import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from go2nav import research
from go2nav.eval.balls_inference import calibrate, invariance
from go2nav.training import balls


def test_missing_or_incomplete_literature_blocks_work(tmp_path):
    with pytest.raises(FileNotFoundError):
        research.require_record('missing', records=tmp_path)
    record = json.loads((research.RECORDS/'A001.json').read_text())
    record['sources'][0].pop('sections')
    (tmp_path/'A001.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='literature'):
        research.require_record('A001', records=tmp_path)


def test_review_is_bound_to_result_bytes(tmp_path):
    result = tmp_path/'result.json'
    result.write_text('{"score": 1}')
    source = json.loads((research.RECORDS/'A001.json').read_text())['sources'][0]
    review = dict(research_id='A001', result=result.name, result_sha256=research.digest(result),
                  reviewed_at='2026-11-15', sources=[source], measurements='score=1', uncertainty='one sample',
                  explanations=['fixture'], next_step='stop', status='passed')
    path = tmp_path/'review.json'
    research.atomic_json(path, review)
    assert research.require_review(path)['status'] == 'passed'
    result.write_text('{"score": 2}')
    with pytest.raises(ValueError, match='stale'):
        research.require_review(path)


def test_overlapping_device_intervals_count_once():
    rows = [dict(start=0, end=3600), dict(start=1800, end=5400), dict(start=7200, end=9000)]
    assert research.device_hours(rows) == 2


def test_reviewed_index_does_not_authorize_changed_data(tmp_path):
    artifact = tmp_path/'train.npy'
    artifact.write_bytes(b'original frozen embeddings')
    hashes = {artifact.name: research.digest(artifact)}
    research.verify_files(tmp_path, hashes)
    artifact.write_bytes(b'changed embeddings, unchanged index')
    with pytest.raises(ValueError, match='changed'):
        research.verify_files(tmp_path, hashes)


def test_predictor_rejects_replaced_encoder_lineage_before_loading(tmp_path):
    from go2nav.training.balls_slots import load_world
    index = tmp_path/'index.json'
    index.write_text('{"encoder": "first.pt"}')
    checkpoint = tmp_path/'predictor.pt'
    torch.save(dict(cache=str(tmp_path), cache_sha256=research.digest(index)), checkpoint)
    index.write_text('{"encoder": "another.pt"}')
    with pytest.raises(ValueError, match='index no longer matches'):
        load_world(checkpoint, device='cpu')


def test_final_queue_rejects_intermediate_or_wrong_run_artifacts(tmp_path):
    from go2nav.eval.await_balls import verify_final
    config = dict(total_steps=100, kind='l0', n=3, train={'epochs': 2}, model={'latent': 128})
    state = dict(step=100, kind='l0', n=3, train=config['train'], config=config['model'])
    path = tmp_path/'model.pt'
    torch.save(state, path)
    verify_final(path, config)
    for change in (dict(step=50), dict(n=1), dict(inference_only=True), dict(train={'epochs': 1})):
        torch.save(state | change, path)
        with pytest.raises(ValueError):
            verify_final(path, config)


def test_incompatible_resume_preserves_existing_files(tmp_path):
    model, head = balls.build('l0', 0, 'cpu')
    identity = dict(kind='l0', n=3, seed=0, train=balls.DEFAULTS, model=balls.asdict(model.config))
    balls.save(tmp_path/'checkpoint.pt', model, head, 'l0', 3, balls.DEFAULTS,
               dict(identity=identity, optimiser={}, scheduler={}, rng=torch.get_rng_state(),
                    cuda_rng=torch.get_rng_state(), step=1))
    (tmp_path/'config.json').write_text(json.dumps(identity))
    (tmp_path/'metrics.jsonl').write_text('{"step": 2}\n')
    before = {p.name: research.digest(p) for p in tmp_path.iterdir()}
    for n, kind, seed, cfg in ((1, 'l0', 0, balls.DEFAULTS), (3, 'l1', 0, balls.DEFAULTS),
                              (3, 'l0', 1, balls.DEFAULTS), (3, 'l0', 0, balls.DEFAULTS | dict(lr=.1))):
        with pytest.raises(ValueError, match='Incompatible'):
            balls.resume_state(tmp_path, n, kind, seed, cfg)
    assert before == {p.name: research.digest(p) for p in tmp_path.iterdir()}
    assert balls.resume_state(tmp_path, 3, 'l0', 0, balls.DEFAULTS)['step'] == 1
    state = torch.load(tmp_path/'checkpoint.pt', weights_only=False)
    del state['identity']
    torch.save(state, tmp_path/'checkpoint.pt')
    assert balls.resume_state(tmp_path, 3, 'l0', 0, balls.DEFAULTS)['step'] == 1


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.pred_proj = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))

    def encode(self, x):
        return self.projector(x.flatten(0, 1)).reshape_as(x)

    def predict(self, z, actions):
        return self.pred_proj((z+actions).flatten(0, 1)).reshape_as(z)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_calibration_uses_fixed_parameters_and_upstream_calibrated_encoder(dtype):
    torch.manual_seed(7)
    model = Toy().to(dtype).eval()
    x = torch.randn(32, 4, 2, dtype=dtype)*3+7
    a = torch.randn(32, 3, 2, dtype=dtype)
    params = copy.deepcopy(dict(model.named_parameters()))
    calibrate(model, lambda: ((x[i:i+8], a[i:i+8]) for i in range(0, 32, 8)))
    with torch.no_grad():
        encoder_input = model.projector[0](x.flatten(0, 1))
        predictor_input = model.pred_proj[0]((model.encode(x)[:, :3]+a).flatten(0, 1))
    for module, values in ((model.projector[1], encoder_input), (model.pred_proj[1], predictor_input)):
        torch.testing.assert_close(module.running_mean, values.mean(0))
        torch.testing.assert_close(module.running_var, values.var(0))
    assert all(torch.equal(p, params[name]) for name, p in model.named_parameters())
    assert all(v['passed'] for mode in invariance(model, x, a).values() for v in mode.values())


def test_atomic_checkpoint_rng_roundtrip(tmp_path):
    torch.manual_seed(3)
    model, head = balls.build('l1', 7, 'cpu')
    rng = torch.get_rng_state()
    path = tmp_path/'checkpoint.pt'
    balls.save(path, model, head, 'l1', 1, balls.DEFAULTS, dict(rng=rng))
    expected = torch.rand(10)
    loaded, loaded_head, state = balls.load(path)
    torch.set_rng_state(state['rng'])
    assert torch.equal(torch.rand(10), expected)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    assert all(torch.equal(v, loaded_head.state_dict()[k]) for k, v in head.state_dict().items())
    assert not list(tmp_path.glob('*.partial'))
