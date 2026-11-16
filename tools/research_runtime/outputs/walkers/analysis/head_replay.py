"""The mover head at every frame of walker units in the validation mazes, on the CPU, for tuning the tracker at 10 Hz."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from go2nav.data.lewm_dataset import Corpus
from go2nav.models.lewm import MOVER_SCALE_M, load_world_model, normalise_frames
torch.set_num_threads(10)
root, out, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
out.mkdir(exist_ok=True)
device = torch.device('cpu')
model = load_world_model(sys.argv[4], device).eval()
c = Corpus(root, length=12, roles=('lewm_val',), kinds=('explore', 'mission'), command_dim=model.config.command_dim, movers=True)
us = sorted(set(c.unit.tolist()))
rng = np.random.default_rng(1); rng.shuffle(us)
done = 0
for u in us:
    if done >= count: break
    folder = root/c.units[u]['path'] if 'path' in c.units[u] else None
    labels = np.asarray(c.movers(u))
    if (labels[:, :, 0] > 0).mean() < .02: continue
    frames = np.asarray(c.frames(u))
    t0 = time.time(); rows = []
    for s in range(0, len(frames), 32):
        x = torch.from_numpy(frames[s:s+32].copy())[None]
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            o = model.movers(model.encode(normalise_frames(x))[0]).float()
        rows.append(torch.cat((o[..., :1].sigmoid(), o[..., 1:]*MOVER_SCALE_M), -1).numpy())
    np.savez_compressed(out/f'{u}.npz', detected=np.concatenate(rows), labels=labels, poses=np.asarray(c.poses(u)), unit=json.dumps(c.units[u]))
    done += 1
    print(done, u, len(frames), round(len(frames)/(time.time()-t0)), 'fps', flush=True)
