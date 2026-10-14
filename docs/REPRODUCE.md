# Reproducing the programme

Every stage below runs from this repository. Each section says what the stage reproduces and how that was checked
against the original programme ([MIGRATION_STATUS.md](MIGRATION_STATUS.md)). Results are preliminary (development
mode). Commands run from the repository root with `PYTHONPATH=.` in the main environment (`env/`), unless stated.

## 0. Environment and artefacts

- `env/setup_rocm_env.sh` creates the main environment (ROCm 7.2, torch 2.12, Genesis 0.4.6, CPU physics).
  `env/setup_render_env.sh` creates the corpus render environment (Genesis 0.3.14, Vulkan); only step 3 uses it.
- `python artifacts/fetch.py vjepa` fetches the V-JEPA 2.1 encoder and source. The shipped checkpoints
  (`artifacts/manifest.yaml`) go into the go2nav cache (`$GO2NAV_CACHE`, default `~/.cache/go2nav`).

## 1. Mazes

    python -m go2nav.mazes.cli configs/mazes/capability_v1.yaml --out outputs/mazes   # also c3v2_sets_v1, c3v3_sets_v1

Regeneration reproduces the registered sets in `data/mazes` byte for byte (`tests/unit/test_mazes.py`).

## 2. Locomotion policy

    python -m go2nav.locomotion.train --out outputs/locomotion/seed11 --seed 11
    python -m go2nav.locomotion.benchmark outputs/locomotion/seed11/model_<N>.pt --out retrained.json
    python -m go2nav.locomotion.benchmark --compare original.json retrained.json

The retrained policy was accepted against the shipped one (`docs/validation/locomotion_retrain_2026-10-06.md`).

## 3. The May corpus (T1's data)

    python -m go2nav.corpus.build outputs/corpus                         # the May plan: 1,450 scenes
    python -m go2nav.corpus.datagen rollout outputs/corpus outputs/datagen --seeds data/corpus/may_2026_rollout_seeds.json \
        [--scenes data/training/t1_temporal/frames.txt]                  # only T1's 80 scenes
    python -m go2nav.corpus.datagen render outputs/corpus outputs/datagen --render-python RENDER_VENV/bin/python \
        --textures --frames-list data/training/t1_temporal/frames.txt   # T1's 18,101 frames; omit for all 69.6 M
    python -m go2nav.data.t1_rows outputs/datagen --out t1_rows.jsonl --render-root outputs/datagen/render

- The scene files are byte-identical to the May build.
- The rollout messages are line-identical to the May rollouts, apart from a scene-id integer (11 scenes checked,
  across all eight families). The seeds file is needed: two scenes were replaced after the May first pass, which
  moved 36 others one place in the name order that the chunk rule (seed 7 + position) uses.
- From all 80 T1 scenes regenerated from scratch, the T1 rows are byte-identical to the shipped rows.
- Render plans and labels are identical.
- The T1 rows equal `data/training/t1_temporal/rows.jsonl`.
- Rendered frames are not bit-reproducible on this Vulkan stack, in the original either (a small fraction of pixels
  differ between runs).
- The T1 row selection ships as data: the original chose it through earlier datasets whose scene roles are not
  reproduced.

## 4. Training recordings and samples (T2-T8 data)

All collections at once (resumable; 428 trials in the layout the sample builders and the chain read):

    python -m go2nav.data.recordings ROOT [--collection NAME ...] [--workers 8]

or one case per process:

    python -m go2nav.data.corridor COLLECTION TRIAL OUT_DIR   # geometry_progress_family, switch_family, short_pulse,
                                                              # balanced_start, balanced_start_horizon
    python -m go2nav.data.collect COLLECTION CASE OUT_DIR     # full_heading, maze_view_training, maze_view_transfer,
                                                              # c3v2_rest_turn

The checked cases reproduce the original physics traces and frames exactly. The exception is the short-pulse frames,
whose recorded floor rendering differs, so the shipped frames are the record.

The decoder contexts also use mission runs, with frames from their verified replays (`go2nav.data.rerender`):

- the C3-v3 on-policy rounds;
- the C3-v2 fresh checks;
- the stage-2 recordings (cohort `s2rec2` in `configs/experiments/r6_dynamics_stage2.yaml`).

Missions are deterministic: rerun missions are identical to the original runs (gate G4).

    python -m go2nav.eval.cohort configs/experiments/decoder_sources.yaml --out outputs/cohorts
    python -m go2nav.eval.cohort configs/experiments/r6_dynamics_stage2.yaml --cohort s2rec2 --out outputs/cohorts
    python -m go2nav.data.mission_sources ROOT --cohorts outputs/cohorts   # links the runs into ROOT, with replays

    python -m go2nav.data.samples all --recordings-root ROOT [--recordings-root ROOT ...] --out data/training

On the original recordings this writes every file of `data/training` except T1's, byte-identical to the shipped ones.

## 5. The learned chain (T1-T8)

    python -m go2nav.training.chain outputs/chain --render-root outputs/datagen/render \
        --recordings-root ROOT [--recordings-root ROOT ...] [--seed-offset K]

This runs T1 → T2 → T3 → T4, T5-0 → T5a → T5b, the T7 decoder cache and three fits with the median choice, the T8
stage-2 refit and T6 (C1). With the original seeds, every stage lands on its original record. C4 and the C1 model are
reproduced, and only C3 changes (see "Retrained chain" in MIGRATION_STATUS). `--seed-offset` gives other seed chains.
The chain writes `outputs/chain/models.yaml`.

## 6. Missions and comparisons

    python -m go2nav.eval.mission --arm C3 --set prelim_test_v1 --maze 30 --recovery off --out runs/C3_30 \
        [--models outputs/chain/models.yaml]
    python -m go2nav.eval.cohort configs/experiments/r4_reserve_exit_v2.yaml --workers 8 --out outputs/cohorts

- Each experiment config in `configs/experiments` is one comparison. A cohort can take `models:` to run on a retrained
  chain.
- The experiments before the final harness ran on the earlier one: the R1, R3 and decoder-source configs set
  `harness: v4` (and `coverage_fix: 'off'` where the run predates the 2 October coverage fix); missions take
  `--harness v4 --coverage-fix off`.
- `python -m go2nav.eval.episode RUN` evaluates a single mission; cohorts evaluate their missions themselves.
- The porting gates compare reruns with the original runs: `tools/porting/mission_check.py` and
  `tools/porting/cohort_check.py --traces`.

## 7. Videos

    python -m go2nav.eval.video RUN_DIR OUT_DIR             # egocentric + chase camera, from a verified replay
    python -m go2nav.eval.hero_video RUN_DIR OUT.mp4        # presentation video (kinematic replay, forecast fan)

`--demo-textures` on a mission draws brick and parquet surfaces for demonstration videos (presentation only; the
models never saw them in training).
