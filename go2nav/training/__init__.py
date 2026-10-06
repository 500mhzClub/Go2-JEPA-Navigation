"""Retraining every learned component from scratch, each with a seed and a lineage manifest.

Stages (each a module with `train` and `evaluate`; see docs/MIGRATION_STATUS.md for status):
- `t1_temporal`: the JEPA predictor from fresh weights on May-corpus video.
- T2-T5 (native adaptation, balanced start, horizon conditioning, motion readouts), T6 (C1), T7 (C3 decoder and C4),
  T8 (stage-2 refits): ported in order.
`common` holds keyed random streams, control normalisation, the frozen-feature store and lineage manifests.
"""
