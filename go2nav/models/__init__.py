"""Learned models.

- `vjepa`: the frozen V-JEPA 2.1 ViT-L encoder (tokens), preprocessing, normalisation and pooling.
- `predictor`: the JEPA predictor (dense future-token prediction from three frames, control history and the action),
  in its 500-ms form (T1-T3) and its 100-800 ms horizon-conditioned form (T4, deployed).
- `motion`: motion heads: the T5 readout, the C3 decoder (T7) and C4 (T7).
- `command`: C1's command-history ridge model (T6).
"""
