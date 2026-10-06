# C3 decoder and C4 fit: ported code reproduces the deployed fit (6 October 2026)

`tools/porting/check_decoder_fit_against_original.py` runs `go2nav.training.decoder.fit` (recipe `decoder_fix`, seed
2026093011, 3,520 updates) on the original stage-2 feature cache, read in place, and compares the result with the
deployed fit's record (`dev_decoder_fits/p3_large_past_frames_s2026093011.json`).

**Result: identical.**
- Selection score: 8.509 mm (C3, eval_onpolicy at 800 ms, mean over movement types), as recorded.
- Every reported median XY error and median ratio matches the record to the printed precision: eval_onpolicy C3 and
  C4, eval_transfer C3, eval_offline C4, all movement types.

So the ported fit, with the same features and seed, gives the same models. Retraining on a retrained predictor (T4)
needs a new cache (`decoder cache --predictor`), since the predicted features depend on T4.
