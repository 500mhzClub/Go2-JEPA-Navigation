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

## Stage-2 refit (T8)

The same check with `--recipe stage2` (seed 2026093012, 3,520 updates; old 32, maze 16, on-policy 1.33 x 5 + 1.35,
patch 6, patch_off 2 per batch) against `stage2_decoder_fits/stage2_p3_large_past_frames_s2026093012.json`:

**Result: identical.** Selection score 9.868 mm, as recorded. eval_onpolicy, eval_transfer, eval_offline and
eval_patch match for C3 and C4, for every movement type and every patch edge bin (approach, entry, on_patch, exit,
off_patch).
