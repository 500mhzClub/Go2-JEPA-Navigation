# Replay of span8w C5's deadlocks

Run from the repository root with `PYTHONPATH=.` and the Genesis environment's python, in order:

1. `stuck_list.py`: the episodes span8w C5 failed and idm100 C5 passed (`outputs/cohorts/lewm_*_c5{,_span8w}`), and
   their stuck decisions (hold, every moving candidate BLOCKED) -> `outputs/lewm/stuck_replay/stuck_list.json`.
2. `stuck_driver.py`: per deadlocked episode (over 100 stuck decisions), rerenders the frames up to the last sampled
   decision (`stuck_render.py`, `go2nav.data.rerender` on a truncated tape, physics checked against the trace) and
   replays the first 10 stuck decisions and those about 25-300 decisions later through span8w C5 and C6 and idm100
   C5 and C6 (`stuck_forecast.py`). Clearance: a half-plane wall fitted per decision to the logged segment clearances
   of the logged forecasts, then the planner's `clearance_check` and `reserve_turns`.
3. `stuck_summary.py`: escapes per forecaster, turn forecasts' dip and translation.

Validation: the replayed span8w C5 forecasts equal the logged ones (0.0 mm); the wall fit's residual is 0.29 mm median.
