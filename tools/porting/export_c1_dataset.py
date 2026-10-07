"""Export C1's (T6) training data from the LeWMQuad-v3 recordings (porting tool).

C1 was fitted on the short-pulse training schedule: every training context of the moving-action switch, geometry
progress and short-pulse recordings, weighted by how often the schedule drew it. For each context this writes, to
data/training/c1_command_model/contexts.jsonl:
- commands: the known (requested) commands of the tape, up to 8 x [vx, vy, yaw rate];
- history: the 420-value applied-command history at the decision (four frames' records, as at run time);
- motions: 8 x [x, y, yaw] true body-frame motion at 100-800 ms, or null where not valid;
- weight: the schedule's draw count.
The rows are produced by the original code (frozen snapshot); `go2nav.training.command_model` rebuilds the design matrix.

Usage: GO2NAV_SNAPSHOT=/path/to/snapshot python tools/porting/export_c1_dataset.py  (programme environment)
"""
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
OUT = REPO/'data/training/c1_command_model'


def main():
    snapshot = Path(os.environ['GO2NAV_SNAPSHOT'])
    sys.path[:0] = [str(snapshot), str(snapshot/'lewm_genesis'), str(snapshot/'lewm_worlds')]
    os.chdir(snapshot)
    from scripts import fit_go2_short_pulse_command_control_development as c1
    schedule = json.loads((c1.source.OUTPUT/'schedule.json').read_text())
    rows = c1.source.load_training_rows()
    data = c1.dataset(rows, c1.features(rows))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'contexts.jsonl').open('w') as stream:
        for i, row in enumerate(rows):
            history = data['x'][i, 0, 40:]  # the 420 history columns (identical across horizons)
            motions = [t['motion'] if t['motion_valid'] else None for t in row['targets'][:8]]
            stream.write(json.dumps(dict(sample_id=row['sample_id'], commands=row['known_commands'], history=history.tolist(),
                                         motions=motions, weight=schedule['context_draw_counts'][row['sample_id']]))+'\n')
    print(dict(contexts=len(rows), draws=int(sum(schedule['context_draw_counts'][r['sample_id']] for r in rows))))


if __name__ == '__main__':
    main()
