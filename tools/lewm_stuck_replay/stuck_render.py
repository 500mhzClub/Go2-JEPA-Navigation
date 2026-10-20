"""Rerender a run's primary frames up to `last_frame` (go2nav.data.rerender with the request tape truncated)."""
import json, sys, time
from pathlib import Path
from go2nav.data import rerender
run, out, last = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if (out/'.done').exists():
    sys.exit(0)
orig = rerender.load
def load(r, name):
    v = orig(r, name)
    if name == 'requests.json':
        return v[:(last+1)*5]
    if name == 'acquisitions.json':
        return v[:last+1]
    return v
rerender.load = load
t = time.time()
n = rerender.replay(run, out)
(out/'.done').write_text(json.dumps(dict(frames=n, wall_s=round(time.time()-t))))
print(run.name, n, round(time.time()-t), flush=True)
