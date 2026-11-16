"""Rebuild the May scene corpus with the original lewm_worlds (snapshot), recording executed lines (porting aid)."""
import hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

ORIGINAL = Path('/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/scene_corpus/minimum_tex_20260520T211541Z')
OUT = Path(sys.argv[1])
lines = defaultdict(set)
ROOT = str(Path.cwd()/'lewm_worlds')

def tracer(frame, event, arg):
    f = frame.f_code.co_filename
    if not f.startswith(ROOT):
        return None
    def local(frame, event, arg):
        if event == 'line':
            lines[frame.f_code.co_filename].add(frame.f_lineno)
        return local
    lines[f].add(frame.f_lineno)
    return local

from lewm_worlds import build_corpus, standard_corpus_plan
started = time.time()
sys.settrace(tracer)
plan = standard_corpus_plan(plan_seed=20260520, train_scenes=1000, val_scenes=150, test_id_scenes=150, test_hard_scenes=150)
result = build_corpus(plan, OUT, emit_genesis=True)
sys.settrace(None)
json.dump({f: sorted(v) for f, v in lines.items()}, open(OUT.parent/'executed_lines.json', 'w'))
same, differ = 0, []
for p in sorted(ORIGINAL.rglob('*')):
    if p.is_file():
        q = OUT/p.relative_to(ORIGINAL)
        if q.exists() and q.read_bytes() == p.read_bytes(): same += 1
        else: differ.append(str(p.relative_to(ORIGINAL)))
print(json.dumps(dict(scenes=result.scene_count, files_identical=same, files_differing=len(differ), first=differ[:5], wall_s=round(time.time()-started))))
