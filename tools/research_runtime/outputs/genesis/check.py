"""P24 2d: a suite's stop rule (PLAN "Phases and gates", as P18's check.py). `check.py suite KIND C [C ...]` -> RUNNING
or BROKEN: in the suite's first 20 missions, at least 5 crashes, or for KIND 'polite' (polite walkers or none) crashes
and contact stops together; with walkers that do not yield, contact stops are what is measured, not a broken mode."""
import glob, json, os, sys
ROOT = '/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/cohorts'

def rows(cohort):
    out = []
    for r in sorted(glob.glob(f'{ROOT}/{cohort}/runs/*')):
        if 'attempt' in r or not os.path.exists(f'{r}/result.json'):
            continue
        res = json.load(open(f'{r}/result.json'))
        err = str(res.get('error') or res.get('status'))
        ev = f'{r}/episode_evaluation.json'
        out.append(dict(run=os.path.basename(r), error=err,
                        crash=not any(k in err for k in ('PoseLost', 'PhysicalStop', 'EPISODE_RECORDED')),
                        contact='DISALLOWED' in err,
                        round_trip=bool(os.path.exists(ev) and json.load(open(ev))['round_trip_success'])))
    return out

kind, cohorts = sys.argv[2], sys.argv[3:]
r = [x for c in cohorts for x in rows(c)]
head = r[:20]
t = dict(done=len(r), round_trips=sum(x['round_trip'] for x in r), crashes=sum(x['crash'] for x in head),
         contacts=sum(x['contact'] for x in head), errors=sorted({x['error'][:80] for x in r if x['crash'] or x['contact']}))
print(json.dumps(t))
print('BROKEN' if t['crashes']+(t['contacts'] if kind == 'polite' else 0) >= 5 else 'RUNNING')
