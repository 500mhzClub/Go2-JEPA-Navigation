"""P18: progress of the cost-decision cohorts and the script's own stop rule.
    check.py smoke            -> PASS or FAIL: no crash, no contact stop, at least 4 of 6 still round trips
    check.py suite C [C ...]  -> RUNNING or BROKEN: a crash or a contact stop in at least 5 of the suite's first 20 missions"""
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
        out.append(dict(run=os.path.basename(r), error=err, crash=not any(k in err for k in ('PoseLost', 'PhysicalStop', 'EPISODE_RECORDED')),
                        contact='DISALLOWED' in err, round_trip=bool(os.path.exists(ev) and json.load(open(ev))['round_trip_success'])))
    return out

def tally(cohorts, first=None):
    r = [x for c in cohorts for x in rows(c)]
    head = r[:first] if first else r
    return dict(done=len(r), round_trips=sum(x['round_trip'] for x in r), crashes=sum(x['crash'] for x in head),
                contacts=sum(x['contact'] for x in head), errors=sorted({x['error'][:80] for x in r if x['crash'] or x['contact']}))

def stood(cohort):
    n = 0
    for r in sorted(glob.glob(f'{ROOT}/{cohort}/runs/*')):
        if 'attempt' in r or not os.path.exists(f'{r}/planning.json'): continue
        p = [x for x in json.load(open(f'{r}/planning.json')) if 'selection' in x]
        n += sum(x['action'] == 'hold' and x['route_status'] == 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER' for x in p) >= 100
    return n

if sys.argv[1] == 'stall':
    t = tally(['cost2_stall_c5']); s = stood('cost2_stall_c5')
    print(json.dumps(t | dict(stood=s)))
    print('PASS' if t['done'] == 6 and not (t['crashes'] or t['contacts']) and s <= 1 else 'FAIL')
elif sys.argv[1] == 'smoke':
    still, walker = tally(['cost_smoke_still_c5']), tally(['cost_smoke_obst1_c5'])
    print('still', json.dumps(still)); print('one polite walker', json.dumps(walker))
    ok = still['done'] == 6 and walker['done'] == 6 and not (still['crashes'] or still['contacts'] or walker['crashes'] or walker['contacts']) \
        and still['round_trips'] >= 4
    print('PASS' if ok else 'FAIL')
else:
    t = tally(sys.argv[2:], first=20)
    print(json.dumps(t))
    print('BROKEN' if t['crashes']+t['contacts'] >= 5 else 'RUNNING')
