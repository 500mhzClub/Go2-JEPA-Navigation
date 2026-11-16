"""P10 step 1 gate (docs/dynamic/PLAN.md): the true pose against wm_rude_model_* on matched missions.
Gate: net gain >= 4 round trips with wins >= 2 x losses. Early stop only when the gate can no longer be met,
or as broken (a mission without an evaluation, i.e. a crash, in >= 5 of the first 20)."""
import glob, json, os
REF, NEW = 'wm_rude_model_obst{}_c5', 'wm_pose_rude_obst{}_c5'
def runs(c):
    out = {}
    for r in glob.glob(f'outputs/cohorts/{c}/runs/*'):
        key = r.rsplit('_C5_', 1)[1]
        ev = f'{r}/episode_evaluation.json'
        if os.path.exists(f'{r}/result.json') and os.path.exists(ev):
            res = json.load(open(f'{r}/result.json'))
            err = str(res.get('error') or res.get('status'))[:40]
            out[key] = dict(rt=json.load(open(ev))['round_trip_success'], err=err,
                            crash=not any(k in err for k in ('PoseLost', 'PhysicalStop', 'EPISODE_RECORDED')))
    return out
wins = losses = done = crashed = 0; remaining_ref_fail = 0; ends = {}
for k in (1, 2):
    ref, new = runs(REF.format(k)), runs(NEW.format(k))
    for key, r in ref.items():
        if key in new:
            done += 1; n = new[key]
            crashed += n['crash']
            wins += bool(n['rt']) and not r['rt']; losses += bool(r['rt']) and not n['rt']
            e = 'RT' if n['rt'] else n['err'].split('(')[0]; ends[e] = ends.get(e, 0)+1
        elif not r['rt']:
            remaining_ref_fail += 1
best = wins+remaining_ref_fail
verdict = 'RUNNING'
if best-losses < 4 or best < 2*losses: verdict = 'FAIL (gate can no longer be met)'
if done <= 20 and crashed >= 5: verdict = 'FAIL (broken: crashes)'
if done == 80: verdict = 'PASS' if wins-losses >= 4 and wins >= 2*losses else 'FAIL'
print(f'matched {done}/80 wins {wins} losses {losses} net {wins-losses} crashed {crashed} ends {ends}')
print(verdict)
