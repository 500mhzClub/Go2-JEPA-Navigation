"""P11 bars (docs/dynamic/PLAN.md), checked against the reference on matched missions.
step 2: vacate_rude_* vs wm_rude_model_* (11 RT, 10 contact stops): RT >= 21 (net >= 10), wins >= 2 x losses, contacts <= 10.
step 3: vacate_obst* vs the fixed record (wo_* with smc_fix_obst1's two missions, 70): RT >= 67, no contacts.
Verdict FAIL as soon as a bar can no longer be met; BROKEN on crashes in >= 5 of the first 20."""
import glob, json, os, sys
step = sys.argv[1] if len(sys.argv) > 1 else '2'
if step == '2':
    REF, NEW = ['wm_rude_model_obst{}_c5'], 'vacate_rude_obst{}_c5'
else:
    REF, NEW = ['wo_obst{}_c5', 'smc_fix_obst{}'], 'vacate_obst{}_c5'
def runs(c):
    out = {}
    for r in glob.glob(f'outputs/cohorts/{c}/runs/*'):
        if 'attempt' in r: continue
        key = r.rsplit('_C5_', 1)[1]
        ev = f'{r}/episode_evaluation.json'
        if os.path.exists(f'{r}/result.json') and os.path.exists(ev):
            res = json.load(open(f'{r}/result.json'))
            err = str(res.get('error') or res.get('status'))
            out[key] = dict(rt=json.load(open(ev))['round_trip_success'], contact='DISALLOWED' in err,
                            crash=not any(k in err for k in ('PoseLost', 'PhysicalStop', 'EPISODE_RECORDED', 'DISALLOWED')),
                            end='RT' if json.load(open(ev))['round_trip_success'] else err.split('(')[0][:40])
    return out
wins = losses = done = crashed = contacts = rt = 0; ends = {}; ref_rt_all = 0; first = []
for k in (1, 2):
    ref = {}
    for c in REF: ref.update(runs(c.format(k)))
    new = runs(NEW.format(k))
    ref_rt_all += sum(r['rt'] for r in ref.values())
    for key, r in ref.items():
        if key in new:
            n = new[key]; done += 1; crashed += n['crash']; contacts += n['contact']; rt += n['rt']
            wins += n['rt'] and not r['rt']; losses += r['rt'] and not n['rt']
            ends[n['end']] = ends.get(n['end'], 0)+1
left = 80-done
print(f'matched {done}/80 rt {rt} wins {wins} losses {losses} net {wins-losses} contacts {contacts} crashed {crashed} ends {ends}')
if done <= 20 and crashed >= 5: print('BROKEN'); sys.exit()
if step == '2':
    best_rt = rt+left
    if contacts > 10 or best_rt < 21 or wins+left < 2*losses or wins+left-losses < 10: print('FAIL'); sys.exit()
    print('PASS' if done == 80 and rt >= 21 and wins >= 2*losses and wins-losses >= 10 else 'RUNNING')
else:
    if contacts > 0 or rt+left < 67: print('FAIL'); sys.exit()
    print('PASS' if done == 80 else 'RUNNING')
