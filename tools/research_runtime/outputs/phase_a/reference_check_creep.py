"""P13 item 2: progress of the range-sensor reference suites, and the stop rule's "broken" check: a crash (an error
other than a pose loss, a physical stop or a recorded episode), or a contact stop, in at least 5 of a suite's first 20
missions (P8's run had 1 contact stop in its 40 matching missions; the first reference run, with ghost trails, 7 in 20)."""
import glob, json, os, sys
SUITES = {'rs_rude': ['rs_rude_obst1_c5', 'rs_rude_obst2_c5'], 'rs_polite': ['rs_obst1_c5', 'rs_obst2_c5'], 'rs_four': ['rs_obst4_c5']}
verdict = 'RUNNING'
for suite, cohorts in SUITES.items():
    done = crashed = rt = contacts = 0
    for c in cohorts:
        for r in sorted(glob.glob(f'outputs/cohorts/{c}/runs/*')):
            if 'attempt' in r or not os.path.exists(f'{r}/result.json'):
                continue
            res = json.load(open(f'{r}/result.json'))
            err = str(res.get('error') or res.get('status'))
            done += 1
            crash = not any(k in err for k in ('PoseLost', 'PhysicalStop', 'EPISODE_RECORDED'))
            crashed += crash and done <= 20
            contacts += 'DISALLOWED' in err and done <= 20
            ev = f'{r}/episode_evaluation.json'
            rt += os.path.exists(ev) and json.load(open(ev))['round_trip_success']
    print(f'{suite}: {done} done, {rt} round trips, {crashed} crashes and {contacts} contact stops in the first 20')
    if crashed >= 5 or contacts >= 5:
        verdict = f'BROKEN ({suite})'
print(verdict)
