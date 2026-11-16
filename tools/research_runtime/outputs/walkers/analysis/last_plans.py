import json, sys
from pathlib import Path
for name in sys.argv[1:]:
    run = Path('outputs/cohorts')/name.split('_C5')[0]/'runs'/name
    hit = next(x for x in json.loads((run/'native/contact_events.json').read_text()) if x['disallowed_contacts'])
    t = hit['timestamp_s']
    rows = [p for p in json.loads((run/'planning.json').read_text()) if 'selection' in p and p['measured_ns']/1e9 <= t]
    print(name, 't', t, 'selection keys', sorted(rows[-1]['selection'].keys())[:60])
    for p in rows[-10:]:
        s = p['selection']
        print('  ', round(p['measured_ns']/1e9-t, 1), 'reason', p.get('reason'), 'act', s.get('action'), 'unpred', s.get('unpredicted_check_action'), 'chk', str(s.get('clearance_check'))[:120], 'moving', [ (t_['id'], t_['velocity_m_s'], t_['motion_updates']) for t_ in s.get('obstacle_tracks') or [] if t_['motion_updates'] >= 3][:2])
