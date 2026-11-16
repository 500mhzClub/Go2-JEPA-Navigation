import collections
import json
import re
import statistics as st
import sys

import numpy as np
from scipy.stats import binom

rows = json.load(open(sys.argv[1]))


def key(r):
    m = re.search(r'obst(\d)|wf_(?:ref|held)', r['run'])
    n = '4' if r['run'].startswith('wf') else m.group(1)
    e = re.search(r'validation_(\d+)_ep(\d)', r['run'])
    return n, e.group(1), e.group(2)


def fail_class(r):
    if r['round_trip']:
        return 'ok'
    e = r['error'] or ''
    if 'PhysicalStop' in e or 'contact' in e.lower():
        return 'contact: '+e[:50]
    if 'Unported' in e:
        return 'unported: '+e[:60]
    if e:
        return 'error: '+e[:60]
    if r['elapsed'] and r['elapsed'] >= 479:
        return 'timeout'
    return f'ended early (terminal)'


for name, g in rows.items():
    print(f'\n=== {name}: {sum(r["round_trip"] for r in g)}/{len(g)}  beacon {sum(r["beacon"] for r in g)}')
    print('  ', collections.Counter(fail_class(r) for r in g).most_common())
    print('   median plan frame ms', round(st.median(r['plan_ms'] for r in g if r['plan_ms']), 0))

print('\n--- exact failure strings, wo+wh+wf')
for name in ('ws', 'wh', 'wo', 'wf_ref', 'wf_held'):
    for r in rows[name]:
        if not r['round_trip'] and (r['error'] or (r['elapsed'] or 0) < 479):
            print(name, r['run'][-16:], round(r['elapsed'] or 0), (r['error'] or '')[:110], '| src', (r['source_error'] or '')[:60])

print('\n--- how far the model reaches, wh + wo (held)')
for name in ('wh', 'wo', 'wf_held'):
    g = rows[name]
    acted = sum(r['acted'] for r in g)
    cut = sum(r['cut'] for r in g)
    near = sum(r['cut_near'] for r in g)
    conf = sum(r['any_confirmed'] for r in g)
    print(f'{name}: plans {acted}; any confirmed track {100*conf/acted:.1f}%; plans on a cut route {100*cut/acted:.1f}%; '
          f'cut plan with confirmed track within 1.3 m of target {100*near/acted:.1f}% of plans; missions with any cut {sum(r["cut"]>0 for r in g)}/{len(g)}')

print('\n--- route status share of plans, by outcome (wo)')
for ok in (True, False):
    c = collections.Counter()
    for r in rows['wo']:
        if r['round_trip'] == ok:
            c.update(r['status'])
    tot = sum(c.values())
    print(ok, [(k.replace('OBSERVED_', '').replace('REQUIRES_', ''), round(100*v/tot, 1)) for k, v in c.most_common(8)])

print('\n--- timeouts: path against shortest, pace, wo+wh')
for name in ('wo', 'wh', 'wf_held'):
    t = [r for r in rows[name] if fail_class(r) == 'timeout']
    ok = [r for r in rows[name] if r['round_trip']]
    def dist(r):
        return sum((r[k] or {}).get('actual_path_m', 0) or 0 for k in ('outbound', 'ret'))
    def short(r):
        return sum((r[k] or {}).get('shortest_path_m', 0) or 0 for k in ('outbound', 'ret'))
    print(name, 'timeouts', len(t), 'beacon reached in', sum(r['beacon'] for r in t),
          '| median metres walked', round(st.median(dist(r) for r in t), 1), 'shortest', round(st.median(short(r) for r in t), 1),
          '| successes: median elapsed', round(st.median(r['elapsed'] for r in ok)), 'walked', round(st.median(dist(r) for r in ok), 1),
          'shortest', round(st.median(short(r) for r in ok), 1))
    c = collections.Counter()
    for r in t:
        c.update(r['status'])
    tot = sum(c.values()) or 1
    print('   timeout plans:', [(k.replace('OBSERVED_', '')[:34], round(100*v/tot, 1)) for k, v in c.most_common(6)],
          'hold-free reasons', collections.Counter(k for r in t for k in r['reasons']).most_common(4))

print('\n--- success time spare against 480 s (wo)')
e = sorted(r['elapsed'] for r in rows['wo'] if r['round_trip'])
print(np.percentile(e, [10, 50, 90, 100]).round(0))

print('\n--- reversing share of plans, and contacts')
for name in rows:
    g = rows[name]
    a = sum(r['acted'] for r in g)
    print(name, f"reverse {100*sum(r['rev'] for r in g)/a:.1f}% forward {100*sum(r['fwd'] for r in g)/a:.1f}% turn {100*sum(r['turn'] for r in g)/a:.1f}% hold {100*sum(r['hold'] for r in g)/a:.1f}%  contact-stopped",
          sum('contact' in fail_class(r) for r in g))

print('\n--- paired (by walker count, maze, episode)')


def pair(a, b):
    A, B = ({key(r): r['round_trip'] for r in rows[n]} for n in (a, b))
    k = set(A) & set(B)
    gain = sum(B[x] and not A[x] for x in k)
    lose = sum(A[x] and not B[x] for x in k)
    return gain, lose, len(k)


for a, b in (('ws', 'wl'), ('ws', 'wh'), ('wh', 'wo'), ('ws', 'wo'), ('wf_ref', 'wf_held')):
    print(a, '->', b, 'gained/lost/n', pair(a, b))

print('\n--- what the bar can resolve: P(pass, net >= -3), simulated from a paired model')
rng = np.random.default_rng(0)
for d in (11, 14):
    for true_net in (0, -3, -5, -8):
        # d discordant pairs; each is a loss with probability p, so E[net] = d(1-2p)
        p = (1-true_net/d)/2
        losses = rng.binomial(d, p, 200000)
        net = d-2*losses
        print(f'discordant {d}: true net {true_net:+d}: pass {100*(net >= -3).mean():.0f}%')
print('\ndrift: chain of k handovers, each passing with true net -2 at d=11:')
p = (1+2/11)/2
for k in (1, 3, 6):
    n = rng.binomial(11, p, (200000, k))
    net = 11-2*n
    print(k, f'all pass {100*(net >= -3).all(1).mean():.0f}%  expected true loss {2*k} of 80')
