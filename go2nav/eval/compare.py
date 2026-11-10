"""Reading a suite sooner (docs/dynamic/PLAN.md P19): two cohorts compared mission by mission, and a bar's state while
a suite still runs.

- `paired A B`: the missions finished in both cohorts (the same maze and episode). Only those whose outcome differs
  carry information: B's gains and losses, and the exact sign test's two-sided p for "no difference". Two runs of the
  same code compared this way measure the scatter a change must exceed.
- `bar --need N --of M C [C ...]`: whether N round trips of M are already secured, already out of reach, or open.

    python -m go2nav.eval.compare paired outputs/cohorts/A outputs/cohorts/B
    python -m go2nav.eval.compare bar --need 62 --of 80 outputs/cohorts/C outputs/cohorts/D
"""
import argparse
import json
import math
import re
from pathlib import Path


def outcomes(cohort):
    """{(maze, episode): round trip made} of a cohort's finished missions."""
    out = {}
    for path in sorted(Path(cohort).glob('runs/*/episode_evaluation.json')):
        match = re.search(r'_(\d+)_ep(\d+)$', path.parent.name)
        if match:
            out[(int(match[1]), int(match[2]))] = bool(json.loads(path.read_text())['round_trip_success'])
    return out


def sign_test(gains, losses):
    """The two-sided exact p of a split at least this uneven, were gains and losses equally likely."""
    n, k = gains+losses, min(gains, losses)
    return 1. if not n else min(1., 2*sum(math.comb(n, i) for i in range(k+1))/2**n)


def paired(a, b):
    both = sorted(set(a) & set(b))
    gains, losses = [m for m in both if b[m] and not a[m]], [m for m in both if a[m] and not b[m]]
    return dict(missions=len(both), first=sum(a[m] for m in both), second=sum(b[m] for m in both), gains=gains,
                losses=losses, p=sign_test(len(gains), len(losses)))


def bar(made, done, total, need):
    """'secured', 'out of reach' or 'open', with the round trips still needed of the missions left."""
    left = total-done
    state = 'secured' if made >= need else 'out of reach' if made+left < need else 'open'
    return dict(state=state, made=made, done=done, needed=max(0, need-made), left=left)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('paired')
    p.add_argument('first')
    p.add_argument('second')
    p = commands.add_parser('bar')
    p.add_argument('cohorts', nargs='+')
    p.add_argument('--need', type=int, required=True)
    p.add_argument('--of', type=int, required=True)
    args = parser.parse_args()
    name = lambda m: f'{m[0]}/{m[1]}'
    if args.command == 'paired':
        out = paired(outcomes(args.first), outcomes(args.second))
        print(f"{out['missions']} missions in both: {out['first']} then {out['second']} round trips")
        print(f"gains {len(out['gains'])}: {' '.join(map(name, out['gains']))}")
        print(f"losses {len(out['losses'])}: {' '.join(map(name, out['losses']))}")
        print(f"sign test p = {out['p']:.3f}")
    else:
        rows = [v for c in args.cohorts for v in outcomes(c).values()]
        out = bar(sum(rows), len(rows), args.of, args.need)
        print(f"{out['state']}: {out['made']} of {out['done']} done; needs {out['needed']} of the {out['left']} left")


if __name__ == '__main__':
    main()
