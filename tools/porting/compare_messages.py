"""Compare a ported rollout's messages.jsonl with an original one, ignoring the scene number (porting tool).

    python tools/porting/compare_messages.py PORTED.jsonl ORIGINAL.jsonl

The original recordings put Python's per-process string hash of the scene name in reset_event and episode_info
(`scene_id`, also copied into render-plan frames); the port uses a fixed number. Every other byte must agree. Prints the first differing line.
"""
import json
import sys


def normalised(line):
    record = json.loads(line)
    for holder in (record.get('payload'), record.get('episode')):  # messages, render-plan frames
        if isinstance(holder, dict) and 'scene_id' in holder:
            holder['scene_id'] = 0
    return json.dumps(record, sort_keys=True, separators=(',', ':'))


def main(ported, original):
    n = differing = 0
    first = None
    with open(ported) as a, open(original) as b:
        for n, (x, y) in enumerate(zip(a, b), 1):
            if x != y and normalised(x) != normalised(y):
                differing += 1
                first = first or (n, x[:300], y[:300])
        rest = sum(1 for _ in a), sum(1 for _ in b)
    print(json.dumps(dict(lines_compared=n, differing=differing, extra_ported=rest[0], extra_original=rest[1])))
    if first:
        print('first difference at line', first[0]); print(' ported:  ', first[1]); print(' original:', first[2])
    return differing == 0 and rest == (0, 0)


if __name__ == '__main__':
    sys.exit(0 if main(*sys.argv[1:3]) else 1)
