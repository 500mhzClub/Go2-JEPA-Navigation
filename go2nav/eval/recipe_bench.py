"""P27's head, measures and references (docs/dynamic/PLAN.md P27 item 2).

    python -m go2nav.eval.recipe_bench references --data ROOT --variant t2 --out DIR     # hold, constant velocity, Oracle-S
    python -m go2nav.eval.recipe_bench score --data ROOT --run RUN                       # the head, then RUN/bench.json
    python -m go2nav.eval.recipe_bench table --references DIR --runs RUN [RUN ...]       # the table, at k = 8

A **forecaster** is `f(bench, e, t0, future)`: for windows whose last history frame is frame t0 (N,) of validation
episode e (N,), under the future actions (N, 16, 2) in pixels, the positions of the agent and the three obstacles after
each step, (N, 16, 4, 2) in pixels. `measure` scores any forecaster the same way:
- **E:** the agent's error, and its error with the tapes shuffled across the windows (the action-shuffle ratio);
- **X:** an obstacle's error on windows where the agent stays over 2 x REACT away, by whether the obstacle bounces,
  beside constant velocity's and hold's;
- **A:** on the 6 pairs of each branch moment's 4 tapes, per obstacle, the predicted response against the true one.

The JEPA's forecaster reads its rolled-out latents through a **head** trained on encoded training frames only
(`ClsHead` for one latent, `TokenHead` for tokens). **R,** retention, is the head's error on encoded validation frames.
**Oracle-S** (`OracleS`) is a supervised predictor on true positions over the same 3-frame history, rolled out the
same way; it sees the kinds on T1 only. Errors are mean Euclidean distances in pixels.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from go2nav.data import reactive_balls as data
from go2nav.eval.paired_forecast_metrics import LayoutBootstrap
from go2nav.models.norm import LayerNorm
from go2nav.sim import reactive_balls as world
from go2nav.training.balls import schedule, to_pixels
from go2nav.training.recipe_bench import HISTORY, load

KS, STEPS = (1, 4, 8, 16), world.BRANCH_STEPS
RESPONSIVE, QUIET = 1.5, .5            # px: a pair responds above the first; a false response is one predicted above
OBJECTS = ('agent', 'green', 'blue', 'yellow')   # the first where the true response is under the second
HALF = world.SIZE/2
PAIRS = np.array(list(itertools.combinations(range(len(world.TAPES)), 2)))


class Bench:
    """A split's labels and branches (numpy) and, with a device, its frames and actions on it."""

    def __init__(self, root, variant, policy='interactive', split='val', device=None):
        frames, self.labels, self.branches = data.load(root, policy, split, variant, mmap=device is None)
        self.positions = np.concatenate((self.labels['agent'][:, :, None], self.labels['obstacle']), 2)   # (E, T, 4, 2)
        self.device = device
        if device is not None:
            self.frames = torch.from_numpy(frames).to(device)
            self.actions = torch.from_numpy(self.labels['actions']).to(device)/world.MAX_ACTION

    def history(self, e, t0):
        """The true positions of the history frames (N, 3, 4, 2)."""
        return self.positions[e[:, None], t0[:, None]+np.arange(1-HISTORY, 1)]


# -- forecasters ------------------------------------------------------------------------------------------------------

def hold(bench, e, t0, future):
    return np.repeat(bench.positions[e, t0][:, None], STEPS, 1)


def constant_velocity(bench, e, t0, future):
    """Each object at its last displacement, the walls unknown."""
    now, before = bench.positions[e, t0], bench.positions[e, t0-1]
    return now[:, None]+np.arange(1, STEPS+1)[None, :, None, None]*(now-before)[:, None]


def mlp(inputs, hidden, outputs, layers=3):
    dims = [inputs]+[hidden]*layers
    body = [m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.GELU())]
    return nn.Sequential(*body, nn.Linear(hidden, outputs))


class OracleS(nn.Module):
    """The supervised reference: from the true positions of 3 frames and the action, each object's next displacement.
    An obstacle's net sees its position, its last two displacements, the agent's position and action, where the agent
    will be relative to it, and (T1) its kind; it adds a correction to the last displacement. The agent's net corrects
    the action from its position and the obstacles' relative positions and displacements."""

    def __init__(self, kinds, hidden=512):
        super().__init__()
        self.kinds = kinds
        self.obstacle = mlp(17, hidden, 2)
        self.agent = mlp(4+4*world.K, hidden, 2)

    def step(self, p, kind, u):
        """p (B, 3, 4, 2) px, kind (B, K, 3) one-hot (zeros on T2), u (B, 2) px, to the next positions (B, 4, 2)."""
        a, o = p[:, -1, 0], p[:, -1, 1:]
        vo, vo2 = o-p[:, -2, 1:], p[:, -2, 1:]-p[:, -3, 1:]
        rel = o-(a+u)[:, None]
        d = rel.norm(dim=-1, keepdim=True)
        scene = torch.cat(((a-HALF)/HALF, u/world.MAX_ACTION), -1)[:, None].expand(-1, world.K, -1)
        x = torch.cat(((o-HALF)/HALF, vo/3, vo2/3, scene, rel/HALF, d/HALF, ((d-world.REACT)/4).clamp(-2, 2), kind), -1)
        o_next = o+vo+3*self.obstacle(x)
        y = torch.cat(((a-HALF)/HALF, u/world.MAX_ACTION, (rel/HALF).flatten(1), (vo/3).flatten(1)), -1)
        return torch.cat(((a+u+3*self.agent(y))[:, None], o_next), 1)

    def rollout(self, p, kind, future):
        """p (B, 3, 4, 2), kind (B, K) int, future (B, steps, 2) px, to (B, steps, 4, 2)."""
        kind = F.one_hot(kind.long(), 3).float()*float(self.kinds)
        out = []
        for k in range(future.shape[1]):
            nxt = self.step(p, kind, future[:, k])
            out.append(nxt)
            p = torch.cat((p[:, 1:], nxt[:, None]), 1)
        return torch.stack(out, 1)


def train_oracle(root, variant, policy='interactive', steps=4000, batch=4096, lr=1e-3, seed=0):
    device = torch.device('cuda')
    torch.manual_seed(2026111650+seed)
    labels = Bench(root, variant, policy, 'train').labels
    positions = torch.from_numpy(np.concatenate((labels['agent'][:, :, None], labels['obstacle']), 2)).to(device)
    actions, kinds = torch.from_numpy(labels['actions']).to(device), torch.from_numpy(labels['kind']).to(device)
    net = OracleS(kinds=variant == 't1').to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(500, steps))
    g = torch.Generator(device=device).manual_seed(2026111651+seed)
    episodes, length = positions.shape[:2]
    for step in range(steps):
        e = torch.randint(0, episodes, (batch,), generator=g, device=device)
        t0 = torch.randint(HISTORY-1, length-STEPS, (batch,), generator=g, device=device)
        frames = t0[:, None]+torch.arange(1-HISTORY, STEPS+1, device=device)
        p = positions[e[:, None], frames]
        out = net.rollout(p[:, :HISTORY], kinds[e], actions[e[:, None], frames[:, HISTORY-1:-1]])
        loss = F.huber_loss(out, p[:, HISTORY:])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        if (step+1) % 500 == 0:
            print(f'oracle {variant} step {step+1}: loss {float(loss.detach()):.4f}', flush=True)
    return net.eval().requires_grad_(False)


def oracle_forecaster(net, batch=8192):
    device = next(net.parameters()).device

    @torch.no_grad()
    def forecast(bench, e, t0, future):
        p, kind = torch.from_numpy(bench.history(e, t0)).to(device), torch.from_numpy(bench.labels['kind'][e]).to(device)
        future = torch.from_numpy(np.asarray(future, np.float32)).to(device)
        return np.concatenate([net.rollout(p[i:i+batch], kind[i:i+batch], future[i:i+batch]).cpu().numpy()
                               for i in range(0, len(e), batch)])
    return forecast


# -- the head ---------------------------------------------------------------------------------------------------------

class ClsHead(nn.Module):
    """One latent (..., D) to the 4 objects' positions (..., 4, 2), in units of half the frame about its centre."""

    def __init__(self, latent, hidden=512):
        super().__init__()
        self.net = mlp(latent, hidden, 2*len(OBJECTS), layers=2)

    def forward(self, z):
        return self.net(z).unflatten(-1, (len(OBJECTS), 2))


class CrossBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm_q, self.norm_x, self.norm_m = LayerNorm(width), LayerNorm(width), LayerNorm(width)
        self.q, self.kv, self.out = nn.Linear(width, width), nn.Linear(width, 2*width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, q, x):
        split = lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k, v = self.kv(self.norm_x(x)).chunk(2, -1)
        a = F.scaled_dot_product_attention(split(self.q(self.norm_q(q))), split(k), split(v))
        q = q+self.out(a.transpose(1, 2).flatten(2))
        return q+self.mlp(self.norm_m(q))


class TokenHead(nn.Module):
    """A frame's tokens (..., S, D) to the 4 objects' positions (..., 4, 2): attention pooling, one learned query an
    object over the tokens (with a learned position embedding), two blocks."""

    def __init__(self, latent, tokens, width=128, heads=4, blocks=2):
        super().__init__()
        self.embed = nn.Linear(latent, width)
        self.position = nn.Parameter(torch.randn(tokens, width)*.02)
        self.query = nn.Parameter(torch.randn(len(OBJECTS), width)*.02)
        self.blocks = nn.ModuleList(CrossBlock(width, heads) for _ in range(blocks))
        self.out = nn.Sequential(LayerNorm(width), nn.Linear(width, 2))

    def forward(self, z):
        lead = z.shape[:-2]
        x = self.embed(z.flatten(0, -3))+self.position
        q = self.query.expand(len(x), -1, -1)
        for block in self.blocks:
            q = block(q, x)
        return self.out(q).unflatten(0, lead)


def build_head(config):
    return TokenHead(config.latent, config.tokens) if config.tokens > 1 else ClsHead(config.latent)


def train_head(model, root, variant, policy='interactive', steps=4000, batch=256, lr=1e-3, seed=0):
    """The head on encoded training frames (the JEPA frozen, in eval mode)."""
    device = next(model.parameters()).device
    torch.manual_seed(2026111660+seed)
    bench = Bench(root, variant, policy, 'train', device)
    positions = torch.from_numpy(bench.positions).to(device)
    head = build_head(model.config).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(200, steps))
    g = torch.Generator(device=device).manual_seed(2026111661+seed)
    episodes, length = positions.shape[:2]
    for step in range(steps):
        e = torch.randint(0, episodes, (batch,), generator=g, device=device)
        t = torch.randint(0, length, (batch,), generator=g, device=device)
        with torch.no_grad():
            z = model.encode(to_pixels(bench.frames[e, t])[:, None])[:, 0]
        loss = F.mse_loss(head(z), (positions[e, t]-HALF)/HALF)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        if (step+1) % 1000 == 0:
            print(f'head step {step+1}: rms {HALF*math.sqrt(float(loss.detach())):.3f} px', flush=True)
    return head.eval().requires_grad_(False)


def jepa_forecaster(model, head, batch=512):
    device = next(model.parameters()).device

    @torch.no_grad()
    def forecast(bench, e, t0, future):
        e, t0 = torch.from_numpy(e).to(device), torch.from_numpy(t0).to(device)
        future = torch.from_numpy(np.asarray(future, np.float32)).to(device)/world.MAX_ACTION
        out = []
        for i in range(0, len(e), batch):
            frames = t0[i:i+batch, None]+torch.arange(1-model.config.history, 1, device=device)
            rows = e[i:i+batch, None]
            z = model.encode(to_pixels(bench.frames[rows, frames]))
            actions = torch.cat((bench.actions[rows, frames[:, :-1]], future[i:i+batch]), 1)
            out.append((head(model.rollout(z, actions, STEPS))*HALF+HALF).cpu().numpy())
        return np.concatenate(out)
    return forecast


@torch.no_grad()
def retention(model, head, bench, batch=1024):
    """R: the head's error on every encoded validation frame, per object."""
    frames = bench.frames.flatten(0, 1)
    out = np.concatenate([(head(model.encode(to_pixels(frames[i:i+batch])[:, None])[:, 0])*HALF+HALF).cpu().numpy()
                          for i in range(0, len(frames), batch)])
    error = np.linalg.norm(out-bench.positions.reshape(-1, len(OBJECTS), 2), axis=-1).mean(0)
    return {name: float(v) for name, v in zip(OBJECTS, error)}


# -- the measures -----------------------------------------------------------------------------------------------------

def mean(values):
    return float(values.mean()) if len(values) else None


def measure(forecast, bench, seed=0, draws=1000, first=HISTORY-1):
    """E, X and A of a forecaster on a bench (the module's description), at each k of KS. A dict for JSON. `first`: the
    earliest last-history frame scored (a model with a longer history needs a later one; references are then scored
    from the same frame)."""
    labels, branches = bench.labels, bench.branches
    episodes, length = bench.positions.shape[:2]
    starts = np.arange(first, length-STEPS, 2)
    e, t0 = np.repeat(np.arange(episodes), len(starts)), np.tile(starts, episodes)
    steps = t0[:, None]+np.arange(STEPS)
    future = labels['actions'][e[:, None], steps]
    truth = bench.positions[e[:, None], steps+1]
    error = lambda f, tape: np.linalg.norm(f(bench, e, t0, tape)-truth, axis=-1)                 # (N, 16, 4)
    perm = np.random.default_rng(seed).permutation(len(e))
    err, shuffled, cv, still = error(forecast, future), error(forecast, future[perm]), error(constant_velocity, future), \
        error(hold, future)
    gap = np.linalg.norm(labels['obstacle']-labels['agent'][:, :, None], axis=-1)                # (E, T, K)
    row = dict(E={}, X={}, A={})
    for k in KS:
        j = k-1
        row['E'][k] = dict(error=mean(err[:, j, 0]), shuffled=mean(shuffled[:, j, 0]),
                           ratio=mean(shuffled[:, j, 0])/max(mean(err[:, j, 0]), 1e-9), cv=mean(cv[:, j, 0]),
                           hold=mean(still[:, j, 0]))
        far = gap[e[:, None], t0[:, None]+np.arange(k+1)].min(1) > 2*world.REACT                  # (N, K)
        bounced = labels['bounce'][e[:, None], steps[:, :k]+1].any(1)
        row['X'][k] = {name: dict(n=int(mask.sum()), error=mean(err[:, j, 1:][mask]), cv=mean(cv[:, j, 1:][mask]),
                                  hold=mean(still[:, j, 1:][mask]))
                       for name, mask in (('bounce', far & bounced), ('straight', far & ~bounced))}
    # A: the branch moments
    moments = branches['t0'].shape[1]
    e, m = np.repeat(np.arange(episodes), moments), np.tile(np.arange(moments), episodes)
    late = branches['t0'][e, m] >= first
    e, m = e[late], m[late]
    t0 = branches['t0'][e, m].astype(np.int64)
    predicted = np.stack([forecast(bench, e, t0, branches['actions'][e, m, tape])
                          for tape in range(len(world.TAPES))], 1)[:, :, :, 1:]                   # (N, 4, 16, K, 2)
    truth = branches['obstacle'][e, m].astype(np.float64)
    touched = np.cumsum(branches['contact'][e, m], 2) > 0                                        # (N, 4, 16, K)
    a, b = PAIRS.T
    shape = (len(e), len(PAIRS), world.K)
    group = np.broadcast_to(e[:, None, None], shape).reshape(-1)
    kind = np.broadcast_to(labels['kind'][e][:, None], shape).reshape(-1)
    bootstrap = LayoutBootstrap(np.arange(episodes), draws=draws)
    for k in KS:
        j = k-1
        true = (truth[:, b, j]-truth[:, a, j]).reshape(-1, 2)
        pred = (predicted[:, b, j].astype(np.float64)-predicted[:, a, j]).reshape(-1, 2)
        contact = (touched[:, a, j] | touched[:, b, j]).reshape(-1)
        tn, pn, err = np.linalg.norm(true, axis=-1), np.linalg.norm(pred, axis=-1), np.linalg.norm(pred-true, axis=-1)
        responsive, quiet = tn > RESPONSIVE, tn < QUIET

        def part(mask):
            r, q = mask & responsive, mask & quiet
            out = dict(pairs=int(mask.sum()), responsive=int(r.sum()), error=mean(err[r]), zero=mean(tn[r]),
                       quiet=int(q.sum()), false_rate=mean((pn[q] > RESPONSIVE).astype(float)), slope=None, ci=None)
            out['ratio'] = out['error']/out['zero'] if r.any() else None
            if mask.sum() > 10 and np.ptp(true[mask]) > 1e-6:
                fit = bootstrap.regression(true[mask], pred[mask], group[mask])
                out['slope'], out['ci'] = fit['slope'], fit['ci95']
            return out
        row['A'][k] = dict(all=part(np.ones(len(tn), bool)), contact=part(contact), distance=part(~contact),
                           **{name: part(kind == i) for i, name in enumerate(world.KINDS)})
    return row


def bars(row, k=8):
    """PLAN P27 item 4, on one row (one seed): each bar's pass or fail; R is None for a reference."""
    E, X, A = row['E'][k], row['X'][k], row['A'][k]['all']
    out = dict(E=E['error'] <= 1. and E['ratio'] >= 5.,
               X=X['bounce']['error'] <= .5*X['bounce']['cv'] and X['straight']['error'] <= X['straight']['cv']+.5,
               A=A['ratio'] <= .5 and A['false_rate'] <= .1 and A['ci'] is not None and A['ci'][0] > .5,
               R=max(row['R'].values()) <= 1. if 'R' in row else None)
    out['all'] = all(v for v in out.values() if v is not None)
    return out


def keyed(row):
    """A row read back from JSON, its k keys as integers again."""
    return {name: ({int(k): v for k, v in part.items()} if name in 'EXA' else part) for name, part in row.items()}


def table(rows, oracle=None, k=8):
    """A markdown table of rows {name: row} at k. With the oracle's row, the share of its gain over constant velocity
    (X, bounce windows) and over zero-response (A) is added."""
    f = lambda v, d=2: '-' if v is None else f'{v:.{d}f}'
    lines = ['| Run | E: agent px | shuffle ratio | X bounce px (cv) | X straight px (cv) | A: error / zero | false | '
             'slope [95%] | R: worst px | X share | A share | bars |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, row in rows.items():
        E, X, A, R = row['E'][k], row['X'][k], row['A'][k]['all'], row.get('R')
        shares = ['-', '-']
        if oracle is not None:
            ox, oa = oracle['X'][k]['bounce'], oracle['A'][k]['all']
            shares = [f((X['bounce']['cv']-X['bounce']['error'])/(ox['cv']-ox['error'])),
                      f((A['zero']-A['error'])/(oa['zero']-oa['error']))]
        ci = '-' if A['ci'] is None else f"[{A['ci'][0]:.2f}, {A['ci'][1]:.2f}]"
        passed = bars(row, k)
        mark = ' '.join(f"{name_}{'✓' if v else '✗'}" for name_, v in passed.items() if v is not None and name_ != 'all')
        lines.append(f"| {name} | {f(E['error'])} | {f(E['ratio'], 1)} | {f(X['bounce']['error'])} ({f(X['bounce']['cv'])}) "
                     f"| {f(X['straight']['error'])} ({f(X['straight']['cv'])}) | {f(A['error'])} / {f(A['zero'])} = "
                     f"{f(A['ratio'])} | {f(A['false_rate'])} | {f(A['slope'])} {ci} | "
                     f"{'-' if R is None else f(max(R.values()))} | {shares[0]} | {shares[1]} | {mark} |")
    return '\n'.join(lines)


# -- commands ---------------------------------------------------------------------------------------------------------

def suffix(first):
    return '' if first == HISTORY-1 else f'_from{first}'


def references(root, variant, out, policy='interactive', first=HISTORY-1):
    """Trains Oracle-S (or loads OUT/oracle_<variant>.pt if it is there) and scores the references."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(.3)
    if (out/f'oracle_{variant}.pt').exists():
        net = OracleS(kinds=variant == 't1').to('cuda')
        net.load_state_dict(torch.load(out/f'oracle_{variant}.pt')['state'])
        net.eval().requires_grad_(False)
    else:
        net = train_oracle(root, variant, policy)
        torch.save(dict(state=net.state_dict(), variant=variant), out/f'oracle_{variant}.pt')
    bench = Bench(root, variant, policy, 'val')
    rows = dict(hold=measure(hold, bench, first=first), cv=measure(constant_velocity, bench, first=first),
                oracle=measure(oracle_forecaster(net), bench, first=first))
    (out/f'references_{variant}{suffix(first)}.json').write_text(json.dumps(rows, indent=1))
    print(table(rows, rows['oracle']))


def score(root, run, policy='interactive', first=HISTORY-1):
    """The head (trained once, kept in RUN/head.pt), then the measures: RUN/bench.json, or RUN/bench_from<first>.json."""
    run, device = Path(run), torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.3)
    model, state = load(run/'model.pt', device)
    variant = state['variant']
    first = max(first, model.config.history-1)
    if (run/'head.pt').exists():
        head = build_head(model.config).to(device)
        head.load_state_dict(torch.load(run/'head.pt'))
        head.eval().requires_grad_(False)
    else:
        head = train_head(model, root, variant, policy)
        torch.save(head.state_dict(), run/'head.pt')
    bench = Bench(root, variant, policy, 'val', device)
    row = measure(jepa_forecaster(model, head), bench, first=first)
    row['R'] = retention(model, head, bench)
    row['meta'] = dict(recipe=state['recipe'], variant=variant, seed=state['seed'], steps=state['step'])
    (run/f'bench{suffix(first)}.json').write_text(json.dumps(row, indent=1))
    print(table({f"{state['recipe']} {variant}": row}))
    print('R:', {k: round(v, 2) for k, v in row['R'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('references')
    r.add_argument('--data', required=True)
    r.add_argument('--variant', choices=['t1', 't2'], required=True)
    r.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--data', required=True)
    s.add_argument('--run', required=True)
    t = sub.add_parser('table')
    for command in (r, s, t):
        command.add_argument('--first', type=int, default=HISTORY-1, help='the earliest last-history frame scored')
    t.add_argument('--references', required=True)
    t.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args()
    if args.command == 'references':
        references(args.data, args.variant, args.out, first=args.first)
    elif args.command == 'score':
        score(args.data, args.run, first=args.first)
    else:
        name = f'bench{suffix(args.first)}.json'
        rows = {Path(run).name: keyed(json.loads((Path(run)/name).read_text())) for run in args.runs}
        for variant in ('t1', 't2'):
            part = {name: row for name, row in rows.items() if row['meta']['variant'] == variant}
            if not part:
                continue
            refs = Path(args.references)/f'references_{variant}{suffix(args.first)}.json'
            refs = {k: keyed(v) for k, v in json.loads(refs.read_text()).items()}
            named = {'constant velocity': refs['cv'], 'Oracle-S': refs['oracle'], **part}
            print(f'\n**{variant.upper()}**\n\n'+table(named, refs['oracle']))


if __name__ == '__main__':
    main()
