"""P34 (docs/dynamic/PLAN.md P34): do the toy's forecasts make better decisions? One fixed planner, several forecasters.

    python -m go2nav.eval.toy_planning run --forecaster truth|oracle|cv|jepa [--run RUN] --out FILE [--episodes 200]

**The task** (`tasks`): the standard reactive-ball world, kinds hidden. The agent starts at the left edge and must
reach the goal straight across at the right edge within `LIMIT` steps (19 at full speed); the three obstacles start
within `BAND_PX` of that line (x 20-44 px), cruising, with kinds drawn as in the data. Success: within `REACHED_PX` of the goal; a contact is the agent touching an
obstacle. The score is success without any contact.

**The planner** (`plan`), fixed and the same for every forecaster: CEM over `HORIZON`-step tapes of displacements
(`SAMPLES` tapes, `ITERATIONS` rounds, `ELITES` kept, norms clipped to 3 px), replanning every step and executing the
first action. A tape's cost is the agent's predicted distance to the goal at its end plus `CLEARANCE_COST` times the
sum over steps and obstacles of how far the predicted gap falls under `CLEARANCE_PX`. Two warm-up steps at rest give
every forecaster its 3-frame history.

**Forecasters** (`forecasters`): each maps the history (the last 3 frames and states) and the candidate tapes to the
agent's and the obstacles' positions after each step:
- `truth`: the simulator itself from a copy of the state (hidden kinds and timers included): the ceiling;
- `oracle`: Oracle-S on true positions (`eval.recipe_bench.OracleS`, T2's);
- `cv`: the agent moved by its actions (walls applied), each obstacle on at its last displacement;
- `jepa`: an object-token JEPA from frames, read by its head (Q0's per-token MLP).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.sim import reactive_balls as world
from go2nav.sim.balls import LOW, HIGH, MAX_ACTION, RADIUS, clip_actions

LIMIT, REACHED_PX, HORIZON, SAMPLES, ITERATIONS, ELITES = 28, 2., 12, 64, 3, 8
BAND_PX = 6.                   # obstacles start within this of the straight line to the goal
CLEARANCE_PX, CLEARANCE_COST, SEED = 2*RADIUS+1.5, 10., 2026111834


def tasks(rng, n):
    """`n` task states and goals: the agent at the left edge, the goal at the right, obstacles across the way."""
    state = world.reset(rng, n)
    y = rng.uniform(16, 48, n)
    state.agent = np.stack((np.full(n, LOW+1.), y), -1)
    state.obstacle = np.stack((rng.uniform(20, 44, (n, world.K)), y[:, None]+rng.uniform(-BAND_PX, BAND_PX, (n, world.K))), -1)
    goal = np.stack((np.full(n, HIGH-1.), y), -1)
    return state, goal


def cost(agent, obstacles, goal, margin=None):
    """Tapes' predicted agent (N, C, H, 2) and obstacles (N, C, H, K, 2), goals (N, 2) to costs (N, C). P34b: `margin`
    (H,) px added to the clearance at each step (the forecaster's own error there)."""
    reach = np.linalg.norm(agent[:, :, -1]-goal[:, None], axis=-1)
    gap = np.linalg.norm(obstacles-agent[..., None, :], axis=-1)
    clearance = CLEARANCE_PX+(0. if margin is None else np.asarray(margin)[None, None, :gap.shape[2], None])
    return reach+CLEARANCE_COST*np.clip(clearance-gap, 0, None).sum((-1, -2))


def plan(forecast, history, goal, rng, margin=None):
    """CEM: the first action of the best tape for each of N tasks (N, 2)."""
    n = len(goal)
    mean, std = np.zeros((n, HORIZON, 2)), np.full((n, HORIZON, 2), 2.)
    to = goal-history['agent'][:, -1]
    mean[:] = (MAX_ACTION*to/np.maximum(np.linalg.norm(to, axis=-1, keepdims=True), 1e-9))[:, None]
    best, best_cost = mean.copy(), np.full(n, np.inf)
    for _ in range(ITERATIONS):
        tapes = clip_actions(mean[:, None]+std[:, None]*rng.normal(size=(n, SAMPLES, HORIZON, 2)))
        tapes[:, 0] = clip_actions(mean)
        agent, obstacles = forecast(history, tapes)
        c = cost(agent, obstacles, goal, margin)
        order = np.argsort(c, 1)[:, :ELITES]
        elite = np.take_along_axis(tapes, order[:, :, None, None], 1)
        mean, std = elite.mean(1), elite.std(1)+.1
        better = c[np.arange(n), order[:, 0]] < best_cost
        best[better] = tapes[np.arange(n), order[:, 0]][better]
        best_cost = np.minimum(best_cost, c[np.arange(n), order[:, 0]])
    return best[:, 0]


# -- forecasters ------------------------------------------------------------------------------------------------------

def truth_forecaster(states):
    """The simulator from copies of each task's current state."""
    def forecast(history, tapes):
        n, c, h, _ = tapes.shape
        s = states[0].take(np.repeat(np.arange(n), c))
        agent, obstacles = [], []
        for k in range(h):
            s = world.step(s, tapes[:, :, k].reshape(n*c, 2))
            agent.append(s.agent), obstacles.append(s.obstacle)
        return (np.stack(agent, 1).reshape(n, c, h, 2), np.stack(obstacles, 1).reshape(n, c, h, world.K, 2))
    return forecast


def agent_path(agent, tapes):
    out, xy = [], np.repeat(agent[:, None], tapes.shape[1], 1)
    for k in range(tapes.shape[2]):
        xy = np.clip(xy+tapes[:, :, k], LOW, HIGH)
        out.append(xy)
    return np.stack(out, 2)


def cv_forecaster():
    def forecast(history, tapes):
        o, prev = history['obstacle'][:, -1], history['obstacle'][:, -2]
        steps = np.arange(1, tapes.shape[2]+1)[None, None, :, None, None]
        obstacles = np.broadcast_to(o[:, None, None]+steps*(o-prev)[:, None, None], (*tapes.shape[:3], world.K, 2))
        return agent_path(history['agent'][:, -1], tapes), obstacles
    return forecast


def oracle_forecaster(path, device):
    from go2nav.eval.recipe_bench import OracleS
    net = OracleS(kinds=False).to(device)
    net.load_state_dict(torch.load(path, map_location='cpu')['state'])
    net.eval()

    @torch.no_grad()
    def forecast(history, tapes):
        n, c, h, _ = tapes.shape
        p = np.concatenate((history['agent'][:, :, None], history['obstacle']), 2)              # (N, 3, 4, 2)
        p = torch.as_tensor(np.repeat(p, c, 0), dtype=torch.float32, device=device)
        out = net.rollout(p, torch.zeros(n*c, world.K, device=device),
                          torch.as_tensor(tapes.reshape(n*c, h, 2), dtype=torch.float32, device=device)).cpu().numpy()
        out = out.reshape(n, c, h, 4, 2)
        return out[:, :, :, 0], out[:, :, :, 1:]
    return forecast


def jepa_forecaster(run, device, chunk=8192):
    from go2nav.eval import recipe_bench as rb
    from go2nav.training.balls import to_pixels
    from go2nav.training.recipe_bench import load
    model, _ = load(Path(run)/'model.pt', device)
    head = rb.build_head(model.config, 'mlp').to(device)
    head.load_state_dict(torch.load(Path(run)/'head_mlp.pt', map_location='cpu'))
    head.eval()

    @torch.no_grad()
    def forecast(history, tapes):
        n, c, h, _ = tapes.shape
        frames = torch.as_tensor(history['frames'], device=device)                              # (N, 3, 64, 64, 3)
        z = model.encode(to_pixels(frames))                                                     # (N, 3, 4, D)
        past = torch.as_tensor(history['actions'], dtype=torch.float32, device=device)/MAX_ACTION   # (N, 2, 2)
        z, past = z.repeat_interleave(c, 0), past.repeat_interleave(c, 0)
        future = torch.as_tensor(tapes.reshape(n*c, h, 2), dtype=torch.float32, device=device)/MAX_ACTION
        out = []
        for i in range(0, n*c, chunk):            # bf16 autocast, as the model was trained; the head in float32
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                rolled = model.rollout(z[i:i+chunk], torch.cat((past[i:i+chunk], future[i:i+chunk]), 1), h)
            out.append((head(rolled.float())*rb.HALF+rb.HALF).cpu().numpy())
        out = np.concatenate(out).reshape(n, c, h, 4, 2)
        return out[:, :, :, 0], out[:, :, :, 1:]
    return forecast


def run(name, episodes, out, run_dir=None, batch=100, margins=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.3)
    rng = np.random.default_rng(SEED)
    margin = None if margins is None else json.loads(Path(margins).read_text())[name]['p90']        # P34b
    results = []
    for start in range(0, episodes, batch):
        n = min(batch, episodes-start)
        state, goal = tasks(np.random.default_rng((SEED, start)), n)
        rows = [world.record(state)]
        actions = []
        for _ in range(2):                                                                      # the warm-up at rest
            state = world.step(state, np.zeros((n, 2)))
            rows.append(world.record(state)), actions.append(np.zeros((n, 2)))
        forecast = dict(truth=lambda: truth_forecaster([state]), cv=cv_forecaster,
                        oracle=lambda: oracle_forecaster('outputs/recipe_bench/references/oracle_t2.pt', device),
                        jepa=lambda: jepa_forecaster(run_dir, device))[name]()
        done, contacts, reached = np.zeros(n, bool), np.zeros(n, int), np.full(n, -1)
        for t in range(LIMIT):
            history = dict(agent=np.stack([r['agent'] for r in rows[-3:]], 1),
                           obstacle=np.stack([r['obstacle'] for r in rows[-3:]], 1),
                           actions=np.stack(actions[-2:], 1))
            if name == 'jepa':
                history['frames'] = world.render(history['agent'].reshape(-1, 2),
                                                 history['obstacle'].reshape(-1, world.K, 2)).reshape(n, 3, 64, 64, 3)
            if name == 'truth':
                forecast = truth_forecaster([state])
            a = np.where(done[:, None], 0., plan(forecast, history, goal, rng, margin))
            state = world.step(state, a)
            rows.append(world.record(state)), actions.append(a)
            contacts += (state.contact.any(-1) & ~done)
            hit = ~done & (np.linalg.norm(state.agent-goal, axis=-1) < REACHED_PX)
            reached[hit] = t+1
            done |= hit
        for i in range(n):
            results.append(dict(reached=int(reached[i]), contacts=int(contacts[i]), kinds=state.kind[i].tolist()))
        print(f'{name}: {start+n} episodes', flush=True)
    summary = dict(forecaster=name, run=run_dir, margins=margin, episodes=len(results),
                   success=float(np.mean([r['reached'] > 0 for r in results])),
                   clean_success=float(np.mean([r['reached'] > 0 and r['contacts'] == 0 for r in results])),
                   contacts=float(np.mean([r['contacts'] for r in results])),
                   steps=float(np.mean([r['reached'] for r in results if r['reached'] > 0] or [np.nan])))
    Path(out).write_text(json.dumps(dict(summary=summary, episodes=results), indent=1))
    print(json.dumps(summary), flush=True)


def calibrate(out, jepa_run, near=2*world.REACT):
    """P34b: each forecaster's error in the agent-obstacle gap, 90th percentile at each step 1..HORIZON, for obstacles
    within `near` px of the agent at the window's last frame, on the standard toy's validation windows (T2) under
    the recorded actions. The simulator's is zero."""
    from go2nav.eval import recipe_bench as rb
    from go2nav.training.recipe_bench import load
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.3)
    bench = rb.Bench('outputs/data/reactive_balls', 't2', 'interactive', 'val', device)
    net = rb.OracleS(kinds=False).to(device)
    net.load_state_dict(torch.load('outputs/recipe_bench/references/oracle_t2.pt', map_location='cpu')['state'])
    model, _ = load(Path(jepa_run)/'model.pt', device)
    head = rb.build_head(model.config, 'mlp').to(device)
    head.load_state_dict(torch.load(Path(jepa_run)/'head_mlp.pt', map_location='cpu'))
    head.eval()

    def cv(bench, e, t0, future):            # the planner's constant velocity: the agent moved by its actions
        out = rb.constant_velocity(bench, e, t0, future)
        out[:, :, 0] = agent_path(bench.positions[e, t0, 0], future[:, None])[:, 0]
        return out
    forecasters = dict(cv=cv, oracle=rb.oracle_forecaster(net.eval()), jepa=rb.jepa_forecaster(model, head))
    episodes, length = bench.positions.shape[:2]
    starts = np.arange(2, length-rb.STEPS, 2)
    e, t0 = np.repeat(np.arange(episodes), len(starts)), np.tile(starts, episodes)
    steps = t0[:, None]+np.arange(rb.STEPS)
    future = bench.labels['actions'][e[:, None], steps]
    truth = bench.positions[e[:, None], steps+1]                                              # (N, 16, 4, 2)
    near_now = np.linalg.norm(bench.positions[e, t0, 1:]-bench.positions[e, t0, :1], axis=-1) < near   # (N, K)
    true_gap = truth[:, :, 1:]-truth[:, :, :1]
    result = dict(truth=dict(p90=[0.]*HORIZON))
    for name, f in forecasters.items():
        pred = f(bench, e, t0, future)
        err = np.linalg.norm((pred[:, :, 1:]-pred[:, :, :1])-true_gap, axis=-1)                # (N, 16, K)
        result[name] = dict(p90=[float(np.quantile(err[:, k][near_now], .9)) for k in range(HORIZON)],
                            median=[float(np.median(err[:, k][near_now])) for k in range(HORIZON)],
                            count=int(near_now.sum()))
        print(name, 'p90 at k = 1, 4, 8, 12:', [round(result[name]['p90'][k], 2) for k in (0, 3, 7, 11)], flush=True)
    Path(out).write_text(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--forecaster', choices=['truth', 'oracle', 'cv', 'jepa'], required=True)
    r.add_argument('--run')
    r.add_argument('--out', required=True)
    r.add_argument('--episodes', type=int, default=200)
    r.add_argument('--margins', help="P34b: the calibrated margins' file (`calibrate`)")
    c = sub.add_parser('calibrate')
    c.add_argument('--jepa', required=True)
    c.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'calibrate':
        calibrate(args.out, args.jepa)
    else:
        run(args.forecaster, args.episodes, args.out, args.run, margins=args.margins)


if __name__ == '__main__':
    main()
