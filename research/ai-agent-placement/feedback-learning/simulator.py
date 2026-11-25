"""Finite-population realization of the transition rule in ../solve.m.

Independent opponent draws, synchronous updates, N humans; AI ratio r=.3.
The p0=0 top boundary goes down with probability one, exactly as MATLAB.
"""
import numpy as np

EPSILON = 0.02
AI_RATIO = 0.3
LEVELS = np.arange(6)


def transition(x, params, action):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    params = np.broadcast_to(np.asarray(params), (len(x), 2))
    action = np.broadcast_to(np.asarray(action, dtype=int), (len(x),))
    pool = x[:, 1:].copy()
    active = action > 0
    pool[np.where(active)[0], action[active] - 1] += AI_RATIO
    total = pool.sum(1, keepdims=True)
    weights = np.divide(pool, total, out=np.full_like(pool, .2), where=total > 1e-12)
    gaps = abs(np.arange(1, 5)[:, None] - np.arange(1, 6)[None, :])
    logits = params[:, 0, None, None] * (gaps - params[:, 1, None, None])
    down = np.sum(weights[:, None, :] / (1 + np.exp(-logits)), axis=2)
    mat = np.zeros((len(x), 6, 6))
    mat[:, 0, 0] = 1 - EPSILON
    mat[:, 0, 1] = EPSILON
    mat[:, 5, 4] = 1
    for h in range(1, 5):
        mat[:, h, h-1] = down[:, h-1]
        mat[:, h, h+1] = 1-down[:, h-1]
    return mat


def sample_step(counts, params, action, rng):
    """Batched multinomial updates, specialized to the two allowed destinations."""
    counts = np.asarray(counts, dtype=np.int64)
    mat = transition(counts/counts.sum(1, keepdims=True), params, action)
    flows = np.zeros((len(counts), 6, 6), dtype=np.int64)
    entrants = rng.binomial(counts[:, 0], EPSILON)
    flows[:, 0, 0] = counts[:, 0]-entrants
    flows[:, 0, 1] = entrants
    flows[:, 5, 4] = counts[:, 5]
    for h in range(1, 5):
        down = rng.binomial(counts[:, h], mat[:, h, h-1])
        flows[:, h, h-1] = down
        flows[:, h, h+1] = counts[:, h]-down
    return flows.sum(1), flows


def initial_counts(x, n):
    raw = x*n
    counts = np.floor(raw).astype(int)
    for k in range(len(x)):
        counts[k, np.argsort(-(raw[k]-counts[k]))[:n-counts[k].sum()]] += 1
    return counts


def trajectories(params, x0, actions, n, seed):
    rng = np.random.default_rng(seed)
    counts = initial_counts(x0, n)
    states = [counts/n]
    flows = []
    for t in range(actions.shape[1]):
        counts, flow = sample_step(counts, params, actions[:, t], rng)
        states.append(counts/n)
        flows.append(flow)
    return np.stack(states, 1), np.stack(flows, 1)


def deterministic(params, x0, actions):
    states = [np.asarray(x0, dtype=float)]
    for t in range(actions.shape[1]):
        x = states[-1]
        states.append(np.einsum('bi,bij->bj', x, transition(x, params, actions[:, t])))
    return np.stack(states, 1)


def make_split(n_scenarios, horizon, n, seed):
    rng = np.random.default_rng(seed)
    params = np.column_stack((rng.choice([-1, 1], n_scenarios)*rng.uniform(.2, 1, n_scenarios),
                              rng.uniform(0, 5, n_scenarios)))
    x0 = rng.dirichlet(np.ones(6)*2, n_scenarios)
    params = np.repeat(params, 6, axis=0)
    x0 = np.repeat(x0, 6, axis=0)
    actions = np.tile(np.arange(6), n_scenarios)[:, None].repeat(horizon, 1)
    states, flows = trajectories(params, x0, actions, n, seed+100)
    return dict(params=params.astype('float32'), actions=actions,
                states=states.astype('float32'), flows=flows, n=np.array(n),
                scenario=np.repeat(np.arange(n_scenarios), 6))


def verify():
    """Independent scalar reference, population conservation and sampling check."""
    rng = np.random.default_rng(987)
    xs = rng.dirichlet(np.ones(6), 32)
    pars = np.column_stack((rng.uniform(-1, 1, 32), rng.uniform(0, 5, 32)))
    acts = rng.integers(0, 6, 32)
    batch = transition(xs, pars, acts)
    refs = []
    for x, (alpha, dc), a in zip(xs, pars, acts):
        y = np.eye(6)[a]
        w = (x[1:]+AI_RATIO*y[1:])/(sum(x[1:])+AI_RATIO*sum(y[1:]))
        t = np.zeros((6, 6)); t[0, :2] = [1-EPSILON, EPSILON]
        for i in range(1, 6):
            p = np.zeros(3)
            for j in range(1, 6):
                s = 1/(1+np.exp(-alpha*(abs(j-i)-dc)))
                pm, pz, pp = s, 0., 1-s
                if i == 5:
                    pp = 0.; pm = pm/(pm+pz)
                p += w[j-1]*np.array([pm, pz, pp])
            if i == 5: t[i, 4:] = p[:2]
            else: t[i, i-1:i+2] = p
        refs.append(t)
    error = float(np.max(abs(batch-np.array(refs))))
    assert error < 1e-12
    assert np.allclose(batch.sum(2), 1)
    # Inactive AI is identical to r=0 because it contributes no active mass.
    assert np.allclose(transition([[1,0,0,0,0,0]], [1,2], 0).sum(2), 1)
    c = initial_counts(xs[:1], 500)
    draws, flows = sample_step(np.repeat(c, 20000, axis=0), pars[:1], acts[0], rng)
    assert np.all(draws.sum(1) == 500)
    assert np.all(flows.sum(2) == c)
    target = c[0]/500 @ transition(c/500, pars[:1], acts[0])[0]
    mc_error = float(abs(draws.mean(0)/500-target).max())
    assert mc_error < .002
    # Explicit individual-player implementation checks the intermediate opponent draw.
    x = c[0]/500; a = acts[0]; alpha, dc = pars[0]
    pool=x[1:].copy()
    if a: pool[a-1] += AI_RATIO
    pool /= pool.sum()
    h = rng.choice(6, 200000, p=x)
    j = rng.choice(np.arange(1,6), len(h), p=pool)
    s = 1/(1+np.exp(-alpha*(abs(j-h)-dc)))
    next_h = np.where(rng.random(len(h)) < s, h-1, h+1)
    next_h[h==5] = 4
    next_h[h==0] = (rng.random((h==0).sum()) < EPSILON).astype(int)
    individual_error = float(abs(np.bincount(next_h,minlength=6)/len(h)-target).max())
    assert individual_error < .005
    return dict(scalar_reference_max_error=error, monte_carlo_max_error=mc_error,
                individual_reference_max_error=individual_error, checks_passed=True)
