"""DM, IPS, SNIPS and DR with explicit probability validation.

Formulas follow the standard contextual-bandit definitions also documented in
https://github.com/st-tech/zr-obp/blob/master/obp/ope/estimators.py .
This module does not depend on OBP or implement joint-slate OPE.
"""

import numpy as np


def contributions(reward, action, propensity, probabilities, predicted_rewards):
    reward, propensity = np.asarray(reward, float), np.asarray(propensity, float)
    action = np.asarray(action)
    pi, q = np.asarray(probabilities, float), np.asarray(predicted_rewards, float)
    if pi.ndim != 2 or q.shape != pi.shape or len(pi) == 0:
        raise ValueError('Expected nonempty [events, actions] arrays')
    n, k = pi.shape
    if any(a.shape != (n,) for a in [reward, action, propensity]):
        raise ValueError('Misaligned event arrays')
    if not np.issubdtype(action.dtype, np.integer) or ((action < 0) | (action >= k)).any():
        raise ValueError('Invalid action index')
    if not all(np.isfinite(a).all() for a in [reward, propensity, pi, q]):
        raise ValueError('Nonfinite OPE input')
    if not np.isin(reward, [0, 1]).all() or ((propensity <= 0) | (propensity > 1)).any():
        raise ValueError('Invalid reward or propensity')
    if (pi < 0).any() or not np.allclose(pi.sum(axis=1), 1, atol=1e-10, rtol=1e-8):
        raise ValueError('Target probabilities must sum to one')
    if ((q < 0) | (q > 1)).any():
        raise ValueError('Click predictions must lie in [0, 1]')
    row = np.arange(n)
    target = pi[row, action]
    weight = target / propensity
    dm = (pi * q).sum(axis=1)
    return {'target_probability': target, 'weight': weight, 'reward': reward,
            'q_logged': q[row, action], 'dm': dm, 'ips': weight * reward,
            'dr': dm + weight * (reward - q[row, action])}


def point_estimates(c):
    mass = float(np.sum(c['weight']))
    if mass <= 0:
        raise ValueError('No observed target-policy support')
    return {'dm': float(np.mean(c['dm'])), 'ips': float(np.mean(c['ips'])),
            'snips': float(np.sum(c['ips']) / mass), 'dr': float(np.mean(c['dr']))}


def summarize(c, blocks, repetitions=1000, seed=42):
    """Resample whole observed time blocks; intervals conditional on fitted policy.

    Blocks need not have equal event counts. SNIPS is recomputed in each draw.
    This is a sensitivity-oriented cluster bootstrap, not a guarantee against
    repeated-user dependence or nonstationarity.
    """
    if repetitions < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    keys, inverse = np.unique(np.asarray(blocks), return_inverse=True)
    if len(inverse) != len(c['weight']) or len(keys) < 2:
        raise ValueError('Need at least two aligned time blocks')
    sums = np.column_stack([np.bincount(inverse, weights=v, minlength=len(keys)) for v in
                            [np.ones(len(inverse)), c['dm'], c['ips'], c['dr'], c['weight']]])
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, 4))
    for i in range(repetitions):
        total = sums[rng.integers(len(keys), size=len(keys))].sum(axis=0)
        draws[i] = [total[1]/total[0], total[2]/total[0],
                    total[2]/total[4] if total[4] > 0 else np.nan, total[3]/total[0]]
    if not np.isfinite(draws).all():
        raise ValueError('Bootstrap has zero support; estimates are unstable')
    w = c['weight']
    values = point_estimates(c)
    result = {'estimates': {name: {'value': values[name],
                                  'ci95': np.quantile(draws[:, i], [.025, .975]).tolist()}
                            for i, name in enumerate(['dm', 'ips', 'snips', 'dr'])},
              'rows': len(w), 'clicks': int(np.sum(c['reward'])),
              'observed_ctr': float(np.mean(c['reward'])), 'time_blocks': len(keys),
              'bootstrap_repetitions': repetitions, 'seed': seed,
              'effective_sample_size': float(w.sum()**2 / np.dot(w, w)),
              'mean_weight': float(w.mean()), 'max_weight': float(w.max()),
              'weight_quantiles': dict(zip(['p50', 'p90', 'p99'], np.quantile(w, [.5, .9, .99]).tolist())),
              'zero_weight_fraction': float(np.mean(w == 0)),
              'top_one_percent_weight_share': float(np.sort(w)[-max(1, int(np.ceil(.01*len(w)))):].sum()/w.sum()),
              'clipping_sensitivity': {}}
    for cap in [5., 10., 20.]:
        clipped = np.minimum(w, cap)
        weighted = clipped*c['reward']
        result['clipping_sensitivity'][str(int(cap))] = {
            'clipped_fraction': float(np.mean(w > cap)),
            'ips': float(weighted.mean()), 'snips': float(weighted.sum()/clipped.sum()),
            'dr': float(np.mean(c['dm']+clipped*(c['reward']-c['q_logged'])))}
    return result
