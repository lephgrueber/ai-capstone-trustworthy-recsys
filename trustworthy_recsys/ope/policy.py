"""Small inspectable categorical-context click model, fitted on training only."""

import numpy as np


def fit_reward_model(frame, columns, n_actions, shrinkage=100.):
    if shrinkage <= 0 or not columns or frame.empty:
        raise ValueError('Need contexts, training events and positive shrinkage')
    action = frame.action_index.to_numpy(int)
    reward = frame.reward.to_numpy(float)
    count = np.bincount(action, minlength=n_actions)
    clicks = np.bincount(action, weights=reward, minlength=n_actions)
    global_ctr = (reward.sum()+1)/(len(reward)+2)
    prior = (clicks + shrinkage*global_ctr)/(count+shrinkage)
    model = {'kind': 'mean of smoothed per-feature/action click rates',
             'shrinkage': shrinkage, 'n_actions': n_actions,
             'columns': list(columns), 'prior': prior.tolist(), 'tables': {}}
    for column in columns:
        table = {}
        for key, group in frame.groupby(column, sort=True):
            indices = group.action_index.to_numpy(int)
            count = np.bincount(indices, minlength=n_actions)
            clicks = np.bincount(indices, weights=group.reward, minlength=n_actions)
            table[str(key)] = ((clicks + shrinkage*prior)/(count+shrinkage)).tolist()
        model['tables'][column] = table
    return model


def predict_rewards(model, frame):
    result = np.zeros((len(frame), model['n_actions']), dtype=float)
    for column in model['columns']:
        table = model['tables'][column]
        unique, inverse = np.unique(frame[column].astype(str), return_inverse=True)
        values = np.array([table.get(value, model['prior']) for value in unique])
        result += values[inverse]
    return result / len(model['columns'])


def policy_probabilities(q, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError('epsilon must lie in [0, 1]')
    probabilities = np.full(q.shape, epsilon / q.shape[1], dtype=float)
    probabilities[np.arange(len(q)), np.argmax(q, axis=1)] += 1-epsilon
    return probabilities
