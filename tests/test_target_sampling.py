"""Training-target caps and rotation (M3 feedback §6)."""

import numpy as np
import pytest

from trustworthy_recsys.retrieval.train import TargetSampler


USERS = np.repeat([0, 1, 2], [10, 3, 1])


def test_spaced_matches_preparation_rule():
    sampler = TargetSampler(USERS, cap=4, mode="spaced")
    chosen = sampler.epoch(0)
    assert chosen.tolist() == [0, 3, 6, 9, 10, 11, 12, 13]
    assert np.array_equal(sampler.epoch(3), chosen)


def test_fixed_and_rotate_share_first_draw_then_rotate():
    fixed = TargetSampler(USERS, cap=2, mode="fixed", seed=7)
    rotate = TargetSampler(USERS, cap=2, mode="rotate", seed=7)
    assert np.array_equal(fixed.epoch(0), rotate.epoch(0)) and np.array_equal(fixed.epoch(0), fixed.epoch(5))
    draws = [rotate.epoch(e) for e in range(4)]
    assert any(not np.array_equal(draws[0], d) for d in draws[1:])
    for draw in draws:
        assert np.bincount(USERS[draw], minlength=3).tolist() == [2, 2, 1]  # cap, no repeats
        assert len(set(draw)) == len(draw)


def test_rotation_expected_frequency_and_validation():
    targets = np.array([5] * 10 + [6] * 3 + [5])
    rotate = TargetSampler(USERS, cap=2, mode="rotate")
    expected = rotate.expected_frequency(targets, 8, epochs=4)
    assert expected[5] == pytest.approx(10 * 2 / 10 + 1) and expected[6] == pytest.approx(2)
    with pytest.raises(ValueError):
        TargetSampler(USERS, cap=0, mode="fixed")
    with pytest.raises(ValueError):
        TargetSampler(np.array([1, 0]), cap=1, mode="fixed")
