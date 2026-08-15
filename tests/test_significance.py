"""Verification of the paired significance tests.

McNemar is easy to get subtly wrong -- the usual mistakes are including the
concordant pairs, or using the chi-square approximation at counts too small for
it. Both inflate significance, which is the direction that would flatter this
project's results. Pinned against hand-computable cases.
"""

from __future__ import annotations

import pytest

from analysis.significance import bootstrap_token_diff, compare, mcnemar_exact


def test_no_disagreement_is_no_evidence():
    """Two policies that never differ cannot be distinguished, whatever their accuracy."""
    assert mcnemar_exact(0, 0) == 1.0


def test_symmetric_counts_give_p_one():
    """Equal disagreement both ways is exactly the null."""
    assert mcnemar_exact(7, 7) == pytest.approx(1.0)
    assert mcnemar_exact(1, 1) == pytest.approx(1.0)


def test_single_lopsided_pair_is_the_coin_flip():
    """One disagreement, one way: two-sided p = 1.0 (a single flip proves nothing).

    2 * P(X <= 0) with n=1 is 2 * 0.5 = 1.0.
    """
    assert mcnemar_exact(1, 0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "only_a,only_b,expected",
    [
        # n=5 all one way: 2 * (1/32) = 0.0625 -- still not significant, which is
        # the point: five items cannot carry a claim.
        (5, 0, 0.0625),
        # n=6 all one way: 2 * (1/64) = 0.03125
        (6, 0, 0.03125),
        # n=10, 9-1 split: 2 * (C(10,0)+C(10,1))/1024 = 2*11/1024
        (9, 1, 2 * 11 / 1024),
    ],
)
def test_known_exact_values(only_a, only_b, expected):
    assert mcnemar_exact(only_a, only_b) == pytest.approx(expected)


def test_direction_does_not_change_p():
    """The test is two-sided, so swapping the policies must not move the p-value."""
    assert mcnemar_exact(9, 2) == pytest.approx(mcnemar_exact(2, 9))


def test_p_never_exceeds_one():
    """Doubling the tail can overshoot at near-even splits if not clamped."""
    for a in range(0, 12):
        for b in range(0, 12):
            assert 0.0 <= mcnemar_exact(a, b) <= 1.0


def test_concordant_pairs_are_excluded():
    """Adding items both policies get right must not change the p-value.

    This is the mistake the test exists to catch: including concordant pairs
    makes every comparison look more certain than it is.
    """
    runs = {
        "a": {},
        "b": {},
    }
    # Six disagreements, all favouring 'a'.
    for i in range(6):
        runs["a"][("cat", i)] = {"correct": 1.0, "tokens": 100, "think_tokens": 0, "did_think": 0.0}
        runs["b"][("cat", i)] = {"correct": 0.0, "tokens": 100, "think_tokens": 0, "did_think": 0.0}
    lean = compare("a", "b", runs).p_value

    # Now add 200 items both get right.
    for i in range(100, 300):
        for label in ("a", "b"):
            runs[label][("cat", i)] = {
                "correct": 1.0, "tokens": 100, "think_tokens": 0, "did_think": 0.0
            }
    padded = compare("a", "b", runs)

    assert padded.p_value == pytest.approx(lean)
    assert (padded.only_a, padded.only_b) == (6, 0)
    # Accuracy moves a lot even though the evidence does not.
    assert padded.acc_a > 0.9 and padded.acc_b > 0.9


def test_bootstrap_ci_brackets_a_known_difference():
    paired = [(200.0, 100.0)] * 50
    low, high = bootstrap_token_diff(paired, iterations=500)
    # A constant difference has no sampling variability.
    assert low == pytest.approx(100.0)
    assert high == pytest.approx(100.0)


def test_bootstrap_ci_contains_zero_for_noise():
    """Symmetric noise around zero must not produce a 'significant' token gap."""
    paired = [(100.0 + d, 100.0) for d in (-30, 30, -20, 20, -10, 10, 0, 5, -5, 1)]
    low, high = bootstrap_token_diff(paired, iterations=2000)
    assert low < 0 < high
