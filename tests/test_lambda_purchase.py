"""Verification of the lambda-purchase analysis.

The claim this module supports — that a length penalty cannot influence the
gradient on a group whose correctness and format are constant — is a headline
result, so the classification it rests on is pinned here rather than trusted.
"""

from __future__ import annotations

import json

import pytest

from analysis.lambda_purchase import ADVANTAGE_EPS, summarise

BUDGET = 768


def row(correct_mean, correct_std, think_std, lam, format_std=0.0, think_rate=0.5, think_mean=100.0):
    """One logged training step.

    `reward_std` is computed from the identity rather than passed in, so a test
    that feeds a cancelling group produces exactly the number TRL would log.
    """
    return {
        "reward": 1.0,
        "reward_std": lam * think_std / BUDGET,
        "rewards/metric_correctness/mean": correct_mean,
        "rewards/metric_correctness/std": correct_std,
        "rewards/metric_format_rate/std": format_std,
        "rewards/metric_mean_think_tokens/mean": think_mean,
        "rewards/metric_mean_think_tokens/std": think_std,
        "rewards/metric_think_rate/mean": think_rate,
    }


def write(tmp_path, rows, name="log_history.json"):
    path = tmp_path / name
    # A real history ends with a summary row carrying no reward; it must be skipped.
    path.write_text(json.dumps(rows + [{"train_runtime": 1.0, "step": len(rows)}]), encoding="utf-8")
    return path


def test_varying_correctness_counts_as_purchase():
    """Lambda can reorder rollouts only when something other than length differs."""
    lam = 0.5
    rows = [row(0.5, 0.53, 100.0, lam), row(1.0, 0.0, 100.0, lam)]
    summary = summarise(write(_tmp(), rows), lam, "t")
    assert summary.graded == 1
    assert summary.all_correct == 1


def test_varying_format_also_counts():
    """format_ok is the other non-length term, so it grants purchase too.

    Easy to overlook: correctness is the obvious one, but a group that is
    uniformly correct while differing in format still lets lambda change the
    ranking rather than only the scale.
    """
    lam = 0.5
    rows = [row(1.0, 0.0, 100.0, lam, format_std=0.35)]
    summary = summarise(write(_tmp(), rows), lam, "t")
    assert summary.graded == 1
    assert summary.all_correct == 0


def test_cancelling_groups_are_split_by_outcome():
    lam = 1.0
    rows = [row(1.0, 0.0, 50.0, lam), row(0.0, 0.0, 50.0, lam), row(1.0, 0.0, 70.0, lam)]
    summary = summarise(write(_tmp(), rows), lam, "t")
    assert (summary.graded, summary.all_correct, summary.all_wrong) == (0, 2, 1)


@pytest.mark.parametrize("lam", [0.05, 0.25, 0.5, 1.0, 2.0])
def test_identity_is_exact_when_constructed_to_hold(lam):
    """A group with constant correctness has reward_std == lambda*std(think)/768.

    Feeding rows built from that identity must report ~zero error at every
    lambda; if it did not, the measured 1e-5 on the real runs would mean nothing.
    """
    rows = [row(1.0, 0.0, think_std, lam) for think_std in (10.0, 116.4, 300.0)]
    summary = summarise(write(_tmp(), rows), lam, "t")
    assert summary.max_cancellation_error == pytest.approx(0.0, abs=1e-12)


def test_attenuation_is_larger_at_small_lambda():
    """The 1e-4 in the denominator is what survives cancellation.

    It is a bigger share of a small reward spread than a large one, so low
    lambda takes slightly smaller steps — the entirety of lambda's effect on a
    cancelling group, and a numerical artefact rather than the reward design.
    """
    think_std = 116.4
    low = summarise(write(_tmp(), [row(1.0, 0.0, think_std, 0.05)]), 0.05, "lo")
    high = summarise(write(_tmp(), [row(1.0, 0.0, think_std, 2.0)]), 2.0, "hi")

    assert low.mean_attenuation < high.mean_attenuation
    expected_low = (0.05 * think_std / BUDGET) / (0.05 * think_std / BUDGET + ADVANTAGE_EPS)
    assert low.mean_attenuation == pytest.approx(expected_low)
    assert high.mean_attenuation > 0.999


def test_summary_row_without_reward_is_ignored():
    """TRL appends a run-summary entry; counting it would shift every fraction."""
    lam = 0.5
    summary = summarise(write(_tmp(), [row(1.0, 0.0, 100.0, lam)]), lam, "t")
    assert summary.steps == 1


def _tmp():
    """Per-call temp dir, so each test writes its own history file."""
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())
