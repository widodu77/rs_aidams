"""Verification of the train/eval split (Phase D precondition).

The split is not incidental plumbing. The trained policy must be evaluated on
items it never saw, and the fixed-policy baselines it is compared against must
be restricted to that same held-out set. Because the baselines already cover all
1240 items, this only works if the split is *deterministic* and *recorded* —
otherwise a rerun silently produces a different eval set and the comparison is
against partially-trained-on data.

Stratification matters for a project-specific reason: reasoning is worth +44.2
points on `irrelevance` and -2.8 on `simple_python`. An unstratified split would
give train and eval different reward landscapes and make them incomparable.

`build_records` needs a tokenizer, so the tests here cover the split logic and
manifest round-trip, which is where the methodological risk lives.
"""

from __future__ import annotations

import json

import pytest

from train.dataset import load_manifest, split_records, write_manifest

CATEGORIES = {
    "simple_python": 400,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
    "irrelevance": 240,
}


@pytest.fixture
def records() -> list[dict]:
    """Stand-ins with the real category sizes; only ids and categories matter."""
    return [
        {"category": cat, "id": f"{cat}_{i}", "prompt": "p", "function": [], "ground_truth": None}
        for cat, n in CATEGORIES.items()
        for i in range(n)
    ]


def test_split_is_deterministic(records):
    """Same seed must give the same split, or the eval set is not reproducible."""
    a_train, a_eval = split_records(records, 0.2, seed=0)
    b_train, b_eval = split_records(records, 0.2, seed=0)

    assert [r["id"] for r in a_eval] == [r["id"] for r in b_eval]
    assert [r["id"] for r in a_train] == [r["id"] for r in b_train]


def test_different_seed_gives_different_split(records):
    _, eval_0 = split_records(records, 0.2, seed=0)
    _, eval_1 = split_records(records, 0.2, seed=1)

    assert {r["id"] for r in eval_0} != {r["id"] for r in eval_1}


def test_split_does_not_depend_on_input_order(records):
    """Reading the dataset files in a different order must not move the split.

    Guards the subtle version of non-determinism: a seeded shuffle over a list
    whose order came from the filesystem is only as stable as that order.
    """
    shuffled = list(reversed(records))
    a_train, a_eval = split_records(records, 0.2, seed=0)
    b_train, b_eval = split_records(shuffled, 0.2, seed=0)

    assert {r["id"] for r in a_eval} == {r["id"] for r in b_eval}
    assert {r["id"] for r in a_train} == {r["id"] for r in b_train}


def test_split_is_a_partition(records):
    """No item may be both trained on and evaluated on, and none may vanish."""
    train, evaluation = split_records(records, 0.2, seed=0)

    train_ids = {r["id"] for r in train}
    eval_ids = {r["id"] for r in evaluation}

    assert train_ids & eval_ids == set()
    assert len(train_ids | eval_ids) == len(records)


def test_split_is_stratified_by_category(records):
    """Each category must contribute ~eval_fraction of its own items."""
    _, evaluation = split_records(records, 0.2, seed=0)

    per_category: dict[str, int] = {}
    for record in evaluation:
        per_category[record["category"]] = per_category.get(record["category"], 0) + 1

    for category, total in CATEGORIES.items():
        assert per_category[category] == round(total * 0.2), category


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.5])
def test_eval_fraction_is_respected(records, fraction):
    _, evaluation = split_records(records, fraction, seed=0)
    assert len(evaluation) == sum(round(n * fraction) for n in CATEGORIES.values())


def test_invalid_fraction_rejected(records):
    with pytest.raises(ValueError):
        split_records(records, 0.0, seed=0)
    with pytest.raises(ValueError):
        split_records(records, 1.0, seed=0)


def test_manifest_round_trips(records, tmp_path):
    """The manifest is what restricts the baselines to the eval set later.

    If it does not round-trip exactly, the trained policy gets compared against
    baselines computed partly on its own training data.
    """
    train, evaluation = split_records(records, 0.2, seed=7)
    path = str(tmp_path / "split_manifest.json")
    write_manifest(path, train, evaluation, seed=7)

    train_ids, eval_ids = load_manifest(path)

    assert train_ids == {(r["category"], r["id"]) for r in train}
    assert eval_ids == {(r["category"], r["id"]) for r in evaluation}
    assert train_ids & eval_ids == set()

    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["seed"] == 7
