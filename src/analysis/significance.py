"""Paired significance tests for the frontier table.

Every policy is evaluated on the *same* 248 held-out items, so the comparisons
are paired and the tests must be too. Two are used:

- **McNemar's exact test** for accuracy. Only the items where two policies
  disagree carry information; the ones they both get right (or both wrong) say
  nothing about which is better. Under the null "the two are equally likely to
  win a disagreement", the count of one-way disagreements is Binomial(n, 0.5),
  and the exact two-sided p follows. No normal approximation, which matters
  because the discordant counts here are small -- often under 20.
- **A paired bootstrap** for token cost. Token counts are heavily skewed (a
  truncated generation is 768, a direct call is 30), so a t-interval on the
  difference is not trustworthy. Resampling items preserves the pairing and
  makes no distributional assumption.

The point of this module is to stop the frontier table from being read as a
ranking. A 2.8-point spread across five lambda looks like a trend and is not
one: at n=248 the standard error on a single accuracy is about 0.6 points, and
paired differences of that size routinely fail to reach significance.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from math import comb

from analysis.pareto import CATEGORIES, DEFAULT_RUNS, score_file
from scoring.bfcl_scorer import load_category
from train.dataset import load_manifest

# Comparisons worth making, as (label_a, label_b). Chosen before seeing the
# results, from the questions the project set out to answer -- not by scanning
# the table for the widest gaps, which would guarantee significance by selection.
DEFAULT_COMPARISONS = [
    # Did training beat the fixed policy it is supposed to improve on?
    ("paired λ=0.5", "always"),
    ("paired λ=1.0", "always"),
    ("norm=batch λ=2.0", "always"),
    # Did training beat simply *prompting* for adaptive behaviour?
    ("paired λ=0.5", "adaptive prompt"),
    ("norm=batch λ=2.0", "adaptive prompt"),
    # Does lambda order the policies within a sweep?
    ("paired λ=0.5", "paired λ=2.0"),
    ("unpaired λ=0.5", "unpaired λ=2.0"),
    # Does the normaliser matter, holding lambda and pairing fixed?
    ("norm=batch λ=2.0", "norm=group λ=2.0"),
    # Did forcing both modes into the group change the deployed policy?
    ("paired λ=2.0", "unpaired λ=2.0"),
]


@dataclass(frozen=True)
class Comparison:
    label_a: str
    label_b: str
    n: int
    acc_a: float
    acc_b: float
    only_a: int  # a correct, b wrong
    only_b: int
    p_value: float
    tokens_a: float
    tokens_b: float
    token_diff: float
    token_ci: tuple[float, float]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value.

    Concordant pairs are excluded by construction: under the null each of the
    `only_a + only_b` disagreements is equally likely to fall either way, so the
    statistic is Binomial(n, 1/2) and the two-sided p is the total probability of
    outcomes at least as lopsided as the one observed.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_token_diff(
    paired: list[tuple[float, float]], iterations: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for the mean paired difference in completion tokens.

    Items are resampled with replacement *as pairs*, which is what preserves the
    fact that both policies answered the same question.
    """
    rng = random.Random(seed)
    n = len(paired)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            a, b = paired[rng.randrange(n)]
            total += a - b
        means.append(total / n)
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def compare(label_a: str, label_b: str, runs: dict) -> Comparison:
    a, b = runs[label_a], runs[label_b]
    keys = sorted(set(a) & set(b))

    only_a = sum(1 for k in keys if a[k]["correct"] and not b[k]["correct"])
    only_b = sum(1 for k in keys if b[k]["correct"] and not a[k]["correct"])
    tokens = [(a[k]["tokens"], b[k]["tokens"]) for k in keys]
    diff = sum(x - y for x, y in tokens) / len(tokens)

    return Comparison(
        label_a=label_a,
        label_b=label_b,
        n=len(keys),
        acc_a=sum(a[k]["correct"] for k in keys) / len(keys),
        acc_b=sum(b[k]["correct"] for k in keys) / len(keys),
        only_a=only_a,
        only_b=only_b,
        p_value=mcnemar_exact(only_a, only_b),
        tokens_a=sum(x for x, _ in tokens) / len(tokens),
        tokens_b=sum(y for _, y in tokens) / len(tokens),
        token_diff=diff,
        token_ci=bootstrap_token_diff(tokens),
    )


def format_table(comparisons: list[Comparison]) -> str:
    lines = [
        f"{'A':18s} {'B':18s} {'accA':>6s} {'accB':>6s} "
        f"{'A>B':>4s} {'B>A':>4s} {'p':>8s}  {'tokens A-B (95% CI)':>26s}"
    ]
    lines.append("-" * len(lines[0]))
    for c in comparisons:
        star = " *" if c.significant else "  "
        lines.append(
            f"{c.label_a:18s} {c.label_b:18s} {c.acc_a:6.1%} {c.acc_b:6.1%} "
            f"{c.only_a:>4} {c.only_b:>4} {c.p_value:8.3f}{star} "
            f"{c.token_diff:+8.1f}  [{c.token_ci[0]:+7.1f},{c.token_ci[1]:+7.1f}]"
        )
    lines.append("")
    lines.append("* p < 0.05, McNemar exact (accuracy). Token CI is a paired bootstrap, 10k resamples.")
    lines.append(
        "A>B counts items A got right and B did not; only those carry information about accuracy."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", default="runs/paired_lam0_05/split_manifest.json")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out", default="results/significance.json")
    args = parser.parse_args()

    import os

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def count_think(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    train_ids, eval_ids = load_manifest(args.split_manifest)
    keep = eval_ids if args.split == "eval" else train_ids
    ground_truth = {(c, s["id"]): s for c in CATEGORIES for s in load_category(c)}

    runs = {}
    for label, path in DEFAULT_RUNS:
        if os.path.exists(path):
            runs[label] = score_file(path, keep, ground_truth, count_think)

    comparisons = []
    for label_a, label_b in DEFAULT_COMPARISONS:
        if label_a in runs and label_b in runs:
            comparisons.append(compare(label_a, label_b, runs))
        else:
            missing = [l for l in (label_a, label_b) if l not in runs]
            print(f"skipped {label_a} vs {label_b}: missing {missing}")

    print()
    print(format_table(comparisons))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump([vars(c) for c in comparisons], fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
