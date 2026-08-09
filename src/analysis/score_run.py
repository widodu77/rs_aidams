"""Score generated runs and produce the per-category breakdown.

Reads the JSONL written by `generate.run_baseline` and computes, per policy and
per category, the two axes the project cares about: accuracy (quality) and
tokens spent reasoning (cost). Runs entirely on CPU.

Beyond accuracy and cost it reports three diagnostics that distinguish a real
result from an artefact:

- `think_rate`   — fraction of items with a non-empty reasoning block. For the
                   fixed policies this should be ~1.0 and ~0.0; anything else
                   means the policy is not being enforced and the comparison is
                   invalid.
- `truncated`    — fraction that hit the generation cap. These produce no call
                   and score zero, so a high rate means the cap, not the model,
                   is driving the accuracy number.
- `no_call_rate` — fraction emitting no call at all. Correct on irrelevance,
                   a failure everywhere else.

Usage:
    PYTHONPATH=src uv run python -m analysis.score_run \
        results/raw/qwen3-1.7b_never.jsonl results/raw/qwen3-1.7b_always.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from transformers import AutoTokenizer

from scoring.bfcl_scorer import load_category, parse_model_output, score_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="JSONL files from generate.run_baseline")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-new-tokens", type=int, default=768, help="cap used at generation time")
    p.add_argument("--out", default="results/baseline_metrics.json")
    return p.parse_args()


def load_records(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    return records


def summarize(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = load_records(args.runs)

    # Ground truth is looked up per category, cached so each file is read once.
    samples: dict[str, dict] = {}
    for category in {r["category"] for r in records}:
        for sample in load_category(category):
            samples[(category, sample["id"])] = sample

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for record in records:
        sample = samples[(record["category"], record["id"])]
        parsed = parse_model_output(record["output_text"])
        result = score_sample(
            sample["function"], sample["ground_truth"], parsed, record["category"]
        )

        think_tokens = (
            len(tokenizer.encode(parsed.think, add_special_tokens=False))
            if parsed.think
            else 0
        )

        buckets[(record["policy"], record["category"])].append(
            {
                "correct": result["correct"],
                "error_type": result["error_type"],
                "completion_tokens": record["completion_tokens"],
                "think_tokens": think_tokens,
                "did_think": float(parsed.did_think),
                "truncated": float(record["completion_tokens"] >= args.max_new_tokens),
                "no_call": float(not parsed.has_calls),
            }
        )

    metrics = {}
    for (policy, category), rows in sorted(buckets.items()):
        metrics[f"{policy}/{category}"] = {
            "policy": policy,
            "category": category,
            "n": len(rows),
            "accuracy": summarize([r["correct"] for r in rows]),
            "mean_completion_tokens": summarize([r["completion_tokens"] for r in rows]),
            "mean_think_tokens": summarize([r["think_tokens"] for r in rows]),
            "think_rate": summarize([r["did_think"] for r in rows]),
            "truncated": summarize([r["truncated"] for r in rows]),
            "no_call_rate": summarize([r["no_call"] for r in rows]),
        }

    # Overall row per policy, weighted by item count.
    for policy in sorted({p for p, _ in buckets}):
        rows = [r for (pol, _), rs in buckets.items() if pol == policy for r in rs]
        metrics[f"{policy}/OVERALL"] = {
            "policy": policy,
            "category": "OVERALL",
            "n": len(rows),
            "accuracy": summarize([r["correct"] for r in rows]),
            "mean_completion_tokens": summarize([r["completion_tokens"] for r in rows]),
            "mean_think_tokens": summarize([r["think_tokens"] for r in rows]),
            "think_rate": summarize([r["did_think"] for r in rows]),
            "truncated": summarize([r["truncated"] for r in rows]),
            "no_call_rate": summarize([r["no_call"] for r in rows]),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    header = f"{'policy':8s} {'category':20s} {'n':>4s} {'acc':>7s} {'tok':>7s} {'think':>7s} {'thinkR':>7s} {'trunc':>6s} {'noCall':>6s}"
    print(header)
    print("-" * len(header))
    for row in metrics.values():
        print(
            f"{row['policy']:8s} {row['category']:20s} {row['n']:4d} "
            f"{row['accuracy']:7.1%} {row['mean_completion_tokens']:7.1f} "
            f"{row['mean_think_tokens']:7.1f} {row['think_rate']:7.1%} "
            f"{row['truncated']:6.1%} {row['no_call_rate']:6.1%}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
