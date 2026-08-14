"""The cost-quality frontier: trained policies against the fixed-policy anchors.

This is the Phase E table. It answers the question the project was set up to ask —
where do the lambda-trained policies land relative to always-think, never-think,
the adaptive *prompt*, and the per-item oracle.

Runs are labelled by **file**, not by the `policy` field inside the records:
`generate.run_vllm` stamps `policy: "adaptive"` on every trained policy, since they
all share the adaptive prompt. Grouping by that field silently merges five adapters
and the untrained baseline into one row.

Everything is restricted to the held-out eval split, including the baselines — they
were generated over all 1240 items, so without that restriction a trained policy
would be compared against different data than it was evaluated on.

Usage:
    PYTHONPATH=src uv run python -m analysis.pareto
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from scoring.bfcl_scorer import load_category, parse_model_output, score_sample
from train.dataset import load_manifest

CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance")

# label -> path. Order is the reading order of the table: the fixed-policy
# anchors, then the paired-rollout sweep (the current result), then the original
# free-sampling sweep. Both sweeps are kept because the comparison between them
# is itself a finding — they were trained on identical data with identical seeds
# and differ only in whether each GRPO group was forced to contain both thinking
# modes. Missing files are skipped with a note, so a partial set still prints.
DEFAULT_RUNS = [
    ("never", "results/raw/vllm/qwen3-1.7b_never.jsonl"),
    ("always", "results/raw/vllm/qwen3-1.7b_always.jsonl"),
    ("adaptive prompt", "results/raw/vllm/qwen3-1.7b_adaptive.jsonl"),
    ("paired λ=0.05", "results/raw/vllm_eval/trained_paired_lam0_05.jsonl"),
    ("paired λ=0.25", "results/raw/vllm_eval/trained_paired_lam0_25.jsonl"),
    ("paired λ=0.5", "results/raw/vllm_eval/trained_paired_lam0_5.jsonl"),
    ("paired λ=1.0", "results/raw/vllm_eval/trained_paired_lam1_0.jsonl"),
    ("paired λ=2.0", "results/raw/vllm_eval/trained_paired_lam2_0.jsonl"),
    ("unpaired λ=0.05", "results/raw/vllm_eval/trained_lam0_05.jsonl"),
    ("unpaired λ=0.25", "results/raw/vllm_eval/trained_lam0_25.jsonl"),
    ("unpaired λ=0.5", "results/raw/vllm_eval/trained_lam0_5.jsonl"),
    ("unpaired λ=1.0", "results/raw/vllm_eval/trained_lam1_0.jsonl"),
    ("unpaired λ=2.0", "results/raw/vllm_eval/trained_lam2_0.jsonl"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-manifest", default="results/split_manifest.json")
    p.add_argument("--split", choices=["train", "eval"], default="eval")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--out", default="results/pareto.json")
    p.add_argument(
        "--run",
        action="append",
        metavar="LABEL=PATH",
        help="replace the default run list (repeatable), e.g. --run 'paired λ=0.5=path.jsonl'",
    )
    return p.parse_args()


def score_file(path: str, keep: set, ground_truth: dict, count_think) -> dict:
    """Score one run, restricted to `keep`. Returns per-item outcomes."""
    items = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["category"], record["id"])
            if key not in keep:
                continue
            sample = ground_truth[key]
            parsed = parse_model_output(record["output_text"])
            result = score_sample(
                sample["function"], sample["ground_truth"], parsed, record["category"]
            )
            items[key] = {
                "correct": result["correct"],
                "tokens": record["completion_tokens"],
                "think_tokens": count_think(parsed.think) if parsed.think else 0,
                "did_think": float(parsed.did_think),
            }
    return items


def mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def count_think(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    train_ids, eval_ids = load_manifest(args.split_manifest)
    keep = eval_ids if args.split == "eval" else train_ids

    ground_truth = {
        (c, s["id"]): s for c in CATEGORIES for s in load_category(c)
    }

    # `--run` uses the last '=' as the separator, because the labels themselves
    # contain one (`paired λ=0.5`).
    selected = DEFAULT_RUNS
    if args.run:
        selected = [(spec.rsplit("=", 1)[0], spec.rsplit("=", 1)[1]) for spec in args.run]

    runs = {}
    for label, path in selected:
        if not os.path.exists(path):
            print(f"missing, skipped: {path}")
            continue
        runs[label] = score_file(path, keep, ground_truth, count_think)

    # ---------------------------------------------------------------- frontier
    print(f"\nheld-out eval split, n={len(keep)}\n")
    header = f"{'policy':18s} {'acc':>7s} {'tokens':>8s} {'think tok':>10s} {'think rate':>11s}"
    print(header)
    print("-" * len(header))

    summary = {}
    for label, items in runs.items():
        row = {
            "n": len(items),
            "accuracy": mean([v["correct"] for v in items.values()]),
            "tokens": mean([v["tokens"] for v in items.values()]),
            "think_tokens": mean([v["think_tokens"] for v in items.values()]),
            "think_rate": mean([v["did_think"] for v in items.values()]),
        }
        summary[label] = row
        print(
            f"{label:18s} {row['accuracy']:7.1%} {row['tokens']:8.1f} "
            f"{row['think_tokens']:10.1f} {row['think_rate']:11.1%}"
        )

    # ---------------------------------------------------------------- oracle
    if "never" in runs and "always" in runs:
        common = set(runs["never"]) & set(runs["always"])
        correct = tokens = 0.0
        for key in common:
            n, a = runs["never"][key], runs["always"][key]
            if n["correct"]:
                correct += 1
                tokens += n["tokens"]
            elif a["correct"]:
                correct += 1
                tokens += a["tokens"]
            else:
                tokens += n["tokens"]
        summary["oracle"] = {
            "n": len(common),
            "accuracy": correct / len(common),
            "tokens": tokens / len(common),
            "think_tokens": float("nan"),
            "think_rate": float("nan"),
        }
        print("-" * len(header))
        print(
            f"{'ORACLE (bound)':18s} {correct/len(common):7.1%} {tokens/len(common):8.1f} "
            f"{'-':>10s} {'-':>11s}"
        )

    # ------------------------------------------------- per-category think rate
    # The prediction from the fp16 break-even table: as lambda rises, thinking
    # should switch off in order — simple_python first, then multiple, then the
    # parallel categories, with irrelevance last.
    print("\n\nthink rate by category (the switch-off prediction)\n")
    # Every run that used the adaptive prompt, i.e. everything whose think rate
    # the model chose rather than had imposed. `never` and `always` are fixed at
    # 0% and ~100% by construction and would only pad the table.
    labels = [l for l in runs if l not in ("never", "always")]
    head = f"{'policy':18s}" + "".join(f"{c[:12]:>14s}" for c in CATEGORIES)
    print(head)
    print("-" * len(head))
    per_category = {}
    for label in labels:
        by_cat = defaultdict(list)
        for (cat, _), v in runs[label].items():
            by_cat[cat].append(v["did_think"])
        per_category[label] = {c: mean(by_cat[c]) for c in CATEGORIES}
        print(f"{label:18s}" + "".join(f"{per_category[label][c]:13.1%} " for c in CATEGORIES))

    # ------------------------------------------------- per-category accuracy
    print("\n\naccuracy by category\n")
    print(head)
    print("-" * len(head))
    for label in runs:
        by_cat = defaultdict(list)
        for (cat, _), v in runs[label].items():
            by_cat[cat].append(v["correct"])
        print(f"{label:18s}" + "".join(f"{mean(by_cat[c]):13.1%} " for c in CATEGORIES))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "think_rate_by_category": per_category}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
