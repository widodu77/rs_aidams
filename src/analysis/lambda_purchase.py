"""How much of the training signal can lambda actually reach?

GRPO standardises advantages within a group: `A = (r - mean) / (std + 1e-4)`.
Write the reward as

    r = w_correct * correct + w_format * format_ok - lambda * think / budget

and consider a group in which `correct` and `format_ok` are the same for every
rollout. Then `r = K - lambda * think / budget`, so both the numerator and the
denominator of `A` carry a factor of `lambda` and it divides straight out. The
length penalty has no influence on that step's gradient at all -- at any value
of lambda, including zero.

Lambda therefore only bites on groups where the *non-length* terms vary, because
only there can it change which rollout ranks highest rather than merely rescale
the spread. This module measures what fraction of training steps that is, and
checks the cancellation identity directly on the steps where it should hold:

    reward_std == lambda * std(think_tokens) / think_token_budget

The residual gradient on a cancelling step is not exactly zero, because of the
`1e-4` guarding the division. That constant is a larger share of a small std
than of a large one, so it makes low-lambda runs take slightly *smaller* steps
than high-lambda ones -- the whole of lambda's effect on such groups, and an
artefact of numerical hygiene rather than of the reward. `attenuation` reports
it as `std / (std + 1e-4)`.

Reads TRL's `log_history.json` directly, so it needs no GPU and no re-run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# TRL hard-codes this in GRPOTrainer: `advantages / (std_rewards + 1e-4)`.
ADVANTAGE_EPS = 1e-4


@dataclass(frozen=True)
class RunSummary:
    """Per-run statistics. Counts are over logged optimiser steps."""

    label: str
    lambda_think: float
    steps: int
    graded: int  # steps where correctness or format varied -> lambda can rank
    all_correct: int
    all_wrong: int
    max_cancellation_error: float  # relative, over cancelling steps
    mean_attenuation: float  # mean std/(std+eps) over cancelling steps
    think_rate_first10: float
    think_rate_last10: float
    think_tokens_first10: float
    think_tokens_last10: float

    @property
    def graded_fraction(self) -> float:
        return self.graded / self.steps if self.steps else 0.0


def _rows(path: Path) -> list[dict]:
    """Training rows only. The final entry is a run summary with no reward."""
    history = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in history if "reward" in row]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarise(path: Path, lambda_think: float, label: str, budget: int = 768) -> RunSummary:
    rows = _rows(path)
    if not rows:
        raise ValueError(f"{path} contains no training rows")

    graded = all_correct = all_wrong = 0
    errors: list[float] = []
    attenuations: list[float] = []

    for row in rows:
        correct_std = row["rewards/metric_correctness/std"]
        format_std = row["rewards/metric_format_rate/std"]
        correct_mean = row["rewards/metric_correctness/mean"]
        reward_std = row["reward_std"]
        think_std = row["rewards/metric_mean_think_tokens/std"]

        # Lambda can change the *ordering* of rollouts only when something other
        # than length differs between them.
        if correct_std > 0 or format_std > 0:
            graded += 1
            continue

        if correct_mean == 1.0:
            all_correct += 1
        elif correct_mean == 0.0:
            all_wrong += 1

        # Everything below this line is a group where the identity must hold.
        predicted = lambda_think * think_std / budget
        if predicted > 0:
            errors.append(abs(reward_std - predicted) / predicted)
        if reward_std > 0:
            attenuations.append(reward_std / (reward_std + ADVANTAGE_EPS))

    think_rate = [row["rewards/metric_think_rate/mean"] for row in rows]
    think_tokens = [row["rewards/metric_mean_think_tokens/mean"] for row in rows]

    return RunSummary(
        label=label,
        lambda_think=lambda_think,
        steps=len(rows),
        graded=graded,
        all_correct=all_correct,
        all_wrong=all_wrong,
        max_cancellation_error=max(errors) if errors else float("nan"),
        mean_attenuation=_mean(attenuations),
        think_rate_first10=_mean(think_rate[:10]),
        think_rate_last10=_mean(think_rate[-10:]),
        think_tokens_first10=_mean(think_tokens[:10]),
        think_tokens_last10=_mean(think_tokens[-10:]),
    )


def format_table(summaries: list[RunSummary]) -> str:
    lines = [
        f"{'lambda':>7} {'steps':>6} {'graded':>7} {'graded%':>8} "
        f"{'allCorr':>8} {'allWrong':>9} {'cancelErr':>10} {'attenuat':>9} "
        f"{'think1st':>9} {'thinkLast':>10} {'tok1st':>8} {'tokLast':>8}"
    ]
    for s in summaries:
        lines.append(
            f"{s.lambda_think:>7} {s.steps:>6} {s.graded:>7} "
            f"{100 * s.graded_fraction:>7.1f}% {s.all_correct:>8} {s.all_wrong:>9} "
            f"{s.max_cancellation_error:>10.2e} {s.mean_attenuation:>9.4f} "
            f"{s.think_rate_first10:>9.3f} {s.think_rate_last10:>10.3f} "
            f"{s.think_tokens_first10:>8.1f} {s.think_tokens_last10:>8.1f}"
        )

    total_steps = sum(s.steps for s in summaries)
    total_graded = sum(s.graded for s in summaries)
    lines.append("")
    lines.append(
        f"across all runs: {total_graded}/{total_steps} steps "
        f"({100 * total_graded / total_steps:.1f}%) gave lambda any purchase; "
        f"on the other {100 * (1 - total_graded / total_steps):.1f}% it cancelled."
    )
    worst = max(s.max_cancellation_error for s in summaries)
    lines.append(
        f"cancellation identity reward_std == lambda*std(think)/768 holds to "
        f"{worst:.1e} relative error on every cancelling step."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LAMBDA=PATH",
        help="e.g. --run 0.05=runs/paired_lam0_05/log_history.json (repeatable)",
    )
    parser.add_argument("--think-token-budget", type=int, default=768)
    parser.add_argument("--out", type=Path, help="optional JSON dump of the summaries")
    args = parser.parse_args()

    summaries = []
    for spec in args.run:
        lam, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--run needs LAMBDA=PATH, got {spec!r}")
        summaries.append(
            summarise(Path(path), float(lam), label=path, budget=args.think_token_budget)
        )

    print(format_table(summaries))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([vars(s) | {"graded_fraction": s.graded_fraction} for s in summaries], indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
