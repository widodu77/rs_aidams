"""Figures for the 4-page paper, built from the scored result files.

Nothing here recomputes a metric. `results/pareto.json` is written by
`analysis.pareto` and `runs/*/log_history.json` by the trainer; this script only
arranges numbers that already exist, so the figures cannot drift from the tables.

    PYTHONPATH=src uv run python paper/make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")

# Serif to match the document body; sizes chosen for a two-column layout, where
# the figure is reduced to ~3.3in and anything under 8pt stops being legible.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)


def load(path: str):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return json.load(fh)


def frontier() -> None:
    """Cost against quality. The whole argument of the paper fits in this axes."""
    summary = load("results/pareto.json")["summary"]

    def point(label):
        row = summary[label]
        return row["tokens"], row["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    # The twelve trained runs whose decision sat in the prompt. Plotted as one
    # cloud rather than twelve labelled points: the claim is that they are
    # indistinguishable, so distinguishing them in the figure would mislead.
    prompt_runs = [
        f"{kind} λ={lam}"
        for kind in ("paired", "unpaired")
        for lam in ("0.05", "0.25", "0.5", "1.0", "2.0")
    ] + ["norm=group λ=2.0", "norm=batch λ=2.0", "decision=prompt λ=2.0"]
    xs, ys = zip(*[point(l) for l in prompt_runs if l in summary])
    ax.scatter(xs, ys, s=14, c="#9aa4b2", marker="o", zorder=2,
               label=f"decision in prompt ({len(xs)} runs)")

    # The gate sweep: five runs, five values of lambda, one location.
    gate = [f"gate λ={lam}" for lam in ("0.05", "0.25", "0.5", "1.0", "2.0")]
    gx, gy = zip(*[point(l) for l in gate if l in summary])
    ax.scatter(gx, gy, s=26, c="#1f4e79", marker="D", zorder=4,
               label="decision in completion (5 runs)")

    anchors = [
        ("never", "never reason", "#c0392b", "v", (-14, 8)),
        ("always", "always reason", "#c0392b", "^", (-16, 9)),
        ("adaptive prompt", "adaptive prompt", "#c0392b", "s", (-30, -13)),
        ("oracle", "per-item oracle", "#1e8449", "*", (7, 2)),
    ]
    for key, name, colour, marker, offset in anchors:
        if key not in summary:
            continue
        x, y = point(key)
        ax.scatter([x], [y], s=60 if marker == "*" else 34, c=colour,
                   marker=marker, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offset,
                    fontsize=7, color=colour)

    # The arrow is the paper's second result: same objective, same lambda, the
    # decision moved from one place to the other.
    if "decision=prompt λ=2.0" in summary and "gate λ=2.0" in summary:
        x0, y0 = point("decision=prompt λ=2.0")
        x1, y1 = point("gate λ=2.0")
        ax.annotate("", xy=(x1 + 8, y1), xytext=(x0 - 8, y0),
                    arrowprops=dict(arrowstyle="->", color="#1f4e79",
                                    lw=0.9, ls=(0, (4, 2))), zorder=3)
        ax.text((x0 + x1) / 2, y0 - 3.4, "move the decision\ninto the completion",
                ha="center", fontsize=6.5, color="#1f4e79")

    ax.set_xlabel("mean completion tokens per item")
    ax.set_ylabel("accuracy (\\%)")
    ax.set_xlim(20, 310)
    ax.set_ylim(69, 95)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "frontier.pdf"))
    plt.close(fig)


def collapse() -> None:
    """Think rate against optimiser step: the null, the movement, and the cliff."""
    fig, ax = plt.subplots(figsize=(3.4, 2.0))

    series = [
        ("paired_lam2_0", "decision in prompt, $\\lambda$=2.0", "#9aa4b2", "-"),
        ("paired_isoff_lam2_0", "same, IS correction off", "#6b7684", (0, (4, 2))),
        ("gate_lam2_0", "decision in completion, $\\lambda$=2.0", "#1f4e79", "-"),
    ]
    for tag, label, colour, style in series:
        path = os.path.join(ROOT, "runs", tag, "log_history.json")
        if not os.path.exists(path):
            print(f"missing, skipped: {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            history = json.load(fh)
        key = "rewards/metric_think_rate/mean"
        pairs = [(r["step"], r[key]) for r in history if key in r]
        steps, rates = zip(*pairs)
        ax.plot(steps, rates, color=colour, ls=style, lw=1.2, label=label)

    # Both reference lines are explained in the caption, so only the one a reader
    # would not otherwise guess gets an inline label.
    ax.axhline(0.5, color="#c0392b", lw=0.8, ls=":", zorder=1)
    ax.axhline(0.174, color="#1e8449", lw=0.9, ls=":", zorder=1)
    ax.text(1.5, 0.19, "oracle", fontsize=6.5, color="#1e8449", va="bottom")

    ax.set_xlabel("optimiser step")
    ax.set_ylabel("think rate (training rollouts)")
    ax.set_xlim(0, 100)
    # The prompt-decision runs sit at the 0.5 forced floor by construction, so
    # the interesting range is the bottom half; capping at 0.7 spends the axes on
    # the part that moves and leaves the legend somewhere it covers no data.
    ax.set_ylim(-0.03, 0.72)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4,
              borderaxespad=0.1, labelspacing=0.25)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "collapse.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    frontier()
    collapse()
    print(f"wrote {OUT}/frontier.pdf and {OUT}/collapse.pdf")
