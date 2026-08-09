# Adaptive Thinking for Tool Use

**Teaching a small language model to decide, for each tool call it makes, whether it's worth thinking first.**

---

## Why this is a question at all

AI agents work by calling functions — `search(query)`, `get_weather(city)`, `book_flight(...)`. Getting those calls right is what makes an agent useful, and it is usually the thing that breaks first in production.

Large models handle this well but cost too much to run at scale. Small models are cheap and less reliable, which is why a lot of recent work uses reinforcement learning to close the gap.

Separately, we know that letting a model reason before it answers often improves the answer. But reasoning isn't free. It costs tokens, which cost money and latency. And it doesn't always help — on an obvious call it's pure waste, and on an ambiguous one it can be the difference between right and wrong.

**Existing methods pick one setting and apply it everywhere.** The model either always reasons or never does. Nobody has trained a small model to make that decision *per call*.

> **Can a ≤2B-parameter model learn, through reinforcement learning, to reason only when the tool call actually warrants it — and where does that land on the cost-quality tradeoff?**

## What we're building

One model that can produce either shape:

```
<think>reasoning…</think><tool_call>{"name": …, "arguments": {…}}</tool_call>   ← thought first
<tool_call>{"name": …, "arguments": {…}}</tool_call>                            ← went straight to it
```

It is trained with GRPO against a reward that is entirely programmatic — no human labels, no learned reward model:

```
reward  =  correctness  +  format_validity  −  λ · think_tokens
```

Correctness comes from the Berkeley Function Calling Leaderboard's own checker: does the call parse, is the function name right, do the arguments validate.

**We never tell the model when to think.** During training it samples both a reasoned and a direct continuation for the same input, and the reward decides which one paid off. The decision rule emerges on its own. The knob `λ` sets the price of thinking — raise it and reasoning has to earn its keep, lower it and reasoning is nearly free.

## What we're trying to show

**1. The tradeoff curve.** Train at several values of `λ`, then plot accuracy against tokens spent, with always-think and never-think as reference points. The question is whether the adaptive model sits *above the line between them* — buying accuracy at a cost neither fixed policy can reach.

**2. That it thinks for sensible reasons.** A model could score well while thinking essentially at random, so this is checked separately:

> **H1** — the model thinks more often as calls get genuinely harder.

"Harder" is measured against fixed properties of the benchmark item: how many candidate functions it had to choose from, how many arguments the call needs, whether several calls are required, whether the right answer was to call nothing at all.

Deliberately *not* measured as "items the model gets wrong" — that would be circular, since a model thinks when it's uncertain and uncertainty tracks its own errors. The correlation would look impressive and mean nothing.

## Early evidence

From the fixed-policy baselines on Qwen3-1.7B (preliminary — `always` on `multiple` is n=41, the rest n=100):

| Category | never-think | always-think |
|---|---|---|
| `simple` | 88.0% | **95.0%** |
| `multiple` | **93.0%** | 82.9% |

Thinking **helps** on simple calls and **hurts** when the model has to pick between several candidate functions. Neither fixed policy wins everywhere — which is the entire premise of the project, visible in real data before any training has happened.

A second observation: Qwen3 already refuses to reason on easy items even when instructed to, opening a `<think>` block and immediately closing it. The base model has a crude gate of its own, which suggests a better, learned one is reachable.

## Three ways this can end

| Outcome | What it means |
|---|---|
| Adaptive beats both fixed policies | Main result — learned per-call gating works |
| The policy collapses to always or never | The reward can't sustain a gate. A real finding, and one the assessment explicitly rewards |
| It gates, but not by difficulty | It found some other signal. Characterizing that is a finding |

There is no outcome where the project has nothing to report.

## Status

| Phase | | |
|---|---|---|
| **A** | Per-sample BFCL scoring | done |
| **B** | Output contract + fixed-prompt baselines | in progress |
| **C** | Reward function + test suite | pending |
| **D** | GRPO training | pending |
| **E** | Tradeoff curve + H1 analysis | pending |
| **F** | Paper, demo, documentation | pending |

A and B are measurement. C and D are the contribution. E is where we find out whether it worked.

Phase A blocked everything downstream: the evaluator *is* the reward function, so no training was possible until a single model output could be scored in isolation. It now can — `src/scoring/bfcl_scorer.py` wraps BFCL's official AST checker as a per-sample scorer, verified against all 1000 single-turn items.

## Setup

```bash
uv sync
```

Run the scorer test suite:

```bash
uv run pytest
```

Generate baselines (GPU) and score them (CPU) as separate steps, so the expensive half runs once and the analysis can be re-derived freely:

```bash
uv run python -m generate.run_baseline --load-4bit --policy never --limit 100
uv run python -m analysis.score_run results/raw/qwen3-1.7b_never.jsonl
```

Dependencies are pinned in `uv.lock` (committed deliberately — it is what makes the experiments reproducible). BFCL is pinned to `bfcl-eval==2026.3.23`, which ships the v4 data inside the package.

## Layout

```
├── src/
│   ├── scoring/       # per-sample BFCL scorer            (Phase A)
│   ├── prompts/       # output contract, policy templates (Phase B)
│   ├── generate/      # GPU: model outputs -> JSONL       (Phase B)
│   ├── analysis/      # CPU: JSONL -> metrics             (Phase B/E)
│   ├── rewards/       # GRPO reward + unit tests          (Phase C)
│   └── train/         # TRL GRPO training                 (Phase D)
├── tests/             # scorer verification suite
├── results/           # raw generations, metrics, plots
├── notes/             # daily research notes
└── docs/              # proposal, design notes
```

## Stack

Qwen3-1.7B (native `<think>` / `<tool_call>` tokens, hybrid thinking mode) · TRL `GRPOTrainer` + LoRA · [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) v4 · 4-bit NF4 locally, Colab for training

## Key references

| | |
|---|---|
| [arXiv:2604.20316](https://arxiv.org/abs/2604.20316) | R2IF — composite rewards for interpretable function calling |
| [arXiv:2508.05118](https://arxiv.org/abs/2508.05118) | EGPO — entropy-enhanced exploration for robust function calling |
| [arXiv:2602.03025](https://arxiv.org/abs/2602.03025) | RC-GRPO — reward-conditioned GRPO for multi-turn tool calling |
| [arXiv:2505.11896](https://arxiv.org/abs/2505.11896) | AdaCoT — Pareto-optimal adaptive CoT triggering via RL |
| [arXiv:2505.16854](https://arxiv.org/abs/2505.16854) | TON — selective reasoning via RL for vision-language models |
| [arXiv:2511.17006](https://arxiv.org/abs/2511.17006) | BATS — budget-aware tool use for agent scaling |
| [arXiv:2507.02076](https://arxiv.org/abs/2507.02076) | Survey — adaptive and controllable test-time compute |

## Use of LLMs

This project uses LLM assistance (Claude, Anthropic) under the conditions set by the AIDAMS program: every concept and line of code must be understood and explainable by the author, and assistance must be labeled in every deliverable.

Assistance to date covers literature mapping and citation verification, proposal structuring, experimental design discussion, and code drafting. Per-file and per-deliverable labeling is maintained in [`notes/`](notes/).

---

*Research Study — bachelors in AI for Data and Management Sciences (AIDAMS), ESSEC Business School / CentraleSupélec, Paris-Saclay. Walid Ajbar, 2026.*
