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

## What we set out to show

**1. The tradeoff curve.** Train at several values of `λ`, then plot accuracy against tokens spent, with always-think and never-think as reference points. The question is whether the adaptive model sits *above the line between them* — buying accuracy at a cost neither fixed policy can reach.

**2. That it thinks for sensible reasons.** A model could score well while thinking essentially at random, so this is checked separately:

> **H1** — the model thinks more often as calls get genuinely harder.

"Harder" is measured against fixed properties of the benchmark item: how many candidate functions it had to choose from, how many arguments the call needs, whether several calls are required, whether the right answer was to call nothing at all.

Deliberately *not* measured as "items the model gets wrong" — that would be circular, since a model thinks when it's uncertain and uncertainty tracks its own errors. The correlation would look impressive and mean nothing.

## Early evidence

Fixed-policy baselines on Qwen3-1.7B, fp16, greedy — **all 1240 single-turn BFCL items, every policy**:

| Category | never-think | always-think | adaptive prompt | | never tok | always tok |
|---|---|---|---|---|---|---|
| `simple` | **92.5%** | 89.8% | 91.2% | | 38 | 203 |
| `multiple` | 93.5% | **95.0%** | 91.0% | | 37 | 213 |
| `parallel` | 69.0% | **84.5%** | 83.5% | | 85 | 368 |
| `parallel_multiple` | 62.5% | **81.5%** | 83.5% | | 96 | 418 |
| `irrelevance` | 39.6% | 83.8% | **85.0%** | | 25 | 209 |
| **overall** | 73.8% | 87.3% | **87.5%** | | 52 | 267 |

Reasoning buys +13.5 points for 5.1× the tokens.

### You cannot get a gate by asking for one

The `adaptive` column is the control that matters. Given a prompt where reasoning is explicitly *optional*, with both output shapes shown and neither privileged, the model reasons on **96.9%** of items and spends **more** tokens than always-think (287 vs 267) for statistically indistinguishable accuracy.

The strongest rival hypothesis to "RL learned a gate" was always *"you didn't need RL — just ask."* That's now ruled out, before any training.

### What a perfect gate would buy

Both policies ran on the same items, so every item is paired and the ceiling is computable:

| | accuracy | mean tokens |
|---|---|---|
| never | 73.8% | 52 |
| always | 87.3% | 267 |
| **per-item oracle** | **91.2%** | **97** |

- thinking **rescues** a failure on 216 items (17.4%)
- thinking **destroys** a success on 49 items (4.0%)
- neither policy is right on 109 items (8.8%) — so 91.2% is the hard ceiling

The oracle peeks at ground truth, so it's a bound rather than a target. But it says where the prize is: **+3.9 accuracy points at 2.8× fewer tokens.** The cost axis, not the accuracy axis. And it thinks on **17.4%** of items — the number to watch during training, where near-0% or near-100% means collapse.

### A methodological caution worth its own line

Earlier baselines here were run with 4-bit NF4 quantization, forced by a 4 GB laptop GPU. Re-running at fp16 on the same items overturned two findings outright: an apparent inversion on `multiple` reversed, and an apparent self-gating behaviour (the model opening `<think>` and immediately closing it on 77/500 items) turned out to occur on **1** item at full precision.

The cause was that quantization was crippling the *never* policy specifically — `parallel` went 18% → 69% — so reasoning was masking quantization damage and every "thinking helps" gap was inflated.

**On a 1.7B model, NF4 didn't merely cost accuracy; it flipped the sign of the reasoning effect on two categories.** The full before/after is preserved in [`notes/2026-08-09.md`](notes/2026-08-09.md), superseded sections intact.

## How it actually ended

**No gate formed.** Twelve trained policies, five values of `λ` spanning 40×, two sampling schemes,
two advantage normalisers. Every one of them reasons on 96% to 99% of items. On the 248 held-out
items:

| | accuracy | tokens | think rate |
|---|---|---|---|
| never | 71.4% | 51 | 0% |
| always | 89.5% | 264 | 98.4% |
| adaptive prompt | 89.1% | 281 | 96.8% |
| trained (12 runs) | 87.9–90.7% | 224–281 | 96.4–99.6% |
| **per-item oracle** | **92.7%** | **99** | 17.4% |

Paired McNemar tests put every accuracy comparison at `p ≥ 0.167`, so the spread across trained runs
carries no information. The tradeoff curve is a point cloud, not a frontier.

### The reason, which is the real result

GRPO standardises advantages inside a group: `A = (r − mean) / (std + 1e-4)`. Whenever correctness
and format are constant within that group, the reward collapses to `K − λ·think/768`, so `λ` scales
the numerator and the denominator alike and divides straight back out. It has no influence on the
gradient at all, at any value.

Measured over 500 logged steps: **`λ` cancels on 68.8% of them**, and the identity
`reward_std = λ·std(think)/768` holds to `9.7e-06` across 344 groups and five `λ`. What survives is
the `1e-4` guarding the division, a numerical stabiliser rather than anything to do with the reward.

### Two repairs, both falsified

Forcing every group to contain both a reasoned and a direct rollout made the deployed policy
**longer** by 32.9 tokens per item (95% CI `+23.1` to `+43.0`), with no accuracy gain. The cause is
that `enable_thinking=False` prefills `<think></think>` into the *prompt*, so a direct rollout's
completion never contains the decision not to think. Reinforcing it raises its probability under a
prompt that is never deployed, while the thinking rollouts are reinforced under the one that is.

Normalising advantages across the batch instead of the group restores `λ`'s meaning on 61% of steps,
but at a magnitude too small to shift the policy, and it still cancels on the other 39%.

Full result in [`notes/2026-08-15.md`](notes/2026-08-15.md), derivation and verification in
[`notes/2026-08-13.md`](notes/2026-08-13.md).

### Then moving the decision into the completion changed everything

The diagnosis named a fix, so we ran it. `--gate-rollouts` keeps **one** prompt for the whole group
and forces the direct half by prepending Qwen3's empty-think marker to its **completion** instead of
its prompt. The decision is then something the policy is trained to produce, and at evaluation the
model can emit it itself.

Same λ, same pairing, same everything else. Only the location of the decision differs:

| policy | acc | tokens | think rate |
|---|---|---|---|
| never (prompted) | 71.4% | 51.4 | 0% |
| always | 89.5% | 264.1 | 98.4% |
| decision in **prompt** λ=2.0 | 89.5% | 266.8 | 98.4% |
| decision in **completion** λ=2.0 | **89.9%** | **59.7** | **0%** |
| oracle (bound) | 92.7% | 99.5 | — |

Paired tests on the same 248 items:

- versus **always-think**: `p = 1.000` on accuracy (13 wins, 12 losses) at **−204.5 tokens/item**
  [−220.1, −190.0]. A 77% cost reduction with no detectable accuracy loss.
- versus **never-think**: **46 wins, 0 losses, `p < 0.001`**, for 8.3 extra tokens. The only
  significant accuracy result in the project, with no regressions at all.

Every completion is a genuine call: correct multi-call output on `parallel`, correct prose
abstention on `irrelevance`. Per-category over `never`: parallel 72.5 → 92.5, parallel_multiple
50.0 → 72.5, irrelevance 37.5 → 93.8.

**Three caveats, all load-bearing.** This is *not* a gate: think rate is 0%, not the oracle's 17.4%.
The model did not learn *when* to reason, it learned not to need it. The comparison to `never` is
confounded, since `never` is prompted and this is trained; the honest control is the other twelve
trained runs, which used the same reward and budget and saved **no tokens at all** (224–281 versus
59.7). And it collapsed: once every rollout stopped reasoning the groups went degenerate and the
policy froze around step 23, so λ=2.0 finds an edge rather than a curve.

### What the study establishes

The prize is real and large (the oracle is both more accurate and 2.7× cheaper than always-think).
Prompting cannot collect it, across three separate wordings. Length-penalised GRPO cannot collect it
either **while the decision sits in the prompt**, for a structural reason rather than a tuning one.
Move the decision into the completion and the same objective, at the same λ, produces always-think
accuracy at never-think cost.

So the headline is not a null. It is that **where the decision lives decides whether the objective
can act on it at all** — and one line of rollout code separates "nothing happens across twelve runs"
from "77% of the tokens disappear at no measurable cost".

Limits: 100 optimiser steps, Qwen3-1.7B, LoRA r=32, one task family, one seed per configuration, and
the winning run collapsed to 0% think rate rather than finding a gate.

### The λ sweep ran, and the objective has a cliff

λ finally controlled something, so we swept it: five gate runs, λ from 0.05 to 2.0, a fortyfold
range. All five landed on the same policy.

| λ | acc | tokens | think rate |
|---|---|---|---|
| 0.05 | 89.5% | 61.0 | 0% |
| 0.25 | 89.5% | 60.7 | 0% |
| 0.5 | 89.9% | 60.9 | 0% |
| 1.0 | 89.5% | 60.7 | 0% |
| 2.0 | 89.9% | 59.7 | 0% |

The endpoints disagree on **one held-out item out of 248** (`p = 1.000`, +1.4 tokens [+0.3, +2.8]).
The adapters are genuinely different, 91–93% byte-identical outputs rather than 100%, so λ moves the
weights. It just moves nothing you can observe. Every run starts near 0.45 think rate and hits
exactly 0.00 by step 27 at the latest, after which the groups are degenerate and training is a no-op.

The reason is a second λ-cancellation, one level up from the first. On a group where correctness and
format agree, the only reward variance is length, so the standardised advantage is ±1 whatever λ is.
λ picks the sign, and the sign never changes: at 1.7B on single-turn BFCL, thinking almost never
flips a correctness term, so 0% think rate is the argmin of the objective for every nonzero λ. There
is no interior optimum, which means no amount of tuning finds one.

So the gate is dead, and it died of the reward rather than the budget. Getting one needs a different
objective, not a different λ: asymmetric error pricing, a per-category budget, or an entropy floor to
stop the collapse. None of those were tested.

What survives is the finding that made the sweep worth running. On this benchmark at this scale,
chain-of-thought before a tool call is **purchasable**: it costs about 200 tokens per item and buys
nothing measurable. That is a real result about tool use, and it is not the one we set out to find.

## Status

| Phase | | |
|---|---|---|
| **A** | Per-sample BFCL scoring | done |
| **B** | Output contract + fixed-prompt baselines | done |
| **C** | Reward function + test suite | done |
| **D** | GRPO training | done, 12 runs |
| **E** | Frontier + significance testing | done |
| **F** | Gate rollouts + λ sweep | done, 6 runs |
| **G** | Paper, demo, documentation | pending |

H1 (does it think more on harder calls?) was never testable: think rate sits at 90–100% in every
category for every policy, so there is no variation to correlate difficulty against.

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

Generation (GPU) and scoring (CPU) are separate steps, so the expensive half runs once and the analysis can be re-derived freely.

**Generation runs on Colab with vLLM** — open [`colab/generate_baselines.ipynb`](colab/generate_baselines.ipynb). All three policies over all 1240 items take ~36 minutes on a free T4. The local HF path (`generate.run_baseline`) still works but is ~25× slower: it batches statically, so every sequence steps until the longest in its batch finishes, wasting 2.36× the decode steps. It also needs 4-bit quantization to fit a 4 GB card, which [distorts the results](notes/2026-08-09.md).

Scoring is CPU-only and runs locally against the full `bfcl-eval` install:

```bash
uv run python -m analysis.score_run results/raw/vllm/qwen3-1.7b_never.jsonl results/raw/vllm/qwen3-1.7b_always.jsonl results/raw/vllm/qwen3-1.7b_adaptive.jsonl --out results/baseline_metrics_vllm.json
```

`results/raw/` holds the superseded NF4 generations; `results/raw/vllm/` holds the fp16 ones. Outputs from the two engines are never mixed in one table.

Dependencies are pinned in `uv.lock` (committed deliberately — it is what makes the experiments reproducible). BFCL is pinned to `bfcl-eval==2026.3.23`, which ships the v4 data inside the package.

## Layout

```
├── src/
│   ├── scoring/       # per-sample BFCL scorer            (Phase A)
│   ├── prompts/       # output contract, policy templates (Phase B)
│   ├── generate/      # GPU: model outputs -> JSONL       (Phase B)
│   ├── analysis/      # CPU: JSONL -> metrics             (Phase B/E)
│   ├── rewards/       # GRPO reward + metric fns          (Phase C)
│   └── train/         # dataset/split + TRL GRPO          (Phase D)
├── colab/             # GPU notebooks: generation, training
├── tests/             # scorer, reward, rollout, stats suites (105 tests)
├── results/
│   ├── raw/           # NF4 generations (superseded)
│   └── raw/vllm/      # fp16 generations (current)
├── notes/             # daily research notes + engineering_log.md (29 entries)
└── docs/              # proposal, design notes
```

## Stack

Qwen3-1.7B (native `<think>` / `<tool_call>` tokens, hybrid thinking mode) · TRL `GRPOTrainer` + LoRA · [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) v4 · vLLM on Colab for generation and training, CPU locally for scoring and analysis

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
