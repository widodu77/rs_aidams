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
| **B** | Output contract + fixed-prompt baselines | done |
| **C** | Reward function + test suite | done |
| **D** | GRPO training | scaffolded, not yet run |
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
├── tests/             # scorer, reward and split suites   (61 tests)
├── results/
│   ├── raw/           # NF4 generations (superseded)
│   └── raw/vllm/      # fp16 generations (current)
├── notes/             # daily research notes + engineering_log.md
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
