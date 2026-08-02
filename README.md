# Adaptive Thinking for Tool Use

**Training a small open language model to reason before complex tool calls — and skip reasoning for routine ones.**

---

## The question

Function calling is how production AI agents actually work: user requests get routed through structured tool calls, and the reliability of those calls is a primary deployment constraint. Frontier models handle this well but cost too much at scale. Small open models are affordable but unreliable, which is why a wave of recent work applies reinforcement learning to close the gap.

Separately, a body of work has shown that explicit chain-of-thought reasoning is sometimes valuable and sometimes pure waste, helpful on ambiguous multi-step problems, a cost with no return on routine ones.

Current tool-use RL methods sit at one extreme or the other. They either **always** emit reasoning before a call, or **never** do. Neither learns a per-call policy.

> **Can a ≤2B-parameter model learn, through reinforcement learning, to invoke reasoning only when the tool call actually warrants it — and where does that adaptive policy land on the cost-quality Pareto frontier?**

## Approach

A single model emits an optional reasoning block before each tool call:

```
<think>…</think><tool_call>{"name": …, "arguments": {…}}</tool_call>   # reasoned
<tool_call>{"name": …, "arguments": {…}}</tool_call>                   # direct
```

Training is GRPO over a fully programmatic reward — no learned reward model, no human labels:

```
R  =  correctness  +  format_validity  −  λ · think_tokens
```

During rollouts the model samples both reasoned and direct continuations for the same input, and the reward decides which one pays off. **No complexity supervision is provided at any point** — the think/no-think gate is emergent, and λ traces out the cost-quality frontier.

### Central hypothesis

> **H1** — Learned think-probability rises monotonically with intrinsic call complexity.

Complexity is measured post-hoc against fixed properties of the benchmark item (BFCL category, candidate tool-set size, argument count and nesting depth, irrelevance-detection items) rather than against model-relative difficulty, which would make the correlation circular.

If H1 holds, the policy is genuinely adaptive rather than a constant policy in disguise. If it fails, the result is a characterization of *where* the gate misfires — which is a finding in its own right.

## Status

Early. Building the evaluation foundation.

| Phase | | |
|---|---|---|
| **A** | Per-sample BFCL scoring | in progress |
| **B** | Output contract + fixed-prompt baselines | pending |
| **C** | Reward function + test suite | pending |
| **D** | GRPO training | pending |
| **E** | Pareto + H1 analysis | pending |
| **F** | Paper, demo, documentation | pending |

Phase A blocks everything downstream: the evaluator *is* the reward function, so no training is possible until a single model output can be scored in isolation.

## Setup

```bash
pip install bfcl-eval
```

Full environment and reproduction instructions will land here as the pipeline stabilizes.

## Layout

```
├── src/
│   ├── scoring/       # per-sample BFCL scorer            (Phase A)
│   ├── prompts/       # output contract, policy templates (Phase B)
│   ├── rewards/       # GRPO reward + unit tests          (Phase C)
│   └── train/         # TRL GRPO training                 (Phase D)
├── experiments/       # run configs and logs
├── results/           # metrics, Pareto plots
├── notes/             # daily research notes
└── docs/              # proposal, design notes
```

## Stack

Qwen2.5-1.5B-Instruct (0.5B for pipeline debugging) · TRL `GRPOTrainer` + LoRA · vLLM rollouts · [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) · Colab Pro

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

Assistance to date covers literature mapping and citation verification, proposal structuring, and experimental design discussion. Per-file and per-deliverable labeling is maintained in [`notes/`](notes/).

---

*Research Study — bachelors in AI for Data and Management Sciences (AIDAMS), ESSEC Business School / CentraleSupélec, Paris-Saclay. Walid Ajbar, 2026.*
