# Research Study Proposal — Draft (v2, locked topic: tool use + RL)

**Student:** Walid Ajbar
**Program:** AIDAMS, ESSEC Business School (joint with CentraleSupelec / Paris-Saclay)
**Supervisor:** Woody Pan (program coordinator); academic supervisor TBD
**RS submission deadline:** End of August 2026
**Proposal deadline:** End of April 2026 (1-page proposal via Google Form)

> Note: an earlier version of this proposal centered on adaptive compute for math reasoning (over-thinking). That direction was archived as `RS_proposal_draft_v1_overthinking_archived.md` after a literature check showed the area had become saturated between October 2025 and April 2026. The current proposal pivots to tool use, where the adaptive-thinking angle is materially less explored.

---

## Working title

**Adaptive thinking for tool use: training small open language models to reason before complex tool calls and skip reasoning for routine ones**

*Alternate title:* "When should an agent think before it calls a tool? Reinforcement learning for cost-aware function calling in small open models"

---

## Google Form fields (ready to paste)

### 1. Project title

> Adaptive thinking for tool use: training small open language models to reason before complex tool calls and skip reasoning for routine ones

### 2. One-sentence project pitch

> I will train a small open-source language model with reinforcement learning to invoke explicit reasoning only before complex or ambiguous tool calls, while skipping reasoning entirely for routine ones. And characterize the resulting cost-quality Pareto frontier on standard function-calling benchmarks against fixed-policy baselines.

### 3. Problem description

> Function calling is the dominant interaction pattern for production AI agents in 2026. Every customer-facing agent (search assistants, coding agents, customer-service bots, voice assistants) routes user requests through structured tool calls, and the reliability of these calls is a primary deployment constraint. Frontier models perform well on tool use but are prohibitively expensive at scale; small open models are cheaper but unreliable, and a recent wave of work has applied reinforcement learning post-training to close the gap (R2IF, FunRL, RC-GRPO, AgenticQwen, ReCall).
>
> A separate line of recent work has documented that explicit chain-of-thought reasoning before model outputs is sometimes helpful (complex multi-step problems, ambiguous inputs) and sometimes harmful or simply wasteful (routine queries with obvious answers). Adaptive-thinking methods have been studied extensively for math reasoning (AALC, ALP, Re-FORC, GRPO-λ, IBPO, BudgetThinker), but the tool-use setting introduces a different structure: a tool call is a discrete, structured action whose correctness can be programmatically verified, and the cost of a wrong call (a failed API request, a hallucinated function name) is qualitatively different from the cost of a verbose answer.
>
> The problem is that current tool-use reinforcement learning methods either always emit reasoning before each call, or never emit reasoning, with no learned per-call policy. Always-thinking inflates inference cost on routine calls; never-thinking damages accuracy on ambiguous ones. No published method, at the small-model scale relevant for production deployment, learns when reasoning before a tool call materially improves the resulting call.
>
> **The Core Question:** Can a small (≤2 billion parameter) language model be trained, via reinforcement learning with a reward that combines tool-call correctness and an explicit reasoning-length penalty, to learn a per-call policy that invokes explicit reasoning only when the tool call complexity warrants it? Where does this adaptive policy sit on the cost-quality Pareto frontier compared to always-thinking and never-thinking baselines, and on which call types does adaptive reasoning fail or backfire?

### 4. Data source

> - Berkeley Function Calling Leaderboard (BFCL) v3 / v4 — the de facto benchmark for function calling, multi-turn evaluation, programmatic correctness — https://gorilla.cs.berkeley.edu/leaderboard.html
> - Nexus Function Calling Benchmark — multi-step composition, additional coverage — https://github.com/nexusflowai/NexusRaven-V2
> - τ-bench (Tau-Bench) — multi-turn dialog with goal completion under tool use, used for the secondary evaluation — https://github.com/sierra-research/tau-bench
> - **Base model:** Qwen3-1.7B — `<think>` (151667), `</think>` (151668) and `<tool_call>` (151657) are native single tokens, and reason-then-call in a single turn is a trained behaviour. Selected over the originally proposed Qwen2.5-1.5B-Instruct after empirical testing (see below); still ≤2B, so the small-model constraint holds
> - **Training framework:** TRL (built-in `GRPOTrainer`, supports custom reward functions and verifiable programmatic rewards; vLLM-accelerated rollouts)
> - **Reward signal:** programmatic — JSON parses, function name matches, arguments validate, multi-turn workflow completes — combined with a length penalty on the reasoning chain
>
> All datasets, models, and tooling are open-source and immediately available. The reward signal is fully programmatic; no learned reward model required.

---

## Defining and measuring call complexity

The core claim — reason "when the call complexity warrants it" — hides two distinct things. Separating them is what makes the contribution testable.

### 1. The decision mechanism: no complexity supervision required

The model is **not** given a complexity signal at inference time, and no complexity labels are used in training. The model may emit an optional `<think>...</think>` block before a call; during GRPO rollouts it samples both think and no-think continuations for the same prompt, and the reward

```
R = correctness − λ · reasoning_length
```

determines which choice pays off on that input. The think/no-think gate is therefore **emergent**: RL discovers the boundary rather than being told where it lies. This mirrors how AdaCoT and TON obtain adaptive triggering, and it means the method requires no annotation effort.

*This distinction matters for framing:* the proposal should not imply the model needs to know a call's complexity a priori. It does not. Complexity enters only as an **analysis axis**, below.

### 2. The analysis axis: how the learned policy is validated

Complexity is defined **post-hoc**, to characterize the policy the model actually learned. Two families, deliberately kept separate:

**Intrinsic complexity (model-independent) — the primary axis.**

| Measure | Rationale |
|---|---|
| BFCL category (simple → multiple → parallel → parallel-multiple → multi-turn) | A ready-made, published difficulty ordering; requires no labeling |
| Tool-set size (number of candidate functions in the prompt) | Proxy for disambiguation load |
| Argument count / nesting depth of the ground-truth call | Structural complexity of the target action |
| Irrelevance-detection items (correct action = emit *no* call) | The maximum-ambiguity case; where reasoning should help most |

**Empirical difficulty (model-relative) — corroborating only.** Per-item `1 − baseline_pass@k` under the never-think baseline.

### 3. The circularity trap, named explicitly

If complexity is defined as "the baseline fails this item" *and* success is defined as "the model thinks on hard items," the resulting correlation can be mechanically trivial: the model thinks when its own uncertainty is high, and uncertainty correlates with baseline failure. The analysis therefore **leads with the intrinsic axes**, which are fixed properties of the benchmark item and cannot be induced by the policy under study. Empirical difficulty is reported second, as corroboration.

### 4. The resulting testable hypothesis

> **H1.** Learned think-probability increases monotonically with intrinsic call complexity.

- **Supported** → the headline result: the policy is genuinely adaptive, not a degenerate constant policy in disguise.
- **Not supported** → a characterization of *where* the gate misfires (e.g. thinking on long-argument calls that are structurally trivial, or skipping reasoning on irrelevance items). This is the negative-result contribution, and it stands independently of whether a parallel paper claims the headline first.

Either outcome gives the per-call breakdown a concrete axis to break down on, which the analysis otherwise lacks.

---

## Why this proposal passes the rubric

The Research Study rubric (8 / 8 / 4 split):

- **Scientific problem solving and rigor (8 points)** — the problem is grounded in a clear gap: published tool-use reinforcement learning methods do not address per-call adaptive thinking, and published adaptive-thinking methods do not address tool use at small-model scale. Literature review covers ~10 papers across both lines.
- **Solution relevance and learnings (8 points)** — original contribution is the per-call adaptive policy plus the cost-quality Pareto characterization for tool use. Code is reproducible end-to-end (TRL + open data + open model).
- **Communication (4 points)** — the demo is intuitive: same trained model handling a routine tool call (no reasoning) and a complex one (explicit reasoning) without manual prompting.

The proposal is **negative-result-friendly**: if the adaptive policy collapses, or fails to beat always-thinking on accuracy, or fails to beat never-thinking on cost, the failure mode itself is a publishable finding (the rubric explicitly rewards this).

---

## How this is differentiated from existing work

| Existing work | What they did | What is left for this project |
|---|---|---|
| **R2IF (April 2026)** | GRPO with composite rewards for interpretable function calling on Qwen 1.5B/3B/7B and Llama-3.2-3B; +34.62% on BFCL | They optimize for accuracy with always-on reasoning. No adaptive thinking budget. |
| **EGPO / "Reasoning through Exploration" (August 2025, orig. titled FunRL)** | Entropy-enhanced advantage in GRPO, strong BFCL results with Qwen2.5-Coder-7B-Instruct | Larger model, accuracy-focused, no per-call reasoning policy. |
| **RC-GRPO (February 2026)** | Reward-conditioned GRPO for multi-turn tool calling on BFCLv4 | Trains on multi-turn workflows, not on the per-call thinking decision. |
| **AgenticQwen (April 2026)** | Multi-round RL on synthetic data for industrial-scale tool use, dual data flywheels | Production-scale focus, not research on adaptive thinking. |
| **BATS / BudgetThinker (November 2025 / August 2025)** | Budget-aware tool use with overall task budget tracking | Track total budget across the trajectory; do not learn a per-call thinking decision. |
| **AdaCoT (May 2025)** | Pareto-optimal adaptive CoT triggering via RL | General-purpose method, not applied to tool use at small scale. |
| **TON (May 2025)** | "Think or Not?" selective reasoning via RL for vision-language models | Different domain (VLM, not tool use). |

The intersection — *small open model, reinforcement learning trained, per-call adaptive reasoning policy, tool use specifically, cost-quality Pareto characterized* — is not yet published.

---

## Compute feasibility (Colab Pro budget)

- **Base model:** Qwen2.5-1.5B-Instruct with LoRA fine-tuning fits comfortably in 40 GB VRAM (Colab Pro A100 / L4)
- **Training time:** approximately 12–18 hours per RL run with vLLM-accelerated rollouts on BFCL data
- **Number of runs:** approximately 5–6 (always-think baseline, never-think baseline, length-penalty variants, the trained adaptive policy, plus 1–2 ablations) — fits comfortably in the ~80 GPU-hours/month budget over 3 months
- **Evaluation:** mostly inference, cheap; BFCL evaluation is automated

---

## Execution plan

Ordered by dependency, not by calendar. Each phase has an exit condition; a phase is not "done" until its exit condition holds, because every downstream phase consumes its output.

**Hard deadline: end of August 2026** (confirmed, AIDAMS RS/CRP deck slide 9). Deliverables: 4-page paper, 5-min recorded presentation with 3–5 slides + demo, code/data/documentation, and a folder of daily research notes.

### A — Per-sample BFCL scoring · *blocks everything*

The evaluator **is** the reward function, so nothing can be trained until a single model output can be scored in isolation.

- Obtain BFCL data; understand the entry schema (question, function definitions, ground-truth call)
- Build a callable `score(prediction, ground_truth) -> float`

> **Resolved (2026-08-02).** The anticipated obstacle — that BFCL's checker would be batch-only and need reimplementing — did not materialise. The `bfcl` CLI is batch, but `bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker()` is already a per-sample function returning `{"valid", "error", "error_type"}`. The official checker can therefore be used directly as the reward signal, which means reported numbers stay leaderboard-comparable and no divergence between reward and reporting has to be documented.
>
> Two real frictions surfaced instead:
>
> 1. **Import coupling.** Importing the checker pulls in `MODEL_CONFIG_MAPPING`, which imports every API model handler, which imports `qwen_agent`, which requires `soundfile`. A missing audio library blocks a pure-string AST checker. The entire chain exists to resolve one boolean, `underscore_to_dot` (a name-mangling quirk for providers that reject `.` in function names). Worked around by pinning `CHECKER_MODEL_NAME` to a config where the flag is False; flagged as a fragility risk for the Colab environment.
> 2. **Decode, not check, is the real work.** The checker expects `{"func_name": {arg: value}}`, whereas the model emits `{"name": ..., "arguments": {...}}`. The shape conversion, plus separating the `<think>` block from the call, is the part that had to be written.

**Exit: met.** `src/scoring/bfcl_scorer.py`, pinned to `bfcl-eval==2026.3.23` (BFCL v4), with a 25-test suite in `tests/`.

Verified in two layers. First, hand-written cases across `simple`, `multiple`, `parallel`, `parallel_multiple` and `irrelevance`, covering correct calls with and without reasoning, wrong function names, wrong argument values, missing required parameters, hallucinated parameters, type errors, wrong call counts, malformed JSON, prose-only output, and both directions of irrelevance. Two properties were confirmed rather than assumed: parallel scoring is order-independent (the model has no reason to emit calls in ground-truth order), and correctness is invariant to whether the model reasoned — the invariant the whole design rests on, since any leakage of the `<think>` block into the correctness term would mean the length penalty is no longer the only pressure acting on the thinking decision.

Second, a full-dataset sweep: every one of the 1000 single-turn items scored with a prediction reconstructed from its own ground truth. This is the check that matters, because hand-written cases only exercise paths someone thought to test. It began at 98.4% and each failure was a defect in the reconstruction rather than the scorer — the scorer was correctly rejecting malformed input. Reaching 100% surfaced three undocumented encoding rules in the BFCL ground-truth format:

1. Optional parameters should be omitted, not filled — BFCL ground truth sometimes lists optional parameters absent from the function schema entirely, so supplying them raises `unexpected_param`
2. `""` among acceptable values is *not* a reliable optionality marker; it also appears on parameters the schema lists as required, so the schema's `required` list is the authority
3. Acceptable-value lists nest recursively through lists as well as dicts

The sweep is retained as a parametrized regression test, so a future `bfcl-eval` bump that shifts the ground-truth encoding fails loudly rather than silently degrading every reward the training loop computes.

### B — Output contract + baseline behavior · *needs A*

- Define a format that lets the reasoning block and the tool call be parsed apart: `<think>…</think><tool_call>{…}</tool_call>` versus bare `<tool_call>…</tool_call>`
- Run the base model under two *fixed prompts*: always-think and never-think

> **Base model changed (2026-08-02): Qwen2.5-1.5B-Instruct → Qwen3-1.7B.** Anticipated by risk register item 3, but for a different reason than the one recorded there — capability to *represent* thinking, not capacity for tool-use complexity.
>
> Qwen2.5-1.5B-Instruct cannot reason and call within a single turn. `<think>` is not in its vocabulary (three tokens) and the model was never trained to emit it; that convention arrived with Qwen3/QwQ. Prompted to reason first, it writes `think:` as prose and then either answers the question directly without calling anything, or stops. Prefilling the opening tag produces well-formed `<think>` blocks, but the model then emits `<|im_end|>` immediately after `</think>` and never reaches the call. Feeding its own completed reasoning back as context yields an empty continuation: once `</think>` is present the turn is over. It does one thing per turn — reasons, or calls, never both.
>
> The workaround (prefilling `<tool_call>` to force a call) was rejected: it would make the always-think policy call on every irrelevance item by construction, destroying the most informative category with an artefact rather than a finding.
>
> This reframes the base-model choice as a substantive design decision rather than an arbitrary one: **a study of *when* to think requires a model that can represent thinking at all.** Qwen3 exposes `enable_thinking` in its chat template, which maps directly onto the two fixed policies and onto the free choice the Phase D policy must make.

**Exit:** accuracy and mean token cost for both fixed policies, broken down by BFCL category. Reported alongside three diagnostics that separate a result from an artefact: think-rate (must be ~1.0 and ~0.0, or the policies are not being enforced), truncation rate at the generation cap (truncated reasoning produces no call and scores zero, so a high rate means the cap rather than the model is driving accuracy), and no-call rate. These are the two anchor points of the Pareto frontier and the axis for H1 — a reportable result obtained before any training.

### C — Reward function · *needs A + B*

`R = correctness + format_validity − λ · think_tokens`

Two design decisions that carry weight:
- **Penalize think tokens only**, not total output length — the thinking decision is the object of study, and penalizing the call itself confounds it
- **Format-term strictness** — too lenient and the model games it; too strict and it never receives usable signal

**Exit:** unit tests pass over hand-constructed cases — correct call with reasoning, correct call without, malformed JSON, right function with wrong arguments, correct refusal on an irrelevance item. Reward bugs are the primary silent failure mode in RL; this test suite is cheap insurance against days of misread training curves.

### D — GRPO training · *needs C*

1. TRL `GRPOTrainer` + LoRA + vLLM rollouts running end-to-end on a tiny subset — "does it execute"
2. **Go/no-go gate:** train on correctness reward *only*, no length penalty. Does accuracy improve over base? If GRPO cannot improve plain correctness on this model, the adaptive question is moot and the project pivots to characterizing why.
3. Only after the gate passes: introduce λ and sweep it

**Instrument think-rate from the first run.** Collapse to all-think or all-no-think is the primary scientific failure mode, and it must be observed as it happens rather than discovered at the end. Mitigation if it appears: stage the training (correctness first, length penalty introduced after stabilization) and cap the penalty.

**Exit:** 3–4 trained policies across λ values, with logged training curves and think-rate traces.

### E — Analysis · *needs D*

- Pareto curve: accuracy versus mean think-tokens, one point per λ plus the two fixed baselines
- **H1 test:** think-rate against intrinsic complexity (see *Defining and measuring call complexity*)
- Failure taxonomy: thought-and-wrong versus skipped-and-wrong
- Qualitative examples selected for the demo video

### F — Deliverables · *continuous, not terminal*

Paper, slides, Loom recording, code, documentation. The **daily research notes folder is itself graded** and must accumulate from day one — it cannot be reconstructed at the end.

---

### Scope boundaries

Deliberately excluded to keep the project completable. Each is a defensible extension, not an omission:

| Excluded | Reason |
|---|---|
| τ-bench secondary evaluation | Multi-turn goal-completion harness is a time sink; the single-turn result stands alone |
| Nexus Function Calling Benchmark | Redundant coverage relative to BFCL for the question asked |
| BFCL multi-turn categories | Evaluation harness complexity; single-turn + irrelevance-detection retained (irrelevance is the most informative category for this question) |
| Hammer-1.5B fallback base model | No headroom to swap base models mid-project |
| SFT comparison baseline | Requires training a second model; the never-think fixed-prompt baseline serves the comparison |

Training runs are kept short — a few hundred GRPO steps on a data subset with LoRA, rather than runs to convergence. For a 4-page paper a clear trend across λ is worth more than a single converged number, and long runs are unreliable under Colab session limits regardless. Debug the pipeline on Qwen2.5-**0.5B**-Instruct before moving to 1.5B.

---

## Key references

Primary tool-use + RL line:
- **R2IF: Aligning Reasoning with Decisions via Composite Rewards for Interpretable LLM Function Calling** (April 2026). arXiv:2604.20316
- **Reasoning through Exploration: A Reinforcement Learning Framework for Robust Function Calling** (Hao et al., August 2025). arXiv:2508.05118 — *cite under this v2 title; v1 was "Exploring Superior Function Calls via RL" (FunRL). Method is EGPO.*
- **RC-GRPO: Reward-Conditioned GRPO for Multi-Turn Tool Calling Agents** (February 2026). arXiv:2602.03025
- **AgenticQwen: Training Small Agentic Language Models with Dual Data Flywheels** (April 2026). arXiv:2604.21590
- **From Self-Evolving Synthetic Data to Verifiable-Reward RL: Multi-turn Tool-Using Agents** (January 2026). arXiv:2601.22607
- **Berkeley Function Calling Leaderboard (BFCL)**. OpenReview link 2GmDdhBdDk
- **Hammer: Robust Function-Calling for On-Device Language Models via Function Masking**. arXiv:2410.04587

Adaptive-thinking line (cross-domain context):
- **AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via RL** (May 2025). arXiv:2505.11896
- **TON: Think or Not? Selective Reasoning via RL for Vision-Language Models** (May 2025). arXiv:2505.16854
- **BATS / Budget-Aware Tool-Use Enables Effective Agent Scaling** (November 2025). arXiv:2511.17006
- **BudgetThinker: Empowering Budget-Aware LLM Reasoning with Control Tokens** (August 2025). arXiv:2508.17196
- **Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning** (November 2025). arXiv:2511.02130
- **Stable Reinforcement Learning for Efficient Reasoning** (Dai, Liu, Si, May 2025). arXiv:2505.18086 — *method is named GRPO-λ; the λ prefix is not part of the paper title.*

Survey anchors:
- **A Survey of Reinforcement Learning for Large Reasoning Models** (Zhang et al., October 2025). arXiv:2509.08827
- **Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs** (July 2025). arXiv:2507.02076

Foundational RL methods:
- **DeepSeek-R1** (Guo et al., January 2025) — GRPO, RLVR
- **DAPO** (Yu et al., 2025) — clip-higher, dynamic sampling
- **DeepSeek-Math** (Shao et al., 2024) — original GRPO

---

## Risk register (named in the proposal because the field moves fast)

1. **Scoop risk on the headline angle.** Adaptive-thinking-for-tool-use is an obvious next step that other groups are likely also considering. Mitigation: structure the contribution so the per-call breakdown analysis and the failure-mode characterization stand on their own even if a parallel "adaptive thinking for tool use" paper appears. The paper is built on multiple sub-contributions, not one headline.

2. **Reward hacking on length.** The length-penalty reward can cause the model to collapse to never-thinking even on calls where reasoning would help. Mitigation: stage the training (correctness reward only, then introduce length penalty after the model has stabilized); cap the penalty.

3. **Model too small for tool-use complexity.** Qwen2.5-1.5B may struggle on multi-turn BFCL even before introducing the adaptive policy. Mitigation: Hammer-1.5B is a fallback base model; the project can also pivot to a 3B base if necessary, sacrificing some compute headroom.

4. **Compute headroom is tight on Colab Pro.** GRPO with vLLM rollouts plus reward computation per step on BFCL trajectories burns GPU time. Mitigation: keep number of training runs to 5–6, use LoRA adapters rather than full fine-tuning, prioritize the most informative ablations.

5. **BFCL evaluation drift.** BFCL versions are updated periodically; results may not match published numbers exactly. Mitigation: pin to one BFCL release for the duration of the project, document the version in the paper.

---

## LLM use disclosure

Per the Research Study rules, any use of large language models during this project will be:
1. Limited to tasks the student fully understands and can explain
2. Clearly labeled in every deliverable

This proposal was drafted with assistance from Claude (Anthropic) for ideation, structuring, literature mapping, and verification of TRL library APIs and BFCL benchmark details. All technical claims, the research question, and the experimental plan reflect the student's own understanding and intent.

---

## Open questions before submission

1. **Academic supervisor.** Reach out to an AIDAMS / CentraleSupelec / Paris-Saclay faculty member working in NLP, reinforcement learning, or AI agents. Candidates to identify in the next two weeks.
2. **Compute back-up plan.** If Colab Pro proves too restrictive, confirm whether ESSEC or Paris-Saclay has a GPU cluster that can host longer training runs.
3. **Corporate Research Project linkage.** This Research Study topic anchors directly to any company building production AI agents (consulting firms automating internal workflows, customer-service automation companies, coding agent products, search assistants). Worth thinking about specific company contacts now so the Research Study can serve as a CRP pitch in September.

---

*Draft last updated: 2026-04-30. Locked topic: tool use + reinforcement learning with adaptive per-call thinking policy on small open models.*
