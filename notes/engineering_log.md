# Engineering log

Every obstacle that forced a change to the code or the architecture, recorded when it happened.
Separate from the daily research notes: those track what was *found*, this tracks what *broke* and
what the fix taught. Newest last.

Format per entry: **symptom** (the literal error where there was one), **root cause**, **fix**, and
**lesson** — the transferable part, which is the reason this file exists.

---

## 1 — BFCL's checker drags an audio library into a string comparison

**Phase A · 2026-08-02**

**Symptom.** Importing `ast_checker` failed on a missing `soundfile`.

**Root cause.** `ast_checker` → `MODEL_CONFIG_MAPPING` → every API model handler → `qwen_agent` →
`soundfile`. The entire chain exists to resolve one boolean, `underscore_to_dot`, which rewrites
`.` to `_` in function names for providers whose APIs reject dots.

**Fix.** Pinned `CHECKER_MODEL_NAME = "gorilla-openfunctions-v2"`, a config where the flag is False,
so names are compared verbatim.

**Lesson.** Before working around a heavy import, find out what it is actually *for*. A 4322-module
dependency chain resolving a single boolean is worth ten minutes of reading. This one came back
twice more — see entries 6 and 7.

---

## 2 — Three undocumented rules in the BFCL ground-truth encoding

**Phase A · 2026-08-02**

**Symptom.** Scoring all 1000 single-turn items with predictions reconstructed from their own
ground truth started at 98.4%, not 100%.

**Root cause.** Every failure was a defect in the reconstruction, not the scorer — the scorer was
correctly rejecting malformed input. Three rules had to be discovered by reading checker errors:

1. Optional parameters should be omitted, not filled: BFCL ground truth sometimes lists optional
   parameters absent from the function schema entirely, so supplying them raises `unexpected_param`.
2. `""` among acceptable values is *not* a reliable optionality marker — it also appears on
   parameters the schema lists as required. The schema's `required` list is the authority.
3. Acceptable-value lists nest through **lists** as well as dicts.

**Fix.** Encoded all three in the test helper; kept the 1000-item sweep as a parametrized
regression test.

**Lesson.** A self-consistency sweep over the whole dataset finds things hand-written tests never
will, because hand-written tests only cover paths someone thought of. And when a check fails,
suspect the *checker of the checker* first.

---

## 3 — `format_ok` conflated two different failures

**Phase A · 2026-08-02**

**Symptom.** None — caught while writing, not by a test. Which is the point of recording it.

**Root cause.** "Emitted malformed JSON" and "emitted no call at all" were the same flag. On
irrelevance items, emitting zero calls is the *correct* answer.

**Fix.** Split into `format_ok` (nothing malformed) and `has_calls` (a call was produced). Whether
"no call" is right is category-dependent and belongs to the scorer, not the parser.

**Lesson.** This flag later became the reward's format term, where the same conflation would have
made *silence* a profitable strategy — see entry 8. A muddled abstraction gets more expensive the
further downstream it travels.

---

## 4 — Qwen2.5 cannot reason and call in the same turn

**Phase B · 2026-08-02**

**Symptom.** Under the always-think policy the model produced a well-formed `<think>` block, closed
it, and immediately emitted `<|im_end|>` — never reaching the tool call. Feeding its own completed
reasoning back as context produced an *empty* continuation.

**Root cause.** `<think>` is not in Qwen2.5's vocabulary (three tokens) and Qwen2.5-Instruct was
never trained to emit it; that convention arrived with Qwen3/QwQ. Once `</think>` is in context,
the model considers the turn over. A property of the model, not the prompt — three successive
prompt fixes each revealed the next layer.

**Fix.** Changed base model to Qwen3-1.7B, where `<think>` (151667) and `</think>` (151668) are
native single tokens and `enable_thinking` is an explicit chat-template flag.

**Rejected fix, and why it matters.** Prefilling `<tool_call>` to force a call would have worked —
and would have made the always-think policy call on every irrelevance item *by construction*,
destroying the most informative category with an artefact.

**Lesson.** A study of *when* to think needs a model that can represent thinking at all. Also: when
a workaround would manufacture the result you are trying to measure, it is not a workaround.

---

## 5 — Qwen2.5 never closes its own tool-call tag

**Phase B · 2026-08-02**

**Symptom.** Three obviously-correct tool calls scored 0.0.

**Root cause.** The model emits `<tool_call>` and the JSON, then goes straight to `<|im_end|>`
without ever emitting `</tool_call>`. Not truncation (19–26 tokens against a 128 cap) and not
special-token stripping (confirmed by decoding with `skip_special_tokens=False`).

**Fix.** Made the closing tag optional — parse with `json.JSONDecoder().raw_decode`, which consumes
exactly one value and reports where it ended, instead of matching a closing delimiter. Leniency
stops there: malformed JSON still fails.

**Lesson.** Requiring the tag would have made the baselines measure the model's tag-emission habits
rather than the effect of reasoning on correctness — confounding the exact comparison the project
exists to make. Decide what a parser is allowed to forgive by asking what you are trying to
measure.

---

## 6 — `[tool.uv.sources]` only overrides *direct* dependencies

**Phase B · 2026-08-02**

**Symptom.** CUDA torch was configured but a CPU build stayed installed.

**Root cause.** torch arrived transitively via `bfcl-eval`, so the PyTorch index was silently
ignored. Found by grepping `uv.lock` for the torch source: `registry = "pypi.org"` gave it away.

**Fix.** Declared `torch` explicitly in `[project.dependencies]`.

**Lesson.** A dependency override that is silently ignored looks exactly like one that worked.
Verify against the lockfile, not the config.

---

## 7 — The Colab dependency fight was avoidable in one line

**Phase B → Colab migration · 2026-08-09**

**Symptom.** A full `bfcl-eval` install fights vLLM over the torch pin. Flagged back in Phase A as
a standing risk.

**Root cause.** Measured instead of assumed: `import bfcl_eval` costs **0.002s and 38 modules**
(the package `__init__.py` is empty), while `from ...ast_checker import ast_checker` costs **9.8s
and 4322 modules**. All the weight is one import line — and generation never needs the checker at
all, only the dataset.

**Fix.** Deferred the `ast_checker` import into `score_sample`. The module then imports under
`pip install --no-deps bfcl-eval==2026.3.23`, which brings the data files and nothing else.

**Lesson.** The Phase A note called this "a fragility risk for the Colab environment". It turned
out to be a one-line change, but only after measuring where the cost actually was. Measure the
import graph before designing around it.

---

## 8 — The format reward would have made silence profitable

**Phase C · 2026-08-09**

**Symptom.** None — caught while writing the reward, and pinned with a test.

**Root cause.** `format_ok` means "nothing malformed was emitted" and is therefore True for prose
containing no tool call at all (entry 3). At a weight near correctness, a model could collect most
of the reward by emitting nothing and never risking a wrong call.

**Fix.** `w_format = 0.2`, so it stays a tie-breaker between two genuine attempts. Test asserts
well-formed prose scores below a correct call.

**Lesson.** In RL, a reward bug does not raise an error — it produces a confidently trained policy
that learned the wrong thing. Every degenerate strategy has to be enumerated and priced *before*
training, because afterwards it just looks like a result.

---

## 9 — Static batching was wasting 2.36x of the GPU

**Colab migration · 2026-08-09**

**Symptom.** The local run took 3h37m for 359 items — 2.7x slower than the 340 ms/step measured in
an earlier smoke test.

**Root cause.** Measured rather than blamed on the card: **14229 decode steps produced 96609 useful
tokens.** HF `generate` batches statically, so every sequence steps until the *longest* in its
batch finishes; one rambling item drags fifteen idle slots along. Length-sorting the same batches
would cut 52% of the steps — but completion length is not knowable in advance.

**Fix.** Moved generation to vLLM, whose continuous batching refills a slot the moment a sequence
completes. 3 policies x 1240 items in **36 minutes**, against a ~16 h local projection.

**Lesson.** The early smoke test measured short, homogeneous batches and gave a number that did not
generalise. Benchmark on the real workload's length distribution, not a convenient sample.

---

## 10 — Same torch version number, different CUDA ABI

**Colab migration · 2026-08-09**

**Symptom.** `ImportError: libcudart.so.13: cannot open shared object file`.

**Root cause.** vLLM 0.26.0 ships a CUDA 13 binary. Colab preinstalls `torch 2.11.0+cu128`. pip saw
that `2.11.0` satisfied vLLM's `torch==2.11.0` requirement and left it alone — so a cu13 `.so` went
looking for a cu13 runtime on a cu12 system. The absence of a "RESTART SESSION" prompt was the
clue: pip had not touched torch at all.

**Fix.** Reinstalled the same torch *version* from the cu130 index.

**Lesson.** A matching version number does not imply a matching CUDA ABI, and pip's resolver cannot
see the difference. Also: two later `libnvrtc.so.13` errors from the same root cause were harmless
— one an optional Hopper kernel library, one in vLLM's *shutdown* path after records were written.
Read where in the lifecycle an error occurs before treating it as fatal.

---

## 11 — Arrow cannot hold BFCL's ground truth

**Phase D · 2026-08-09**

**Symptom.** `pyarrow.lib.ArrowInvalid: Could not convert 'true' with type str: tried to convert to
boolean`, raised from `Dataset.from_list`.

**Root cause.** `datasets` builds an Arrow schema by type inference, and neither `function` nor
`ground_truth` can be given one: every item's function schema has a different shape, and BFCL
ground truth mixes types inside a single acceptable-values list — `"formatted": [true, ""]`, which
is entry 2's rule 2 resurfacing in a new place.

**Fix.** Carry both columns as JSON strings; decode at the TRL boundary in `rewards.reward`, leaving
`compute_reward`'s direct-call path taking real Python objects.

**Verification.** Reproduced both directions locally with an ephemeral pyarrow (`uv run --with
pyarrow`): the old shape fails with the identical error, the new one builds a 1240-row all-string
table with zero round-trip mismatches.

**Lesson.** The dangerous version of this bug is the one that does *not* raise. Had the reward
silently received a JSON string instead of a schema, every rollout would have scored zero with no
error — a flat reward curve for hours before anyone suspected the data path. Hence the test
asserting encoded and decoded columns give identical rewards.

---

## 12 — TRL removed `max_prompt_length`

**Phase D · 2026-08-09**

**Symptom.** `TypeError: GRPOConfig.__init__() got an unexpected keyword argument
'max_prompt_length'`.

**Root cause.** TRL 1.9.2 dropped the field; there is no replacement.

**Fix.** Removed the argument. Its absence is harmless here — the worst observed BFCL prompt is 905
tokens against a 32k context, and truncating a prompt would delete function schemas and make the
item unanswerable, so prompt truncation was never something to want.

**Verification.** Rather than guessing, introspected `dataclasses.fields(GRPOConfig)` and
`inspect.signature(GRPOTrainer.__init__)` against the installed TRL. `max_prompt_length` was the
*only* rejected argument; everything else matched. `trl` is now pinned `>=1.9,<2`.

**Lesson.** Fast-moving library APIs are worth introspecting rather than reading about, especially
when the failure costs a GPU session. Two smoke-test failures in a row (this and entry 11) both
landed before any GPU work — which is exactly what a five-step smoke test is for.

---

## 13 — An unused library broke LoRA by *raising* from a feature-detection check

**Phase D · 2026-08-09**

**Symptom.**

```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

raised from `peft/tuners/lora/model.py` during `get_peft_model`.

**Root cause.** Nothing in this project uses torchao — it is a quantization library, and LoRA here
runs in fp16. But PEFT builds a LoRA layer by walking a list of *dispatchers*, one per backend
(bitsandbytes, gptq, awq, hqq, torchao, …), asking each whether it applies. `dispatch_torchao`
calls `is_torchao_available()`, and that helper **raises** on a too-old version rather than
returning False. Colab preinstalls torchao 0.10.0. So an unused, unwanted, unreferenced backend
aborted adapter injection.

**Fix.** `pip uninstall -y torchao`. When the package is absent the same helper returns False
cleanly and the dispatcher moves on. Uninstalling was preferred over upgrading: it is
deterministic and cannot drag a new dependency into a torch install that is already delicately
balanced (entry 10). The verify cell now asserts torchao's absence as a precondition.

**Lesson.** A feature-detection predicate should answer the question it was asked. `is_X_available()`
raising instead of returning False turns an optional backend into a hard dependency on *not*
having an old version of it — a failure mode that is invisible from your own code and dependency
list, because the offending package is neither imported nor requested. When a traceback names a
library you have never heard of in this context, check whether you are on a plugin-discovery path
before assuming you need it.

This is the third distinct way the *preinstalled Colab environment* has broken a run (entries 10,
12, 13). The pattern is worth naming: Colab is not a clean machine, and every install fights
whatever was already there.

---

## 14 — The `--no-deps` trick did not survive contact with training

**Phase D · 2026-08-09**

**Symptom.** Inside the training loop, on the first batch of rollouts:

```
ModuleNotFoundError: No module named 'anthropic'
```

raised from `score_sample` → the deferred `ast_checker` import.

**Root cause.** My own design error, and the interesting kind: a fix that was correct for one
environment, reused in another where its premise did not hold. Entry 7 made the checker import
lazy so that **generation** could run under `pip install --no-deps bfcl-eval` — generation only
reads the dataset and never scores. I then copied `--no-deps` into the training notebook, where
scoring is the entire point: the reward *is* the checker, called once per rollout. The lazy import
did exactly what it was designed to do, and deferred the failure from startup to the first reward
computation.

**Why the obvious fix was wrong.** Installing bfcl-eval's real dependencies was measured before
being attempted: importing `ast_checker` pulls **4322 modules across 81 third-party top-level
packages** — `anthropic`, `boto3`, `cryptography`, `black`, `tree_sitter`, `qwen_agent`,
`soundfile` and so on. Beyond being absurd, it would also pull bfcl-eval's torch pin and upgrade
the torch that trl/peft/transformers are already working against — re-opening entry 10.

**Fix.** Went back to what entry 1 established: the whole chain exists to resolve **one boolean**.
`ast_checker.py` uses `MODEL_CONFIG_MAPPING` in exactly one place, line 86:

```python
model_name_escaped = model_name.replace("_", "/")
if "." in function_name:
    if MODEL_CONFIG_MAPPING[model_name_escaped].underscore_to_dot:
```

Since `CHECKER_MODEL_NAME` is pinned to a config where that flag is False, the value is known
statically. `_install_model_config_stub()` injects a minimal `bfcl_eval.constants.model_config`
into `sys.modules` before importing the checker, so the handler chain is never touched. The real
import is still preferred and used wherever the full install exists; the stub is a fallback.

The stub serves **only** the pinned model and raises `KeyError` for anything else — guessing False
for an OpenAI/Mistral/Google-style handler would silently compare function names the wrong way,
which is a wrong-answers bug, not a crash.

**Verification.** Scored all 1000 single-turn items twice in separate processes — once normally,
once with the handler packages made unimportable to force the stub path — and compared verdicts
item by item. **1000 items, 0 mismatches.** Two tests now pin it: that the stub refuses unknown
models, and that `CHECKER_MODEL_NAME` contains no underscore (since `ast_checker` looks up
`name.replace("_", "/")`, the stub's key and the lookup only agree while that holds).

**Lesson.** Two of them. First: when you narrow an environment to make one job fit, check every
other job that will run in it — `--no-deps` was a correct fix carried into a context whose
requirements were strictly larger, and the laziness that made it work is exactly what hid the
problem until mid-training. Second: the answer was already in entry 1, written seven days earlier.
"This 4322-module chain resolves one boolean" was recorded as an observation; it turned out to be
the fix. Keeping the log is what made that reusable.

---

## 15 — The first smoke test passed, and the passing numbers were the problem

**Phase D · 2026-08-09**

**Symptom.** None. Five steps ran, the adapter saved, no error. The problems were in the logged
metrics, which is the only reason they were caught before a 200-step run.

**Finding A — three of five steps produced no gradient.**

```
step 2:  frac_reward_zero_std 1    loss 0    grad_norm 0
step 3:  frac_reward_zero_std 1    loss 0    grad_norm 0
step 5:  frac_reward_zero_std 1    loss 0    grad_norm 0
```

GRPO computes advantage *within* a group of rollouts on the same prompt. When every rollout scores
identically the advantage is zero and the step teaches nothing. Step 5 is the benign-looking
version: all four rollouts correct, reward 1.2 across the board, no learning.

This is structural at lambda = 0 and it sharpens what the go/no-go gate actually measures. With
correctness + format only, every saturated group is dead weight — and `simple_python` is 92.5%
correct under *never*, so saturation is the common case. Introducing lambda partly fixes it:
rollouts that agree on correctness still differ in think length, so the group keeps variance. The
group size was also only 4; raised to 8.

Related: `entropy` logged at 0.04–0.12, which is low — near-identical rollouts produce
near-identical rewards, compounding the problem. Worth noting that EGPO (arXiv:2508.05118, already
in the reference table) is precisely about entropy-enhanced exploration for function calling. That
citation moved from background to directly relevant.

**Finding B — the configuration could not finish in a Colab session.**

Smoke ran ~37.5 s/step at **4 completions x 256 tokens**. The intended config was 32 completions x
768 tokens — 24x the generation work, so ~15 min/step, ~75 h for 300 steps.

Root cause is entry 9 all over again, in a new place: HF `generate` batches statically, and the
rollout loop runs it hundreds of times. Fixed the same way — `use_vllm=True` with
`vllm_mode="colocate"`, which keeps vLLM in-process (the only shape that fits one Colab GPU).
Batch also cut to 16 completions/step, and steps to 200, consistent with the proposal's "a few
hundred GRPO steps rather than runs to convergence".

Installing vLLM into the training environment re-opens entry 10, so the notebook's install cell now
has a **required order**: vLLM first (it has the strongest opinions about torch), then trl/peft,
then the cu130 torch repair — which must come *after* vLLM or vLLM's install reverts it — then the
torchao uninstall.

**Lesson.** A smoke test that only answers "did it crash" is worth much less than one that logs the
quantities you would have to reason about later. Both findings were invisible in the pass/fail
outcome and obvious in the metrics — and the metric functions existed only because collapse was
identified in advance as the primary failure mode. Instrument for the failure you expect, then read
the instruments even when the run succeeds.

Also worth noting: entries 9, 10 and 15 are one problem wearing three hats. Static batching cost
2.36x in generation, then ~24x in training; and each time vLLM was the fix, the CUDA ABI issue came
along with it.

---

## 16 — `--index-url` replaces PyPI, and took numpy down with it

**Phase D · 2026-08-09**

**Symptom.** A 40-frame traceback ending in

```
ModuleNotFoundError: Could not import module 'AutoModel'.
Are this object's requirements defined correctly?
```

whose actual cause was four exceptions deeper:

```
AttributeError: module 'numpy._core._multiarray_umath'
                has no attribute '_blas_supports_fpe'
```

**Root cause.** The fix from entry 10 — `pip install --force-reinstall torch==X --index-url
https://download.pytorch.org/whl/cu130` — has a side effect I did not account for. **`--index-url`
*replaces* PyPI rather than adding to it**, unlike `--extra-index-url`. Combined with
`--force-reinstall`, which reinstalls the entire dependency tree rather than just the named
package, pip pulled every one of torch's dependencies from the PyTorch index, which serves older
pinned versions. numpy was downgraded underneath compiled extensions in transformers and vLLM that
were built against a newer one, and a C-level attribute vanished.

It did not bite in the generation notebook because less was installed there for the downgrade to
break.

**Fix.** Restore numpy from PyPI explicitly after the torch repair
(`pip install -U numpy --index-url https://pypi.org/simple`), and reorder so `trl`/`peft`/
`datasets` install *after* the torch repair rather than before — that way they resolve their
compiled dependencies against the torch that will actually be used. The verify cell now imports
and prints numpy first, and imports `AutoModelForCausalLM` explicitly, so a recurrence shows up as
one line instead of a mystery.

**Lesson.** Two. First, `--force-reinstall` plus `--index-url` is a much bigger hammer than it
looks: one replaces the whole package universe, the other reinstalls the whole dependency tree, and
together they silently rewrite packages you never named. Second, and more general — **the error
message named `AutoModel`, which had nothing to do with the problem.** Python's lazy-import
machinery re-raises through several layers, so the top of a traceback is often the last thing to
notice a failure rather than the first thing to cause it. Read to the bottom of the chain before
forming a hypothesis.

Install-order dependencies have now accumulated to six steps, each justified by a different entry
in this log. That the ordering is load-bearing is itself the finding: this environment is not
reproducible by listing packages, only by listing packages *and* the sequence.

---

## 17 — The diagnostic cell was reporting the kernel, not the disk

**Phase D · 2026-08-09**

**Symptom.** After applying the entry-16 fix, the verify cell still reported

```
numpy     : 2.0.2
torch     : 2.11.0+cu128 | CUDA 12.8
```

i.e. *neither* repair appeared to have happened — the numpy upgrade and the cu130 torch reinstall
both looked like no-ops.

**Root cause.** They may well have happened; the cell could not see it. The install cell began with
`import torch` in order to read `torch.__version__` and build the pin for the cu130 reinstall. That
single line loads torch **and numpy** into the kernel. Every pip command afterwards rewrites files
on disk, but `sys.modules` keeps serving the already-imported modules for the life of the session.
So the versions printed were whatever was loaded before the installs ran, and the failure was
indistinguishable from pip having done nothing.

Colab compounds this by only sometimes offering the RESTART SESSION button, which had been treated
as the signal that a restart was needed.

**Fix.** Three changes:

- The install cell no longer imports anything heavy. `importlib.metadata.version("torch")` reads
  package metadata **without importing the package**, so the kernel stays clean.
- The cell ends with a check run in a **subprocess** (`!python -c "import numpy, torch; ..."`),
  which gets a fresh interpreter and therefore reports what is genuinely on disk.
- The restart is documented as mandatory rather than conditional on Colab offering it.

Also pinned `numpy>=2.2` explicitly rather than a bare `-U`: `_blas_supports_fpe` arrived in numpy
2.1, and an unqualified upgrade can be held at 2.0.x by another package's pin without saying so.

**Lesson.** In a long-lived interpreter, "what is installed" and "what is imported" are different
questions, and pip only answers the first. Any check meant to verify an installation has to run in
a process that started *after* it — otherwise a successful install and a failed one look identical.
This is the second time in two entries that the visible symptom pointed away from the cause
(entry 16's traceback blamed `AutoModel`); both cost a full round trip, and both were cheap to make
self-diagnosing once seen.

---

## 18 — OOM in the loss pass: the logits tensor, not the activations

**Phase D · 2026-08-09**

**Symptom.** vLLM colocate initialised, CUDA graphs captured, rollouts generated — then

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 150.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 47.81 MiB is free.
Of the allocated memory 13.78 GiB is allocated by PyTorch, with
457.95 MiB reserved by PyTorch but unallocated.
```

raised from `_get_per_token_logps_and_entropies` → SDPA attention. Generation succeeded; the
*loss* pass ran out.

**Root cause.** Three tenants on a 14.56 GiB card at once:

| | |
|---|---|
| vLLM colocate (own weights + KV cache, at 0.35) | ~5.1 GiB |
| training model, fp16 | ~3.4 GiB |
| **logits in the log-prob forward** | **~1.9 GiB, twice** |

The last one is the part that is easy to miss. Gradient checkpointing was already on, so
activations were not the problem — but GRPO needs *per-token log-probabilities*, and computing
them materialises `batch x completion_tokens x vocab`. With Qwen3's 151936-token vocabulary that
is `8 x 768 x 151936 x 2 bytes ≈ 1.9 GiB` in fp16, and TRL does it twice per step (policy, then
reference with the adapter disabled). Nothing about that scales with model size, which is why "a
1.7B model on a 15 GiB card" felt like it should be comfortable and was not.

**Fix.** Three changes, in decreasing order of effect:

1. `per_device_train_batch_size` 8 → **2**. This is the number of completions in one forward, so
   it divides the logits tensor directly (~1.9 GiB → ~0.5 GiB). Group size stays 8 via
   `gradient_accumulation_steps` 4, so the GRPO group is unchanged — only the forward is split.
2. `vllm_enable_sleep_mode=True`. In colocate mode vLLM otherwise holds its ~5 GiB for the whole
   run; sleeping offloads it between generation phases, which is exactly the window the loss pass
   needs.
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set before torch initialises CUDA. The OOM
   reported 458 MiB reserved-but-unallocated: alternating rollouts and loss allocates large,
   differently-shaped tensors every step, which fragments the caching allocator.

The script now prints the predicted logits size at startup, so the dominant term is visible before
the run rather than after it.

**Lesson.** For RL on language models, peak memory is often set by `vocab_size`, not by parameter
count — and the batch dimension that controls it is the *forward* batch, which can be decoupled
from the algorithmic group size via gradient accumulation. Worth knowing that these two knobs look
identical in a config file and do completely different things: one is a memory lever, the other is
a statistics lever.

---

## 19 — A warning I dismissed as cosmetic turned out to be load-bearing

**Phase D · 2026-08-09**

**Symptom.**

```
pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, cumem allocator is not supported on current platform.
```

raised while constructing vLLM inside `GRPOTrainer`, after enabling sleep mode (entry 18, fix 2).

**Root cause.** Sleep mode offloads vLLM's weights and KV cache via its **cumem allocator**, and
that allocator needs `libnvrtc.so.13` — the CUDA 13 runtime-compilation library, which the cu130
torch wheel does not install (it brings `libcudart.so.13` but not NVRTC).

The same missing library had already appeared, twice, during baseline generation:

```
ImportError: libnvrtc.so.13 ... from vllm.cumem_allocator import
```

I assessed it as harmless at the time and that assessment was correct *for generation* — it
appeared once for an optional Hopper kernel library and once in vLLM's shutdown path, after all
records had been written. Entry 10 says so explicitly. But "harmless in the shutdown path of a
generation run" does not generalise to "harmless", and the moment sleep mode needed the same
allocator it became fatal at construction.

**Fix.** Inverted the flag: sleep mode is now opt-in (`--vllm-sleep`) rather than default, so a
missing NVRTC cannot break a run that never needed it. Memory headroom comes from
`--per-device-batch-size 2` instead, which was always the safer lever — it shrinks the logits
tensor directly and depends on nothing. Sleep mode remains available for anyone who first gets
`import vllm.cumem_allocator` working.

**Lesson.** This is entry 14's shape again, from the other direction. There, a fix correct for one
environment was carried into another where its premise failed. Here, a *diagnosis* correct for one
context — "this error is cosmetic" — was carried into another where the same missing dependency was
required. Both are the same underlying mistake: treating a conclusion as unconditional when it was
actually scoped to circumstances that later changed.

Practical version: when you write off an error as harmless, write down *why* it is harmless. Entry
10 recorded "it appears in the shutdown path", and that note is what made this diagnosis take
minutes instead of another blind round trip.

---

## 20 — Phase D runs. The missing library was there all along, under another name

**Phase D · 2026-08-09**

**Resolution of entry 19.** `pip install nvidia-cuda-nvrtc` alone did not fix it — the import still
failed. The library *was* on disk, in `site-packages/nvidia/<component>/lib`, but shipped as
`libnvrtc.so.13.x.y` with no bare `libnvrtc.so.13` soname, and its directory was not on
`LD_LIBRARY_PATH`. torch preloads the CUDA libraries *it* needs, which is why nothing else had
noticed. Creating the symlink and exporting the directory made `import vllm.cumem_allocator`
succeed, and sleep mode with it.

Both steps are now in the notebook: the install cell adds `nvidia-cuda-nvrtc`, and the verify cell
creates the symlink, sets `LD_LIBRARY_PATH`, and tests the import in a subprocess before any GPU
time is spent.

**First complete GRPO run: 5/5 steps, adapter saved, no OOM.** The working shape on a 15 GiB T4 is
`--use-vllm --vllm-sleep --per-device-batch-size 2 --gradient-accumulation-steps 4
--num-generations 8 --max-completion-length 768`.

**Measured throughput: ~85 s/step**, and it tracks completion length closely (38 s at a 190-token
mean, 125 s at 476). Two consequences: 200 steps is ~4.7 h per run, so gate plus a six-point lambda
sweep is ~33 h; and sleep mode reloads the model from disk on every wake, visible as a
`Loading safetensors checkpoint shards` line per step, costing 4–12 s each.

**Finding: lambda = 0 is the worst case for gradient signal, not the neutral one.**
`frac_reward_zero_std` was 1.0 on three of five steps *even at group size 8* — steps 1, 4 and 5 had
all eight rollouts correct, reward 1.2 across the board, `grad_norm 0`. With only correctness and
format, the reward has very few distinct levels, and a model already at 87% accuracy saturates
groups constantly.

But look at the same steps' `metric_mean_think_tokens/std`: 57.4, 45.1, 11.1. The rollouts differ
in reasoning length even when they agree on correctness. **So any lambda > 0 gives those groups
non-zero variance and a usable gradient.** The go/no-go gate as specified in the proposal —
correctness-only — is therefore the configuration least likely to learn, which inverts how it
should be read: weak movement at lambda = 0 is expected and is not evidence that GRPO cannot work
here.

Also logged for later: `entropy` sits at 0.05–0.13 (low rollout diversity), and
`sampling/importance_sampling_ratio` ranges 0.2–2.2 with a min of 0 on one step — the vLLM-versus-
training-model numerics gap that TRL corrects for, worth watching if training behaves oddly.

**Lesson.** A dependency can be installed, present on disk, and still unusable — packaging, soname
and loader path are three separate things. And the scientific lesson is bigger than the
engineering one: instrumenting `frac_reward_zero_std` turned "the gate did not move much" from a
result into an artefact that was predicted in advance. That metric existed only because collapse
was named as the primary failure mode before any training ran.

---

## 21 — A 47% bigger GPU produced *less* free memory

**Phase D · 2026-08-12**

**Symptom.** Moved from a 15 GiB T4 to a 22 GiB L4 (Colab Pro) and immediately OOM'd in the
backward pass, on a configuration that had been running fine on the smaller card.

```
T4:  VRAM before training:  7.65 GiB free of 14.56  ( 6.91 GiB held)
L4:  VRAM before training:  7.86 GiB free of 22.03  (14.17 GiB held)
```

A 7.5 GiB larger card yielded **0.2 GiB more free memory**.

**Root cause.** Two compounding mistakes, both mine.

1. **`vllm_gpu_memory_utilization` is a fraction of *total*, not an absolute budget.** Keeping it at
   the T4-tuned 0.35 handed vLLM `0.35 x 22.03 = 7.7 GiB` instead of 5.1 — the extra capacity was
   silently consumed by the very component that had been squeezed.
2. **I raised the forward batch from 2 to 8 at the same time**, on the reasoning that a bigger card
   could afford it. That took the logits tensor from ~0.43 GiB back to ~1.74, doubled across the
   policy and reference passes.

Together they consumed the entire upgrade and then some.

**Fix.** On compute capability >= 8: forward batch 4 (logits ~0.87 GiB), and
`--vllm-gpu-memory-utilization 0.25`, which is 5.5 GiB — enough for vLLM's own 3.4 GiB of weights
plus ~2.1 GiB of KV cache, ample for 8 sequences at `max_model_len 2048`. The GRPO group stays at 8
on both hardware paths via gradient accumulation, so the algorithm is identical and only the memory
behaviour differs.

**Lesson.** Fractional resource settings do not port across hardware — they *look* like they scale
correctly and in fact re-tune themselves in the wrong direction. Worse, I changed two variables at
once while moving to new hardware, so the first OOM carried no information about which one was
responsible; only the `VRAM before training` line added in entry 18 made the split diagnosable
without another run.

Bigger hardware is not automatically more headroom. It is more headroom only if every consumer is
expressed in absolute terms or re-tuned deliberately.

---

## 22 — The fix was correct and did not apply: a stale variable in the kernel

**Phase D · 2026-08-12**

**Symptom.** After pushing the entry-21 fix and re-opening the notebook, the run printed

```
forward batch 8, vllm=on, sleep=off
peak logits tensor ~1.74 GiB per forward
```

— the *old* configuration — and OOM'd again in the same place. The corrected values were in the
notebook on disk and in git.

**Root cause.** The hardware settings were computed in one notebook cell and passed to the training
script as a `$FLAGS` string. Re-opening the notebook from GitHub replaced the **cells**, but
`FLAGS` was already bound in the running kernel from the previous version, and the training cell
consumed that binding. Editing the definition of a variable does not rebind it in a live session.

This is entry 17 in a different costume. There it was `sys.modules` holding an old package; here it
is a plain Python name. Both are the same failure: **a long-lived interpreter serving state from
before the fix, in a way that is indistinguishable from the fix not working.**

**Fix.** Deleted the mechanism rather than the symptom. `train_grpo.py` now resolves
`per_device_batch_size`, `gradient_accumulation_steps`, `vllm_gpu_memory_utilization` and sleep
mode itself, from `torch.cuda.get_device_capability()` in the process that will use them, and
prints what it chose at startup. All four remain overridable; only unset ones are filled in.

The notebook passes no hardware flags at all now, so there is no second place for the truth to live
and nothing left to go stale.

**Lesson.** When configuration has to match the environment, compute it *in the process that runs
in that environment*. Passing it across a boundary — notebook cell to subprocess, kernel to script
— creates a copy, and copies go stale silently. The general form: prefer deriving state over
propagating it, especially anywhere a long-lived session is involved.

Three separate failures now (17, 21, 22) have come from configuration and environment disagreeing
across a process or session boundary. Every one of them cost a full GPU cycle to diagnose, and
every one was invisible in the error message.

**Addendum, Phase E setup.** It happened a fourth time, in the third distinct flavour: the notebook
was re-opened from GitHub (updating the *cells*) but section 2's `git pull` cell was not re-run, so
`/content/rs_aidams` stayed on the previous commit. The evaluation loop then emitted five identical
`unrecognized arguments: --split-manifest` errors and copied nothing — which reads like a Drive
failure rather than a stale checkout.

So the three update paths in this setup are genuinely independent, and that is worth stating
plainly because it keeps catching us:

| what | updates when |
|---|---|
| notebook cells | the notebook is re-opened from GitHub |
| repo code | section 2's `git pull` cell is re-run |
| loaded modules and variables | the session is restarted |

The eval cell now probes `run_vllm --help` for the flag it is about to use and fails with an
instruction naming the exact cell to re-run. General principle, and the one to carry forward:
**where several things must be in sync and only some of them update automatically, make the
consumer assert what it needs rather than assume it.** A cheap up-front check converts a confusing
downstream symptom into a one-line diagnosis.

---

## 23 — Making the prompt conversational silently changed the completion type

**Phase D · 2026-08-13**

**Symptom.** The paired-rollout sweep died, and the visible error was a red herring:

```
FileNotFoundError: 'runs/paired_lam0_05/log_history.json'
```

That file is only written *after* `trainer.train()` returns, so its absence means training failed;
the real error was in the `!python` output above it.

**Root cause.** Paired rollouts need the prompt kept as a message list rather than a pre-rendered
string, because a rendered string cannot be re-templated per thinking mode. But TRL branches on
`is_conversational({"prompt": prompts[0]})` when it decodes completions: with a string prompt it
passes the completion through as a string, and with a message-list prompt it wraps it as
`[{"role": "assistant", "content": ...}]`.

The reward functions call `parse_model_output(completion)`, which runs a regex. A regex over a
`list` raises `TypeError`, so training died on the first reward computation.

**Fix.** A `completion_text()` normaliser at the TRL boundary in `rewards/reward.py`, handling
string, message-list and single-dict forms. `compute_reward` and `score_completion` keep taking
raw text, so the direct-call path and the standard training path are untouched. Two tests pin it,
including one asserting that conversational and plain completions produce *identical* rewards.

**Lesson.** Changing the type of one field changed the type of a different field, several layers
away, through a branch inside the library. Nothing in the paired-rollout change mentioned
completions at all. Worth noting how lucky the crash was: had `parse_model_output` been tolerant of
lists it would have found no tool call, scored every rollout zero, and produced a training run that
learned from nothing while looking healthy. **A strict parser converted a silent wrong-answers bug
into a loud one** — the same property that made the `format_ok`/`has_calls` split (entry 3) worth
keeping.

Also worth recording as a debugging habit: the reported exception was three layers downstream of
the fault. `FileNotFoundError` on an output artefact almost always means "the thing that writes it
failed", and the useful error is upstream.
