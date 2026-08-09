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
