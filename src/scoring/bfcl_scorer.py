"""Per-sample BFCL scoring.

This module is essentially the foundation of the whole project: the GRPO reward
function calls into it once per rollout, so it has to score a *single* model
output in isolation and return a number.

Pretty much two things happen here:

1. `parse_model_output` decodes our output contract
   (`<think>...</think><tool_call>{"name": f, "arguments": {...}}</tool_call>`)
   into the shape BFCL's AST checker expects, which is `{f: {...}}`. That shape
   conversion is the part that actually had to be written.

2. `score_sample` calls BFCL's official `ast_checker` and reduces its verdict
   dict to a float.

Using the official checker (rather than a reimplementation) means the numbers
reported in the paper are directly comparable to the leaderboard.

Reading order below: the two jobs first, then the data they pass between them,
then the loader, and finally the import plumbing that makes BFCL's checker
usable at all. Nothing below the "plumbing" banner is part of the idea.

BFCL version pinned: bfcl-eval==2026.3.23 (BFCL v4 data).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import bfcl_eval
from bfcl_eval.constants.enums import Language

# ===========================================================================
# JOB 1 — decode the model's output into checker-ready calls
# ===========================================================================

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

_TOOL_CALL_OPEN = "<tool_call>"
_JSON_DECODER = json.JSONDecoder()


def parse_model_output(text: str) -> ParsedOutput:
    """Decode raw generated text into checker-ready calls.

    Deliberately lenient about *surrounding* prose but strict about the JSON
    inside `<tool_call>` blocks: a model that emits malformed JSON has failed,
    and the reward must be able to see that.
    """
    # An empty `<think></think>` is not reasoning. Qwen3's chat template emits
    # exactly that when `enable_thinking=False`, so counting it would report the
    # never-policy as reasoning on every single item — and think-rate is the
    # primary collapse diagnostic for training.
    think_match = _THINK_RE.search(text)
    think = (think_match.group(1).strip() or None) if think_match else None

    calls: list[dict] = []
    errors: list[str] = []
    format_ok = True

    for obj, error in _iter_tool_calls(text):
        if error is not None:
            format_ok = False
            errors.append(error)
            continue

        if not isinstance(obj, dict):
            format_ok = False
            errors.append(f"tool_call payload is not an object: {obj!r}")
            continue

        name = obj.get("name")
        args = obj.get("arguments", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            format_ok = False
            errors.append(f"tool_call missing 'name' or 'arguments': {obj!r}")
            continue

        calls.append({name: args})  # {"name": f, "arguments": a} -> {f: a}

    # Zero calls is deliberately NOT flagged here: on irrelevance items it is the
    # correct answer. `format_ok` means only "nothing malformed was emitted";
    # ask `has_calls` whether a call was produced at all. Keeping the two apart
    # is what stops silence from earning the reward's format term later.
    return ParsedOutput(think=think, calls=calls, format_ok=format_ok, errors=errors)


def _iter_tool_calls(text: str):
    """Yield `(parsed_object, error)` for each `<tool_call>` in the text.

    The closing `</tool_call>` is **optional by design**. Qwen often emits the
    opening tag and the JSON, then stops — and it does so far more when not
    reasoning (134 unclosed calls under `never` against 15 under `always`).
    Requiring the tag would penalise the never-policy for a tag habit and
    inflate the exact reasoning gap this project exists to measure.

    So instead of matching a closing delimiter, `raw_decode` consumes exactly
    one JSON value and reports where it ended. Leniency stops there: malformed
    JSON still fails, so `format_ok` keeps its meaning for the reward.
    """
    position = 0
    while True:
        start = text.find(_TOOL_CALL_OPEN, position)
        if start == -1:
            return

        json_start = start + len(_TOOL_CALL_OPEN)
        while json_start < len(text) and text[json_start].isspace():
            json_start += 1

        try:
            obj, end = _JSON_DECODER.raw_decode(text, json_start)
        except json.JSONDecodeError as exc:
            # Advance past the opening tag rather than giving up, so one
            # malformed call cannot hide well-formed ones later in the output.
            yield None, f"malformed JSON in tool_call: {exc}"
            position = json_start
            continue

        yield obj, None
        position = end


# ===========================================================================
# JOB 2 — judge the decoded call
# ===========================================================================


def score_sample(
    functions: list[dict],
    ground_truth: list[dict],
    parsed: ParsedOutput,
    test_category: str,
    language: Language = Language.PYTHON,
) -> dict:
    """Score one decoded output. Returns {"correct": float, "error_type": str}.

    Note what is absent: `parsed.think` is never read. Correctness has to be
    blind to whether the model reasoned, otherwise the length penalty stops
    being the only pressure acting on the thinking decision.
    """
    # Relevance-style categories are not AST problems: there is no ground-truth
    # call to compare against, only the presence or absence of one.
    if "irrelevance" in test_category:
        correct = len(parsed.calls) == 0
        return {
            "correct": float(correct),
            "error_type": "" if correct else "irrelevance:should_not_have_called",
        }

    if "relevance" in test_category:
        correct = len(parsed.calls) > 0
        return {
            "correct": float(correct),
            "error_type": "" if correct else "relevance:should_have_called",
        }

    if not parsed.calls:
        return {"correct": 0.0, "error_type": "parse:no_valid_call"}

    ast_checker = _load_ast_checker()

    result = ast_checker(
        functions,
        parsed.calls,
        ground_truth,
        language,
        test_category,
        CHECKER_MODEL_NAME,
    )
    valid = bool(result["valid"])
    # ast_checker leaves a stale `error_type` populated even when valid is True,
    # so never branch on error_type without checking valid first.
    return {
        "correct": float(valid),
        "error_type": "" if valid else result.get("error_type", "unknown"),
    }


# ===========================================================================
# THE DATA CARRIER — what job 1 hands to job 2
# ===========================================================================


@dataclass
class ParsedOutput:
    """Result of decoding a raw model generation."""

    think: str | None  # reasoning text, or None if the model emitted no think block
    calls: list[dict]  # checker-shaped calls: [{"func_name": {arg: value}}]
    format_ok: bool  # did every tool_call block that WAS emitted parse cleanly?
    errors: list[str] = field(default_factory=list)

    @property
    def did_think(self) -> bool:
        return self.think is not None

    @property
    def has_calls(self) -> bool:
        return len(self.calls) > 0


# ===========================================================================
# LOADING THE BFCL DATA
# ===========================================================================

DATA_DIR = os.path.join(os.path.dirname(bfcl_eval.__file__), "data")


def load_category(test_category: str) -> list[dict]:
    """Load a BFCL category, joining each question to its ground truth by id."""
    q_path = os.path.join(DATA_DIR, f"BFCL_v4_{test_category}.json")
    a_path = os.path.join(DATA_DIR, "possible_answer", f"BFCL_v4_{test_category}.json")

    def read_jsonl(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    questions = read_jsonl(q_path)

    # Irrelevance categories ship no possible_answer file: there is no correct
    # call to compare against, only the absence of one.
    if not os.path.exists(a_path):
        return [{**q, "ground_truth": None} for q in questions]

    answers = {a["id"]: a["ground_truth"] for a in read_jsonl(a_path)}
    return [{**q, "ground_truth": answers.get(q["id"])} for q in questions]


# ===========================================================================
# PLUMBING — getting BFCL's checker to import at all
#
# None of this is part of the idea. It exists because importing the official
# checker drags in every API model handler to resolve a single boolean.
# ===========================================================================

# `ast_checker` threads `model_name` through to look up exactly one boolean:
# `underscore_to_dot`, which rewrites "." to "_" in function names for providers
# that reject dots. Pinned to a config where it is False, so names compare
# verbatim. Changing this pin changes how every function name is matched.
CHECKER_MODEL_NAME = "gorilla-openfunctions-v2"

_AST_CHECKER = None


def _load_ast_checker():
    """Import BFCL's checker, stubbing the handler chain only if it is absent.

    Imported lazily rather than at module load: `import bfcl_eval` is free, but
    importing the checker costs ~10s and 4322 modules. Deferring it lets
    generation run under `pip install --no-deps bfcl-eval`, which brings the
    data files and nothing else. Where the full install exists, nothing is
    stubbed and the upstream path is used verbatim.
    """
    global _AST_CHECKER
    if _AST_CHECKER is None:
        try:
            from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        except ImportError:
            _install_model_config_stub()
            from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        _AST_CHECKER = ast_checker
    return _AST_CHECKER


def _install_model_config_stub() -> None:
    """Supply `MODEL_CONFIG_MAPPING` directly instead of importing 81 packages.

    `ast_checker` imports it at module level and uses it in one place, to read
    `underscore_to_dot`. Resolving that boolean the real way pulls 4322 modules
    (anthropic, boto3, tree_sitter, ...) plus bfcl-eval's torch pin, which would
    fight the torch that trl/peft are built against.

    The stub serves only `CHECKER_MODEL_NAME` and raises `KeyError` for anything
    else: guessing False for another provider would silently compare names the
    wrong way, which is a wrong-answers bug rather than a crash.

    See engineering log entries 1, 7 and 14.
    """
    import sys
    import types

    class _StubConfig:
        underscore_to_dot = False

    class _StubMapping(dict):
        def __getitem__(self, key):
            if key != CHECKER_MODEL_NAME:
                raise KeyError(
                    f"model_config stub serves only {CHECKER_MODEL_NAME!r}, got {key!r}. "
                    "Install bfcl-eval with its full dependencies to use another model."
                )
            return _StubConfig

    module = types.ModuleType("bfcl_eval.constants.model_config")
    module.MODEL_CONFIG_MAPPING = _StubMapping()
    sys.modules["bfcl_eval.constants.model_config"] = module
