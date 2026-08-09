"""Per-sample BFCL scoring.

This module is the foundation of the whole project: the GRPO reward function
calls into it once per rollout, so it must score a *single* model output in
isolation and return a number.

Two things happen here:

1. `parse_model_output` decodes our output contract
   (`<think>...</think><tool_call>{...}</tool_call>`) into the structure BFCL's
   AST checker expects. The checker wants `{"func_name": {arg: value}}`, which
   is NOT the `{"name": ..., "arguments": {...}}` shape the model emits, so a
   shape conversion is required.

2. `score_sample` calls BFCL's official `ast_checker` and reduces its verdict
   dict to a float.

Using the official checker (rather than a reimplementation) means the numbers
reported in the paper are directly comparable to the leaderboard.

BFCL version pinned: bfcl-eval==2026.3.23 (BFCL v4 data).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import bfcl_eval
from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

DATA_DIR = os.path.join(os.path.dirname(bfcl_eval.__file__), "data")

# `ast_checker` threads a `model_name` through purely to look up one boolean:
# `underscore_to_dot`. That flag rewrites "." to "_" in function names for
# providers (OpenAI, Mistral, Google) whose APIs reject dots. A locally trained
# Qwen model has no such restriction, so we pin to a model config where the flag
# is False, i.e. function names are compared verbatim.
CHECKER_MODEL_NAME = "gorilla-openfunctions-v2"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

_TOOL_CALL_OPEN = "<tool_call>"
_JSON_DECODER = json.JSONDecoder()


def _iter_tool_calls(text: str):
    """Yield `(parsed_object, error)` for each `<tool_call>` in the text.

    The closing `</tool_call>` is treated as **optional**, which is a deliberate
    decision rather than laziness.

    `<tool_call>` and `</tool_call>` are single added tokens in Qwen2.5's
    vocabulary (151657 / 151658) belonging to its native tool-calling format. In
    practice the model emits the opening token and the JSON, then goes straight
    to `<|im_end|>` without ever emitting the closing token. Requiring it would
    mean the baselines measure Qwen's tag-emission habits rather than the effect
    of reasoning on call correctness — confounding the exact comparison this
    project exists to make.

    So instead of matching a closing delimiter, the JSON object is consumed with
    `raw_decode`, which reads exactly one value and reports where it ended. This
    tolerates a missing closing tag and trailing text, while still failing on
    genuinely malformed JSON — so `format_ok` keeps its meaning for the reward.
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
            yield None, f"malformed JSON in tool_call: {exc}"
            position = json_start
            continue

        yield obj, None
        position = end


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


def parse_model_output(text: str) -> ParsedOutput:
    """Decode raw generated text into checker-ready calls.

    Deliberately lenient about *surrounding* prose but strict about the JSON
    inside `<tool_call>` blocks: a model that emits malformed JSON has failed,
    and the reward must be able to see that.
    """
    # An empty `<think></think>` is not reasoning. Qwen3's chat template emits
    # exactly that when `enable_thinking=False`, so counting it as a think block
    # would report the never-policy as reasoning on every single item.
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

        # Shape conversion: {"name": f, "arguments": {...}} -> {f: {...}}
        calls.append({name: args})

    # Deliberately NOT an error here: emitting zero calls is the correct
    # behaviour on irrelevance items. Whether "no call" is right or wrong is a
    # category-dependent judgment, so it belongs to `score_sample`, not the
    # parser. `format_ok` therefore means only "nothing malformed was emitted";
    # use `has_calls` to ask whether a call was produced at all.
    return ParsedOutput(think=think, calls=calls, format_ok=format_ok, errors=errors)


def score_sample(
    functions: list[dict],
    ground_truth: list[dict],
    parsed: ParsedOutput,
    test_category: str,
    language: Language = Language.PYTHON,
) -> dict:
    """Score one decoded output. Returns {"correct": float, "error_type": str}.

    Relevance-style categories are handled separately because they are not AST
    problems: for `irrelevance` the correct behaviour is to emit NO call at all,
    so there is nothing for the AST checker to compare against.
    """
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

    result = ast_checker(
        functions,
        parsed.calls,
        ground_truth,
        language,
        test_category,
        CHECKER_MODEL_NAME,
    )
    valid = bool(result["valid"])
    # NOTE: ast_checker leaves a stale `error_type` populated even when valid is
    # True, so never branch on error_type without checking valid first.
    return {
        "correct": float(valid),
        "error_type": "" if valid else result.get("error_type", "unknown"),
    }


def load_category(test_category: str) -> list[dict]:
    """Load a BFCL category, joining each question to its ground truth by id."""
    q_path = os.path.join(DATA_DIR, f"BFCL_v4_{test_category}.json")
    a_path = os.path.join(DATA_DIR, "possible_answer", f"BFCL_v4_{test_category}.json")

    def read_jsonl(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    questions = read_jsonl(q_path)

    # Irrelevance categories have no possible_answer file: there is no correct
    # call to compare against, only the absence of one.
    if not os.path.exists(a_path):
        return [{**q, "ground_truth": None} for q in questions]

    answers = {a["id"]: a["ground_truth"] for a in read_jsonl(a_path)}
    return [{**q, "ground_truth": answers.get(q["id"])} for q in questions]
