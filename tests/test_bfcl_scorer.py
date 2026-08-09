"""Verification of the per-sample BFCL scorer (Phase A exit condition).

These tests run against the real BFCL v4 data shipped in `bfcl-eval`, not
fixtures, so they also detect the scorer silently breaking if the pinned BFCL
version is ever bumped.

The scorer is the reward function. A scorer that quietly returns 1.0 for wrong
calls would make every downstream training run meaningless while still
producing plausible-looking curves, so each category is checked for both
directions: correct output scores 1.0, and each distinct failure mode scores 0.0.
"""

from __future__ import annotations

import json

import pytest

from scoring.bfcl_scorer import load_category, parse_model_output, score_sample


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def pick_value(acceptable: list, is_required: bool):
    """Choose a concrete argument value from BFCL's acceptable-values list.

    Returns the sentinel `None` to mean "omit this parameter entirely".

    Three rules, each one established by running this helper across the full
    dataset and reading the resulting checker errors:

    1. `""` among the acceptable values usually marks the parameter optional,
       and optional parameters are best omitted: BFCL ground truth sometimes
       lists optional parameters that do not appear in the function schema at
       all, and supplying those raises `unexpected_param`.

    2. `""` is NOT a reliable optionality marker on its own — it also appears on
       parameters that the schema lists as required (e.g. `[true, ""]` for a
       required `formatted` flag). The schema's `required` list is the
       authority, so a required parameter is always supplied.

    3. Acceptable-value lists nest recursively through both dicts *and* lists,
       so unwrapping must recurse through both.
    """
    if not is_required and any(value == "" for value in acceptable):
        return None
    # A required parameter may still list "" first; take the first real value.
    for value in acceptable:
        if value != "":
            return concretize(value)
    return None


def concretize(value):
    """Recursively collapse nested acceptable-value lists into a concrete value."""
    if isinstance(value, dict):
        result = {}
        for key, inner in value.items():
            # Inside a dict, every value is itself an acceptable-values list.
            # Nested parameters carry no schema, so treat them as required and
            # let the "" fallback in pick_value decide.
            picked = pick_value(inner, is_required=True) if isinstance(inner, list) else concretize(inner)
            if picked is not None:
                result[key] = picked
        return result
    if isinstance(value, list):
        return [concretize(item) for item in value]
    return value


def build_call(gt_entry: dict, functions: list[dict]) -> dict:
    """Turn one ground-truth entry into a valid `{"name", "arguments"}` call.

    Needs the function schemas, not just the ground truth, because optionality
    can only be determined from the schema's `required` list (see rule 2 above).
    """
    (func_name, params), = gt_entry.items()
    schema = next((f for f in functions if f["name"] == func_name), None)
    required = set(schema["parameters"].get("required", [])) if schema else set()

    arguments = {}
    for param, acceptable in params.items():
        value = pick_value(acceptable, is_required=param in required)
        if value is not None:
            arguments[param] = value
    return {"name": func_name, "arguments": arguments}


def render(calls: list[dict], think: str | None = None) -> str:
    """Render calls into our output contract, as the model would emit it."""
    prefix = f"<think>{think}</think>" if think is not None else ""
    body = "".join(
        f"<tool_call>{json.dumps(call)}</tool_call>" for call in calls
    )
    return prefix + body


def score_text(sample: dict, text: str, category: str) -> dict:
    parsed = parse_model_output(text)
    return score_sample(sample["function"], sample["ground_truth"], parsed, category)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def test_parser_separates_think_from_call():
    parsed = parse_model_output(
        '<think>reasoning here</think><tool_call>{"name":"f","arguments":{"a":1}}</tool_call>'
    )
    assert parsed.did_think
    assert parsed.think == "reasoning here"
    assert parsed.calls == [{"f": {"a": 1}}]
    assert parsed.format_ok


def test_parser_handles_absent_think_block():
    parsed = parse_model_output('<tool_call>{"name":"f","arguments":{}}</tool_call>')
    assert not parsed.did_think
    assert parsed.has_calls


def test_malformed_json_flags_format_not_silently_dropped():
    parsed = parse_model_output('<tool_call>{"name":"f", "arguments":{"a":1,}}</tool_call>')
    assert not parsed.format_ok
    assert not parsed.has_calls
    assert parsed.errors


def test_closing_tag_is_optional():
    """Qwen2.5 emits `<tool_call>` + JSON then `<|im_end|>`, never the closing tag.

    Requiring `</tool_call>` would make the baselines measure tag-emission
    habits rather than the effect of reasoning, so it is optional by design.
    """
    parsed = parse_model_output('<tool_call>{"name":"f","arguments":{"a":1}}')
    assert parsed.calls == [{"f": {"a": 1}}]
    assert parsed.format_ok


def test_closing_tag_still_accepted_when_present():
    parsed = parse_model_output('<tool_call>{"name":"f","arguments":{"a":1}}</tool_call>')
    assert parsed.calls == [{"f": {"a": 1}}]


def test_multiple_unclosed_calls_all_parse():
    parsed = parse_model_output(
        '<tool_call>{"name":"f","arguments":{"a":1}}\n<tool_call>{"name":"g","arguments":{"b":2}}'
    )
    assert parsed.calls == [{"f": {"a": 1}}, {"g": {"b": 2}}]


def test_leniency_does_not_extend_to_malformed_json():
    """Optional closing tag must not weaken the format signal the reward needs."""
    parsed = parse_model_output('<tool_call>{"name":"f", "arguments":{"a":1,}}')
    assert not parsed.format_ok
    assert not parsed.has_calls


def test_no_call_is_not_a_format_error():
    """Zero calls is correct on irrelevance items, so it must not read as malformed."""
    parsed = parse_model_output("I cannot help with that request.")
    assert parsed.format_ok
    assert not parsed.has_calls


# --------------------------------------------------------------------------
# simple
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_sample():
    return load_category("simple_python")[0]


def test_simple_correct_scores_one(simple_sample):
    call = build_call(simple_sample["ground_truth"][0], simple_sample["function"])
    assert score_text(simple_sample, render([call]), "simple")["correct"] == 1.0


def test_simple_reasoning_does_not_change_correctness(simple_sample):
    """Correctness must be independent of whether the model reasoned.

    This is the invariant the whole project rests on: if the presence of a
    <think> block leaked into the correctness term, the length penalty would no
    longer be the only pressure acting on the thinking decision.
    """
    call = build_call(simple_sample["ground_truth"][0], simple_sample["function"])
    with_think = score_text(simple_sample, render([call], think="some reasoning"), "simple")
    without = score_text(simple_sample, render([call]), "simple")
    assert with_think["correct"] == without["correct"] == 1.0


def test_simple_wrong_function_name_scores_zero(simple_sample):
    call = build_call(simple_sample["ground_truth"][0], simple_sample["function"])
    call["name"] = "definitely_not_the_right_function"
    assert score_text(simple_sample, render([call]), "simple")["correct"] == 0.0


def test_simple_missing_required_param_scores_zero(simple_sample):
    call = build_call(simple_sample["ground_truth"][0], simple_sample["function"])
    call["arguments"].pop(next(iter(call["arguments"])))
    assert score_text(simple_sample, render([call]), "simple")["correct"] == 0.0


def test_simple_prose_only_scores_zero(simple_sample):
    assert score_text(simple_sample, "The area is 25 square units.", "simple")["correct"] == 0.0


# --------------------------------------------------------------------------
# multiple — right function must be chosen from several candidates
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multiple_sample():
    return load_category("multiple")[0]


def test_multiple_correct_scores_one(multiple_sample):
    call = build_call(multiple_sample["ground_truth"][0], multiple_sample["function"])
    assert score_text(multiple_sample, render([call]), "multiple")["correct"] == 1.0


def test_multiple_dotted_function_name_preserved(multiple_sample):
    """Function names contain dots (e.g. `triangle_properties.get`).

    `CHECKER_MODEL_NAME` pins `underscore_to_dot=False` so names compare
    verbatim. If that pin were wrong, dots would be rewritten to underscores and
    every call in this category would score zero.
    """
    call = build_call(multiple_sample["ground_truth"][0], multiple_sample["function"])
    assert "." in call["name"]
    assert score_text(multiple_sample, render([call]), "multiple")["correct"] == 1.0


def test_multiple_choosing_wrong_candidate_scores_zero(multiple_sample):
    correct_name = next(iter(multiple_sample["ground_truth"][0]))
    others = [f["name"] for f in multiple_sample["function"] if f["name"] != correct_name]
    assert others, "fixture expected more than one candidate function"

    call = build_call(multiple_sample["ground_truth"][0], multiple_sample["function"])
    call["name"] = others[0]
    assert score_text(multiple_sample, render([call]), "multiple")["correct"] == 0.0


# --------------------------------------------------------------------------
# parallel — several calls, order should not matter
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parallel_sample():
    return load_category("parallel")[0]


def test_parallel_correct_scores_one(parallel_sample):
    calls = [build_call(gt, parallel_sample["function"]) for gt in parallel_sample["ground_truth"]]
    assert len(calls) > 1, "fixture expected multiple calls"
    assert score_text(parallel_sample, render(calls), "parallel")["correct"] == 1.0


def test_parallel_is_order_independent(parallel_sample):
    """Routed to `parallel_function_checker_no_order`, so reversal must still pass.

    Verified rather than assumed: the model has no reason to emit calls in
    ground-truth order, and if order mattered the reward would be silently noisy.
    """
    calls = [build_call(gt, parallel_sample["function"]) for gt in parallel_sample["ground_truth"]]
    assert score_text(parallel_sample, render(list(reversed(calls))), "parallel")["correct"] == 1.0


def test_parallel_dropping_a_call_scores_zero(parallel_sample):
    calls = [build_call(gt, parallel_sample["function"]) for gt in parallel_sample["ground_truth"]]
    assert score_text(parallel_sample, render(calls[:1]), "parallel")["correct"] == 0.0


# --------------------------------------------------------------------------
# parallel_multiple — several calls to different functions
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parallel_multiple_sample():
    return load_category("parallel_multiple")[0]


def test_parallel_multiple_correct_scores_one(parallel_multiple_sample):
    calls = [build_call(gt, parallel_multiple_sample["function"]) for gt in parallel_multiple_sample["ground_truth"]]
    assert score_text(parallel_multiple_sample, render(calls), "parallel_multiple")["correct"] == 1.0


def test_parallel_multiple_is_order_independent(parallel_multiple_sample):
    calls = [build_call(gt, parallel_multiple_sample["function"]) for gt in parallel_multiple_sample["ground_truth"]]
    reversed_score = score_text(
        parallel_multiple_sample, render(list(reversed(calls))), "parallel_multiple"
    )
    assert reversed_score["correct"] == 1.0


def test_parallel_multiple_dropping_a_call_scores_zero(parallel_multiple_sample):
    calls = [build_call(gt, parallel_multiple_sample["function"]) for gt in parallel_multiple_sample["ground_truth"]]
    assert score_text(parallel_multiple_sample, render(calls[:1]), "parallel_multiple")["correct"] == 0.0


# --------------------------------------------------------------------------
# irrelevance — correct behaviour is to emit no call at all
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def irrelevance_sample():
    return load_category("irrelevance")[0]


def test_irrelevance_abstaining_scores_one(irrelevance_sample):
    result = score_text(irrelevance_sample, "None of the available tools apply here.", "irrelevance")
    assert result["correct"] == 1.0


def test_irrelevance_calling_anyway_scores_zero(irrelevance_sample):
    text = render([{"name": "some_function", "arguments": {}}])
    assert score_text(irrelevance_sample, text, "irrelevance")["correct"] == 0.0


def test_irrelevance_has_no_ground_truth(irrelevance_sample):
    """No possible_answer file exists for irrelevance: there is no correct call."""
    assert irrelevance_sample["ground_truth"] is None


# --------------------------------------------------------------------------
# full-dataset sweep — the actual Phase A exit criterion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category", ["simple_python", "multiple", "parallel", "parallel_multiple"]
)
def test_ground_truth_derived_predictions_score_perfectly(category):
    """Score every item using a prediction derived from its own ground truth.

    Hand-written cases only exercise the paths someone thought to test. This
    sweep is the real check: a prediction reconstructed from an item's own
    ground truth must score 1.0, on every item, or either the scorer or the
    reconstruction is wrong.

    Getting this from 98.4% to 100% surfaced three encoding rules in the BFCL
    ground-truth format that were not obvious from reading a single example:
    optional-parameter omission, `""` appearing on *required* parameters, and
    acceptable-value lists nesting through lists as well as dicts.

    It also guards the pinned dependency: if `bfcl-eval` is ever bumped and the
    ground-truth encoding shifts, this fails loudly instead of silently
    degrading every reward the training loop computes.
    """
    data = load_category(category)
    failures = []

    for sample in data:
        calls = [build_call(gt, sample["function"]) for gt in sample["ground_truth"]]
        parsed = parse_model_output(render(calls))
        result = score_sample(sample["function"], sample["ground_truth"], parsed, category)
        if result["correct"] != 1.0:
            failures.append((sample["id"], result["error_type"]))

    assert not failures, (
        f"{len(failures)}/{len(data)} items failed in {category}: {failures[:5]}"
    )
