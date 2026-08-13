"""Verification of the GRPO reward function (Phase C exit condition).

The reward is what the model optimises, so a defect here does not produce an
error — it produces a trained policy that confidently learned the wrong thing.
The tests are therefore organised around the properties the training run
depends on, not around line coverage:

1. Correctness dominates. No amount of length saving can make a wrong call
   outrank a right one, and no format bonus can either. If this fails, the
   cheapest policy wins and the run collapses to never-think regardless of
   accuracy.
2. Thinking is priced, and priced only by think tokens. Two outputs with the
   same call and different reasoning lengths must be separated, and the length
   of the *call* must not affect the price.
3. The penalty is bounded and monotone, so `lambda_think` means the same thing
   across runs — the tradeoff curve is parameterised by it.
4. Abstention on irrelevance is rewarded, and rewarded more than a call. This is
   the category most easily destroyed by a reward that quietly assumes a call is
   always wanted.

Real BFCL items are used rather than fixtures, for the same reason as in the
scorer suite: it catches the pinned BFCL version shifting underneath us.
"""

from __future__ import annotations

import pytest

from rewards.reward import (
    RewardConfig,
    compute_reward,
    make_reward_fn,
    score_completion,
    think_cost,
)
from scoring.bfcl_scorer import load_category, parse_model_output


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def word_tokens(text: str) -> int:
    """Stand-in tokenizer: whitespace count.

    The reward takes the token counter as an argument precisely so the tests do
    not need to download a tokenizer, and so a change in Qwen3's vocabulary
    cannot break assertions that are about reward arithmetic.
    """
    return len(text.split())


@pytest.fixture(scope="module")
def simple_item() -> dict:
    """A real `simple_python` item, with a known-correct call for it."""
    sample = load_category("simple_python")[0]
    (func_name, params), = sample["ground_truth"][0].items()
    args = {key: values[0] for key, values in params.items() if values[0] != ""}
    return {"sample": sample, "func_name": func_name, "args": args}


def call_text(func_name: str, args: dict) -> str:
    import json

    return f'<tool_call>{json.dumps({"name": func_name, "arguments": args})}</tool_call>'


def reward_for(item: dict, completion: str, config: RewardConfig | None = None) -> dict:
    return score_completion(
        completion,
        item["sample"],
        "simple_python",
        word_tokens,
        config,
    )


# --------------------------------------------------------------------------
# 1. correctness dominates
# --------------------------------------------------------------------------


def test_correct_call_beats_wrong_call(simple_item):
    correct = reward_for(simple_item, call_text(simple_item["func_name"], simple_item["args"]))
    wrong = reward_for(simple_item, call_text("no_such_function", simple_item["args"]))

    assert correct["correct"] == 1.0
    assert wrong["correct"] == 0.0
    assert correct["reward"] > wrong["reward"]


def test_correctness_outweighs_maximum_think_saving(simple_item):
    """A right answer that thought at full budget must beat a wrong instant one.

    This is the constraint that keeps `lambda_think` in a usable range: once the
    price of thinking exceeds the value of being right, the optimal policy is to
    answer wrongly and cheaply, and the experiment measures nothing.
    """
    config = RewardConfig(lambda_think=0.5, think_token_budget=10)
    thought = " ".join(["reasoning"] * 50)
    correct_slow = reward_for(
        simple_item,
        f"<think>{thought}</think>" + call_text(simple_item["func_name"], simple_item["args"]),
        config,
    )
    wrong_fast = reward_for(simple_item, call_text("no_such_function", {}), config)

    assert correct_slow["think_cost"] == pytest.approx(config.lambda_think)  # clipped at budget
    assert correct_slow["reward"] > wrong_fast["reward"]


def test_format_bonus_cannot_outrank_correctness(simple_item):
    """Well-formed prose must not beat a correct call.

    `format_ok` is True for output containing no tool call at all, so the format
    weight is the one term that could make silence profitable.
    """
    silent = reward_for(simple_item, "I am afraid I cannot help with that.")
    correct = reward_for(simple_item, call_text(simple_item["func_name"], simple_item["args"]))

    assert silent["format_ok"] == 1.0
    assert silent["correct"] == 0.0
    assert correct["reward"] > silent["reward"]


def test_malformed_json_loses_the_format_term(simple_item):
    broken = reward_for(simple_item, '<tool_call>{"name": "f", "arguments": {oops}}</tool_call>')

    assert broken["format_ok"] == 0.0
    assert broken["correct"] == 0.0
    assert broken["reward"] < 0.0 or broken["reward"] == 0.0


# --------------------------------------------------------------------------
# 2. thinking is priced, and only think tokens are priced
# --------------------------------------------------------------------------


def test_thinking_costs_when_it_does_not_help(simple_item):
    """Same correct call, one with reasoning: the reasoned one must score lower.

    This is the pressure the entire project depends on. Without it the model has
    no reason to ever skip thinking and the adaptive policy cannot form.
    """
    call = call_text(simple_item["func_name"], simple_item["args"])
    direct = reward_for(simple_item, call)
    reasoned = reward_for(simple_item, "<think>let me consider this carefully</think>" + call)

    assert direct["correct"] == reasoned["correct"] == 1.0
    assert reasoned["did_think"] == 1.0 and direct["did_think"] == 0.0
    assert reasoned["reward"] < direct["reward"]


def test_longer_reasoning_costs_more(simple_item):
    call = call_text(simple_item["func_name"], simple_item["args"])
    short = reward_for(simple_item, "<think>one two</think>" + call)
    long = reward_for(simple_item, "<think>" + " ".join(["word"] * 40) + "</think>" + call)

    assert long["think_tokens"] > short["think_tokens"]
    assert long["reward"] < short["reward"]


def test_call_length_is_not_priced(simple_item):
    """A longer *call* must not be cheaper or dearer than a short one.

    Penalising completion length instead of think length would systematically
    disadvantage the parallel categories, which need several call blocks by
    construction — turning a category effect into an apparent reasoning effect.
    """
    call = call_text(simple_item["func_name"], simple_item["args"])
    one = reward_for(simple_item, call)
    padded = reward_for(simple_item, call + call + call)

    assert one["think_cost"] == padded["think_cost"] == 0.0


def test_empty_think_block_is_not_charged(simple_item):
    """Qwen3 self-gates by emitting `<think></think>`; that must cost nothing.

    Observed on 18% of `simple_python` items under the always policy. An empty
    block is the model declining to reason, so charging it would price a
    decision the model did not make.
    """
    call = call_text(simple_item["func_name"], simple_item["args"])
    gated = reward_for(simple_item, "<think>\n</think>\n\n" + call)
    direct = reward_for(simple_item, call)

    assert gated["did_think"] == 0.0
    assert gated["think_cost"] == 0.0
    assert gated["reward"] == direct["reward"]


# --------------------------------------------------------------------------
# 3. the penalty is bounded and monotone in lambda
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [0, 1, 100, 768, 5000])
def test_think_cost_stays_within_lambda(tokens):
    config = RewardConfig(lambda_think=0.3, think_token_budget=768)
    cost = think_cost(tokens, config)
    assert 0.0 <= cost <= config.lambda_think


def test_think_cost_is_monotone():
    config = RewardConfig()
    costs = [think_cost(n, config) for n in (0, 10, 100, 500, 768, 1000)]
    assert costs == sorted(costs)


def test_lambda_zero_makes_thinking_free(simple_item):
    """The lambda=0 end of the sweep must reproduce the unpenalised reward.

    It is the control point of the tradeoff curve: if thinking still costs
    something at lambda=0, every point on the curve is offset.
    """
    config = RewardConfig(lambda_think=0.0)
    call = call_text(simple_item["func_name"], simple_item["args"])
    direct = reward_for(simple_item, call, config)
    reasoned = reward_for(simple_item, "<think>" + " ".join(["w"] * 200) + "</think>" + call, config)

    assert reasoned["reward"] == direct["reward"]


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        RewardConfig(think_token_budget=0)
    with pytest.raises(ValueError):
        RewardConfig(lambda_think=-0.1)


# --------------------------------------------------------------------------
# 4. abstention on irrelevance
# --------------------------------------------------------------------------


def test_abstaining_on_irrelevance_beats_calling():
    sample = load_category("irrelevance")[0]
    abstain = score_completion(
        "No suitable function is available.", sample, "irrelevance", word_tokens
    )
    called = score_completion(
        call_text("some_function", {"x": 1}), sample, "irrelevance", word_tokens
    )

    assert abstain["correct"] == 1.0
    assert called["correct"] == 0.0
    assert abstain["reward"] > called["reward"]


def test_thinking_still_costs_on_a_correct_abstention():
    """Reasoning its way to a correct abstention is still charged for.

    Otherwise irrelevance becomes a free place to think, and the learned gate
    would be shaped by that loophole rather than by difficulty.
    """
    sample = load_category("irrelevance")[0]
    direct = score_completion("No suitable function.", sample, "irrelevance", word_tokens)
    reasoned = score_completion(
        "<think>none of these fit the request</think>No suitable function.",
        sample,
        "irrelevance",
        word_tokens,
    )

    assert direct["correct"] == reasoned["correct"] == 1.0
    assert reasoned["reward"] < direct["reward"]


# --------------------------------------------------------------------------
# 5. TRL adapter
# --------------------------------------------------------------------------


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_trl_reward_fn_returns_aligned_floats(simple_item):
    """TRL calls `fn(prompts, completions, **columns)` and wants a list of floats.

    Checked explicitly because a shape mismatch here surfaces as an opaque
    trainer error many minutes into a GPU run.
    """
    sample = simple_item["sample"]
    call = call_text(simple_item["func_name"], simple_item["args"])
    completions = [call, "<think>hmm let me think</think>" + call, "garbage"]

    reward_fn = make_reward_fn(FakeTokenizer())
    rewards = reward_fn(
        prompts=["p"] * 3,
        completions=completions,
        function=[sample["function"]] * 3,
        ground_truth=[sample["ground_truth"]] * 3,
        category=["simple_python"] * 3,
    )

    assert len(rewards) == 3
    assert all(isinstance(r, float) for r in rewards)
    # direct-correct > reasoned-correct > wrong
    assert rewards[0] > rewards[1] > rewards[2]


def test_metric_fns_are_logging_only_and_report_the_collapse_signals(simple_item):
    """Think-rate must be observable per step, at weight 0.0.

    Collapse to all-think or all-no-think is the primary scientific failure mode
    and it is invisible in the reward curve — a policy that stopped reasoning and
    one that started emitting malformed calls both flatten it. These functions
    carry TRL's reward signature so they can be registered alongside the real
    reward and logged without touching the gradient.
    """
    from rewards.reward import make_metric_fns

    sample = simple_item["sample"]
    call = call_text(simple_item["func_name"], simple_item["args"])
    completions = [call, "<think>hmm let me think</think>" + call, "garbage"]
    columns = {
        "function": [sample["function"]] * 3,
        "ground_truth": [sample["ground_truth"]] * 3,
        "category": ["simple_python"] * 3,
    }

    metrics = {fn.__name__: fn(None, completions, **columns) for fn in make_metric_fns(FakeTokenizer())}

    assert metrics["metric_think_rate"] == [0.0, 1.0, 0.0]
    assert metrics["metric_correctness"] == [1.0, 1.0, 0.0]
    # Prose parses cleanly — it is simply not a call. That distinction is the
    # format_ok / has_calls split established in Phase A.
    assert metrics["metric_format_rate"] == [1.0, 1.0, 1.0]
    assert metrics["metric_mean_think_tokens"] == [0.0, 4.0, 0.0]


def test_trl_reward_fn_accepts_json_encoded_columns(simple_item):
    """Columns arrive JSON-encoded from `train.dataset`; both forms must work.

    Arrow cannot infer a schema for the nested `function` / `ground_truth`
    structures, so the dataset carries them as strings. If the reward silently
    mishandled that, every training reward would be computed against a string
    instead of a schema — scoring zero everywhere, with no error raised.
    """
    import json

    from rewards.reward import make_metric_fns

    sample = simple_item["sample"]
    call = call_text(simple_item["func_name"], simple_item["args"])
    completions = [call, "<think>hmm</think>" + call]

    decoded = {
        "function": [sample["function"]] * 2,
        "ground_truth": [sample["ground_truth"]] * 2,
        "category": ["simple_python"] * 2,
    }
    encoded = {
        "function": [json.dumps(sample["function"])] * 2,
        "ground_truth": [json.dumps(sample["ground_truth"])] * 2,
        "category": ["simple_python"] * 2,
    }

    reward_fn = make_reward_fn(FakeTokenizer())
    assert reward_fn(None, completions, **encoded) == reward_fn(None, completions, **decoded)

    # And the correctness metric, which scores independently of the reward.
    correctness = next(f for f in make_metric_fns(FakeTokenizer()) if "correctness" in f.__name__)
    assert correctness(None, completions, **encoded) == [1.0, 1.0]


def test_trl_reward_fn_accepts_conversational_completions(simple_item):
    """TRL wraps completions as messages when the PROMPT column is conversational.

    The paired-rollout path requires conversational prompts (a pre-rendered string
    cannot be re-templated per thinking mode), which silently changes the shape
    TRL hands the reward: `[{"role": "assistant", "content": ...}]` instead of a
    string. Unhandled, the parser finds no tool call inside a list and scores every
    rollout zero — no crash, just a training run that learns from nothing.
    """
    from rewards.reward import make_metric_fns

    sample = simple_item["sample"]
    call = call_text(simple_item["func_name"], simple_item["args"])
    plain = [call, "<think>hmm</think>" + call]
    conversational = [[{"role": "assistant", "content": c}] for c in plain]
    columns = {
        "function": [sample["function"]] * 2,
        "ground_truth": [sample["ground_truth"]] * 2,
        "category": ["simple_python"] * 2,
    }

    reward_fn = make_reward_fn(FakeTokenizer())
    assert reward_fn(None, conversational, **columns) == reward_fn(None, plain, **columns)

    think_rate = next(f for f in make_metric_fns(FakeTokenizer()) if "think_rate" in f.__name__)
    assert think_rate(None, conversational, **columns) == [0.0, 1.0]


def test_completion_text_rejects_unknown_shapes():
    from rewards.reward import completion_text

    assert completion_text("plain") == "plain"
    assert completion_text([{"role": "assistant", "content": "x"}]) == "x"
    assert completion_text({"role": "assistant", "content": "y"}) == "y"
    with pytest.raises(TypeError):
        completion_text(42)


def test_reward_fn_has_a_stable_name():
    """TRL logs reward functions under __name__; a collision would merge traces."""
    from rewards.reward import make_metric_fns

    names = [make_reward_fn(FakeTokenizer()).__name__] + [
        fn.__name__ for fn in make_metric_fns(FakeTokenizer())
    ]
    assert names[0] == "reward"
    assert len(set(names)) == len(names)


def test_compute_reward_reports_every_component(simple_item):
    """The breakdown is the only way to diagnose a collapsed run.

    A policy that stopped thinking and one that started emitting malformed calls
    both flatten the reward curve; only the components tell them apart.
    """
    call = call_text(simple_item["func_name"], simple_item["args"])
    parsed = parse_model_output("<think>a b c</think>" + call)
    result = compute_reward(
        simple_item["sample"]["function"],
        simple_item["sample"]["ground_truth"],
        parsed,
        "simple_python",
        think_tokens=3,
    )

    assert set(result) == {
        "reward",
        "correct",
        "format_ok",
        "think_cost",
        "think_tokens",
        "did_think",
        "error_type",
    }
    config = RewardConfig()
    expected = config.w_correct * 1.0 + config.w_format * 1.0 - think_cost(3, config)
    assert result["reward"] == pytest.approx(expected)
