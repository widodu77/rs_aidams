"""The GRPO reward function.

This is the object the whole project turns on. There is no learned reward model
and no human labels: the reward is computed programmatically from the BFCL
checker, so the training signal is exactly the metric the results are reported
against.

    reward = w_correct * correct + w_format * format_ok - lambda_think * cost

`cost` is the *reasoning* length only, not the total completion length. That
distinction is load-bearing. Penalising total tokens would charge the model for
the tool call itself, and calls are not equally long across categories — a
`parallel_multiple` item needs several call blocks and would be penalised for
being that category. The decision under study is whether to *think*, so only
think tokens carry a price.

Nothing here is category-aware beyond what `score_sample` already does. The
reward must not encode a hand-written rule about when reasoning is appropriate,
because "when to think" is precisely what the model is supposed to learn. The
reward states only what thinking *costs*; whether it was worth paying is decided
by whether correctness went up.
"""

from __future__ import annotations

from dataclasses import dataclass

from scoring.bfcl_scorer import ParsedOutput, parse_model_output, score_sample


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the composite reward.

    `lambda_think` is the experimental knob: the whole Phase E tradeoff curve is
    produced by training at several values of it and plotting where each run
    lands. Everything else is held fixed across runs.
    """

    lambda_think: float = 0.2

    # Correctness is the anchor and is fixed at 1.0, so every other weight reads
    # as a fraction of "one correct call".
    w_correct: float = 1.0

    # Deliberately small. `format_ok` means only "nothing malformed was
    # emitted", and it is True for an output containing no tool call at all —
    # prose scores a clean format. At a weight near 1.0 that makes silence a
    # competitive strategy: the model could collect most of the reward by
    # emitting nothing and never risking a wrong call. At 0.2 it stays a
    # tie-breaker between two attempts, which is all it is meant to be, and
    # correctness dominates any choice between calling and not calling.
    w_format: float = 0.2

    # Think cost is normalised by this budget rather than charged per raw token.
    #
    # With a raw token count, `lambda_think` has to be ~1e-3 to be comparable to
    # a correctness of 1.0, and its meaning silently changes whenever the
    # generation cap changes. Normalising to a fraction of the budget makes the
    # penalty bounded in [0, lambda_think], so lambda_think reads directly as
    # "the most that thinking can ever cost, in units of correctness" — which is
    # the quantity the tradeoff curve is parameterised by.
    #
    # Defaults to the Phase B generation cap so a full-length think block costs
    # exactly lambda_think.
    think_token_budget: int = 768

    def __post_init__(self) -> None:
        if self.think_token_budget <= 0:
            raise ValueError("think_token_budget must be positive")
        if self.lambda_think < 0:
            raise ValueError("lambda_think must be non-negative")


def think_cost(think_tokens: int, config: RewardConfig) -> float:
    """Normalised price of a reasoning block, in [0, lambda_think].

    Clipped at the budget so an over-long block cannot be charged more than a
    full one. Without the clip a truncated generation — which already scores
    zero correctness because it never reached the call — would be punished twice
    and dominate the gradient with a signal about the cap rather than about
    reasoning.
    """
    fraction = min(max(think_tokens, 0) / config.think_token_budget, 1.0)
    return config.lambda_think * fraction


def compute_reward(
    functions: list[dict],
    ground_truth: list[dict] | None,
    parsed: ParsedOutput,
    test_category: str,
    think_tokens: int,
    config: RewardConfig | None = None,
) -> dict:
    """Score one rollout. Returns the total plus every component separately.

    The breakdown is returned rather than just the scalar because a GRPO run that
    goes wrong is almost always diagnosed by looking at which term moved: a
    policy collapsing to never-think and one collapsing to malformed output
    produce similarly flat reward curves but completely different component
    traces.
    """
    config = config or RewardConfig()

    result = score_sample(functions, ground_truth, parsed, test_category)
    correct = result["correct"]
    format_ok = float(parsed.format_ok)
    cost = think_cost(think_tokens, config)

    return {
        "reward": config.w_correct * correct + config.w_format * format_ok - cost,
        "correct": correct,
        "format_ok": format_ok,
        "think_cost": cost,
        "think_tokens": think_tokens,
        "did_think": float(parsed.did_think),
        "error_type": result["error_type"],
    }


def score_completion(
    completion: str,
    sample: dict,
    test_category: str,
    count_think_tokens,
    config: RewardConfig | None = None,
) -> dict:
    """Parse a raw generation and reward it.

    `count_think_tokens` is injected rather than a tokenizer being constructed
    here, so that this module stays importable on CPU without loading one, and
    so training reuses the exact tokenizer already in memory.
    """
    parsed = parse_model_output(completion)
    think_tokens = count_think_tokens(parsed.think) if parsed.think else 0
    return compute_reward(
        sample["function"],
        sample["ground_truth"],
        parsed,
        test_category,
        think_tokens,
        config,
    )


def make_reward_fn(tokenizer, config: RewardConfig | None = None):
    """Build a reward function with TRL's `GRPOTrainer` signature.

    TRL calls reward functions as `fn(prompts, completions, **columns)`, where
    every extra dataset column arrives as a list aligned with the batch. It
    expects a plain list of floats back, so the component breakdown is dropped
    here — `compute_reward` is the entry point to use when the breakdown is
    wanted for logging or analysis.
    """
    config = config or RewardConfig()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def reward_fn(prompts=None, completions=None, **columns) -> list[float]:
        functions = columns["function"]
        ground_truth = columns["ground_truth"]
        categories = columns["category"]

        rewards = []
        for completion, fns, gt, category in zip(
            completions, functions, ground_truth, categories
        ):
            parsed = parse_model_output(completion)
            think_tokens = count(parsed.think) if parsed.think else 0
            rewards.append(
                compute_reward(fns, gt, parsed, category, think_tokens, config)["reward"]
            )
        return rewards

    return reward_fn
