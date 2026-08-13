"""Paired rollouts: force every GRPO group to contain both thinking modes.

This exists because of two failures found in the first lambda sweep
(notes/2026-08-13.md), which turned out to have a single root cause.

**Failure 1 — the policy never gated.** Think-rate stayed at 98-100% at every
lambda. GRPO computes advantage *within* a group of rollouts on the same prompt,
and the base model emits `<think>` on ~97% of adaptive-prompt samples. So a group
of 8 rollouts essentially never contained a no-think continuation, the comparison
"reason vs do not" was never made, and no reward could price an action absent
from the data. **GRPO can only reinforce behaviour it samples.**

**Failure 2 — lambda had no effect on length either.** With
`scale_rewards="group"` (TRL's default), advantage is `(r - mean) / std`. On an
all-correct group correctness and format are constant, so `r = const - lambda *
think/768` and `std` is exactly `lambda * std(think)/768` — lambda appears in
numerator and denominator and **cancels exactly**. Confirmed against logged
`reward_std` to four significant figures across four runs.

Pairing fixes both, and the second is the non-obvious part: once a group holds
both modes, reward differences come from correctness *and* length, so lambda
changes which rollout **ranks** highest rather than merely rescaling the spread.
Group normalisation divides out a uniform scale; it cannot divide out a change of
ordering.

There is quantitative reason to expect this to bite rather than just relocate the
problem: the per-item oracle found 866 of 1240 items where *both* policies answer
correctly. On that majority a paired group presents two correct rollouts
differing only in length — exactly the clean "shorter wins" signal the gate needs.

This is also what the proposal specified all along: *"During training it samples
both a reasoned and a direct continuation for the same input, and the reward
decides which one paid off."* The first implementation sampled freely and assumed
both modes would appear.
"""

from __future__ import annotations

from prompts.contract import chat_template_kwargs


def make_paired_rollout_func(tokenizer, think_fraction: float = 0.5, policy: str = "adaptive"):
    """Build a TRL `rollout_func` that splits each group across thinking modes.

    TRL calls `rollout_func(prompts, trainer)` with the raw per-process prompt
    slice (no duplication) and expects back a dict with `prompt_ids`,
    `completion_ids` and `logprobs`, each holding `n_prompts * num_generations`
    entries in prompt-major order.

    Rather than driving vLLM directly, this reuses `trainer.vllm_generation`,
    whose `generate()` returns precisely that 4-tuple. Calling it twice — once
    per mode — and interleaving keeps the sampling path, the weight syncing and
    the logprob bookkeeping identical to the standard one. The only thing that
    changes is which chat template each half was rendered with.

    `enable_thinking=False` makes Qwen3's template close an empty
    `<think></think>` inside the *prompt*, so the completion begins outside the
    block. The parser therefore reports `did_think=False` and the reward charges
    zero think tokens — the direct continuation is priced correctly and for free.
    """
    if not 0.0 < think_fraction < 1.0:
        raise ValueError("think_fraction must be strictly between 0 and 1")

    def token_ids(messages, enable_thinking: bool) -> list[int]:
        """Tokenise one conversation, normalising what the tokenizer hands back.

        `apply_chat_template(..., tokenize=True)` does not have a stable return
        type across transformers versions: v5 returns a `BatchEncoding` (a dict)
        rather than a flat list, and a single conversation may also come back
        nested one level. Passing a dict on to vLLM fails far downstream inside
        its input validator with

            TypeError: '>' not supported between instances of 'str' and 'int'

        because `max()` over a dict returns its largest *key*. Normalising here,
        and asserting the element type, keeps that failure local and legible.
        """
        kwargs = chat_template_kwargs(policy)
        kwargs["enable_thinking"] = enable_thinking
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **kwargs,
        )
        if isinstance(out, dict):  # BatchEncoding, despite return_dict=False
            out = out["input_ids"]
        if out and isinstance(out[0], list):  # nested single conversation
            out = out[0]
        out = list(out)
        if not all(isinstance(token, int) for token in out):
            offender = next(t for t in out if not isinstance(t, int))
            raise TypeError(
                "chat template produced non-integer token ids "
                f"(saw {type(offender).__name__}); vLLM needs list[int]."
            )
        return out

    def render(prompts, enable_thinking: bool) -> list[list[int]]:
        return [token_ids(messages, enable_thinking) for messages in prompts]

    def rollout_func(prompts, trainer) -> dict:
        group_size = getattr(trainer, "num_generations", None) or trainer.args.num_generations
        n_think = max(1, round(group_size * think_fraction))
        n_direct = group_size - n_think
        if n_direct < 1:
            raise ValueError(
                f"think_fraction={think_fraction} leaves no direct rollouts at "
                f"num_generations={group_size}; the group would contain one mode only, "
                "which is the failure this function exists to fix."
            )

        thought = trainer.vllm_generation.generate(render(prompts, True), None, n_think)
        direct = trainer.vllm_generation.generate(render(prompts, False), None, n_direct)

        # Both calls return prompt-major batches: prompt 0's completions, then
        # prompt 1's, and so on. Re-interleave so each prompt's full group is
        # contiguous, because TRL identifies a group by consecutive positions.
        prompt_ids: list = []
        completion_ids: list = []
        logprobs: list = []
        for index in range(len(prompts)):
            t0, t1 = index * n_think, (index + 1) * n_think
            d0, d1 = index * n_direct, (index + 1) * n_direct
            prompt_ids.extend(thought[0][t0:t1] + direct[0][d0:d1])
            completion_ids.extend(thought[1][t0:t1] + direct[1][d0:d1])
            logprobs.extend(thought[2][t0:t1] + direct[2][d0:d1])

        expected = len(prompts) * group_size
        if len(completion_ids) != expected:
            raise RuntimeError(
                f"paired rollouts produced {len(completion_ids)} completions, expected "
                f"{expected}. TRL groups by position, so a miscount silently mixes "
                "rollouts from different prompts into one advantage computation."
            )

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
        }

    return rollout_func
