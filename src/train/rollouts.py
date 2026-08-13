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

    **The contract, which cost four failed runs to establish.** TRL's
    `RepeatSampler` is built with `mini_repeat_count=num_generations`, so every
    prompt is already duplicated `num_generations` times, contiguously, *before*
    the batch reaches here. `prompts` is therefore not a list of distinct items:
    at `num_generations=8` a batch of 8 rows is one item repeated eight times.
    `rollout_func` owes back exactly `len(prompts)` completions — one per row,
    not one per row times the group size.

    That single misreading produced three different errors in a row, because it
    is wrong by a factor of `num_generations` and nothing checks it until the
    reward functions are asked for one value per completion.

    Generation goes through `trainer.vllm_generation.generate`, which keeps the
    sampling path, weight syncing and logprob bookkeeping identical to the
    standard one. It is called with `num_generations=1`: that argument is a
    *stride* used to undo the sampler's duplication
    (`prompts[::num_generations]`, then `n=num_generations` per unique prompt),
    and a stride of 1 means "take every row as given, one completion each" —
    which is what is wanted here, since each row has already been rendered in its
    own thinking mode and rows within a group are deliberately no longer
    identical.

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
        if len(prompts) % group_size:
            raise ValueError(
                f"got {len(prompts)} prompts, which is not a multiple of "
                f"num_generations={group_size}. TRL's sampler emits each prompt exactly "
                "num_generations times in a row, so a ragged batch means that assumption "
                "no longer holds and mode assignment by position would be meaningless."
            )

        # A row's position inside its block of `group_size` decides its mode. The
        # block is one item repeated, so this is the whole intervention: the same
        # prompt is rendered with thinking enabled for the first `n_think` rows
        # and disabled for the rest, which is what puts both modes in a group.
        rendered: list[list[int]] = []
        for index, messages in enumerate(prompts):
            position = index % group_size
            if messages != prompts[index - position]:
                raise RuntimeError(
                    f"prompt at position {index} differs from the first prompt of its "
                    "group. Modes are assigned by position on the assumption that each "
                    "block of num_generations rows is one repeated item; if that is no "
                    "longer true the two modes would be compared across different items."
                )
            rendered.append(token_ids(messages, enable_thinking=position < n_think))

        # num_generations=1 means stride 1: no de-duplication, one completion per
        # row. Anything larger would discard rows -- including, here, every direct
        # row -- since generate() slices `prompts[::num_generations]`.
        prompt_ids, completion_ids, logprobs, _ = trainer.vllm_generation.generate(
            rendered, None, 1
        )
        if len(completion_ids) != len(prompts):
            raise RuntimeError(
                f"vllm_generation.generate returned {len(completion_ids)} completions for "
                f"{len(prompts)} prompts; TRL requires exactly one per prompt row."
            )

        # vLLM returns per-token top-k logprobs, shape
        # (batch, completion_len, num_logprobs). TRL's standard path reduces this
        # to the sampled token's logprob before use, and rollout_func must hand
        # back the same 2-D shape. Returning the raw 3-D form survives padding and
        # stacking and fails only in the importance-sampling correction, as
        # "size of tensor a (64) must match tensor b (361)" -- two completion
        # lengths, naming nothing about logprobs.
        logprobs = [[lp[0] for lp in sequence] for sequence in logprobs]

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
        }

    return rollout_func
