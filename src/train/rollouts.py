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


def no_think_marker(tokenizer, messages, policy: str) -> list[int]:
    """The tokens Qwen3's template appends when thinking is disabled.

    Derived by rendering the same conversation both ways and taking the
    difference, rather than hard-coding `<think>\\n\\n</think>\\n\\n`. The literal
    string differs between template versions, and a stale constant would silently
    force the wrong tokens into every direct rollout.

    Asserts the no-think rendering is the thinking rendering plus a **suffix**.
    If a future template instead injected the marker earlier in the prompt, the
    difference would not be a suffix and this returns nothing usable — better to
    fail here than to splice tokens into the middle of a completion.
    """
    def render(enable_thinking: bool) -> list[int]:
        kwargs = chat_template_kwargs(policy)
        kwargs["enable_thinking"] = enable_thinking
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
        )
        if isinstance(out, dict):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    thinking, direct = render(True), render(False)
    if len(direct) <= len(thinking) or direct[: len(thinking)] != thinking:
        raise RuntimeError(
            "the no-think rendering is not the thinking rendering plus a suffix; "
            "this template cannot be used for gate rollouts without re-deriving the marker"
        )
    return direct[len(thinking) :]


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


def make_gate_rollout_func(tokenizer, think_fraction: float = 0.5, policy: str = "gate"):
    """Paired rollouts where the decision lives in the *completion*.

    Phase E measured the flaw in `make_paired_rollout_func`: it creates the
    direct branch by re-rendering the prompt with `enable_thinking=False`, which
    makes Qwen3's template append an empty `<think></think>` to the **prompt**.
    The completion therefore begins after the decision not to think, so the
    decision is never something the policy is trained to produce. Reinforcing
    those rollouts raises their likelihood under a prompt that is never deployed,
    and the measured effect was the opposite of the intended one: +32.9 tokens
    per item against the identical unpaired run, with no accuracy gain
    (`notes/2026-08-15.md`).

    Here both halves of the group share **one** prompt, rendered exactly as it
    will be at evaluation. The direct half is forced by prepending the same
    marker tokens to its *completion* instead. Consequences:

    - the decision is inside the completion, so the gradient reaches it;
    - both branches are on-policy for the deployed prompt;
    - the parser already scores an empty think block as `did_think=False` with
      zero think tokens (`bfcl_scorer.py`, and deliberately so), meaning the
      reward prices the direct branch correctly with no changes;
    - at evaluation the model can emit that marker itself, which is precisely
      the gate the project is trying to produce.

    **This returns `logprobs=None`**, because vLLM reports logprobs only for
    tokens it generated and the forced marker has none. TRL treats `None` as
    "no sampling logprobs available" and computes everything from the model, but
    the importance-sampling correction is guarded on a config flag rather than on
    the value, so `vllm_importance_sampling_correction=False` is required. That
    correction has been nearly inert throughout this project
    (`sampling_logp_difference/mean` between 0.005 and 0.018 in every logged
    run), but it is still a change of optimiser, so the comparison run must
    disable it too.
    """
    if not 0.0 < think_fraction < 1.0:
        raise ValueError("think_fraction must be strictly between 0 and 1")

    def token_ids(messages) -> list[int]:
        kwargs = chat_template_kwargs(policy)
        kwargs["enable_thinking"] = True  # identical for both halves, by design
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
        )
        if isinstance(out, dict):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        out = list(out)
        if not all(isinstance(token, int) for token in out):
            offender = next(t for t in out if not isinstance(t, int))
            raise TypeError(
                f"chat template produced non-integer token ids (saw {type(offender).__name__})"
            )
        return out

    def rollout_func(prompts, trainer) -> dict:
        group_size = getattr(trainer, "num_generations", None) or trainer.args.num_generations
        n_think = max(1, round(group_size * think_fraction))
        if group_size - n_think < 1:
            raise ValueError(
                f"think_fraction={think_fraction} leaves no direct rollouts at "
                f"num_generations={group_size}"
            )
        if len(prompts) % group_size:
            raise ValueError(
                f"got {len(prompts)} prompts, not a multiple of num_generations={group_size}; "
                "TRL's sampler emits each prompt exactly num_generations times in a row"
            )

        marker = no_think_marker(tokenizer, prompts[0], policy)

        # One prompt per row, identical within a group. Rows past `n_think` are
        # sent with the marker appended so generation continues *after* it; the
        # marker is moved back into the completion below.
        prompt_ids: list[list[int]] = []
        sent: list[list[int]] = []
        forced: list[bool] = []
        for index, messages in enumerate(prompts):
            position = index % group_size
            if messages != prompts[index - position]:
                raise RuntimeError(
                    f"prompt at position {index} differs from the first of its group"
                )
            base = token_ids(messages)
            prompt_ids.append(base)
            is_direct = position >= n_think
            forced.append(is_direct)
            sent.append(base + marker if is_direct else base)

        _, completion_ids, _, _ = trainer.vllm_generation.generate(sent, None, 1)
        if len(completion_ids) != len(prompts):
            raise RuntimeError(
                f"vllm_generation.generate returned {len(completion_ids)} completions for "
                f"{len(prompts)} prompts; TRL requires exactly one per prompt row."
            )

        completions = [
            list(marker) + list(c) if is_direct else list(c)
            for c, is_direct in zip(completion_ids, forced)
        ]

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completions,
            # See the docstring: the forced marker has no sampled logprob, so the
            # importance-sampling correction must be disabled for this path.
            "logprobs": None,
        }

    return rollout_func
