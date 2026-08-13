"""Verification of paired rollouts.

The dangerous failure here is silent. TRL identifies a GRPO group by *consecutive
positions* in the returned batch, so if mode assignment is off by one the trainer
happily computes advantages across rollouts belonging to different prompts. There
is no error, no shape mismatch — just a quietly wrong gradient. These tests pin
the ordering and the counts.

Every test feeds prompts the way TRL actually does: `RepeatSampler` is
constructed with `mini_repeat_count=num_generations`, so each item arrives
already repeated `num_generations` times, contiguously. Reading that batch as a
list of *distinct* prompts is what broke the first four paired runs, and a test
suite that passes unique prompts would keep the bug alive, so `repeated()` below
is used everywhere rather than a bare list.

A fake vLLM backend stands in for `trainer.vllm_generation`: the real one needs a
GPU, and what needs testing is the bookkeeping around it, not vLLM itself.
"""

from __future__ import annotations

import pytest

from train.rollouts import make_paired_rollout_func


class FakeTokenizer:
    """Returns list[int], like the real tokenizer must.

    Token ids encode (mode, prompt identity) so assertions can recover both:
    `[mode, item_index]`. An earlier version of this fake returned
    `[mode, "prompt name"]` — an int and a *string* — which let a real bug
    through: transformers v5 hands back a `BatchEncoding`, vLLM called `max()`
    on it, got a dict key, and died with an unrelated-looking TypeError. A fake
    whose types do not match the real contract tests nothing.
    """

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
    ):
        mode = 1 if kwargs.get("enable_thinking") else 0
        return [mode, int(messages[0]["content"])]


class DictReturningTokenizer(FakeTokenizer):
    """transformers v5 shape: a BatchEncoding-like dict rather than a flat list."""

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": super().apply_chat_template(messages, **kwargs)}


class NestedTokenizer(FakeTokenizer):
    """Some versions wrap a single conversation one level deep."""

    def apply_chat_template(self, messages, **kwargs):
        return [super().apply_chat_template(messages, **kwargs)]


class StringTokenIdTokenizer:
    """The exact shape that broke the first paired run, for the guard to catch."""

    def apply_chat_template(self, messages, **kwargs):
        return ["input_ids", "attention_mask"]


class FakeVLLM:
    """Returns exactly one completion per input row, in order.

    That is what the real backend does when called with `num_generations=1`:
    the argument is a *stride* used to undo the sampler's duplication
    (`prompts[::num_generations]`, then `n` outputs per unique prompt), so a
    stride of 1 takes every row as given. This fake asserts the stride is 1,
    because any larger value would silently drop rows — at stride 8 it would
    keep one thinking row per group and discard all four direct ones, which is
    precisely the collapse paired rollouts exist to prevent.

    Completion lengths differ by mode, as they do in reality: reasoned rollouts
    run hundreds of tokens, direct ones a few dozen. Logprobs are returned in
    vLLM's real 3-D shape `(batch, completion_len, num_logprobs)` with two
    candidates per position, so a caller that forgets to reduce to the top-1 is
    caught here rather than 2500 lines into the trainer.
    """

    def __init__(self):
        self.calls = []

    def generate(self, prompt_ids, images, num_generations, profiler=None):
        assert num_generations == 1, (
            f"stride {num_generations} would slice prompts[::{num_generations}] and "
            "discard rows that were deliberately rendered in different modes"
        )
        self.calls.append((list(prompt_ids), num_generations))
        prompts, completions, logprobs = [], [], []
        for position, ids in enumerate(prompt_ids):
            mode, name = ids[0], ids[1]
            prompts.append(ids)
            # Reasoned completions are longer, exactly as they are in a real run.
            completion = [mode, name, position] + [9] * (5 if mode else 1)
            completions.append(completion)
            logprobs.append([[-0.5, -9.0] for _ in completion])
        return prompts, completions, logprobs, None


class ShortVLLM(FakeVLLM):
    """Returns fewer completions than prompts, to prove the count check fires."""

    def generate(self, prompt_ids, images, num_generations, profiler=None):
        out = super().generate(prompt_ids, images, num_generations, profiler)
        return tuple(field[:-1] if field else field for field in out[:3]) + (None,)


class FakeTrainer:
    def __init__(self, group_size):
        self.num_generations = group_size
        self.vllm_generation = FakeVLLM()


def messages(index):
    """Prompt identity as an int, so token ids can stay list[int]."""
    return [{"role": "user", "content": str(index)}]


def repeated(indices, group_size):
    """The batch shape TRL actually delivers: each item repeated in place.

    `RepeatSampler(mini_repeat_count=num_generations)` yields
    `[0, 0, 0, 0, 1, 1, 1, 1, ...]`, not `[0, 1, 0, 1, ...]`.
    """
    return [messages(index) for index in indices for _ in range(group_size)]


def test_returns_exactly_one_completion_per_prompt_row():
    """The contract that four consecutive failed runs got wrong.

    Prompts arrive pre-repeated, so the answer owed back is `len(prompts)` — not
    `len(prompts) * num_generations`. Overshooting by that factor is invisible
    until the reward functions are asked for one value per completion, where it
    surfaces as "returned 64 rewards, but 8 were expected".
    """
    trainer = FakeTrainer(group_size=8)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)
    prompts = repeated([0], group_size=8)

    out = fn(prompts, trainer)

    assert len(out["completion_ids"]) == len(prompts) == 8
    assert len(out["prompt_ids"]) == len(prompts)
    assert len(out["logprobs"]) == len(prompts)


def test_every_group_contains_both_modes():
    """The whole point: no group may be single-mode.

    If a group holds only reasoned rollouts, GRPO never compares "reason" against
    "do not", which is exactly the failure that produced the flat first sweep.
    """
    trainer = FakeTrainer(group_size=8)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    out = fn(repeated([0, 1], group_size=8), trainer)
    completions = out["completion_ids"]

    for start in range(0, len(completions), 8):
        modes = {c[0] for c in completions[start : start + 8]}
        assert modes == {0, 1}, f"group at {start} is single-mode: {modes}"


def test_groups_are_contiguous_and_prompt_pure():
    """Each block of `num_generations` must belong to exactly one prompt.

    TRL groups by position, so a mis-ordered batch mixes prompts inside one
    advantage computation — wrong, and completely silent.
    """
    trainer = FakeTrainer(group_size=8)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    out = fn(repeated([0, 1, 2], group_size=8), trainer)
    completions = out["completion_ids"]

    assert len(completions) == 3 * 8
    for index, start in enumerate(range(0, len(completions), 8)):
        names = {c[1] for c in completions[start : start + 8]}
        assert names == {index}, f"group {index} mixes prompts: {names}"


def test_all_returned_fields_stay_aligned():
    """prompt_ids[i] must describe completion_ids[i], including its mode."""
    trainer = FakeTrainer(group_size=4)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    out = fn(repeated([0, 1], group_size=4), trainer)

    assert len(out["prompt_ids"]) == len(out["completion_ids"]) == len(out["logprobs"])
    for prompt, completion in zip(out["prompt_ids"], out["completion_ids"]):
        assert prompt[0] == completion[0], "prompt rendered in a different mode than its completion"
        assert prompt[1] == completion[1], "prompt and completion belong to different items"


@pytest.mark.parametrize(
    "group_size,fraction,expected_think",
    [(8, 0.5, 4), (8, 0.25, 2), (4, 0.5, 2), (8, 0.75, 6)],
)
def test_split_sizes(group_size, fraction, expected_think):
    trainer = FakeTrainer(group_size=group_size)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=fraction)

    out = fn(repeated([0], group_size), trainer)
    modes = [c[0] for c in out["completion_ids"]]

    assert len(modes) == group_size
    assert sum(modes) == expected_think


def test_both_modes_are_actually_rendered():
    """One generate call, whose rows carry both chat-template variants.

    Two calls were not needed: mode is a property of how each row is rendered,
    and rendering is per row, so a single batched call covers both.
    """
    trainer = FakeTrainer(group_size=8)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    fn(repeated([0], group_size=8), trainer)

    assert len(trainer.vllm_generation.calls) == 1
    sent, _ = trainer.vllm_generation.calls[0]
    assert len(sent) == 8
    assert [ids[0] for ids in sent] == [1, 1, 1, 1, 0, 0, 0, 0]


def test_degenerate_fractions_are_rejected():
    """A fraction that empties one side recreates the bug this module fixes."""
    with pytest.raises(ValueError):
        make_paired_rollout_func(FakeTokenizer(), think_fraction=1.0)
    with pytest.raises(ValueError):
        make_paired_rollout_func(FakeTokenizer(), think_fraction=0.0)

    # Valid fraction, but too small for the group size to leave any direct rollouts.
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.99)
    with pytest.raises(ValueError, match="one mode only"):
        fn(repeated([0], group_size=2), FakeTrainer(group_size=2))


def test_ragged_batch_is_rejected():
    """Mode is assigned by position, which assumes whole groups.

    A batch that is not a multiple of num_generations means the sampler's
    repetition no longer lines up, and every position-based assumption in this
    function is void.
    """
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)
    with pytest.raises(ValueError, match="not a multiple"):
        fn(repeated([0], group_size=8)[:5], FakeTrainer(group_size=8))


def test_non_uniform_group_is_rejected():
    """Each block of num_generations rows must be one repeated item.

    If TRL ever stopped repeating prompts in place, mode assignment by position
    would compare a thinking rollout on one item against a direct rollout on
    another — a wrong gradient with no error anywhere.
    """
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)
    interleaved = [messages(index % 2) for index in range(8)]  # 0,1,0,1,... not 0,0,0,0,1,1,1,1

    with pytest.raises(RuntimeError, match="differs from the first prompt of its group"):
        fn(interleaved, FakeTrainer(group_size=8))


def test_tokenizer_return_shapes_are_normalised():
    """transformers does not have a stable return type for apply_chat_template.

    v5 returns a BatchEncoding (dict); some versions nest a single conversation
    one level. Both must come back as flat list[int], because vLLM validates
    `max(prompt_token_ids)` and a dict silently yields its largest *key*.
    """
    for tokenizer in (FakeTokenizer(), DictReturningTokenizer(), NestedTokenizer()):
        fn = make_paired_rollout_func(tokenizer, think_fraction=0.5)
        out = fn(repeated([0], group_size=4), FakeTrainer(group_size=4))
        for ids in out["prompt_ids"]:
            assert all(isinstance(t, int) for t in ids), f"{type(tokenizer).__name__} leaked non-ints"


def test_non_integer_token_ids_fail_locally():
    """Guard converts a far-downstream vLLM TypeError into a legible one.

    The first paired run died inside vLLM's input validator with
    "'>' not supported between instances of 'str' and 'int'", which says nothing
    about chat templating.
    """
    fn = make_paired_rollout_func(StringTokenIdTokenizer(), think_fraction=0.5)
    with pytest.raises(TypeError, match="non-integer token ids"):
        fn(repeated([0], group_size=4), FakeTrainer(group_size=4))


def test_logprobs_are_reduced_to_one_float_per_token():
    """vLLM hands back top-k logprobs per position; TRL expects only the sampled one.

    TRL's standard path does `[[lp[0] for lp in seq] for seq in logprobs]` right
    after generating, so `rollout_func` owes it the same 2-D shape. Returning the
    raw 3-D form does not fail at the boundary — it survives all the way to the
    importance-sampling correction and dies there as

        RuntimeError: size of tensor a (64) must match tensor b (361) at dim 1

    where 64 and 361 are two *completion lengths*, naming nothing about logprobs.
    """
    trainer = FakeTrainer(group_size=8)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    out = fn(repeated([0, 1], group_size=8), trainer)

    for completion, sequence in zip(out["completion_ids"], out["logprobs"]):
        assert len(sequence) == len(completion), "logprobs must align token-for-token"
        for value in sequence:
            assert isinstance(value, float), f"expected a float per token, got {value!r}"
            assert value == -0.5, "reduction must keep the sampled token, i.e. index 0"


def test_short_backend_response_is_caught():
    """A backend returning too few completions must fail loudly, not silently.

    An undersized batch would otherwise reach the reward functions and be
    reported there as a reward-count error, several layers from its cause.
    """
    trainer = FakeTrainer(group_size=8)
    trainer.vllm_generation = ShortVLLM()
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    with pytest.raises(RuntimeError, match="exactly one per prompt row"):
        fn(repeated([0, 1], group_size=8), trainer)
