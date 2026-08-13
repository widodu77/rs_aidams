"""Verification of paired rollouts.

The dangerous failure here is silent. TRL identifies a GRPO group by *consecutive
positions* in the returned batch, so if interleaving is off by one the trainer
happily computes advantages across rollouts belonging to different prompts. There
is no error, no shape mismatch — just a quietly wrong gradient. These tests pin
the ordering and the counts.

A fake vLLM backend stands in for `trainer.vllm_generation`: the real one needs a
GPU, and what needs testing is the bookkeeping around it, not vLLM itself.
"""

from __future__ import annotations

import pytest

from train.rollouts import make_paired_rollout_func


class FakeTokenizer:
    """Records which thinking mode each render used, in the token ids themselves."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        marker = 1 if kwargs.get("enable_thinking") else 0
        # Encode (prompt identity, mode) so downstream assertions can recover both.
        return [marker, messages[0]["content"]]


class FakeVLLM:
    """Returns prompt-major batches, tagging every completion with its origin."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt_ids, images, num_generations, profiler=None):
        self.calls.append((prompt_ids, num_generations))
        prompts, completions, logprobs = [], [], []
        for ids in prompt_ids:
            mode, name = ids[0], ids[1]
            for k in range(num_generations):
                prompts.append(ids)
                completions.append([mode, name, k])
                logprobs.append([[0.0]])
        return prompts, completions, logprobs, None


class FakeTrainer:
    def __init__(self, group_size):
        self.num_generations = group_size
        self.vllm_generation = FakeVLLM()


def messages(name):
    return [{"role": "user", "content": name}]


def test_every_group_contains_both_modes():
    """The whole point: no group may be single-mode.

    If a group holds only reasoned rollouts, GRPO never compares "reason" against
    "do not", which is exactly the failure that produced the flat first sweep.
    """
    trainer = FakeTrainer(group_size=8)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    out = fn([messages("a"), messages("b")], trainer)
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

    out = fn([messages("a"), messages("b"), messages("c")], trainer)
    completions = out["completion_ids"]

    assert len(completions) == 3 * 8
    for index, start in enumerate(range(0, len(completions), 8)):
        names = {c[1] for c in completions[start : start + 8]}
        assert names == {"abc"[index]}, f"group {index} mixes prompts: {names}"


def test_all_returned_fields_stay_aligned():
    """prompt_ids[i] must describe completion_ids[i], including its mode."""
    trainer = FakeTrainer(group_size=4)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    out = fn([messages("a"), messages("b")], trainer)

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

    out = fn([messages("a")], trainer)
    modes = [c[0] for c in out["completion_ids"]]

    assert len(modes) == group_size
    assert sum(modes) == expected_think


def test_both_modes_are_actually_requested():
    """One vLLM call per mode, with enable_thinking set differently."""
    trainer = FakeTrainer(group_size=8)
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.5)

    fn([messages("a")], trainer)

    assert len(trainer.vllm_generation.calls) == 2
    modes = [call[0][0][0] for call in trainer.vllm_generation.calls]
    assert set(modes) == {0, 1}, "both chat-template variants must be rendered"


def test_degenerate_fractions_are_rejected():
    """A fraction that empties one side recreates the bug this module fixes."""
    with pytest.raises(ValueError):
        make_paired_rollout_func(FakeTokenizer(), think_fraction=1.0)
    with pytest.raises(ValueError):
        make_paired_rollout_func(FakeTokenizer(), think_fraction=0.0)

    # Valid fraction, but too small for the group size to leave any direct rollouts.
    fn = make_paired_rollout_func(FakeTokenizer(), think_fraction=0.99)
    with pytest.raises(ValueError, match="one mode only"):
        fn([messages("a")], FakeTrainer(group_size=2))
