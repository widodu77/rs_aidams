"""Verification of gate rollouts.

The property that matters, and the one Phase E got wrong, is **where the
decision lives**. `make_paired_rollout_func` puts it in the prompt, so the
completion begins after the choice and the policy is never trained to make it.
These tests pin the corrected invariant: every row in a group carries the *same*
prompt, and the direct rows carry the no-think marker at the head of their
**completion**.

A fake tokenizer stands in for Qwen3. It reproduces the one behaviour that
matters here: rendering with `enable_thinking=False` returns the thinking
rendering plus a marker suffix.
"""

from __future__ import annotations

import pytest

from train.rollouts import make_gate_rollout_func, no_think_marker

MARKER = [901, 902]


class FakeTokenizer:
    """Thinking render is `[7, item]`; no-think render appends the marker."""

    def apply_chat_template(self, messages, **kwargs):
        base = [7, int(messages[0]["content"])]
        return base + MARKER if kwargs.get("enable_thinking") is False else base


class NoMarkerTokenizer(FakeTokenizer):
    """A template where disabling thinking changes nothing."""

    def apply_chat_template(self, messages, **kwargs):
        return [7, int(messages[0]["content"])]


class MidInsertTokenizer(FakeTokenizer):
    """A template that injects the marker somewhere other than the end."""

    def apply_chat_template(self, messages, **kwargs):
        base = [7, int(messages[0]["content"])]
        return [7] + MARKER + base[1:] if kwargs.get("enable_thinking") is False else base


class FakeVLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt_ids, images, num_generations, profiler=None):
        assert num_generations == 1
        self.calls.append([list(p) for p in prompt_ids])
        prompts, completions, logprobs = [], [], []
        for position, ids in enumerate(prompt_ids):
            prompts.append(list(ids))
            completions.append([50, position])
            logprobs.append([[-0.5, -9.0], [-0.5, -9.0]])
        return prompts, completions, logprobs, None


class FakeTrainer:
    def __init__(self, group_size):
        self.num_generations = group_size
        self.vllm_generation = FakeVLLM()


def messages(index):
    return [{"role": "user", "content": str(index)}]


def repeated(indices, group_size):
    return [messages(i) for i in indices for _ in range(group_size)]


def test_marker_is_derived_not_hardcoded():
    assert no_think_marker(FakeTokenizer(), messages(0), "gate") == MARKER


def test_template_without_a_marker_is_rejected():
    """If disabling thinking changes nothing, there is no marker to force."""
    with pytest.raises(RuntimeError, match="not the thinking rendering plus a suffix"):
        no_think_marker(NoMarkerTokenizer(), messages(0), "gate")


def test_marker_inserted_mid_prompt_is_rejected():
    """Splicing tokens into the middle of a completion would corrupt it silently."""
    with pytest.raises(RuntimeError, match="not the thinking rendering plus a suffix"):
        no_think_marker(MidInsertTokenizer(), messages(0), "gate")


def test_every_row_in_a_group_shares_one_prompt():
    """The whole point. Phase E gave the two halves *different* prompts, which put
    the direct rollouts off-policy for deployment and inverted the intended effect."""
    trainer = FakeTrainer(group_size=8)
    fn = make_gate_rollout_func(FakeTokenizer(), think_fraction=0.5, policy="gate")

    out = fn(repeated([0, 1], group_size=8), trainer)

    for start in range(0, 16, 8):
        block = out["prompt_ids"][start : start + 8]
        assert all(p == block[0] for p in block), "prompts differ within a group"
        assert MARKER != block[0][-len(MARKER) :], "marker leaked into the prompt"


def test_marker_leads_the_direct_completions_only():
    trainer = FakeTrainer(group_size=8)
    fn = make_gate_rollout_func(FakeTokenizer(), think_fraction=0.5, policy="gate")

    out = fn(repeated([0], group_size=8), trainer)
    completions = out["completion_ids"]

    for i, c in enumerate(completions):
        if i < 4:
            assert c[: len(MARKER)] != MARKER, f"row {i} should be a thinking rollout"
        else:
            assert c[: len(MARKER)] == MARKER, f"row {i} should carry the marker"


def test_marker_is_sent_to_the_backend_for_direct_rows():
    """Generation must continue *after* the marker, not re-emit it."""
    trainer = FakeTrainer(group_size=4)
    fn = make_gate_rollout_func(FakeTokenizer(), think_fraction=0.5, policy="gate")

    fn(repeated([0], group_size=4), trainer)

    sent = trainer.vllm_generation.calls[0]
    assert sent[0][-len(MARKER) :] != MARKER
    assert sent[2][-len(MARKER) :] == MARKER
    assert sent[3][-len(MARKER) :] == MARKER


def test_logprobs_are_none():
    """The forced marker has no sampled logprob, so none are returned at all.

    TRL then computes per-token logps from the model. It only tolerates this with
    `vllm_importance_sampling_correction=False`, which `--gate-rollouts` sets;
    the flag guards that code path rather than a None check.
    """
    trainer = FakeTrainer(group_size=4)
    fn = make_gate_rollout_func(FakeTokenizer(), think_fraction=0.5, policy="gate")
    assert fn(repeated([0], group_size=4), trainer)["logprobs"] is None


def test_counts_and_ragged_batches():
    trainer = FakeTrainer(group_size=8)
    fn = make_gate_rollout_func(FakeTokenizer(), think_fraction=0.5, policy="gate")

    out = fn(repeated([0, 1, 2], group_size=8), trainer)
    assert len(out["completion_ids"]) == len(out["prompt_ids"]) == 24

    with pytest.raises(ValueError, match="not a multiple"):
        fn(repeated([0], group_size=8)[:5], trainer)


def test_non_uniform_group_is_rejected():
    fn = make_gate_rollout_func(FakeTokenizer(), think_fraction=0.5, policy="gate")
    interleaved = [messages(i % 2) for i in range(8)]
    with pytest.raises(RuntimeError, match="differs from the first of its group"):
        fn(interleaved, FakeTrainer(group_size=8))


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_degenerate_fractions_are_rejected(fraction):
    with pytest.raises(ValueError):
        make_gate_rollout_func(FakeTokenizer(), think_fraction=fraction, policy="gate")
