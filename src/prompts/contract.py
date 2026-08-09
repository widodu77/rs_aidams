"""The output contract, and the two fixed policies that bracket the frontier.

The contract is deliberately identical across policies: the model always emits
tool calls in the same `<tool_call>{...}</tool_call>` form, and the only thing
that varies is whether a `<think>` block precedes them. That matters because the
scorer must be blind to which policy produced an output — if parsing differed
between policies, any accuracy gap would be confounded by parsing artefacts
rather than reflecting the effect of reasoning.

The two policies here are enforced by *prompting*, not training. They provide
the two anchor points of the cost-quality frontier:

- `always` — upper bound on reasoning cost, and the accuracy that unconditional
  reasoning buys
- `never`  — lower bound on cost, and the accuracy floor without reasoning

The Phase D adaptive policy is trained to choose between these per call, and is
measured against both.
"""

from __future__ import annotations

import json

SYSTEM_TEMPLATE = """You have access to the following functions:

<functions>
{functions}
</functions>

To call a function, emit a tool call in exactly this form:
<tool_call>{{"name": "<function name>", "arguments": {{<arguments>}}}}</tool_call>

Rules:
- Respond by emitting tool call(s). Never compute or state the answer yourself, even if the answer is obvious.
- Emit one <tool_call> block per call. If the request needs several calls, emit several blocks.
- Use only functions listed above, and only parameters defined in their schema.
- If none of the available functions can serve the request, emit no <tool_call> block and reply in plain text saying that no suitable function is available.
{policy}"""

# The "never compute the answer yourself" rule is not redundant. Without it a
# 1.5B model shown an easy question (triangle area, factorial) reasons its way
# to the numeric answer in prose and never calls anything — scoring zero on an
# item it fully understood. That failure hits the `always` policy hardest,
# because reasoning is precisely what invites it to solve the problem directly,
# which would bias the baseline comparison against reasoning for the wrong
# reason.

# The irrelevance instruction above is load-bearing: without an explicit licence
# to abstain, a model prompted to call functions will call one for every input,
# and the irrelevance category collapses to 0% for both policies.

# Each policy instruction ends with a literal shape example. A 1.5B model told
# to "reason inside a <think> block" will happily emit the word `think:` as
# prose instead of the tag; showing the exact shape fixes that far more reliably
# than describing it.
POLICY_INSTRUCTIONS = {
    "always": (
        "- Before the tool call(s), reason briefly inside a single "
        "<think>...</think> block. Your entire reply must follow exactly this shape:\n"
        '<think>your reasoning here</think><tool_call>{"name": "...", "arguments": {...}}</tool_call>'
    ),
    "never": (
        "- Do not reason and do not explain. Your entire reply must contain "
        "nothing but the tool call block(s):\n"
        '<tool_call>{"name": "...", "arguments": {...}}</tool_call>'
    ),
    # Used at Phase D, where the decision is the model's to make rather than the
    # prompt's. Kept here so the contract has a single home. Note it shows both
    # shapes, so neither is privileged by the prompt.
    "adaptive": (
        "- You may optionally reason inside a single <think>...</think> block "
        "before the tool call(s), if and only if the request warrants it. Either "
        "of these shapes is valid:\n"
        '<think>your reasoning here</think><tool_call>{"name": "...", "arguments": {...}}</tool_call>\n'
        '<tool_call>{"name": "...", "arguments": {...}}</tool_call>'
    ),
}

POLICIES = tuple(POLICY_INSTRUCTIONS)

# Extra arguments passed to the tokenizer's chat template per policy.
#
# Qwen3 exposes `enable_thinking`, which is exactly the switch these policies
# need: True lets the model open a `<think>` block itself, False makes the
# template close an empty one so the model starts already outside it.
#
# This replaced an earlier prefill mechanism built for Qwen2.5-1.5B-Instruct,
# which could not do the job at all. `<think>` is not in Qwen2.5's vocabulary
# (three tokens) and the model was never trained to emit it — that convention
# arrived with Qwen3/QwQ. Prefilling the opening tag produced well-formed
# `<think>` blocks, but the model then emitted `<|im_end|>` immediately after
# `</think>` and never reached the tool call. Feeding its own completed
# reasoning back as context produced an empty continuation: once `</think>` is
# present, Qwen2.5 considers the turn finished. It does one thing per turn — it
# reasons, or it calls, never both.
#
# In Qwen3 `<think>` (151667) and `</think>` (151668) are native single tokens
# and reason-then-call is a trained behaviour, which is why the base model
# changed. For a project about *when* to think, the base model has to be able to
# represent thinking at all.
CHAT_TEMPLATE_KWARGS = {
    "always": {"enable_thinking": True},
    "never": {"enable_thinking": False},
    # Phase D: the model must be free to choose, so thinking is available but
    # not forced. The reward, not the template, decides whether it is used.
    "adaptive": {"enable_thinking": True},
}


def chat_template_kwargs(policy: str) -> dict:
    """Return chat-template arguments for a policy."""
    if policy not in CHAT_TEMPLATE_KWARGS:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    return dict(CHAT_TEMPLATE_KWARGS[policy])


def build_system_prompt(functions: list[dict], policy: str) -> str:
    """Render the system prompt for one BFCL item under a given policy."""
    if policy not in POLICY_INSTRUCTIONS:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    return SYSTEM_TEMPLATE.format(
        functions=json.dumps(functions, indent=2),
        policy=POLICY_INSTRUCTIONS[policy],
    )


def build_messages(sample: dict, policy: str) -> list[dict]:
    """Build the chat messages for one BFCL item.

    BFCL nests questions as `question[turn][message]`. Only the first turn is
    used here: multi-turn categories are out of scope (see the proposal's scope
    boundaries), and every single-turn category has exactly one turn.
    """
    system = build_system_prompt(sample["function"], policy)
    user_turn = sample["question"][0]
    return [{"role": "system", "content": system}, *user_turn]
