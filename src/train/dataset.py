"""Build the GRPO training dataset, and the train/eval split.

Two things happen here, and the second one is the methodologically load-bearing
half.

**Prompt rendering.** The prompt is rendered to a *string* with the tokenizer's
chat template, using the same `prompts.contract` functions that produced the
baselines. This is deliberate: TRL applies its own chat template only when the
prompt column is conversational (a list of messages). Handing it a pre-rendered
string means there is exactly one prompt-construction path in the project, so a
trained policy is answering byte-identical prompts to the ones the baselines
were measured on. Any drift there would silently invalidate the comparison the
whole project rests on.

**The split.** The trained policy has to be evaluated on items it never trained
on, and the fixed-policy baselines it is compared against have to be restricted
to that same held-out set. Because the baselines cover all 1240 items, any split
can be scored retroactively without regenerating anything — but only if the
split is deterministic and recorded. Hence a fixed seed and a written manifest.

The split is stratified by category. BFCL categories differ enormously in how
much reasoning is worth (`irrelevance` +44 points, `simple_python` -2.8), so an
unstratified split would change the reward landscape between train and eval and
make the two incomparable.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from prompts.contract import build_messages, chat_template_kwargs
from scoring.bfcl_scorer import load_category

DEFAULT_CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance")

# The policy used for training prompts. `adaptive` is the only correct choice:
# the model must be free to decide, so the prompt shows both output shapes and
# privileges neither. Which one it picks is what the reward is there to teach.
TRAIN_POLICY = "adaptive"


def build_records(
    tokenizer,
    categories=DEFAULT_CATEGORIES,
    policy: str = TRAIN_POLICY,
    conversational: bool = False,
) -> list[dict]:
    """Render every item into a training record.

    `conversational=True` keeps the prompt as a list of messages instead of a
    pre-rendered string. That is required by the paired-rollout path: it has to
    render each prompt *twice*, once with `enable_thinking=True` and once with
    `False`, which is impossible once the template has already been applied.
    TRL's own guidance is the same — chat templating belongs inside
    `rollout_func`, at the backend boundary.

    The default stays False so the standard path keeps exactly one
    prompt-construction route, byte-identical to the one the baselines used.
    """
    template_kwargs = chat_template_kwargs(policy)
    records = []
    for category in categories:
        for sample in load_category(category):
            messages = build_messages(sample, policy)
            prompt = (
                messages
                if conversational
                else tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            )
            records.append(
                {
                    "prompt": prompt,
                    # JSON-encoded, NOT nested objects. `datasets` builds an
                    # Arrow schema by type inference, and neither column can be
                    # given one: every item's function schema has a different
                    # shape, and BFCL ground truth mixes types inside a single
                    # acceptable-values list (e.g. `"formatted": [true, ""]`),
                    # which Arrow rejects outright with
                    #   ArrowInvalid: Could not convert 'true' with type str
                    #
                    # Encoding to strings also guarantees the structures reach
                    # the reward function exactly as written, with no Arrow type
                    # coercion silently reshaping them in between.
                    # `rewards.reward` decodes them at the TRL boundary.
                    "function": json.dumps(sample["function"]),
                    "ground_truth": json.dumps(sample["ground_truth"]),
                    "category": category,
                    "id": sample["id"],
                }
            )
    return records


def split_records(
    records: list[dict], eval_fraction: float = 0.2, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Stratified, deterministic train/eval split.

    Seeded and stratified per category so the split is reproducible from the
    seed alone and both halves see the same category mix.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    train, evaluation = [], []
    for category in sorted(by_category):
        # Sort by id first so the shuffle depends only on the seed, never on the
        # order the dataset files happened to be read in.
        items = sorted(by_category[category], key=lambda r: r["id"])
        rng = random.Random(f"{seed}:{category}")
        rng.shuffle(items)
        cut = round(len(items) * eval_fraction)
        evaluation.extend(items[:cut])
        train.extend(items[cut:])

    return train, evaluation


def write_manifest(path: str, train: list[dict], evaluation: list[dict], seed: int) -> None:
    """Record which ids landed in which half.

    The manifest is what lets `analysis.score_run` restrict the fixed-policy
    baselines to the eval set later. Without it the trained policy would be
    compared against baselines computed partly on its own training data.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": seed,
                "train_ids": [[r["category"], r["id"]] for r in train],
                "eval_ids": [[r["category"], r["id"]] for r in evaluation],
            },
            fh,
            indent=2,
        )


def load_manifest(path: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Read a split manifest back as `(train_ids, eval_ids)` sets."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return (
        {(c, i) for c, i in data["train_ids"]},
        {(c, i) for c, i in data["eval_ids"]},
    )


def build_datasets(
    tokenizer,
    categories=DEFAULT_CATEGORIES,
    eval_fraction: float = 0.2,
    seed: int = 0,
    manifest_path: str | None = "results/split_manifest.json",
    conversational: bool = False,
    policy: str = TRAIN_POLICY,
):
    """Build HF Datasets for training. Returns `(train_ds, eval_ds)`.

    `policy` selects the prompt wording. It must match whatever the trained
    adapter is later evaluated under, or the policy is being scored on prompts it
    never saw. The split itself is independent of it: `split_records` keys on
    category and item id, so every policy yields the same 992/248 partition.
    """
    from datasets import Dataset

    records = build_records(tokenizer, categories, policy=policy, conversational=conversational)
    train, evaluation = split_records(records, eval_fraction, seed)

    if manifest_path:
        write_manifest(manifest_path, train, evaluation, seed)

    return Dataset.from_list(train), Dataset.from_list(evaluation)
