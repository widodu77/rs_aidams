"""Generate model outputs for a fixed policy and write them to JSONL.

This is the only step that needs a GPU. It deliberately does no scoring: the
raw generations are written to disk, and every downstream step (scoring,
per-category breakdowns, Pareto curves, H1 analysis) reads that file and runs on
CPU. Keeping the split means the expensive step runs once and the analysis can
be re-run and re-derived freely, and it is what lets the Phase D training step
be a single command on a remote GPU with all iteration happening locally.

Usage:
    PYTHONPATH=src uv run python -m generate.run_baseline \
        --model Qwen/Qwen2.5-0.5B-Instruct --policy never --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompts.contract import POLICIES, build_messages, chat_template_kwargs
from scoring.bfcl_scorer import load_category

DEFAULT_CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument(
        "--load-4bit",
        action="store_true",
        help="NF4 quantization. Required for 1.5B on a 4 GB card: fp16 weights alone "
        "are ~3.1 GB against ~3.4 GB free, leaving no room for KV cache.",
    )
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    p.add_argument("--limit", type=int, default=None, help="items per category (default: all)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", default=None, help="output JSONL (default: results/raw/<model>_<policy>.jsonl)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="keep records already present in --out and generate only what is missing. "
        "Generation is slow enough that losing a partial run to an interruption is "
        "expensive, and greedy decoding makes completed records exactly reproducible, "
        "so resuming costs nothing in validity.",
    )
    return p.parse_args()


def load_completed(out_path: str) -> set[tuple[str, str]]:
    """Return `(category, id)` pairs already generated, repairing a partial file.

    A run killed mid-write can leave a truncated final line. Valid records are
    read back and the file rewritten from them, so the malformed remnant is
    dropped and the corresponding item is simply regenerated.
    """
    if not os.path.exists(out_path):
        return set()

    records = []
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Truncated trailing line from an interrupted write.
                break

    with open(out_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    return {(r["category"], r["id"]) for r in records}


def load_model(model_id: str, load_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Decoder-only models must pad on the left for batched generation, otherwise
    # right-padding puts pad tokens between the prompt and the first generated
    # token and the continuation is conditioned on padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # NF4 with double quantization: the standard accuracy-preserving
            # setup. Compute stays in fp16 — only the stored weights are 4-bit,
            # so quality loss is small while weights drop from ~3.1 GB to ~1 GB.
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        quantization_config=quantization_config,
    )
    model.eval()
    return tokenizer, model


def build_work_items(categories: list[str], limit: int | None) -> list[dict]:
    items = []
    for category in categories:
        data = load_category(category)
        if limit is not None:
            data = data[:limit]
        for sample in data:
            items.append({"category": category, "sample": sample})
    return items


@torch.inference_mode()
def generate_batch(tokenizer, model, batch: list[dict], policy: str, max_new_tokens: int) -> list[dict]:
    template_kwargs = chat_template_kwargs(policy)
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(item["sample"], policy),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for item in batch
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        # Greedy: baselines must be deterministic so the comparison against the
        # trained policy reflects the policy, not sampling noise.
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_len = encoded["input_ids"].shape[1]
    completions = outputs[:, prompt_len:]

    results = []
    for item, completion in zip(batch, completions):
        # Strip padding before counting, or every completion in the batch reports
        # the length of the longest one.
        trimmed = completion[completion != tokenizer.pad_token_id]
        text = tokenizer.decode(trimmed, skip_special_tokens=True)
        results.append(
            {
                "id": item["sample"]["id"],
                "category": item["category"],
                "policy": policy,
                "output_text": text,
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(trimmed.shape[0]),
            }
        )
    return results


def main() -> None:
    args = parse_args()

    out_path = args.out or os.path.join(
        "results", "raw", f"{args.model.split('/')[-1]}_{args.policy}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"model={args.model}  policy={args.policy}  4bit={args.load_4bit}  categories={args.categories}")

    completed = load_completed(out_path) if args.resume else set()
    work = build_work_items(args.categories, args.limit)
    if completed:
        before = len(work)
        work = [w for w in work if (w["category"], w["sample"]["id"]) not in completed]
        print(f"resuming: {len(completed)} already present, {before - len(work)} skipped")

    if not work:
        print(f"nothing to do; {out_path} is already complete")
        return

    tokenizer, model = load_model(args.model, load_4bit=args.load_4bit)
    print(f"loaded. VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"{len(work)} items to generate")

    started = time.time()
    written = 0
    with open(out_path, "a" if completed else "w", encoding="utf-8") as fh:
        for start in range(0, len(work), args.batch_size):
            batch = work[start : start + args.batch_size]
            for record in generate_batch(tokenizer, model, batch, args.policy, args.max_new_tokens):
                fh.write(json.dumps(record) + "\n")
                written += 1
            # Flush each batch so an interrupted run loses at most one batch
            # rather than a buffer's worth of expensive generations.
            fh.flush()
            elapsed = time.time() - started
            rate = written / elapsed if elapsed else 0
            print(f"  {written}/{len(work)}  ({rate:.2f} items/s)", flush=True)

    print(f"wrote {written} records to {out_path} in {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
