"""Generate model outputs with vLLM. Same contract, same JSONL, different engine.

This is `generate.run_baseline` re-targeted at vLLM for remote GPUs. It exists
because HF `generate` batches statically: every sequence in a batch steps until
the *longest* one finishes, so one rambling item drags fifteen idle slots along
with it. Measured on the local 359-item run, 14229 decode steps were spent to
produce 96609 useful tokens — a 2.36x waste factor, and 52% of the steps would
vanish under perfect length grouping. vLLM's continuous batching refills a slot
as soon as a sequence completes, which is exactly that waste.

Deliberately NOT a rewrite of the pipeline:

- Prompts are built by `prompts.contract` and rendered with the HF tokenizer's
  chat template, then handed to vLLM as plain strings. Using `llm.chat()` would
  introduce a second prompt-construction path that could drift from the one the
  Phase B baselines were produced with; this way the prompt bytes are identical
  and `enable_thinking` flows through the code path already validated.
- The output JSONL schema is unchanged, so `analysis.score_run` reads these
  files without modification.

Note on comparability: vLLM and HF `generate` do not produce byte-identical
output even at temperature 0, because the kernels and batching differ. Runs from
the two engines must therefore not be mixed within one reported table — a
baseline set should be regenerated wholesale on whichever engine is used.

Usage:
    PYTHONPATH=src python -m generate.run_vllm --policy always
"""

from __future__ import annotations

import argparse
import json
import os
import time

from transformers import AutoTokenizer

from prompts.contract import POLICIES, build_messages, chat_template_kwargs
from scoring.bfcl_scorer import load_category

DEFAULT_CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    p.add_argument("--limit", type=int, default=None, help="items per category (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="items per vLLM call. Not a batch size — vLLM schedules internally. "
        "This only bounds how much work a Colab disconnect can destroy, since "
        "results are flushed after each chunk.",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        help="'auto' picks bfloat16 where supported. Colab's T4 is Turing and has "
        "no bfloat16, so vLLM falls back to float16 there automatically.",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--out", default=None, help="output JSONL (default: results/raw/<model>_<policy>.jsonl)")
    p.add_argument("--resume", action="store_true", help="skip items already present in --out")
    return p.parse_args()


def load_completed(out_path: str) -> set[tuple[str, str]]:
    """Return `(category, id)` pairs already generated, repairing a partial file.

    Mirrors `run_baseline.load_completed`: a run killed mid-write can leave a
    truncated final line, so valid records are read back and the file rewritten
    from them. On Colab this matters more than locally — sessions are killed on
    idle, not just on error.
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
                break

    with open(out_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    return {(r["category"], r["id"]) for r in records}


def build_work_items(categories: list[str], limit: int | None) -> list[dict]:
    items = []
    for category in categories:
        data = load_category(category)
        if limit is not None:
            data = data[:limit]
        for sample in data:
            items.append({"category": category, "sample": sample})
    return items


def render_prompts(tokenizer, items: list[dict], policy: str) -> list[str]:
    template_kwargs = chat_template_kwargs(policy)
    return [
        tokenizer.apply_chat_template(
            build_messages(item["sample"], policy),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for item in items
    ]


def main() -> None:
    args = parse_args()

    # Imported here rather than at module top so `--help` works, and so an
    # import error names vLLM specifically instead of failing during argparse.
    from vllm import LLM, SamplingParams

    out_path = args.out or os.path.join(
        "results", "raw", f"{args.model.split('/')[-1].lower()}_{args.policy}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"model={args.model}  policy={args.policy}  categories={args.categories}")

    completed = load_completed(out_path) if args.resume else set()
    work = build_work_items(args.categories, args.limit)
    if completed:
        before = len(work)
        work = [w for w in work if (w["category"], w["sample"]["id"]) not in completed]
        print(f"resuming: {len(completed)} already present, {before - len(work)} skipped")

    if not work:
        print(f"nothing to do; {out_path} is already complete")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
    )
    # Greedy, matching the HF baselines: the comparison against the trained
    # policy must reflect the policy, not sampling noise.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    print(f"{len(work)} items to generate")
    started = time.time()
    written = 0

    with open(out_path, "a" if completed else "w", encoding="utf-8") as fh:
        for start in range(0, len(work), args.chunk_size):
            chunk = work[start : start + args.chunk_size]
            prompts = render_prompts(tokenizer, chunk, args.policy)
            outputs = llm.generate(prompts, sampling)

            for item, output in zip(chunk, outputs):
                completion = output.outputs[0]
                fh.write(
                    json.dumps(
                        {
                            "id": item["sample"]["id"],
                            "category": item["category"],
                            "policy": args.policy,
                            "output_text": completion.text,
                            # vLLM reports both lengths exactly, so unlike the HF
                            # path there is no padding to strip before counting.
                            "prompt_tokens": len(output.prompt_token_ids),
                            "completion_tokens": len(completion.token_ids),
                        }
                    )
                    + "\n"
                )
                written += 1
            fh.flush()
            elapsed = time.time() - started
            print(f"  {written}/{len(work)}  ({written/elapsed:.2f} items/s)", flush=True)

    print(f"wrote {written} records to {out_path} in {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
