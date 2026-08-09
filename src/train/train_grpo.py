"""GRPO training with TRL + LoRA (Phase D).

Trains the adaptive policy: the model sees a prompt where reasoning is optional
and the reward decides whether it paid off. Nothing here tells the model when to
think — that is the thing being learned, and any hand-written rule about it
entering the reward would make the result circular.

Two things this script deliberately does NOT do:

- It does not tune the prompt per policy. Training uses `adaptive` only, which
  is the prompt whose fixed-policy behaviour is already measured (96.9%
  think-rate, 87.5% accuracy, 287.5 tokens). That measurement is the "prompting
  alone" control, so anything the trained policy gains over it is attributable
  to training rather than to phrasing.
- It does not evaluate. Evaluation goes through `generate.run_vllm --adapter`
  and `analysis.score_run`, i.e. the exact pipeline that produced the baselines.
  Scoring a trained policy through a different path than its baselines is how
  comparisons quietly break.

Go/no-go gate (from the proposal): run first with `--lambda-think 0.0`, which
reduces the reward to correctness + format. If GRPO cannot improve plain
correctness over the base model, the adaptive question is moot and the project
pivots to characterising why. Only after that passes is lambda swept.

Usage:
    PYTHONPATH=src python -m train.train_grpo --lambda-think 0.0 --output runs/gate
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output", required=True, help="output directory for the adapter")
    p.add_argument(
        "--lambda-think",
        type=float,
        default=0.0,
        help="price of reasoning. 0.0 is the go/no-go gate (correctness + format only). "
        "The informative sweep from the fp16 baselines is {0.05, 0.1, 0.25, 0.5, 1.0, 2.0}: "
        "four of five categories switch off below 0.55, then nothing changes until "
        "irrelevance at 2.11.",
    )
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="rollouts per prompt. GRPO's advantage is computed within this group, so "
        "it must be >1 or every advantage is zero and nothing is learned.",
    )
    p.add_argument("--max-completion-length", type=int, default=768, help="matches the baselines' cap")
    # NOTE: there is deliberately no --max-prompt-length. TRL removed
    # `max_prompt_length` from GRPOConfig (absent in 1.9.2), and its absence is
    # harmless here: the worst observed BFCL prompt is 905 tokens against a 32k
    # context. Truncating a prompt would delete function schemas and make the
    # item unanswerable, so left-truncation was never something to want.
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--per-device-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0, help="rollouts must be sampled, not greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Heavy imports live here so --help works without a GPU stack present.
    import torch
    from datasets import Dataset  # noqa: F401  (imported for its side effect on availability)
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from rewards.reward import RewardConfig, make_metric_fns, make_reward_fn
    from train.dataset import build_datasets

    os.makedirs(args.output, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds = build_datasets(
        tokenizer,
        eval_fraction=args.eval_fraction,
        seed=args.split_seed,
        manifest_path=os.path.join(args.output, "split_manifest.json"),
    )
    print(f"train {len(train_ds)}  eval {len(eval_ds)}")

    reward_config = RewardConfig(
        lambda_think=args.lambda_think,
        think_token_budget=args.max_completion_length,
    )
    with open(os.path.join(args.output, "reward_config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(reward_config) if hasattr(reward_config, "__dict__") else {
            "lambda_think": reward_config.lambda_think,
            "w_correct": reward_config.w_correct,
            "w_format": reward_config.w_format,
            "think_token_budget": reward_config.think_token_budget,
        }, fh, indent=2)

    # The real reward first, then logging-only metrics at weight 0.0. TRL logs
    # every reward function separately, so this yields per-step traces of
    # think-rate, correctness and format without any of them touching the
    # gradient. Collapse to all-think or all-no-think is the primary failure
    # mode and it is invisible in the reward curve alone.
    reward_funcs = [make_reward_fn(tokenizer, reward_config), *make_metric_fns(tokenizer)]
    reward_weights = [1.0] + [0.0] * (len(reward_funcs) - 1)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # All attention and MLP projections. Restricting LoRA to attention only
        # is common but tends to underfit format-shaping behaviour, and the
        # decision being learned here is expressed through output format.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    config = GRPOConfig(
        output_dir=args.output,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        # Truncated completions are NOT masked out of the loss (TRL's default).
        # A rollout that hit the cap never reached its tool call, so it scores
        # zero correctness and pays the full think penalty — which is exactly
        # the signal wanted. Rambling past the budget is a real failure the
        # policy should learn to avoid, not noise to be hidden from it. The fp16
        # baselines put this at 2-3.5% of items, so it is a small effect either
        # way; recorded here because the opposite choice is defensible and
        # should be a decision rather than an accident.
        mask_truncated_completions=False,
        reward_weights=reward_weights,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        # T4 is Turing: no bf16, so fp16 is the fallback. Matches the precision
        # the fp16 baselines were generated at.
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output)

    # The log history carries the per-step metric traces; keeping it next to the
    # adapter means a run can be diagnosed later without re-reading stdout.
    with open(os.path.join(args.output, "log_history.json"), "w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    print(f"saved adapter and logs to {args.output}")


if __name__ == "__main__":
    main()
