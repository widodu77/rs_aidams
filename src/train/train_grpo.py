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
import glob
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
    # 100 steps at the measured ~85 s/step is ~2.4 h per run, so a five-point
    # lambda sweep fits in roughly 12 h across Colab sessions. At 200 it would be
    # ~33 h including the gate, which free-tier sessions cannot absorb. The
    # proposal already scopes this: "a few hundred GRPO steps on a data subset
    # with LoRA, rather than runs to convergence" — for a 4-page paper a clear
    # trend across lambda beats one converged point.
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="rollouts per prompt. GRPO's advantage is computed within this group, so "
        "it must be >1 or every advantage is zero and nothing is learned. Bigger groups "
        "also reduce how often a group saturates (all rollouts scoring identically), "
        "which produces a zero-variance step that teaches nothing — measured at 3 of 5 "
        "steps in the first smoke test at group size 4.",
    )
    p.add_argument(
        "--use-vllm",
        action="store_true",
        help="route rollout generation through vLLM in-process (colocate mode). "
        "Effectively mandatory on Colab: HF `generate` batches statically, so every "
        "sequence in a rollout batch steps until the longest finishes. Measured at "
        "~15 min/step for 32 completions x 768 tokens, i.e. ~75 h for 200 steps.",
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.35,
        help="fraction of VRAM reserved for vLLM in colocate mode. It holds its own copy "
        "of the weights (3.2 GiB for Qwen3-1.7B fp16), so this budget minus that is the "
        "KV cache. Too high starves training; too low starves the rollout batch. Lower "
        "this first on OOM.",
    )
    p.add_argument(
        "--vllm-sleep",
        action="store_true",
        help="offload vLLM's weights and KV cache between generation phases, freeing ~5 GiB "
        "for the loss pass. OPT-IN, because it requires vLLM's cumem allocator, which needs "
        "`libnvrtc.so.13` — not installed by the cu130 torch wheel. Without it vLLM fails at "
        "construction with 'cumem allocator is not supported on current platform'. Enable "
        "only after `pip install nvidia-cuda-nvrtc` succeeds and `import vllm.cumem_allocator` "
        "works. Reducing --per-device-batch-size is the safer lever.",
    )
    p.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=2048,
        help="bounds vLLM's KV cache per sequence. Worst BFCL prompt is 905 tokens and "
        "completions are capped at --max-completion-length, so prompt+completion fits "
        "well inside this. Capping it is what bought 21x concurrency during baseline "
        "generation.",
    )
    p.add_argument("--max-completion-length", type=int, default=768, help="matches the baselines' cap")
    # NOTE: there is deliberately no --max-prompt-length. TRL removed
    # `max_prompt_length` from GRPOConfig (absent in 1.9.2), and its absence is
    # harmless here: the worst observed BFCL prompt is 905 tokens against a 32k
    # context. Truncating a prompt would delete function schemas and make the
    # item unanswerable, so left-truncation was never something to want.
    p.add_argument("--learning-rate", type=float, default=1e-5)
    # per_device_batch_size is the number of completions in ONE forward pass, and
    # it is the single biggest lever on peak memory — not because of activations
    # (gradient checkpointing handles those) but because of logits. TRL needs
    # per-token log-probabilities, so the forward materialises
    #     batch x completion_tokens x vocab
    # = 8 x 768 x 151936 x 2 bytes ~ 1.9 GiB in fp16, computed twice (policy and
    # reference). At 8 that OOMs a T4 alongside vLLM; at 2 it is ~0.5 GiB.
    #
    # 2 x 4 = 8 completions per optimizer step, i.e. one prompt at group size 8.
    # Raising gradient_accumulation_steps buys more prompts per step (a less noisy
    # gradient) at linear time cost, without touching peak memory.
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="rollout sampling temperature. Above the usual 1.0 on purpose: the first "
        "full run logged entropy at 0.05-0.13, and low rollout diversity means groups "
        "agree, which means zero advantage and no gradient (frac_reward_zero_std hit 1.0 "
        "on three of five steps). More diverse rollouts are the cheapest way to keep "
        "groups informative. Affects training only — evaluation stays greedy, matching "
        "how the fixed-policy baselines were generated.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-steps", type=int, default=1)
    # Frequent enough that an interrupted Colab session loses minutes, not hours.
    # At ~85 s/step a checkpoint every 10 steps caps the loss at ~14 minutes.
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from the latest checkpoint in --output if one exists. Colab "
        "sessions drop, and a run interrupted at step 4 of 50 otherwise starts over. "
        "Safe to pass on a fresh run: with no checkpoint present it simply starts "
        "from scratch.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Must be set before torch initialises CUDA. The rollout / loss alternation
    # allocates and frees large, differently-shaped tensors every step (logits in
    # particular), which fragments the caching allocator badly — the first OOM
    # here reported 458 MiB reserved but unallocated. Expandable segments let the
    # allocator grow a region rather than hunting for an exact-fit block.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    completions_per_step = args.per_device_batch_size * args.gradient_accumulation_steps
    print(f"train {len(train_ds)}  eval {len(eval_ds)}")
    print(
        f"{completions_per_step} completions/step "
        f"({completions_per_step // args.num_generations} prompts x {args.num_generations} rollouts), "
        f"forward batch {args.per_device_batch_size}, "
        f"vllm={'on' if args.use_vllm else 'OFF — expect ~15 min/step'}"
        f"{'' if not args.use_vllm else (', sleep=on' if args.vllm_sleep else ', sleep=off')}"
    )
    # Peak memory is dominated by the logits tensor in the log-prob forward, so
    # it is worth stating up front rather than discovering via OOM.
    logits_gib = args.per_device_batch_size * args.max_completion_length * 151936 * 2 / 1024**3
    print(f"peak logits tensor ~{logits_gib:.2f} GiB per forward (x2 for the reference pass)")
    if completions_per_step % args.num_generations:
        raise ValueError(
            f"per_device_batch_size * gradient_accumulation_steps "
            f"({completions_per_step}) must be divisible by num_generations "
            f"({args.num_generations}); GRPO groups rollouts by prompt."
        )

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
        # Rollout generation, not the optimizer, is what makes a GRPO step
        # expensive. Colocate keeps vLLM in this process rather than requiring a
        # separate server, which is the only shape that fits a single Colab GPU.
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_len,
        # Opt-in: needs vLLM's cumem allocator, which needs libnvrtc.so.13.
        vllm_enable_sleep_mode=args.use_vllm and args.vllm_sleep,
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

    # Headroom check before the first step. On a 15 GiB card this run has three
    # tenants — vLLM's colocated engine, the training model, and the transient
    # logits/backward allocations — and when it fails it does so as
    # `CUBLAS_STATUS_ALLOC_FAILED on cublasCreate`, which does not look like OOM.
    # Printing the free bytes turns that into a number that can be reasoned about.
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(
            f"VRAM before training: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB "
            f"({(total - free) / 2**30:.2f} GiB already held by vLLM + weights)"
        )
        if free < 2 * 2**30:
            print(
                "  WARNING: under 2 GiB free. The backward pass needs room for gradients "
                "and a cuBLAS workspace. Lower --per-device-batch-size or "
                "--vllm-gpu-memory-utilization, or enable --vllm-sleep."
            )

    # `resume_from_checkpoint=True` raises if no checkpoint exists, so probe first
    # and let --resume be harmless on a fresh run.
    resume_from = None
    if args.resume:
        checkpoints = glob.glob(os.path.join(args.output, "checkpoint-*"))
        if checkpoints:
            resume_from = True
            latest = max(checkpoints, key=lambda p: int(p.rsplit("-", 1)[-1]))
            print(f"resuming from {latest}")
        else:
            print("--resume given but no checkpoint found; starting fresh")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(args.output)

    # The log history carries the per-step metric traces; keeping it next to the
    # adapter means a run can be diagnosed later without re-reading stdout.
    with open(os.path.join(args.output, "log_history.json"), "w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    print(f"saved adapter and logs to {args.output}")


if __name__ == "__main__":
    main()
