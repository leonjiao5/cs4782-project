import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from code.config import RESULTS_DIR
from code.data import load_train_corpus, format_training_example
from code.model import load_base_model, inject_lora_baseline, inject_peft_dora_baseline, inject_dora
from code.utils import set_seed, load_config


def build_dataset(corpus: list[dict], tokenizer) -> Dataset:
    rows = [
        {"text": format_training_example(item["problem"], item["solution"], tokenizer)}
        for item in corpus
    ]
    return Dataset.from_list(rows)


def main(args):
    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model

    set_seed(cfg["seed"])

    if args.method == "none":
        print("method=none: zero-shot baseline needs no training, exiting.")
        return

    load_in_4bit = cfg.get("load_in_4bit", False)
    dtype = torch.float16 if load_in_4bit else torch.bfloat16
    model, tokenizer = load_base_model(
        cfg["model_name"],
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=cfg.get("trust_remote_code", False),
        attn_implementation=cfg.get("attn_implementation") or None,
    )

    rank = cfg.get("rank", 16)
    alpha = cfg.get("alpha", 32)
    dropout = cfg.get("dropout", 0.05)
    target_modules = cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"])

    if args.method == "lora":
        model = inject_lora_baseline(model, target_modules, rank, alpha, dropout)
    elif args.method == "peft_dora":
        model = inject_peft_dora_baseline(model, target_modules, rank, alpha, dropout)
    elif args.method == "dora":
        model = inject_dora(model, target_modules, rank, alpha, dropout)
    elif args.method == "full":
        raise NotImplementedError("full fine-tuning not yet wired up")

    model.print_trainable_parameters()

    # Frozen backbone + gradient checkpointing: inputs must require grad for backward
    # through checkpointed segments (applies to peft and scratch DoRA).
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    corpus = load_train_corpus(args.tier)
    train_dataset = build_dataset(corpus, tokenizer)
    print(f"Training on {len(train_dataset)} examples (tier={args.tier})")

    run_id = args.run_id or f"{args.method}_{args.tier}_{datetime.now().strftime('%m%d_%H%M')}"
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({"method": args.method, "tier": args.tier, **cfg}, f, indent=2)

    effective_batch = cfg.get("batch_size", 16)
    per_device_batch = cfg.get("per_device_batch_size", 1)
    grad_accum = max(1, effective_batch // per_device_batch)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.get("lr", 2e-4),
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        bf16=(
            not load_in_4bit
            and (torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        ),
        fp16=(load_in_4bit and torch.cuda.is_available()),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 5),
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=cfg.get("max_seq_len", 4096),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    if args.method == "dora":
        # Save only the trainable adapter weights (m, lora_A, lora_B).
        # Avoids writing a full ~14 GB model copy; base weights reload from HF at eval time.
        dora_state = {
            n: p.data.cpu()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        torch.save(dora_state, os.path.join(output_dir, "dora_adapters.pt"))
        with open(os.path.join(output_dir, "dora_config.json"), "w") as f:
            json.dump(
                {
                    "rank": rank,
                    "alpha": alpha,
                    "dropout": dropout,
                    "target_modules": list(target_modules),
                    "model_name": cfg["model_name"],
                },
                f,
                indent=2,
            )
    else:
        trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)
    print(f"Checkpoint saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/configs/default.yaml")
    parser.add_argument("--model", default=None, help="Override model_name in config")
    parser.add_argument(
        "--method",
        choices=["dora", "lora", "peft_dora", "full", "none"],
        default="lora",
    )
    parser.add_argument(
        "--tier",
        default="train",
        help="Training data tier; any name matching data/{tier}/{tier}.jsonl",
    )
    parser.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "checkpoints"))
    parser.add_argument("--run_id", default=None,
                        help="Override auto-generated run ID (method_tier_timestamp)")
    parser.add_argument("--resume", default=None,
                        help="Path to a Trainer checkpoint dir to resume from")
    main(parser.parse_args())
