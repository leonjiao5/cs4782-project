import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

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
    model, tokenizer = load_base_model(cfg["model_name"], dtype=dtype, load_in_4bit=load_in_4bit)

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

    corpus = load_train_corpus(args.tier)
    train_dataset = build_dataset(corpus, tokenizer)
    print(f"Training on {len(train_dataset)} examples (tier={args.tier})")

    run_id = f"{args.method}_{args.tier}_{datetime.now().strftime('%m%d_%H%M')}"
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({"method": args.method, "tier": args.tier, **cfg}, f, indent=2)

    effective_batch = cfg.get("batch_size", 16)
    per_device_batch = cfg.get("per_device_batch_size", 1)
    grad_accum = max(1, effective_batch // per_device_batch)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.get("lr", 2e-4),
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        bf16=(not load_in_4bit and torch.cuda.is_available()),
        fp16=(load_in_4bit and torch.cuda.is_available()),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch",
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_len", 4096),
    )

    trainer.train()
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
        choices=["train_light", "train", "train_scale"],
        default="train",
    )
    parser.add_argument("--output_dir", default="results/checkpoints")
    main(parser.parse_args())
