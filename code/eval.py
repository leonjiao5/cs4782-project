import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import Counter

import torch

from code.config import RESULTS_DIR
from code.data import load_aime, format_prompt, extract_boxed_answer
from code.model import load_base_model
from code.utils import is_correct, pass_at_k


def load_eval_model(checkpoint: str, model_override: str | None, load_in_4bit: bool):
    """
    Load model for eval:
    - If checkpoint dir has adapter_config.json  → peft adapter on top of base model
    - Otherwise (HF model name or plain dir)     → base model only (zero-shot)
    """
    adapter_cfg_path = os.path.join(checkpoint, "adapter_config.json")
    dtype = torch.float16 if load_in_4bit else torch.bfloat16

    if os.path.isdir(checkpoint) and os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_name = model_override or adapter_cfg["base_model_name_or_path"]
        model, tokenizer = load_base_model(base_name, dtype=dtype, load_in_4bit=load_in_4bit)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
    else:
        base_name = model_override or checkpoint
        model, tokenizer = load_base_model(base_name, dtype=dtype, load_in_4bit=load_in_4bit)

    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompt: str,
    n: int,
    temperature: float,
    max_new_tokens: int = 2048,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    if n == 1 or temperature == 0.0:
        gen_kwargs = dict(do_sample=False, num_return_sequences=1)
    else:
        gen_kwargs = dict(do_sample=True, temperature=temperature, num_return_sequences=n)

    gen_kwargs.update(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # out shape: (num_return_sequences, prompt_len + new_tokens)
    # repeat greedy output to match expected n when temperature==0
    responses = [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        for seq in out
    ]
    if temperature == 0.0 and n > 1:
        responses = responses * n
    return responses[:n]


def main(args):
    model, tokenizer = load_eval_model(
        args.checkpoint,
        model_override=args.model,
        load_in_4bit=args.load_in_4bit,
    )

    year = int(args.benchmark.replace("aime", ""))
    problems = load_aime(year)
    print(f"Evaluating {args.checkpoint} on AIME {year} ({len(problems)} problems)")

    n_correct_greedy = 0
    results_per_problem = []

    for i, item in enumerate(problems):
        prompt = format_prompt(item["problem"], tokenizer)
        responses = generate_responses(
            model, tokenizer, prompt, n=args.n_samples, temperature=args.temperature
        )

        corrects = [is_correct(r, item["answer"]) for r in responses]
        pred_answers = [extract_boxed_answer(r) for r in responses]

        if corrects[0]:
            n_correct_greedy += 1

        majority_correct = False
        if args.n_samples > 1:
            valid = [a for a in pred_answers if a is not None]
            if valid:
                majority_answer = Counter(valid).most_common(1)[0][0]
                majority_correct = majority_answer.strip() == item["answer"].strip()

        results_per_problem.append({
            "problem_id": item["id"],
            "answer": item["answer"],
            "predicted": pred_answers,
            "correct": corrects,
            "majority_correct": majority_correct,
        })

        mark = "✓" if corrects[0] else "✗"
        print(f"  [{i+1:2d}/{len(problems)}] {mark}  pred={pred_answers[0]}  gold={item['answer']}")

    accuracy = n_correct_greedy / len(problems)

    # pass@k summary (only meaningful when n_samples > 1)
    pass_k_summary = {}
    if args.n_samples > 1:
        for k in [1, 2, 4, 8, 16]:
            if k > args.n_samples:
                break
            per_prob = [
                pass_at_k(sum(r["correct"]), args.n_samples, k)
                for r in results_per_problem
            ]
            pass_k_summary[f"pass@{k}"] = sum(per_prob) / len(per_prob)

    print(f"\n{'='*45}")
    print(f"Benchmark : AIME {year}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Greedy acc: {accuracy:.1%}  ({n_correct_greedy}/{len(problems)})")
    for label, val in pass_k_summary.items():
        print(f"{label:10s}: {val:.3f}")

    output = {
        "checkpoint": args.checkpoint,
        "benchmark": args.benchmark,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "n_correct": n_correct_greedy,
        "n_total": len(problems),
        "accuracy": accuracy,
        **pass_k_summary,
        "problems": results_per_problem,
    }

    tables_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    if args.output:
        out_path = args.output
    else:
        ckpt_name = os.path.basename(args.checkpoint.rstrip("/")) or "base"
        out_path = os.path.join(tables_dir, f"{ckpt_name}_{args.benchmark}.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Local peft checkpoint dir, or HF model name for zero-shot")
    parser.add_argument("--benchmark", choices=["aime2024", "aime2025"], default="aime2024")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=None, help="Base model name override")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output", default=None, help="Override output JSON path")
    main(parser.parse_args())
