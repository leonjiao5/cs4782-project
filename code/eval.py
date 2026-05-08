import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import Counter

import torch

from code.config import RESULTS_DIR
from code.data import (load_aime, load_amc12, load_math, format_prompt,
                        extract_boxed_answer, extract_mc_answer,
                        AIME_EVAL_SYSTEM_PROMPT, AMC12_EVAL_SYSTEM_PROMPT,
                        AMC12_EVAL_SYSTEM_PROMPT_STRONG, SYSTEM_PROMPT)
from code.model import load_base_model
from code.utils import is_correct, pass_at_k


def load_eval_model(
    checkpoint: str,
    model_override: str | None,
    load_in_4bit: bool,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
):
    """
    Load model for eval:
    - adapter_config.json present  → peft adapter (lora / peft_dora)
    - dora_config.json present     → scratch DoRA (loads adapter weights only)
    - otherwise                    → base model only (zero-shot)
    """
    adapter_cfg_path = os.path.join(checkpoint, "adapter_config.json")
    dora_cfg_path    = os.path.join(checkpoint, "dora_config.json")
    dtype = torch.float16 if load_in_4bit else torch.bfloat16

    if os.path.isdir(checkpoint) and os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_name = model_override or adapter_cfg["base_model_name_or_path"]
        model, tokenizer = load_base_model(
            base_name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()

    elif os.path.isdir(checkpoint) and os.path.exists(dora_cfg_path):
        with open(dora_cfg_path) as f:
            dora_cfg = json.load(f)
        base_name = model_override or dora_cfg["model_name"]
        model, tokenizer = load_base_model(
            base_name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        from code.model import inject_dora
        model = inject_dora(
            model,
            dora_cfg["target_modules"],
            dora_cfg["rank"],
            dora_cfg["alpha"],
            dora_cfg.get("dropout", 0.05),
        )
        adapter_path = os.path.join(checkpoint, "dora_adapters.pt")
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"Warning: unexpected keys in dora_adapters.pt: {unexpected[:5]}")

    else:
        base_name = model_override or checkpoint
        model, tokenizer = load_base_model(
            base_name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

    model.eval()
    return model, tokenizer


MC_LETTERS = ["A", "B", "C", "D", "E"]
# Suffix appended to the prompt so the next token is naturally a choice letter.
MC_LOGIT_SUFFIX = "\nThe answer is ("


def _letter_logits(model, tokenizer, text: str) -> tuple[str, dict]:
    """Append MC_LOGIT_SUFFIX to text, forward pass, return argmax letter + score dict."""
    inputs = tokenizer(text + MC_LOGIT_SUFFIX, return_tensors="pt").to(model.device)
    letter_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in MC_LETTERS}
    with torch.no_grad():
        last_logits = model(**inputs).logits[0, -1, :]
    scores = {l: last_logits[tid].item() for l, tid in letter_ids.items()}
    return max(scores, key=scores.__getitem__), scores


def score_mc_logit(model, tokenizer, prompt: str) -> tuple[str, dict]:
    """Single forward pass on the bare prompt — no generation."""
    return _letter_logits(model, tokenizer, prompt)


def score_mc_twopass(model, tokenizer, prompt: str,
                     max_new_tokens: int = 4096, top_p: float = 1.0) -> tuple[str, str, dict]:
    """
    Generate full reasoning trace first, then read answer logits from the
    end of the generated text. Returns (predicted_letter, reasoning_text, logit_scores).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
        )
    reasoning = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    # Second pass: read letter logits from end of prompt + reasoning
    full_text = prompt + reasoning
    predicted, scores = _letter_logits(model, tokenizer, full_text)
    return predicted, reasoning, scores


def generate_responses(
    model,
    tokenizer,
    prompt: str,
    n: int,
    temperature: float,
    max_new_tokens: int = 4096,
    top_p: float = 1.0,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    if n == 1 or temperature == 0.0:
        gen_kwargs = dict(do_sample=False, num_return_sequences=1)
    else:
        gen_kwargs = dict(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kwargs.update(
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
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
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    if args.benchmark.startswith("aime"):
        year = int(args.benchmark.replace("aime", ""))
        problems = load_aime(year)
        system_prompt = AIME_EVAL_SYSTEM_PROMPT
        is_mc = False
    elif args.benchmark.startswith("amc12"):
        year = int(args.benchmark.replace("amc12", ""))
        problems = load_amc12(year)
        system_prompt = AMC12_EVAL_SYSTEM_PROMPT_STRONG if args.strong_prompt else AMC12_EVAL_SYSTEM_PROMPT
        is_mc = True
    elif args.benchmark == "math":
        problems = load_math()
        system_prompt = SYSTEM_PROMPT
        is_mc = False
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    # Shard the problem list before resume so each shard has its own partial file
    if args.num_shards > 1:
        problems = problems[args.shard_idx::args.num_shards]
        print(f"[shard {args.shard_idx}/{args.num_shards}] {len(problems)} problems")

    if args.scoring == "logit" and not is_mc:
        raise ValueError("--scoring logit is only supported for AMC12 benchmarks")

    # Resolve output path early so we can resume from it
    tables_dir_early = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(tables_dir_early, exist_ok=True)
    if args.output:
        out_path_early = args.output
    else:
        ckpt_name_early = os.path.basename(args.checkpoint.rstrip("/")) or "base"
        out_path_early = os.path.join(tables_dir_early, f"{ckpt_name_early}_{args.benchmark}.json")
    partial_path = out_path_early + ".partial"

    # Resume: load already-completed problems from a previous interrupted run
    done_ids: set = set()
    results_per_problem: list = []
    n_correct_greedy: int = 0
    for resume_path in [partial_path, out_path_early]:
        if os.path.exists(resume_path):
            try:
                prev = json.load(open(resume_path))
                if "problems" in prev:
                    results_per_problem = prev["problems"]
                    n_correct_greedy = sum(1 for r in results_per_problem if r["correct"][0])
                    done_ids = {r["problem_id"] for r in results_per_problem}
                    print(f"[resume] loaded {len(done_ids)} completed problems from {resume_path}")
                    break
            except Exception:
                pass

    remaining = [item for item in problems if item["id"] not in done_ids]
    print(f"Evaluating {args.checkpoint} on {args.benchmark.upper()} "
          f"({len(problems)} problems, scoring={args.scoring})"
          + (f" — resuming, {len(remaining)} remaining" if done_ids else ""))

    for i, item in enumerate(remaining):
        prompt = format_prompt(item["problem"], tokenizer, system_prompt=system_prompt)

        if args.scoring == "logit":
            pred, logit_scores = score_mc_logit(model, tokenizer, prompt)
            pred_answers = [pred]
            responses = []
            corrects = [pred == item["answer"].upper()]
            majority_correct = corrects[0]
            extra = {"logit_scores": logit_scores}
        elif args.scoring == "twopass":
            pred, reasoning, logit_scores = score_mc_twopass(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens, top_p=args.top_p,
            )
            pred_answers = [pred]
            responses = [reasoning]
            corrects = [pred == item["answer"].upper()]
            majority_correct = corrects[0]
            extra = {"logit_scores": logit_scores}
        else:
            responses = generate_responses(
                model, tokenizer, prompt, n=args.n_samples, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, top_p=args.top_p,
            )
            if is_mc:
                pred_answers = [extract_mc_answer(r) for r in responses]
                corrects = [a is not None and a == item["answer"].upper() for a in pred_answers]
            else:
                pred_answers = [extract_boxed_answer(r) for r in responses]
                corrects = [is_correct(r, item["answer"]) for r in responses]

            majority_correct = False
            if args.n_samples > 1:
                valid = [a for a in pred_answers if a is not None]
                if valid:
                    majority_answer = Counter(valid).most_common(1)[0][0]
                    gold = item["answer"].upper() if is_mc else item["answer"].strip()
                    majority_correct = majority_answer.strip().upper() == gold
            extra = {}

        if corrects[0]:
            n_correct_greedy += 1

        result_entry: dict = {
            "problem_id": item["id"],
            "answer": item["answer"],
            "predicted": pred_answers,
            "responses": responses,
            "correct": corrects,
            "majority_correct": majority_correct,
            **extra,
        }
        if args.benchmark == "math":
            result_entry["level"] = item.get("level")
            result_entry["type"]  = item.get("type")
        results_per_problem.append(result_entry)

        mark = "✓" if corrects[0] else "✗"
        overall_i = len(done_ids) + i + 1
        print(f"  [{overall_i:2d}/{len(problems)}] {mark}  pred={pred_answers[0]}  gold={item['answer']}", flush=True)

        # Write partial results every 10 problems so a timeout doesn't lose everything
        if overall_i % 10 == 0:
            with open(partial_path, "w") as _pf:
                json.dump({"problems": results_per_problem, "n_correct": n_correct_greedy,
                           "n_total": overall_i}, _pf)

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
    print(f"Benchmark : {args.benchmark.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Greedy acc: {accuracy:.1%}  ({n_correct_greedy}/{len(problems)})")
    for label, val in pass_k_summary.items():
        print(f"{label:10s}: {val:.3f}")

    output = {
        "checkpoint": args.checkpoint,
        "benchmark": args.benchmark,
        "scoring": args.scoring,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "n_correct": n_correct_greedy,
        "n_total": len(problems),
        "accuracy": accuracy,
        **pass_k_summary,
        "problems": results_per_problem,
    }

    if args.benchmark == "math":
        from collections import defaultdict
        level_correct: dict = defaultdict(int)
        level_total: dict = defaultdict(int)
        type_correct: dict = defaultdict(int)
        type_total: dict = defaultdict(int)
        for item, res in zip(problems, results_per_problem):
            lv = str(item.get("level", "?"))
            tp = item.get("type", "?") or "?"
            level_total[lv] += 1
            type_total[tp] += 1
            if res["correct"][0]:
                level_correct[lv] += 1
                type_correct[tp] += 1
        output["by_level"] = {
            lv: round(level_correct[lv] / level_total[lv], 4)
            for lv in sorted(level_total)
        }
        output["by_type"] = {
            tp: round(type_correct[tp] / type_total[tp], 4)
            for tp in sorted(type_total)
        }
        print("\nBy level:")
        for lv in sorted(level_total):
            print(f"  Level {lv}: {level_correct[lv]}/{level_total[lv]} "
                  f"({level_correct[lv]/level_total[lv]:.1%})")
        print("By type:")
        for tp in sorted(type_total):
            print(f"  {tp}: {type_correct[tp]}/{type_total[tp]} "
                  f"({type_correct[tp]/type_total[tp]:.1%})")

    out_path = out_path_early
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results → {out_path}")

    if os.path.exists(partial_path):
        os.remove(partial_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Local peft checkpoint dir, or HF model name for zero-shot")
    parser.add_argument("--benchmark",
                        choices=["aime2024", "aime2025", "amc122024", "amc122025", "math"],
                        default="aime2024")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=None, help="Base model name override")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model from HF",
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional attention backend, e.g. sdpa or flash_attention_2",
    )
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling probability (only used when temperature > 0)")
    parser.add_argument("--scoring", choices=["greedy", "logit", "twopass"], default="greedy",
                        help="'greedy': full generation+parse; 'logit': single fwd pass over A-E; 'twopass': generate reasoning then read answer logits (AMC12 only)")
    parser.add_argument("--strong_prompt", action="store_true",
                        help="Use stronger AMC12 format instruction that explicitly forbids writing numbers")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max tokens to generate (used by greedy and twopass scoring)")
    parser.add_argument("--output", default=None, help="Override output JSON path")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards (for parallel MATH eval)")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="Which shard this job processes (0-indexed, stride-based)")
    main(parser.parse_args())
