import glob
import json
import os
from pathlib import Path

from code.config import DATA_DIR

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the following competition math problem "
    "step by step, showing your full reasoning. "
    "Write your final answer inside \\boxed{}."
)


def load_aime(year: int) -> list[dict]:
    aime_dir = os.path.join(DATA_DIR, f"aime_{year}")
    problems = []
    for path in sorted(glob.glob(os.path.join(aime_dir, "*.json"))):
        with open(path) as f:
            item = json.load(f)
        problems.append({
            "id": Path(path).stem,
            "problem": item["problem"],
            "answer": str(item["answer"]),
        })
    return problems


def load_train_corpus(name: str) -> list[dict]:
    """name: 'train_light', 'train', or 'train_scale'"""
    jsonl_path = os.path.join(DATA_DIR, name, f"{name}.jsonl")
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_prompt(problem: str, tokenizer=None) -> str:
    """Eval-time prompt: system + user, with generation prompt appended."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback when tokenizer has no chat_template (Llama-3-style tags).
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{problem}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def format_training_example(problem: str, solution: str, tokenizer=None) -> str:
    """Training-time text: full conversation including the assistant solution."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # Fallback when tokenizer has no chat_template (Llama-3-style tags).
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{problem}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{solution}<|eot_id|>"
    )


def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None
