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

AIME_EVAL_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the following competition math problem "
    "step by step, showing your full reasoning. "
    "Write your final answer inside \\boxed{}. "
    "The answer is an integer between 0 and 999 inclusive."
)

AMC12_EVAL_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the following multiple-choice competition math problem "
    "step by step, showing your full reasoning. "
    "The answer is one of the choices A, B, C, D, or E. "
    "Write your final answer as a single letter inside \\boxed{}, e.g. \\boxed{A}."
)

AMC12_EVAL_SYSTEM_PROMPT_STRONG = (
    "You are an expert mathematician. Solve the following multiple-choice competition math problem "
    "step by step, showing your full reasoning. "
    "The answer is one of the choices A, B, C, D, or E. "
    "You MUST end your response with your final answer as a single capital letter inside \\boxed{}, "
    "for example \\boxed{A} or \\boxed{C}. "
    "Do not write a number — write only the letter corresponding to the correct choice."
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


def load_amc12(year: int) -> list[dict]:
    amc12_dir = os.path.join(DATA_DIR, f"amc12_{year}")
    problems = []
    for path in sorted(glob.glob(os.path.join(amc12_dir, "*.json"))):
        with open(path) as f:
            item = json.load(f)
        problems.append({
            "id": Path(path).stem,
            "problem": item["problem"],
            "answer": str(item["answer"]).strip().upper(),
        })
    return problems


def extract_mc_answer(text: str) -> str | None:
    """Extract last A/B/C/D/E choice from \\boxed{} or plain text."""
    import re
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        candidate = boxed.strip().upper()
        if len(candidate) == 1 and candidate in "ABCDE":
            return candidate
    matches = re.findall(r'\b([A-E])\b', text)
    return matches[-1] if matches else None


def load_math() -> list[dict]:
    """Load the sampled hendrycks/math test problems (levels 1, 3, 5, 25% each)."""
    jsonl_path = os.path.join(DATA_DIR, "math_l135", "math_l135.jsonl")
    problems = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
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


def format_prompt(problem: str, tokenizer=None, system_prompt: str = None) -> str:
    """Eval-time prompt: system + user, with generation prompt appended."""
    sp = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sp},
        {"role": "user", "content": problem},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback: Qwen ChatML format (<|im_start|>/<|im_end|>).
    return (
        f"<|im_start|>system\n{sp}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_training_example(problem: str, solution: str, tokenizer=None) -> str:
    """Training-time text: full conversation including the assistant solution."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # Fallback: Qwen ChatML format (<|im_start|>/<|im_end|>).
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{solution}<|im_end|>\n"
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
