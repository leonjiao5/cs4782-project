import math
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_correct(pred: str, gold: str) -> bool:
    import re
    from code.data import extract_boxed_answer

    try:
        gold_int = int(str(gold).strip())
    except ValueError:
        # Non-integer gold: fall back to exact string match
        extracted = extract_boxed_answer(pred)
        return extracted is not None and extracted.strip() == str(gold).strip()

    # (a) last \boxed{...} — try direct int parse
    extracted = extract_boxed_answer(pred)
    if extracted is not None:
        try:
            return int(extracted.strip()) == gold_int
        except ValueError:
            pass
        # (b) coerce \boxed{anything} — grab last integer inside
        nums = re.findall(r'-?\d+', extracted)
        if nums:
            try:
                if int(nums[-1]) == gold_int:
                    return True
            except ValueError:
                pass

    # (c) natural-language answer patterns
    for pat in [r'(?:answer|result)\s+is\s+(\d+)', r'=\s*(\d+)\s*[.$\n]', r'\\boxed\{(\d+)\}']:
        matches = re.findall(pat, pred)
        if matches:
            try:
                if int(matches[-1]) == gold_int:
                    return True
            except ValueError:
                pass

    # (d) last standalone integer 0-999 in the response
    all_ints = re.findall(r'\b(\d{1,3})\b', pred)
    if all_ints:
        try:
            if int(all_ints[-1]) == gold_int:
                return True
        except ValueError:
            pass

    return False


def pass_at_k(n_correct: int, n_samples: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, Codex paper)."""
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n_samples - n_correct, k) / math.comb(n_samples, k)
