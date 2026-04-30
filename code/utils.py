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
    from code.data import extract_boxed_answer
    answer = extract_boxed_answer(pred)
    if answer is None:
        return False
    return answer.strip() == str(gold).strip()


def pass_at_k(n_correct: int, n_samples: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, Codex paper)."""
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n_samples - n_correct, k) / math.comb(n_samples, k)
