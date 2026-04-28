import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_correct(pred: str, gold: str) -> bool:
    raise NotImplementedError


def pass_at_k(n_correct: int, n_samples: int, k: int) -> float:
    raise NotImplementedError
