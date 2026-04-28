import torch
import torch.nn as nn


class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.lora_A = None
        self.lora_B = None
        self.m = None  # per-column magnitude vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def apply_dora_to_module(model: nn.Module, target_names, rank: int, alpha: float) -> nn.Module:
    raise NotImplementedError
