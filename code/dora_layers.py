import torch
import torch.nn as nn


class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=0)

        with torch.no_grad():
            initial_weights = base.weight
            self.m = nn.Parameter(initial_weights.norm(p=2, dim=1)) 

        self.dropout = nn.Dropout(p=dropout)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def apply_dora_to_module(model: nn.Module, target_names, rank: int, alpha: float) -> nn.Module:
    raise NotImplementedError
