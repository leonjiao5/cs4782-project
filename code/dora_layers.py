import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base.in_features
        out_features = base.out_features
        device = base.weight.device
        dtype = base.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros((rank, in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=0)

        with torch.no_grad():
            initial_weights = base.weight
            self.m = nn.Parameter(initial_weights.norm(p=2, dim=1).clone())

        for param in self.base.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        self.merged = False
        self._cached_base_weight = None

    def _compute_scaled_weight(self) -> torch.Tensor:
        base_weights = self.base.weight
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        weight_eff = base_weights + lora_update

        # Eq. 11 detach trick: keep denominator constant for gradients.
        row_norm = weight_eff.norm(p=2, dim=1).detach().clamp_min(1e-12)
        return (self.m / row_norm).unsqueeze(1) * weight_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)

        weight_scaled = self._compute_scaled_weight()
        return F.linear(self.dropout(x), weight_scaled, self.base.bias)

    def merge(self) -> None:
        if self.merged:
            return
        if self._cached_base_weight is None:
            self._cached_base_weight = self.base.weight.detach().clone()
        with torch.no_grad():
            self.base.weight.copy_(self._compute_scaled_weight())
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        if self._cached_base_weight is None:
            raise RuntimeError("Cannot unmerge before merge has cached base weight.")
        with torch.no_grad():
            self.base.weight.copy_(self._cached_base_weight)
        self.merged = False


def apply_dora_to_module(
    model: nn.Module,
    target_names: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.05,
):
    target_set = set(target_names)
    trainable_params = []

    def should_replace(module_qualified_name: str, module_name: str) -> bool:
        return module_name in target_set or module_qualified_name in target_set

    def recurse(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in module.named_children():
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and should_replace(qualified_name, child_name):
                dora_layer = DoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                for param in dora_layer.base.parameters():
                    param.requires_grad = False
                setattr(module, child_name, dora_layer)
                trainable_params.extend([dora_layer.m, dora_layer.lora_A, dora_layer.lora_B])
                continue
            recurse(child, qualified_name)

    recurse(model)
    return model, trainable_params
