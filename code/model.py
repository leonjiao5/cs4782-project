import torch
import torch.nn as nn


def load_base_model(model_name: str, dtype=torch.bfloat16, device: str = "cuda", load_in_4bit: bool = False):
    raise NotImplementedError


def inject_dora(model: nn.Module, target_modules, rank: int, alpha: float) -> nn.Module:
    raise NotImplementedError


def inject_lora_baseline(model: nn.Module, target_modules, rank: int, alpha: float) -> nn.Module:
    raise NotImplementedError


def inject_peft_dora_baseline(model: nn.Module, target_modules, rank: int, alpha: float) -> nn.Module:
    raise NotImplementedError
