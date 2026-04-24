import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer

class DoraLayer(nn.Module, LoraLayer):
    def __init__(self):
        super().__init__()