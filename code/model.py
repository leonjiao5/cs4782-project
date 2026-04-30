import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(
    model_name: str,
    dtype=torch.bfloat16,
    device: str = "cuda",
    load_in_4bit: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_cfg, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )

    return model, tokenizer


def inject_lora_baseline(
    model: nn.Module,
    target_modules,
    rank: int,
    alpha: float,
    dropout: float = 0.05,
) -> nn.Module:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def inject_peft_dora_baseline(
    model: nn.Module,
    target_modules,
    rank: int,
    alpha: float,
    dropout: float = 0.05,
) -> nn.Module:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
    )
    return get_peft_model(model, config)


def inject_dora(model: nn.Module, target_modules, rank: int, alpha: float) -> nn.Module:
    raise NotImplementedError("scratch DoRA not yet implemented")
