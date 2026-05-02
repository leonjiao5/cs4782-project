import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import types

from code.dora_layers import apply_dora_to_module


def load_base_model(
    model_name: str,
    dtype=torch.bfloat16,
    device: str = "cuda",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from_pretrained_kw: dict = {"device_map": "auto", "trust_remote_code": trust_remote_code}
    if attn_implementation:
        from_pretrained_kw["attn_implementation"] = attn_implementation

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            **from_pretrained_kw,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **from_pretrained_kw,
        )

    # Qwen and some other CausalLMs ship with pad_token_id=None on config; align with tokenizer
    # so Trainer/collators and generation behave consistently.
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

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


def inject_dora(
    model: nn.Module,
    target_modules,
    rank: int,
    alpha: float,
    dropout: float = 0.05,
) -> nn.Module:
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    for param in model.parameters():
        param.requires_grad = False

    model, trainable_params = apply_dora_to_module(
        model=model,
        target_names=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    if not trainable_params:
        raise ValueError(f"No target linear modules matched target_modules={target_modules}")

    model._dora_trainable_params = trainable_params

    def _print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100.0 * trainable / total if total else 0.0
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")

    model.print_trainable_parameters = types.MethodType(_print_trainable_parameters, model)
    return model
