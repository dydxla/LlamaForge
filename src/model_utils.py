import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto"
    )
    return model


def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model
