from peft import LoraConfig, get_peft_model


def get_peft_config(
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        bias: str = "none",
        task_type: str = "CAUSAL_LM"
):
    return LoraConfig(
        r=r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout, 
        bias=bias,
        task_type=task_type
    )


def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        # target_modules=["q_proj", "k_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model