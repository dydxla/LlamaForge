from llamaforge.finetune import FinetuneTrainer

## default param.
# model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
# method: str = "lora",
# dataset_dir: str = "./datasets",
# output_dir: str = "./output",
# deepspeed_config: str = "../../configs/deepspeed/ds_config.json"
# per_device_train_batch_size = 2,
# gradient_accumulation_steps = 2,
# learning_rate = 2e-4,
# num_train_epochs = 3,
# logging_steps = 10,
# max_seq_length = 256

_conf = {
    "output_dir": "./output",
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "logging_dir": "./output/logs",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "evaluation_strategy": "no",
    "fp16": True,
    "bf16": False,
    "deepspeed": "../../configs/deepspeed/ds_config.json",
    "report_to": "none",
    "optim": "paged_adamw_32bit",
    "save_steps": 0,
    "weight_decay": 0.001,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 256,
    "packing": False,
    "dataset_text_field": "text",
    "dataset_kwargs": {
        "add_special_tokens": True, 
        "append_concat_token": False
    }
}

_lora_conf = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}


trainer = FinetuneTrainer()
