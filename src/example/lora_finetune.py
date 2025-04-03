from llamaforge.finetune import FinetuneTrainer
import torch
import os

# Hugging Face 토큰 설정 (환경 변수에서 가져오기)
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

# 기본 학습 설정
train_conf = {
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

# LoRA 설정
lora_conf = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# 예제 1: Llama 모델 파인튜닝
def finetune_llama():
    print("Llama 모델 파인튜닝 예제")
    trainer = FinetuneTrainer(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        dataset_path="./datasets/llama_dataset",
        model_dtype=torch.float16,
        template_type="chatbot",
        initial_configs=train_conf,
        initial_lora_configs=lora_conf,
        hf_token=hf_token
    )
    trainer.run_finetune()

# 예제 2: DeepSeek 모델 파인튜닝
def finetune_deepseek():
    print("DeepSeek 모델 파인튜닝 예제")
    # DeepSeek 모델에 맞게 LoRA 설정 조정
    deepseek_lora_conf = lora_conf.copy()
    deepseek_lora_conf["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    trainer = FinetuneTrainer(
        model_name="deepseek-ai/deepseek-llm-7b-base",
        dataset_path="./datasets/deepseek_dataset",
        model_dtype=torch.float16,
        template_type="chatbot",
        initial_configs=train_conf,
        initial_lora_configs=deepseek_lora_conf,
        hf_token=hf_token
    )
    trainer.run_finetune()

# 예제 3: Mistral 모델 파인튜닝
def finetune_mistral():
    print("Mistral 모델 파인튜닝 예제")
    # Mistral 모델에 맞게 LoRA 설정 조정
    mistral_lora_conf = lora_conf.copy()
    mistral_lora_conf["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    trainer = FinetuneTrainer(
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_path="./datasets/mistral_dataset",
        model_dtype=torch.float16,
        template_type="chatbot",
        initial_configs=train_conf,
        initial_lora_configs=mistral_lora_conf,
        hf_token=hf_token
    )
    trainer.run_finetune()

# 예제 4: 커스텀 설정으로 파인튜닝
def finetune_custom():
    print("커스텀 설정으로 파인튜닝 예제")
    # 커스텀 학습 설정
    custom_train_conf = train_conf.copy()
    custom_train_conf["per_device_train_batch_size"] = 1
    custom_train_conf["gradient_accumulation_steps"] = 4
    custom_train_conf["learning_rate"] = 1e-4
    
    # 커스텀 LoRA 설정
    custom_lora_conf = lora_conf.copy()
    custom_lora_conf["r"] = 8
    custom_lora_conf["lora_alpha"] = 16
    
    trainer = FinetuneTrainer(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        dataset_path="./datasets/custom_dataset",
        model_dtype=torch.float16,
        template_type="chatbot",
        initial_configs=custom_train_conf,
        initial_lora_configs=custom_lora_conf,
        hf_token=hf_token
    )
    trainer.run_finetune()

if __name__ == "__main__":
    # 실행할 예제 선택 (주석 해제)
    # finetune_llama()
    # finetune_deepseek()
    # finetune_mistral()
    # finetune_custom()
    
    # 기본 예제 실행
    trainer = FinetuneTrainer(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        dataset_path="./datasets",
        model_dtype=torch.float16,
        template_type="chatbot",
        initial_configs=train_conf,
        initial_lora_configs=lora_conf,
        hf_token=hf_token
    )
    trainer.run_finetune()
