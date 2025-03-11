import torch, platform, inspect
from transformers import TrainingArguments
from trl import SFTTrainer
from llamaforge.finetune.configs import _conf
from llamaforge.finetune.core.train.base_trainer import BaseTrainer
from llamaforge.finetune.core.models import load_model
from llamaforge.finetune.core.tokenizers import load_tokenizer
from llamaforge.finetune.core.datasets import load_and_template_datasets
from llamaforge.finetune.utils import get_peft_config


class FinetuneTrainer(BaseTrainer):
    def __init__(
            self, 
            model_name: str, 
            dataset_path: str, 
            model_dtype = torch.float16, 
            template_type: str = "chatbot",
            initial_configs = _conf,
    ):
        super().__init__(initial_configs)
        self.model_name = model_name
        self.dataset_path = dataset_path

        # Load model and tokenizer
        self.model = load_model(model_name, torch_dtype=model_dtype)
        self.tokenizer = load_tokenizer(model_name, template_type=template_type)

        # Load dataset
        self.train_dataset, self.test_dataset = load_and_template_datasets(self.tokenizer, data_path=self.dataset_path)

        # Initialize TrainingArguments
        
    def __create_training_args(
            self, 
            output_dir, 
            deepspeed_config
    ):
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=_conf["per_device_train_batch_size"],
            gradient_accumulation_steps=_conf["gradient_accumulation_steps"],
            learning_rate=_conf["learning_rate"],
            num_train_epochs=_conf["num_train_epochs"],
            logging_dir=f"{output_dir}/logs",
            logging_steps=_conf["logging_steps"],
            save_strategy="epoch",
            evaluation_strategy="no",
            fp16=True,
            deepspeed=deepspeed_config,
            report_to="none",
        )

    def __create_trainer(
            model, 
            tokenizer, 
            train_dataset, 
            test_dataset, 
            training_args,
            peft_config,
    ):
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            dataset_text_field="text",
            max_seq_length=_conf["max_seq_length"],
            tokenizer=tokenizer,
            peft_config=peft_config,
            packing=False,
            dataset_kwargs={
                "add_special_tokens": True,
                "append_concat_token": False,
            },
        )

    def run_finetune(
            self,
            model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
            method: str = "lora",
            dataset_dir: str = "./datasets",
            output_dir: str = "./output",
            deepspeed_config: str = "../../configs/deepspeed/ds_config.json",
            fp16: bool = True,
            **kwargs
    ):
        if platform.system()=="Windows":
            deepspeed_config = None
        peft_config = get_peft_config if method=="lora" else None

        # 모든 파라미터를 자동으로 가져오기
        training_args_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        sft_trainer_keys = set(inspect.signature(SFTTrainer.__init__).parameters.keys())

        # 불필요한 키 제거
        training_args_keys -= {'self', 'args', 'kwargs'}
        sft_trainer_keys -= {'self', 'args', 'kwargs'}

        # kwargs에서 분리
        training_args_kwargs = {k: v for k, v in kwargs.items() if k in training_args_keys}
        sft_trainer_kwargs = {k: v for k, v in kwargs.items() if k in sft_trainer_keys}

        # TrainingArguments 및 SFTTrainer 생성
        training_args = self.__create_training_args(output_dir, deepspeed_config, **training_args_kwargs)
        
        sft_trainer = self.__create_trainer(
            self.model,
            self.tokenizer,
            self.train_dataset,
            self.test_dataset,
            training_args,
            peft_config,
            **sft_trainer_kwargs
        )

        sft_trainer.train()

