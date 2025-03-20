import torch, platform, inspect
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from llamaforge.finetune.configs import _train_args, _train_conf, _lora_conf
from llamaforge.finetune.core.train.base_trainer import BaseTrainer
from llamaforge.finetune.core.models import load_model
from llamaforge.finetune.core.tokenizers import load_tokenizer
from llamaforge.finetune.core.datasets import load_and_template_datasets
from llamaforge.finetune.utils import get_peft_config


class FinetuneTrainer(BaseTrainer):
    def __init__(
            self, 
            model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
            dataset_path: str = "./datasets", 
            model_dtype = torch.float16, 
            template_type: str = "chatbot",
            initial_configs = None,
            initial_lora_configs = None
    ):
        super().__init__(initial_configs, initial_lora_configs)
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
            **kwargs
    ):
        return TrainingArguments(**kwargs)

    def __create_trainer(
            self,
            model, 
            tokenizer, 
            train_dataset, 
            test_dataset, 
            training_args,
            peft_config,
            **kwargs
    ):
        default_args = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
        default_args.update(kwargs)
        return SFTTrainer(**default_args)

    def run_finetune(
            self,
            method: str = "lora",
            args: dict = None,
            lora_args: dict = None,
            **kwargs
    ):
        """
        run finetuning method

        Args:
            method (str): lora or mora or None
            args (dict): training arguments
            lora_args (dict): lora configs (when. method=="lora")

        Returns:
            
        """

        # 기본 arguments 정의
        if not args:
            args = self.configs
        if not lora_args:
            lora_args = self.lora_configs
        if platform.system()=="Windows":
            args["deepspeed"] = None

        # 모든 파라미터를 자동으로 가져오기
        training_args_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        sft_trainer_keys = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
        lora_args_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())

        # 불필요한 키 제거
        training_args_keys -= {'self', 'args', 'kwargs'}
        sft_trainer_keys -= {'self', 'args', 'kwargs'}
        lora_args_keys -= {'self', 'args', 'kwargs'}

        # kwargs에서 분리
        training_args_kwargs = {k: v for k, v in args.items() if k in training_args_keys}
        sft_trainer_kwargs = {k: v for k, v in args.items() if k in sft_trainer_keys}

        if kwargs:
            for k, v in kwargs.items():
                if k in training_args_keys and k not in training_args_kwargs:
                    training_args_kwargs.update({k:v})
                elif k in sft_trainer_keys and k not in sft_trainer_kwargs:
                    sft_trainer_kwargs.update({k:v})
                elif k in lora_args_keys and k not in lora_args:
                    lora_args.update({k:v})
                else:
                    raise KeyError(f"Config '{k}' is invalid param.")

        # config 정의
        peft_config = get_peft_config(**lora_args) if method=="lora" else None

        # TrainingArguments 및 SFTTrainer 생성
        training_args = self.__create_training_args(**training_args_kwargs)
        
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

