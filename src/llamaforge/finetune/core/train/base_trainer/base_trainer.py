from llamaforge.finetune.configs import _train_args, _train_conf, _lora_conf


class BaseTrainer:
    
    TRAIN_ARGS = _train_args
    TRAIN_PARAMS = _train_conf
    LORA_ARGS = _lora_conf

    def __init__(
            self, 
            initial_configs: dict = None, 
            initial_lora_configs: dict = None
    ):
        self.configs = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "logging_steps": 10,
        }

        self.lora_configs = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }

        if not initial_configs:
            self.configs = {**self.TRAIN_ARGS, **self.TRAIN_PARAMS}
        else:
            self.configs.update(initial_configs)
        if not initial_lora_configs:
            self.lora_configs = self.LORA_ARGS
        else:
            self.lora_configs.update(initial_lora_configs)

    def __repr__(self,):
        return f"BaseTrainer(configs={self.configs})"
    
    def update_config(self, key, value):
        if key in self.configs:
            self.configs[key] = value
        else:
            raise KeyError(f"Config '{key}' does not exist in configs.")
        
    def get_config(self, key):
        return self.configs.get(key, None)

    def show_configs(self):
        return self.configs