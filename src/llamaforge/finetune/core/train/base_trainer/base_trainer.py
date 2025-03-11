from llamaforge.finetune.configs import _conf


class BaseTrainer:
    def __init__(self, initial_configs=_conf):
        self.configs = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "logging_steps": 10,
            "max_seq_length": 256
        }
        if initial_configs:
            self.configs.update(initial_configs)

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