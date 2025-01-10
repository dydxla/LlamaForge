from typing import List, Dict
from .func import eval_boolq
from configs import config

class BenchmarkEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # self.eval_hellaswag = eval_hellaswag
        self.eval_boolq = eval_boolq
        # self.eval_squad = eval_squad
        # self.eval_mmlu = eval_mmlu
        self.config = config
        self.batch_size = self.config['batch_size']
        self.device = self.config['device']
        self.dataset_split_type = "validation"

    def get_data_path(self, benchmark: str):
        """
        Retrieves the dataset for a given benchmark. If `data_path` is specified and non-empty,
        return the path from config; otherwise, return None value.

        Args:
            benchmark (str): Name of the benchmark.

        Returns:
            str: data_path or None
        """
        # Get data path from config
        data_paths = self.config.get("data_paths", {})
        data_path = data_paths.get(benchmark, "")

        if data_path:  # If a data path is provided from config file.
            return data_path
        else:  # Otherwise, return None.
            return None
        
    def run_benchmarks(self, benchmarks: List) -> Dict:
        results = {}
        for benchmark in benchmarks:
            method_name = f"eval_{benchmark}"
            try:
                if hasattr(self, method_name):
                    results[benchmark] = getattr(self, method_name)(
                                            model=self.model,
                                            tokenizer=self.tokenizer,
                                            data_path=self.get_data_path(benchmark),
                                            batch_size=self.batch_size,
                                            device=self.device,
                                            split_type=self.dataset_split_type
                    )
                else:
                    raise AttributeError(f"No evaluation method for benchmark '{benchmark}'")
            except Exception as e:
                print(f"Error: {e}")
        return results