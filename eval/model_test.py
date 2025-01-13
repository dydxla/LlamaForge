from typing import List
from benchmarks import BenchmarkEvaluator
from configs import config
from models import load_model, load_tokenizer

def eval_model(model_path: str = None, tokenizer_path: str = None, benchmarks: List = None):
    # load model
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    benchmarks_list = benchmarks if benchmarks[0] else config['benchmarks']
    evaluator = BenchmarkEvaluator(model, tokenizer)
    results = evaluator.run_benchmarks(benchmarks_list)
    # ***** run metric code ******
    #
    #
    return results
    
