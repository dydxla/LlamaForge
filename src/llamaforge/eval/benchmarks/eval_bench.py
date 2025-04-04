from typing import List, Dict, Tuple, Union
from llamaforge.eval.benchmarks.func import eval_boolq, eval_squad
from llamaforge.eval.configs import config
import traceback
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkEvaluator:
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        """
        벤치마크 평가기 초기화

        Args:
            model_path (str, optional): 모델 경로. 기본값은 config.yaml의 model_path
            tokenizer_path (str, optional): 토크나이저 경로. 기본값은 model_path와 동일
        """
        self.config = config
        self.model_path = model_path or self.config['model_path']
        self.tokenizer_path = tokenizer_path or self.model_path
        
        # 모델과 토크나이저 로드
        logger.info(f"Loading model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # 평가 함수 설정
        self.eval_boolq = eval_boolq
        self.eval_squad = eval_squad
        
        # 기본 설정
        self.batch_size = self.config['batch_size']
        self.device = self.config['device']
        self.dataset_split_type = "validation"

    def get_data_path(self, benchmark: str) -> str:
        """
        벤치마크에 대한 데이터셋 경로를 반환

        Args:
            benchmark (str): 벤치마크 이름

        Returns:
            str: 데이터셋 경로 또는 빈 문자열
        """
        data_paths = self.config.get("data_paths", {})
        return data_paths.get(benchmark, "")

    def run_benchmarks(self, benchmarks: Union[str, List[str]] = None) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        지정된 벤치마크들을 실행하고 결과를 반환

        Args:
            benchmarks (Union[str, List[str]], optional): 
                실행할 벤치마크 이름 또는 이름 목록.
                기본값은 config.yaml의 모든 벤치마크

        Returns:
            Dict[str, Union[float, Tuple[float, float]]]: 
                벤치마크별 평가 결과
                - BoolQ: float (정확도)
                - SQuAD: Tuple[float, float] (EM 점수, F1 점수)

        Examples:
            >>> evaluator = BenchmarkEvaluator()
            >>> # 단일 벤치마크 실행
            >>> result = evaluator.run_benchmarks("boolq")
            >>> # 여러 벤치마크 실행
            >>> results = evaluator.run_benchmarks(["boolq", "squad"])
            >>> # 모든 벤치마크 실행
            >>> results = evaluator.run_benchmarks()
        """
        if benchmarks is None:
            benchmarks = self.config['benchmarks']
        elif isinstance(benchmarks, str):
            benchmarks = [benchmarks]
        
        results = {}
        for benchmark in benchmarks:
            method_name = f"eval_{benchmark}"
            try:
                if hasattr(self, method_name):
                    logger.info(f"Running {benchmark} benchmark...")
                    results[benchmark] = getattr(self, method_name)(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        data_path=self.get_data_path(benchmark),
                        batch_size=self.batch_size,
                        device=self.device,
                        split_type=self.dataset_split_type
                    )
                    logger.info(f"Completed {benchmark} benchmark")
                else:
                    logger.error(f"No evaluation method for benchmark '{benchmark}'")
            except Exception as e:
                logger.error(f"Error in {benchmark}: {str(e)}")
                logger.debug(traceback.format_exc())
        return results