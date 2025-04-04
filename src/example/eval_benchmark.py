"""
벤치마크 평가 예제 스크립트

이 스크립트는 LlamaForge의 벤치마크 평가 기능을 사용하는 다양한 방법을 보여줍니다.
"""

from llamaforge.eval.benchmarks import BenchmarkEvaluator
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_results(results):
    """벤치마크 결과를 보기 좋게 출력"""
    for benchmark, result in results.items():
        if isinstance(result, tuple):  # SQuAD 결과
            em, f1 = result
            logger.info(f"{benchmark.upper()} Results:")
            logger.info(f"  Exact Match: {em:.2f}%")
            logger.info(f"  F1 Score: {f1:.2f}%")
        else:  # BoolQ 결과
            logger.info(f"{benchmark.upper()} Accuracy: {result:.2f}%")

def main():
    # 1. 기본 설정으로 평가기 생성
    logger.info("1. 기본 설정으로 평가기 생성")
    evaluator = BenchmarkEvaluator()
    
    # 2. 단일 벤치마크 실행 (BoolQ)
    logger.info("\n2. BoolQ 벤치마크 실행")
    boolq_result = evaluator.run_benchmarks("boolq")
    print_results(boolq_result)
    
    # 3. 여러 벤치마크 실행 (BoolQ + SQuAD)
    logger.info("\n3. BoolQ와 SQuAD 벤치마크 실행")
    results = evaluator.run_benchmarks(["boolq", "squad"])
    print_results(results)
    
    # 4. 모든 벤치마크 실행
    logger.info("\n4. 모든 벤치마크 실행")
    all_results = evaluator.run_benchmarks()
    print_results(all_results)
    
    # 5. 커스텀 모델로 평가기 생성
    logger.info("\n5. 커스텀 모델로 평가기 생성")
    custom_evaluator = BenchmarkEvaluator(
        model_path="path/to/your/model",  # 사용자 모델 경로
        tokenizer_path="path/to/your/tokenizer"  # 사용자 토크나이저 경로
    )
    
    # 커스텀 모델로 BoolQ 평가
    logger.info("커스텀 모델로 BoolQ 평가")
    custom_boolq_result = custom_evaluator.run_benchmarks("boolq")
    print_results(custom_boolq_result)

if __name__ == "__main__":
    main()