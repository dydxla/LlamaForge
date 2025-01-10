import argparse
from evaluate import eval_model

def main():
    # ArgumentParser 설정
    parser = argparse.ArgumentParser(description="Evaluate a model using specified benchmarks and metrics.")
    
    # 모델 경로
    parser.add_argument(
        "-m", "--model_path", type=str, default=None,
        help="Path to the pretrained model. If not specified, uses the default path from config."
    )
    
    # 토크나이저 경로
    parser.add_argument(
        "-t", "--tokenizer_path", type=str, default=None,
        help="Path to the tokenizer. If not specified, uses the default path from config."
    )
    
    # 벤치마크 리스트 (선택적, 기본값은 config에서 로드)
    parser.add_argument(
        "-b", "--benchmarks", nargs="+", default=None,
        help="List of benchmarks to evaluate (e.g., 'hellaswag boolq'). Defaults to all benchmarks in config."
    )
    
    # 파싱된 인자 가져오기
    args = parser.parse_args()
    
    # # eval_model 함수 호출
    # print("Running evaluation...")
    # results = eval_model(
    #     model_path=args.model_path,
    #     tokenizer_path=args.tokenizer_path,
    #     benchmarks=args.benchmarks
    # )
    
    # # 결과 출력
    # print("\nEvaluation Results:")
    # for benchmark, result in results.items():
    #     print(f"{benchmark}: {result}")
    print("model_path : ", args.model_path)
    print("tokenizer_path : ", args.tokenizer_path)
    print("benchmarks : ", args.benchmarks)

if __name__ == "__main__":
    main()
