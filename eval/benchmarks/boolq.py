from datasets import load_dataset
from evaluate import load as load_metric
import torch

def eval_boolq(model, tokenizer, data_path=None, batch_size=4, device="cuda", split_type="validation"):
    """
    BoolQ 벤치마크 평가 함수.

    Args:
        model: 로드된 모델 객체.
        tokenizer: 로드된 토크나이저 객체.
        data_path: BoolQ 데이터셋 경로. None 이면 HuggingFace에서 로드.
        batch_size: 배치 크기.
        device: 실행 장치 ex-("cuda", "cpu")
        
    Returns:
        dict: Accuracy 점수가 포함된 평가 결과
    """
    # 데이터 로드
    if data_path:
        dataset = load_dataset("json", data_files=data_path)[split_type]
    else:
        dataset = load_dataset("boolq", split=split_type)

    # 모델 평가
    model.eval()
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    
    predictions = []
    references = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        questions = batch["question"]
        contexts = batch["passage"]
        answers = batch["answer"]   # True/False 정답
        
        # boolq 프롬프트 처리
        prompt = load_prompt("boolq", passage=contexts, question=questions)

        # 입력 토큰화
        inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)
        
        # 입력을 모델에 넣어 시퀀스 생성
        with torch.no_grad():
            output_ids = model.generate(    # shape : (batch_size, sequence_length)
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        # 생성된 시퀀스를 토크나이저를 통해 디코딩
        decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]   # List(batch size)
        
        
        
        

        

def boolq_process(split_type="validation"):
    dataset = load_dataset("boolq", split=split_type)
    metric = load_metric("accuracy")
    