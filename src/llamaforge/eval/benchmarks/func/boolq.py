import os
from datetime import datetime
from datasets import load_dataset
from evaluate import load as load_metric
import torch
from typing import List
from llamaforge.eval.prompts import PromptManager
from llamaforge.eval.prompts.templates import get_boolq_answer
from llamaforge.eval.configs import config

def is_correct(gentext: str):
    prompt_type = config['prompt_type']['boolq']
    ans_text = get_boolq_answer(prompt_type=prompt_type)
    if ans_text in gentext:
        prediction_str = gentext.split(ans_text)[-1].strip()
    else:
        prediction_str = gentext

    # 간단히 'yes'/'no'가 들어있는지 체크
    prediction_str_lower = prediction_str.lower()
    if "yes" in prediction_str_lower and "no" not in prediction_str_lower:
        return "True"
    elif "no" in prediction_str_lower and "yes" not in prediction_str_lower:
        return "False"
    else:
        # 애매한 경우, 좀 더 세부 파싱 로직이 필요할 수 있음
        # 일단 'yes'/'no' 중 하나 선택
        return "True" if "yes" in prediction_str_lower else "False"
    

def boolq_res_process(predictions: List, references: List):
    metric = load_metric("accuracy")
    accuracy_score = metric.compute(predictions=[1 if ch=='True' else 0 for ch in predictions], references=[1 if ch=='True' else 0 for ch in references])["accuracy"]
    print(f"BoolQ Accuracy (on {len(predictions)} samples): {accuracy_score:.2f}")
    os.makedirs(config['log_root_dir'], exist_ok=True)

    # 파일 경로 설정
    file_path = os.path.join(config['log_root_dir'], "boolq.log")
    
    content = f"""BoolQ Accuracy (on {len(predictions)} samples): {accuracy_score:.2f}"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content_with_timestamp = f"[{timestamp}] {content}"
    
    # 파일 생성 및 내용 저장
    with open(file_path, "a") as log_file:
        log_file.write(content_with_timestamp + "\n")
    
    return accuracy_score



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
    # model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    
    predictions = []
    references = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        questions = batch["question"]
        contexts = batch["passage"]
        answers = batch["answer"]   # True/False 정답
        references.extend(list(map(str, answers)))
        
        # boolq 프롬프트 처리
        prompt_manager = PromptManager()
        prompt = [prompt_manager.get_prompt(
            benchmark="boolq",
            prompt_type=config['prompt_type']['boolq'],
            passage=context,
            question=question) for context, question in zip(contexts, questions)
            ]

        # 입력 토큰화
        inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)
        
        # 입력을 모델에 넣어 시퀀스 생성
        with torch.no_grad():
            output_ids = model.generate(    # shape : (batch_size, sequence_length)
                **inputs,
                max_new_tokens=config['max_new_tokens'],
                do_sample=False
            )
        
        # 생성된 시퀀스를 토크나이저를 통해 디코딩
        decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]   # List(batch size)
        
        # 배치 내의 각 생성된 텍스트에 대해 answer 값 가져오기
        for gentext in decoded:
            predictions.append(is_correct(gentext))

    boolq_accuracy = boolq_res_process(predictions=predictions, references=references)
    return boolq_accuracy