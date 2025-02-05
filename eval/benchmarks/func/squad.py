import re
import string
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from prompts import PromptManager
from configs import config


def normalize_answer(s: str) -> str:
    """
    문자열을 소문자로 변환, 구두점 제거, 관사 제거, 공백 정리하는 메서드

    Args:
        s (str): 처리하고자 하는 문자열

    Returns:
        str : 처리된 문자열
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(prediction: str, ground_truth: str) -> int:
    """
    Exact Match 여부를 1 또는 0으로 반환하는 메서드

    Args:
        prediction (str): 예측 되어져 나온 답변 문자열
        ground_truth (str): 실제 답변 문자열

    Returns:
        int : Exact Match 여부 (0 or 1)
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    예측과 정답 간의 단어 단위 F1 점수를 계산하는 메서드

    Args:
        prediction (str): 예측 되어져 나온 답변 문자열
        ground_truth (str): 실제 답변 문자열

    Returns:
        float : 계산된 F1 점수
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)

    common_tokens = set(pred_tokens) & set(gt_tokens)
    if len(common_tokens) == 0:
        return 0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def eval_squad(model, tokenizer, data_path=None, batch_size=4, device="cuda", split_type="validation"):
    """
    Squad v1.1 벤치마크 평가 함수.

    Args:
        model: 로드된 모델 객체.
        tokenizer: 로드된 토크나이저 객체.
        data_path: squad 데이터셋 경로. None 이면 HuggingFace에서 로드.
        batch_size
        device: 실행 장치 ex-("cuda", "cpu")
        split_type: train or validation
        
    Returns:
        dict: Accuracy 점수가 포함된 평가 결과
    """
    # 데이터 로드
    if data_path:
        dataset = load_dataset("json", data_files=data_path)[split_type]
    else:
        dataset = load_dataset("squad", split=split_type)

    # 모델 평가
    model.eval()
    # model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    
    exact_matches = []
    f1_scores = []

    for example in tqdm(dataset):
        question = example["question"]
        context = example["context"]
        answer = example["answer"]["text"]
        # references.extend(list(map(str, answers)))
        
        # boolq 프롬프트 처리
        prompt_manager = PromptManager()
        prompt = prompt_manager.get_prompt(
            benchmark="squad",
            prompt_type=config['prompt_type']['squad'],
            passage=context,
            question=question)
            
        # 입력 토큰화
        inputs = tokenizer.encode(prompt, padding=True, return_tensors="pt").to(model.device)
        
        # 입력을 모델에 넣어 시퀀스 생성
        with torch.no_grad():
            output_ids = model.generate(    # shape : (batch_size, sequence_length)
                inputs,
                max_new_tokens=config['max_new_tokens'],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # 생성된 전체 텍스트 디코딩
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 프롬프트 부분을 제거하여 생성된 답변만 추출
        generated_answer = generated_text[len(prompt):].strip()
        # 생성된 답변에 여러 줄이 포함된다면 첫 번째 줄만 사용
        generated_answer = generated_answer.split("\n")[0].strip()

        # 모든 정답 중 최고 점수를 선택 (여러 정답 중 가장 좋은 점수)
        em = max(compute_exact(generated_answer, ans) for ans in answer)
        f1 = max(compute_f1(generated_answer, ans) for ans in answer)

        exact_matches.append(em)
        f1_scores.append(f1)

    # 평균 점수 계산 및 출력
    avg_em = sum(exact_matches) / len(exact_matches) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100

    print(f"Average Exact Match (EM): {avg_em:.2f}%")
    print(f"Average F1 Score: {avg_f1:.2f}%")

    return avg_em, avg_f1