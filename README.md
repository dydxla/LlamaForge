# LlamaForge 🦙

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Mac-blue)

</div>

LlamaForge는 대규모 언어 모델(LLM)을 쉽게 파인튜닝할 수 있도록 도와주는 오픈소스 라이브러리입니다. Llama, DeepSeek, Grok 등 Hugging Face에서 지원하는 다양한 LLM 모델을 LoRA를 통해 효율적으로 파인튜닝할 수 있으며, Windows, Linux, Mac 환경 모두에서 사용할 수 있습니다.

## ✨ 주요 기능

- 🚀 간단한 API로 다양한 LLM 모델 파인튜닝 (Llama, DeepSeek, Grok 등)
- 🔧 LoRA를 통한 효율적인 파인튜닝 지원
- 📊 BoolQ, SQuAD 등 표준 벤치마크를 통한 모델 성능 평가
- 🌐 Windows, Linux, Mac 환경 지원
- 🛠️ 커스터마이즈 가능한 학습 설정
- 📊 기본 제공되는 데이터셋 템플릿

## 📋 요구사항

- Python 3.10 이상
- CUDA 지원 GPU (권장)
- Hugging Face 계정 및 토큰
- 사용하려는 모델에 대한 접근 권한 (예: Llama 모델의 경우 Meta AI 승인 필요)

## 🚀 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/dydxla/LlamaForge.git
cd LlamaForge
```

2. 패키지 설치:
```bash
pip install -e .
```

## 🔑 Hugging Face 설정

1. [Hugging Face](https://huggingface.co/) 계정 생성
2. Access Token 생성:
   - Settings → Access Tokens → New token
   - 토큰 생성 시 read 권한 부여
3. 모델 접근 권한 얻기:
   - 필요한 경우 해당 모델의 라이선스 동의 (예: Llama 모델의 경우 [Meta AI](https://ai.meta.com/llama/)에서 신청)
   - Hugging Face에서 해당 모델에 대한 접근 권한 요청

## 💻 사용 방법

### 기본 사용법

```python
from llamaforge.finetune import FinetuneTrainer

# Hugging Face 토큰 설정 (환경 변수로 설정했거나 직접 전달)
trainer = FinetuneTrainer(
    model_name="meta-llama/Llama-2-7b-chat-hf",  # 또는 다른 LLM 모델 (예: deepseek-ai/deepseek-llm-7b-base)
    dataset_path="your_dataset_path",  # 데이터셋 경로
    hf_token="your_huggingface_token"  # 선택사항: 환경 변수로 설정했다면 생략 가능
)

# 파인튜닝 시작
trainer.run_finetune()
```

### 환경 변수로 토큰 설정

Windows PowerShell:
```powershell
$env:HUGGING_FACE_HUB_TOKEN = "your_token"
```

Linux/Mac:
```bash
export HUGGING_FACE_HUB_TOKEN="your_token"
```

### 커스텀 설정

```python
from llamaforge.finetune import FinetuneTrainer

trainer = FinetuneTrainer(
    model_name="deepseek-ai/deepseek-llm-7b-base",  # 예: DeepSeek 모델
    dataset_path="your_dataset_path",
    model_dtype=torch.float16,  # 모델 데이터 타입
    template_type="chatbot",    # 데이터셋 템플릿 타입
    initial_configs={           # 학습 설정
        "output_dir": "./output",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "fp16": True,
    },
    initial_lora_configs={      # LoRA 설정
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],  # 모델에 따라 다를 수 있음
    }
)

# 파인튜닝 시작
trainer.run_finetune()
```

### 모델 성능 평가

```python
from llamaforge.eval.benchmarks import BenchmarkEvaluator

# 기본 설정으로 평가기 생성
evaluator = BenchmarkEvaluator(
    model_path="path/to/your/model",      # 평가할 모델 경로
    tokenizer_path="path/to/tokenizer"    # 토크나이저 경로 (선택사항)
)

# 단일 벤치마크 실행 (BoolQ)
boolq_result = evaluator.run_benchmarks("boolq")
print(f"BoolQ Accuracy: {boolq_result['boolq']:.2f}%")

# 여러 벤치마크 실행 (BoolQ + SQuAD)
results = evaluator.run_benchmarks(["boolq", "squad"])
print(f"BoolQ Accuracy: {results['boolq']:.2f}%")
em, f1 = results['squad']
print(f"SQuAD - Exact Match: {em:.2f}%, F1 Score: {f1:.2f}%")

# 모든 벤치마크 실행
all_results = evaluator.run_benchmarks()
```

평가 설정은 `src/llamaforge/eval/configs/config.yaml`에서 커스터마이즈할 수 있습니다:
```yaml
# 평가하고자 하는 벤치마크
benchmarks:
    - boolq
    - squad

# 평가 지표 설정
metrics:
    - exact_match
    - f1

# 실행 옵션
device: cuda      # 실행 장치 ("cuda", "cpu")
max_new_tokens: 100     # 모델 출력의 최대 길이
batch_size: 4       # 배치 크기
```

## 📚 데이터셋 형식

데이터셋은 다음과 같은 JSON 형식을 따릅니다:

```json
[
    {
        "instruction": "질문 또는 지시사항",
        "output": "응답 또는 출력"
    },
    ...
]
```

데이터셋은 내부적으로 다음과 같은 OpenAI 메시지 형식으로 변환됩니다:
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are the AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."
        },
        {
            "role": "user",
            "content": "질문 또는 지시사항"
        },
        {
            "role": "assistant",
            "content": "응답 또는 출력"
        }
    ]
}
```

## 🤝 기여하기

프로젝트에 기여하고 싶으시다면:

1. 이 저장소를 포크합니다
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 Apache License 2.0 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👥 작성자

- dydxla - [GitHub](https://github.com/dydxla)

## 🙏 감사의 글

- [Meta AI](https://ai.meta.com/) - Llama 모델 제공
- [DeepSeek AI](https://deepseek.ai/) - DeepSeek 모델 제공
- [Hugging Face](https://huggingface.co/) - 훌륭한 라이브러리들 제공
- [PEFT](https://github.com/huggingface/peft) - LoRA 구현
- [TRL](https://github.com/huggingface/trl) - SFT Trainer 구현