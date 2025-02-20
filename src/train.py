import os
import argparse
from src.tokenizer_utils import prepare_tokenizer
from src.model_utils import load_model, apply_lora
from src.dataset_utils import load_and_template_datasets
from src.training_pipeline import create_training_args, create_trainer

def main():

    """
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    OUTPUT_DIR = "./llama-int8-q-lora"
    DEEPSPEED_CONFIG = "./ds_config.json"
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data")
    """

    # ArgumentParser 설정
    parser = argparse.ArgumentParser(description="Training LLM Models.")
    
    # 모델 경로
    parser.add_argument(
        "-m", "--model_path", 
        type=str, 
        # default=None,
        required=True,
        help="Path to the pretrained model."
    )
    
    # 템플릿 타입
    parser.add_argument(
        "-t", "--template", 
        # default=None,
        required=True,
        help="Template types. (e.g., 'llama', 'qa', 'chatbot', 'summarization', 'instruction')"
    )

    # 데이터 폴더
    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        default="koalpaca",
        help="Dataset folder path." 
    )

    # 모델 출력 폴더 경로
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./output/default",
        help="model training output directory."
    )
    
    # 파싱된 인자 가져오기
    args = parser.parse_args()

    print("Runing model training...")

    model_path, template_type, data_dir, output_dir = args.model_path, args.template, args.data_dir, args.output_dir

    # 1. Prepare
    tokenizer = prepare_tokenizer(model_path, template_type)    # 토크나이저 세팅
    model = load_model(model_path)
    model = apply_lora(model)

    # 2. Data load
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data")

    train_dataset, test_dataset = load_and_template_datasets(
        tokenizer,
        data_path=os.path.join(DATA_DIR, data_dir)
    )

    # 3. Training set
    OUTPUT_DIR = output_dir
    DEEPSPEED_CONFIG = "./ds_config.json"
    training_args = create_training_args(OUTPUT_DIR, DEEPSPEED_CONFIG)
    trainer = create_trainer(model, tokenizer, train_dataset, test_dataset, training_args)

    # 4. Start Train
    trainer.train()

    # 5. Save model
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
