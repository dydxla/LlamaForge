import os
from src.tokenizer_utils import prepare_tokenizer
from src.model_utils import load_model, apply_lora
from src.dataset_utils import load_and_template_datasets
from src.training_pipeline import create_training_args, create_trainer

def main():
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    OUTPUT_DIR = "./llama-int8-q-lora"
    DEEPSPEED_CONFIG = "./ds_config.json"
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data")

    # 1. Prepare
    tokenizer = prepare_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)
    model = apply_lora(model)

    # 2. Data load
    train_dataset, test_dataset = load_and_template_datasets(
        tokenizer,
        train_path=os.path.join(DATA_DIR, "train_dataset.json"),
        test_path=os.path.join(DATA_DIR, "test_dataset.json"),
    )

    # 3. Training set
    training_args = create_training_args(OUTPUT_DIR, DEEPSPEED_CONFIG)
    trainer = create_trainer(model, tokenizer, train_dataset, test_dataset, training_args)

    # 4. Start Train
    trainer.train()

    # 5. Save model
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
