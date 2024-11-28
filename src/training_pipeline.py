from transformers import TrainingArguments
from trl import SFTTrainer

def create_training_args(output_dir, deepspeed_config):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True,
        deepspeed=deepspeed_config,
        report_to="none",
    )

def create_trainer(model, tokenizer, train_dataset, test_dataset, training_args):
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": True,
            "append_concat_token": False,
        },
    )
