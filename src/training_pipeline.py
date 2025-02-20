from transformers import TrainingArguments
from trl import SFTTrainer
from configs import conf

def create_training_args(output_dir, deepspeed_config):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=conf["per_device_train_batch_size"],
        gradient_accumulation_steps=conf["gradient_accumulation_steps"],
        learning_rate=conf["learning_rate"],
        num_train_epochs=conf["num_train_epochs"],
        logging_dir=f"{output_dir}/logs",
        logging_steps=conf["logging_steps"],
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
        max_seq_length=conf["max_seq_length"],
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": True,
            "append_concat_token": False,
        },
    )
