from datasets import load_dataset

def load_and_template_datasets(tokenizer, train_path, test_path):
    train_dataset = load_dataset(
        "json", 
        data_files=train_path, 
        split="train"
    )
    test_dataset = load_dataset(
        "json", 
        data_files=test_path, 
        split="train"
    )
    
    def template_dataset(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    return train_dataset, test_dataset
