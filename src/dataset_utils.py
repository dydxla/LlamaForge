import os
from datasets import load_dataset

def load_hf_dataset(dataset_name):
    # Convert dataset to OAI messages
    system_message = """You are the AI assistant created by BC Card to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""

    # Load dataset from the hub
    dataset = load_dataset(dataset_name)

    columns_to_remove = list(dataset["train"].features)

    # Convert to messages format
    dataset = dataset.map(
        lambda sample: {
            'messages': [
                {"role": "system", "content": system_message},
                {"role": "user", "content": sample['instruction'].replace("'",'').replace('"','').replace('\0xa0',' ')},
                {"role": "assistant", "content": sample['output'].replace("'",'').replace('"','').replace('\0xa0',' ')}
            ]
        },
    )

    # Remove unused columns
    dataset = dataset.map(remove_columns=columns_to_remove, batched=False)

    # Split dataset: 90% for training, 10% for testing
    train_test_split = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset

def dataset_files_is_exist(dataset_path):
    """
    Method to check whether the dataset path exists.

    Args:
        dataset_path (str): dataset directory.
        
    Returns:
        bool: exist or not
    """
    if not os.path.exists(dataset_path):
        print(f"{dataset_path} does not exist. Creating default dataset.")
        return False
    else: return True

def load_and_template_datasets(tokenizer, data_path):
    if dataset_files_is_exist(data_path):
        
        if os.path.splitext(train_path) == '.json':
            train_dataset = load_dataset(
                "json", 
                data_files=train_path, 
                split="train"
            )
            if dataset_files_is_exist(test_path):
                test_dataset = load_dataset(
                    "json", 
                    data_files=test_path, 
                    split="train"
                )
    else:
        train_dataset, test_dataset = load_hf_dataset("beomi/KoAlpaca-v1.1a")
    
    def template_dataset(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    return train_dataset, test_dataset
