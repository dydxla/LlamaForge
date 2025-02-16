import os, glob
from datasets import load_dataset


def dataset_train_test_split(dataset):
    # Split dataset: 90% for training, 10% for testing
    train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return train_dataset, test_dataset


def dataset_processing(dataset):
    # Convert dataset to OAI messages
    system_message = """You are the AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""

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

    train_dataset, test_dataset = dataset_train_test_split(dataset)

    return train_dataset, test_dataset


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


def load_cache_dataset(dataset_path):
    """
    load dataset on cache.
    """
    file_ext = os.path.splitext(dataset_path)[-1][1:]

    dataset = load_dataset(
        file_ext,
        data_files=dataset_path
    )

    return dataset


def dataset_files_is_exist(dataset_path):
    """
    Method to check whether the dataset path exists.

    Args:
        dataset_path (str): dataset path.
        
    Returns:
        bool: exist or not
    """
    if not os.path.exists(dataset_path):
        print(f"{dataset_path} does not exist. Creating default dataset.")
        return False
    else: return True


def dataset_folders_is_exist(dataset_dir):
    """
    Method to check whether the dataset directory exists.
    
    Args:
        dataset_dir (str): dataset directory.
        
    Returns:
        bool: exist or not.
    """
    if not os.path.isdir(dataset_dir):
        print(f"{dataset_dir} does not exist. Creating default dataset.")
        return False
    else: return True


def load_and_template_datasets(tokenizer, data_path):
    """
    dataset load and processing to template.

    Args:
        tokenizer (tokenizer): tokenizer LLM
        data_path (str): dataset path.
        
    Returns:
        dataset: train dataset
        dataset: validation dataset
    """
    if dataset_folders_is_exist(data_path):    # if data path is exist.
        dataset_file_list = glob.glob(os.path.join(data_path, "train", "*"))
        train_datasets = []
        test_datasets = []
        for dataset_file in dataset_file_list:
            dataset = load_cache_dataset(dataset_file)
            train_dataset, test_dataset = dataset_processing(dataset)

    else:
        train_dataset, test_dataset = load_hf_dataset("beomi/KoAlpaca-v1.1a")
    
    def template_dataset(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    return train_dataset, test_dataset
