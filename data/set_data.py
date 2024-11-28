from datasets import load_dataset

# Convert dataset to OAI messages
system_message = """You are the AI assistant created by BC Card to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""

# Load dataset from the hub
dataset = load_dataset("beomi/KoAlpaca-v1.1a")

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

# Save to JSON files
train_dataset.to_json("train_dataset.json", orient="records", force_ascii=False)
test_dataset.to_json("test_dataset.json", orient="records", force_ascii=False)
