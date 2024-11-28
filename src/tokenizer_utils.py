import torch
from transformers import AutoTokenizer

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

def prepare_tokenizer(model_name):
    """
    Prepares and configures a tokenizer for the specified model.

    Args:
        model_name (str): The name of the model to load the tokenizer for.

    Returns:
        AutoTokenizer: The prepared tokenizer with padding token and chat template configured.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    return tokenizer