import torch
from transformers import AutoTokenizer
from llamaforge.finetune.configs.templates import TEMPLATES

def load_tokenizer(
        model_name: str, 
        template_type: str = "chatbot"
    ):
    """
    Prepares and configures a tokenizer for the specified model.

    Args:
        model_name (str): The name of the model to load the tokenizer for.
        template_type (str): Template type you want to use.
        
    Returns:
        AutoTokenizer: The prepared tokenizer with padding token and chat template configured.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = TEMPLATES[template_type]
    return tokenizer