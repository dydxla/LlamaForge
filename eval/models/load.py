from configs import config
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_path: str = None):
    try:
        if not model_path:
            model_path = config['model_path']
        model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map="auto",
                    torch_dtype=torch.float16
        )
        return model
    except ValueError as value_error:
        print(f"ValueError: {value_error} - Check if the model path is correct and contains valid model files.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        
def load_tokenizer(tokenizer_path: str = None):
    try:
        if not tokenizer_path:
            tokenizer_path = config['model_path']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    except ValueError as value_error:
        print(f"ValueError: {value_error} - Check if the tokenizer path is correct and contains valid tokenizer files.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the tokenizer: {e}")
        
    