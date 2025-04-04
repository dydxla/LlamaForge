from typing import List, Dict

def get_squad_templates() -> Dict:
    return {
            "basic": """Context: {passage}

                Question: {question}

                Answer:"""
            }