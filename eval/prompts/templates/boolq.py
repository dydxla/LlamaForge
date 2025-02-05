from typing import List, Dict

def get_boolq_templates() -> Dict:
    return {
            "basic": """Passage: {passage}
                Question: {question}
                Answer (yes or no):""",
            "simple": """Refer to the passage below and write the correct answer (yes or no) to the question.
                Passage: {passage}
                Question: {question}
                Answer (yes or no):""",
            "detailed": """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                Question: {question} 
                Context: {passage} 
                Answer:""",
            "fewshot-cot": """Below are examples of answering yes/no questions given a passage. 
                We include a step-by-step reasoning (chain-of-thought) before arriving at the final yes/no answer.

                Example 1:
                Passage: "A cat is one of the mammals and is commonly kept as a pet at home."
                Question: "Is a cat a reptile?"
                Reasoning: A cat is a mammal. It is not a reptile.
                Answer: no

                Example 2:
                Passage: "Chocolate is made from ingredients extracted from cocoa beans. It is usually a sweet snack."
                Question: "Is cocoa bean the raw material for chocolate?"
                Reasoning: Chocolate is made from cocoa beans. The answer to the question is 'yes.'
                Answer: yes

                Now here is a new question:

                Passage: "{passage}"
                Question: "{question}"
                Let's think step-by-step (chain of thought) then decide the answer as yes or no.

                Reasoning:""",
            }

def get_boolq_answer(prompt_type: str) -> str:
    if prompt_type in ['basic', 'simple']:
        return "Answer (yes or no):"
    elif prompt_type in ['detailed', 'fewshot-cot']:
        return "Answer:"
    else:
        raise ValueError(f"Invalid prompt type ({prompt_type}) in benchmark <boolq>.")