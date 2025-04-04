"""
Evaluation functions for different benchmarks.
Currently supports:
- BoolQ: Boolean Question Answering
- SQuAD: Reading Comprehension
"""

from llamaforge.eval.benchmarks.func.boolq import eval_boolq
from llamaforge.eval.benchmarks.func.squad import eval_squad

__all__ = ['eval_boolq', 'eval_squad']