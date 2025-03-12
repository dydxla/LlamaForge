# LlamaForge
Scale LLMs fine-tuning (with LoRA, FP16 optimization, and DeepSpeed)—making large-scale language models accessible and efficient for multi-GPU setups.

In addition, it supports benchmark index evaluation not only for open LLM models but also for individual LLM models and fine-tuned models.


## Finetune
-------------
```python
from llamaforge.finetune import FinetuneTrainer
trainer = FinetuneTrainer()
trainer.run_finetune()
```