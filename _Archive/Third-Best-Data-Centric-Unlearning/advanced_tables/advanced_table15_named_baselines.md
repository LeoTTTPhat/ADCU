| baseline_type | name | track | failure_detection | false_alarm | retain_utility | mean_risk_ucb | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unlearning_method | filtered retraining | Pretrained-LoRA | 1.000 | -- | 0.167 | 0.672 | 6 |
| unlearning_method | gradient-ascent | LoRA-SFT/Pretrained-LoRA | 1.000 | -- | 0.431 | 0.716 | 9 |
| unlearning_method | negative SFT | LoRA-SFT/Pretrained-LoRA | 1.000 | 1.000 | 0.320 | 0.543 | 9 |
| unlearning_method | NPO | Pretrained-LoRA/Qwen-TOFU-LoRA | 1.000 | -- | 0.231 | 0.556 | 13 |
| unlearning_method | SimNPO | Pretrained-LoRA | 1.000 | -- | 0.167 | 0.549 | 6 |
| audit_method | retriever-only | DenseRAG/LoRA-SFT/PEFT-LoRA | 0.720 | 0.000 | 0.965 | 0.636 | 31 |
| audit_method | membership inference | DenseRAG/PEFT-LoRA/Pretrained-LoRA/Qwen-TOFU-LoRA | 1.000 | -- | 0.446 | 0.699 | 57 |
| audit_method | extraction-only | DenseRAG/PEFT-LoRA/Pretrained-LoRA/Qwen-TOFU-LoRA | 0.632 | -- | 0.446 | 0.518 | 57 |
| audit_method | influence-style | DenseRAG/PEFT-LoRA/Pretrained-LoRA/Qwen-TOFU-LoRA | 1.000 | -- | 0.446 | 0.688 | 57 |
