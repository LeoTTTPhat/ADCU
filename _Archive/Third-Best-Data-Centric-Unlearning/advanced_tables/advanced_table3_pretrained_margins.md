| track | method | active_forget_targets | mean_forget_margin | retain_perplexity | retain_completion_accuracy | adcu_decision | failure_detection_rate | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained-LoRA | pretrained LoRA NPO unlearning | 0.3333 | -1.1573 | 77.8779 | 0.1667 | escalate/fail | 1.0 | 6 |
| Pretrained-LoRA | pretrained LoRA SimNPO unlearning | 0.3333 | -1.1361 | 77.5389 | 0.1667 | escalate/fail | 1.0 | 6 |
| Pretrained-LoRA | pretrained LoRA filtered retraining | 0.5 | -0.495 | 69.0911 | 0.1667 | escalate/fail | 1.0 | 6 |
| Pretrained-LoRA | pretrained LoRA full SFT | 0.6667 | -0.4245 | 73.9102 | 0.1667 | escalate/fail | 1.0 | 6 |
| Pretrained-LoRA | pretrained LoRA gradient-ascent unlearning | 0.5 | -0.9501 | 111.1017 | 0.1667 | escalate/fail | 1.0 | 6 |
| Pretrained-LoRA | pretrained LoRA negative SFT unlearning | 0.1667 | -1.4597 | 72.9715 | 0.0 | escalate/fail | 1.0 | 6 |
| Qwen-TOFU-LoRA | Qwen2.5-0.5B TOFU-like NPO unlearning | 1.0 | -0.4114 | 8.5194 | 1.0 | fail | 1.0 | 1 |
| Qwen-TOFU-LoRA | Qwen2.5-0.5B TOFU-like full SFT | 1.0 | -0.1918 | 8.9517 | 1.0 | fail | 1.0 | 1 |
