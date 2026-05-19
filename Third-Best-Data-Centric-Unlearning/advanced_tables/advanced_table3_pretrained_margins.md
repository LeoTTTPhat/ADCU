| track | method | active_forget_targets | mean_forget_margin | retain_perplexity | retain_completion_accuracy | adcu_decision | failure_detection_rate | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained-LoRA | pretrained LoRA NPO unlearning | 0.0 | -2.0091 | 72.2828 | 0.0 | escalate | 1.0 | 3 |
| Pretrained-LoRA | pretrained LoRA SimNPO unlearning | 0.0 | -1.9647 | 71.5352 | 0.0 | escalate | 1.0 | 3 |
| Pretrained-LoRA | pretrained LoRA filtered retraining | 0.0 | -0.9395 | 55.9968 | 0.0 | escalate | 1.0 | 3 |
| Pretrained-LoRA | pretrained LoRA full SFT | 0.3333 | -0.8107 | 63.1552 | 0.0 | escalate/fail | 1.0 | 3 |
| Pretrained-LoRA | pretrained LoRA gradient-ascent unlearning | 0.0 | -1.6783 | 127.0763 | 0.0 | escalate | 1.0 | 3 |
| Pretrained-LoRA | pretrained LoRA negative SFT unlearning | 0.0 | -2.3428 | 64.688 | 0.0 | escalate | 1.0 | 3 |
| Qwen-TOFU-LoRA | Qwen2.5-0.5B TOFU-like NPO unlearning | 1.0 | -0.4114 | 8.5194 | 1.0 | fail | 1.0 | 1 |
| Qwen-TOFU-LoRA | Qwen2.5-0.5B TOFU-like full SFT | 1.0 | -0.1918 | 8.9517 | 1.0 | fail | 1.0 | 1 |
