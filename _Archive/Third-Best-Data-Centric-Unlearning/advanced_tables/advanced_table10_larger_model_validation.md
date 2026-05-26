| model | method | forget_risk_before | forget_risk_after | clean_counterfactual_distance | retain_utility | adcu_decision | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SmolLM2-135M (synthetic) | NPO | 0.929 | 0.681 | 0.500 | 0.333 | escalate/fail | 3 |
| SmolLM2-135M (synthetic) | SimNPO | 0.929 | 0.681 | 0.500 | 0.333 | escalate/fail | 3 |
| SmolLM2-135M (synthetic) | filtered retraining | 0.929 | 0.929 | 1.000 | 0.333 | fail | 3 |
| SmolLM2-135M (synthetic) | full SFT | 0.929 | 0.929 | 1.000 | 0.333 | fail | 3 |
| SmolLM2-135M (synthetic) | gradient-ascent | 0.929 | 0.852 | 0.833 | 0.333 | fail | 3 |
| SmolLM2-135M (synthetic) | negative SFT | 0.929 | 0.587 | 0.333 | 0.000 | escalate/fail | 3 |
| Qwen2.5-0.5B (synthetic) | NPO | 0.511 | 0.416 | 0.000 | 0.000 | escalate | 3 |
| Qwen2.5-0.5B (synthetic) | SimNPO | 0.511 | 0.416 | 0.000 | 0.000 | escalate | 3 |
| Qwen2.5-0.5B (synthetic) | filtered retraining | 0.511 | 0.416 | 0.000 | 0.000 | escalate | 3 |
| Qwen2.5-0.5B (synthetic) | full SFT | 0.511 | 0.511 | 0.167 | 0.000 | escalate/fail | 3 |
| Qwen2.5-0.5B (synthetic) | gradient-ascent | 0.511 | 0.416 | 0.000 | 0.000 | escalate | 3 |
| Qwen2.5-0.5B (synthetic) | negative SFT | 0.511 | 0.416 | 0.000 | 0.000 | escalate | 3 |
| Qwen2.5-0.5B (TOFU) | TOFU NPO | 0.643 | 0.643 | 0.500 | 1.000 | fail | 1 |
| Qwen2.5-0.5B (TOFU) | TOFU full SFT | 0.643 | 0.643 | 0.500 | 1.000 | fail | 1 |
