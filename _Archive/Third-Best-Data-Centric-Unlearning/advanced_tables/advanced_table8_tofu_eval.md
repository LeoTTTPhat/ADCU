| method | tofu_split | split_size | eval_n | eval_mode | answer_preferred_rate | mean_answer_margin | mean_answer_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B TOFU-like NPO unlearning | forget01 | 40 | 40 | full | 1.0 | 2.6162 | 2.0098 |
| Qwen2.5-0.5B TOFU-like NPO unlearning | forget01_perturbed | 40 | 5 | deterministic_subset_5_of_40 | 1.0 | 3.5929 | 0.8516 |
| Qwen2.5-0.5B TOFU-like NPO unlearning | real_authors | 100 | 5 | deterministic_subset_5_of_100 | 1.0 | 0.0 | 0.5613 |
| Qwen2.5-0.5B TOFU-like NPO unlearning | retain99 | 3960 | 5 | deterministic_subset_5_of_3960 | 1.0 | 1.8546 | 2.2326 |
| Qwen2.5-0.5B TOFU-like NPO unlearning | retain_perturbed | 400 | 5 | deterministic_subset_5_of_400 | 1.0 | 1.8546 | 2.2326 |
| Qwen2.5-0.5B TOFU-like NPO unlearning | world_facts | 117 | 5 | deterministic_subset_5_of_117 | 0.4 | -2.0521 | 4.854 |
| Qwen2.5-0.5B TOFU-like full SFT | forget01 | 40 | 40 | full | 1.0 | 3.4098 | 2.0209 |
| Qwen2.5-0.5B TOFU-like full SFT | forget01_perturbed | 40 | 5 | deterministic_subset_5_of_40 | 1.0 | 4.4914 | 0.8811 |
| Qwen2.5-0.5B TOFU-like full SFT | real_authors | 100 | 5 | deterministic_subset_5_of_100 | 1.0 | 0.0 | 0.4016 |
| Qwen2.5-0.5B TOFU-like full SFT | retain99 | 3960 | 5 | deterministic_subset_5_of_3960 | 1.0 | 2.4063 | 2.2736 |
| Qwen2.5-0.5B TOFU-like full SFT | retain_perturbed | 400 | 5 | deterministic_subset_5_of_400 | 1.0 | 2.4063 | 2.2736 |
| Qwen2.5-0.5B TOFU-like full SFT | world_facts | 117 | 5 | deterministic_subset_5_of_117 | 0.4 | -1.8328 | 4.3498 |
