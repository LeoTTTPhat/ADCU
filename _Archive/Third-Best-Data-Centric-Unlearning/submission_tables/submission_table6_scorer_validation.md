| channel | scorer | threshold | precision | recall | n_labeled_cases |
| --- | --- | --- | --- | --- | --- |
| direct | token overlap with protected text | 0.4 | 0.5 | 1.0 | 1920 |
| paraphrase | alias/fact/expected-term overlap | 0.35 | 0.25 | 1.0 | 1920 |
| retrieval | derived artifact id or citation hit | 0.2 | 1.0 | 1.0 | 1920 |
| counterfactual | two-sample S_minus versus S_clean distance | 0.2 | 1.0 | 1.0 | 1920 |
| extraction | extraction probe plus semantic/direct hit | 0.08 | 0.75 | 1.0 | 1920 |
| watermark | exact canary/provenance-token hit | 0.2 | 0.6667 | 1.0 | 1920 |
