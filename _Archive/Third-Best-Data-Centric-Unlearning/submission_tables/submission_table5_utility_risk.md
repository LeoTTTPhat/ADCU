| track | deletion_method | mean_risk_ucb | mean_retain_utility | mean_audit_calls | failure_detection_rate |
| --- | --- | --- | --- | --- | --- |
| FineTuneSim | adapter replacement failure | 0.8379 | 0.965 | 25.33 | 1.0 |
| FineTuneSim | filtered retraining | 0.508 | 0.985 | 25.33 | 0.0 |
| FineTuneSim | gradient-ascent unlearning | 0.7155 | 0.965 | 25.33 | 1.0 |
| FineTuneSim | negative fine-tuning | 0.6303 | 0.955 | 25.33 | 0.8889 |
| FineTuneSim | over-unlearned adapter | 0.508 | 0.42 | 25.33 | 0.0 |
| HybridReal | RAG-only deletion | 0.8772 | 0.98 | 25.33 | 1.0 |
| HybridReal | adapter-only unlearning | 0.8772 | 0.98 | 25.33 | 1.0 |
| HybridReal | full provenance purge | 0.508 | 0.98 | 25.33 | 0.0 |
| HybridReal | graph-pivot expansion retained | 0.8772 | 0.98 | 25.33 | 1.0 |
| HybridReal | synthetic derivative retained | 1.0 | 0.98 | 25.33 | 1.0 |
| NaturalFEVER | answer-only paraphrase leak | 0.8616 | 0.98 | 25.78 | 1.0 |
| NaturalFEVER | citation cache retained | 0.6975 | 0.98 | 25.78 | 1.0 |
| NaturalFEVER | full provenance purge | 0.5068 | 0.98 | 25.78 | 0.0 |
| NaturalFEVER | natural summary retained | 0.6709 | 0.98 | 25.78 | 1.0 |
| RealRAG | backdoor-triggered RAG leakage | 0.7155 | 0.98 | 25.33 | 1.0 |
| RealRAG | cache-not-purged deletion | 0.7155 | 0.98 | 25.33 | 1.0 |
| RealRAG | index-only deletion | 0.7155 | 0.98 | 25.33 | 1.0 |
| RealRAG | provenance-guided deletion | 0.508 | 0.98 | 25.33 | 0.0 |
| RealRAG | shadow-copy deletion | 0.6696 | 0.98 | 25.33 | 1.0 |
