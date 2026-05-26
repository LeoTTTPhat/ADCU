| track | data | model | retriever | deletion_targets | probes | seeds | runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RealRAG | FEVER retain + synthetic private targets | black-box QA simulator | lexical/SVD | 1/3/5 | 12/24/48 | 5 | ~8 min CPU |
| NaturalFEVER | public FEVER evidence deletions | black-box QA simulator | lexical | 1/3/5 | 12/24/48 | 5 | ~5 min CPU |
| LoRA-SFT | synthetic SFT records | controlled low-rank adapter | none | 2 | 32 | 2 | ~2 min CPU |
| Pretrained-LoRA | synthetic private completions | SmolLM2-135M/Qwen2.5-0.5B PEFT | none | 1 | 24 | 3 per model | ~30-90 min CPU |
| Qwen-TOFU-LoRA | TOFU forget01/retain99 | Qwen2.5-0.5B PEFT | none | 2 | 24 | 1 | ~25 min CPU |
| DenseRAG | FEVER retain + derivatives | black-box QA simulator | BM25/SVD/MiniLM/E5/BGE/cross | 2 | 24 | 3 | ~10 min CPU |
