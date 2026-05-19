# ADCU-Bench Model Cache Manifest

This artifact is designed to run without silent model downloads. The advanced
experiments use the following HuggingFace checkpoints when they are present in
the local cache.

| Component | Checkpoint | Required for |
| --- | --- | --- |
| PEFT sequence-classification validation | `hf-internal-testing/tiny-random-distilbert` | `PEFT-LoRA` |
| Pretrained causal-LM LoRA | `HuggingFaceTB/SmolLM2-135M-Instruct` | `Pretrained-LoRA` |
| Sentence-transformer retrieval | `sentence-transformers/all-MiniLM-L6-v2` | `DenseRAG/neural` |
| E5 retrieval | `intfloat/e5-base-v2` | `DenseRAG/e5` |
| BGE retrieval | `BAAI/bge-small-en-v1.5` | `DenseRAG/bge` |
| Larger pretrained public-forget LoRA | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen-TOFU-LoRA` |
| Public forget/retain benchmark | `locuslab/TOFU` official configs | `Qwen-TOFU-LoRA`, `TOFU-FullEval` |

The local machine used for the current run has safetensors weights for
`HuggingFaceTB/SmolLM2-135M-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`,
`intfloat/e5-base-v2`, and `BAAI/bge-small-en-v1.5`. TinyLlama contained
tokenizer/config cache entries but no local safetensors weights, so it was not
used for the executed larger-checkpoint run.

Expected runtime on an Apple Silicon laptop CPU is roughly:

| Step | Approximate runtime |
| --- | ---: |
| Submission experiments and tables | 1-2 minutes |
| Attack suite | under 1 minute |
| Advanced LoRA, Qwen-TOFU, default TOFU split evaluation, and DenseRAG experiments | 10-14 minutes |
| Exhaustive Qwen likelihood scoring for every official TOFU example with `ADCU_TOFU_EVAL_LIMIT=full` | hours on CPU; use GPU for submission-scale runs |
| Tests and PDF build | under 1 minute |

Runtime varies with the HuggingFace cache state and BLAS/PyTorch backend.
