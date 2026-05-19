# ADCU-Bench Card

## Purpose

ADCU-Bench evaluates whether deletion-audit methods detect residual dependence on deleted data in RAG, fine-tuned, and hybrid foundation-model systems.

## Benchmark Tracks

1. **RealRAG:** FEVER evidence passages form the retain corpus; synthetic private deletion targets are injected as raw chunks, summaries, synthetic QA records, and caches.
2. **DenseRAG:** TF-IDF, SVD dense retrieval, MiniLM, E5, BGE, and a local cross-encoder-style reranker evaluate whether stronger retrieval changes residual-dependence exposure.
3. **FineTuneSim:** controlled adapter-memory simulator for repeatable unlearning stress tests.
4. **LoRA-SFT:** real PyTorch low-rank adapter trained with SFT-style examples; deletion is evaluated after filtered retraining, gradient-ascent unlearning, and negative SFT.
5. **Pretrained-LoRA:** cached SmolLM2-135M PEFT LoRA causal-LM validation over three seeds with filtered retraining, gradient ascent, negative SFT, NPO, and SimNPO.
6. **Qwen-TOFU-LoRA:** cached Qwen2.5-0.5B PEFT LoRA validation on a compact public TOFU forget/retain split.
7. **TOFU-FullEval:** official TOFU split coverage, full `forget01` evaluation, deterministic bounded Qwen likelihood evaluation on larger retain/auxiliary splits by default, and an explicit exhaustive mode via `ADCU_TOFU_EVAL_LIMIT=full`.
8. **HybridReal:** RAG records create synthetic derivatives that survive in caches, summaries, or adapters after source deletion.

## Audit Methods

- ExactMatch
- CanaryOnly
- RetrieverHit
- MembershipInference
- ExtractionAudit
- InfluenceProxy
- UniformProbes
- ADCU-NoValuation
- ADCU

## Metrics

- Failure detection rate
- False alarm rate
- Direct leakage
- Paraphrase leakage
- Retrieval dependence
- Counterfactual dependence
- Extraction risk
- Watermark hit
- Retain utility
- Retain perplexity and retain completion accuracy for pretrained LoRA
- Top-k deleted-derivative hit rates for retrievers
- Audit calls
- Bootstrap 95% confidence interval for detection rate
- Per-seed variance
- Weight/tolerance sensitivity
- Synthetic labeled scorer validation
- Utility--risk summary with audit-call latency proxy
- Triggered leakage and graph-pivot attack outcomes
- NaturalFEVER deletion audit case study over public FEVER evidence
- Answer-only black-box audit mode
- Retrieval-off counterfactual attribution mode
- Block-bootstrap and score-jitter robustness diagnostics

Optional semantic scorer validation on deployment samples can use
`benchmark/human_adjudication_template.md`.

## Reproducibility Commands

Code repository: https://github.com/LeoTTTPhat/ADCU

```bash
python3.11 scripts/run_adcu_submission_experiments.py
python3.11 scripts/run_adcu_advanced_experiments.py
python3.11 scripts/make_adcu_submission_tables.py
python3.11 scripts/make_adcu_advanced_tables.py
python3.11 -m pytest -q
```

One-command reproduction:

```bash
scripts/reproduce_adcu_artifact.sh
```

Exhaustive Qwen scoring over every example in every official TOFU split is
enabled with:

```bash
ADCU_TOFU_EVAL_LIMIT=full scripts/reproduce_adcu_artifact.sh
```

See `benchmark/model_cache_manifest.md` for exact HuggingFace checkpoint names,
cache expectations, and approximate runtime. See
`benchmark/human_adjudication_template.md` for optional deployment-side
validation of semantic scorer precision and recall. The artifact also includes
`benchmark/model_assisted_adjudication_draft.csv`, a 30-row redacted
NaturalFEVER labeling draft for humans to confirm; it is explicitly not reported
as human adjudication.

## Release Notes

All private records are synthetic. Case-study outputs are redacted by default. The benchmark is designed to test technical audit evidence, not to certify legal compliance.
