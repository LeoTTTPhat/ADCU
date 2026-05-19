# Auditable Data-Centric Unlearning for LLM/RAG Systems

Code repository: https://github.com/LeoTTTPhat/ADCU

## Working Title

**Auditable Data-Centric Unlearning in Retrieval-Augmented and Fine-Tuned Foundation Models**

## One-Sentence Thesis

Deletion compliance for LLM systems should be evaluated as a data-centric, auditable behavior-removal problem: after a user deletion request, the system should provide measurable evidence that the deleted data no longer contributes to retrieval, generation, fine-tuned behavior, or downstream answers.

## Why This Is Timely

TKDE has recently published challenge-level work on federated foundation models that explicitly identifies private data use, continual learning, unlearning, watermarking, and efficiency as important open problems. This project turns those themes into a concrete data engineering contribution for RAG and fine-tuned foundation-model systems: a deletion-risk metric, an efficient black-box audit algorithm, a benchmark, and an attack suite.

Reference anchor:

- Fan et al., "Ten Challenging Problems in Federated Foundation Models," IEEE TKDE, 37(7):4314-4337, 2025. DOI: 10.1109/TKDE.2025.3555328.

## Core Question

After a deletion request, how do we know the contribution of a user's data is really removed from downstream behavior?

This project answers the question by combining:

- **Provenance and watermarking:** know which outputs, retrieved chunks, embeddings, adapters, prompts, and generated synthetic records may be causally downstream of deleted data.
- **Data valuation:** estimate which deleted records had high operational influence before deletion and therefore deserve stronger audit coverage.
- **Black-box behavioral tests:** probe whether the post-deletion system still reproduces, retrieves, paraphrases, relies on, or semantically leaks deleted information.

## Contribution Package

1. **Formal deletion-risk metric:** a unified score that measures residual behavioral dependence on deletion-target data across RAG, fine-tuning, and hybrid pipelines.
2. **Efficient audit algorithm:** valuation-guided probe generation and sequential black-box testing that reduces audit cost while preserving statistical confidence.
3. **RAG/fine-tuning/unlearning benchmark:** standardized deletion targets, provenance labels, canaries, utility tasks, retain sets, and attack scenarios.
4. **Attack suite:** systematic demonstrations of where existing unlearning claims fail under paraphrase, retriever, chunking, shadow-copy, adapter, and evaluation-evasion attacks.

## Files

- [paper_blueprint.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/paper_blueprint.md): manuscript-ready research framing.
- [formal_metric.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/formal_metric.md): deletion-risk metric and statistical audit objective.
- [audit_algorithm.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/audit_algorithm.md): proposed efficient auditing method.
- [benchmark_protocol.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/benchmark_protocol.md): benchmark design for RAG, fine-tuning, and hybrid systems.
- [attack_suite.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/attack_suite.md): failure modes and red-team tests.
- [execution_plan.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/execution_plan.md): 10-week execution plan and minimal viable paper.
- [references.bib](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/references.bib): seed bibliography.

## Executable Prototype

The idea is also implemented as a lightweight Python package in [src/adcu](/Users/phatttt/Documents/Claude/Projects/TKDE-03/src/adcu):

- `data.py`: deletion targets, artifacts, probes, responses, and evidence vectors.
- `provenance.py`: target-to-artifact provenance graph.
- `probes.py`: direct, paraphrase, semantic, retrieval, bridge, and extraction probe generation.
- `scorers.py`: direct leakage, paraphrase leakage, retrieval dependence, counterfactual, extraction, and watermark scoring.
- `risk.py`: weighted deletion-risk score, valuation priorities, and Hoeffding upper confidence bounds.
- `audit.py`: ADCU black-box auditor with pass/fail/escalate reports.
- `benchmark.py`: deterministic synthetic RAG/fine-tuning/hybrid deletion scenario.
- `attack_suite.py`: twelve attack cases from the research plan, including triggered RAG leakage and graph-pivot reconstruction.

Run the full synthetic audit and attack suite:

```bash
python3.11 scripts/run_adcu_all.py
```

Run individual smoke tracks:

```bash
python3.11 scripts/run_adcu_rag_smoke.py
python3.11 scripts/run_adcu_finetune_smoke.py
python3.11 scripts/run_adcu_hybrid_attack.py
```

Run tests:

```bash
python3.11 -m pytest tests/test_adcu_audit.py -q
python3.11 -m pytest -q
```

Verified result in this workspace: `9 passed`.

## TKDE Submission Draft

The current TKDE-style manuscript draft is in [manuscript_adcu](/Users/phatttt/Documents/Claude/Projects/TKDE-03/manuscript_adcu).

Build it with:

```bash
cd manuscript_adcu
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Generated submission-support artifacts:

- [experiments/adcu_results](/Users/phatttt/Documents/Claude/Projects/TKDE-03/experiments/adcu_results): JSON/CSV experiment outputs.
- [experiments/adcu_submission_results](/Users/phatttt/Documents/Claude/Projects/TKDE-03/experiments/adcu_submission_results): repeated submission-scale outputs over five seeds, three target sizes, and three audit budgets.
- [tables](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/tables): Markdown and LaTeX tables.
- [submission_tables](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/submission_tables): repeated-experiment Markdown and LaTeX tables.
- [figures/adcu](/Users/phatttt/Documents/Claude/Projects/TKDE-03/figures/adcu): SVG figures.
- [figures/adcu_submission](/Users/phatttt/Documents/Claude/Projects/TKDE-03/figures/adcu_submission): repeated-experiment SVG figures.
- [submission_implementation_plan.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/submission_implementation_plan.md): remaining work to reach a full TKDE-ready submission.

Run the repeated submission-scale experiments:

```bash
python3.11 scripts/run_adcu_submission_experiments.py
python3.11 scripts/make_adcu_submission_tables.py
python3.11 scripts/make_adcu_submission_figures.py
python3.11 scripts/run_adcu_advanced_experiments.py
python3.11 scripts/make_adcu_advanced_tables.py
```

Advanced artifacts:

- [experiments/adcu_advanced_results](/Users/phatttt/Documents/Claude/Projects/TKDE-03/experiments/adcu_advanced_results): LoRA-SFT and DenseRAG JSON/CSV outputs.
- [advanced_tables](/Users/phatttt/Documents/Claude/Projects/TKDE-03/Third-Best-Data-Centric-Unlearning/advanced_tables): bootstrap confidence, pretrained-margin, retriever-hit, probe-ablation, per-seed, and redacted case-study tables.
- [benchmark/adcu_bench_card.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/benchmark/adcu_bench_card.md): reusable benchmark card.
- [benchmark/model_cache_manifest.md](/Users/phatttt/Documents/Claude/Projects/TKDE-03/benchmark/model_cache_manifest.md): model-cache and runtime manifest.

One-command reproduction:

```bash
scripts/reproduce_adcu_artifact.sh
```
