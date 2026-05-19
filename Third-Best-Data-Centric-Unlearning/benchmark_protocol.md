# ADCU-Bench Protocol

## Purpose

ADCU-Bench evaluates whether an audit method can detect residual dependence on deleted data while preserving retain utility in RAG, fine-tuned, and hybrid foundation-model systems.

## Benchmark Principles

- Deletion targets must have known provenance.
- Some targets should be high influence and some low influence.
- The benchmark must include direct and derived data artifacts.
- Audits must evaluate more than exact string memorization.
- Retain utility must be reported with deletion risk.
- Attack scenarios must be part of the benchmark, not an afterthought.

## Tracks

### Track 1: RAG Corpus Deletion

Data unit:

- Document, chunk, source, user profile, or private passage.

System:

- Retriever plus generator.
- Optional reranker and citation module.

Deletion operation:

- Remove target chunks and embeddings from the index.
- Remove source metadata and cached retrieval traces.

Audit goal:

- Detect whether deleted facts are still retrievable, cited, paraphrased, or used in generated answers through shadow copies, summaries, neighboring chunks, or cached contexts.

Candidate datasets:

- Natural Questions Open.
- HotpotQA.
- FEVER.
- Synthetic private-user profiles injected into public corpora.
- Domain-specific mini-corpora in legal, medical, or finance QA.

### Track 2: Fine-Tuning Record Unlearning

Data unit:

- Instruction example, preference pair, user-specific record, or canary-bearing training example.

System:

- Small open model with LoRA/SFT adaptation.

Deletion operation:

- Adapter unlearning, gradient ascent on forget data, fine-tuning from filtered data, model editing, or adapter replacement.

Audit goal:

- Detect whether deleted information remains in model behavior under direct, paraphrased, and extraction prompts.

Candidate datasets:

- Dolly/Alpaca-style instruction subsets.
- Synthetic user-specific records.
- Preference pairs with private labels.
- Domain adaptation records with inserted canaries.

### Track 3: Hybrid RAG + Fine-Tuning

Data unit:

- User document used both in RAG and in synthetic instruction generation or fine-tuning.

System:

- Corpus ingestion creates chunks, summaries, embeddings, and synthetic QA pairs.
- Synthetic QA pairs fine-tune an adapter.

Deletion operation:

- Remove corpus artifacts and run an unlearning method on fine-tuned artifacts.

Audit goal:

- Detect cross-channel leakage where the raw source is deleted from the index but its derivative survives in the adapter, cache, summary, or synthetic data.

This is the main stress test and should be highlighted as the paper's most important empirical setting.

## Deletion Target Construction

Each target should include:

- Raw protected record.
- Paraphrase set.
- Entity and relation annotations.
- Canary or watermark token when appropriate.
- Derived chunks.
- Embedding IDs.
- Synthetic QA derivatives.
- Fine-tuning record IDs.
- Known answer facts.
- Retain-neighbor records that should not be deleted.

## Evaluation Metrics

### Deletion Risk

- Direct leakage rate.
- Paraphrased leakage rate.
- Deleted retrieval hit rate.
- Derived artifact hit rate.
- Counterfactual answer dependence.
- Extraction success rate.
- Watermark/canary hit rate.
- ADCU risk score and upper confidence bound.

### Audit Quality

- Violation detection AUROC.
- Precision/recall for failed deletion targets.
- Time to first violation.
- Detection per 1,000 model calls.
- False pass rate under controlled ground truth.
- False escalation rate.

### Retain Utility

- Exact match, F1, accuracy, or task-specific utility.
- Faithfulness and context precision for RAG.
- Helpfulness or instruction-following score for fine-tuned models.
- Safety/refusal behavior where relevant.
- Utility degradation relative to pre-deletion system.

### Efficiency

- Model calls.
- Retriever calls.
- LLM-as-judge calls.
- Wall-clock time.
- GPU hours.
- Index rebuild time.
- Cost-normalized detection.

## Baseline Unlearning Methods

### RAG

- Delete from vector index only.
- Delete from raw corpus and rebuild index.
- Delete plus cache purge.
- Delete plus synthetic derivative purge.
- Delete plus provenance-graph purge.

### Fine-Tuning

- No unlearning.
- Filtered retraining.
- LoRA adapter retraining from filtered data.
- Gradient ascent on forget set.
- Negative preference fine-tuning.
- Model editing.
- Adapter replacement.

### Hybrid

- RAG-only deletion.
- Fine-tuning-only unlearning.
- Naive artifact deletion.
- Full provenance-guided deletion.

## Baseline Audits

- Exact string match.
- Canary-only check.
- Retriever hit check.
- Membership inference.
- Random probe suite.
- Uniform paraphrase suite.
- Exhaustive probe suite.
- ADCU-Audit.

## Required Tables

### Table 1: Main Audit Detection

Rows: audit methods.

Columns: RAG, fine-tuning, hybrid, average detection AUROC, detection per 1,000 calls.

### Table 2: Residual Dependence By Channel

Rows: unlearning methods.

Columns: direct leakage, paraphrase leakage, retrieval dependence, derivative dependence, extraction risk, ADCU risk.

### Table 3: Retain Utility

Rows: unlearning methods.

Columns: retain quality, forget risk, utility-risk tradeoff, cost.

### Table 4: Attack Suite

Rows: attacks.

Columns: exact-match audit, retriever audit, membership audit, ADCU-Audit.

## Required Figures

1. Data provenance graph for a deletion target.
2. ADCU audit pipeline.
3. Risk-versus-cost curve.
4. Channel-wise residual dependence heatmap.
5. Case study showing a deletion target surviving through synthetic derivative or adapter memory.

## Reporting Rules

- Report model name, adapter method, retriever, embedding model, index type, chunking policy, and cache policy.
- Report all deletion artifacts: raw record, chunks, embeddings, summaries, synthetic examples, adapters, and caches.
- Report whether the auditor has black-box, gray-box, or white-box access.
- Report statistical confidence for deletion-risk scores.
- Report retain utility alongside every deletion-risk claim.
- Never claim legal compliance; claim measured technical audit evidence.

