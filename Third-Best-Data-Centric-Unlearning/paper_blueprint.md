# Paper Blueprint

## Title

Auditable Data-Centric Unlearning in Retrieval-Augmented and Fine-Tuned Foundation Models

## Target Venue

IEEE Transactions on Knowledge and Data Engineering (TKDE)

## Paper Angle

Machine unlearning for foundation-model systems is often framed as an algorithmic model-update problem. In deployed LLM applications, however, user data can influence behavior through many data objects: raw records, document chunks, vector embeddings, retriever indexes, synthetic summaries, prompt caches, fine-tuning examples, adapters, evaluation traces, and downstream logs. This makes deletion a data engineering problem as much as a model training problem.

The paper proposes **Auditable Data-Centric Unlearning (ADCU)**: a framework for measuring and auditing residual behavioral dependence on deleted data in RAG and fine-tuned foundation-model systems.

## One-Sentence Claim

ADCU is the first data-centric audit framework that combines provenance, data valuation, and black-box behavioral testing to quantify deletion risk across RAG, fine-tuning, and hybrid foundation-model pipelines.

## Research Questions

1. How can deletion risk be defined in a way that covers both retrieval-time data dependence and model-parameter dependence?
2. Can provenance and pre-deletion data valuation identify which deletion targets require stronger audit coverage?
3. Can black-box behavioral tests detect residual dependence even when retriever logs, model weights, or training details are unavailable?
4. Which common unlearning claims fail under adversarial paraphrase, shadow-copy, chunk-boundary, adapter, and evaluation-evasion attacks?
5. How much audit cost can be saved by valuation-guided sequential testing compared with exhaustive probe suites?

## System Model

The system is a foundation-model application with one or more data-bearing components:

- **RAG corpus:** source documents, chunks, metadata, embeddings, and indexes.
- **Fine-tuning corpus:** instruction records, preference pairs, domain records, or synthetic derivatives.
- **Adaptation artifacts:** LoRA adapters, merged weights, prompt memories, caches, rerankers, and policy filters.
- **Operational traces:** retrieved contexts, generated answers, query logs, evaluation examples, synthetic augmentations, and monitoring samples.

A deletion request targets a set of user-owned data units `D_del`. The post-deletion system `S_minus` should behave as if the protected contribution of `D_del` has been removed while preserving utility on a retain distribution `Q_ret`.

## Proposed Contributions

1. **Deletion-risk metric.** A formal metric that combines direct leakage, paraphrased leakage, retrieval dependence, counterfactual answer dependence, and extraction risk into a calibrated audit score.
2. **ADCU audit algorithm.** An efficient algorithm that uses provenance metadata and data valuation to select high-risk probes, then runs sequential black-box tests until risk is certified below a threshold or a violation is found.
3. **ADCU-Bench.** A benchmark for RAG, fine-tuning, and hybrid unlearning with deletion targets, retain tasks, provenance labels, canaries, paraphrase families, and attack scenarios.
4. **Attack suite.** A red-team suite showing that naive deletion, retriever-only removal, adapter-only unlearning, and surface-form memorization checks can all pass weak audits while retaining deleted-data behavior.
5. **Deployment guidance.** Practical recommendations for data registries, provenance ledgers, watermark/canary design, audit logs, and cost-bounded deletion verification.

## Novelty Wedge

The paper should avoid competing only as "another unlearning algorithm." The stronger TKDE wedge is **auditable data operations for foundation-model systems**:

- It evaluates the full data pipeline, not only model weights.
- It treats unlearning as a measurable compliance workflow.
- It connects provenance, valuation, watermarking, and black-box tests.
- It produces an operational benchmark and attack suite.

## Key Definitions

- **Deletion target:** a record, document, chunk, source, client, or derived artifact subject to deletion.
- **Residual dependence:** post-deletion behavior that remains predictably influenced by the deletion target.
- **Audit probe:** a query designed to expose direct recall, paraphrased recall, retrieval dependence, counterfactual answer dependence, or extraction risk.
- **Deletion risk:** a calibrated probability or upper confidence bound that residual dependence exceeds an acceptable threshold.
- **Retain utility:** performance on non-deleted data and tasks that should remain stable after unlearning.

## Main Method Sketch

1. Build a provenance graph from raw user records to chunks, embeddings, synthetic derivatives, fine-tuning records, adapters, caches, and evaluation traces.
2. Estimate pre-deletion operational value for each deletion target using retrieval frequency, answer dependence, influence proxies, Shapley-style marginal utility, or cached counterfactual tests.
3. Generate audit probes from high-risk deletion targets:
   - direct probes,
   - paraphrase probes,
   - semantic neighbor probes,
   - bridge/multi-hop probes,
   - extraction probes,
   - retrieval-only probes.
4. Run the post-deletion system as a black box and collect behavioral evidence.
5. Compute deletion risk with uncertainty bounds.
6. Certify, reject, or escalate to stronger unlearning.

## Experimental Claims To Prove

1. ADCU detects residual deleted-data behavior that standard exact-match, membership, and retriever-hit audits miss.
2. Valuation-guided auditing achieves higher violation detection per model call than random or uniform probe sampling.
3. RAG-only deletion is insufficient when deleted data has been distilled into summaries, synthetic records, or fine-tuning artifacts.
4. Fine-tuning-only unlearning is insufficient when deleted content remains in vector indexes, caches, or retrieval memories.
5. ADCU can provide statistically meaningful audit certificates while preserving retain utility reporting.

## Paper Outline

1. **Introduction**
   - Deletion requests are increasingly relevant for LLM systems.
   - RAG and fine-tuning create multi-channel data dependence.
   - Existing unlearning evaluations are too narrow.
   - ADCU provides measurable, auditable deletion risk.

2. **Background and Related Work**
   - Machine unlearning.
   - RAG provenance and source attribution.
   - Data valuation and influence estimation.
   - Watermarking, canaries, membership inference, and extraction.
   - Compliance and right-to-be-forgotten workflows.

3. **Problem Formulation**
   - Data-bearing pipeline model.
   - Deletion request semantics.
   - Residual dependence.
   - Retain utility and audit cost.

4. **Deletion-Risk Metric**
   - Behavioral channels.
   - Risk aggregation.
   - Statistical confidence.
   - Certification objective.

5. **ADCU Audit Algorithm**
   - Provenance graph.
   - Valuation-guided probe allocation.
   - Probe generation.
   - Sequential black-box testing.
   - Audit report.

6. **Benchmark**
   - RAG tasks.
   - Fine-tuning tasks.
   - Hybrid tasks.
   - Deletion targets, retain sets, and provenance labels.
   - Baselines and metrics.

7. **Experiments**
   - Main detection results.
   - Cost-quality tradeoff.
   - Attack suite.
   - Ablations.
   - Case studies.

8. **Discussion**
   - Limits of black-box audits.
   - Privacy-preserving audit design.
   - Legal and operational interpretation.
   - Deployment recommendations.

9. **Conclusion**

## Risk And Mitigation

- **Risk: legal overclaiming.** Avoid claiming legal compliance. Claim measurable technical evidence and auditability.
- **Risk: black-box tests cannot prove absence.** Use statistical certification with explicit assumptions and confidence bounds.
- **Risk: too broad.** Keep the benchmark centered on two concrete pipelines: RAG deletion and LoRA/SFT unlearning, with hybrid as the key stress test.
- **Risk: valuation novelty overlap.** Position valuation as audit allocation and risk prioritization, not the sole contribution.

