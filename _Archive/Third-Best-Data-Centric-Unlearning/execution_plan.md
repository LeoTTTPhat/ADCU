# Execution Plan

## Minimal Viable Paper

The fastest credible TKDE version should focus on:

1. RAG corpus deletion.
2. Fine-tuning record unlearning with LoRA/SFT.
3. One hybrid pipeline where deleted RAG documents survive through synthetic derivatives or adapters.
4. ADCU deletion-risk metric.
5. Valuation-guided black-box audit.
6. Attack suite against weak audits.

Avoid trying to solve every form of machine unlearning. The paper wins by making deletion auditing measurable across realistic foundation-model data pipelines.

## 10-Week Plan

### Week 1: Finalize Formulation

- Lock system model, deletion target taxonomy, and risk metric.
- Decide default evidence scorers.
- Define pass, fail, and escalate semantics.
- Draft Sections 1-3.

### Week 2: Build RAG Deletion Harness

- Implement document ingestion, chunking, embeddings, vector index, retrieval, and generation.
- Inject private synthetic user profiles into HotpotQA/FEVER/NQ-style corpora.
- Build deletion operation variants: index-only, corpus rebuild, cache purge, provenance purge.

### Week 3: Build Fine-Tuning Harness

- Create synthetic private instruction records with canaries and paraphrase families.
- Fine-tune a small open model or adapter.
- Implement unlearning baselines: no unlearning, filtered retraining, gradient ascent, negative fine-tuning, adapter replacement.

### Week 4: Build Hybrid Pipeline

- Generate synthetic QA derivatives from private RAG records.
- Fine-tune adapter on derivatives.
- Delete original source and test whether behavior survives through the adapter.
- Add provenance graph linking raw documents to derived artifacts.

### Week 5: Implement ADCU-Audit

- Implement probe generation.
- Implement lexical, embedding, NLI/semantic, retrieval, and canary scorers.
- Implement risk aggregation and confidence bounds.
- Implement sequential budget allocation.

### Week 6: Baseline Audits

- Exact-match audit.
- Canary audit.
- Retriever-hit audit.
- Membership-style prompt audit.
- Random probe audit.
- Uniform paraphrase audit.
- Exhaustive probe upper bound.

### Week 7: Attack Suite

- Implement paraphrase survival, chunk-boundary leakage, shadow-copy retrieval, synthetic derivative retention, adapter merge residue, evaluation evasion, retriever-reranker disagreement, retain-neighbor over-unlearning, multi-hop reconstruction, and watermark suppression.

### Week 8: Main Experiments

- Run RAG, fine-tuning, and hybrid tracks.
- Produce detection AUROC, deletion risk, retain utility, and cost-normalized detection.
- Run ablations: no valuation, no provenance, no paraphrases, no sequential stopping.

### Week 9: Analysis And Figures

- Generate risk-versus-cost curves.
- Generate channel-wise residual dependence heatmaps.
- Write case studies.
- Prepare failure examples with protected content redacted.

### Week 10: Manuscript Draft

- Complete full TKDE-style draft.
- Tighten claims around auditability rather than legal compliance.
- Add reproducibility checklist and benchmark card.
- Prepare artifact package.

## Implementation Skeleton

Recommended folder expansion if this becomes executable:

```text
adcu/
  provenance.py
  probes.py
  risk.py
  audit.py
  scorers.py
  rag_harness.py
  finetune_harness.py
scripts/
  run_adcu_rag_smoke.py
  run_adcu_finetune_smoke.py
  run_adcu_hybrid_attack.py
experiments/
  adcu_results/
benchmark/
  adcu_protocol.md
```

## First Pilot

The best first pilot is synthetic but realistic:

- Create 100 user profiles with private attributes.
- Insert them into a small RAG corpus.
- Create 300 synthetic QA pairs from 30 profiles.
- Fine-tune a small adapter on the QA pairs.
- Delete 10 profiles.
- Compare:
  - RAG index deletion only,
  - adapter unlearning only,
  - full provenance deletion,
  - filtered retraining.
- Audit with:
  - exact match,
  - retriever hit,
  - random probes,
  - ADCU-Audit.

Expected result:

- Index deletion passes retriever checks but fails hybrid probes.
- Adapter-only unlearning passes model probes but fails RAG retrieval.
- Exact-match checks miss paraphrase leakage.
- ADCU-Audit detects failures with fewer calls than uniform probing.

## Acceptance Strategy

Lead with the data engineering story:

- Real LLM applications are not a single model.
- Data deletion requires tracking and auditing a graph of artifacts.
- Current unlearning evaluations are too model-centric and too weak behaviorally.
- ADCU supplies the missing operational layer: provenance, valuation-guided auditing, risk certification, and attacks.

## Potential Paper Positioning

This can be positioned as a sibling or successor to DataVal-FM:

- DataVal-FM asks: which data helps, harms, or matters?
- ADCU asks: after deletion, does formerly important data still matter?

The connection is strong, but the paper should be separate because the metric, audit workflow, benchmark, and attacks are centered on deletion verification.

