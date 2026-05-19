# TKDE Submission Implementation Plan

## Current Status

The ADCU track now has:

- A research dossier.
- An executable Python prototype in `src/adcu`.
- RAG, fine-tuning, and hybrid synthetic experiment harnesses.
- A ten-case attack suite.
- JSON/CSV result exports.
- Markdown and LaTeX tables.
- SVG figures.
- A compiling IEEE/TKDE-style manuscript draft in `manuscript_adcu`.

Verified commands:

```bash
python3.11 scripts/run_adcu_main_experiments.py
python3.11 scripts/run_adcu_attack_suite.py
python3.11 scripts/make_adcu_tables.py
python3.11 scripts/make_adcu_figures.py
python3.11 -m pytest -q
cd manuscript_adcu && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Current verification result:

- `9 passed`
- `manuscript_adcu/main.pdf` builds successfully.

## Submission Target

Primary target:

- IEEE Transactions on Knowledge and Data Engineering regular submission.

Conditional target:

- TKDE special issue on data/knowledge-empowered GenAI, only if real RAG and at least one credible fine-tuning experiment are completed before the submission deadline.

## What Is Submission-Ready Now

The following pieces are ready as a scaffold:

- Formal deletion-risk metric.
- ADCU audit algorithm.
- Benchmark schema.
- Attack suite taxonomy.
- Synthetic pilot validating the full pipeline.
- Manuscript skeleton and tables.
- Reproducible scripts.

The following pieces are not yet strong enough for final TKDE submission:

- Real retrieval experiments on public QA corpora.
- Real LoRA/SFT unlearning experiments.
- Larger-scale statistical repeats.
- More complete related work.
- Artifact packaging instructions.

## Phase 1: Real RAG Experiments

Goal:

Replace the synthetic RAG track with a real corpus and retriever while preserving the same ADCU audit interface.

Implementation tasks:

1. Add `src/adcu/rag_real.py`.
2. Create a public QA corpus split using HotpotQA, FEVER, or NQ-style data already present in `data/rag_raw`.
3. Inject synthetic private user records into the corpus.
4. Generate chunks, source IDs, and provenance edges.
5. Implement simple lexical retrieval first.
6. Add dense retrieval if embeddings are available locally.
7. Run deletion variants:
   - index-only deletion,
   - corpus rebuild,
   - cache purge,
   - shadow-copy deletion,
   - provenance-guided deletion.
8. Export results into `experiments/adcu_results/rag_real_*`.

Required table:

- RAG deletion method versus direct leakage, paraphrase leakage, retrieval dependence, retain QA utility, and ADCU risk.

## Phase 2: Fine-Tuning Experiments

Goal:

Add at least one open-weight adapter or lightweight simulator with real training traces.

Implementation tasks:

1. Add `src/adcu/finetune_real.py`.
2. Build synthetic private instruction examples.
3. Add retain instruction examples.
4. Train or simulate adapters:
   - no unlearning,
   - filtered retraining,
   - negative fine-tuning,
   - gradient-ascent unlearning,
   - adapter replacement.
5. Probe deleted facts using direct, paraphrased, extraction, and canary-suppression prompts.
6. Export results into `experiments/adcu_results/finetune_real_*`.

Minimum credible path:

- Use a small local/open model if available.
- If not available, present the deterministic adapter-memory simulator as a controlled benchmark and clearly label it as such.

## Phase 3: Hybrid Derivative-Retention Experiment

Goal:

Show the paper's main novelty: deletion fails when RAG documents are removed but synthetic derivatives or adapters survive.

Implementation tasks:

1. Add `src/adcu/hybrid_real.py`.
2. Generate synthetic QA pairs from injected private RAG documents.
3. Feed those examples into the fine-tuning track.
4. Delete original RAG source only.
5. Compare:
   - RAG-only deletion,
   - adapter-only unlearning,
   - synthetic derivative purge,
   - full provenance purge.
6. Audit all variants with baseline audits and ADCU.

Required figure:

- Provenance graph showing raw record to chunks, embeddings, synthetic QA, adapter, and cache.

## Phase 4: Statistical Repeats And Ablations

Required repeats:

- At least 5 random seeds for synthetic target injection.
- At least 3 deletion-target sizes.
- At least 3 audit budgets.

Required ablations:

- No provenance.
- No valuation.
- No paraphrase probes.
- No retrieval probes.
- No extraction probes.
- Uniform budget allocation.

Key claim to support:

- ADCU improves failure detection per audit call over weak baselines while preserving retain-utility reporting.

## Phase 5: Manuscript Upgrade

Upgrade `manuscript_adcu` from pilot draft to submission draft:

1. Expand Introduction to position deletion verification as TKDE data engineering.
2. Expand Related Work to cover unlearning, RAG provenance, extraction, watermarking, and data valuation.
3. Add system architecture figure.
4. Replace pilot-only statements with real experiment results.
5. Add threat model and assumptions.
6. Add limitations around black-box certification.
7. Add reproducibility appendix.
8. Keep legal language careful: claim measurable technical audit evidence, not legal compliance.

## Final Submission Checklist

- Main PDF builds without LaTeX errors.
- Every result table is generated from scripts.
- All experiment scripts run from a clean checkout.
- All random seeds are logged.
- Model, retriever, embedding model, chunking policy, and cache policy are logged.
- Deletion target construction is documented.
- Attack suite is documented.
- Artifact README includes exact commands.
- Claims distinguish synthetic pilot, controlled benchmark, and real experiments.
- No raw protected synthetic content appears unredacted in paper examples unless explicitly generated for benchmark release.

