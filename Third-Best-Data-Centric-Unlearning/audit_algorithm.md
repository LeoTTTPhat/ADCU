# ADCU Audit Algorithm

## Name

**ADCU-Audit:** Auditable Data-Centric Unlearning Audit

## Inputs

- Pre-deletion system `S_0`, when available.
- Post-deletion system `S_-`.
- Deletion targets `D_del`.
- Retain query set `Q_ret`.
- Provenance graph `G`.
- Probe generator `P`.
- Evidence scorers `E`.
- Risk tolerance `tau`.
- Confidence parameter `delta`.
- Audit budget `B`.

## Outputs

- Pass, fail, or escalate decision.
- Per-target deletion-risk scores.
- Aggregate valuation-weighted deletion risk.
- Retain utility report.
- Evidence table with violating probes.
- Provenance paths that explain likely residual dependence.

## Algorithm

1. **Construct or import provenance graph.**
   - Link raw records to chunks, embeddings, summaries, synthetic examples, adapters, caches, prompts, retrieval logs, generated answers, and evaluation records.
   - Mark direct and derived artifacts for each deletion target.

2. **Estimate pre-deletion influence.**
   - For RAG: use retrieval frequency, context inclusion, answer-support attribution, citation frequency, and leave-source-out utility deltas.
   - For fine-tuning: use loss, gradient/influence proxies, memorization canaries, validation-slice utility deltas, and nearest-neighbor clusters.
   - For hybrid systems: propagate influence through the provenance graph.

3. **Allocate audit budget by risk.**
   - Assign each target a priority `pi(z)`.
   - Reserve a minimum probe budget for all targets.
   - Allocate extra probes to high-value, highly connected, or high-sensitivity targets.

4. **Generate audit probes.**
   - Direct fact probes.
   - Paraphrase probes.
   - Semantic-neighbor probes.
   - Multi-hop bridge probes.
   - Extraction prompts.
   - Retrieval-only probes.
   - Counterfactual probes comparing deleted and retain alternatives.

5. **Run black-box system calls.**
   - Query `S_-` without exposing deletion labels.
   - Record answers, retrieved contexts, citations, confidence, refusal behavior, and metadata available to normal users or auditors.

6. **Score behavioral evidence.**
   - Direct leakage: string, entity, and span overlap with protected content.
   - Paraphrased leakage: semantic similarity and entailment.
   - Retrieval dependence: deleted artifact or derivative appears in retrieved context.
   - Counterfactual dependence: answer matches deleted-source fact when a deletion-neutral answer should differ.
   - Extraction risk: protected content recovered under adversarial prompting.
   - Watermark/provenance hit: canary or watermark appears.

7. **Update confidence bounds.**
   - Compute empirical per-target risk.
   - Compute upper confidence bounds.
   - Stop early if pass or fail is statistically clear.

8. **Check retain utility.**
   - Evaluate `S_-` on `Q_ret`.
   - Flag over-unlearning if retain utility loss exceeds tolerance.

9. **Produce audit report.**
   - Decision.
   - Risk scores.
   - Confidence intervals.
   - Top violating probes.
   - Suspected provenance paths.
   - Recommended remediation.

## Pseudocode

```text
ADCU_AUDIT(S_minus, D_del, Q_ret, G, B, tau, delta):
    for z in D_del:
        V[z] <- estimate_pre_deletion_influence(z, G)
        P[z] <- generate_probe_pool(z, G)

    pi <- normalize_priorities(V)
    A <- allocate_budget(D_del, pi, B)

    for round in 1..max_rounds:
        for z in active_targets:
            probes <- sample_probes(P[z], A[z])
            outputs <- query_black_box(S_minus, probes)
            evidence[z] <- evidence[z] union score(outputs, z, G)
            ucb[z] <- risk_ucb(evidence[z], delta)

        if weighted_sum(pi, ucb) <= tau and retain_ok(S_minus, Q_ret):
            return PASS(report)

        if any_observed_violation(evidence, tau):
            return FAIL(report)

        active_targets <- targets_with_uncertain_ucb(ucb, tau)
        A <- reallocate_remaining_budget(active_targets, pi)

    return ESCALATE(report)
```

## Efficiency Levers

- Reuse cached pre-deletion retrieval and generation logs.
- Prioritize high-value and high-provenance-degree targets.
- Use cheap lexical and embedding scorers before expensive LLM-as-judge scoring.
- Use sequential confidence bounds to stop early.
- Batch probes by target and prompt template.
- Reuse paraphrase families across similar records.

## Baselines

- No audit beyond delete-from-corpus check.
- Exact-match leakage audit.
- Random probe audit.
- Uniform probe audit.
- Membership-inference audit.
- Retriever-hit audit.
- Leave-one-target-out utility audit.
- Full exhaustive probe suite.

## Expected Empirical Result

ADCU-Audit should detect more residual deletion failures per 1,000 model calls than random, uniform, exact-match, or retriever-only audits, especially in hybrid RAG-plus-fine-tuning systems.

