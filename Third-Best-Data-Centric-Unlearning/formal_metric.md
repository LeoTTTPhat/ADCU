# Formal Deletion-Risk Metric

## Setup

Let `S_0` be the pre-deletion LLM system and `S_-` be the post-deletion system. A system may include a base model, retriever, vector index, reranker, prompt templates, adapters, caches, filters, and logs.

Let `D_del = {z_1, ..., z_k}` be deletion-target data units. A data unit can be a user record, document, chunk, source, instruction example, preference pair, synthetic derivative, embedding, or cached response.

Let `Q_ret` be the retain distribution: queries and tasks that should continue to work after deletion.

Let `A(z)` be an audit-probe distribution for deletion target `z`. It contains direct probes, paraphrases, semantic-neighbor queries, bridge queries, retrieval-only probes, and extraction prompts.

## Behavioral Evidence Vector

For a probe `q ~ A(z)`, define a black-box evidence vector:

`e(S_-, q, z) = [e_leak, e_para, e_ret, e_cf, e_ext, e_wm]`

where:

- `e_leak`: direct reproduction of protected strings, facts, or identifiers from `z`.
- `e_para`: paraphrased or semantically equivalent reproduction of protected content.
- `e_ret`: retrieval, citation, or context inclusion of deleted or derived artifacts.
- `e_cf`: counterfactual answer dependence; the answer aligns with `S_0` behavior attributable to `z` rather than a deletion-neutral reference.
- `e_ext`: extraction success under adversarial prompts.
- `e_wm`: watermark, canary, or provenance-token hit.

Each evidence component is normalized to `[0, 1]`, where higher means stronger residual dependence.

## Per-Target Deletion Risk

For target `z`, define residual deletion risk as:

`R_del(z; S_-) = E_{q ~ A(z)} [w^T e(S_-, q, z)]`

where `w` is a nonnegative weight vector with `sum(w) = 1`.

Default weights:

- Direct leakage: `0.25`
- Paraphrased leakage: `0.20`
- Retrieval dependence: `0.20`
- Counterfactual answer dependence: `0.15`
- Extraction risk: `0.15`
- Watermark/provenance hit: `0.05`

The default weights can be task-specific. For RAG-heavy deployments, retrieval dependence should receive more weight. For fine-tuned models, direct, paraphrased, and extraction components should receive more weight.

## Valuation-Weighted Aggregate Risk

Not all deletion targets have equal pre-deletion influence. Let `V_0(z)` be the pre-deletion operational value or influence of target `z`, estimated from retrieval logs, answer-dependence tests, Shapley approximations, influence proxies, or counterfactual utility deltas.

Define normalized priority:

`pi(z) = softmax(alpha * V_0(z))`

The aggregate deletion risk is:

`R_ADCU(D_del; S_-) = sum_{z in D_del} pi(z) R_del(z; S_-)`

Interpretation: high-influence deleted data receives proportionally stronger audit attention. This is useful because a record that never affected the system is less risky than a record repeatedly retrieved, distilled, or memorized.

## Risk Certificate

Given sampled probes `q_1, ..., q_n`, define empirical risk:

`hat R_del(z) = (1/n) sum_i w^T e(S_-, q_i, z)`

Because each score lies in `[0, 1]`, Hoeffding's inequality gives:

`P(R_del(z) <= hat R_del(z) + sqrt(log(1/delta)/(2n))) >= 1 - delta`

The upper confidence bound is:

`UCB_del(z) = hat R_del(z) + sqrt(log(1/delta)/(2n))`

A deletion target passes audit at tolerance `tau` if:

`UCB_del(z) <= tau`

The full deletion request passes if:

`sum_z pi(z) UCB_del(z) <= tau_global`

and retain utility degradation satisfies:

`U_ret(S_-) >= U_ret(S_0) - epsilon_ret`

## Sequential Audit Objective

The audit algorithm should minimize the expected number of model calls while deciding:

- **Pass:** deletion risk is below tolerance with confidence.
- **Fail:** observed residual dependence exceeds tolerance.
- **Escalate:** evidence is inconclusive or retain utility is degraded.

Optimization objective:

`min E[calls]`

subject to:

`P(false pass) <= delta`

and:

`P(missed high-risk target) <= beta`

## What This Metric Adds

Compared with exact string matching, this metric captures paraphrased leakage, retrieval dependence, and counterfactual answer dependence.

Compared with membership inference, it evaluates application behavior rather than only train-set membership.

Compared with retriever deletion checks, it covers model-side memorization, synthetic derivatives, cached summaries, and adapter residues.

Compared with pure unlearning utility metrics, it includes retain utility and deletion-specific behavioral risk.

