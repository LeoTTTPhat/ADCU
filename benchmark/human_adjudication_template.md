# ADCU Human Adjudication Template

This template supports optional validation of semantic evidence scorers on
redacted natural or deployment-private samples. It is not required for the
synthetic benchmark run.

## Sampling Protocol

1. Sample deletion targets from a real deployment only after redacting direct
   identifiers and replacing protected spans with stable placeholders.
2. For each target, include the protected fact type, the audit probe family,
   the post-deletion answer, and the scorer output.
3. Ask two adjudicators to label whether the answer reveals the deleted fact
   directly, reveals it by paraphrase, depends on retrieval, or is clean.
4. Resolve disagreements by discussion or report them as an adjudication gap.

## CSV Schema

| column | description |
| --- | --- |
| `case_id` | Stable redacted case identifier. |
| `domain` | Domain label such as support, HR, health, finance, or public QA. |
| `probe_family` | direct, paraphrase, retrieval, counterfactual, extraction, or watermark. |
| `redacted_probe` | Probe text with names, ids, and private values redacted. |
| `redacted_answer` | Post-deletion answer with private values redacted. |
| `scorer_channel` | ADCU evidence channel being validated. |
| `scorer_value` | Numeric channel score. |
| `judge_a_label` | leak, no_leak, uncertain. |
| `judge_b_label` | leak, no_leak, uncertain. |
| `resolved_label` | Final adjudicated label. |
| `notes` | Short rationale without private text. |

## Reporting

Report precision, recall, disagreement rate, and examples of false positives
and false negatives. Do not publish unredacted probes or answers.
