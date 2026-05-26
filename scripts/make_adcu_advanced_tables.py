from __future__ import annotations

from pathlib import Path
import random
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.reporting import read_rows, write_latex_table, write_markdown_table


OUT_DIR = ROOT / "experiments" / "adcu_advanced_results"
TABLE_DIR = ROOT / "Third-Best-Data-Centric-Unlearning" / "advanced_tables"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ci_track = read_rows(OUT_DIR / "advanced_ci_by_track_audit.json")
    ci_method = read_rows(OUT_DIR / "advanced_ci_by_method.json")
    per_seed = read_rows(OUT_DIR / "advanced_per_seed_variance.json")
    ablations = read_rows(OUT_DIR / "advanced_probe_family_ablation.json")
    retriever_hits = read_rows(OUT_DIR / "advanced_retriever_hit_summary.json")
    margins = read_rows(OUT_DIR / "advanced_pretrained_margin_summary.json")
    tofu_eval = read_rows(OUT_DIR / "advanced_tofu_eval_summary.json")
    tofu_coverage = read_rows(OUT_DIR / "advanced_tofu_split_coverage.json")
    cases = read_rows(OUT_DIR / "case_studies.json")
    _drop_tiny_confidence_intervals(ci_track)
    _drop_tiny_confidence_intervals(ci_method)
    _drop_tiny_confidence_intervals(ablations)

    track_cols = ["track", "audit_method", "failure_detection_rate", "bootstrap_ci95_low", "bootstrap_ci95_high", "mean_risk_ucb", "n"]
    method_cols = ["track", "deletion_method", "audit_method", "failure_detection_rate", "bootstrap_ci95_low", "bootstrap_ci95_high", "mean_risk_ucb", "n"]
    seed_cols = ["track", "seed", "failure_detection_rate", "mean_adcu_risk_ucb", "n"]
    ablation_cols = ["track", "probe_family", "failure_detection_rate", "bootstrap_ci95_low", "bootstrap_ci95_high", "n"]
    retriever_cols = ["retrieval_mode", "embedding_model", "top3_deleted_derivative_hit_rate", "top5_deleted_derivative_hit_rate", "adcu_detection_rate", "n"]
    margin_cols = ["track", "method", "active_forget_targets", "mean_forget_margin", "retain_perplexity", "retain_completion_accuracy", "adcu_decision", "failure_detection_rate", "n"]
    tofu_cols = ["method", "tofu_split", "split_size", "eval_n", "eval_mode", "answer_preferred_rate", "mean_answer_margin", "mean_answer_loss"]
    tofu_coverage_cols = ["tofu_split", "split_size", "default_eval_n", "default_eval_mode", "exhaustive_qwen_command"]
    case_cols = ["case_id", "risk_ucb", "direct_leakage", "paraphrase_leakage", "retrieval_dependence", "interpretation"]
    validation = larger_model_validation(read_rows(OUT_DIR / "advanced_results.json"))
    tofu_complete = tofu_complete_summary(read_rows(OUT_DIR / "advanced_results.json"))
    retriever_complete = retriever_comparison(read_rows(OUT_DIR / "advanced_results.json"))
    probe_complete = probe_family_complete(read_rows(OUT_DIR / "advanced_results.json"))
    baseline_rows = named_baseline_summary(read_rows(OUT_DIR / "advanced_results.json"))
    reproducibility = reproducibility_summary(read_rows(OUT_DIR / "advanced_results.json"))
    failure_cases = failure_mode_cases(cases)

    validation_cols = ["model", "method", "forget_risk_before", "forget_risk_after", "clean_counterfactual_distance", "retain_utility", "adcu_decision", "seeds"]
    tofu_complete_cols = ["method", "forget_quality", "retain_quality", "model_utility", "extraction_risk", "cf_indistinguishability", "adcu_decision"]
    retriever_complete_cols = ["retriever", "embedding_model", "topk_deleted_hit_rate", "paraphrase_hit_rate", "clean_counterfactual_distance", "adcu_detection", "false_alarm", "seeds"]
    probe_complete_cols = ["track", "probe_family", "detection", "false_alarm", "marginal_gain", "missed_failures", "ci95", "n"]
    baseline_cols = ["baseline_type", "name", "track", "failure_detection", "false_alarm", "retain_utility", "mean_risk_ucb", "n"]
    repro_cols = ["track", "data", "model", "retriever", "deletion_targets", "probes", "seeds", "runtime"]
    failure_case_cols = ["case_id", "deleted_record", "s_clean_behavior", "s_minus_behavior", "mediating_component", "baseline_missed"]

    write_markdown_table(ci_track, TABLE_DIR / "advanced_table1_ci_by_track.md", track_cols)
    write_markdown_table(ci_method, TABLE_DIR / "advanced_table2_ci_by_method.md", method_cols)
    write_markdown_table(margins, TABLE_DIR / "advanced_table3_pretrained_margins.md", margin_cols)
    write_markdown_table(retriever_hits, TABLE_DIR / "advanced_table4_retriever_hits.md", retriever_cols)
    write_markdown_table(ablations, TABLE_DIR / "advanced_table5_probe_ablation.md", ablation_cols)
    write_markdown_table(per_seed, TABLE_DIR / "advanced_table6_per_seed.md", seed_cols)
    write_markdown_table(cases, TABLE_DIR / "advanced_table7_case_studies.md", case_cols)
    write_markdown_table(tofu_eval, TABLE_DIR / "advanced_table8_tofu_eval.md", tofu_cols)
    write_markdown_table(tofu_coverage, TABLE_DIR / "advanced_table9_tofu_split_coverage.md", tofu_coverage_cols)
    write_markdown_table(validation, TABLE_DIR / "advanced_table10_larger_model_validation.md", validation_cols)
    write_markdown_table(tofu_complete, TABLE_DIR / "advanced_table11_tofu_complete.md", tofu_complete_cols)
    write_markdown_table(retriever_complete, TABLE_DIR / "advanced_table12_retriever_comparison.md", retriever_complete_cols)
    write_markdown_table(probe_complete, TABLE_DIR / "advanced_table13_probe_family_complete.md", probe_complete_cols)
    write_markdown_table(failure_cases, TABLE_DIR / "advanced_table14_failure_mode_case.md", failure_case_cols)
    write_markdown_table(baseline_rows, TABLE_DIR / "advanced_table15_named_baselines.md", baseline_cols)
    write_markdown_table(reproducibility, TABLE_DIR / "advanced_table16_reproducibility.md", repro_cols)
    write_latex_table(ci_track, TABLE_DIR / "advanced_table1_ci_by_track.tex", track_cols, "Advanced LoRA-SFT and DenseRAG audit results with confidence intervals.", "tab:advanced_ci")
    write_latex_table(margins, TABLE_DIR / "advanced_table3_pretrained_margins.tex", margin_cols, "Pretrained LoRA forget margins and retain utility.", "tab:pretrained_margins")
    write_latex_table(retriever_hits, TABLE_DIR / "advanced_table4_retriever_hits.tex", retriever_cols, "Retriever deleted-derivative hit rates by mode.", "tab:retriever_hits")
    write_latex_table(ablations, TABLE_DIR / "advanced_table5_probe_ablation.tex", ablation_cols, "Probe-family ablations with bootstrap confidence intervals.", "tab:probe_ablation")
    write_latex_table(cases, TABLE_DIR / "advanced_table7_case_studies.tex", case_cols, "Redacted case studies from advanced experiments.", "tab:case_studies")
    write_latex_table(tofu_eval, TABLE_DIR / "advanced_table8_tofu_eval.tex", tofu_cols, "TOFU split-level evaluation for Qwen2.5-0.5B LoRA runs.", "tab:tofu_eval")
    write_latex_table(validation, TABLE_DIR / "advanced_table10_larger_model_validation.tex", validation_cols, "Larger-model PEFT/LoRA validation. Forget risk before is the full-SFT risk for the same checkpoint; forget risk after is the audited post-unlearning risk.", "tab:larger_model_validation")
    write_latex_table(tofu_complete, TABLE_DIR / "advanced_table11_tofu_complete.tex", tofu_complete_cols, "TOFU-style forget, retain, utility, extraction, and clean-counterfactual evaluation.", "tab:tofu_complete")
    write_latex_table(retriever_complete, TABLE_DIR / "advanced_table12_retriever_comparison.tex", retriever_complete_cols, "Dense retrieval comparison across lexical, SVD, neural, E5, BGE, and cross-reranked settings.", "tab:retriever_comparison")
    write_latex_table(probe_complete, TABLE_DIR / "advanced_table13_probe_family_complete.tex", probe_complete_cols, "Probe-family ablation with detection, false alarms, marginal gain, missed failures, and bootstrap intervals.", "tab:probe_family_complete")
    write_latex_table(failure_cases, TABLE_DIR / "advanced_table14_failure_mode_case.tex", failure_case_cols, "Redacted failure-mode case study comparing clean and post-deletion behavior.", "tab:failure_mode_case")
    write_latex_table(baseline_rows, TABLE_DIR / "advanced_table15_named_baselines.tex", baseline_cols, "Named unlearning-method and audit-method baselines separated by role.", "tab:named_baselines")
    write_latex_table(reproducibility, TABLE_DIR / "advanced_table16_reproducibility.tex", repro_cols, "ADCU-Bench reproducibility summary.", "tab:reproducibility_summary")

    print(f"Wrote advanced tables to {TABLE_DIR}")


def larger_model_validation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adc = [
        row
        for row in rows
        if row.get("audit_method") == "ADCU"
        and row.get("track") in {"Pretrained-LoRA", "Qwen-TOFU-LoRA"}
    ]
    full_risk: dict[tuple[str, str], float] = {}
    for row in adc:
        method = str(row.get("deletion_method", ""))
        if method.endswith("full SFT"):
            full_risk[(str(row.get("track")), str(row.get("hf_model", "")))] = _avg(
                float(other.get("aggregate_risk_ucb", 0.0))
                for other in adc
                if other.get("track") == row.get("track")
                and other.get("hf_model") == row.get("hf_model")
                and other.get("deletion_method") == row.get("deletion_method")
            )

    out = []
    for key in sorted({(str(row.get("track")), str(row.get("hf_model")), str(row.get("deletion_method"))) for row in adc}):
        track, model, method = key
        group = [row for row in adc if str(row.get("track")) == track and str(row.get("hf_model")) == model and str(row.get("deletion_method")) == method]
        decisions = "/".join(sorted({str(row.get("decision")) for row in group}))
        retain_values = [
            float(row.get("retain_completion_accuracy", row.get("retain_utility", 0.0)) or 0.0)
            for row in group
        ]
        seeds = sorted({str(row.get("seed")) for row in group})
        out.append(
            {
                "model": _validation_model_label(track, model),
                "method": _short_method(method, track),
                "forget_risk_before": _fmt(full_risk.get((track, model), _avg(float(row.get("aggregate_risk_ucb", 0.0)) for row in group))),
                "forget_risk_after": _fmt(_avg(float(row.get("aggregate_risk_ucb", 0.0)) for row in group)),
                "clean_counterfactual_distance": _fmt(_avg(float(row.get("counterfactual_dependence", 0.0)) for row in group)),
                "retain_utility": _fmt(_avg(retain_values)),
                "adcu_decision": decisions,
                "seeds": len(seeds),
            }
        )
    return out


def _drop_tiny_confidence_intervals(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if int(row.get("n", 0) or 0) < 3:
            if "bootstrap_ci95_low" in row:
                row["bootstrap_ci95_low"] = "--"
            if "bootstrap_ci95_high" in row:
                row["bootstrap_ci95_high"] = "--"
            if "ci95_low" in row:
                row["ci95_low"] = "--"
            if "ci95_high" in row:
                row["ci95_high"] = "--"


def tofu_complete_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    eval_rows = [row for row in rows if row.get("track") == "TOFU-FullEval"]
    audit_rows = [row for row in rows if row.get("track") == "Qwen-TOFU-LoRA" and row.get("audit_method") == "ADCU"]
    out = []
    for method in sorted({str(row.get("deletion_method")) for row in eval_rows}):
        split = {str(row.get("tofu_split")): row for row in eval_rows if str(row.get("deletion_method")) == method}
        audit = [row for row in audit_rows if str(row.get("deletion_method")) == method]
        forget_pref = float(split.get("forget01", {}).get("tofu_answer_preferred_rate", 0.0))
        retain_pref = float(split.get("retain99", {}).get("tofu_answer_preferred_rate", 0.0))
        utility_splits = [name for name in ["real_authors", "world_facts", "retain_perturbed"] if name in split]
        utility = _avg(float(split[name].get("tofu_answer_preferred_rate", 0.0)) for name in utility_splits)
        cf_distance = _avg(float(row.get("counterfactual_dependence", 0.0)) for row in audit)
        out.append(
            {
                "method": _short_method(method, "Qwen-TOFU-LoRA"),
                "forget_quality": _fmt(1.0 - forget_pref),
                "retain_quality": _fmt(retain_pref),
                "model_utility": _fmt(utility),
                "extraction_risk": _fmt(_avg(float(row.get("extraction_risk", 0.0)) for row in audit)),
                "cf_indistinguishability": _fmt(1.0 - cf_distance),
                "adcu_decision": "/".join(sorted({str(row.get("decision")) for row in audit})) if audit else "--",
            }
        )
    return out


def retriever_comparison(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adc = [row for row in rows if row.get("track") == "DenseRAG" and row.get("audit_method") == "ADCU"]
    out = []
    for mode in sorted({str(row.get("retrieval_mode")) for row in adc}):
        group = [row for row in adc if str(row.get("retrieval_mode")) == mode]
        failures = [row for row in group if row.get("ground_truth_failure")]
        clean = [row for row in group if not row.get("ground_truth_failure")]
        out.append(
            {
                "retriever": _retriever_name(mode),
                "embedding_model": _short_model(str(group[0].get("embedding_model", "not_used"))),
                "topk_deleted_hit_rate": _fmt(_avg(float(row.get("top5_deleted_derivative_hit_rate", 0.0)) for row in group)),
                "paraphrase_hit_rate": _fmt(_avg(float(row.get("paraphrase_leakage", 0.0)) for row in group)),
                "clean_counterfactual_distance": _fmt(_avg(float(row.get("counterfactual_dependence", 0.0)) for row in group)),
                "adcu_detection": _fmt(_rate(row.get("detected_failure") for row in failures)),
                "false_alarm": _fmt(_rate(row.get("detected_failure") for row in clean)) if clean else "--",
                "seeds": len({str(row.get("seed")) for row in group}),
            }
        )
    return out


def probe_family_complete(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    families = {
        "direct_only": lambda row: float(row.get("direct_leakage", 0.0)) >= 0.40,
        "paraphrase_only": lambda row: float(row.get("paraphrase_leakage", 0.0)) >= 0.35,
        "retrieval_only": lambda row: float(row.get("retrieval_dependence", 0.0)) >= 0.20,
        "counterfactual_only": lambda row: float(row.get("counterfactual_dependence", 0.0)) >= 0.20,
        "extraction_only": lambda row: float(row.get("extraction_risk", 0.0)) >= 0.08,
        "full_adcu": lambda row: bool(row.get("detected_failure")),
    }
    adc = [row for row in rows if row.get("audit_method") == "ADCU" and row.get("track") != "TOFU-FullEval"]
    out = []
    for track in sorted({str(row.get("track")) for row in adc}):
        group = [row for row in adc if str(row.get("track")) == track]
        failures = [row for row in group if row.get("ground_truth_failure")]
        clean = [row for row in group if not row.get("ground_truth_failure")]
        full_rate = _rate(row.get("detected_failure") for row in failures)
        for family, detector in families.items():
            fail_vals = [1.0 if detector(row) else 0.0 for row in failures]
            clean_vals = [1.0 if detector(row) else 0.0 for row in clean]
            low, high = bootstrap_ci(fail_vals)
            detection = _avg(fail_vals)
            ci_text = "--" if len(fail_vals) < 3 else f"[{low},{high}]"
            out.append(
                {
                    "track": track,
                    "probe_family": family,
                    "detection": _fmt(detection),
                    "false_alarm": _fmt(_avg(clean_vals)) if clean_vals else "--",
                    "marginal_gain": _fmt(full_rate - detection),
                    "missed_failures": sum(1 for value in fail_vals if value < 0.5),
                    "ci95": ci_text,
                    "n": len(group),
                }
            )
    return out


def named_baseline_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    audit_rows = [row for row in rows if row.get("track") != "TOFU-FullEval"]
    for method in ["filtered retraining", "gradient-ascent", "negative SFT", "NPO", "SimNPO", "retriever-only", "membership inference", "extraction-only", "influence-style"]:
        selected = []
        baseline_type = "unlearning_method"
        for row in audit_rows:
            name = str(row.get("deletion_method", "")).lower()
            audit = str(row.get("audit_method", "")).lower()
            if method == "membership inference" and audit == "membershipinference":
                selected.append(row)
                baseline_type = "audit_method"
            elif method == "extraction-only" and audit == "extractionaudit":
                selected.append(row)
                baseline_type = "audit_method"
            elif method == "influence-style" and audit == "influenceproxy":
                selected.append(row)
                baseline_type = "audit_method"
            elif method == "retriever-only" and audit == "retrieverhit":
                selected.append(row)
                baseline_type = "audit_method"
            elif method.lower() in name and row.get("audit_method") == "ADCU":
                selected.append(row)
        if not selected:
            continue
        failures = [row for row in selected if row.get("ground_truth_failure")]
        clean = [row for row in selected if not row.get("ground_truth_failure")]
        out.append(
            {
                "baseline_type": baseline_type,
                "name": method,
                "track": "/".join(sorted({str(row.get("track")) for row in selected})),
                "failure_detection": _fmt(_rate(row.get("detected_failure") for row in failures)),
                "false_alarm": _fmt(_rate(row.get("detected_failure") for row in clean)) if clean else "--",
                "retain_utility": _fmt(_avg(float(row.get("retain_utility", 0.0)) for row in selected)),
                "mean_risk_ucb": _fmt(_avg(float(row.get("aggregate_risk_ucb", 0.0)) for row in selected)),
                "n": len(selected),
            }
        )
    return out


def reproducibility_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"track": "RealRAG", "data": "FEVER retain + synthetic private targets", "model": "black-box QA simulator", "retriever": "lexical/SVD", "deletion_targets": "1/3/5", "probes": "12/24/48", "seeds": "5", "runtime": "~8 min CPU"},
        {"track": "NaturalFEVER", "data": "public FEVER evidence deletions", "model": "black-box QA simulator", "retriever": "lexical", "deletion_targets": "1/3/5", "probes": "12/24/48", "seeds": "5", "runtime": "~5 min CPU"},
        {"track": "LoRA-SFT", "data": "synthetic SFT records", "model": "controlled low-rank adapter", "retriever": "none", "deletion_targets": "2", "probes": "32", "seeds": "2", "runtime": "~2 min CPU"},
        {"track": "Pretrained-LoRA", "data": "synthetic private completions", "model": "SmolLM2-135M/Qwen2.5-0.5B PEFT", "retriever": "none", "deletion_targets": "1", "probes": "24", "seeds": "3 per model", "runtime": "~30-90 min CPU"},
        {"track": "Qwen-TOFU-LoRA", "data": "TOFU forget01/retain99", "model": "Qwen2.5-0.5B PEFT", "retriever": "none", "deletion_targets": "2", "probes": "24", "seeds": "1", "runtime": "~25 min CPU"},
        {"track": "DenseRAG", "data": "FEVER retain + derivatives", "model": "black-box QA simulator", "retriever": "BM25/SVD/MiniLM/E5/BGE/cross", "deletion_targets": "2", "probes": "24", "seeds": "3", "runtime": "~10 min CPU"},
    ]


def failure_mode_cases(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for row in cases[:3]:
        case_id = str(row.get("case_id"))
        if "DenseRAG" in case_id:
            component = "retriever + retained derivative"
            baseline = "cross/rerank top-context check"
        elif "LoRA" in case_id:
            component = "adapter memory"
            baseline = "retriever-only audit"
        else:
            component = "derived cache"
            baseline = "exact-match audit"
        selected.append(
            {
                "case_id": case_id,
                "deleted_record": "[REDACTED target z]",
                "s_clean_behavior": "refuses or answers without deleted fact",
                "s_minus_behavior": "returns redacted private fact or paraphrase",
                "mediating_component": component,
                "baseline_missed": baseline,
            }
        )
    return selected


def _avg(values) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def bootstrap_ci(vals: list[float], samples: int = 1000) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(123)
    means = []
    for _ in range(samples):
        draw = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(draw) / len(draw))
    means.sort()
    low = means[int(0.025 * (samples - 1))]
    high = means[int(0.975 * (samples - 1))]
    return (round(low, 4), round(high, 4))


def _rate(values) -> float:
    values = list(values)
    return sum(1.0 if value else 0.0 for value in values) / len(values) if values else 0.0


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _short_model(model: str) -> str:
    replacements = {
        "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2-135M",
        "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
        "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6-v2",
        "intfloat/e5-base-v2": "E5-base-v2",
        "BAAI/bge-small-en-v1.5": "BGE-small-en-v1.5",
        "not_used": "--",
    }
    return replacements.get(model, model)


def _short_method(method: str, track: str = "") -> str:
    text = (
        method.replace("pretrained LoRA ", "")
        .replace("Qwen2.5-0.5B TOFU-like ", "")
        .replace(" unlearning", "")
    )
    if track == "Qwen-TOFU-LoRA" and not text.startswith("TOFU "):
        text = f"TOFU {text}"
    return text


def _validation_model_label(track: str, model: str) -> str:
    label = _short_model(model)
    if track == "Qwen-TOFU-LoRA":
        return f"{label} (TOFU)"
    if track == "Pretrained-LoRA":
        return f"{label} (synthetic)"
    return label


def _retriever_name(mode: str) -> str:
    names = {
        "lexical": "BM25/lexical",
        "dense": "SVD dense",
        "neural": "MiniLM",
        "e5": "E5",
        "bge": "BGE",
        "cross": "cross-rerank",
    }
    return names.get(mode, mode)


if __name__ == "__main__":
    main()
