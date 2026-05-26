from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.reporting import read_rows, write_latex_table, write_markdown_table


OUT_DIR = ROOT / "experiments" / "adcu_submission_results"
TABLE_DIR = ROOT / "Third-Best-Data-Centric-Unlearning" / "submission_tables"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    track = read_rows(OUT_DIR / "track_audit_summary.json")
    methods = read_rows(OUT_DIR / "method_summary.json")
    budgets = read_rows(OUT_DIR / "budget_summary.json")
    sensitivity = read_rows(OUT_DIR / "weight_tolerance_sensitivity.json")
    utility = read_rows(OUT_DIR / "utility_risk_summary.json")
    scorer = read_rows(OUT_DIR / "scorer_validation_summary.json")
    cf_headline = read_rows(OUT_DIR / "cf_headline_summary.json")
    floor_roc = read_rows(OUT_DIR / "floor_roc_summary.json")
    independent = read_rows(OUT_DIR / "independent_label_summary.json")
    mid_size = read_rows(OUT_DIR / "mid_size_scaling_summary.json")
    blackbox = read_rows(OUT_DIR / "blackbox_summary.json")
    counterfactual = read_rows(OUT_DIR / "counterfactual_ablation_summary.json")
    robustness = read_rows(OUT_DIR / "robustness_summary.json")
    natural_case = read_rows(OUT_DIR / "natural_fever_case_study.json")

    track = [row for row in track if row.get("audit_method") not in {"ADCU-BlackBox", "ADCU-RetrievalOffCF"}]
    track_cols = ["track", "audit_method", "failure_detection_rate", "false_alarm_rate", "mean_risk_ucb", "mean_cf_distance", "mean_floor_normalized_risk", "mean_calls"]
    method_cols = ["track", "deletion_method", "direct_leakage", "paraphrase_leakage", "retrieval_dependence", "extraction_risk", "retain_utility", "risk_ucb", "cf_distance", "floor_normalized_risk"]
    budget_cols = ["budget", "failure_detection_rate", "mean_risk_ucb", "mean_floor_normalized_risk", "mean_calls"]
    sensitivity_cols = ["weight_profile", "tolerance", "failure_detection_rate", "false_alarm_rate", "n"]
    utility_cols = ["track", "deletion_method", "mean_risk_ucb", "mean_retain_utility", "mean_audit_calls", "failure_detection_rate"]
    scorer_cols = ["channel", "scorer", "threshold", "precision", "recall", "n_labeled_cases"]
    cf_headline_cols = ["track", "mean_rhat_cf", "mean_ucb_cf", "mean_floor", "floor_normalized_risk", "delta", "tolerance", "failure_detection_rate", "false_alarm_rate", "n"]
    floor_roc_cols = ["score", "tolerance", "detection_rate", "false_alarm_rate", "delta", "n"]
    independent_cols = ["track", "label_source", "labeled_failures", "labeled_clean", "detection_rate", "false_alarm_rate"]
    mid_size_cols = ["regime", "track", "audit_method", "retain_corpus_n", "deletion_targets", "budget", "failure_detection_rate", "false_alarm_rate", "mean_rhat_cf", "mean_ucb_cf", "floor_normalized_risk", "n"]
    blackbox_cols = ["track", "audit_method", "failure_detection_rate", "mean_direct", "mean_paraphrase", "mean_retrieval", "mean_risk_ucb", "n"]
    counterfactual_cols = ["track", "deletion_method", "full_risk_ucb", "retrieval_off_risk_ucb", "risk_delta", "full_detection_rate", "retrieval_off_detection_rate"]
    robustness_cols = ["track", "block_bootstrap_low", "block_bootstrap_high", "score_jitter_stability", "mean_effective_probe_families", "n"]
    natural_case_cols = ["case_id", "deletion_method", "ground_truth", "decision", "risk_ucb", "retrieval_dependence", "redacted_claim", "redacted_answer"]

    write_markdown_table(track, TABLE_DIR / "submission_table1_audit_summary.md", track_cols)
    write_markdown_table(methods, TABLE_DIR / "submission_table2_method_summary.md", method_cols)
    write_markdown_table(budgets, TABLE_DIR / "submission_table3_budget_summary.md", budget_cols)
    write_markdown_table(sensitivity, TABLE_DIR / "submission_table4_weight_sensitivity.md", sensitivity_cols)
    write_markdown_table(utility, TABLE_DIR / "submission_table5_utility_risk.md", utility_cols)
    write_markdown_table(scorer, TABLE_DIR / "submission_table6_scorer_validation.md", scorer_cols)
    write_markdown_table(cf_headline, TABLE_DIR / "submission_table11_cf_headline.md", cf_headline_cols)
    write_markdown_table(floor_roc, TABLE_DIR / "submission_table12_floor_roc.md", floor_roc_cols)
    write_markdown_table(independent, TABLE_DIR / "submission_table13_independent_labels.md", independent_cols)
    write_markdown_table(mid_size, TABLE_DIR / "submission_table14_mid_size_scaling.md", mid_size_cols)
    write_markdown_table(blackbox, TABLE_DIR / "submission_table7_blackbox.md", blackbox_cols)
    write_markdown_table(counterfactual, TABLE_DIR / "submission_table8_counterfactual_ablation.md", counterfactual_cols)
    write_markdown_table(robustness, TABLE_DIR / "submission_table9_robustness.md", robustness_cols)
    write_markdown_table(natural_case, TABLE_DIR / "submission_table10_natural_fever_case.md", natural_case_cols)
    write_latex_table(track, TABLE_DIR / "submission_table1_audit_summary.tex", track_cols, "Repeated audit detection summary across seeds, deletion-target sizes, and budgets.", "tab:submission_audit")
    write_latex_table(methods, TABLE_DIR / "submission_table2_method_summary.tex", method_cols, "Repeated residual-dependence summary for ADCU.", "tab:submission_methods")
    write_latex_table(budgets, TABLE_DIR / "submission_table3_budget_summary.tex", budget_cols, "ADCU detection as audit budget varies.", "tab:submission_budget")
    write_latex_table(sensitivity, TABLE_DIR / "submission_table4_weight_sensitivity.tex", sensitivity_cols, "Weight and tolerance sensitivity in the repeated harness.", "tab:weight_sensitivity")
    write_latex_table(utility, TABLE_DIR / "submission_table5_utility_risk.tex", utility_cols, "Utility--risk summary for ADCU across repeated deletion cases.", "tab:utility_risk")
    write_latex_table(scorer, TABLE_DIR / "submission_table6_scorer_validation.tex", scorer_cols, "Held-out released labeled-set validation of evidence-channel scorers.", "tab:scorer_validation")
    write_latex_table(cf_headline, TABLE_DIR / "submission_table11_cf_headline.tex", cf_headline_cols, "Headline counterfactual deletion results using the two-sample $\\hat R_{\\mathrm{cf}}$ statistic.", "tab:cf_headline")
    write_latex_table(floor_roc, TABLE_DIR / "submission_table12_floor_roc.tex", floor_roc_cols, "Floor-normalized counterfactual-risk ROC with declared $\\delta$.", "tab:floor_roc")
    write_latex_table(independent, TABLE_DIR / "submission_table13_independent_labels.tex", independent_cols, "Evaluation against deletion-operation labels independent of ADCU channel scores.", "tab:independent_labels")
    write_latex_table(mid_size, TABLE_DIR / "submission_table14_mid_size_scaling.tex", mid_size_cols, "Mid-size RealRAG scaling check on a 5,000-record FEVER retain corpus.", "tab:mid_size_scaling")
    write_latex_table(blackbox, TABLE_DIR / "submission_table7_blackbox.tex", blackbox_cols, "Pure black-box answer-only auditing compared with metadata-aware ADCU.", "tab:blackbox")
    write_latex_table(counterfactual, TABLE_DIR / "submission_table8_counterfactual_ablation.tex", counterfactual_cols, "Retrieval-off counterfactual ablation for deletion-risk attribution.", "tab:counterfactual_ablation")
    write_latex_table(robustness, TABLE_DIR / "submission_table9_robustness.tex", robustness_cols, "Robustness diagnostics for correlated probes and score jitter.", "tab:robustness")
    write_latex_table(natural_case, TABLE_DIR / "submission_table10_natural_fever_case.tex", natural_case_cols, "Natural-data FEVER deletion audit case study with redacted observed outputs.", "tab:natural_fever_case")
    print(f"Wrote submission tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
