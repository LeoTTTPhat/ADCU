from __future__ import annotations

from pathlib import Path
import sys

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

    track_cols = ["track", "audit_method", "failure_detection_rate", "bootstrap_ci95_low", "bootstrap_ci95_high", "mean_risk_ucb", "n"]
    method_cols = ["track", "deletion_method", "audit_method", "failure_detection_rate", "bootstrap_ci95_low", "bootstrap_ci95_high", "mean_risk_ucb", "n"]
    seed_cols = ["track", "seed", "failure_detection_rate", "mean_adcu_risk_ucb", "n"]
    ablation_cols = ["track", "probe_family", "failure_detection_rate", "bootstrap_ci95_low", "bootstrap_ci95_high", "n"]
    retriever_cols = ["retrieval_mode", "embedding_model", "top3_deleted_derivative_hit_rate", "top5_deleted_derivative_hit_rate", "adcu_detection_rate", "n"]
    margin_cols = ["track", "method", "active_forget_targets", "mean_forget_margin", "retain_perplexity", "retain_completion_accuracy", "adcu_decision", "failure_detection_rate", "n"]
    tofu_cols = ["method", "tofu_split", "split_size", "eval_n", "eval_mode", "answer_preferred_rate", "mean_answer_margin", "mean_answer_loss"]
    tofu_coverage_cols = ["tofu_split", "split_size", "default_eval_n", "default_eval_mode", "exhaustive_qwen_command"]
    case_cols = ["case_id", "risk_ucb", "direct_leakage", "paraphrase_leakage", "retrieval_dependence", "interpretation"]

    write_markdown_table(ci_track, TABLE_DIR / "advanced_table1_ci_by_track.md", track_cols)
    write_markdown_table(ci_method, TABLE_DIR / "advanced_table2_ci_by_method.md", method_cols)
    write_markdown_table(margins, TABLE_DIR / "advanced_table3_pretrained_margins.md", margin_cols)
    write_markdown_table(retriever_hits, TABLE_DIR / "advanced_table4_retriever_hits.md", retriever_cols)
    write_markdown_table(ablations, TABLE_DIR / "advanced_table5_probe_ablation.md", ablation_cols)
    write_markdown_table(per_seed, TABLE_DIR / "advanced_table6_per_seed.md", seed_cols)
    write_markdown_table(cases, TABLE_DIR / "advanced_table7_case_studies.md", case_cols)
    write_markdown_table(tofu_eval, TABLE_DIR / "advanced_table8_tofu_eval.md", tofu_cols)
    write_markdown_table(tofu_coverage, TABLE_DIR / "advanced_table9_tofu_split_coverage.md", tofu_coverage_cols)
    write_latex_table(ci_track, TABLE_DIR / "advanced_table1_ci_by_track.tex", track_cols, "Advanced LoRA-SFT and DenseRAG audit results with confidence intervals.", "tab:advanced_ci")
    write_latex_table(margins, TABLE_DIR / "advanced_table3_pretrained_margins.tex", margin_cols, "Pretrained LoRA forget margins and retain utility.", "tab:pretrained_margins")
    write_latex_table(retriever_hits, TABLE_DIR / "advanced_table4_retriever_hits.tex", retriever_cols, "Retriever deleted-derivative hit rates by mode.", "tab:retriever_hits")
    write_latex_table(ablations, TABLE_DIR / "advanced_table5_probe_ablation.tex", ablation_cols, "Probe-family ablations with bootstrap confidence intervals.", "tab:probe_ablation")
    write_latex_table(cases, TABLE_DIR / "advanced_table7_case_studies.tex", case_cols, "Redacted case studies from advanced experiments.", "tab:case_studies")
    write_latex_table(tofu_eval, TABLE_DIR / "advanced_table8_tofu_eval.tex", tofu_cols, "TOFU split-level evaluation for Qwen2.5-0.5B LoRA runs.", "tab:tofu_eval")

    print(f"Wrote advanced tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
