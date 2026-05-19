from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.reporting import (
    detection_summary,
    read_rows,
    residual_summary,
    write_latex_table,
    write_markdown_table,
)


OUT_DIR = ROOT / "experiments" / "adcu_results"
TABLE_DIR = ROOT / "Third-Best-Data-Centric-Unlearning" / "tables"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(OUT_DIR / "main_results.json")
    attacks = read_rows(OUT_DIR / "attack_suite.json")

    detection = detection_summary(rows)
    residual = residual_summary(rows)

    det_cols = ["track", "audit_method", "failure_detection_rate", "false_alarm_rate", "mean_risk_ucb", "detections_per_1000_calls"]
    res_cols = ["track", "deletion_method", "direct_leakage", "paraphrase_leakage", "retrieval_dependence", "extraction_risk", "retain_utility", "risk_ucb"]
    atk_cols = ["attack_id", "decision", "risk_ucb", "violations", "fooled_weak_audits"]

    write_markdown_table(detection, TABLE_DIR / "table1_detection_summary.md", det_cols)
    write_markdown_table(residual, TABLE_DIR / "table2_residual_channels.md", res_cols)
    write_markdown_table(attacks, TABLE_DIR / "table3_attack_suite.md", atk_cols)
    write_latex_table(detection, TABLE_DIR / "table1_detection_summary.tex", det_cols, "Audit detection summary.", "tab:adcu_detection")
    write_latex_table(residual, TABLE_DIR / "table2_residual_channels.tex", res_cols, "Residual dependence by deletion method.", "tab:adcu_residual")
    write_latex_table(attacks, TABLE_DIR / "table3_attack_suite.tex", atk_cols, "Attack suite outcomes.", "tab:adcu_attacks")

    print(f"Wrote ADCU tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()

