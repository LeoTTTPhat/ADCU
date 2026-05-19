from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.reporting import read_rows, write_svg_bar


OUT_DIR = ROOT / "experiments" / "adcu_submission_results"
FIG_DIR = ROOT / "figures" / "adcu_submission"


def main() -> None:
    track = read_rows(OUT_DIR / "track_audit_summary.json")
    budgets = read_rows(OUT_DIR / "budget_summary.json")

    hybrid = [row for row in track if row["track"] == "HybridReal"]
    for row in hybrid:
        row["label"] = row["audit_method"]
    for row in budgets:
        row["label"] = f"budget={row['budget']}"

    write_svg_bar(
        hybrid,
        FIG_DIR / "hybrid_real_detection_rate.svg",
        "HybridReal Failure Detection Across Audit Methods",
        "label",
        "failure_detection_rate",
    )
    write_svg_bar(
        budgets,
        FIG_DIR / "budget_detection_rate.svg",
        "ADCU Failure Detection Across Audit Budgets",
        "label",
        "failure_detection_rate",
    )
    print(f"Wrote submission figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

