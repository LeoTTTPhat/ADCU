from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.reporting import detection_summary, read_rows, residual_summary, write_svg_bar, write_svg_pipeline


OUT_DIR = ROOT / "experiments" / "adcu_results"
FIG_DIR = ROOT / "figures" / "adcu"


def main() -> None:
    rows = read_rows(OUT_DIR / "main_results.json")
    detection = [
        row
        for row in detection_summary(rows)
        if row["track"] == "Hybrid"
    ]
    residual = [
        row
        for row in residual_summary(rows)
        if row["track"] == "Hybrid"
    ]
    for row in detection:
        row["label"] = row["audit_method"]
    for row in residual:
        row["label"] = row["deletion_method"]

    write_svg_pipeline(FIG_DIR / "adcu_pipeline.svg")
    write_svg_bar(
        detection,
        FIG_DIR / "hybrid_detection_rate.svg",
        "Hybrid Track Failure Detection Rate",
        "label",
        "failure_detection_rate",
    )
    write_svg_bar(
        residual,
        FIG_DIR / "hybrid_deletion_risk.svg",
        "Hybrid Residual Deletion Risk",
        "label",
        "risk_ucb",
    )
    print(f"Wrote ADCU figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

