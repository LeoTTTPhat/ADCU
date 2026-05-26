from __future__ import annotations

from csv import DictWriter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.submission_experiments import (
    aggregate_submission_rows,
    run_natural_fever_case_study,
    run_mid_size_scaling_experiment,
    run_submission_experiments,
    summarize_mid_size_scaling,
)


OUT_DIR = ROOT / "experiments" / "adcu_submission_results"


def write_rows(name: str, rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (OUT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    rows = run_submission_experiments(ROOT / "data")
    write_rows("submission_results", rows)
    summaries = aggregate_submission_rows(rows)
    for name, summary_rows in summaries.items():
        write_rows(name, summary_rows)
    write_rows("natural_fever_case_study", run_natural_fever_case_study(ROOT / "data"))
    mid_size_rows = run_mid_size_scaling_experiment(ROOT / "data")
    write_rows("mid_size_scaling_results", mid_size_rows)
    write_rows("mid_size_scaling_summary", summarize_mid_size_scaling(mid_size_rows))

    print(f"Wrote {len(rows)} repeated submission experiment rows to {OUT_DIR}")
    print("ADCU summary by track:")
    for row in summaries["track_audit_summary"]:
        if row["audit_method"] == "ADCU":
            print(
                f"{row['track']:12s} detection={row['failure_detection_rate']:.3f} "
                f"false_alarm={row['false_alarm_rate']:.3f} "
                f"risk={row['mean_risk_ucb']:.3f} calls={row['mean_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
