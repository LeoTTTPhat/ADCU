from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu import build_synthetic_scenario, run_finetune_track, run_hybrid_track, run_rag_track
from adcu.reporting import detection_summary, residual_summary, write_rows


OUT_DIR = ROOT / "experiments" / "adcu_results"


def main() -> None:
    scenario = build_synthetic_scenario()
    rows = []
    rows.extend(run_rag_track(scenario))
    rows.extend(run_finetune_track(scenario))
    rows.extend(run_hybrid_track(scenario))

    write_rows(rows, OUT_DIR, "main_results")

    dict_rows = [row.to_dict() for row in rows]
    summary_rows = detection_summary(dict_rows)
    residual_rows = residual_summary(dict_rows)
    write_dict_rows(summary_rows, "detection_summary")
    write_dict_rows(residual_rows, "residual_summary")

    print(f"Wrote {len(rows)} main experiment rows to {OUT_DIR}")
    print("Top ADCU rows:")
    for row in dict_rows:
        if row["audit_method"] == "ADCU":
            print(
                f"{row['track']:8s} {row['deletion_method'][:30]:30s} "
                f"decision={row['decision']:8s} risk={row['aggregate_risk_ucb']:.3f} "
                f"detected={row['detected_failure']}"
            )


def write_dict_rows(rows: list[dict[str, object]], stem: str) -> None:
    from csv import DictWriter
    import json

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    with (OUT_DIR / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

