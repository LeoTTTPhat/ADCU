from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu import ADCUAuditor, build_attack_suite, build_synthetic_scenario, summarize_channels


OUT_DIR = ROOT / "experiments" / "adcu_results"


def main() -> None:
    from csv import DictWriter
    import json

    scenario = build_synthetic_scenario()
    rows: list[dict[str, object]] = []
    for attack in build_attack_suite(scenario):
        retain_utility = 0.40 if attack.attack_id == "retain_neighbor_over_unlearning" else 0.98
        report = ADCUAuditor(scenario.graph, attack.system).audit(
            scenario.targets,
            budget=24,
            retain_utility=retain_utility,
            retain_tolerance=0.2,
        )
        channels = summarize_channels(scenario.graph, attack.system, scenario.targets)
        rows.append(
            {
                "attack_id": attack.attack_id,
                "name": attack.name,
                "decision": report.decision,
                "risk_ucb": round(report.aggregate_risk_ucb, 4),
                "violations": len(report.violating_probes),
                "retain_utility": retain_utility,
                "fooled_weak_audits": ", ".join(attack.fooled_weak_audits),
                **{key: round(value, 4) for key, value in channels.items()},
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "attack_suite.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (OUT_DIR / "attack_suite.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} attack rows to {OUT_DIR}")
    for row in rows:
        print(f"{row['name']:32s} decision={row['decision']:8s} risk={row['risk_ucb']:.3f}")


if __name__ == "__main__":
    main()

