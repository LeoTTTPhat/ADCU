from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu import ADCUAuditor, build_synthetic_scenario, make_leaky_system


def main() -> None:
    scenario = build_synthetic_scenario()
    system = make_leaky_system(
        scenario,
        leak_target_ids={"user_alpha", "user_beta"},
        retrieve_target_ids={"user_alpha"},
        paraphrase_only=True,
        canary_suppressed=True,
    )
    report = ADCUAuditor(scenario.graph, system).audit(scenario.targets, budget=24, retain_utility=0.96)

    print("Hybrid RAG + fine-tuning derivative attack")
    print(f"decision: {report.decision}")
    print(f"aggregate_risk_ucb: {report.aggregate_risk_ucb:.3f}")
    print(f"violating_probes: {', '.join(report.violating_probes) or 'none'}")


if __name__ == "__main__":
    main()

