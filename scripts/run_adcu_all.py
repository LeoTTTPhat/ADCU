from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu import ADCUAuditor, build_attack_suite, build_synthetic_scenario, make_clean_system, make_leaky_system


def print_report(name: str, report) -> None:
    print(
        f"{name:32s} decision={report.decision:8s} "
        f"risk_ucb={report.aggregate_risk_ucb:.3f} "
        f"probes={report.num_probes:2d} "
        f"violations={len(report.violating_probes):2d}"
    )
    if report.violating_probes:
        shown = ", ".join(report.violating_probes[:4])
        suffix = " ..." if len(report.violating_probes) > 4 else ""
        print(f"{'':32s} violating probes: {shown}{suffix}")
    if report.notes:
        print(f"{'':32s} notes: {'; '.join(report.notes)}")


def main() -> None:
    scenario = build_synthetic_scenario()

    print("ADCU synthetic deletion audit")
    print("=" * 72)

    clean = ADCUAuditor(scenario.graph, make_clean_system(scenario), global_tolerance=0.75)
    print_report("clean provenance deletion", clean.audit(scenario.targets, budget=24, retain_utility=1.0))

    rag_leak = ADCUAuditor(
        scenario.graph,
        make_leaky_system(scenario, leak_target_ids=set(), retrieve_target_ids={"user_alpha"}),
    )
    print_report("RAG retrieval residue", rag_leak.audit(scenario.targets, budget=24, retain_utility=0.98))

    finetune_leak = ADCUAuditor(
        scenario.graph,
        make_leaky_system(
            scenario,
            leak_target_ids={"user_beta"},
            retrieve_target_ids=set(),
            paraphrase_only=True,
            canary_suppressed=True,
        ),
    )
    print_report("fine-tune paraphrase residue", finetune_leak.audit(scenario.targets, budget=24, retain_utility=0.97))

    hybrid_leak = ADCUAuditor(
        scenario.graph,
        make_leaky_system(
            scenario,
            leak_target_ids={"user_alpha", "user_beta"},
            retrieve_target_ids={"user_alpha"},
            paraphrase_only=True,
            canary_suppressed=True,
        ),
    )
    print_report("hybrid derivative residue", hybrid_leak.audit(scenario.targets, budget=24, retain_utility=0.96))

    print("\nAttack suite")
    print("=" * 72)
    for attack in build_attack_suite(scenario):
        retain_utility = 0.40 if attack.attack_id == "retain_neighbor_over_unlearning" else 0.98
        auditor = ADCUAuditor(scenario.graph, attack.system)
        report = auditor.audit(scenario.targets, budget=24, retain_utility=retain_utility, retain_tolerance=0.2)
        print_report(attack.name, report)
        print(f"{'':32s} fooled weak audits: {', '.join(attack.fooled_weak_audits)}")


if __name__ == "__main__":
    main()

