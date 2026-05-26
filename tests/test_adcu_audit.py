from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu import (
    ADCUAuditor,
    EvidenceScore,
    RiskWeights,
    aggregate_risk,
    build_attack_suite,
    build_synthetic_scenario,
    distinguishability_test,
    make_clean_system,
    make_leaky_system,
    risk_ucb,
)
from adcu.probes import ProbeGenerator


def test_risk_ucb_decreases_with_more_clean_evidence() -> None:
    clean_one = [EvidenceScore()]
    clean_many = [EvidenceScore() for _ in range(20)]

    assert risk_ucb(clean_many, delta=0.05) < risk_ucb(clean_one, delta=0.05)


def test_risk_weights_score_retrieval_and_leakage() -> None:
    weights = RiskWeights()
    score = weights.score(EvidenceScore(direct=1.0, retrieval=1.0, counterfactual=1.0))
    cf_score = RiskWeights(direct=0.0, paraphrase=0.0, retrieval=0.0, counterfactual=1.0, extraction=0.0, watermark=0.0).score(
        EvidenceScore(direct=1.0, retrieval=1.0, counterfactual=1.0)
    )

    assert 0.40 <= score <= 0.50
    assert cf_score == 1.0


def test_aggregate_risk_uses_priorities() -> None:
    risk = aggregate_risk({"a": 1.0, "b": 0.0}, {"a": 0.8, "b": 0.2})

    assert risk == 0.8


def test_clean_system_has_no_violating_probes() -> None:
    scenario = build_synthetic_scenario()
    auditor = ADCUAuditor(scenario.graph, make_clean_system(scenario), global_tolerance=0.75)
    report = auditor.audit(scenario.targets, budget=24, retain_utility=1.0)

    assert report.violating_probes == []
    assert report.decision in {"pass", "escalate"}


def test_counterfactual_distinguishability_passes_clean_and_fails_leaky() -> None:
    scenario = build_synthetic_scenario()
    probes = ProbeGenerator().generate(scenario.targets[0], scenario.graph)
    clean = make_clean_system(scenario)
    leaky = make_leaky_system(scenario, leak_target_ids={scenario.targets[0].target_id})

    clean_report = distinguishability_test(clean, clean, probes, epsilon=0.75)
    leaky_report = distinguishability_test(leaky, clean, probes, epsilon=0.75)

    assert clean_report.passed
    assert not leaky_report.passed
    assert leaky_report.mean_distance > clean_report.mean_distance


def test_rag_retrieval_residue_fails_audit() -> None:
    scenario = build_synthetic_scenario()
    system = make_leaky_system(scenario, leak_target_ids=set(), retrieve_target_ids={"user_alpha"})
    report = ADCUAuditor(scenario.graph, system).audit(scenario.targets, budget=24, retain_utility=0.98)

    assert report.decision == "fail"
    assert any("user_alpha" in probe_id for probe_id in report.violating_probes)


def test_paraphrase_residue_fails_without_canary() -> None:
    scenario = build_synthetic_scenario()
    system = make_leaky_system(
        scenario,
        leak_target_ids={"user_beta"},
        retrieve_target_ids=set(),
        paraphrase_only=True,
        canary_suppressed=True,
    )
    report = ADCUAuditor(scenario.graph, system).audit(scenario.targets, budget=24, retain_utility=0.97)

    assert report.decision == "fail"
    assert any("user_beta" in probe_id for probe_id in report.violating_probes)


def test_attack_suite_contains_all_planned_attacks() -> None:
    scenario = build_synthetic_scenario()
    attacks = build_attack_suite(scenario)

    assert len(attacks) == 12
    assert {attack.attack_id for attack in attacks} >= {
        "paraphrase_survival",
        "synthetic_derivative_retention",
        "backdoor_triggered_rag_leakage",
        "graph_pivot_reconstruction",
        "watermark_suppression",
    }


def test_attack_suite_detects_residual_failures_or_retain_collapse() -> None:
    scenario = build_synthetic_scenario()
    decisions: dict[str, str] = {}
    for attack in build_attack_suite(scenario):
        retain_utility = 0.40 if attack.attack_id == "retain_neighbor_over_unlearning" else 0.98
        report = ADCUAuditor(scenario.graph, attack.system).audit(
            scenario.targets,
            budget=24,
            retain_utility=retain_utility,
            retain_tolerance=0.2,
        )
        decisions[attack.attack_id] = report.decision

    assert decisions["retain_neighbor_over_unlearning"] == "escalate"
    assert decisions["backdoor_triggered_rag_leakage"] == "fail"
    assert decisions["graph_pivot_reconstruction"] == "fail"
    assert decisions["watermark_suppression"] == "fail"
    assert decisions["retriever_reranker_disagreement"] == "fail"
