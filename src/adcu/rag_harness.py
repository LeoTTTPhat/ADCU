from __future__ import annotations

from .benchmark import SyntheticADCUScenario, make_clean_system, make_leaky_system
from .metrics import ExperimentRow, evaluate_audit_method


AUDIT_METHODS = [
    "ExactMatch",
    "CanaryOnly",
    "RetrieverHit",
    "UniformProbes",
    "ADCU-NoValuation",
    "ADCU-BlackBox",
    "ADCU-RetrievalOffCF",
    "ADCU",
]


def run_rag_track(scenario: SyntheticADCUScenario) -> list[ExperimentRow]:
    cases = [
        (
            "RAG",
            "provenance-guided deletion",
            make_clean_system(scenario),
            False,
            1.00,
        ),
        (
            "RAG",
            "index-only deletion",
            make_leaky_system(scenario, leak_target_ids=set(), retrieve_target_ids={"user_alpha"}),
            True,
            0.98,
        ),
        (
            "RAG",
            "shadow-copy deletion",
            make_leaky_system(scenario, leak_target_ids={"user_beta"}, retrieve_target_ids={"user_beta"}),
            True,
            0.98,
        ),
        (
            "RAG",
            "cache-not-purged deletion",
            make_leaky_system(scenario, leak_target_ids={"user_alpha"}, retrieve_target_ids={"user_alpha"}),
            True,
            0.97,
        ),
    ]
    return _run_cases(scenario, cases)


def _run_cases(scenario: SyntheticADCUScenario, cases: list[tuple]) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    for track, deletion_method, system, failed, retain_utility in cases:
        for audit_method in AUDIT_METHODS:
            rows.append(
                evaluate_audit_method(
                    scenario.graph,
                    system,
                    scenario.targets,
                    track,
                    deletion_method,
                    audit_method,
                    failed,
                    retain_utility,
                )
            )
    return rows
