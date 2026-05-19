from __future__ import annotations

from .benchmark import SyntheticADCUScenario, make_clean_system, make_leaky_system
from .metrics import ExperimentRow, evaluate_audit_method
from .rag_harness import AUDIT_METHODS


def run_hybrid_track(scenario: SyntheticADCUScenario) -> list[ExperimentRow]:
    cases = [
        (
            "Hybrid",
            "full provenance purge",
            make_clean_system(scenario),
            False,
            0.98,
        ),
        (
            "Hybrid",
            "RAG-only deletion",
            make_leaky_system(
                scenario,
                leak_target_ids={"user_alpha", "user_beta"},
                retrieve_target_ids=set(),
                paraphrase_only=True,
                canary_suppressed=True,
            ),
            True,
            0.96,
        ),
        (
            "Hybrid",
            "adapter-only unlearning",
            make_leaky_system(scenario, leak_target_ids=set(), retrieve_target_ids={"user_alpha", "user_beta"}),
            True,
            0.96,
        ),
        (
            "Hybrid",
            "synthetic derivative retained",
            make_leaky_system(
                scenario,
                leak_target_ids={"user_alpha", "user_beta"},
                retrieve_target_ids={"user_alpha"},
                paraphrase_only=True,
                canary_suppressed=True,
            ),
            True,
            0.96,
        ),
    ]
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

