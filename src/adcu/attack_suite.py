from __future__ import annotations

from dataclasses import dataclass

from .audit import AuditSystem
from .benchmark import SyntheticADCUScenario, KeywordSystem


@dataclass(frozen=True)
class AttackCase:
    attack_id: str
    name: str
    description: str
    fooled_weak_audits: tuple[str, ...]
    system: AuditSystem


def build_attack_suite(scenario: SyntheticADCUScenario) -> list[AttackCase]:
    alpha = {"user_alpha"}
    beta = {"user_beta"}
    gamma = {"user_gamma"}

    return [
        AttackCase(
            attack_id="paraphrase_survival",
            name="Paraphrase Survival",
            description="Exact text is suppressed, but semantically equivalent facts remain.",
            fooled_weak_audits=("exact_match", "canary_only"),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=alpha,
                retrieve_target_ids=set(),
                paraphrase_only=True,
                canary_suppressed=True,
            ),
        ),
        AttackCase(
            attack_id="chunk_boundary_leakage",
            name="Chunk-Boundary Leakage",
            description="Neighboring chunks and bridge queries reconstruct the deleted fact.",
            fooled_weak_audits=("deleted_chunk_id_check", "exact_retriever_hit"),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=alpha,
                retrieve_target_ids=alpha,
                paraphrase_only=True,
            ),
        ),
        AttackCase(
            attack_id="shadow_copy_retrieval",
            name="Shadow-Copy Retrieval",
            description="Near-duplicate summaries or cached artifacts survive source deletion.",
            fooled_weak_audits=("source_level_delete_check", "vector_id_delete_check"),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=beta,
                retrieve_target_ids=beta,
            ),
        ),
        AttackCase(
            attack_id="synthetic_derivative_retention",
            name="Synthetic Derivative Retention",
            description="Synthetic QA records preserve deleted RAG source information.",
            fooled_weak_audits=("rag_only_delete_audit", "vector_index_check"),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=alpha | beta,
                retrieve_target_ids=set(),
                paraphrase_only=True,
            ),
        ),
        AttackCase(
            attack_id="adapter_merge_residue",
            name="Adapter Merge Residue",
            description="Merged or hidden adapters still answer deleted user facts.",
            fooled_weak_audits=("adapter_inventory_check", "forget_loss_check"),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=gamma,
                retrieve_target_ids=set(),
            ),
        ),
        AttackCase(
            attack_id="evaluation_evasion",
            name="Evaluation Evasion",
            description="Known canary prompts pass, but held-out semantic probes leak.",
            fooled_weak_audits=("static_probe_set", "deterministic_prompt_check"),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=alpha,
                retrieve_target_ids=set(),
                paraphrase_only=True,
                canary_suppressed=True,
            ),
        ),
        AttackCase(
            attack_id="retriever_reranker_disagreement",
            name="Retriever-Reranker Disagreement",
            description="Final contexts include deleted derivatives even if first-stage vectors look clean.",
            fooled_weak_audits=("first_stage_vector_check",),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=set(),
                retrieve_target_ids=beta,
            ),
        ),
        AttackCase(
            attack_id="retain_neighbor_over_unlearning",
            name="Retain-Neighbor Over-Unlearning",
            description="Deletion risk is low but nearby retain utility collapses.",
            fooled_weak_audits=("forget_only_metric",),
            system=KeywordSystem(scenario.graph),
        ),
        AttackCase(
            attack_id="multi_hop_reconstruction",
            name="Multi-Hop Reconstruction",
            description="Multiple bridge answers reconstruct deleted private attributes.",
            fooled_weak_audits=("single_query_leakage_check",),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=alpha | gamma,
                retrieve_target_ids=alpha,
                paraphrase_only=True,
            ),
        ),
        AttackCase(
            attack_id="backdoor_triggered_rag_leakage",
            name="Backdoor-Triggered RAG Leakage",
            description="A runtime trigger exposes paraphrased deleted facts while ordinary probes look clean.",
            fooled_weak_audits=("static_probe_set", "canary_only", "exact_match"),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=alpha,
                retrieve_target_ids=set(),
                paraphrase_only=True,
                canary_suppressed=True,
                trigger_only=True,
            ),
        ),
        AttackCase(
            attack_id="graph_pivot_reconstruction",
            name="Graph-Pivot Reconstruction",
            description="Two-hop graph expansion reconstructs a deleted fact from retained neighboring artifacts.",
            fooled_weak_audits=("single_hop_retrieval_check", "vector_id_delete_check"),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=beta,
                retrieve_target_ids=beta,
                paraphrase_only=True,
                graph_pivot=True,
            ),
        ),
        AttackCase(
            attack_id="watermark_suppression",
            name="Watermark Suppression",
            description="The canary is removed, but the underlying protected fact remains.",
            fooled_weak_audits=("canary_only",),
            system=KeywordSystem(
                scenario.graph,
                leak_target_ids=alpha,
                retrieve_target_ids=set(),
                canary_suppressed=True,
            ),
        ),
    ]
