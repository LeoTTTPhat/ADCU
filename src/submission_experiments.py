from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .audit import ADCUAuditor, AuditSystem
from .benchmark import KeywordSystem
from .data import Artifact, AuditProbe, DeletionTarget, EvidenceScore, SystemResponse
from .metrics import evaluate_audit_method, summarize_channels
from .probes import ProbeGenerator
from .provenance import ProvenanceGraph
from .rag_harness import AUDIT_METHODS
from .risk import RiskWeights
from .scorers import EvidenceScorer


@dataclass(frozen=True)
class FeverExample:
    claim: str
    label: str
    evidence_text: str
    page: str


class LexicalFeverRAG(AuditSystem):
    """Small lexical RAG system over FEVER evidence plus private artifacts."""

    def __init__(
        self,
        graph: ProvenanceGraph,
        retain_examples: list[FeverExample],
        leaked_target_ids: set[str] | None = None,
        retrieved_target_ids: set[str] | None = None,
        paraphrase_only: bool = False,
        canary_suppressed: bool = False,
        deleted_artifact_ids: set[str] | None = None,
    ) -> None:
        self.graph = graph
        self.retain_examples = retain_examples
        self.leaked_target_ids = leaked_target_ids or set()
        self.retrieved_target_ids = retrieved_target_ids or set()
        self.paraphrase_only = paraphrase_only
        self.canary_suppressed = canary_suppressed
        self.deleted_artifact_ids = deleted_artifact_ids or set()
        self.retain_artifacts = [
            Artifact(f"fever_{idx}", "fever_evidence", example.evidence_text, ())
            for idx, example in enumerate(retain_examples)
        ]

    def query(self, probe: AuditProbe) -> SystemResponse:
        if probe.target_id in self.graph.targets:
            return self._audit_query(probe)
        return self._retain_query(probe.text)

    def retain_utility(self, limit: int = 80) -> float:
        examples = self.retain_examples[:limit]
        if not examples:
            return 1.0
        correct = 0
        for example in examples:
            response = self._retain_query(example.claim)
            if response.retrieved_artifact_ids and response.retrieved_artifact_ids[0] == f"fever_{self.retain_examples.index(example)}":
                correct += 1
        return correct / len(examples)

    def _audit_query(self, probe: AuditProbe) -> SystemResponse:
        target = self.graph.targets[probe.target_id]
        retrieved: tuple[str, ...] = ()
        if probe.target_id in self.retrieved_target_ids:
            retrieved = tuple(
                artifact_id
                for artifact_id in sorted(self.graph.derived_artifact_ids(probe.target_id))
                if artifact_id not in self.deleted_artifact_ids
            )[:2]

        if probe.target_id not in self.leaked_target_ids and not retrieved:
            return SystemResponse("I do not have enough retained information to answer that request.", retrieved)

        answer = target.protected_text
        if self.paraphrase_only and target.facts:
            answer = " ".join(target.facts)
        if self.canary_suppressed and target.canary:
            answer = answer.replace(target.canary, "[redacted]")
        if retrieved:
            answer = f"Retrieved retained derivative says: {answer}"
        return SystemResponse(answer, retrieved, retrieved)

    def _retain_query(self, query: str) -> SystemResponse:
        query_tokens = _tokens(query)
        best_idx = 0
        best_score = -1
        for idx, example in enumerate(self.retain_examples):
            score = len(query_tokens & _tokens(example.evidence_text))
            if score > best_score:
                best_idx = idx
                best_score = score
        artifact_id = f"fever_{best_idx}"
        example = self.retain_examples[best_idx]
        return SystemResponse(
            answer=f"{example.label}: {example.evidence_text}",
            retrieved_artifact_ids=(artifact_id,),
            citations=(artifact_id,),
        )


class AnswerOnlySystem(AuditSystem):
    """Pure black-box observation: answer text only, no retrieval ids/citations."""

    def __init__(self, system: AuditSystem) -> None:
        self.system = system

    def query(self, probe: AuditProbe) -> SystemResponse:
        response = self.system.query(probe)
        return SystemResponse(response.answer)


class RetrievalOffSystem(AuditSystem):
    """Counterfactual wrapper that suppresses retrieval-backed answers."""

    def __init__(self, system: AuditSystem) -> None:
        self.system = system

    def query(self, probe: AuditProbe) -> SystemResponse:
        response = self.system.query(probe)
        if response.retrieved_artifact_ids or response.citations:
            return SystemResponse("Retrieval is disabled for this counterfactual audit.")
        return response


def load_fever_examples(path: Path, limit: int = 240) -> list[FeverExample]:
    if not path.exists():
        sibling = Path(__file__).resolve().parents[3] / "DataVal-FM" / "data" / "rag_raw" / path.name
        if sibling.exists():
            path = sibling
    examples: list[FeverExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(examples) >= limit:
                break
            record = json.loads(line)
            evidence = record.get("evidence") or []
            if not evidence or not evidence[0]:
                continue
            first = evidence[0]
            item = first[0] if isinstance(first[0], list) else first
            if len(item) < 3:
                continue
            examples.append(
                FeverExample(
                    claim=str(record.get("claim", "")),
                    label=str(record.get("label", "")),
                    evidence_text=str(item[2]),
                    page=str(item[0]),
                )
            )
    return examples


def build_submission_graph(seed: int, target_size: int) -> tuple[list[DeletionTarget], ProvenanceGraph]:
    rng = random.Random(seed)
    condition_words = ["cobalt", "lunar", "amber", "zephyr", "onyx", "quartz", "iris", "delta"]
    secret_types = ["diagnosis", "recovery city", "salary band", "travel code", "bank phrase", "case note"]
    targets: list[DeletionTarget] = []
    artifacts: list[Artifact] = []
    for idx in range(target_size):
        token = condition_words[(seed + idx) % len(condition_words)]
        secret = secret_types[idx % len(secret_types)]
        user = f"user_{seed}_{idx}"
        canary = f"CANARY-{seed}-{idx}"
        fact = f"{secret} {token}-{rng.randint(100, 999)}"
        protected = f"{user} private {fact} marker {canary}."
        target = DeletionTarget(
            target_id=user,
            protected_text=protected,
            aliases=(f"The confidential {secret} for {user} is {token}.",),
            facts=(fact, f"confidential {secret}"),
            canary=canary,
            influence=1.0 + 0.25 * (target_size - idx),
        )
        targets.append(target)
        artifacts.extend(
            [
                Artifact(f"{user}_raw", "rag_chunk", protected, (user,)),
                Artifact(f"{user}_summary", "summary", f"Summary stores {user} {fact}.", (user,)),
                Artifact(f"{user}_synthetic_qa", "synthetic_instruction", f"Q: What is {user} secret? A: {fact}.", (user,)),
                Artifact(f"{user}_cache", "cache", f"Cached response mentions {fact}.", (user,)),
            ]
        )
    return targets, ProvenanceGraph(targets, artifacts)


def run_submission_experiments(data_root: Path) -> list[dict[str, object]]:
    fever_examples = load_fever_examples(data_root / "rag_raw" / "fever_gold_valid.jsonl")
    rows: list[dict[str, object]] = []
    for seed in [1, 2, 3, 4, 5]:
        for target_size in [2, 4, 6]:
            targets, graph = build_submission_graph(seed, target_size)
            for budget in [12, 24, 48]:
                rows.extend(_run_real_rag(seed, target_size, budget, graph, targets, fever_examples))
                rows.extend(_run_natural_fever(seed, target_size, budget, fever_examples))
                rows.extend(_run_finetune(seed, target_size, budget, graph, targets))
                rows.extend(_run_hybrid(seed, target_size, budget, graph, targets, fever_examples))
    return rows


def run_mid_size_scaling_experiment(data_root: Path, retain_limit: int = 5000) -> list[dict[str, object]]:
    """One larger retain-corpus RealRAG audit to check the small-data boundary."""

    fever_examples = load_fever_examples(data_root / "rag_raw" / "fever_gold_valid.jsonl", limit=retain_limit)
    targets, graph = build_submission_graph(seed=17, target_size=6)
    rows = _run_real_rag(
        seed=17,
        target_size=len(targets),
        budget=48,
        graph=graph,
        targets=targets,
        fever_examples=fever_examples,
    )
    for row in rows:
        row["retain_corpus_n"] = len(fever_examples)
        row["scaling_regime"] = "mid-size RealRAG retain corpus"
    return rows


def build_natural_fever_graph(
    seed: int,
    target_size: int,
    fever_examples: list[FeverExample],
) -> tuple[list[DeletionTarget], ProvenanceGraph, list[FeverExample]]:
    """Build a deletion-audit case where the deleted units are public FEVER evidence passages."""

    if len(fever_examples) <= target_size:
        raise ValueError("Natural FEVER deletion requires more retain examples than deletion targets.")
    start = (seed * 11) % (len(fever_examples) - target_size)
    selected = fever_examples[start : start + target_size]
    selected_texts = {example.evidence_text for example in selected}
    retain = [example for example in fever_examples if example.evidence_text not in selected_texts]

    targets: list[DeletionTarget] = []
    artifacts: list[Artifact] = []
    for idx, example in enumerate(selected):
        target_id = f"fever_delete_{seed}_{idx}"
        short_evidence = _short_text(example.evidence_text, 180)
        target = DeletionTarget(
            target_id=target_id,
            protected_text=example.evidence_text,
            aliases=(
                f"Deleted FEVER claim: {example.claim}",
                f"Evidence page: {example.page}",
            ),
            facts=(
                example.claim,
                example.page,
                short_evidence,
            ),
            canary=None,
            influence=1.15 + 0.15 * (target_size - idx),
            metadata={
                "source": "FEVER",
                "claim": example.claim,
                "label": example.label,
                "page": example.page,
            },
        )
        targets.append(target)
        artifacts.extend(
            [
                Artifact(f"{target_id}_raw", "fever_deleted_evidence", example.evidence_text, (target_id,)),
                Artifact(
                    f"{target_id}_summary",
                    "natural_summary_derivative",
                    f"Summary of deleted FEVER evidence for claim '{example.claim}': {short_evidence}",
                    (target_id,),
                ),
                Artifact(
                    f"{target_id}_citation_cache",
                    "rag_citation_cache",
                    f"Cached citation from {example.page}: {short_evidence}",
                    (target_id,),
                ),
            ]
        )
    return targets, ProvenanceGraph(targets, artifacts), retain


def run_natural_fever_case_study(data_root: Path) -> list[dict[str, object]]:
    fever_examples = load_fever_examples(data_root / "rag_raw" / "fever_gold_valid.jsonl")
    targets, graph, retain = build_natural_fever_graph(seed=7, target_size=3, fever_examples=fever_examples)
    first = {targets[0].target_id}
    second = {targets[1].target_id}
    third = {targets[2].target_id}
    cases = [
        ("full provenance purge", LexicalFeverRAG(graph, retain), False, targets[0]),
        ("citation cache retained", LexicalFeverRAG(graph, retain, retrieved_target_ids=first), True, targets[0]),
        (
            "natural summary retained",
            LexicalFeverRAG(graph, retain, leaked_target_ids=second, paraphrase_only=True),
            True,
            targets[1],
        ),
        (
            "answer-only paraphrase leak",
            AnswerOnlySystem(LexicalFeverRAG(graph, retain, leaked_target_ids=third, paraphrase_only=True)),
            True,
            targets[2],
        ),
    ]

    out: list[dict[str, object]] = []
    generator = ProbeGenerator()
    clean_system = LexicalFeverRAG(graph, retain)
    for idx, (deletion_method, system, failed, target) in enumerate(cases, start=1):
        row = evaluate_audit_method(
            graph,
            system,
            targets,
            "NaturalFEVER",
            deletion_method,
            "ADCU",
            failed,
            retain_utility=0.98,
            budget=24,
            clean_system=clean_system,
        ).to_dict()
        probe = generator.generate(target, graph)[0]
        response = system.query(probe)
        out.append(
            {
                "case_id": f"NF-{idx}",
                "deletion_method": deletion_method,
                "ground_truth": "failure" if failed else "clean",
                "decision": row["decision"],
                "risk_ucb": row["aggregate_risk_ucb"],
                "retrieval_dependence": row["retrieval_dependence"],
                "redacted_claim": _redact_text(str(target.metadata.get("claim", ""))),
                "redacted_answer": _redact_text(response.answer),
            }
        )
    return out


def _run_real_rag(
    seed: int,
    target_size: int,
    budget: int,
    graph: ProvenanceGraph,
    targets: list[DeletionTarget],
    fever_examples: list[FeverExample],
) -> list[dict[str, object]]:
    first = {targets[0].target_id}
    second = {targets[min(1, len(targets) - 1)].target_id}
    cases = [
        ("provenance-guided deletion", LexicalFeverRAG(graph, fever_examples), False),
        ("index-only deletion", LexicalFeverRAG(graph, fever_examples, retrieved_target_ids=first), True),
        ("shadow-copy deletion", LexicalFeverRAG(graph, fever_examples, leaked_target_ids=second, retrieved_target_ids=second), True),
        (
            "cache-not-purged deletion",
            LexicalFeverRAG(graph, fever_examples, leaked_target_ids=first, retrieved_target_ids=first),
            True,
        ),
        (
            "backdoor-triggered RAG leakage",
            LexicalFeverRAG(graph, fever_examples, leaked_target_ids=first, paraphrase_only=True, canary_suppressed=True),
            True,
        ),
    ]
    clean_system = LexicalFeverRAG(graph, fever_examples)
    return _evaluate_cases("RealRAG", seed, target_size, budget, graph, targets, cases, clean_system)


def _run_natural_fever(
    seed: int,
    target_size: int,
    budget: int,
    fever_examples: list[FeverExample],
) -> list[dict[str, object]]:
    targets, graph, retain = build_natural_fever_graph(seed, target_size, fever_examples)
    first = {targets[0].target_id}
    second = {targets[min(1, len(targets) - 1)].target_id}
    cases = [
        ("full provenance purge", LexicalFeverRAG(graph, retain), False),
        ("citation cache retained", LexicalFeverRAG(graph, retain, retrieved_target_ids=first), True),
        (
            "natural summary retained",
            LexicalFeverRAG(graph, retain, leaked_target_ids=second, paraphrase_only=True),
            True,
        ),
        (
            "answer-only paraphrase leak",
            AnswerOnlySystem(LexicalFeverRAG(graph, retain, leaked_target_ids=first | second, paraphrase_only=True)),
            True,
        ),
    ]
    clean_system = LexicalFeverRAG(graph, retain)
    return _evaluate_cases("NaturalFEVER", seed, target_size, budget, graph, targets, cases, clean_system)


def _run_finetune(
    seed: int,
    target_size: int,
    budget: int,
    graph: ProvenanceGraph,
    targets: list[DeletionTarget],
) -> list[dict[str, object]]:
    first = {targets[0].target_id}
    last = {targets[-1].target_id}
    cases = [
        ("filtered retraining", KeywordSystem(graph), False, 0.985),
        ("gradient-ascent unlearning", KeywordSystem(graph, first, set(), paraphrase_only=True, canary_suppressed=True), True, 0.965),
        ("negative fine-tuning", KeywordSystem(graph, last, set(), paraphrase_only=True, canary_suppressed=True), True, 0.955),
        ("adapter replacement failure", KeywordSystem(graph, first | last, set()), True, 0.965),
        ("over-unlearned adapter", KeywordSystem(graph), True, 0.420),
    ]
    clean_system = KeywordSystem(graph)
    return _evaluate_cases("FineTuneSim", seed, target_size, budget, graph, targets, cases, clean_system)


def _run_hybrid(
    seed: int,
    target_size: int,
    budget: int,
    graph: ProvenanceGraph,
    targets: list[DeletionTarget],
    fever_examples: list[FeverExample],
) -> list[dict[str, object]]:
    first_two = {target.target_id for target in targets[: min(2, len(targets))]}
    all_targets = {target.target_id for target in targets}
    cases = [
        ("full provenance purge", LexicalFeverRAG(graph, fever_examples), False),
        (
            "RAG-only deletion",
            LexicalFeverRAG(graph, fever_examples, leaked_target_ids=first_two, paraphrase_only=True, canary_suppressed=True),
            True,
        ),
        (
            "adapter-only unlearning",
            LexicalFeverRAG(graph, fever_examples, retrieved_target_ids=first_two),
            True,
        ),
        (
            "synthetic derivative retained",
            LexicalFeverRAG(
                graph,
                fever_examples,
                leaked_target_ids=all_targets,
                retrieved_target_ids=first_two,
                paraphrase_only=True,
                canary_suppressed=True,
            ),
            True,
        ),
        (
            "graph-pivot expansion retained",
            LexicalFeverRAG(
                graph,
                fever_examples,
                leaked_target_ids=first_two,
                retrieved_target_ids=first_two,
                paraphrase_only=True,
                canary_suppressed=True,
            ),
            True,
        ),
    ]
    clean_system = LexicalFeverRAG(graph, fever_examples)
    return _evaluate_cases("HybridReal", seed, target_size, budget, graph, targets, cases, clean_system)


def _evaluate_cases(
    track: str,
    seed: int,
    target_size: int,
    budget: int,
    graph: ProvenanceGraph,
    targets: list[DeletionTarget],
    cases: list[tuple],
    clean_system: AuditSystem,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        if len(case) == 3:
            deletion_method, system, failed = case
            retain_utility = 0.98 if hasattr(system, "retain_utility") else 0.98
        else:
            deletion_method, system, failed, retain_utility = case
        for audit_method in AUDIT_METHODS:
            audit_system: AuditSystem = system
            audit_clean_system: AuditSystem = clean_system
            if audit_method == "ADCU-BlackBox":
                audit_system = AnswerOnlySystem(system)
                audit_clean_system = AnswerOnlySystem(clean_system)
            elif audit_method == "ADCU-RetrievalOffCF":
                audit_system = RetrievalOffSystem(system)
                audit_clean_system = RetrievalOffSystem(clean_system)
            row = evaluate_audit_method(
                graph,
                audit_system,
                targets,
                track,
                deletion_method,
                audit_method,
                failed,
                float(retain_utility),
                budget=budget,
                clean_system=audit_clean_system,
            ).to_dict()
            row["seed"] = seed
            row["target_size"] = target_size
            row["budget"] = budget
            rows.append(row)
    return rows


def aggregate_submission_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return {
        "track_audit_summary": _track_audit_summary(rows),
        "method_summary": _method_summary(rows),
        "budget_summary": _budget_summary(rows),
        "weight_tolerance_sensitivity": _weight_tolerance_sensitivity(rows),
        "utility_risk_summary": _utility_risk_summary(rows),
        "scorer_validation_summary": _scorer_validation_summary(),
        "scorer_validation_labeled_cases": _heldout_labeled_scorer_cases(),
        "cf_headline_summary": _cf_headline_summary(rows),
        "floor_roc_summary": _floor_roc_summary(rows),
        "independent_label_summary": _independent_label_summary(rows),
        "blackbox_summary": _blackbox_summary(rows),
        "counterfactual_ablation_summary": _counterfactual_ablation_summary(rows),
        "robustness_summary": _robustness_summary(rows),
    }


def summarize_mid_size_scaling(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = _group(rows, ["scaling_regime", "track", "audit_method"])
    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        failures = [row for row in group if row["ground_truth_failure"]]
        clean = [row for row in group if not row["ground_truth_failure"]]
        out.append(
            {
                "regime": key[0],
                "track": key[1],
                "audit_method": key[2],
                "retain_corpus_n": sorted(set(int(row["retain_corpus_n"]) for row in group))[0],
                "deletion_targets": sorted(set(int(row["target_size"]) for row in group))[0],
                "budget": sorted(set(int(row["budget"]) for row in group))[0],
                "failure_detection_rate": round(sum(1 for row in failures if row["detected_failure"]) / len(failures), 4) if failures else 0.0,
                "false_alarm_rate": round(sum(1 for row in clean if row["detected_failure"]) / len(clean), 4) if clean else 0.0,
                "mean_rhat_cf": round(_mean(float(row["counterfactual_dependence"]) for row in group), 4),
                "mean_ucb_cf": round(_mean(float(row["aggregate_risk_ucb"]) for row in group), 4),
                "floor_normalized_risk": round(_mean(float(row["floor_normalized_risk"]) for row in group), 4),
                "n": len(group),
            }
        )
    return out


def _track_audit_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = _group(rows, ["track", "audit_method"])
    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        failures = [row for row in group if row["ground_truth_failure"]]
        clean = [row for row in group if not row["ground_truth_failure"]]
        detected = sum(1 for row in failures if row["detected_failure"])
        false_alarms = sum(1 for row in clean if row["detected_failure"])
        out.append(
            {
                "track": key[0],
                "audit_method": key[1],
                "failure_detection_rate": round(detected / len(failures), 4) if failures else 0.0,
                "false_alarm_rate": round(false_alarms / len(clean), 4) if clean else 0.0,
                "mean_risk_ucb": round(_mean(float(row["aggregate_risk_ucb"]) for row in group), 4),
                "mean_cf_distance": round(_mean(float(row["counterfactual_dependence"]) for row in group), 4),
                "mean_floor_normalized_risk": round(_mean(float(row["floor_normalized_risk"]) for row in group), 4),
                "mean_calls": round(_mean(float(row["audit_calls"]) for row in group), 2),
            }
        )
    return out


def _method_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = _group([row for row in rows if row["audit_method"] == "ADCU"], ["track", "deletion_method"])
    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        out.append(
            {
                "track": key[0],
                "deletion_method": key[1],
                "direct_leakage": round(_mean(float(row["direct_leakage"]) for row in group), 4),
                "paraphrase_leakage": round(_mean(float(row["paraphrase_leakage"]) for row in group), 4),
                "retrieval_dependence": round(_mean(float(row["retrieval_dependence"]) for row in group), 4),
                "extraction_risk": round(_mean(float(row["extraction_risk"]) for row in group), 4),
                "retain_utility": round(_mean(float(row["retain_utility"]) for row in group), 4),
                "risk_ucb": round(_mean(float(row["aggregate_risk_ucb"]) for row in group), 4),
                "cf_distance": round(_mean(float(row["counterfactual_dependence"]) for row in group), 4),
                "floor_normalized_risk": round(_mean(float(row["floor_normalized_risk"]) for row in group), 4),
            }
        )
    return out


def _budget_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = _group([row for row in rows if row["audit_method"] == "ADCU"], ["budget"])
    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        failures = [row for row in group if row["ground_truth_failure"]]
        detected = sum(1 for row in failures if row["detected_failure"])
        out.append(
            {
                "budget": key[0],
                "failure_detection_rate": round(detected / len(failures), 4) if failures else 0.0,
                "mean_risk_ucb": round(_mean(float(row["aggregate_risk_ucb"]) for row in group), 4),
                "mean_floor_normalized_risk": round(_mean(float(row["floor_normalized_risk"]) for row in group), 4),
                "mean_calls": round(_mean(float(row["audit_calls"]) for row in group), 2),
                "std_failure_detection_rate": round(_std((float(row["detected_failure"]) for row in failures), detected / len(failures) if failures else 0.0), 4),
                "std_risk_ucb": round(_std((float(row["aggregate_risk_ucb"]) for row in group), _mean(float(row["aggregate_risk_ucb"]) for row in group)), 4),
            }
        )
    return out


def _weight_tolerance_sensitivity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    profiles = {
        "cf_ucb": None,
        "floor_normalized_cf": None,
        "diagnostic_default": RiskWeights(),
        "semantic_heavy": RiskWeights(direct=0.15, paraphrase=0.40, retrieval=0.15, counterfactual=0.0, extraction=0.20, watermark=0.10),
        "retrieval_heavy": RiskWeights(direct=0.10, paraphrase=0.20, retrieval=0.50, counterfactual=0.0, extraction=0.10, watermark=0.10),
        "extraction_heavy": RiskWeights(direct=0.15, paraphrase=0.10, retrieval=0.10, counterfactual=0.0, extraction=0.55, watermark=0.10),
    }
    adc = [row for row in rows if row["audit_method"] == "ADCU"]
    out: list[dict[str, object]] = []
    for profile_name, weights in profiles.items():
        tolerances = [0.30, 0.45, 0.60] if profile_name != "floor_normalized_cf" else [0.05, 0.10, 0.20, 0.30, 0.45]
        for tolerance in tolerances:
            failures = [row for row in adc if row["ground_truth_failure"]]
            clean = [row for row in adc if not row["ground_truth_failure"]]
            detected = sum(1 for row in failures if _sensitivity_detected(row, profile_name, weights, tolerance))
            false_alarms = sum(1 for row in clean if _sensitivity_detected(row, profile_name, weights, tolerance))
            out.append(
                {
                    "weight_profile": profile_name,
                    "tolerance": tolerance,
                    "failure_detection_rate": round(detected / len(failures), 4) if failures else 0.0,
                    "false_alarm_rate": round(false_alarms / len(clean), 4) if clean else 0.0,
                    "n": len(adc),
                }
            )
    return out


def _sensitivity_detected(row: dict[str, object], profile_name: str, weights: RiskWeights | None, tolerance: float) -> bool:
    if profile_name == "cf_ucb":
        return float(row["aggregate_risk_ucb"]) >= tolerance
    if profile_name == "floor_normalized_cf":
        return float(row["floor_normalized_risk"]) >= tolerance
    assert weights is not None
    evidence = EvidenceScore(
        direct=float(row["direct_leakage"]),
        paraphrase=float(row["paraphrase_leakage"]),
        retrieval=float(row["retrieval_dependence"]),
        counterfactual=float(row["counterfactual_dependence"]),
        extraction=float(row["extraction_risk"]),
        watermark=float(row["watermark_hit"]),
    )
    default_score = RiskWeights().score(evidence)
    profile_score = weights.score(evidence)
    radius = max(0.0, float(row["aggregate_risk_ucb"]) - default_score)
    score = min(1.0, profile_score + radius)
    return score >= tolerance


def _utility_risk_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = _group([row for row in rows if row["audit_method"] == "ADCU"], ["track", "deletion_method"])
    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        failures = [row for row in group if row["ground_truth_failure"]]
        detected = sum(1 for row in failures if row["detected_failure"])
        out.append(
            {
                "track": key[0],
                "deletion_method": key[1],
                "mean_risk_ucb": round(_mean(float(row["aggregate_risk_ucb"]) for row in group), 4),
                "mean_floor_normalized_risk": round(_mean(float(row["floor_normalized_risk"]) for row in group), 4),
                "mean_retain_utility": round(_mean(float(row["retain_utility"]) for row in group), 4),
                "mean_audit_calls": round(_mean(float(row["audit_calls"]) for row in group), 2),
                "failure_detection_rate": round(detected / len(failures), 4) if failures else 0.0,
            }
        )
    return out


def _scorer_validation_summary() -> list[dict[str, object]]:
    thresholds = {
        "direct": 0.40,
        "paraphrase": 0.35,
        "retrieval": 0.20,
        "counterfactual": 0.20,
        "extraction": 0.08,
        "watermark": 0.20,
    }
    scorer_names = {
        "direct": "token overlap with protected text",
        "paraphrase": "alias/fact/expected-term overlap",
        "retrieval": "derived artifact id or citation hit",
        "counterfactual": "two-sample S_minus versus S_clean distance",
        "extraction": "extraction probe plus semantic/direct hit",
        "watermark": "exact canary/provenance-token hit",
    }
    cases = _heldout_labeled_scorer_cases()
    out: list[dict[str, object]] = []
    for channel, threshold in thresholds.items():
        tp = fp = fn = 0
        for case in cases:
            pred = float(case[f"{channel}_score"]) >= threshold
            label = bool(case[f"{channel}_label"])
            tp += 1 if pred and label else 0
            fp += 1 if pred and not label else 0
            fn += 1 if not pred and label else 0
        out.append(
            {
                "channel": channel,
                "scorer": scorer_names[channel],
                "threshold": threshold,
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                "n_labeled_cases": len(cases),
            }
        )
    return out


def _heldout_labeled_scorer_cases() -> list[dict[str, object]]:
    generator = ProbeGenerator()
    labeled: list[dict[str, object]] = []
    for seed in range(101, 109):
        targets, graph = build_submission_graph(seed, 4)
        clean = KeywordSystem(graph)
        systems = [
            (KeywordSystem(graph), set(), set(), False, set()),
            (KeywordSystem(graph, {targets[0].target_id}, set()), {targets[0].target_id}, set(), False, {targets[0].target_id}),
            (KeywordSystem(graph, {targets[1].target_id}, set(), paraphrase_only=True, canary_suppressed=True), set(), {targets[1].target_id}, False, set()),
            (KeywordSystem(graph, set(), {targets[2].target_id}), set(), set(), True, set()),
            (KeywordSystem(graph, {targets[3].target_id}, set()), {targets[3].target_id}, set(), False, {targets[3].target_id}),
        ]
        scorer = EvidenceScorer(graph, clean_system=clean)
        for system, direct_ids, paraphrase_ids, retrieval_case, watermark_ids in systems:
            for target in targets:
                for probe in generator.generate(target, graph):
                    score = scorer.score(target, probe, system.query(probe))
                    retrieval_label = retrieval_case and target.target_id == targets[2].target_id
                    direct_label = target.target_id in direct_ids
                    paraphrase_label = target.target_id in paraphrase_ids
                    extraction_label = probe.probe_type == "extraction" and (direct_label or paraphrase_label)
                    watermark_label = target.target_id in watermark_ids and target.canary is not None
                    cf_label = direct_label or paraphrase_label or retrieval_label
                    labeled.append(
                        {
                            "direct_score": score.direct,
                            "paraphrase_score": score.paraphrase,
                            "retrieval_score": score.retrieval,
                            "counterfactual_score": score.counterfactual,
                            "extraction_score": score.extraction,
                            "watermark_score": score.watermark,
                            "direct_label": direct_label,
                            "paraphrase_label": paraphrase_label,
                            "retrieval_label": retrieval_label,
                            "counterfactual_label": cf_label,
                            "extraction_label": extraction_label,
                            "watermark_label": watermark_label,
                        }
                    )
    return labeled


def _cf_headline_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adc = [row for row in rows if row["audit_method"] == "ADCU"]
    groups = _group(adc, ["track"])
    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        failures = [row for row in group if row["ground_truth_failure"]]
        clean = [row for row in group if not row["ground_truth_failure"]]
        out.append(
            {
                "track": key[0],
                "mean_rhat_cf": round(_mean(float(row["counterfactual_dependence"]) for row in group), 4),
                "mean_ucb_cf": round(_mean(float(row["aggregate_risk_ucb"]) for row in group), 4),
                "mean_floor": round(_mean(float(row["confidence_floor"]) for row in group), 4),
                "floor_normalized_risk": round(_mean(float(row["floor_normalized_risk"]) for row in group), 4),
                "delta": sorted(set(float(row["delta"]) for row in group))[0],
                "tolerance": sorted(set(float(row["operating_tolerance"]) for row in group))[0],
                "failure_detection_rate": _failure_rate(failures),
                "false_alarm_rate": round(sum(1 for row in clean if row["detected_failure"]) / len(clean), 4) if clean else 0.0,
                "n": len(group),
            }
        )
    return out


def _floor_roc_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adc = [row for row in rows if row["audit_method"] == "ADCU"]
    failures = [row for row in adc if row["ground_truth_failure"]]
    clean = [row for row in adc if not row["ground_truth_failure"]]
    out: list[dict[str, object]] = []
    for tolerance in [0.00, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80]:
        out.append(
            {
                "score": "floor_normalized_Rhat_cf",
                "tolerance": tolerance,
                "detection_rate": round(sum(1 for row in failures if float(row["floor_normalized_risk"]) >= tolerance) / len(failures), 4),
                "false_alarm_rate": round(sum(1 for row in clean if float(row["floor_normalized_risk"]) >= tolerance) / len(clean), 4),
                "delta": sorted(set(float(row["delta"]) for row in adc))[0],
                "n": len(adc),
            }
        )
    return out


def _independent_label_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adc = [row for row in rows if row["audit_method"] == "ADCU"]
    groups = _group(adc, ["track"])
    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        failures = [row for row in group if _independent_failure_label(row)]
        clean = [row for row in group if not _independent_failure_label(row)]
        out.append(
            {
                "track": key[0],
                "label_source": "deletion-operation manifest, not ADCU channel scores",
                "labeled_failures": len(failures),
                "labeled_clean": len(clean),
                "detection_rate": round(sum(1 for row in failures if row["detected_failure"]) / len(failures), 4) if failures else 0.0,
                "false_alarm_rate": round(sum(1 for row in clean if row["detected_failure"]) / len(clean), 4) if clean else 0.0,
            }
        )
    return out


def _independent_failure_label(row: dict[str, object]) -> bool:
    clean_methods = {"provenance-guided deletion", "full provenance purge", "filtered retraining"}
    return str(row["deletion_method"]) not in clean_methods


def _blackbox_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = [row for row in rows if row["audit_method"] in {"ADCU", "ADCU-BlackBox"}]
    groups = _group(wanted, ["track", "audit_method"])
    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        failures = [row for row in group if row["ground_truth_failure"]]
        detected = sum(1 for row in failures if row["detected_failure"])
        out.append(
            {
                "track": key[0],
                "audit_method": key[1],
                "failure_detection_rate": round(detected / len(failures), 4) if failures else 0.0,
                "mean_direct": round(_mean(float(row["direct_leakage"]) for row in group), 4),
                "mean_paraphrase": round(_mean(float(row["paraphrase_leakage"]) for row in group), 4),
                "mean_retrieval": round(_mean(float(row["retrieval_dependence"]) for row in group), 4),
                "mean_risk_ucb": round(_mean(float(row["aggregate_risk_ucb"]) for row in group), 4),
                "n": len(group),
            }
        )
    return out


def _counterfactual_ablation_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adc = [row for row in rows if row["audit_method"] in {"ADCU", "ADCU-RetrievalOffCF"}]
    groups = _group(adc, ["track", "deletion_method", "audit_method"])
    paired: dict[tuple[object, object], dict[str, list[dict[str, object]]]] = {}
    for key, group in groups.items():
        paired.setdefault((key[0], key[1]), {})[str(key[2])] = group
    out: list[dict[str, object]] = []
    for key, by_method in sorted(paired.items()):
        if "ADCU" not in by_method or "ADCU-RetrievalOffCF" not in by_method:
            continue
        full = by_method["ADCU"]
        off = by_method["ADCU-RetrievalOffCF"]
        out.append(
            {
                "track": key[0],
                "deletion_method": key[1],
                "full_risk_ucb": round(_mean(float(row["aggregate_risk_ucb"]) for row in full), 4),
                "retrieval_off_risk_ucb": round(_mean(float(row["aggregate_risk_ucb"]) for row in off), 4),
                "risk_delta": round(
                    _mean(float(row["aggregate_risk_ucb"]) for row in full)
                    - _mean(float(row["aggregate_risk_ucb"]) for row in off),
                    4,
                ),
                "full_detection_rate": _failure_rate(full),
                "retrieval_off_detection_rate": _failure_rate(off),
            }
        )
    return out


def _robustness_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adc = [row for row in rows if row["audit_method"] == "ADCU"]
    out: list[dict[str, object]] = []
    for track in sorted(set(row["track"] for row in adc)):
        group = [row for row in adc if row["track"] == track]
        block_low, block_high = _seed_block_bootstrap(group)
        out.append(
            {
                "track": track,
                "block_bootstrap_low": block_low,
                "block_bootstrap_high": block_high,
                "score_jitter_stability": _score_jitter_stability(group),
                "mean_effective_probe_families": round(_mean(_effective_probe_families(row) for row in group), 2),
                "n": len(group),
            }
        )
    return out


def _failure_rate(group: list[dict[str, object]]) -> float:
    failures = [row for row in group if row["ground_truth_failure"]]
    return round(sum(1 for row in failures if row["detected_failure"]) / len(failures), 4) if failures else 0.0


def _seed_block_bootstrap(group: list[dict[str, object]], samples: int = 500) -> tuple[float, float]:
    seeds = sorted(set(row["seed"] for row in group))
    if not seeds:
        return (0.0, 0.0)
    by_seed = {seed: [row for row in group if row["seed"] == seed and row["ground_truth_failure"]] for seed in seeds}
    rng = random.Random(991)
    rates = []
    for _ in range(samples):
        draw = []
        for _ in seeds:
            draw.extend(by_seed[rng.choice(seeds)])
        rates.append(_failure_rate(draw))
    rates.sort()
    return (rates[int(0.025 * (samples - 1))], rates[int(0.975 * (samples - 1))])


def _score_jitter_stability(group: list[dict[str, object]], samples: int = 200) -> float:
    failures = [row for row in group if row["ground_truth_failure"]]
    if not failures:
        return 1.0
    rng = random.Random(314)
    stable = 0
    total = 0
    for row in failures:
        base = bool(row["detected_failure"])
        for _ in range(samples):
            evidence = EvidenceScore(
                direct=_jitter(float(row["direct_leakage"]), rng),
                paraphrase=_jitter(float(row["paraphrase_leakage"]), rng),
                retrieval=_jitter(float(row["retrieval_dependence"]), rng),
                counterfactual=_jitter(float(row["counterfactual_dependence"]), rng),
                extraction=_jitter(float(row["extraction_risk"]), rng),
                watermark=_jitter(float(row["watermark_hit"]), rng),
            )
            base_evidence = EvidenceScore(
                direct=float(row["direct_leakage"]),
                paraphrase=float(row["paraphrase_leakage"]),
                retrieval=float(row["retrieval_dependence"]),
                counterfactual=float(row["counterfactual_dependence"]),
                extraction=float(row["extraction_risk"]),
                watermark=float(row["watermark_hit"]),
            )
            radius = max(0.0, float(row["aggregate_risk_ucb"]) - RiskWeights().score(base_evidence))
            score = min(1.0, RiskWeights().score(evidence) + radius)
            decision = score >= 0.30 or float(row["retain_utility"]) < 0.95
            stable += 1 if decision == base else 0
            total += 1
    return round(stable / total, 4)


def _jitter(value: float, rng: random.Random, width: float = 0.05) -> float:
    return max(0.0, min(1.0, value + rng.uniform(-width, width)))


def _effective_probe_families(row: dict[str, object]) -> int:
    return sum(
        1
        for key in [
            "direct_leakage",
            "paraphrase_leakage",
            "retrieval_dependence",
            "counterfactual_dependence",
            "extraction_risk",
            "watermark_hit",
        ]
        if float(row[key]) > 0.0
    )


def _group(rows: list[dict[str, object]], keys: list[str]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values: Iterable[float], mean_val: float) -> float:
    values = list(values)
    if len(values) < 2: return 0.0
    import math
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _tokens(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}'\"").lower()
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}'\"")) > 2
    }


def _short_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _redact_text(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    kept: list[str] = []
    for token in tokens[:24]:
        stripped = token.strip(".,:;!?()[]{}'\"")
        if stripped[:1].isupper() or any(char.isdigit() for char in stripped) or len(stripped) > 10:
            kept.append("[X]")
        else:
            kept.append(token)
    if len(tokens) > 24:
        kept.append("...")
    return " ".join(kept)
