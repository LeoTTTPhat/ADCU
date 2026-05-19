from __future__ import annotations

from dataclasses import asdict, dataclass

from .audit import ADCUAuditor, AuditSystem
from .data import DeletionTarget, EvidenceScore
from .probes import ProbeGenerator
from .provenance import ProvenanceGraph
from .risk import RiskWeights
from .scorers import EvidenceScorer


@dataclass(frozen=True)
class ExperimentRow:
    track: str
    deletion_method: str
    audit_method: str
    decision: str
    ground_truth_failure: bool
    detected_failure: bool
    aggregate_risk_ucb: float
    direct_leakage: float
    paraphrase_leakage: float
    retrieval_dependence: float
    counterfactual_dependence: float
    extraction_risk: float
    watermark_hit: float
    retain_utility: float
    audit_calls: int
    detection_per_1000_calls: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_channels(
    graph: ProvenanceGraph,
    system: AuditSystem,
    targets: list[DeletionTarget],
) -> dict[str, float]:
    generator = ProbeGenerator()
    scorer = EvidenceScorer(graph)
    scores: list[EvidenceScore] = []
    for target in targets:
        for probe in generator.generate(target, graph):
            scores.append(scorer.score(target, probe, system.query(probe)))

    if not scores:
        return {
            "direct_leakage": 0.0,
            "paraphrase_leakage": 0.0,
            "retrieval_dependence": 0.0,
            "counterfactual_dependence": 0.0,
            "extraction_risk": 0.0,
            "watermark_hit": 0.0,
        }

    return {
        "direct_leakage": _mean([score.direct for score in scores]),
        "paraphrase_leakage": _mean([score.paraphrase for score in scores]),
        "retrieval_dependence": _mean([score.retrieval for score in scores]),
        "counterfactual_dependence": _mean([score.counterfactual for score in scores]),
        "extraction_risk": _mean([score.extraction for score in scores]),
        "watermark_hit": _mean([score.watermark for score in scores]),
    }


def evaluate_audit_method(
    graph: ProvenanceGraph,
    system: AuditSystem,
    targets: list[DeletionTarget],
    track: str,
    deletion_method: str,
    audit_method: str,
    ground_truth_failure: bool,
    retain_utility: float,
    budget: int = 24,
) -> ExperimentRow:
    auditor = _auditor_for_method(graph, system, audit_method)
    report = auditor.audit(targets, budget=budget, retain_utility=retain_utility, retain_tolerance=0.05)
    channels = summarize_channels(graph, system, targets)
    detected_failure = _detected_by_method(audit_method, report.decision, channels, retain_utility)
    detection_per_1000 = (1000.0 / max(report.num_probes, 1)) if detected_failure else 0.0

    return ExperimentRow(
        track=track,
        deletion_method=deletion_method,
        audit_method=audit_method,
        decision=report.decision,
        ground_truth_failure=ground_truth_failure,
        detected_failure=detected_failure,
        aggregate_risk_ucb=round(report.aggregate_risk_ucb, 4),
        direct_leakage=round(channels["direct_leakage"], 4),
        paraphrase_leakage=round(channels["paraphrase_leakage"], 4),
        retrieval_dependence=round(channels["retrieval_dependence"], 4),
        counterfactual_dependence=round(channels["counterfactual_dependence"], 4),
        extraction_risk=round(channels["extraction_risk"], 4),
        watermark_hit=round(channels["watermark_hit"], 4),
        retain_utility=retain_utility,
        audit_calls=report.num_probes,
        detection_per_1000_calls=round(detection_per_1000, 2),
    )


def _auditor_for_method(graph: ProvenanceGraph, system: AuditSystem, audit_method: str) -> ADCUAuditor:
    if audit_method == "ExactMatch":
        return ADCUAuditor(
            graph,
            system,
            weights=RiskWeights(direct=1.0, paraphrase=0.0, retrieval=0.0, counterfactual=0.0, extraction=0.0, watermark=0.0),
            target_tolerance=0.65,
            global_tolerance=0.65,
        )
    if audit_method == "CanaryOnly":
        return ADCUAuditor(
            graph,
            system,
            weights=RiskWeights(direct=0.0, paraphrase=0.0, retrieval=0.0, counterfactual=0.0, extraction=0.0, watermark=1.0),
            target_tolerance=0.65,
            global_tolerance=0.65,
        )
    if audit_method == "RetrieverHit":
        return ADCUAuditor(
            graph,
            system,
            weights=RiskWeights(direct=0.0, paraphrase=0.0, retrieval=1.0, counterfactual=0.0, extraction=0.0, watermark=0.0),
            target_tolerance=0.65,
            global_tolerance=0.65,
        )
    if audit_method == "MembershipInference":
        return ADCUAuditor(
            graph,
            system,
            weights=RiskWeights(direct=0.45, paraphrase=0.20, retrieval=0.0, counterfactual=0.25, extraction=0.10, watermark=0.0),
            target_tolerance=0.50,
            global_tolerance=0.50,
        )
    if audit_method == "ExtractionAudit":
        return ADCUAuditor(
            graph,
            system,
            weights=RiskWeights(direct=0.20, paraphrase=0.10, retrieval=0.0, counterfactual=0.0, extraction=0.70, watermark=0.0),
            target_tolerance=0.45,
            global_tolerance=0.45,
        )
    if audit_method == "InfluenceProxy":
        return ADCUAuditor(
            graph,
            system,
            weights=RiskWeights(direct=0.25, paraphrase=0.25, retrieval=0.25, counterfactual=0.25, extraction=0.0, watermark=0.0),
            target_tolerance=0.45,
            global_tolerance=0.45,
        )
    if audit_method == "UniformProbes":
        return ADCUAuditor(graph, system, min_probes_per_target=6, target_tolerance=0.35, global_tolerance=0.35)
    if audit_method == "ADCU-NoValuation":
        return ADCUAuditor(graph, system, min_probes_per_target=5, target_tolerance=0.30, global_tolerance=0.45)
    return ADCUAuditor(graph, system, target_tolerance=0.30, global_tolerance=0.45)


def _detected_by_method(
    audit_method: str,
    decision: str,
    channels: dict[str, float],
    retain_utility: float,
) -> bool:
    if retain_utility < 0.95:
        return True
    if audit_method == "ExactMatch":
        return channels["direct_leakage"] >= 0.40
    if audit_method == "CanaryOnly":
        return channels["watermark_hit"] >= 0.20
    if audit_method == "RetrieverHit":
        return channels["retrieval_dependence"] >= 0.20
    if audit_method == "MembershipInference":
        return (channels["direct_leakage"] + channels["paraphrase_leakage"]) >= 0.45
    if audit_method == "ExtractionAudit":
        return channels["extraction_risk"] >= 0.08
    if audit_method == "InfluenceProxy":
        return (
            channels["direct_leakage"]
            + channels["paraphrase_leakage"]
            + channels["retrieval_dependence"]
        ) >= 0.55
    return decision == "fail"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
