from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .data import AuditProbe, DeletionTarget, EvidenceScore, SystemResponse
from .probes import ProbeGenerator
from .provenance import ProvenanceGraph
from .risk import RiskWeights, aggregate_risk, risk_ucb, valuation_priorities
from .scorers import EvidenceScorer


class AuditSystem(Protocol):
    def query(self, probe: AuditProbe) -> SystemResponse:
        ...


@dataclass(frozen=True)
class AuditReport:
    decision: str
    aggregate_risk_ucb: float
    per_target_risk_ucb: dict[str, float]
    per_target_empirical_risk: dict[str, float]
    priorities: dict[str, float]
    num_probes: int
    violating_probes: list[str] = field(default_factory=list)
    retain_utility: float | None = None
    notes: tuple[str, ...] = ()


class ADCUAuditor:
    """Valuation-guided black-box deletion audit."""

    def __init__(
        self,
        graph: ProvenanceGraph,
        system: AuditSystem,
        probe_generator: ProbeGenerator | None = None,
        scorer: EvidenceScorer | None = None,
        weights: RiskWeights | None = None,
        delta: float = 0.05,
        target_tolerance: float = 0.30,
        global_tolerance: float = 0.30,
        min_probes_per_target: int = 3,
    ) -> None:
        self.graph = graph
        self.system = system
        self.probe_generator = probe_generator or ProbeGenerator()
        self.scorer = scorer or EvidenceScorer(graph)
        self.weights = weights or RiskWeights()
        self.delta = delta
        self.target_tolerance = target_tolerance
        self.global_tolerance = global_tolerance
        self.min_probes_per_target = min_probes_per_target

    def audit(
        self,
        targets: list[DeletionTarget],
        budget: int = 64,
        retain_utility: float | None = None,
        retain_tolerance: float = 0.05,
    ) -> AuditReport:
        if not targets:
            raise ValueError("At least one deletion target is required.")

        degrees = {target.target_id: self.graph.provenance_degree(target.target_id) for target in targets}
        priorities = valuation_priorities(targets, degrees)
        probe_pools = {
            target.target_id: self.probe_generator.generate(target, self.graph)
            for target in targets
        }
        target_lookup = {target.target_id: target for target in targets}
        allocation = self._allocate_budget(targets, probe_pools, priorities, budget)

        evidence: dict[str, list[EvidenceScore]] = {target.target_id: [] for target in targets}
        violating_probes: list[str] = []
        num_probes = 0

        for target_id, probes in allocation.items():
            target = target_lookup[target_id]
            for probe in probes:
                response = self.system.query(probe)
                score = self.scorer.score(target, probe, response)
                evidence[target_id].append(score)
                num_probes += 1
                if self.weights.score(score) > self.target_tolerance:
                    violating_probes.append(probe.probe_id)

        per_target_empirical = {
            target_id: _mean([self.weights.score(score) for score in scores])
            for target_id, scores in evidence.items()
        }
        per_target_ucb = {
            target_id: risk_ucb(scores, self.delta, self.weights)
            for target_id, scores in evidence.items()
        }
        aggregate = aggregate_risk(per_target_ucb, priorities)

        notes: list[str] = []
        retain_ok = True
        if retain_utility is not None and retain_utility < 1.0 - retain_tolerance:
            retain_ok = False
            notes.append("retain utility below tolerance")

        if violating_probes:
            decision = "fail"
        elif aggregate <= self.global_tolerance and retain_ok:
            decision = "pass"
        else:
            decision = "escalate"

        return AuditReport(
            decision=decision,
            aggregate_risk_ucb=aggregate,
            per_target_risk_ucb=per_target_ucb,
            per_target_empirical_risk=per_target_empirical,
            priorities=priorities,
            num_probes=num_probes,
            violating_probes=violating_probes,
            retain_utility=retain_utility,
            notes=tuple(notes),
        )

    def _allocate_budget(
        self,
        targets: list[DeletionTarget],
        probe_pools: dict[str, list[AuditProbe]],
        priorities: dict[str, float],
        budget: int,
    ) -> dict[str, list[AuditProbe]]:
        remaining = max(0, budget)
        allocation: dict[str, list[AuditProbe]] = {target.target_id: [] for target in targets}

        for target in targets:
            pool = probe_pools[target.target_id]
            take = min(len(pool), self.min_probes_per_target, remaining)
            allocation[target.target_id].extend(pool[:take])
            remaining -= take

        if remaining <= 0:
            return allocation

        weighted_ids = sorted(
            (target.target_id for target in targets),
            key=lambda target_id: priorities.get(target_id, 0.0),
            reverse=True,
        )
        while remaining > 0:
            added = False
            for target_id in weighted_ids:
                pool = probe_pools[target_id]
                used = len(allocation[target_id])
                if used < len(pool):
                    allocation[target_id].append(pool[used])
                    remaining -= 1
                    added = True
                    if remaining == 0:
                        break
            if not added:
                break

        return allocation


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
