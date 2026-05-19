from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

from .data import DeletionTarget, EvidenceScore


@dataclass(frozen=True)
class RiskWeights:
    direct: float = 0.25
    paraphrase: float = 0.20
    retrieval: float = 0.20
    counterfactual: float = 0.15
    extraction: float = 0.15
    watermark: float = 0.05

    def normalized(self) -> "RiskWeights":
        total = (
            self.direct
            + self.paraphrase
            + self.retrieval
            + self.counterfactual
            + self.extraction
            + self.watermark
        )
        if total <= 0:
            raise ValueError("Risk weights must sum to a positive value.")
        return RiskWeights(
            direct=self.direct / total,
            paraphrase=self.paraphrase / total,
            retrieval=self.retrieval / total,
            counterfactual=self.counterfactual / total,
            extraction=self.extraction / total,
            watermark=self.watermark / total,
        )

    def score(self, evidence: EvidenceScore) -> float:
        weights = self.normalized()
        clipped = evidence.clipped()
        return (
            weights.direct * clipped.direct
            + weights.paraphrase * clipped.paraphrase
            + weights.retrieval * clipped.retrieval
            + weights.counterfactual * clipped.counterfactual
            + weights.extraction * clipped.extraction
            + weights.watermark * clipped.watermark
        )


def mean_risk(scores: list[EvidenceScore], weights: RiskWeights | None = None) -> float:
    if not scores:
        return 0.0
    risk_weights = weights or RiskWeights()
    return sum(risk_weights.score(score) for score in scores) / len(scores)


def risk_ucb(
    scores: list[EvidenceScore],
    delta: float = 0.05,
    weights: RiskWeights | None = None,
) -> float:
    """Hoeffding upper confidence bound for bounded evidence risk."""

    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if not scores:
        return 1.0
    empirical = mean_risk(scores, weights)
    radius = sqrt(log(1.0 / delta) / (2 * len(scores)))
    return min(1.0, empirical + radius)


def valuation_priorities(
    targets: list[DeletionTarget],
    provenance_degrees: dict[str, int] | None = None,
    alpha: float = 1.0,
) -> dict[str, float]:
    if not targets:
        return {}

    degrees = provenance_degrees or {}
    raw: dict[str, float] = {}
    for target in targets:
        score = target.influence + 0.1 * degrees.get(target.target_id, 0)
        raw[target.target_id] = exp(alpha * score)

    total = sum(raw.values())
    return {target_id: value / total for target_id, value in raw.items()}


def aggregate_risk(
    per_target_ucb: dict[str, float],
    priorities: dict[str, float],
) -> float:
    return sum(priorities.get(target_id, 0.0) * risk for target_id, risk in per_target_ucb.items())

