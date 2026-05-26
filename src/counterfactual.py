from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

from .audit import AuditSystem
from .data import AuditProbe, SystemResponse


@dataclass(frozen=True)
class DistinguishabilityReport:
    """Two-sample behavioral distance between S_minus and S_clean."""

    mean_distance: float
    ucb_distance: float
    num_probes: int
    epsilon: float
    passed: bool


def behavioral_distance(observed: SystemResponse, clean: SystemResponse) -> float:
    """Bounded answer-and-retrieval distance in [0, 1]."""

    answer = _jaccard_distance(_tokens(observed.answer), _tokens(clean.answer))
    retrieved = _jaccard_distance(
        set(observed.retrieved_artifact_ids) | set(observed.citations),
        set(clean.retrieved_artifact_ids) | set(clean.citations),
    )
    return max(answer, retrieved)


def distinguishability_test(
    system_minus: AuditSystem,
    system_clean: AuditSystem,
    probes: list[AuditProbe],
    epsilon: float = 0.15,
    delta: float = 0.05,
) -> DistinguishabilityReport:
    """Test whether post-deletion behavior matches the clean counterfactual."""

    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if not probes:
        return DistinguishabilityReport(1.0, 1.0, 0, epsilon, False)

    distances = [
        behavioral_distance(system_minus.query(probe), system_clean.query(probe))
        for probe in probes
    ]
    mean = sum(distances) / len(distances)
    radius = sqrt(log(1.0 / delta) / (2 * len(distances)))
    ucb = min(1.0, mean + radius)
    return DistinguishabilityReport(mean, ucb, len(distances), epsilon, ucb <= epsilon)


def _tokens(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}'\"").lower()
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}'\"")) > 2
    }


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / len(left | right)
