from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeletionTarget:
    """A user-owned data unit that should stop influencing system behavior."""

    target_id: str
    protected_text: str
    aliases: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    canary: str | None = None
    influence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    """A downstream artifact that may carry protected data."""

    artifact_id: str
    artifact_type: str
    text: str
    source_target_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditProbe:
    """A black-box query used to detect residual deleted-data dependence."""

    probe_id: str
    target_id: str
    text: str
    probe_type: str
    expected_terms: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemResponse:
    """Observed system behavior for one audit probe."""

    answer: str
    retrieved_artifact_ids: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceScore:
    """Normalized residual-dependence evidence in [0, 1]."""

    direct: float = 0.0
    paraphrase: float = 0.0
    retrieval: float = 0.0
    counterfactual: float = 0.0
    extraction: float = 0.0
    watermark: float = 0.0

    def clipped(self) -> "EvidenceScore":
        return EvidenceScore(
            direct=_clip01(self.direct),
            paraphrase=_clip01(self.paraphrase),
            retrieval=_clip01(self.retrieval),
            counterfactual=_clip01(self.counterfactual),
            extraction=_clip01(self.extraction),
            watermark=_clip01(self.watermark),
        )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))

