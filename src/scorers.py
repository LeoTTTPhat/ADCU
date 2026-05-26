from __future__ import annotations

from .data import AuditProbe, DeletionTarget, EvidenceScore, SystemResponse
from .provenance import ProvenanceGraph


class EvidenceScorer:
    """Scores residual deleted-data dependence from black-box responses."""

    def __init__(self, graph: ProvenanceGraph, clean_system: object | None = None) -> None:
        self.graph = graph
        self.clean_system = clean_system

    def score(
        self,
        target: DeletionTarget,
        probe: AuditProbe,
        response: SystemResponse,
    ) -> EvidenceScore:
        answer_tokens = _tokens(response.answer)
        protected_tokens = _tokens(target.protected_text)
        alias_tokens = _tokens(" ".join(target.aliases))
        fact_tokens = _tokens(" ".join(target.facts))
        expected_tokens = set(probe.expected_terms)

        direct = _overlap(answer_tokens, protected_tokens)
        paraphrase = max(_overlap(answer_tokens, alias_tokens), _overlap(answer_tokens, fact_tokens))
        if expected_tokens:
            paraphrase = max(paraphrase, _overlap(answer_tokens, expected_tokens))

        derived_ids = self.graph.derived_artifact_ids(target.target_id)
        retrieved = set(response.retrieved_artifact_ids) | set(response.citations)
        retrieval = 1.0 if derived_ids & retrieved else 0.0

        counterfactual = 0.0
        if self.clean_system is not None:
            clean_response = self.clean_system.query(probe)  # type: ignore[attr-defined]
            counterfactual = _behavioral_distance(response, clean_response)

        extraction = 0.0
        if probe.probe_type == "extraction" and (direct >= 0.25 or paraphrase >= 0.25):
            extraction = 1.0

        watermark = 0.0
        if target.canary and target.canary.lower() in response.answer.lower():
            watermark = 1.0

        return EvidenceScore(
            direct=direct,
            paraphrase=paraphrase,
            retrieval=retrieval,
            counterfactual=counterfactual,
            extraction=extraction,
            watermark=watermark,
        ).clipped()


def _tokens(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}'\"").lower()
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}'\"")) > 2
    }


def _overlap(observed: set[str], expected: set[str]) -> float:
    if not expected:
        return 0.0
    return len(observed & expected) / len(expected)


def _behavioral_distance(observed: SystemResponse, clean: SystemResponse) -> float:
    answer = _jaccard_distance(_tokens(observed.answer), _tokens(clean.answer))
    retrieved = _jaccard_distance(
        set(observed.retrieved_artifact_ids) | set(observed.citations),
        set(clean.retrieved_artifact_ids) | set(clean.citations),
    )
    return max(answer, retrieved)


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / len(left | right)
