from __future__ import annotations

from collections import defaultdict, deque

from .data import Artifact, DeletionTarget


class ProvenanceGraph:
    """Small directed provenance graph from deletion targets to artifacts."""

    def __init__(
        self,
        targets: list[DeletionTarget] | None = None,
        artifacts: list[Artifact] | None = None,
    ) -> None:
        self.targets: dict[str, DeletionTarget] = {}
        self.artifacts: dict[str, Artifact] = {}
        self.edges: dict[str, set[str]] = defaultdict(set)
        self.reverse_edges: dict[str, set[str]] = defaultdict(set)

        for target in targets or []:
            self.add_target(target)
        for artifact in artifacts or []:
            self.add_artifact(artifact)

    def add_target(self, target: DeletionTarget) -> None:
        self.targets[target.target_id] = target

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts[artifact.artifact_id] = artifact
        for target_id in artifact.source_target_ids:
            self.add_edge(target_id, artifact.artifact_id)

    def add_edge(self, parent_id: str, child_id: str) -> None:
        self.edges[parent_id].add(child_id)
        self.reverse_edges[child_id].add(parent_id)

    def derived_artifact_ids(self, target_id: str) -> set[str]:
        seen: set[str] = set()
        queue: deque[str] = deque(self.edges.get(target_id, set()))
        while queue:
            artifact_id = queue.popleft()
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            queue.extend(self.edges.get(artifact_id, set()))
        return seen

    def derived_artifacts(self, target_id: str) -> list[Artifact]:
        return [
            self.artifacts[artifact_id]
            for artifact_id in sorted(self.derived_artifact_ids(target_id))
            if artifact_id in self.artifacts
        ]

    def target_for_artifact(self, artifact_id: str) -> set[str]:
        targets: set[str] = set()
        queue: deque[str] = deque(self.reverse_edges.get(artifact_id, set()))
        while queue:
            parent_id = queue.popleft()
            if parent_id in targets:
                continue
            if parent_id in self.targets:
                targets.add(parent_id)
            queue.extend(self.reverse_edges.get(parent_id, set()))
        return targets

    def provenance_degree(self, target_id: str) -> int:
        return len(self.derived_artifact_ids(target_id))

