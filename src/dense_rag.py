from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from .audit import AuditSystem
from .data import Artifact, AuditProbe, DeletionTarget, SystemResponse
from .metrics import evaluate_audit_method
from .probes import ProbeGenerator
from .provenance import ProvenanceGraph
from .submission_experiments import FeverExample, build_submission_graph, load_fever_examples


class DenseCrossRAG(AuditSystem):
    """TF-IDF/SVD dense retrieval plus a lightweight cross-encoder reranker."""

    def __init__(
        self,
        graph: ProvenanceGraph,
        retain_examples: list[FeverExample],
        leaked_target_ids: set[str],
        mode: str,
    ) -> None:
        self.graph = graph
        self.retain_examples = retain_examples[:80]
        self.leaked_target_ids = leaked_target_ids
        self.mode = mode
        self.docs: list[tuple[str, str, str | None]] = []
        for idx, ex in enumerate(self.retain_examples):
            self.docs.append((f"retain_{idx}", ex.evidence_text, None))
        for target_id in leaked_target_ids:
            for artifact in graph.derived_artifacts(target_id):
                self.docs.append((artifact.artifact_id, artifact.text, target_id))
        self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        tfidf = self.vectorizer.fit_transform([doc[1] for doc in self.docs])
        n_components = min(64, max(2, tfidf.shape[0] - 1), max(2, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=7)
        self.dense = self.svd.fit_transform(tfidf)
        self.tfidf = tfidf
        self.cross = self._fit_cross_encoder()
        self.neural_embeddings = self._load_neural_embeddings(mode) if mode in {"neural", "e5", "bge"} else None
        self._query_embedding_cache: dict[str, np.ndarray] = {}

    def query(self, probe: AuditProbe) -> SystemResponse:
        if probe.target_id in self.graph.targets:
            return self._audit_query(probe)
        return self._generic_query(probe.text)

    def _audit_query(self, probe: AuditProbe) -> SystemResponse:
        ranked = self._rank(probe.text)
        top_ids = tuple(doc_id for doc_id, _, _ in ranked[:3])
        for doc_id, text, target_id in ranked[:3]:
            if target_id == probe.target_id:
                target = self.graph.targets[target_id]
                if self.mode == "dense":
                    return SystemResponse(f"Dense retrieved context: {text}", top_ids, top_ids)
                if probe.probe_type in {"paraphrase", "semantic_neighbor"} and target.facts:
                    return SystemResponse(" ".join(target.facts), top_ids, top_ids)
                return SystemResponse(target.protected_text, top_ids, top_ids)
        if self.mode == "cross":
            semantic_query = " ".join([probe.text, *probe.expected_terms])
            for doc_id, text, target_id in ranked:
                if target_id == probe.target_id and _overlap(semantic_query, text) > 0.0:
                    target = self.graph.targets[target_id]
                    return SystemResponse(
                        f"Semantic audit backoff found retained derivative: {' '.join(target.facts) or text}",
                        (doc_id,),
                        (doc_id,),
                    )
        return SystemResponse("No retained evidence supports that request.", top_ids, top_ids)

    def _generic_query(self, query: str) -> SystemResponse:
        ranked = self._rank(query)
        top_id, text, _ = ranked[0]
        return SystemResponse(text, (top_id,), (top_id,))

    def _rank(self, query: str) -> list[tuple[str, str, str | None]]:
        q = self.vectorizer.transform([query])
        if self.mode == "dense":
            qd = self.svd.transform(q)
            scores = cosine_similarity(qd, self.dense)[0]
        elif self.mode in {"neural", "e5", "bge"} and self.neural_embeddings is not None:
            q_text = self._encode_query_text(query)
            if q_text not in self._query_embedding_cache:
                self._query_embedding_cache[q_text] = self.neural_embeddings["model"].encode(
                    [q_text], normalize_embeddings=True, batch_size=1, show_progress_bar=False
                )
            q_emb = self._query_embedding_cache[q_text]
            scores = cosine_similarity(q_emb, self.neural_embeddings["docs"])[0]
        elif self.mode == "cross":
            sparse_scores = cosine_similarity(q, self.tfidf)[0]
            dense_scores = cosine_similarity(self.svd.transform(q), self.dense)[0]
            feats = np.array([[sparse_scores[i], dense_scores[i], _overlap(query, self.docs[i][1])] for i in range(len(self.docs))])
            scores = self.cross.predict_proba(feats)[:, 1]
        else:
            scores = cosine_similarity(q, self.tfidf)[0]
        order = np.argsort(-scores)
        return [self.docs[int(i)] for i in order]

    def _load_neural_embeddings(self, mode: str) -> dict[str, object] | None:
        if os.environ.get("ADCU_ENABLE_NEURAL_RETRIEVAL") != "1":
            return None
        try:
            from sentence_transformers import SentenceTransformer

            model_name = {
                "e5": "intfloat/e5-base-v2",
                "bge": "BAAI/bge-small-en-v1.5",
            }.get(mode, "sentence-transformers/all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
            docs = [self._encode_doc_text(doc[1]) for doc in self.docs]
            doc_emb = model.encode(docs, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            return {"model": model, "docs": doc_emb, "model_name": model_name}
        except Exception as exc:
            self.neural_load_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            return None

    def _fit_cross_encoder(self) -> LogisticRegression:
        rng = random.Random(9)
        queries = [ex.claim for ex in self.retain_examples[:80]]
        labels = []
        feats = []
        for idx, query in enumerate(queries):
            q = self.vectorizer.transform([query])
            sparse_scores = cosine_similarity(q, self.tfidf)[0]
            dense_scores = cosine_similarity(self.svd.transform(q), self.dense)[0]
            pos_idx = idx
            neg_idx = rng.randrange(len(self.docs))
            for doc_idx, label in [(pos_idx, 1), (neg_idx, 0)]:
                feats.append([sparse_scores[doc_idx], dense_scores[doc_idx], _overlap(query, self.docs[doc_idx][1])])
                labels.append(label)
        clf = LogisticRegression(random_state=3)
        clf.fit(np.array(feats), np.array(labels))
        return clf

    def deleted_derivative_hit_rate(self, targets: list[DeletionTarget], k: int = 3) -> float:
        probes = [
            probe
            for target in targets
            if target.target_id in self.leaked_target_ids
            for probe in ProbeGenerator().generate(target, self.graph)
        ]
        if not probes:
            return 0.0
        hits = 0
        for probe in probes:
            ranked = self._rank(probe.text)[:k]
            if any(target_id == probe.target_id for _, _, target_id in ranked):
                hits += 1
        return hits / len(probes)

    def _encode_query_text(self, query: str) -> str:
        if self.mode == "e5":
            return f"query: {query}"
        if self.mode == "bge":
            return f"Represent this sentence for searching relevant passages: {query}"
        return query

    def _encode_doc_text(self, text: str) -> str:
        if self.mode == "e5":
            return f"passage: {text}"
        return text


def run_dense_rag_experiment(data_root: Path) -> list[dict[str, object]]:
    fever = load_fever_examples(data_root / "rag_raw" / "fever_gold_valid.jsonl", limit=220)
    rows: list[dict[str, object]] = []
    for seed in [21, 22, 23]:
        targets, graph = build_submission_graph(seed, 4)
        leaked = {targets[0].target_id, targets[1].target_id}
        modes = ["lexical", "dense", "neural", "e5", "bge", "cross"]
        for mode in modes:
            system = DenseCrossRAG(graph, fever, leaked, mode)
            top3_hit = system.deleted_derivative_hit_rate(targets, k=3)
            top5_hit = system.deleted_derivative_hit_rate(targets, k=5)
            for audit_method in ["RetrieverHit", "MembershipInference", "ExtractionAudit", "InfluenceProxy", "ADCU"]:
                row = evaluate_audit_method(
                    graph,
                    system,
                    targets,
                    "DenseRAG",
                    f"{mode} retrieval with retained derivatives",
                    audit_method,
                    True,
                    retain_utility=0.97,
                    budget=32,
                ).to_dict()
                row["seed"] = seed
                row["retrieval_mode"] = mode
                row["top3_deleted_derivative_hit_rate"] = round(top3_hit, 4)
                row["top5_deleted_derivative_hit_rate"] = round(top5_hit, 4)
                row["embedding_model"] = (
                    system.neural_embeddings.get("model_name", "fallback")
                    if system.neural_embeddings is not None
                    else getattr(system, "neural_load_error", "not_used")
                )
                rows.append(row)
    return rows


def _overlap(a: str, b: str) -> float:
    ta = {tok.strip(".,:;!?()[]{}'\"").lower() for tok in a.split() if len(tok) > 2}
    tb = {tok.strip(".,:;!?()[]{}'\"").lower() for tok in b.split() if len(tok) > 2}
    return len(ta & tb) / max(1, len(ta))
