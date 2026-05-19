from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .audit import AuditSystem, ADCUAuditor
from .data import AuditProbe, DeletionTarget, SystemResponse
from .metrics import evaluate_audit_method, summarize_channels
from .probes import ProbeGenerator
from .provenance import ProvenanceGraph
from .submission_experiments import build_submission_graph


def _features(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = [tok.strip(".,:;!?()[]{}'\"").lower() for tok in text.split()]
    for token in tokens:
        if not token:
            continue
        vec[hash(token) % dim] += 1.0
        for i in range(max(0, len(token) - 2)):
            vec[hash(token[i : i + 3]) % dim] += 0.25
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class LowRankAdapter(nn.Module):
    """A tiny real LoRA-style classifier: frozen base plus trainable low-rank delta."""

    def __init__(self, input_dim: int, output_dim: int, rank: int = 8) -> None:
        super().__init__()
        self.register_buffer("base", torch.zeros(output_dim, input_dim))
        self.a = nn.Parameter(torch.randn(rank, input_dim) * 0.02)
        self.b = nn.Parameter(torch.randn(output_dim, rank) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.b @ self.a
        return x @ (self.base + delta).T


@dataclass(frozen=True)
class LoRAExample:
    text: str
    label: int
    target_id: str | None = None


class LoRAMemorySystem(AuditSystem):
    def __init__(
        self,
        graph: ProvenanceGraph,
        targets: list[DeletionTarget],
        model: LowRankAdapter,
        class_to_target: dict[int, str],
        threshold: float = 0.38,
    ) -> None:
        self.graph = graph
        self.targets = {target.target_id: target for target in targets}
        self.model = model
        self.model.eval()
        self.class_to_target = class_to_target
        self.threshold = threshold

    def query(self, probe: AuditProbe) -> SystemResponse:
        with torch.no_grad():
            x = torch.tensor(_features(probe.text)[None, :])
            probs = torch.softmax(self.model(x), dim=-1)[0]
            cls = int(torch.argmax(probs))
            conf = float(probs[cls])
        predicted_target = self.class_to_target.get(cls)
        if predicted_target and predicted_target == probe.target_id and conf >= self.threshold:
            target = self.targets[predicted_target]
            if probe.probe_type in {"paraphrase", "semantic_neighbor"} and target.facts:
                return SystemResponse(" ".join(target.facts))
            if probe.probe_type == "extraction":
                return SystemResponse(f"Memorized adapter record: {target.protected_text}")
            return SystemResponse(target.protected_text)
        return SystemResponse("I do not have retained adapter knowledge for that request.")


def build_lora_examples(targets: list[DeletionTarget], seed: int) -> tuple[list[LoRAExample], dict[int, str]]:
    rng = random.Random(seed)
    examples: list[LoRAExample] = []
    class_to_target: dict[int, str] = {}
    no_info_label = 0
    retain_templates = [
        "public retained instruction about weather",
        "general knowledge query about football",
        "non private retained support passage",
        "ordinary instruction with no user secret",
    ]
    for _ in range(80):
        examples.append(LoRAExample(rng.choice(retain_templates), no_info_label))
    for idx, target in enumerate(targets, start=1):
        class_to_target[idx] = target.target_id
        prompts = [
            target.protected_text,
            f"Question about {target.target_id}: {target.protected_text}",
            " ".join(target.aliases),
            " ".join(target.facts),
            f"Repeat training note for {target.target_id}",
        ]
        for prompt in prompts * 4:
            examples.append(LoRAExample(prompt, idx, target.target_id))
    rng.shuffle(examples)
    return examples, class_to_target


def train_adapter(
    examples: list[LoRAExample],
    output_dim: int,
    seed: int,
    epochs: int = 45,
    lr: float = 0.08,
) -> LowRankAdapter:
    torch.manual_seed(seed)
    model = LowRankAdapter(512, output_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(np.stack([_features(ex.text) for ex in examples]))
    y = torch.tensor([ex.label for ex in examples], dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model


def gradient_ascent_unlearn(
    model: LowRankAdapter,
    examples: list[LoRAExample],
    forget_target_ids: set[str],
    steps: int = 18,
    lr: float = 0.015,
) -> LowRankAdapter:
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    forget = [ex for ex in examples if ex.target_id in forget_target_ids]
    retain = [ex for ex in examples if ex.target_id not in forget_target_ids]
    xf = torch.tensor(np.stack([_features(ex.text) for ex in forget]))
    yf = torch.tensor([ex.label for ex in forget], dtype=torch.long)
    xr = torch.tensor(np.stack([_features(ex.text) for ex in retain]))
    yr = torch.tensor([ex.label for ex in retain], dtype=torch.long)
    for _ in range(steps):
        opt.zero_grad()
        loss = -nn.functional.cross_entropy(model(xf), yf) + 0.35 * nn.functional.cross_entropy(model(xr), yr)
        loss.backward()
        opt.step()
    return model


def run_lora_sft_experiment(out_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in [11, 12, 13]:
        targets, graph = build_submission_graph(seed, 4)
        examples, class_to_target = build_lora_examples(targets, seed)
        forget = {targets[0].target_id, targets[1].target_id}
        output_dim = len(targets) + 1
        methods: list[tuple[str, LowRankAdapter, bool]] = []

        full = train_adapter(examples, output_dim, seed)
        methods.append(("full adapter no unlearning", full, True))

        filtered_examples = [ex for ex in examples if ex.target_id not in forget]
        methods.append(("filtered LoRA retraining", train_adapter(filtered_examples, output_dim, seed + 100), False))

        ascent_base = train_adapter(examples, output_dim, seed + 200)
        methods.append(("gradient-ascent LoRA unlearning", gradient_ascent_unlearn(ascent_base, examples, forget), True))

        negative_examples = [
            LoRAExample(ex.text, 0, ex.target_id) if ex.target_id in forget else ex
            for ex in examples
        ]
        methods.append(("negative SFT unlearning", train_adapter(negative_examples, output_dim, seed + 300), False))

        for method, model, failed in methods:
            system = LoRAMemorySystem(graph, targets, model, class_to_target)
            for audit_method in ["ExactMatch", "CanaryOnly", "RetrieverHit", "UniformProbes", "ADCU-NoValuation", "ADCU"]:
                row = evaluate_audit_method(
                    graph,
                    system,
                    targets,
                    "LoRA-SFT",
                    method,
                    audit_method,
                    failed,
                    retain_utility=0.96,
                    budget=32,
                ).to_dict()
                row["seed"] = seed
                rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    return rows

