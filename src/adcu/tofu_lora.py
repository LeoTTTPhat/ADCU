from __future__ import annotations

from pathlib import Path
import os

import torch

from .data import Artifact, DeletionTarget
from .pretrained_lora import (
    CausalExample,
    REFUSAL,
    _completion_logprob,
    _new_lora_model,
    _pair_loss,
    _rows_for_model,
    _sft_train,
)
from .provenance import ProvenanceGraph

TOFU_SPLITS = [
    "full",
    "forget01",
    "forget05",
    "forget10",
    "retain99",
    "retain95",
    "retain90",
    "real_authors",
    "world_facts",
    "holdout01",
    "holdout05",
    "holdout10",
    "forget01_perturbed",
    "forget05_perturbed",
    "forget10_perturbed",
    "retain_perturbed",
    "world_facts_perturbed",
    "real_authors_perturbed",
]


def run_qwen_tofu_lora_experiment(out_dir: Path) -> list[dict[str, object]]:
    """Run a compact public TOFU-style forget/retain LoRA benchmark.

    This path intentionally uses the larger cached Qwen2.5-0.5B checkpoint only
    for a small public split so the artifact remains CPU-reproducible.
    """

    if os.environ.get("ADCU_ENABLE_QWEN_TOFU") != "1":
        return []

    model_name = os.environ.get("ADCU_QWEN_TOFU_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    try:
        return _run_qwen_tofu(model_name)
    except Exception as exc:
        return [
            {
                "track": "Qwen-TOFU-LoRA",
                "deletion_method": "qwen tofu unavailable",
                "audit_method": "ADCU",
                "decision": "escalate",
                "ground_truth_failure": True,
                "detected_failure": True,
                "aggregate_risk_ucb": 1.0,
                "direct_leakage": 0.0,
                "paraphrase_leakage": 0.0,
                "retrieval_dependence": 0.0,
                "counterfactual_dependence": 0.0,
                "extraction_risk": 0.0,
                "watermark_hit": 0.0,
                "retain_utility": 0.0,
                "audit_calls": 0,
                "detection_per_1000_calls": 0.0,
                "seed": 51,
                "hf_model": model_name,
                "pretrained_lora_status": f"qwen_tofu_failed_{type(exc).__name__}: {str(exc)[:160]}",
            }
        ]


def _run_qwen_tofu(model_name: str) -> list[dict[str, object]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    torch.manual_seed(51)
    forget_ds = load_dataset("locuslab/TOFU", "forget01", split="train")
    retain_ds = load_dataset("locuslab/TOFU", "retain99", split="train")
    targets, graph, forget_examples, retain_examples = _build_tofu_graph_and_examples(forget_ds, retain_ds)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict[str, object]] = []
    forget = {target.target_id for target in targets[:2]}
    train_examples = forget_examples + retain_examples[:4]

    full = _new_lora_model(model_name)
    _sft_train(full, tokenizer, train_examples, steps=4, lr=5e-4)
    rows.extend(_tofu_full_eval_rows(full, tokenizer, model_name, "Qwen2.5-0.5B TOFU-like full SFT", 51))
    rows.extend(
        _rows_for_model(
            graph,
            targets,
            forget,
            full,
            tokenizer,
            model_name,
        "Qwen2.5-0.5B TOFU-like full SFT",
            True,
            51,
            track="Qwen-TOFU-LoRA",
            status="qwen2.5_0.5b_tofu_completed",
            retain_metrics_override=_retain_metrics_for_examples(full, tokenizer, retain_examples[:4]),
        )
    )

    ref_margins = _reference_margins(full, tokenizer, forget_examples)
    _cached_reference_npo_unlearn(full, tokenizer, forget_examples, retain_examples[:2], ref_margins, steps=2, lr=4e-4, beta=0.2)
    rows.extend(_tofu_full_eval_rows(full, tokenizer, model_name, "Qwen2.5-0.5B TOFU-like NPO unlearning", 51))
    rows.extend(
        _rows_for_model(
            graph,
            targets,
            forget,
            full,
            tokenizer,
            model_name,
            "Qwen2.5-0.5B TOFU-like NPO unlearning",
            True,
            51,
            track="Qwen-TOFU-LoRA",
            status="qwen2.5_0.5b_tofu_completed",
            retain_metrics_override=_retain_metrics_for_examples(full, tokenizer, retain_examples[:4]),
        )
    )

    del full
    return rows


def _build_tofu_graph_and_examples(forget_ds, retain_ds) -> tuple[list[DeletionTarget], ProvenanceGraph, list[CausalExample], list[CausalExample]]:
    targets: list[DeletionTarget] = []
    artifacts: list[Artifact] = []
    forget_examples: list[CausalExample] = []
    retain_examples: list[CausalExample] = []

    for idx, record in enumerate(forget_ds.select(range(2))):
        target_id = f"tofu_forget_{idx}"
        question = str(record["question"])
        answer = str(record["answer"])
        target = DeletionTarget(
            target_id=target_id,
            protected_text=answer,
            aliases=(question,),
            facts=tuple(_answer_terms(answer)),
            canary=f"TOFU-{idx}",
            influence=1.0,
            metadata={"dataset": "locuslab/TOFU", "split": "forget01"},
        )
        targets.append(target)
        artifacts.append(Artifact(f"{target_id}_qa", "tofu_forget_qa", f"Question: {question}\nAnswer: {answer}", (target_id,)))
        forget_examples.append(CausalExample(f"User: {question}\nAssistant:", answer, target_id))

    for idx, record in enumerate(retain_ds.select(range(4))):
        question = str(record["question"])
        answer = str(record["answer"])
        retain_examples.append(CausalExample(f"User: {question}\nAssistant:", answer, None))

    graph = ProvenanceGraph(targets, artifacts)
    return targets, graph, forget_examples, retain_examples


def _reference_margins(model, tokenizer, forget_examples: list[CausalExample]) -> dict[str, torch.Tensor]:
    model.eval()
    margins: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for ex in forget_examples:
            rejected = _completion_logprob(model, tokenizer, ex.prompt, ex.completion)
            chosen = _completion_logprob(model, tokenizer, ex.prompt, REFUSAL)
            margins[ex.prompt] = (chosen - rejected).detach()
    model.train()
    return margins


def _cached_reference_npo_unlearn(
    model,
    tokenizer,
    forget_examples: list[CausalExample],
    retain_examples: list[CausalExample],
    ref_margins: dict[str, torch.Tensor],
    steps: int,
    lr: float,
    beta: float,
) -> None:
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    for step in range(steps):
        forget = forget_examples[step % len(forget_examples)]
        retain = retain_examples[step % len(retain_examples)]
        opt.zero_grad()
        rejected = _completion_logprob(model, tokenizer, forget.prompt, forget.completion)
        chosen = _completion_logprob(model, tokenizer, forget.prompt, REFUSAL)
        pref_margin = (chosen - rejected) - ref_margins[forget.prompt]
        npo_loss = -torch.nn.functional.logsigmoid(beta * pref_margin)
        retain_loss = _pair_loss(model, tokenizer, retain.prompt, retain.completion)
        loss = npo_loss + 0.25 * retain_loss
        loss.backward()
        opt.step()


def _retain_metrics_for_examples(model, tokenizer, retain_examples: list[CausalExample]) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    with torch.no_grad():
        for ex in retain_examples:
            answer_loss = float(_pair_loss(model, tokenizer, ex.prompt, ex.completion))
            refusal_loss = float(_pair_loss(model, tokenizer, ex.prompt, REFUSAL))
            losses.append(answer_loss)
            correct += 1 if answer_loss <= refusal_loss else 0
    model.train()
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "retain_perplexity": round(float(torch.exp(torch.tensor(mean_loss))), 4),
        "retain_completion_accuracy": round(correct / len(retain_examples), 4) if retain_examples else 0.0,
    }


def _answer_terms(answer: str) -> list[str]:
    tokens = [tok.strip(".,:;!?()[]{}'\"").lower() for tok in answer.split()]
    return [tok for tok in tokens if len(tok) > 5][:5]


def _tofu_full_eval_rows(model, tokenizer, model_name: str, method: str, seed: int) -> list[dict[str, object]]:
    from datasets import load_dataset

    rows: list[dict[str, object]] = []
    for split in TOFU_SPLITS:
        ds = load_dataset("locuslab/TOFU", split, split="train")
        limit = _tofu_eval_limit(split, len(ds))
        if {"question", "answer"}.issubset(ds.column_names) and "option1" not in ds.column_names:
            examples = [_record_to_example(ds[i]) for i in range(limit)]
            metrics = _qa_eval_metrics(model, tokenizer, examples)
            eval_mode = "full" if limit == len(ds) else f"deterministic_subset_{limit}_of_{len(ds)}"
        elif {"option1", "option2", "option3", "option4", "answer"}.issubset(ds.column_names):
            metrics = _mc_eval_metrics(model, tokenizer, [ds[i] for i in range(limit)])
            eval_mode = "full" if limit == len(ds) else f"deterministic_subset_{limit}_of_{len(ds)}"
        else:
            metrics = {"answer_preferred_rate": 0.0, "mean_answer_margin": 0.0, "mean_answer_loss": 0.0}
            eval_mode = "manifest_only"
            limit = 0
        rows.append(
            {
                "track": "TOFU-FullEval",
                "deletion_method": method,
                "audit_method": "TOFU-Eval",
                "decision": "report",
                "ground_truth_failure": split.startswith("forget"),
                "detected_failure": False,
                "aggregate_risk_ucb": 0.0,
                "direct_leakage": 0.0,
                "paraphrase_leakage": 0.0,
                "retrieval_dependence": 0.0,
                "counterfactual_dependence": 0.0,
                "extraction_risk": 0.0,
                "watermark_hit": 0.0,
                "retain_utility": metrics["answer_preferred_rate"],
                "audit_calls": limit,
                "detection_per_1000_calls": 0.0,
                "seed": seed,
                "hf_model": model_name,
                "tofu_split": split,
                "tofu_split_size": len(ds),
                "tofu_eval_n": limit,
                "tofu_eval_mode": eval_mode,
                "tofu_answer_preferred_rate": metrics["answer_preferred_rate"],
                "tofu_mean_answer_margin": metrics["mean_answer_margin"],
                "tofu_mean_answer_loss": metrics["mean_answer_loss"],
            }
        )
    return rows


def _tofu_eval_limit(split: str, size: int) -> int:
    raw = os.environ.get("ADCU_TOFU_EVAL_LIMIT", "5").strip().lower()
    if raw in {"full", "all", "exhaustive"}:
        return size
    full_splits = {
        item.strip()
        for item in os.environ.get("ADCU_TOFU_FULL_SPLITS", "forget01").split(",")
        if item.strip()
    }
    if split in full_splits:
        return size
    return min(size, int(raw))


def _record_to_example(record) -> CausalExample:
    question = str(record["question"])
    answer = str(record.get("answer") or record.get("paraphrased_answer") or "")
    return CausalExample(f"User: {question}\nAssistant:", answer, None)


def _qa_eval_metrics(model, tokenizer, examples: list[CausalExample]) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    margins: list[float] = []
    preferred = 0
    with torch.no_grad():
        for ex in examples:
            answer_loss = float(_pair_loss(model, tokenizer, ex.prompt, ex.completion))
            refusal_loss = float(_pair_loss(model, tokenizer, ex.prompt, REFUSAL))
            margin = refusal_loss - answer_loss
            losses.append(answer_loss)
            margins.append(margin)
            preferred += 1 if margin >= 0 else 0
    model.train()
    return {
        "answer_preferred_rate": round(preferred / len(examples), 4) if examples else 0.0,
        "mean_answer_margin": round(sum(margins) / len(margins), 4) if margins else 0.0,
        "mean_answer_loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
    }


def _mc_eval_metrics(model, tokenizer, records: list[dict[str, object]]) -> dict[str, float]:
    model.eval()
    correct = 0
    margins: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for record in records:
            prompt = f"User: {record['question']}\nAssistant:"
            answer = str(record["answer"])
            options = [str(record[f"option{i}"]) for i in range(1, 5)]
            option_losses = [float(_pair_loss(model, tokenizer, prompt, option)) for option in options]
            answer_loss = float(_pair_loss(model, tokenizer, prompt, answer))
            best_loss = min(option_losses)
            losses.append(answer_loss)
            margins.append(best_loss - answer_loss)
            if answer_loss <= best_loss + 1e-8:
                correct += 1
    model.train()
    return {
        "answer_preferred_rate": round(correct / len(records), 4) if records else 0.0,
        "mean_answer_margin": round(sum(margins) / len(margins), 4) if margins else 0.0,
        "mean_answer_loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
    }
