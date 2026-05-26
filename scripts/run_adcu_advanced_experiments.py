from __future__ import annotations

from csv import DictWriter
import json
from math import sqrt
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.dense_rag import run_dense_rag_experiment
from adcu.lora_sft import run_lora_sft_experiment
from adcu.peft_lora import run_peft_lora_validation
from adcu.pretrained_lora import run_pretrained_lora_experiment
from adcu.tofu_lora import run_qwen_tofu_lora_experiment


OUT_DIR = ROOT / "experiments" / "adcu_advanced_results"


def write_rows(name: str, rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (OUT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def ci_summary(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        vals = [1.0 if row["detected_failure"] else 0.0 for row in group if row["ground_truth_failure"]]
        risks = [float(row["aggregate_risk_ucb"]) for row in group]
        rate = sum(vals) / len(vals) if vals else 0.0
        se = sqrt(rate * (1 - rate) / len(vals)) if vals else 0.0
        boot_low, boot_high = bootstrap_ci(vals)
        record = {name: value for name, value in zip(keys, key)}
        record.update(
            {
                "failure_detection_rate": round(rate, 4),
                "ci95_low": round(max(0.0, rate - 1.96 * se), 4),
                "ci95_high": round(min(1.0, rate + 1.96 * se), 4),
                "bootstrap_ci95_low": boot_low,
                "bootstrap_ci95_high": boot_high,
                "mean_risk_ucb": round(sum(risks) / len(risks), 4) if risks else 0.0,
                "n": len(group),
            }
        )
        out.append(record)
    return out


def bootstrap_ci(vals: list[float], samples: int = 1000) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(123)
    means = []
    for _ in range(samples):
        draw = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(draw) / len(draw))
    means.sort()
    low = means[int(0.025 * (samples - 1))]
    high = means[int(0.975 * (samples - 1))]
    return (round(low, 4), round(high, 4))


def per_seed_variance(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((row["track"], row["seed"]), []).append(row)
    out = []
    for (track, seed), group in sorted(grouped.items()):
        failures = [row for row in group if row["ground_truth_failure"]]
        detected = [1.0 if row["detected_failure"] else 0.0 for row in failures]
        risks = [float(row["aggregate_risk_ucb"]) for row in group if row["audit_method"] == "ADCU"]
        out.append(
            {
                "track": track,
                "seed": seed,
                "failure_detection_rate": round(sum(detected) / len(detected), 4) if detected else 0.0,
                "mean_adcu_risk_ucb": round(sum(risks) / len(risks), 4) if risks else 0.0,
                "n": len(group),
            }
        )
    return out


def probe_family_ablation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    families = {
        "direct_only": lambda row: float(row["direct_leakage"]) >= 0.40,
        "paraphrase_only": lambda row: float(row["paraphrase_leakage"]) >= 0.35,
        "retrieval_only": lambda row: float(row["retrieval_dependence"]) >= 0.20,
        "extraction_only": lambda row: float(row["extraction_risk"]) >= 0.08,
        "full_adcu": lambda row: bool(row["detected_failure"]),
    }
    out = []
    adc_rows = [row for row in rows if row["audit_method"] == "ADCU" and row["ground_truth_failure"]]
    for track in sorted(set(row["track"] for row in adc_rows)):
        group = [row for row in adc_rows if row["track"] == track]
        for family, detector in families.items():
            vals = [1.0 if detector(row) else 0.0 for row in group]
            low, high = bootstrap_ci(vals)
            if len(vals) < 3:
                low, high = "--", "--"
            out.append(
                {
                    "track": track,
                    "probe_family": family,
                    "failure_detection_rate": round(sum(vals) / len(vals), 4) if vals else 0.0,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "n": len(vals),
                }
            )
    return out


def retriever_hit_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    dense = [row for row in rows if row["track"] == "DenseRAG" and row["audit_method"] == "ADCU"]
    out = []
    for mode in sorted(set(row["retrieval_mode"] for row in dense)):
        group = [row for row in dense if row["retrieval_mode"] == mode]
        out.append(
            {
                "retrieval_mode": mode,
                "embedding_model": sorted(set(str(row.get("embedding_model", "")) for row in group))[0],
                "top3_deleted_derivative_hit_rate": round(sum(float(row.get("top3_deleted_derivative_hit_rate", 0.0)) for row in group) / len(group), 4),
                "top5_deleted_derivative_hit_rate": round(sum(float(row.get("top5_deleted_derivative_hit_rate", 0.0)) for row in group) / len(group), 4),
                "adcu_detection_rate": round(sum(1.0 if row["detected_failure"] else 0.0 for row in group) / len(group), 4),
                "n": len(group),
            }
        )
    return out


def pretrained_margin_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adc = [
        row
        for row in rows
        if row["track"] in {"Pretrained-LoRA", "Qwen-TOFU-LoRA"} and row["audit_method"] == "ADCU"
    ]
    out = []
    for track, method in sorted(set((row["track"], row["deletion_method"]) for row in adc)):
        group = [row for row in adc if row["track"] == track and row["deletion_method"] == method]
        decisions = sorted(set(str(row["decision"]) for row in group))
        out.append(
            {
                "track": track,
                "method": method,
                "active_forget_targets": round(sum(float(row.get("active_forget_targets", 0.0)) for row in group) / len(group), 4),
                "mean_forget_margin": round(sum(float(row.get("mean_forget_margin", 0.0)) for row in group) / len(group), 4),
                "retain_perplexity": round(sum(float(row.get("retain_perplexity", 0.0)) for row in group) / len(group), 4),
                "retain_completion_accuracy": round(sum(float(row.get("retain_completion_accuracy", 0.0)) for row in group) / len(group), 4),
                "adcu_decision": "/".join(decisions),
                "failure_detection_rate": round(sum(1.0 if row["detected_failure"] else 0.0 for row in group) / len(group), 4),
                "n": len(group),
            }
        )
    return out


def tofu_eval_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = {
        "forget01",
        "retain99",
        "real_authors",
        "world_facts",
        "forget01_perturbed",
        "retain_perturbed",
    }
    tofu = [
        row
        for row in rows
        if row["track"] == "TOFU-FullEval"
        and row.get("tofu_split") in wanted
    ]
    out = []
    for row in sorted(tofu, key=lambda item: (str(item["deletion_method"]), str(item["tofu_split"]))):
        out.append(
            {
                "method": row["deletion_method"],
                "tofu_split": row["tofu_split"],
                "split_size": row["tofu_split_size"],
                "eval_n": row["tofu_eval_n"],
                "eval_mode": row["tofu_eval_mode"],
                "answer_preferred_rate": row["tofu_answer_preferred_rate"],
                "mean_answer_margin": row["tofu_mean_answer_margin"],
                "mean_answer_loss": row["tofu_mean_answer_loss"],
            }
        )
    return out


def tofu_split_coverage(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    tofu = [row for row in rows if row["track"] == "TOFU-FullEval"]
    seen = {}
    for row in tofu:
        seen.setdefault(row["tofu_split"], row)
    out = []
    for split, row in sorted(seen.items()):
        out.append(
            {
                "tofu_split": split,
                "split_size": row["tofu_split_size"],
                "default_eval_n": row["tofu_eval_n"],
                "default_eval_mode": row["tofu_eval_mode"],
                "exhaustive_qwen_command": "ADCU_TOFU_EVAL_LIMIT=full",
            }
        )
    return out


def case_studies(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for wanted in [
        ("PEFT-LoRA", "tiny HF PEFT LoRA validation", "ADCU"),
        ("Pretrained-LoRA", "pretrained LoRA NPO unlearning", "ADCU"),
        ("LoRA-SFT", "full adapter no unlearning", "ADCU"),
        ("DenseRAG", "dense retrieval with retained derivatives", "RetrieverHit"),
        ("DenseRAG", "e5 retrieval with retained derivatives", "ADCU"),
        ("DenseRAG", "lexical retrieval with retained derivatives", "ADCU"),
    ]:
        for row in rows:
            if (row["track"], row["deletion_method"], row["audit_method"]) == wanted and row["detected_failure"]:
                selected.append(
                    {
                        "case_id": f"{wanted[0]}::{wanted[1]}::{wanted[2]}",
                        "risk_ucb": row["aggregate_risk_ucb"],
                        "direct_leakage": row["direct_leakage"],
                        "paraphrase_leakage": row["paraphrase_leakage"],
                        "retrieval_dependence": row["retrieval_dependence"],
                        "redacted_probe": "[REDACTED USER] deletion audit probe",
                        "redacted_output": "[REDACTED SECRET] residual behavior detected",
                        "interpretation": "Residual dependence remains after the nominal deletion/unlearning operation.",
                    }
                )
                break
    return selected


def main() -> None:
    rows = []
    lora = run_lora_sft_experiment(OUT_DIR)
    peft = run_peft_lora_validation(OUT_DIR)
    pretrained = run_pretrained_lora_experiment(OUT_DIR)
    qwen_tofu = run_qwen_tofu_lora_experiment(OUT_DIR)
    dense = run_dense_rag_experiment(ROOT / "data")
    rows.extend(lora)
    rows.extend(peft)
    rows.extend(pretrained)
    rows.extend(qwen_tofu)
    rows.extend(dense)

    write_rows("advanced_results", rows)
    audit_rows = [row for row in rows if row["track"] != "TOFU-FullEval"]
    write_rows("advanced_ci_by_track_audit", ci_summary(audit_rows, ["track", "audit_method"]))
    write_rows("advanced_ci_by_method", ci_summary(audit_rows, ["track", "deletion_method", "audit_method"]))
    write_rows("advanced_per_seed_variance", per_seed_variance(audit_rows))
    write_rows("advanced_probe_family_ablation", probe_family_ablation(audit_rows))
    write_rows("advanced_retriever_hit_summary", retriever_hit_summary(audit_rows))
    write_rows("advanced_pretrained_margin_summary", pretrained_margin_summary(audit_rows))
    write_rows("advanced_tofu_eval_summary", tofu_eval_summary(rows))
    write_rows("advanced_tofu_split_coverage", tofu_split_coverage(rows))
    write_rows("case_studies", case_studies(audit_rows))

    print(f"Wrote {len(rows)} advanced rows to {OUT_DIR}")
    for row in ci_summary(audit_rows, ["track", "audit_method"]):
        if row["audit_method"] == "ADCU":
            print(
                f"{row['track']:10s} ADCU detection={row['failure_detection_rate']:.3f} "
                f"CI=[{row['ci95_low']:.3f},{row['ci95_high']:.3f}] risk={row['mean_risk_ucb']:.3f}"
            )


if __name__ == "__main__":
    main()
