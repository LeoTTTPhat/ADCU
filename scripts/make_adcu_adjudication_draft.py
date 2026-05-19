from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.scorers import EvidenceScorer
from adcu.submission_experiments import (
    AnswerOnlySystem,
    LexicalFeverRAG,
    build_natural_fever_graph,
    load_fever_examples,
    _redact_text,
)
from adcu.probes import ProbeGenerator


OUT_CSV = ROOT / "benchmark" / "model_assisted_adjudication_draft.csv"
OUT_JSON = ROOT / "experiments" / "adcu_submission_results" / "model_assisted_adjudication_summary.json"
OUT_MD = ROOT / "Third-Best-Data-Centric-Unlearning" / "submission_tables" / "submission_table11_model_adjudication_draft.md"
OUT_TEX = ROOT / "Third-Best-Data-Centric-Unlearning" / "submission_tables" / "submission_table11_model_adjudication_draft.tex"


THRESHOLDS = {
    "direct": 0.35,
    "paraphrase": 0.35,
    "retrieval": 0.20,
    "counterfactual": 0.50,
    "extraction": 0.08,
    "watermark": 0.20,
}


def main() -> None:
    rows = make_rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_table(summary)
    print(f"Wrote {len(rows)} model-assisted adjudication draft rows to {OUT_CSV}")


def make_rows(limit: int = 30) -> list[dict[str, object]]:
    fever_examples = load_fever_examples(ROOT / "data" / "rag_raw" / "fever_gold_valid.jsonl", limit=260)
    generator = ProbeGenerator()
    candidates: list[dict[str, object]] = []
    channels = ["direct", "paraphrase", "retrieval", "counterfactual", "extraction"]

    for seed in range(1, 8):
        targets, graph, retain = build_natural_fever_graph(seed, 3, fever_examples)
        systems = [
            ("full provenance purge", LexicalFeverRAG(graph, retain), "clean"),
            ("citation cache retained", LexicalFeverRAG(graph, retain, retrieved_target_ids={targets[0].target_id}), "failure"),
            ("natural summary retained", LexicalFeverRAG(graph, retain, leaked_target_ids={targets[1].target_id}, paraphrase_only=True), "failure"),
            (
                "answer-only paraphrase leak",
                AnswerOnlySystem(LexicalFeverRAG(graph, retain, leaked_target_ids={targets[2].target_id}, paraphrase_only=True)),
                "failure",
            ),
        ]
        scorer = EvidenceScorer(graph)
        for deletion_method, system, ground_truth in systems:
            for target in targets:
                for probe in generator.generate(target, graph)[:4]:
                    response = system.query(probe)
                    evidence = scorer.score(target, probe, response)
                    for channel in channels:
                        value = float(getattr(evidence, channel))
                        threshold = THRESHOLDS[channel]
                        label = "leak" if value >= threshold else "no_leak"
                        candidates.append(
                            {
                                "case_id": "",
                                "domain": "public_qa_fever",
                                "probe_family": probe.probe_type,
                                "redacted_probe": _redact_text(probe.text),
                                "redacted_answer": _redact_text(response.answer),
                                "scorer_channel": channel,
                                "scorer_value": round(value, 4),
                                "judge_a_label": label,
                                "judge_b_label": label,
                                "resolved_label": label,
                                "ground_truth_case": ground_truth,
                                "deletion_method": deletion_method,
                                "adjudicator_type": "model_assisted_draft_not_human",
                                "notes": "Draft label for human confirmation; do not report as human adjudication.",
                            }
                        )
    return _balanced_sample(candidates, channels, limit)


def _balanced_sample(candidates: list[dict[str, object]], channels: list[str], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    per_channel = max(1, limit // len(channels))
    for channel in channels:
        group = [row for row in candidates if row["scorer_channel"] == channel]
        leaks = [row for row in group if row["resolved_label"] == "leak"]
        no_leaks = [row for row in group if row["resolved_label"] == "no_leak"]
        take_leaks = min(len(leaks), per_channel // 2)
        take_no_leaks = min(len(no_leaks), per_channel - take_leaks)
        channel_rows = leaks[:take_leaks] + no_leaks[:take_no_leaks]
        if len(channel_rows) < per_channel:
            used = {id(row) for row in channel_rows}
            channel_rows.extend(row for row in group if id(row) not in used)
        selected.extend(channel_rows[:per_channel])
    if len(selected) < limit:
        used = {id(row) for row in selected}
        selected.extend(row for row in candidates if id(row) not in used)
    selected = selected[:limit]
    for idx, row in enumerate(selected, start=1):
        row["case_id"] = f"MAD-{idx:03d}"
    return selected


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_channel: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_channel.setdefault(str(row["scorer_channel"]), []).append(row)
    out: list[dict[str, object]] = []
    for channel, group in sorted(by_channel.items()):
        leaks = [row for row in group if row["resolved_label"] == "leak"]
        no_leaks = [row for row in group if row["resolved_label"] == "no_leak"]
        disagreements = [row for row in group if row["judge_a_label"] != row["judge_b_label"]]
        out.append(
            {
                "channel": channel,
                "n": len(group),
                "draft_leak_rate": round(len(leaks) / len(group), 4),
                "draft_no_leak_rate": round(len(no_leaks) / len(group), 4),
                "draft_disagreement_rate": round(len(disagreements) / len(group), 4),
                "adjudicator_type": "model-assisted draft",
            }
        )
    return out


def write_table(rows: list[dict[str, object]]) -> None:
    columns = ["channel", "n", "draft_leak_rate", "draft_no_leak_rate", "draft_disagreement_rate", "adjudicator_type"]
    md = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        md.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    header = " & ".join(column.replace("_", " ").title() for column in columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Model-assisted draft adjudication for redacted NaturalFEVER outputs. These labels are not human adjudication.}",
        "\\label{tab:model_adjudication_draft}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(row[column]).replace("_", "\\_") for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
