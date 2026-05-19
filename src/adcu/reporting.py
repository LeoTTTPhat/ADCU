from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable

from .metrics import ExperimentRow


def write_rows(rows: list[ExperimentRow], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dict_rows = [row.to_dict() for row in rows]
    (out_dir / f"{stem}.json").write_text(json.dumps(dict_rows, indent=2), encoding="utf-8")
    if not dict_rows:
        return
    with (out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dict_rows)


def read_rows(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def detection_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["track"]), str(row["audit_method"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (track, audit_method), group in sorted(groups.items()):
        failures = [row for row in group if bool(row["ground_truth_failure"])]
        clean = [row for row in group if not bool(row["ground_truth_failure"])]
        true_detected = sum(1 for row in failures if bool(row["detected_failure"]))
        false_detected = sum(1 for row in clean if bool(row["detected_failure"]))
        calls = sum(int(row["audit_calls"]) for row in group)
        summary.append(
            {
                "track": track,
                "audit_method": audit_method,
                "failure_detection_rate": round(true_detected / len(failures), 4) if failures else 0.0,
                "false_alarm_rate": round(false_detected / len(clean), 4) if clean else 0.0,
                "mean_risk_ucb": round(_mean([float(row["aggregate_risk_ucb"]) for row in group]), 4),
                "detections_per_1000_calls": round(1000.0 * true_detected / calls, 2) if calls else 0.0,
            }
        )
    return summary


def residual_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if str(row["audit_method"]) != "ADCU":
            continue
        key = (str(row["track"]), str(row["deletion_method"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (track, deletion_method), group in sorted(groups.items()):
        summary.append(
            {
                "track": track,
                "deletion_method": deletion_method,
                "direct_leakage": round(_mean([float(row["direct_leakage"]) for row in group]), 4),
                "paraphrase_leakage": round(_mean([float(row["paraphrase_leakage"]) for row in group]), 4),
                "retrieval_dependence": round(_mean([float(row["retrieval_dependence"]) for row in group]), 4),
                "extraction_risk": round(_mean([float(row["extraction_risk"]) for row in group]), 4),
                "retain_utility": round(_mean([float(row["retain_utility"]) for row in group]), 4),
                "risk_ucb": round(_mean([float(row["aggregate_risk_ucb"]) for row in group]), 4),
            }
        )
    return summary


def write_markdown_table(rows: list[dict[str, object]], path: Path, columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(rows: list[dict[str, object]], path: Path, columns: list[str], caption: str, label: str) -> None:
    header = " & ".join(_latex_escape(column.replace("_", " ").title()) for column in columns)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [_latex_escape(row.get(column, "")) for column in columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_bar(
    rows: list[dict[str, object]],
    path: Path,
    title: str,
    label_key: str,
    value_key: str,
    width: int = 980,
    height: int = 460,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin = {"left": 230, "right": 45, "top": 56, "bottom": 50}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_value = max(float(row[value_key]) for row in rows) or 1.0
    bar_h = min(24, plot_h / max(len(rows), 1) * 0.62)
    gap = (plot_h - bar_h * len(rows)) / max(len(rows) - 1, 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{_xml(title)}</text>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = margin["left"] + tick * plot_w / max_value
        parts.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{height-margin["bottom"]}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.2g}</text>')
    for idx, row in enumerate(rows):
        y = margin["top"] + idx * (bar_h + gap)
        value = float(row[value_key])
        w = value * plot_w / max_value
        label = str(row[label_key])
        parts.append(f'<text x="{margin["left"]-10}" y="{y+bar_h*0.68:.1f}" text-anchor="end" font-family="Arial" font-size="12">{_xml(label)}</text>')
        parts.append(f'<rect x="{margin["left"]}" y="{y:.1f}" width="{w:.1f}" height="{bar_h}" fill="#2f6fbb" rx="3"/>')
        parts.append(f'<text x="{margin["left"]+w+6:.1f}" y="{y+bar_h*0.68:.1f}" font-family="Arial" font-size="12">{value:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_pipeline(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes = [
        "Deletion\\nRequest",
        "Provenance\\nGraph",
        "Value-Guided\\nProbe Allocation",
        "Black-Box\\nBehavior Tests",
        "Risk UCB +\\nRetain Utility",
    ]
    width, height = 1020, 210
    box_w = 165
    gap = 34
    x0, y = 40, 76
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#555"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="510" y="33" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">ADCU Audit Pipeline</text>',
    ]
    for idx, label in enumerate(boxes):
        x = x0 + idx * (box_w + gap)
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="78" fill="#f4f7fb" stroke="#2f6fbb" stroke-width="1.5" rx="6"/>')
        for j, line in enumerate(label.split("\\n")):
            parts.append(f'<text x="{x+box_w/2}" y="{y+31+j*18}" text-anchor="middle" font-family="Arial" font-size="13">{_xml(line)}</text>')
        if idx < len(boxes) - 1:
            parts.append(f'<line x1="{x+box_w+7}" y1="{y+39}" x2="{x+box_w+gap-8}" y2="{y+39}" stroke="#555" stroke-width="1.4" marker-end="url(#arrow)"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def _xml(value: object) -> str:
    return html.escape(str(value), quote=True)
