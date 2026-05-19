from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "adcu_pdf"


def main() -> None:
    import matplotlib.pyplot as plt
    global FancyBboxPatch
    from matplotlib.patches import FancyBboxPatch

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline(plt)
    _bars(plt)
    _budget(plt)
    _provenance(plt)
    print(f"Wrote PDF figures to {FIG_DIR}")


def _pipeline(plt) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    ax.axis("off")
    labels = ["Deletion\nrequest", "Provenance\ngraph", "Value-guided\nprobes", "Black-box\nresponses", "Risk UCB +\nretain utility"]
    xs = [0.02, 0.22, 0.42, 0.62, 0.82]
    for x, label in zip(xs, labels):
        box = FancyBboxPatch((x, 0.28), 0.15, 0.42, boxstyle="round,pad=0.02", facecolor="#f4f7fb", edgecolor="#2f6fbb")
        ax.add_patch(box)
        ax.text(x + 0.075, 0.49, label, ha="center", va="center", fontsize=8)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + 0.19, 0.49), xytext=(x + 0.16, 0.49), arrowprops={"arrowstyle": "->", "color": "#555"})
    ax.set_title("ADCU audit pipeline", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adcu_pipeline.pdf")
    plt.close(fig)


def _bars(plt) -> None:
    rows = json.loads((ROOT / "experiments" / "adcu_submission_results" / "track_audit_summary.json").read_text())
    rows = [r for r in rows if r["track"] == "HybridReal"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    labels = [r["audit_method"] for r in rows]
    vals = [r["failure_detection_rate"] for r in rows]
    ax.barh(labels, vals, color="#2f6fbb")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Failure detection rate")
    ax.set_title("HybridReal detection by audit method", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hybrid_real_detection.pdf")
    plt.close(fig)


def _budget(plt) -> None:
    rows = json.loads((ROOT / "experiments" / "adcu_submission_results" / "budget_summary.json").read_text())
    fig, ax1 = plt.subplots(figsize=(5.4, 2.8))
    budgets = [r["budget"] for r in rows]
    det = [r["failure_detection_rate"] for r in rows]
    risk = [r["mean_risk_ucb"] for r in rows]
    ax1.plot(budgets, det, marker="o", label="Detection", color="#2f6fbb")
    ax1.set_ylim(0.9, 1.02)
    ax1.set_xlabel("Audit budget")
    ax1.set_ylabel("Detection")
    ax2 = ax1.twinx()
    ax2.plot(budgets, risk, marker="s", label="Risk UCB", color="#d98c2b")
    ax2.set_ylabel("Mean risk UCB")
    ax1.set_title("Budget versus detection and risk", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "budget_detection_risk.pdf")
    plt.close(fig)


def _provenance(plt) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    ax.axis("off")
    nodes = {
        "Raw private\nrecord": (0.08, 0.55),
        "Chunk /\nembedding": (0.32, 0.75),
        "Summary /\ncache": (0.32, 0.35),
        "Synthetic\nQA": (0.58, 0.55),
        "Adapter /\nanswer": (0.82, 0.55),
    }
    for label, (x, y) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - 0.07, y - 0.11), 0.14, 0.16, boxstyle="round,pad=0.02", facecolor="#fff8ed", edgecolor="#d98c2b"))
        ax.text(x, y - 0.03, label, ha="center", va="center", fontsize=8)
    edges = [
        ("Raw private\nrecord", "Chunk /\nembedding"),
        ("Raw private\nrecord", "Summary /\ncache"),
        ("Chunk /\nembedding", "Synthetic\nQA"),
        ("Summary /\ncache", "Synthetic\nQA"),
        ("Synthetic\nQA", "Adapter /\nanswer"),
    ]
    for a, b in edges:
        ax.annotate("", xy=nodes[b], xytext=nodes[a], arrowprops={"arrowstyle": "->", "color": "#555"})
    ax.set_title("Example deletion-target provenance graph", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "provenance_case.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
