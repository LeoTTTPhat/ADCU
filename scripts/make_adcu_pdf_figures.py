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
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    colors = {
        "target": ("#e8f1ff", "#2f6fbb"),
        "removed": ("#f4f4f4", "#9a9a9a"),
        "residual": ("#fff1dc", "#d97904"),
        "answer": ("#ffe4e6", "#c4314b"),
        "audit": ("#eaf7ed", "#23824a"),
    }

    def box(label: str, x: float, y: float, w: float, h: float, kind: str, fontsize: int = 7) -> None:
        face, edge = colors[kind]
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="#1f2933")

    box_sizes: dict[str, tuple[float, float]] = {}

    def node_box(name: str, label: str, w: float, h: float, kind: str, fontsize: int = 7) -> None:
        box_sizes[name] = (w, h)
        box(label, *nodes[name], w, h, kind, fontsize=fontsize)

    def anchor(name: str, side: str, offset: float = 0.0) -> tuple[float, float]:
        x, y = nodes[name]
        w, h = box_sizes[name]
        if side == "left":
            return (x - w / 2, y + offset)
        if side == "right":
            return (x + w / 2, y + offset)
        if side == "top":
            return (x + offset, y + h / 2)
        if side == "bottom":
            return (x + offset, y - h / 2)
        raise ValueError(f"unknown side: {side}")

    def arrow(
        src: str,
        src_side: str,
        dst: str,
        dst_side: str,
        color: str = "#59636e",
        dashed: bool = False,
        src_offset: float = 0.0,
        dst_offset: float = 0.0,
        rad: float = 0.0,
    ) -> None:
        start = anchor(src, src_side, src_offset)
        end = anchor(dst, dst_side, dst_offset)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.0,
                "linestyle": "--" if dashed else "-",
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 9,
                "connectionstyle": f"arc3,rad={rad}",
            },
        )

    ax.text(0.50, 0.985, "Case-study provenance graph", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(0.10, 0.88, "target", ha="center", fontsize=7, color="#4b5563")
    ax.text(0.39, 0.88, "derived artifacts", ha="center", fontsize=7, color="#4b5563")
    ax.text(0.70, 0.88, "behavior", ha="center", fontsize=7, color="#4b5563")
    ax.text(0.91, 0.88, "audit", ha="center", fontsize=7, color="#4b5563")

    nodes = {
        "target": (0.10, 0.56),
        "chunk": (0.28, 0.73),
        "embedding": (0.28, 0.56),
        "summary": (0.28, 0.39),
        "cache": (0.48, 0.73),
        "synthetic": (0.48, 0.56),
        "adapter": (0.48, 0.39),
        "retriever": (0.69, 0.73),
        "answer": (0.69, 0.56),
        "memory": (0.69, 0.39),
        "audit": (0.91, 0.56),
    }

    node_box("target", "Deleted\nrecord $z$", 0.16, 0.16, "target", fontsize=8.2)
    node_box("chunk", "Raw chunk\nremoved", 0.15, 0.11, "removed", fontsize=7.0)
    node_box("embedding", "Embedding\nstale", 0.15, 0.11, "residual", fontsize=7.0)
    node_box("summary", "Summary\nderivative", 0.15, 0.11, "residual", fontsize=7.0)
    node_box("cache", "Citation\ncache", 0.15, 0.11, "residual", fontsize=7.0)
    node_box("synthetic", "Synthetic\nQA", 0.15, 0.11, "residual", fontsize=7.0)
    node_box("adapter", "LoRA\nupdate", 0.15, 0.11, "residual", fontsize=7.0)
    node_box("retriever", "Retriever\nhit", 0.15, 0.11, "answer", fontsize=7.0)
    node_box("answer", "Paraphrased\nanswer", 0.15, 0.11, "answer", fontsize=7.0)
    node_box("memory", "Adapter\nmemory", 0.15, 0.11, "answer", fontsize=7.0)
    node_box("audit", "ADCU evidence\n\ndirect\nparaphrase\nretrieval\ncounterfactual\nextraction", 0.17, 0.46, "audit", fontsize=6.6)

    edges = [
        ("target", "right", "chunk", "left", "#9a9a9a", True, 0.045, 0.0, 0.04),
        ("target", "right", "embedding", "left", "#d97904", False, 0.0, 0.0, 0.0),
        ("target", "right", "summary", "left", "#d97904", False, -0.045, 0.0, -0.04),
        ("chunk", "right", "cache", "left", "#d97904", False, 0.0, 0.0, 0.0),
        ("embedding", "right", "synthetic", "left", "#d97904", False, 0.0, 0.0, 0.0),
        ("summary", "right", "adapter", "left", "#d97904", False, 0.0, 0.0, 0.0),
        ("synthetic", "bottom", "adapter", "top", "#d97904", False, 0.0, 0.0, 0.0),
        ("cache", "right", "retriever", "left", "#c4314b", False, 0.0, 0.0, 0.0),
        ("synthetic", "right", "answer", "left", "#c4314b", False, 0.0, 0.0, 0.0),
        ("adapter", "right", "memory", "left", "#c4314b", False, 0.0, 0.0, 0.0),
        ("retriever", "right", "audit", "left", "#23824a", True, 0.0, 0.11, 0.0),
        ("answer", "right", "audit", "left", "#23824a", True, 0.0, 0.0, 0.0),
        ("memory", "right", "audit", "left", "#23824a", True, 0.0, -0.11, 0.0),
    ]
    for src, src_side, dst, dst_side, color, dashed, src_offset, dst_offset, rad in edges:
        arrow(src, src_side, dst, dst_side, color=color, dashed=dashed, src_offset=src_offset, dst_offset=dst_offset, rad=rad)

    legend = [
        Line2D([0], [0], color="#9a9a9a", lw=1.2, linestyle="--", label="deleted path"),
        Line2D([0], [0], color="#d97904", lw=1.2, label="residual artifact"),
        Line2D([0], [0], color="#c4314b", lw=1.2, label="behavioral leak"),
        Line2D([0], [0], color="#23824a", lw=1.2, linestyle="--", label="audit evidence"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=6.0)

    fig.tight_layout(pad=0.4)
    fig.savefig(FIG_DIR / "provenance_case.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
