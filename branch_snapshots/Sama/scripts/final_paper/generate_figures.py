"""Generate the paper figures for ``docs/final_paper`` as vector PDFs.

Every figure is generated directly from a validated source CSV under
``outputs/``, mirroring the existing milestone plotting scripts but emitting
publication-ready PDF files into ``docs/final_paper/figures/``.  No plotted
value is edited by hand.

Sources (see ``docs/final_paper/README.md`` and ``EVIDENCE_MAP.md``):

  * kitti_vs_waymo_map50_95.pdf     -> outputs/milestone_6/generalization_analysis/tables/kitti_vs_waymo_comparison.csv
  * generalization_ratio_map50_95.pdf -> same CSV
  * class_wise_degradation.pdf      -> outputs/milestone_6/generalization_analysis/tables/class_wise_degradation.csv
  * object_size_recall.pdf          -> outputs/milestone_7/object_size_analysis/object_size_summary.csv
  * pedestrian_cyclist_fn_rate.pdf  -> outputs/milestone_7/safety_error_analysis/safety_false_negative_summary.csv
  * deployment_tradeoff.pdf         -> outputs/milestone_7/deployment_tradeoff/deployment_suitability_table.csv

Usage::

    python scripts/final_paper/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "final_paper" / "figures"

M5 = ROOT / "outputs" / "milestone_5" / "figures"
M6_TABLES = ROOT / "outputs" / "milestone_6" / "generalization_analysis" / "tables"
M7 = ROOT / "outputs" / "milestone_7"

DETECTOR_ORDER = ["yolo", "rtdetr", "faster_rcnn", "retinanet"]
DISPLAY = {
    "yolo": "YOLOv8s",
    "rtdetr": "RT-DETR-L",
    "faster_rcnn": "Faster R-CNN",
    "retinanet": "RetinaNet",
}
COLORS = {
    "yolo": "#4C72B0",
    "rtdetr": "#DD8452",
    "faster_rcnn": "#C44E52",
    "retinanet": "#55A868",
}
DATASET_LABEL = {"kitti": "KITTI validation", "waymo": "Waymo external"}


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["detector"].isin(DETECTOR_ORDER)]
    df["detector"] = pd.Categorical(df["detector"], categories=DETECTOR_ORDER, ordered=True)
    return df.sort_values("detector").reset_index(drop=True)


def _xticklabels() -> list[str]:
    return [DISPLAY[d] for d in DETECTOR_ORDER]


def fig_kitti_vs_waymo_map50_95() -> None:
    df = _load(M6_TABLES / "kitti_vs_waymo_comparison.csv")
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - width / 2, df["KITTI_mAP50_95"], width, label="KITTI validation", color="#8DA0CB")
    ax.bar(x + width / 2, df["Waymo_mAP50_95"], width, label="Waymo external", color="#FC8D62")
    ax.set_ylabel("mAP@0.50:0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(_xticklabels())
    ax.set_ylim(0, df["KITTI_mAP50_95"].max() * 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "kitti_vs_waymo_map50_95.pdf")


def fig_generalization_ratio() -> None:
    df = _load(M6_TABLES / "kitti_vs_waymo_comparison.csv")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(x, df["mAP50_95_generalization_ratio"], color=[COLORS[d] for d in df["detector"]])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="No drop (ratio = 1.0)")
    ax.set_ylabel("Generalization ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(_xticklabels())
    ax.set_ylim(0, df["mAP50_95_generalization_ratio"].max() * 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "generalization_ratio_map50_95.pdf")


def fig_class_wise_degradation() -> None:
    df = _load(M6_TABLES / "class_wise_degradation.csv")
    classes = ["Vehicle", "Pedestrian", "Cyclist"]
    x = np.arange(len(classes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, det in enumerate(DETECTOR_ORDER):
        sub = df[df["detector"] == det].set_index("class_name").loc[classes]
        ax.bar(x + (i - 1.5) * width, sub["absolute_drop"], width, label=DISPLAY[det], color=COLORS[det])
    ax.set_ylabel("Absolute drop in AP@0.50:0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "class_wise_degradation.pdf")


def fig_object_size_recall() -> None:
    df = _load(M7 / "object_size_analysis" / "object_size_summary.csv")
    df = df[df["class_name"] == "all"]
    sizes = ["small", "medium", "large"]
    size_labels = ["Small", "Medium", "Large"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    x = np.arange(3)
    width = 0.2
    for ax, ds in zip(axes, ["kitti", "waymo"]):
        sub = df[df["dataset"] == ds]
        for i, det in enumerate(DETECTOR_ORDER):
            r = sub[sub["detector"] == det].set_index("size_category").loc[sizes]
            ax.bar(x + (i - 1.5) * width, r["recall"], width, label=DISPLAY[det], color=COLORS[det])
        ax.set_xticks(x)
        ax.set_xticklabels(size_labels)
        ax.set_title(DATASET_LABEL[ds])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Recall")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "object_size_recall.pdf")


def fig_pedestrian_cyclist_fn_rate() -> None:
    df = _load(M7 / "safety_error_analysis" / "safety_false_negative_summary.csv")
    df = df[df["class_name"] == "pedestrian+cyclist"]
    agg = df.groupby(["dataset", "detector"]).agg(tp=("tp", "sum"), fn=("fn", "sum")).reset_index()
    agg["fn_rate"] = agg["fn"] / (agg["tp"] + agg["fn"])
    x = np.arange(len(DETECTOR_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for offset, ds in zip([-width / 2, width / 2], ["kitti", "waymo"]):
        sub = agg[agg["dataset"] == ds].set_index("detector").loc[DETECTOR_ORDER]
        ax.bar(x + offset, sub["fn_rate"], width, label=DATASET_LABEL[ds])
    ax.set_xticks(x)
    ax.set_xticklabels(_xticklabels())
    ax.set_ylabel("Pedestrian + cyclist FN rate")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "pedestrian_cyclist_fn_rate.pdf")


def fig_deployment_tradeoff() -> None:
    df = _load(M7 / "deployment_tradeoff" / "deployment_suitability_table.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for _, r in df.iterrows():
        ax.scatter(
            r["waymo_mean_inference_ms"],
            r["Waymo_mAP50_95"],
            s=140,
            color=COLORS[r["detector"]],
            label=DISPLAY[r["detector"]],
        )
    ax.set_xlabel("Mean inference time (ms/image, Waymo)")
    ax.set_ylabel("Waymo mAP@0.50:0.95")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "deployment_tradeoff.pdf")


def main() -> None:
    print("=" * 79)
    print("Final paper - generate figures from validated artifacts")
    print("=" * 79)
    fig_kitti_vs_waymo_map50_95()
    fig_generalization_ratio()
    fig_class_wise_degradation()
    fig_object_size_recall()
    fig_pedestrian_cyclist_fn_rate()
    fig_deployment_tradeoff()
    print("Done.")


if __name__ == "__main__":
    main()
