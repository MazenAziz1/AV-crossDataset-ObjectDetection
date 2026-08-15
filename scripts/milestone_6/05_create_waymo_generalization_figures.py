import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DISPLAY_NAMES = {
    "yolo": "YOLOv8s",
    "rtdetr": "RT-DETR-L",
    "retinanet": "RetinaNet",
    "faster_rcnn": "Faster R-CNN",
}
DETECTOR_ORDER = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]
COLORS = {
    "yolo": "#4C72B0",
    "rtdetr": "#DD8452",
    "retinanet": "#55A868",
    "faster_rcnn": "#C44E52",
}
CLASS_COLORS = {"Vehicle": "#4C72B0", "Pedestrian": "#55A868", "Cyclist": "#DD8452"}


def project_root():
    return Path(__file__).resolve().parents[2]


def load_comparison():
    p = project_root() / "outputs" / "milestone_6" / "generalization_analysis" / "tables" / "kitti_vs_waymo_comparison.csv"
    df = pd.read_csv(p)
    df["display"] = df["detector"].map(DISPLAY_NAMES)
    return df


def load_class_degradation():
    p = project_root() / "outputs" / "milestone_6" / "generalization_analysis" / "tables" / "class_wise_degradation.csv"
    df = pd.read_csv(p)
    df["display"] = df["detector"].map(DISPLAY_NAMES)
    return df


def load_waymo_summary():
    p = project_root() / "outputs" / "milestone_6" / "waymo_external_validation" / "tables" / "waymo_external_summary.csv"
    df = pd.read_csv(p)
    df["display"] = df["detector"].map(DISPLAY_NAMES)
    return df


def add_value_labels(ax, rects, fmt="{:.3f}"):
    for r in rects:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2.0, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)


def fig_kitti_vs_waymo_map50_95():
    df = load_comparison().set_index("detector").loc[DETECTOR_ORDER].reset_index()
    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_k = ax.bar(x - width / 2, df["KITTI_mAP50_95"], width, label="KITTI validation", color="#8DA0CB")
    bars_w = ax.bar(x + width / 2, df["Waymo_mAP50_95"], width, label="Waymo external", color="#FC8D62")
    add_value_labels(ax, bars_k)
    add_value_labels(ax, bars_w)

    ax.set_ylabel("mAP@0.50:0.95")
    ax.set_title("KITTI vs Waymo mAP@0.50:0.95 (no retraining)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["display"])
    ax.set_ylim(0, max(df["KITTI_mAP50_95"].max(), 1.0) * 1.08)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = project_root() / "outputs" / "milestone_6" / "figures" / "kitti_vs_waymo_map50_95.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out.name)


def fig_generalization_ratio():
    df = load_comparison().set_index("detector").loc[DETECTOR_ORDER].reset_index()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, df["mAP50_95_generalization_ratio"], color=[COLORS[d] for d in df["detector"]])
    add_value_labels(ax, bars)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="No drop (ratio = 1.0)")
    ax.set_ylabel("Generalization Ratio (Waymo / KITTI mAP@0.50:0.95)")
    ax.set_title("Generalization Ratio (higher = more stable across domains)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["display"])
    ax.set_ylim(0, max(df["mAP50_95_generalization_ratio"].max(), 0.2) * 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = project_root() / "outputs" / "milestone_6" / "figures" / "generalization_ratio_map50_95.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out.name)


def fig_class_wise_degradation():
    df = load_class_degradation()
    classes = ["Vehicle", "Pedestrian", "Cyclist"]
    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, det in enumerate(DETECTOR_ORDER):
        sub = df[df["detector"] == det].set_index("class_name").loc[classes]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, sub["absolute_drop"], width,
                      label=DISPLAY_NAMES[det], color=COLORS[det])
        add_value_labels(ax, bars)

    ax.set_ylabel("Absolute drop in AP@0.50:0.95 (KITTI - Waymo)")
    ax.set_title("Class-wise AP@0.50:0.95 degradation by detector")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = project_root() / "outputs" / "milestone_6" / "figures" / "class_wise_degradation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out.name)


def fig_inference_time():
    df = load_waymo_summary().set_index("detector").loc[DETECTOR_ORDER].reset_index()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, df["mean_inference_ms"], color=[COLORS[d] for d in df["detector"]])
    add_value_labels(ax, bars, fmt="{:.1f}")
    ax.set_ylabel("Mean inference time (ms / image)")
    ax.set_title("Waymo external validation - mean inference time")
    ax.set_xticks(x)
    ax.set_xticklabels(df["display"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = project_root() / "outputs" / "milestone_6" / "figures" / "waymo_inference_time_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out.name)


def main():
    print("=" * 79)
    print("Milestone 6 - Phase 7: Generate generalization figures")
    print("=" * 79)
    fig_kitti_vs_waymo_map50_95()
    fig_generalization_ratio()
    fig_class_wise_degradation()
    fig_inference_time()


if __name__ == "__main__":
    main()
