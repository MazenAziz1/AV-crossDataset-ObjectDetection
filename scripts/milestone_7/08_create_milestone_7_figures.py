import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common

DISP = common.DETECTOR_DISPLAY
COLORS = {"yolo": "#4C72B0", "rtdetr": "#DD8452", "retinanet": "#55A868", "faster_rcnn": "#C44E52"}
DETECTORS = common.DETECTORS
SIZE_LABELS = {"small": "Small", "medium": "Medium", "large": "Large"}


def out(name):
    return common.M7_OUT / "figures" / name


def fig_small_medium_large_recall():
    df = pd.read_csv(common.M7_OUT / "object_size_analysis" / "object_size_summary.csv")
    df = df[df["class_name"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, dataset in zip(axes, common.DATASETS):
        sub = df[df["dataset"] == dataset]
        x = np.arange(3)
        width = 0.2
        for i, d in enumerate(DETECTORS):
            r = sub[sub["detector"] == d].set_index("size_category").loc[["small", "medium", "large"]]
            ax.bar(x + (i - 1.5) * width, r["recall"], width, label=DISP[d], color=COLORS[d])
        ax.set_xticks(x); ax.set_xticklabels(["Small", "Medium", "Large"])
        ax.set_title(common.DATASET_DISPLAY[dataset]); ax.set_ylim(0, 1.0)
        ax.set_ylabel("Recall" if dataset == "kitti" else "")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Recall by object size (operating point, conf>=0.25)")
    fig.tight_layout()
    fig.savefig(out("small_medium_large_recall.png"), dpi=150); plt.close(fig)


def fig_small_object_failure_rate():
    df = pd.read_csv(common.M7_OUT / "object_size_analysis" / "small_object_failure_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DETECTORS)); width = 0.38
    for offset, dataset in zip([-width / 2, width / 2], common.DATASETS):
        sub = df[df["dataset"] == dataset].set_index("detector").loc[DETECTORS]
        ax.bar(x + offset, sub["miss_rate"], width, label=common.DATASET_DISPLAY[dataset])
    ax.set_xticks(x); ax.set_xticklabels([DISP[d] for d in DETECTORS])
    ax.set_ylabel("Small-object miss rate"); ax.set_ylim(0, 1.0)
    ax.set_title("Small-object failure rate (miss rate, class-agnostic)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("small_object_failure_rate.png"), dpi=150); plt.close(fig)


def fig_pedestrian_cyclist_fn_rate():
    df = pd.read_csv(common.M7_OUT / "safety_error_analysis" / "safety_false_negative_summary.csv")
    df = df[df["class_name"] == "pedestrian+cyclist"]
    agg = df.groupby(["dataset", "detector"]).agg({"tp": "sum", "fn": "sum"}).reset_index()
    agg["fn_rate"] = agg["fn"] / (agg["tp"] + agg["fn"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DETECTORS)); width = 0.38
    for offset, dataset in zip([-width / 2, width / 2], common.DATASETS):
        sub = agg[agg["dataset"] == dataset].set_index("detector").loc[DETECTORS]
        ax.bar(x + offset, sub["fn_rate"], width, label=common.DATASET_DISPLAY[dataset])
    ax.set_xticks(x); ax.set_xticklabels([DISP[d] for d in DETECTORS])
    ax.set_ylabel("Pedestrian+cyclist FN rate"); ax.set_ylim(0, 1.0)
    ax.set_title("Safety-critical false-negative rate")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("pedestrian_cyclist_false_negative_rate.png"), dpi=150); plt.close(fig)


def fig_failure_type_breakdown():
    df = pd.read_csv(common.M7_OUT / "safety_error_analysis" / "failure_type_summary.csv")
    err = ["false_negative", "false_positive", "localization_error", "class_confusion", "over_detection"]
    labels = ["Miss (FN)", "False pos.", "Localization", "Confusion", "Over-detect."]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, dataset in zip(axes, common.DATASETS):
        sub = df[df["dataset"] == dataset].set_index("detector").loc[DETECTORS]
        bottom = np.zeros(len(DETECTORS))
        for e, lbl in zip(err, labels):
            vals = sub[e].values
            ax.bar(np.arange(len(DETECTORS)), vals, bottom=bottom, label=lbl)
            bottom += vals
        ax.set_xticks(np.arange(len(DETECTORS))); ax.set_xticklabels([DISP[d] for d in DETECTORS])
        ax.set_title(common.DATASET_DISPLAY[dataset])
        ax.set_ylabel("Count" if dataset == "kitti" else "")
    axes[0].legend(fontsize=8)
    fig.suptitle("Failure-type breakdown by detector")
    fig.tight_layout()
    fig.savefig(out("failure_type_breakdown.png"), dpi=150); plt.close(fig)


def fig_class_confusion_heatmap():
    df = pd.read_csv(common.M7_OUT / "safety_error_analysis" / "class_confusion_summary.csv")
    names = ["Vehicle", "Pedestrian", "Cyclist"]
    mat = np.zeros((3, 3))
    for _, r in df.iterrows():
        if r["class_name"] in names and r["gt_class_name"] in names:
            mat[names.index(r["class_name"]), names.index(r["gt_class_name"])] += r["count"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Reds")
    ax.set_xticks(range(3)); ax.set_xticklabels(names)
    ax.set_yticks(range(3)); ax.set_yticklabels(names)
    ax.set_xlabel("Ground-truth class"); ax.set_ylabel("Predicted class")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_title("Class confusion (predicted vs ground truth)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out("class_confusion_heatmap.png"), dpi=150); plt.close(fig)


def fig_deployment_suitability():
    df = pd.read_csv(common.M7_OUT / "deployment_tradeoff" / "deployment_suitability_table.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in df.iterrows():
        ax.scatter(r["waymo_mean_inference_ms"], r["Waymo_mAP50_95"],
                   s=120, color=COLORS[r["detector"]], label=DISP[r["detector"]])
    ax.set_xlabel("Mean inference time (ms / image, Waymo)")
    ax.set_ylabel("Waymo mAP@0.50:0.95")
    ax.set_title("Deployment trade-off: speed vs external accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("deployment_suitability_comparison.png"), dpi=150); plt.close(fig)


def main():
    print("=" * 79)
    print("Milestone 7 - Step 8: Figures")
    print("=" * 79)
    fig_small_medium_large_recall()
    fig_small_object_failure_rate()
    fig_pedestrian_cyclist_fn_rate()
    fig_failure_type_breakdown()
    fig_class_confusion_heatmap()
    fig_deployment_suitability()

    md = ["# Milestone 7 Figures", "",
          "| Figure | Source table |", "|---|---|"]
    for f, src in [
        ("small_medium_large_recall.png", "object_size_summary.csv"),
        ("small_object_failure_rate.png", "small_object_failure_summary.csv"),
        ("pedestrian_cyclist_false_negative_rate.png", "safety_false_negative_summary.csv"),
        ("failure_type_breakdown.png", "failure_type_summary.csv"),
        ("class_confusion_heatmap.png", "class_confusion_summary.csv"),
        ("deployment_suitability_comparison.png", "deployment_suitability_table.csv"),
    ]:
        md.append(f"| {f} | {src} |")
    md.append("")
    (common.M7_OUT / "figures" / "MILESTONE_7_FIGURES.md").write_text("\n".join(md), encoding="utf-8")
    print("Saved 6 figures + MILESTONE_7_FIGURES.md")


if __name__ == "__main__":
    main()
