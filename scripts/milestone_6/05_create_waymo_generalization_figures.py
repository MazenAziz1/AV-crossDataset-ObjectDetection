from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

GENERALIZATION_DIR = PROJECT / "outputs" / "milestone_6" / "generalization_analysis"
TABLES_DIR = GENERALIZATION_DIR / "tables"

COMPARISON_CSV = TABLES_DIR / "kitti_vs_waymo_comparison.csv"
RATIO_CSV = TABLES_DIR / "generalization_ratio_table.csv"
CLASS_DEGRADATION_CSV = TABLES_DIR / "class_wise_degradation.csv"
LARGEST_CLASS_DROP_CSV = TABLES_DIR / "largest_class_degradation_by_detector.csv"

FIGURE_DIR = PROJECT / "outputs" / "milestone_6" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MANIFEST_JSON = FIGURE_DIR / "milestone_6_figure_manifest.json"
OUTPUT_MANIFEST_MD = FIGURE_DIR / "MILESTONE_6_FIGURES.md"


DISPLAY_NAMES = {
    "yolo": "YOLO",
    "rtdetr": "RT-DETR",
    "retinanet": "RetinaNet",
    "faster_rcnn": "Faster R-CNN",
}


def detector_label(detector):
    return DISPLAY_NAMES.get(detector, detector)


def save_current_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def grouped_bar_chart(
    df,
    detector_col,
    value_a_col,
    value_b_col,
    label_a,
    label_b,
    title,
    ylabel,
    output_path,
):
    labels = [detector_label(x) for x in df[detector_col].tolist()]
    x = np.arange(len(labels))
    width = 0.35

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.bar(x - width / 2, df[value_a_col].values, width, label=label_a)
    ax.bar(x + width / 2, df[value_b_col].values, width, label=label_b)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)


def single_bar_chart(
    df,
    detector_col,
    value_col,
    title,
    ylabel,
    output_path,
    annotate=True,
):
    labels = [detector_label(x) for x in df[detector_col].tolist()]
    x = np.arange(len(labels))
    values = df[value_col].values

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    bars = ax.bar(x, values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    if annotate:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    save_current_figure(output_path)


def class_wise_degradation_chart(class_df, output_path):
    ap_df = class_df[class_df["metric"] == "AP50_95"].copy()

    pivot = ap_df.pivot(
        index="detector",
        columns="class_name",
        values="absolute_drop",
    )

    preferred_order = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]
    pivot = pivot.reindex([d for d in preferred_order if d in pivot.index])

    labels = [detector_label(x) for x in pivot.index.tolist()]
    classes = pivot.columns.tolist()

    x = np.arange(len(labels))
    width = 0.22

    fig = plt.figure(figsize=(11, 6))
    ax = plt.gca()

    for idx, class_name in enumerate(classes):
        offsets = x + (idx - (len(classes) - 1) / 2) * width
        values = pivot[class_name].values
        ax.bar(offsets, values, width, label=class_name)

    ax.set_title("Class-wise AP50-95 Degradation from KITTI to Waymo")
    ax.set_ylabel("Absolute AP50-95 Drop")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(title="Class")
    ax.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)


def class_generalization_ratio_chart(class_df, output_path):
    ap_df = class_df[class_df["metric"] == "AP50_95"].copy()

    pivot = ap_df.pivot(
        index="detector",
        columns="class_name",
        values="generalization_ratio",
    )

    preferred_order = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]
    pivot = pivot.reindex([d for d in preferred_order if d in pivot.index])

    labels = [detector_label(x) for x in pivot.index.tolist()]
    classes = pivot.columns.tolist()

    x = np.arange(len(labels))
    width = 0.22

    fig = plt.figure(figsize=(11, 6))
    ax = plt.gca()

    for idx, class_name in enumerate(classes):
        offsets = x + (idx - (len(classes) - 1) / 2) * width
        values = pivot[class_name].values
        ax.bar(offsets, values, width, label=class_name)

    ax.set_title("Class-wise AP50-95 Generalization Ratio")
    ax.set_ylabel("Waymo AP50-95 / KITTI AP50-95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(title="Class")
    ax.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)


def main():
    print("=" * 100)
    print("STEP 6/10 - Create Milestone 6 figures")
    print("=" * 100)

    errors = []

    for path in [COMPARISON_CSV, RATIO_CSV, CLASS_DEGRADATION_CSV, LARGEST_CLASS_DROP_CSV]:
        if not path.exists():
            errors.append(f"Missing required input: {path}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        print("STEP 6/10 FAILED ❌")
        raise SystemExit(1)

    comparison_df = pd.read_csv(COMPARISON_CSV)
    ratio_df = pd.read_csv(RATIO_CSV)
    class_df = pd.read_csv(CLASS_DEGRADATION_CSV)
    largest_df = pd.read_csv(LARGEST_CLASS_DROP_CSV)

    # Keep the generalization ranking order for most plots.
    comparison_df = comparison_df.sort_values(
        by="rank_by_mAP50_95_generalization",
        ascending=True,
    )
    ratio_df = ratio_df.sort_values(
        by="rank_by_mAP50_95_generalization",
        ascending=True,
    )

    figures = {}

    path = FIGURE_DIR / "kitti_vs_waymo_map50_95.png"
    grouped_bar_chart(
        comparison_df,
        "detector",
        "KITTI_mAP50_95",
        "Waymo_mAP50_95",
        "KITTI",
        "Waymo",
        "KITTI vs Waymo mAP50-95",
        "mAP50-95",
        path,
    )
    figures["kitti_vs_waymo_map50_95"] = str(path.relative_to(PROJECT))
    print("Created:", path)

    path = FIGURE_DIR / "kitti_vs_waymo_map50.png"
    grouped_bar_chart(
        comparison_df,
        "detector",
        "KITTI_mAP50",
        "Waymo_mAP50",
        "KITTI",
        "Waymo",
        "KITTI vs Waymo mAP50",
        "mAP50",
        path,
    )
    figures["kitti_vs_waymo_map50"] = str(path.relative_to(PROJECT))
    print("Created:", path)

    path = FIGURE_DIR / "generalization_ratio_map50_95.png"
    single_bar_chart(
        ratio_df,
        "detector",
        "mAP50_95_generalization_ratio",
        "mAP50-95 Generalization Ratio",
        "Waymo mAP50-95 / KITTI mAP50-95",
        path,
    )
    figures["generalization_ratio_map50_95"] = str(path.relative_to(PROJECT))
    print("Created:", path)

    path = FIGURE_DIR / "generalization_ratio_map50.png"
    single_bar_chart(
        ratio_df,
        "detector",
        "mAP50_generalization_ratio",
        "mAP50 Generalization Ratio",
        "Waymo mAP50 / KITTI mAP50",
        path,
    )
    figures["generalization_ratio_map50"] = str(path.relative_to(PROJECT))
    print("Created:", path)

    path = FIGURE_DIR / "class_wise_degradation_ap50_95.png"
    class_wise_degradation_chart(class_df, path)
    figures["class_wise_degradation_ap50_95"] = str(path.relative_to(PROJECT))
    print("Created:", path)

    path = FIGURE_DIR / "class_wise_generalization_ratio_ap50_95.png"
    class_generalization_ratio_chart(class_df, path)
    figures["class_wise_generalization_ratio_ap50_95"] = str(path.relative_to(PROJECT))
    print("Created:", path)

    path = FIGURE_DIR / "waymo_inference_time_comparison.png"
    waymo_speed_df = comparison_df.sort_values(
        by="Waymo_mean_inference_ms",
        ascending=True,
    )
    single_bar_chart(
        waymo_speed_df,
        "detector",
        "Waymo_mean_inference_ms",
        "Waymo Mean Inference Time",
        "Mean Inference Time (ms/image)",
        path,
        annotate=True,
    )
    figures["waymo_inference_time_comparison"] = str(path.relative_to(PROJECT))
    print("Created:", path)

    path = FIGURE_DIR / "largest_class_degradation_by_detector.png"
    largest_plot_df = largest_df.copy()
    single_bar_chart(
        largest_plot_df,
        "detector",
        "largest_absolute_degradation",
        "Largest Class-Level AP50-95 Degradation by Detector",
        "Largest Absolute AP50-95 Drop",
        path,
        annotate=True,
    )
    figures["largest_class_degradation_by_detector"] = str(path.relative_to(PROJECT))
    print("Created:", path)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "input_tables": {
            "comparison_csv": str(COMPARISON_CSV.relative_to(PROJECT)),
            "ratio_csv": str(RATIO_CSV.relative_to(PROJECT)),
            "class_degradation_csv": str(CLASS_DEGRADATION_CSV.relative_to(PROJECT)),
            "largest_class_drop_csv": str(LARGEST_CLASS_DROP_CSV.relative_to(PROJECT)),
        },
        "figures": figures,
    }

    OUTPUT_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 6 Figures")
    md.append("")
    md.append(f"Created at: `{manifest['created_at']}`")
    md.append("")
    md.append("## Generated Figures")
    md.append("")
    for name, rel_path in figures.items():
        md.append(f"- `{name}`: `{rel_path}`")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append(
        "These figures visualize KITTI-to-Waymo domain-shift degradation, "
        "generalization ratios, class-wise performance drops, and Waymo inference-time behavior."
    )
    md.append("")

    OUTPUT_MANIFEST_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("Created:", OUTPUT_MANIFEST_JSON)
    print("Created:", OUTPUT_MANIFEST_MD)

    print()
    print("STEP 6/10 COMPLETE ✅")
    print("Milestone 6 figures are ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()