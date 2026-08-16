from pathlib import Path
from datetime import datetime
import json
import zipfile
import shutil

import pandas as pd
import matplotlib.pyplot as plt


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

FIGURE_DIR = PROJECT / "outputs" / "milestone_7" / "figures"
BUNDLE_DIR = PROJECT / "outputs" / "milestone_7" / "report_bundle"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_MANIFEST_JSON = FIGURE_DIR / "milestone_7_figure_manifest.json"
FIGURE_MANIFEST_MD = FIGURE_DIR / "MILESTONE_7_FIGURES.md"

BUNDLE_ZIP = BUNDLE_DIR / "milestone_7_report_source_bundle.zip"
BUNDLE_MANIFEST_JSON = BUNDLE_DIR / "milestone_7_report_source_bundle_manifest.json"
BUNDLE_MANIFEST_MD = BUNDLE_DIR / "MILESTONE_7_REPORT_SOURCE_BUNDLE.md"

M5_KITTI_SUMMARY = PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "tables" / "comparison_summary_full.csv"
M6_WAYMO_SUMMARY = PROJECT / "outputs" / "milestone_6" / "waymo_external_validation" / "tables" / "waymo_external_summary.csv"
M6_GENERALIZATION_RATIO = PROJECT / "outputs" / "milestone_6" / "generalization_analysis" / "tables" / "generalization_ratio_table.csv"

OBJECT_SIZE_SUMMARY = PROJECT / "outputs" / "milestone_7" / "object_size_analysis" / "object_size_summary.csv"
OBJECT_SIZE_COMPARISON = PROJECT / "outputs" / "milestone_7" / "object_size_analysis" / "object_size_dataset_comparison.csv"

SAFETY_FN_SUMMARY = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "safety_false_negative_summary.csv"
FAILURE_TYPE_SUMMARY = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "failure_type_summary.csv"
DEPLOYMENT_SUMMARY = PROJECT / "outputs" / "milestone_7" / "deployment_tradeoff" / "deployment_suitability_summary.csv"

GALLERY_MANIFEST = PROJECT / "outputs" / "milestone_7" / "failure_cases" / "failure_case_manifest.json"
FAILURE_PANEL_WAYMO = PROJECT / "outputs" / "milestone_7" / "failure_cases" / "panels" / "failure_case_panel_waymo.png"
FAILURE_PANEL_KITTI = PROJECT / "outputs" / "milestone_7" / "failure_cases" / "panels" / "failure_case_panel_kitti.png"
FAILURE_PANEL_SAFETY = PROJECT / "outputs" / "milestone_7" / "failure_cases" / "panels" / "failure_case_panel_safety_vru.png"

DETECTOR_ORDER = {
    "yolo": 0,
    "rtdetr": 1,
    "retinanet": 2,
    "faster_rcnn": 3,
}

DETECTOR_LABELS = {
    "yolo": "YOLO",
    "rtdetr": "RT-DETR",
    "retinanet": "RetinaNet",
    "faster_rcnn": "Faster R-CNN",
}


def rel(path: Path):
    return str(path.relative_to(PROJECT))


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def detector_sort_key(detector):
    return DETECTOR_ORDER.get(str(detector), 99)


def detector_label(detector):
    return DETECTOR_LABELS.get(str(detector), str(detector))


def save_bar_chart(df, x_col, y_col, title, ylabel, output_path, sort_by_detector=True):
    data = df.copy()

    if sort_by_detector and "detector" in data.columns:
        data = data.sort_values("detector", key=lambda s: s.map(detector_sort_key))

    labels = [
        detector_label(v) if x_col == "detector" else str(v)
        for v in data[x_col].tolist()
    ]
    values = data[y_col].astype(float).tolist()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return output_path


def save_grouped_bar_chart(df, group_col, series_col, value_col, title, ylabel, output_path):
    data = df.copy()

    pivot = data.pivot_table(
        index=group_col,
        columns=series_col,
        values=value_col,
        aggfunc="first",
    )

    if group_col == "detector":
        pivot = pivot.loc[
            sorted(pivot.index, key=detector_sort_key)
        ]

    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return output_path


def save_scatter(df, output_path):
    data = df.copy()

    plt.figure(figsize=(8, 5))

    for _, row in data.iterrows():
        x = float(row["waymo_fps"])
        y = float(row["waymo_mAP50_95"])
        label = detector_label(row["detector"])
        plt.scatter(x, y, s=90)
        plt.text(x, y, f" {label}", va="center")

    plt.title("Waymo Accuracy-Speed Trade-off")
    plt.xlabel("Waymo FPS")
    plt.ylabel("Waymo mAP50-95")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return output_path


def save_failure_type_chart(failure_df, dataset, output_path):
    data = failure_df.copy()
    data = data[
        (data["dataset"] == dataset)
        & (data["failure_type"] != "true_positive")
    ]

    grouped = (
        data.groupby(["detector", "failure_type"], as_index=False)["count"]
        .sum()
    )

    pivot = grouped.pivot_table(
        index="detector",
        columns="failure_type",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    pivot = pivot.loc[
        sorted(pivot.index, key=detector_sort_key)
    ]

    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_title(f"{dataset.upper()} Failure-Type Burden")
    ax.set_ylabel("Failure event count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return output_path


def require_paths(paths):
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print("ERROR missing:", p)
        raise SystemExit(1)


def create_figures():
    print("Creating figures...")

    kitti_df = read_csv(M5_KITTI_SUMMARY)
    waymo_df = read_csv(M6_WAYMO_SUMMARY)
    gen_df = read_csv(M6_GENERALIZATION_RATIO)
    obj_df = read_csv(OBJECT_SIZE_SUMMARY)
    safety_df = read_csv(SAFETY_FN_SUMMARY)
    failure_df = read_csv(FAILURE_TYPE_SUMMARY)
    deploy_df = read_csv(DEPLOYMENT_SUMMARY)

    figures = []

    # 1. Deployment score.
    fig = save_bar_chart(
        deploy_df,
        "detector",
        "deployment_tradeoff_score",
        "Milestone 7 Safety-Weighted Deployment Trade-off Score",
        "Score",
        FIGURE_DIR / "deployment_tradeoff_score.png",
    )
    figures.append(("deployment_tradeoff_score", fig))

    # 2. Waymo VRU FNR all sizes.
    waymo_vru = safety_df[
        (safety_df["dataset"] == "waymo")
        & (safety_df["class_name"] == "Vulnerable_Road_Users")
        & (safety_df["object_size_bin"] == "all_sizes")
    ]

    fig = save_bar_chart(
        waymo_vru,
        "detector",
        "false_negative_rate",
        "Waymo Vulnerable-Road-User False Negative Rate",
        "False Negative Rate",
        FIGURE_DIR / "waymo_vru_false_negative_rate.png",
    )
    figures.append(("waymo_vru_false_negative_rate", fig))

    # 3. Waymo small VRU FNR.
    waymo_small_vru = safety_df[
        (safety_df["dataset"] == "waymo")
        & (safety_df["class_name"] == "Vulnerable_Road_Users")
        & (safety_df["object_size_bin"] == "small")
    ]

    fig = save_bar_chart(
        waymo_small_vru,
        "detector",
        "false_negative_rate",
        "Waymo Small Vulnerable-Road-User False Negative Rate",
        "False Negative Rate",
        FIGURE_DIR / "waymo_small_vru_false_negative_rate.png",
    )
    figures.append(("waymo_small_vru_false_negative_rate", fig))

    # 4. Waymo small-object recall by class.
    waymo_small_classes = obj_df[
        (obj_df["dataset"] == "waymo")
        & (obj_df["object_size_bin"] == "small")
        & (obj_df["class_name"].isin(["Vehicle", "Pedestrian", "Cyclist", "Vulnerable_Road_Users", "All_Classes"]))
    ]

    fig = save_grouped_bar_chart(
        waymo_small_classes,
        "detector",
        "class_name",
        "recall",
        "Waymo Small-Object Recall by Class",
        "Recall",
        FIGURE_DIR / "waymo_small_object_recall_by_class.png",
    )
    figures.append(("waymo_small_object_recall_by_class", fig))

    # 5. KITTI vs Waymo small all-class recall.
    small_all = obj_df[
        (obj_df["object_size_bin"] == "small")
        & (obj_df["class_name"] == "All_Classes")
    ]

    fig = save_grouped_bar_chart(
        small_all,
        "detector",
        "dataset",
        "recall",
        "Small-Object Recall: KITTI vs Waymo",
        "Recall",
        FIGURE_DIR / "small_object_recall_kitti_vs_waymo.png",
    )
    figures.append(("small_object_recall_kitti_vs_waymo", fig))

    # 6. Generalization ratio mAP50-95.
    gen_plot = gen_df.copy()

    if "generalization_ratio_mAP50_95" not in gen_plot.columns:
        # fallback for earlier naming
        possible = [c for c in gen_plot.columns if "ratio" in c.lower() and "50_95" in c.lower()]
        if possible:
            gen_plot = gen_plot.rename(columns={possible[0]: "generalization_ratio_mAP50_95"})

    fig = save_bar_chart(
        gen_plot,
        "detector",
        "generalization_ratio_mAP50_95",
        "KITTI-to-Waymo Generalization Ratio mAP50-95",
        "Waymo mAP50-95 / KITTI mAP50-95",
        FIGURE_DIR / "generalization_ratio_map50_95.png",
    )
    figures.append(("generalization_ratio_map50_95", fig))

    # 7. Accuracy-speed scatter.
    fig = save_scatter(
        deploy_df,
        FIGURE_DIR / "waymo_accuracy_speed_tradeoff.png",
    )
    figures.append(("waymo_accuracy_speed_tradeoff", fig))

    # 8. Waymo failure type burden.
    fig = save_failure_type_chart(
        failure_df,
        "waymo",
        FIGURE_DIR / "waymo_failure_type_burden.png",
    )
    figures.append(("waymo_failure_type_burden", fig))

    # 9. KITTI failure type burden.
    fig = save_failure_type_chart(
        failure_df,
        "kitti",
        FIGURE_DIR / "kitti_failure_type_burden.png",
    )
    figures.append(("kitti_failure_type_burden", fig))

    # 10. Waymo mAP50-95.
    fig = save_bar_chart(
        waymo_df,
        "detector",
        "mAP50_95",
        "Waymo External Validation mAP50-95",
        "mAP50-95",
        FIGURE_DIR / "waymo_external_map50_95.png",
    )
    figures.append(("waymo_external_map50_95", fig))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "figure_count": len(figures),
        "figures": [
            {
                "name": name,
                "path": rel(path),
            }
            for name, path in figures
        ],
        "notes": [
            "Figures summarize Milestone 7 object-size, safety, failure-case, and deployment trade-off analyses.",
            "Failure-case visual panels are stored under outputs/milestone_7/failure_cases/panels.",
        ],
    }

    FIGURE_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 7 Figures")
    md.append("")
    md.append(f"Created at: `{manifest['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append("## Figure Files")
    md.append("")
    for item in manifest["figures"]:
        md.append(f"- `{item['name']}`: `{item['path']}`")
    md.append("")
    md.append("## Failure-Case Panels")
    md.append("")
    for panel in [FAILURE_PANEL_WAYMO, FAILURE_PANEL_KITTI, FAILURE_PANEL_SAFETY]:
        if panel.exists():
            md.append(f"- `{rel(panel)}`")
    md.append("")

    FIGURE_MANIFEST_MD.write_text("\n".join(md), encoding="utf-8")

    print("Created figures:", len(figures))
    for name, path in figures:
        print("  ", name, "->", rel(path))

    print("Created:", FIGURE_MANIFEST_JSON)
    print("Created:", FIGURE_MANIFEST_MD)

    return manifest


def add_file_to_zip(zipf, path, arcname=None):
    if not path.exists():
        return False

    if arcname is None:
        arcname = rel(path)

    zipf.write(path, arcname)
    return True


def create_report_bundle(figure_manifest):
    print()
    print("Creating report source bundle...")

    bundle_files = []

    # Core configs.
    bundle_files.extend((PROJECT / "configs" / "analysis" / "milestone_7").glob("*"))

    # Milestone 5/6 source comparison tables.
    bundle_files.extend([
        M5_KITTI_SUMMARY,
        M6_WAYMO_SUMMARY,
        M6_GENERALIZATION_RATIO,
    ])

    # Milestone 7 data audit.
    bundle_files.extend((PROJECT / "outputs" / "milestone_7" / "data_audit").glob("*"))

    # Object-size analysis.
    bundle_files.extend((PROJECT / "outputs" / "milestone_7" / "object_size_analysis").glob("*"))

    # Safety/failure summaries only.
    safety_dir = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis"
    keep_safety_names = {
        "detection_error_index.json",
        "DETECTION_ERROR_INDEX.md",
        "detection_error_summary.csv",
        "detection_core_tp_fp_fn_summary.csv",
        "safety_false_negative_summary.csv",
        "safety_false_negative_summary.json",
        "top_safety_critical_images.csv",
        "top_safety_critical_images.json",
        "safety_dataset_comparison.csv",
        "safety_dataset_comparison.json",
        "safety_false_negative_analysis_summary.json",
        "MILESTONE_7_SAFETY_FALSE_NEGATIVE_ANALYSIS.md",
        "failure_type_summary.csv",
        "failure_type_summary.json",
        "false_positive_summary.csv",
        "localization_error_summary.csv",
        "class_confusion_summary.csv",
        "duplicate_detection_summary.csv",
        "top_failure_type_images.csv",
        "failure_case_candidate_rows.csv",
        "failure_type_analysis_summary.json",
        "MILESTONE_7_FAILURE_TYPE_ANALYSIS.md",
    }

    for path in safety_dir.glob("*"):
        if path.name in keep_safety_names:
            bundle_files.append(path)

    # Deployment trade-off.
    bundle_files.extend((PROJECT / "outputs" / "milestone_7" / "deployment_tradeoff").glob("*"))

    # Figures.
    bundle_files.extend(FIGURE_DIR.glob("*.png"))
    bundle_files.extend([
        FIGURE_MANIFEST_JSON,
        FIGURE_MANIFEST_MD,
    ])

    # Failure-case manifests and panels.
    failure_dir = PROJECT / "outputs" / "milestone_7" / "failure_cases"
    bundle_files.extend([
        failure_dir / "failure_case_manifest.json",
        failure_dir / "FAILURE_CASE_GALLERY.md",
        FAILURE_PANEL_WAYMO,
        FAILURE_PANEL_KITTI,
        FAILURE_PANEL_SAFETY,
    ])

    # Remove missing + duplicates + avoid raw huge full detection index CSV.
    clean_files = []
    seen = set()

    for path in bundle_files:
        if not path.exists() or not path.is_file():
            continue

        if path.name == "detection_error_index.csv":
            continue

        key = str(path.resolve()).lower()
        if key in seen:
            continue

        seen.add(key)
        clean_files.append(path)

    if BUNDLE_ZIP.exists():
        BUNDLE_ZIP.unlink()

    with zipfile.ZipFile(BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for path in clean_files:
            add_file_to_zip(zipf, path)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "bundle_zip": rel(BUNDLE_ZIP),
        "file_count": len(clean_files),
        "bundle_size_mb": round(BUNDLE_ZIP.stat().st_size / (1024 * 1024), 3),
        "excluded_large_files": [
            "outputs/milestone_7/safety_error_analysis/detection_error_index.csv"
        ],
        "included_files": [
            rel(path)
            for path in clean_files
        ],
        "instructions": [
            "Upload milestone_7_report_source_bundle.zip to ChatGPT to generate the injection-ready Milestone 7 DOCX report.",
            "The full detection_error_index.csv is excluded to keep the bundle small; summaries and manifests are included instead.",
        ],
    }

    BUNDLE_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 7 Report Source Bundle")
    md.append("")
    md.append(f"Created at: `{manifest['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append(f"- Bundle: `{manifest['bundle_zip']}`")
    md.append(f"- File count: `{manifest['file_count']}`")
    md.append(f"- Bundle size MB: `{manifest['bundle_size_mb']}`")
    md.append("")
    md.append("## Excluded Large Files")
    md.append("")
    for item in manifest["excluded_large_files"]:
        md.append(f"- `{item}`")
    md.append("")
    md.append("## Included Files")
    md.append("")
    for item in manifest["included_files"]:
        md.append(f"- `{item}`")
    md.append("")

    BUNDLE_MANIFEST_MD.write_text("\n".join(md), encoding="utf-8")

    print("Created:", BUNDLE_ZIP)
    print("Created:", BUNDLE_MANIFEST_JSON)
    print("Created:", BUNDLE_MANIFEST_MD)
    print("Bundle size MB:", manifest["bundle_size_mb"])
    print("Files included:", manifest["file_count"])

    return manifest


def main():
    print("=" * 100)
    print("STEP 9/10 - Create Milestone 7 figures and report source bundle")
    print("=" * 100)

    required = [
        M5_KITTI_SUMMARY,
        M6_WAYMO_SUMMARY,
        M6_GENERALIZATION_RATIO,
        OBJECT_SIZE_SUMMARY,
        OBJECT_SIZE_COMPARISON,
        SAFETY_FN_SUMMARY,
        FAILURE_TYPE_SUMMARY,
        DEPLOYMENT_SUMMARY,
        FAILURE_PANEL_WAYMO,
        FAILURE_PANEL_KITTI,
        FAILURE_PANEL_SAFETY,
    ]

    require_paths(required)

    figure_manifest = create_figures()
    bundle_manifest = create_report_bundle(figure_manifest)

    print()
    print("=" * 100)
    print("Milestone 7 figures and report bundle created")
    print("=" * 100)
    print("Figure count:", figure_manifest["figure_count"])
    print("Bundle:", BUNDLE_ZIP)
    print("Bundle size MB:", bundle_manifest["bundle_size_mb"])

    print()
    print("STEP 9/10 COMPLETE ✅")
    print("Figures and report source bundle are ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()