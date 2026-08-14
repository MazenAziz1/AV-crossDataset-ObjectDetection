from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

KITTI_CSV = PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "tables" / "comparison_summary_full.csv"
WAYMO_CSV = PROJECT / "outputs" / "milestone_6" / "waymo_external_validation" / "tables" / "waymo_external_summary.csv"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_6" / "generalization_analysis"
TABLES_DIR = OUTPUT_DIR / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]


def safe_ratio(numerator, denominator):
    if denominator is None:
        return None
    if pd.isna(denominator) or float(denominator) == 0.0:
        return None
    if pd.isna(numerator):
        return None
    return float(numerator) / float(denominator)


def safe_drop(kitti_value, waymo_value):
    if pd.isna(kitti_value) or pd.isna(waymo_value):
        return None
    return float(kitti_value) - float(waymo_value)


def safe_drop_percent(kitti_value, waymo_value):
    drop = safe_drop(kitti_value, waymo_value)
    if drop is None:
        return None
    if float(kitti_value) == 0.0:
        return None
    return 100.0 * drop / float(kitti_value)


def round_float(value, digits=6):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return round(float(value), digits)


def main():
    print("=" * 100)
    print("STEP 5/10 - KITTI vs Waymo generalization analysis")
    print("=" * 100)

    errors = []

    if not KITTI_CSV.exists():
        errors.append(f"Missing KITTI baseline CSV: {KITTI_CSV}")

    if not WAYMO_CSV.exists():
        errors.append(f"Missing Waymo external CSV: {WAYMO_CSV}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        print("STEP 5/10 FAILED ❌")
        raise SystemExit(1)

    kitti_df = pd.read_csv(KITTI_CSV)
    waymo_df = pd.read_csv(WAYMO_CSV)

    required_detectors = {"yolo", "rtdetr", "retinanet", "faster_rcnn"}

    kitti_detectors = set(kitti_df["detector"].tolist())
    waymo_detectors = set(waymo_df["detector"].tolist())

    if kitti_detectors != required_detectors:
        errors.append(f"KITTI detector set mismatch: {kitti_detectors}")

    if waymo_detectors != required_detectors:
        errors.append(f"Waymo detector set mismatch: {waymo_detectors}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        print("STEP 5/10 FAILED ❌")
        raise SystemExit(1)

    merged = kitti_df.merge(
        waymo_df,
        on="detector",
        suffixes=("_KITTI", "_Waymo"),
    )

    aggregate_rows = []

    for _, row in merged.iterrows():
        detector = row["detector"]

        kitti_map50 = row["mAP50_KITTI"]
        waymo_map50 = row["mAP50_Waymo"]

        kitti_map5095 = row["mAP50_95_KITTI"]
        waymo_map5095 = row["mAP50_95_Waymo"]

        aggregate_rows.append({
            "detector": detector,

            "KITTI_num_images": int(row["num_images_KITTI"]),
            "Waymo_num_images": int(row["num_images_Waymo"]),

            "KITTI_mAP50": kitti_map50,
            "Waymo_mAP50": waymo_map50,
            "mAP50_absolute_drop": safe_drop(kitti_map50, waymo_map50),
            "mAP50_drop_percent": safe_drop_percent(kitti_map50, waymo_map50),
            "mAP50_generalization_ratio": safe_ratio(waymo_map50, kitti_map50),

            "KITTI_mAP50_95": kitti_map5095,
            "Waymo_mAP50_95": waymo_map5095,
            "mAP50_95_absolute_drop": safe_drop(kitti_map5095, waymo_map5095),
            "mAP50_95_drop_percent": safe_drop_percent(kitti_map5095, waymo_map5095),
            "mAP50_95_generalization_ratio": safe_ratio(waymo_map5095, kitti_map5095),

            "KITTI_mean_inference_ms": row["mean_inference_ms_KITTI"],
            "Waymo_mean_inference_ms": row["mean_inference_ms_Waymo"],
            "inference_ms_difference_Waymo_minus_KITTI": safe_drop(
                row["mean_inference_ms_Waymo"],
                row["mean_inference_ms_KITTI"],
            ),
            "Waymo_to_KITTI_inference_ratio": safe_ratio(
                row["mean_inference_ms_Waymo"],
                row["mean_inference_ms_KITTI"],
            ),
        })

    aggregate_df = pd.DataFrame(aggregate_rows)

    # Rank detectors by the primary generalization metric.
    aggregate_df = aggregate_df.sort_values(
        by=["mAP50_95_generalization_ratio", "Waymo_mAP50_95"],
        ascending=[False, False],
    ).reset_index(drop=True)

    aggregate_df.insert(1, "rank_by_mAP50_95_generalization", aggregate_df.index + 1)

    class_rows = []

    for _, row in merged.iterrows():
        detector = row["detector"]

        for class_name in CLASSES:
            for metric in ["AP50", "AP50_95"]:
                kitti_col = f"{class_name}_{metric}_KITTI"
                waymo_col = f"{class_name}_{metric}_Waymo"

                if kitti_col not in merged.columns or waymo_col not in merged.columns:
                    continue

                kitti_value = row[kitti_col]
                waymo_value = row[waymo_col]

                class_rows.append({
                    "detector": detector,
                    "class_name": class_name,
                    "metric": metric,
                    "KITTI_value": kitti_value,
                    "Waymo_value": waymo_value,
                    "absolute_drop": safe_drop(kitti_value, waymo_value),
                    "drop_percent": safe_drop_percent(kitti_value, waymo_value),
                    "generalization_ratio": safe_ratio(waymo_value, kitti_value),
                })

    class_df = pd.DataFrame(class_rows)

    # AP50-95 class-wise degradation summary.
    class_ap5095 = class_df[class_df["metric"] == "AP50_95"].copy()
    class_ap5095 = class_ap5095.sort_values(
        by=["detector", "absolute_drop"],
        ascending=[True, False],
    )

    largest_degradation_rows = []

    for detector in sorted(required_detectors):
        detector_rows = class_ap5095[class_ap5095["detector"] == detector].copy()

        if detector_rows.empty:
            continue

        largest = detector_rows.sort_values(
            by="absolute_drop",
            ascending=False,
        ).iloc[0]

        smallest_ratio = detector_rows.sort_values(
            by="generalization_ratio",
            ascending=True,
        ).iloc[0]

        largest_degradation_rows.append({
            "detector": detector,
            "largest_absolute_degradation_class": largest["class_name"],
            "largest_absolute_degradation": largest["absolute_drop"],
            "largest_absolute_degradation_percent": largest["drop_percent"],
            "lowest_generalization_ratio_class": smallest_ratio["class_name"],
            "lowest_class_generalization_ratio": smallest_ratio["generalization_ratio"],
        })

    largest_degradation_df = pd.DataFrame(largest_degradation_rows)

    ratio_table = aggregate_df[[
        "rank_by_mAP50_95_generalization",
        "detector",
        "KITTI_mAP50_95",
        "Waymo_mAP50_95",
        "mAP50_95_absolute_drop",
        "mAP50_95_drop_percent",
        "mAP50_95_generalization_ratio",
        "KITTI_mAP50",
        "Waymo_mAP50",
        "mAP50_absolute_drop",
        "mAP50_drop_percent",
        "mAP50_generalization_ratio",
    ]].copy()

    # Create rounded display versions for markdown readability.
    aggregate_display = aggregate_df.copy()
    class_display = class_df.copy()
    largest_display = largest_degradation_df.copy()
    ratio_display = ratio_table.copy()

    for df in [aggregate_display, class_display, largest_display, ratio_display]:
        for col in df.columns:
            if df[col].dtype.kind in "fc":
                df[col] = df[col].apply(lambda x: round_float(x, 6))

    aggregate_csv = TABLES_DIR / "kitti_vs_waymo_comparison.csv"
    aggregate_json = TABLES_DIR / "kitti_vs_waymo_comparison.json"

    ratio_csv = TABLES_DIR / "generalization_ratio_table.csv"
    ratio_json = TABLES_DIR / "generalization_ratio_table.json"

    class_csv = TABLES_DIR / "class_wise_degradation.csv"
    class_json = TABLES_DIR / "class_wise_degradation.json"

    largest_csv = TABLES_DIR / "largest_class_degradation_by_detector.csv"
    largest_json = TABLES_DIR / "largest_class_degradation_by_detector.json"

    summary_json = OUTPUT_DIR / "generalization_analysis_summary.json"
    summary_md = OUTPUT_DIR / "MILESTONE_6_GENERALIZATION_ANALYSIS.md"

    aggregate_display.to_csv(aggregate_csv, index=False)
    aggregate_json.write_text(aggregate_display.to_json(orient="records", indent=2), encoding="utf-8")

    ratio_display.to_csv(ratio_csv, index=False)
    ratio_json.write_text(ratio_display.to_json(orient="records", indent=2), encoding="utf-8")

    class_display.to_csv(class_csv, index=False)
    class_json.write_text(class_display.to_json(orient="records", indent=2), encoding="utf-8")

    largest_display.to_csv(largest_csv, index=False)
    largest_json.write_text(largest_display.to_json(orient="records", indent=2), encoding="utf-8")

    best_generalizer = aggregate_display.iloc[0].to_dict()
    worst_generalizer = aggregate_display.iloc[-1].to_dict()

    best_waymo_accuracy = aggregate_display.sort_values(
        by="Waymo_mAP50_95",
        ascending=False,
    ).iloc[0].to_dict()

    fastest_waymo = aggregate_display.sort_values(
        by="Waymo_mean_inference_ms",
        ascending=True,
    ).iloc[0].to_dict()

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "inputs": {
            "kitti_csv": str(KITTI_CSV.relative_to(PROJECT)),
            "waymo_csv": str(WAYMO_CSV.relative_to(PROJECT)),
        },
        "primary_metric": "mAP50_95_generalization_ratio",
        "formula": "Generalization Ratio = Waymo metric / KITTI metric",
        "best_generalizer_by_mAP50_95_ratio": best_generalizer,
        "worst_generalizer_by_mAP50_95_ratio": worst_generalizer,
        "best_waymo_detector_by_mAP50_95": best_waymo_accuracy,
        "fastest_waymo_detector": fastest_waymo,
        "outputs": {
            "kitti_vs_waymo_comparison_csv": str(aggregate_csv.relative_to(PROJECT)),
            "generalization_ratio_table_csv": str(ratio_csv.relative_to(PROJECT)),
            "class_wise_degradation_csv": str(class_csv.relative_to(PROJECT)),
            "largest_class_degradation_csv": str(largest_csv.relative_to(PROJECT)),
            "summary_md": str(summary_md.relative_to(PROJECT)),
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 6 - KITTI vs Waymo Generalization Analysis")
    md.append("")
    md.append(f"Created at: `{summary['created_at']}`")
    md.append("")
    md.append("## Objective")
    md.append("")
    md.append(
        "This analysis compares Milestone 5 KITTI in-domain validation results "
        "against Milestone 6 Waymo external-validation results using the same locked "
        "KITTI-trained checkpoints. No retraining or fine-tuning is used."
    )
    md.append("")
    md.append("## Formula")
    md.append("")
    md.append("`Generalization Ratio = Waymo metric / KITTI metric`")
    md.append("")
    md.append("A ratio closer to `1.0` indicates better cross-dataset stability.")
    md.append("")
    md.append("## Aggregate KITTI vs Waymo Comparison")
    md.append("")
    md.append(aggregate_display.to_markdown(index=False))
    md.append("")
    md.append("## Generalization Ratio Table")
    md.append("")
    md.append(ratio_display.to_markdown(index=False))
    md.append("")
    md.append("## Largest Class-Level Degradation by Detector")
    md.append("")
    md.append(largest_display.to_markdown(index=False))
    md.append("")
    md.append("## Key Findings")
    md.append("")
    md.append(
        f"- Best generalizer by mAP50-95 ratio: `{best_generalizer['detector']}` "
        f"with ratio `{best_generalizer['mAP50_95_generalization_ratio']}`."
    )
    md.append(
        f"- Best Waymo detector by mAP50-95: `{best_waymo_accuracy['detector']}` "
        f"with Waymo mAP50-95 `{best_waymo_accuracy['Waymo_mAP50_95']}`."
    )
    md.append(
        f"- Fastest Waymo detector: `{fastest_waymo['detector']}` "
        f"with mean inference `{fastest_waymo['Waymo_mean_inference_ms']}` ms."
    )
    md.append("")
    md.append("## Interpretation Draft")
    md.append("")
    md.append(
        "All detectors show substantial degradation when transferred from KITTI to the "
        "Waymo external subset. This confirms a significant domain shift between the "
        "in-domain KITTI validation split and the external Waymo subset. The observed "
        "drop should be discussed as the central cross-dataset generalization result "
        "of Milestone 6 rather than as a training failure, because the evaluation uses "
        "locked KITTI-trained checkpoints without retraining."
    )
    md.append("")

    summary_md.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Aggregate KITTI vs Waymo comparison")
    print("=" * 100)
    print(aggregate_display.to_string(index=False))

    print()
    print("=" * 100)
    print("Generalization ratio table")
    print("=" * 100)
    print(ratio_display.to_string(index=False))

    print()
    print("=" * 100)
    print("Largest class-level degradation")
    print("=" * 100)
    print(largest_display.to_string(index=False))

    print()
    print("Created:", aggregate_csv)
    print("Created:", ratio_csv)
    print("Created:", class_csv)
    print("Created:", largest_csv)
    print("Created:", summary_json)
    print("Created:", summary_md)

    print()
    print("STEP 5/10 COMPLETE ✅")
    print("KITTI vs Waymo generalization analysis is complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()