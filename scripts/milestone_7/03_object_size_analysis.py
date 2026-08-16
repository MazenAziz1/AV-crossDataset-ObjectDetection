from pathlib import Path
from datetime import datetime
import csv
import json
from collections import defaultdict


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

INPUT_CSV = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_error_index.csv"
INPUT_MANIFEST_JSON = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_error_index.json"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_7" / "object_size_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OBJECT_SIZE_SUMMARY_CSV = OUTPUT_DIR / "object_size_summary.csv"
OBJECT_SIZE_SUMMARY_JSON = OUTPUT_DIR / "object_size_summary.json"

SMALL_OBJECT_FAILURE_CSV = OUTPUT_DIR / "small_object_failure_summary.csv"
SMALL_OBJECT_FAILURE_JSON = OUTPUT_DIR / "small_object_failure_summary.json"

DATASET_COMPARISON_CSV = OUTPUT_DIR / "object_size_dataset_comparison.csv"
DATASET_COMPARISON_JSON = OUTPUT_DIR / "object_size_dataset_comparison.json"

SUMMARY_JSON = OUTPUT_DIR / "object_size_analysis_summary.json"
SUMMARY_MD = OUTPUT_DIR / "MILESTONE_7_OBJECT_SIZE_ANALYSIS.md"


SIZE_ORDER = {
    "small": 0,
    "medium": 1,
    "large": 2,
    "all_sizes": 3,
}

CLASS_ORDER = {
    "Vehicle": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Vulnerable_Road_Users": 3,
    "All_Classes": 4,
}

DETECTOR_ORDER = {
    "yolo": 0,
    "rtdetr": 1,
    "retinanet": 2,
    "faster_rcnn": 3,
}

SAFETY_CLASSES = {"Pedestrian", "Cyclist"}


def safe_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def round_or_blank(value, digits=6):
    if value is None:
        return ""
    return round(float(value), digits)


def init_counter():
    return {
        "tp": 0,
        "fn": 0,
        "gt_objects": 0,
        "normalized_area_sum": 0.0,
        "normalized_area_count": 0,
    }


def add_event(counter, failure_type, normalized_area):
    if failure_type == "true_positive":
        counter["tp"] += 1
        counter["gt_objects"] += 1
    elif failure_type == "false_negative":
        counter["fn"] += 1
        counter["gt_objects"] += 1

    if normalized_area is not None:
        counter["normalized_area_sum"] += normalized_area
        counter["normalized_area_count"] += 1


def make_summary_row(dataset, detector, class_name, object_size_bin, counter):
    tp = counter["tp"]
    fn = counter["fn"]
    gt_objects = counter["gt_objects"]

    recall = safe_div(tp, gt_objects)
    false_negative_rate = safe_div(fn, gt_objects)

    mean_normalized_area = None
    if counter["normalized_area_count"] > 0:
        mean_normalized_area = counter["normalized_area_sum"] / counter["normalized_area_count"]

    return {
        "dataset": dataset,
        "detector": detector,
        "class_name": class_name,
        "object_size_bin": object_size_bin,
        "tp": tp,
        "fn": fn,
        "gt_objects": gt_objects,
        "recall": round_or_blank(recall),
        "false_negative_rate": round_or_blank(false_negative_rate),
        "mean_normalized_area": round_or_blank(mean_normalized_area, 8),
    }


def sort_summary_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            CLASS_ORDER.get(r["class_name"], 99),
            SIZE_ORDER.get(r["object_size_bin"], 99),
        ),
    )


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest():
    if not INPUT_MANIFEST_JSON.exists():
        return {}
    return json.loads(INPUT_MANIFEST_JSON.read_text(encoding="utf-8"))


def main():
    print("=" * 100)
    print("STEP 4/10 - Object-size robustness analysis")
    print("=" * 100)

    if not INPUT_CSV.exists():
        print("ERROR: Missing input:", INPUT_CSV)
        print("STEP 4/10 FAILED ❌")
        raise SystemExit(1)

    manifest = load_manifest()

    counters = defaultdict(init_counter)

    rows_read = 0
    gt_events_used = 0
    ignored_rows = 0

    print("Reading detection error index:")
    print(INPUT_CSV)

    with INPUT_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows_read += 1

            failure_type = row.get("failure_type")

            # Object-size analysis should use GT-centered events only.
            # Avoid counting localization_error and class_confusion as extra GT objects.
            if failure_type not in {"true_positive", "false_negative"}:
                ignored_rows += 1
                continue

            dataset = row.get("dataset", "")
            detector = row.get("detector", "")
            class_name = row.get("analysis_class_name", "")
            object_size_bin = row.get("object_size_bin", "")
            normalized_area = safe_float(row.get("object_size_score"))

            if not dataset or not detector or not class_name or not object_size_bin:
                ignored_rows += 1
                continue

            gt_events_used += 1

            # Per class and size.
            add_event(
                counters[(dataset, detector, class_name, object_size_bin)],
                failure_type,
                normalized_area,
            )

            # Per class across all sizes.
            add_event(
                counters[(dataset, detector, class_name, "all_sizes")],
                failure_type,
                normalized_area,
            )

            # All classes by size.
            add_event(
                counters[(dataset, detector, "All_Classes", object_size_bin)],
                failure_type,
                normalized_area,
            )

            # All classes across all sizes.
            add_event(
                counters[(dataset, detector, "All_Classes", "all_sizes")],
                failure_type,
                normalized_area,
            )

            # Vulnerable road users by size.
            if class_name in SAFETY_CLASSES:
                add_event(
                    counters[(dataset, detector, "Vulnerable_Road_Users", object_size_bin)],
                    failure_type,
                    normalized_area,
                )

                add_event(
                    counters[(dataset, detector, "Vulnerable_Road_Users", "all_sizes")],
                    failure_type,
                    normalized_area,
                )

            if rows_read % 250000 == 0:
                print(f"  read {rows_read:,} rows")

    summary_rows = []

    for key, counter in counters.items():
        dataset, detector, class_name, object_size_bin = key

        if counter["gt_objects"] == 0:
            continue

        summary_rows.append(
            make_summary_row(
                dataset,
                detector,
                class_name,
                object_size_bin,
                counter,
            )
        )

    summary_rows = sort_summary_rows(summary_rows)

    summary_fieldnames = [
        "dataset",
        "detector",
        "class_name",
        "object_size_bin",
        "tp",
        "fn",
        "gt_objects",
        "recall",
        "false_negative_rate",
        "mean_normalized_area",
    ]

    write_csv(OBJECT_SIZE_SUMMARY_CSV, summary_rows, summary_fieldnames)
    write_json(OBJECT_SIZE_SUMMARY_JSON, summary_rows)

    small_rows = [
        r for r in summary_rows
        if r["object_size_bin"] == "small"
    ]

    small_rows = sorted(
        small_rows,
        key=lambda r: (
            r["dataset"],
            CLASS_ORDER.get(r["class_name"], 99),
            -float(r["false_negative_rate"]) if r["false_negative_rate"] != "" else 0,
        ),
    )

    write_csv(SMALL_OBJECT_FAILURE_CSV, small_rows, summary_fieldnames)
    write_json(SMALL_OBJECT_FAILURE_JSON, small_rows)

    # KITTI vs Waymo comparison for object-size robustness.
    lookup = {
        (
            r["detector"],
            r["class_name"],
            r["object_size_bin"],
            r["dataset"],
        ): r
        for r in summary_rows
    }

    comparison_rows = []

    comparison_keys = sorted(
        set(
            (r["detector"], r["class_name"], r["object_size_bin"])
            for r in summary_rows
        ),
        key=lambda k: (
            DETECTOR_ORDER.get(k[0], 99),
            CLASS_ORDER.get(k[1], 99),
            SIZE_ORDER.get(k[2], 99),
        ),
    )

    for detector, class_name, object_size_bin in comparison_keys:
        kitti = lookup.get((detector, class_name, object_size_bin, "kitti"))
        waymo = lookup.get((detector, class_name, object_size_bin, "waymo"))

        if not kitti or not waymo:
            continue

        kitti_recall = safe_float(kitti["recall"])
        waymo_recall = safe_float(waymo["recall"])
        kitti_fnr = safe_float(kitti["false_negative_rate"])
        waymo_fnr = safe_float(waymo["false_negative_rate"])

        recall_drop = None
        if kitti_recall is not None and waymo_recall is not None:
            recall_drop = kitti_recall - waymo_recall

        fnr_increase = None
        if kitti_fnr is not None and waymo_fnr is not None:
            fnr_increase = waymo_fnr - kitti_fnr

        comparison_rows.append({
            "detector": detector,
            "class_name": class_name,
            "object_size_bin": object_size_bin,

            "KITTI_gt_objects": kitti["gt_objects"],
            "Waymo_gt_objects": waymo["gt_objects"],

            "KITTI_recall": kitti["recall"],
            "Waymo_recall": waymo["recall"],
            "recall_drop_KITTI_minus_Waymo": round_or_blank(recall_drop),

            "KITTI_false_negative_rate": kitti["false_negative_rate"],
            "Waymo_false_negative_rate": waymo["false_negative_rate"],
            "fnr_increase_Waymo_minus_KITTI": round_or_blank(fnr_increase),
        })

    comparison_fieldnames = [
        "detector",
        "class_name",
        "object_size_bin",
        "KITTI_gt_objects",
        "Waymo_gt_objects",
        "KITTI_recall",
        "Waymo_recall",
        "recall_drop_KITTI_minus_Waymo",
        "KITTI_false_negative_rate",
        "Waymo_false_negative_rate",
        "fnr_increase_Waymo_minus_KITTI",
    ]

    write_csv(DATASET_COMPARISON_CSV, comparison_rows, comparison_fieldnames)
    write_json(DATASET_COMPARISON_JSON, comparison_rows)

    # Key findings.
    vulnerable_small_waymo = [
        r for r in summary_rows
        if r["dataset"] == "waymo"
        and r["object_size_bin"] == "small"
        and r["class_name"] in {"Pedestrian", "Cyclist", "Vulnerable_Road_Users"}
    ]

    vulnerable_small_waymo = sorted(
        vulnerable_small_waymo,
        key=lambda r: (
            -float(r["false_negative_rate"]) if r["false_negative_rate"] != "" else 0,
            DETECTOR_ORDER.get(r["detector"], 99),
        ),
    )

    all_classes_small = [
        r for r in summary_rows
        if r["class_name"] == "All_Classes"
        and r["object_size_bin"] == "small"
    ]

    best_small_by_dataset = {}

    for dataset in ["kitti", "waymo"]:
        rows = [
            r for r in all_classes_small
            if r["dataset"] == dataset
        ]

        rows = sorted(
            rows,
            key=lambda r: (
                -float(r["recall"]) if r["recall"] != "" else -1,
                DETECTOR_ORDER.get(r["detector"], 99),
            ),
        )

        best_small_by_dataset[dataset] = rows[0] if rows else None

    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "input": str(INPUT_CSV.relative_to(PROJECT)),
        "rows_read": rows_read,
        "gt_events_used": gt_events_used,
        "ignored_rows": ignored_rows,
        "object_size_policy": {
            "method": "target_box_normalized_area_quantiles",
            "normalized_area_definition": "bbox_area / image_area",
            "small": "normalized_area <= 0.004157071390230814",
            "medium": "0.004157071390230814 < normalized_area <= 0.015589650670960799",
            "large": "normalized_area > 0.015589650670960799",
            "target_boxes_used": 39086,
        },
        "dataset_object_size_distribution_from_error_index_manifest": manifest.get("dataset_summary", {}),
        "best_small_object_recall_by_dataset_all_classes": best_small_by_dataset,
        "worst_waymo_small_vulnerable_rows": vulnerable_small_waymo[:10],
        "outputs": {
            "object_size_summary_csv": str(OBJECT_SIZE_SUMMARY_CSV.relative_to(PROJECT)),
            "small_object_failure_csv": str(SMALL_OBJECT_FAILURE_CSV.relative_to(PROJECT)),
            "object_size_dataset_comparison_csv": str(DATASET_COMPARISON_CSV.relative_to(PROJECT)),
            "summary_md": str(SUMMARY_MD.relative_to(PROJECT)),
        },
    }

    write_json(SUMMARY_JSON, summary_payload)

    md = []
    md.append("# Milestone 7 Object-Size Robustness Analysis")
    md.append("")
    md.append(f"Created at: `{summary_payload['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append("## Object Size Policy")
    md.append("")
    md.append("- Method: `target_box_normalized_area_quantiles`")
    md.append("- Normalized area: `bbox_area / image_area`")
    md.append("- Small: `normalized_area <= 0.004157071390230814`")
    md.append("- Medium: `0.004157071390230814 < normalized_area <= 0.015589650670960799`")
    md.append("- Large: `normalized_area > 0.015589650670960799`")
    md.append("- Target boxes used to define thresholds: `39086`")
    md.append("")
    md.append("## Processing Summary")
    md.append("")
    md.append(f"- Rows read from detection error index: `{rows_read}`")
    md.append(f"- GT-centered TP/FN events used: `{gt_events_used}`")
    md.append(f"- Non-object-size rows ignored: `{ignored_rows}`")
    md.append("")
    md.append("## Best Small-Object Recall by Dataset, All Classes")
    md.append("")
    md.append("| Dataset | Detector | Recall | False Negative Rate | GT Objects |")
    md.append("|---|---|---:|---:|---:|")
    for dataset, row in best_small_by_dataset.items():
        if row:
            md.append(
                f"| {dataset} | {row['detector']} | {row['recall']} | "
                f"{row['false_negative_rate']} | {row['gt_objects']} |"
            )
    md.append("")
    md.append("## Worst Waymo Small Vulnerable-Road-User Rows")
    md.append("")
    md.append("| Detector | Class | Recall | False Negative Rate | GT Objects | TP | FN |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for row in vulnerable_small_waymo[:10]:
        md.append(
            f"| {row['detector']} | {row['class_name']} | {row['recall']} | "
            f"{row['false_negative_rate']} | {row['gt_objects']} | {row['tp']} | {row['fn']} |"
        )
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append(f"- `{OBJECT_SIZE_SUMMARY_CSV.relative_to(PROJECT)}`")
    md.append(f"- `{SMALL_OBJECT_FAILURE_CSV.relative_to(PROJECT)}`")
    md.append(f"- `{DATASET_COMPARISON_CSV.relative_to(PROJECT)}`")
    md.append(f"- `{SUMMARY_JSON.relative_to(PROJECT)}`")
    md.append("")

    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Object-size analysis created")
    print("=" * 100)
    print("Rows read:", rows_read)
    print("GT-centered TP/FN events used:", gt_events_used)
    print("Ignored non-object-size rows:", ignored_rows)
    print("Created:", OBJECT_SIZE_SUMMARY_CSV)
    print("Created:", OBJECT_SIZE_SUMMARY_JSON)
    print("Created:", SMALL_OBJECT_FAILURE_CSV)
    print("Created:", SMALL_OBJECT_FAILURE_JSON)
    print("Created:", DATASET_COMPARISON_CSV)
    print("Created:", DATASET_COMPARISON_JSON)
    print("Created:", SUMMARY_JSON)
    print("Created:", SUMMARY_MD)

    print()
    print("Best small-object recall by dataset, all classes:")
    for dataset, row in best_small_by_dataset.items():
        if row:
            print(
                f"  {dataset}: {row['detector']} | recall={row['recall']} | "
                f"FNR={row['false_negative_rate']} | gt={row['gt_objects']}"
            )

    print()
    print("Worst Waymo small vulnerable-road-user rows:")
    for row in vulnerable_small_waymo[:8]:
        print(
            f"  {row['detector']} / {row['class_name']} | "
            f"recall={row['recall']} | FNR={row['false_negative_rate']} | "
            f"gt={row['gt_objects']} | tp={row['tp']} | fn={row['fn']}"
        )

    print()
    print("STEP 4/10 COMPLETE ✅")
    print("Object-size robustness analysis is ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()