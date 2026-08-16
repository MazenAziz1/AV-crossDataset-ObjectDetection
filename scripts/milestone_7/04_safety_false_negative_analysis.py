from pathlib import Path
from datetime import datetime
import csv
import json
from collections import defaultdict, Counter


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

INPUT_CSV = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_error_index.csv"
OBJECT_SIZE_SUMMARY_CSV = PROJECT / "outputs" / "milestone_7" / "object_size_analysis" / "object_size_summary.csv"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAFETY_FN_SUMMARY_CSV = OUTPUT_DIR / "safety_false_negative_summary.csv"
SAFETY_FN_SUMMARY_JSON = OUTPUT_DIR / "safety_false_negative_summary.json"

TOP_IMAGES_CSV = OUTPUT_DIR / "top_safety_critical_images.csv"
TOP_IMAGES_JSON = OUTPUT_DIR / "top_safety_critical_images.json"

SAFETY_DATASET_COMPARISON_CSV = OUTPUT_DIR / "safety_dataset_comparison.csv"
SAFETY_DATASET_COMPARISON_JSON = OUTPUT_DIR / "safety_dataset_comparison.json"

SAFETY_SUMMARY_JSON = OUTPUT_DIR / "safety_false_negative_analysis_summary.json"
SAFETY_SUMMARY_MD = OUTPUT_DIR / "MILESTONE_7_SAFETY_FALSE_NEGATIVE_ANALYSIS.md"


SAFETY_CLASSES = {"Pedestrian", "Cyclist"}
ALL_SAFETY_LABEL = "Vulnerable_Road_Users"

DETECTOR_ORDER = {
    "yolo": 0,
    "rtdetr": 1,
    "retinanet": 2,
    "faster_rcnn": 3,
}

CLASS_ORDER = {
    "Pedestrian": 0,
    "Cyclist": 1,
    ALL_SAFETY_LABEL: 2,
}

SIZE_ORDER = {
    "small": 0,
    "medium": 1,
    "large": 2,
    "all_sizes": 3,
}


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


def rounded(value, digits=6):
    if value is None:
        return ""
    return round(float(value), digits)


def init_summary_counter():
    return {
        "tp": 0,
        "fn": 0,
        "gt_objects": 0,
        "small_fn": 0,
        "medium_fn": 0,
        "large_fn": 0,
        "critical_fn": 0,
        "normalized_area_sum_fn": 0.0,
        "normalized_area_count_fn": 0,
    }


def add_gt_event(counter, failure_type, size_bin, normalized_area):
    if failure_type == "true_positive":
        counter["tp"] += 1
        counter["gt_objects"] += 1

    elif failure_type == "false_negative":
        counter["fn"] += 1
        counter["gt_objects"] += 1
        counter["critical_fn"] += 1

        if size_bin == "small":
            counter["small_fn"] += 1
        elif size_bin == "medium":
            counter["medium_fn"] += 1
        elif size_bin == "large":
            counter["large_fn"] += 1

        if normalized_area is not None:
            counter["normalized_area_sum_fn"] += normalized_area
            counter["normalized_area_count_fn"] += 1


def summary_row(dataset, detector, class_name, size_bin, c):
    recall = safe_div(c["tp"], c["gt_objects"])
    fnr = safe_div(c["fn"], c["gt_objects"])

    mean_missed_area = None
    if c["normalized_area_count_fn"] > 0:
        mean_missed_area = c["normalized_area_sum_fn"] / c["normalized_area_count_fn"]

    return {
        "dataset": dataset,
        "detector": detector,
        "class_name": class_name,
        "object_size_bin": size_bin,
        "tp": c["tp"],
        "fn": c["fn"],
        "gt_objects": c["gt_objects"],
        "recall": rounded(recall),
        "false_negative_rate": rounded(fnr),
        "small_fn": c["small_fn"],
        "medium_fn": c["medium_fn"],
        "large_fn": c["large_fn"],
        "critical_fn": c["critical_fn"],
        "mean_normalized_area_of_missed_objects": rounded(mean_missed_area, 8),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def top_image_init():
    return {
        "dataset": "",
        "detector": "",
        "image_id": "",
        "image_path": "",
        "missed_pedestrians": 0,
        "missed_cyclists": 0,
        "small_missed_vru": 0,
        "medium_missed_vru": 0,
        "large_missed_vru": 0,
        "total_safety_false_negatives": 0,
        "max_best_same_class_iou": 0.0,
        "max_best_wrong_class_iou": 0.0,
        "safety_critical_score": 0.0,
        "image_width": "",
        "image_height": "",
        "total_gt_in_image": "",
        "total_predictions_in_image": "",
    }


def update_top_image(item, row):
    class_name = row.get("analysis_class_name", "")
    size_bin = row.get("object_size_bin", "")

    item["dataset"] = row.get("dataset", "")
    item["detector"] = row.get("detector", "")
    item["image_id"] = row.get("image_id", "")
    item["image_path"] = row.get("image_path", "")
    item["image_width"] = row.get("image_width", "")
    item["image_height"] = row.get("image_height", "")
    item["total_gt_in_image"] = row.get("total_gt_in_image", "")
    item["total_predictions_in_image"] = row.get("total_predictions_in_image", "")

    if class_name == "Pedestrian":
        item["missed_pedestrians"] += 1
    elif class_name == "Cyclist":
        item["missed_cyclists"] += 1

    if size_bin == "small":
        item["small_missed_vru"] += 1
    elif size_bin == "medium":
        item["medium_missed_vru"] += 1
    elif size_bin == "large":
        item["large_missed_vru"] += 1

    item["total_safety_false_negatives"] += 1

    best_same = safe_float(row.get("best_same_class_iou"))
    best_wrong = safe_float(row.get("best_wrong_class_iou"))

    if best_same is not None:
        item["max_best_same_class_iou"] = max(item["max_best_same_class_iou"], best_same)

    if best_wrong is not None:
        item["max_best_wrong_class_iou"] = max(item["max_best_wrong_class_iou"], best_wrong)

    # Transparent scoring:
    # every missed VRU = 1.0
    # small missed VRU gets +0.5 because it is harder and safety-relevant
    # cyclist miss gets +0.25 because cyclists were highly vulnerable in M6/M7 outputs
    item["safety_critical_score"] += 1.0

    if size_bin == "small":
        item["safety_critical_score"] += 0.5

    if class_name == "Cyclist":
        item["safety_critical_score"] += 0.25


def sort_summary(rows):
    return sorted(
        rows,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            CLASS_ORDER.get(r["class_name"], 99),
            SIZE_ORDER.get(r["object_size_bin"], 99),
        ),
    )


def main():
    print("=" * 100)
    print("STEP 5/10 - Safety-oriented false negative analysis")
    print("=" * 100)

    if not INPUT_CSV.exists():
        print("ERROR: Missing input:", INPUT_CSV)
        print("STEP 5/10 FAILED ❌")
        raise SystemExit(1)

    counters = defaultdict(init_summary_counter)
    image_counter = defaultdict(top_image_init)

    rows_read = 0
    gt_safety_events_used = 0
    safety_false_negative_rows = 0
    ignored_rows = 0

    print("Reading detection error index:")
    print(INPUT_CSV)

    with INPUT_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows_read += 1

            failure_type = row.get("failure_type", "")
            class_name = row.get("analysis_class_name", "")
            dataset = row.get("dataset", "")
            detector = row.get("detector", "")
            size_bin = row.get("object_size_bin", "")

            if failure_type not in {"true_positive", "false_negative"}:
                ignored_rows += 1
                continue

            if class_name not in SAFETY_CLASSES:
                ignored_rows += 1
                continue

            if not dataset or not detector or not size_bin:
                ignored_rows += 1
                continue

            normalized_area = safe_float(row.get("object_size_score"))

            gt_safety_events_used += 1

            # Individual safety class by size.
            add_gt_event(
                counters[(dataset, detector, class_name, size_bin)],
                failure_type,
                size_bin,
                normalized_area,
            )

            # Individual safety class across all sizes.
            add_gt_event(
                counters[(dataset, detector, class_name, "all_sizes")],
                failure_type,
                size_bin,
                normalized_area,
            )

            # Combined vulnerable road users by size.
            add_gt_event(
                counters[(dataset, detector, ALL_SAFETY_LABEL, size_bin)],
                failure_type,
                size_bin,
                normalized_area,
            )

            # Combined vulnerable road users across all sizes.
            add_gt_event(
                counters[(dataset, detector, ALL_SAFETY_LABEL, "all_sizes")],
                failure_type,
                size_bin,
                normalized_area,
            )

            if failure_type == "false_negative":
                safety_false_negative_rows += 1
                key = (
                    dataset,
                    detector,
                    row.get("image_id", ""),
                    row.get("image_path", ""),
                )
                update_top_image(image_counter[key], row)

            if rows_read % 250000 == 0:
                print(f"  read {rows_read:,} rows")

    summary_rows = []

    for key, c in counters.items():
        dataset, detector, class_name, size_bin = key

        if c["gt_objects"] == 0:
            continue

        summary_rows.append(
            summary_row(dataset, detector, class_name, size_bin, c)
        )

    summary_rows = sort_summary(summary_rows)

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
        "small_fn",
        "medium_fn",
        "large_fn",
        "critical_fn",
        "mean_normalized_area_of_missed_objects",
    ]

    write_csv(SAFETY_FN_SUMMARY_CSV, summary_rows, summary_fieldnames)
    write_json(SAFETY_FN_SUMMARY_JSON, summary_rows)

    top_image_rows = list(image_counter.values())

    for item in top_image_rows:
        item["max_best_same_class_iou"] = round(float(item["max_best_same_class_iou"]), 6)
        item["max_best_wrong_class_iou"] = round(float(item["max_best_wrong_class_iou"]), 6)
        item["safety_critical_score"] = round(float(item["safety_critical_score"]), 6)

    top_image_rows = sorted(
        top_image_rows,
        key=lambda r: (
            -r["safety_critical_score"],
            -r["total_safety_false_negatives"],
            -r["small_missed_vru"],
            DETECTOR_ORDER.get(r["detector"], 99),
        ),
    )

    top_image_fieldnames = [
        "dataset",
        "detector",
        "image_id",
        "image_path",
        "missed_pedestrians",
        "missed_cyclists",
        "small_missed_vru",
        "medium_missed_vru",
        "large_missed_vru",
        "total_safety_false_negatives",
        "max_best_same_class_iou",
        "max_best_wrong_class_iou",
        "safety_critical_score",
        "image_width",
        "image_height",
        "total_gt_in_image",
        "total_predictions_in_image",
    ]

    write_csv(TOP_IMAGES_CSV, top_image_rows, top_image_fieldnames)
    write_json(TOP_IMAGES_JSON, top_image_rows[:250])

    # Dataset comparison: KITTI vs Waymo for safety classes.
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
    keys = sorted(
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

    for detector, class_name, size_bin in keys:
        kitti = lookup.get((detector, class_name, size_bin, "kitti"))
        waymo = lookup.get((detector, class_name, size_bin, "waymo"))

        if not kitti or not waymo:
            continue

        kitti_fnr = safe_float(kitti["false_negative_rate"])
        waymo_fnr = safe_float(waymo["false_negative_rate"])
        kitti_recall = safe_float(kitti["recall"])
        waymo_recall = safe_float(waymo["recall"])

        comparison_rows.append({
            "detector": detector,
            "class_name": class_name,
            "object_size_bin": size_bin,
            "KITTI_gt_objects": kitti["gt_objects"],
            "Waymo_gt_objects": waymo["gt_objects"],
            "KITTI_fn": kitti["fn"],
            "Waymo_fn": waymo["fn"],
            "KITTI_recall": kitti["recall"],
            "Waymo_recall": waymo["recall"],
            "recall_drop_KITTI_minus_Waymo": rounded(
                kitti_recall - waymo_recall
                if kitti_recall is not None and waymo_recall is not None
                else None
            ),
            "KITTI_false_negative_rate": kitti["false_negative_rate"],
            "Waymo_false_negative_rate": waymo["false_negative_rate"],
            "fnr_increase_Waymo_minus_KITTI": rounded(
                waymo_fnr - kitti_fnr
                if kitti_fnr is not None and waymo_fnr is not None
                else None
            ),
        })

    comparison_fieldnames = [
        "detector",
        "class_name",
        "object_size_bin",
        "KITTI_gt_objects",
        "Waymo_gt_objects",
        "KITTI_fn",
        "Waymo_fn",
        "KITTI_recall",
        "Waymo_recall",
        "recall_drop_KITTI_minus_Waymo",
        "KITTI_false_negative_rate",
        "Waymo_false_negative_rate",
        "fnr_increase_Waymo_minus_KITTI",
    ]

    write_csv(SAFETY_DATASET_COMPARISON_CSV, comparison_rows, comparison_fieldnames)
    write_json(SAFETY_DATASET_COMPARISON_JSON, comparison_rows)

    # Key rows for console/report.
    waymo_all_vru = [
        r for r in summary_rows
        if r["dataset"] == "waymo"
        and r["class_name"] == ALL_SAFETY_LABEL
        and r["object_size_bin"] == "all_sizes"
    ]

    waymo_all_vru = sorted(
        waymo_all_vru,
        key=lambda r: (
            float(r["false_negative_rate"]) if r["false_negative_rate"] != "" else -1
        ),
        reverse=True,
    )

    waymo_small_vru = [
        r for r in summary_rows
        if r["dataset"] == "waymo"
        and r["class_name"] == ALL_SAFETY_LABEL
        and r["object_size_bin"] == "small"
    ]

    waymo_small_vru = sorted(
        waymo_small_vru,
        key=lambda r: (
            float(r["false_negative_rate"]) if r["false_negative_rate"] != "" else -1
        ),
        reverse=True,
    )

    best_waymo_safety = sorted(
        waymo_all_vru,
        key=lambda r: (
            float(r["false_negative_rate"]) if r["false_negative_rate"] != "" else 999
        ),
    )

    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "input": str(INPUT_CSV.relative_to(PROJECT)),
        "rows_read": rows_read,
        "gt_safety_events_used": gt_safety_events_used,
        "safety_false_negative_rows": safety_false_negative_rows,
        "ignored_rows": ignored_rows,
        "safety_focus": {
            "classes": sorted(SAFETY_CLASSES),
            "combined_label": ALL_SAFETY_LABEL,
            "rationale": "False negatives for pedestrians and cyclists are safety-critical in intelligent-vehicle perception.",
        },
        "worst_waymo_vru_all_sizes": waymo_all_vru[0] if waymo_all_vru else None,
        "worst_waymo_vru_small": waymo_small_vru[0] if waymo_small_vru else None,
        "best_waymo_vru_all_sizes": best_waymo_safety[0] if best_waymo_safety else None,
        "top_safety_critical_images": top_image_rows[:20],
        "outputs": {
            "safety_false_negative_summary_csv": str(SAFETY_FN_SUMMARY_CSV.relative_to(PROJECT)),
            "top_safety_critical_images_csv": str(TOP_IMAGES_CSV.relative_to(PROJECT)),
            "safety_dataset_comparison_csv": str(SAFETY_DATASET_COMPARISON_CSV.relative_to(PROJECT)),
            "summary_md": str(SAFETY_SUMMARY_MD.relative_to(PROJECT)),
        },
    }

    write_json(SAFETY_SUMMARY_JSON, summary_payload)

    md = []
    md.append("# Milestone 7 Safety-Oriented False Negative Analysis")
    md.append("")
    md.append(f"Created at: `{summary_payload['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append(
        "This analysis focuses on false negatives for pedestrians and cyclists, "
        "because missed vulnerable road users are safety-critical for intelligent-vehicle perception."
    )
    md.append("")
    md.append("## Processing Summary")
    md.append("")
    md.append(f"- Rows read: `{rows_read}`")
    md.append(f"- GT-centered safety events used: `{gt_safety_events_used}`")
    md.append(f"- Safety false-negative rows: `{safety_false_negative_rows}`")
    md.append(f"- Ignored rows: `{ignored_rows}`")
    md.append("")
    md.append("## Waymo Vulnerable Road User False Negative Rate, All Sizes")
    md.append("")
    md.append("| Detector | TP | FN | GT Objects | Recall | FNR |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in waymo_all_vru:
        md.append(
            f"| {r['detector']} | {r['tp']} | {r['fn']} | {r['gt_objects']} | "
            f"{r['recall']} | {r['false_negative_rate']} |"
        )
    md.append("")
    md.append("## Waymo Small Vulnerable Road User False Negative Rate")
    md.append("")
    md.append("| Detector | TP | FN | GT Objects | Recall | FNR |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in waymo_small_vru:
        md.append(
            f"| {r['detector']} | {r['tp']} | {r['fn']} | {r['gt_objects']} | "
            f"{r['recall']} | {r['false_negative_rate']} |"
        )
    md.append("")
    md.append("## Top Safety-Critical Images")
    md.append("")
    md.append("| Dataset | Detector | Image ID | Missed Pedestrians | Missed Cyclists | Small Misses | Score |")
    md.append("|---|---|---|---:|---:|---:|---:|")
    for r in top_image_rows[:20]:
        md.append(
            f"| {r['dataset']} | {r['detector']} | {r['image_id']} | "
            f"{r['missed_pedestrians']} | {r['missed_cyclists']} | "
            f"{r['small_missed_vru']} | {r['safety_critical_score']} |"
        )
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append(f"- `{SAFETY_FN_SUMMARY_CSV.relative_to(PROJECT)}`")
    md.append(f"- `{TOP_IMAGES_CSV.relative_to(PROJECT)}`")
    md.append(f"- `{SAFETY_DATASET_COMPARISON_CSV.relative_to(PROJECT)}`")
    md.append(f"- `{SAFETY_SUMMARY_JSON.relative_to(PROJECT)}`")
    md.append("")

    SAFETY_SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Safety-oriented false negative analysis created")
    print("=" * 100)
    print("Rows read:", rows_read)
    print("GT-centered safety events used:", gt_safety_events_used)
    print("Safety false-negative rows:", safety_false_negative_rows)
    print("Ignored rows:", ignored_rows)

    print()
    print("Created:", SAFETY_FN_SUMMARY_CSV)
    print("Created:", SAFETY_FN_SUMMARY_JSON)
    print("Created:", TOP_IMAGES_CSV)
    print("Created:", TOP_IMAGES_JSON)
    print("Created:", SAFETY_DATASET_COMPARISON_CSV)
    print("Created:", SAFETY_DATASET_COMPARISON_JSON)
    print("Created:", SAFETY_SUMMARY_JSON)
    print("Created:", SAFETY_SUMMARY_MD)

    print()
    print("Waymo vulnerable-road-user FNR, all sizes:")
    for r in waymo_all_vru:
        print(
            f"  {r['detector']}: recall={r['recall']} | "
            f"FNR={r['false_negative_rate']} | TP={r['tp']} | FN={r['fn']} | GT={r['gt_objects']}"
        )

    print()
    print("Waymo small vulnerable-road-user FNR:")
    for r in waymo_small_vru:
        print(
            f"  {r['detector']}: recall={r['recall']} | "
            f"FNR={r['false_negative_rate']} | TP={r['tp']} | FN={r['fn']} | GT={r['gt_objects']}"
        )

    print()
    print("Top 10 safety-critical images:")
    for r in top_image_rows[:10]:
        print(
            f"  {r['dataset']} / {r['detector']} / {r['image_id']} | "
            f"ped_FN={r['missed_pedestrians']} | cyc_FN={r['missed_cyclists']} | "
            f"small_FN={r['small_missed_vru']} | score={r['safety_critical_score']}"
        )

    print()
    print("STEP 5/10 COMPLETE ✅")
    print("Safety-oriented false negative analysis is ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()