from pathlib import Path
from datetime import datetime
import csv
import json
import heapq
from collections import defaultdict


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

INPUT_CSV = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_error_index.csv"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_TYPE_SUMMARY_CSV = OUTPUT_DIR / "failure_type_summary.csv"
FAILURE_TYPE_SUMMARY_JSON = OUTPUT_DIR / "failure_type_summary.json"

FALSE_POSITIVE_SUMMARY_CSV = OUTPUT_DIR / "false_positive_summary.csv"
FALSE_POSITIVE_SUMMARY_JSON = OUTPUT_DIR / "false_positive_summary.json"

LOCALIZATION_ERROR_SUMMARY_CSV = OUTPUT_DIR / "localization_error_summary.csv"
LOCALIZATION_ERROR_SUMMARY_JSON = OUTPUT_DIR / "localization_error_summary.json"

CLASS_CONFUSION_SUMMARY_CSV = OUTPUT_DIR / "class_confusion_summary.csv"
CLASS_CONFUSION_SUMMARY_JSON = OUTPUT_DIR / "class_confusion_summary.json"

DUPLICATE_DETECTION_SUMMARY_CSV = OUTPUT_DIR / "duplicate_detection_summary.csv"
DUPLICATE_DETECTION_SUMMARY_JSON = OUTPUT_DIR / "duplicate_detection_summary.json"

TOP_FAILURE_IMAGES_CSV = OUTPUT_DIR / "top_failure_type_images.csv"
TOP_FAILURE_IMAGES_JSON = OUTPUT_DIR / "top_failure_type_images.json"

FAILURE_CASE_CANDIDATES_CSV = OUTPUT_DIR / "failure_case_candidate_rows.csv"
FAILURE_CASE_CANDIDATES_JSON = OUTPUT_DIR / "failure_case_candidate_rows.json"

SUMMARY_JSON = OUTPUT_DIR / "failure_type_analysis_summary.json"
SUMMARY_MD = OUTPUT_DIR / "MILESTONE_7_FAILURE_TYPE_ANALYSIS.md"


DETECTOR_ORDER = {
    "yolo": 0,
    "rtdetr": 1,
    "retinanet": 2,
    "faster_rcnn": 3,
}

CLASS_ORDER = {
    "Vehicle": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Vulnerable_Road_Users": 3,
    "All_Classes": 4,
    "not_applicable": 5,
}

SIZE_ORDER = {
    "small": 0,
    "medium": 1,
    "large": 2,
    "all_sizes": 3,
    "not_applicable": 4,
}

FAILURE_ORDER = {
    "false_negative": 0,
    "false_positive": 1,
    "localization_error": 2,
    "class_confusion": 3,
    "duplicate_detection": 4,
    "true_positive": 5,
}

RISK_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "none": 4,
    "unknown": 5,
}

RISK_WEIGHT = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
    "none": 0.0,
    "unknown": 0.5,
}

SAFETY_CLASSES = {"Pedestrian", "Cyclist"}

HIGH_CONFIDENCE_THRESHOLD = 0.25
TOP_CANDIDATE_LIMIT = 300


def safe_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def rounded(value, digits=6):
    if value is None:
        return ""
    return round(float(value), digits)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def init_agg():
    return {
        "count": 0,

        "score_sum": 0.0,
        "score_count": 0,

        "iou_sum": 0.0,
        "iou_count": 0,

        "best_same_sum": 0.0,
        "best_same_count": 0,

        "best_wrong_sum": 0.0,
        "best_wrong_count": 0,

        "high_confidence_count": 0,
        "safety_relevant_count": 0,
    }


def is_safety_relevant(row):
    values = [
        row.get("analysis_class_name", ""),
        row.get("gt_class_name", ""),
        row.get("pred_class_name", ""),
    ]
    return any(v in SAFETY_CLASSES for v in values)


def add_agg(agg, row):
    agg["count"] += 1

    score = safe_float(row.get("score"))
    iou = safe_float(row.get("iou"))
    best_same = safe_float(row.get("best_same_class_iou"))
    best_wrong = safe_float(row.get("best_wrong_class_iou"))

    if score is not None:
        agg["score_sum"] += score
        agg["score_count"] += 1

        if score >= HIGH_CONFIDENCE_THRESHOLD:
            agg["high_confidence_count"] += 1

    if iou is not None:
        agg["iou_sum"] += iou
        agg["iou_count"] += 1

    if best_same is not None:
        agg["best_same_sum"] += best_same
        agg["best_same_count"] += 1

    if best_wrong is not None:
        agg["best_wrong_sum"] += best_wrong
        agg["best_wrong_count"] += 1

    if is_safety_relevant(row):
        agg["safety_relevant_count"] += 1


def mean_from_agg(agg, sum_key, count_key, digits=6):
    count = agg[count_key]
    if count == 0:
        return ""
    return round(agg[sum_key] / count, digits)


def normalize_blank(value, fallback="not_applicable"):
    if value is None or value == "":
        return fallback
    return value


def failure_summary_row(key, agg):
    dataset, detector, failure_type, class_name, size_bin, risk_level = key

    return {
        "dataset": dataset,
        "detector": detector,
        "failure_type": failure_type,
        "analysis_class_name": class_name,
        "object_size_bin": size_bin,
        "risk_level": risk_level,
        "count": agg["count"],
        "high_confidence_count": agg["high_confidence_count"],
        "safety_relevant_count": agg["safety_relevant_count"],
        "mean_score": mean_from_agg(agg, "score_sum", "score_count", 8),
        "mean_iou": mean_from_agg(agg, "iou_sum", "iou_count", 6),
        "mean_best_same_class_iou": mean_from_agg(agg, "best_same_sum", "best_same_count", 6),
        "mean_best_wrong_class_iou": mean_from_agg(agg, "best_wrong_sum", "best_wrong_count", 6),
    }


def fp_summary_row(key, agg):
    dataset, detector, pred_class_name, risk_level = key

    return {
        "dataset": dataset,
        "detector": detector,
        "pred_class_name": pred_class_name,
        "risk_level": risk_level,
        "false_positive_count": agg["count"],
        "high_confidence_fp_count": agg["high_confidence_count"],
        "safety_relevant_count": agg["safety_relevant_count"],
        "mean_score": mean_from_agg(agg, "score_sum", "score_count", 8),
        "mean_iou_to_any_gt": mean_from_agg(agg, "iou_sum", "iou_count", 6),
    }


def localization_summary_row(key, agg):
    dataset, detector, class_name, size_bin, risk_level = key

    return {
        "dataset": dataset,
        "detector": detector,
        "class_name": class_name,
        "object_size_bin": size_bin,
        "risk_level": risk_level,
        "localization_error_count": agg["count"],
        "high_confidence_count": agg["high_confidence_count"],
        "safety_relevant_count": agg["safety_relevant_count"],
        "mean_score": mean_from_agg(agg, "score_sum", "score_count", 8),
        "mean_iou": mean_from_agg(agg, "iou_sum", "iou_count", 6),
    }


def confusion_summary_row(key, agg):
    dataset, detector, gt_class_name, pred_class_name, size_bin, risk_level = key

    return {
        "dataset": dataset,
        "detector": detector,
        "gt_class_name": gt_class_name,
        "pred_class_name": pred_class_name,
        "object_size_bin": size_bin,
        "risk_level": risk_level,
        "class_confusion_count": agg["count"],
        "high_confidence_count": agg["high_confidence_count"],
        "safety_relevant_count": agg["safety_relevant_count"],
        "mean_score": mean_from_agg(agg, "score_sum", "score_count", 8),
        "mean_iou": mean_from_agg(agg, "iou_sum", "iou_count", 6),
    }


def duplicate_summary_row(key, agg):
    dataset, detector, class_name, size_bin, risk_level = key

    return {
        "dataset": dataset,
        "detector": detector,
        "class_name": class_name,
        "object_size_bin": size_bin,
        "risk_level": risk_level,
        "duplicate_detection_count": agg["count"],
        "high_confidence_count": agg["high_confidence_count"],
        "safety_relevant_count": agg["safety_relevant_count"],
        "mean_score": mean_from_agg(agg, "score_sum", "score_count", 8),
        "mean_iou": mean_from_agg(agg, "iou_sum", "iou_count", 6),
    }


def top_image_init():
    return {
        "dataset": "",
        "detector": "",
        "image_id": "",
        "image_path": "",

        "false_negative_count": 0,
        "false_positive_count": 0,
        "localization_error_count": 0,
        "class_confusion_count": 0,
        "duplicate_detection_count": 0,

        "high_confidence_false_positive_count": 0,
        "safety_relevant_failure_count": 0,
        "total_non_tp_failure_events": 0,
        "failure_type_score": 0.0,

        "image_width": "",
        "image_height": "",
        "total_gt_in_image": "",
        "total_predictions_in_image": "",
    }


def update_image_failure(item, row):
    failure_type = row.get("failure_type", "")
    score = safe_float(row.get("score"))
    risk = row.get("risk_level", "unknown")

    item["dataset"] = row.get("dataset", "")
    item["detector"] = row.get("detector", "")
    item["image_id"] = row.get("image_id", "")
    item["image_path"] = row.get("image_path", "")
    item["image_width"] = row.get("image_width", "")
    item["image_height"] = row.get("image_height", "")
    item["total_gt_in_image"] = row.get("total_gt_in_image", "")
    item["total_predictions_in_image"] = row.get("total_predictions_in_image", "")

    if failure_type == "false_negative":
        item["false_negative_count"] += 1
    elif failure_type == "false_positive":
        item["false_positive_count"] += 1
        if score is not None and score >= HIGH_CONFIDENCE_THRESHOLD:
            item["high_confidence_false_positive_count"] += 1
    elif failure_type == "localization_error":
        item["localization_error_count"] += 1
    elif failure_type == "class_confusion":
        item["class_confusion_count"] += 1
    elif failure_type == "duplicate_detection":
        item["duplicate_detection_count"] += 1

    item["total_non_tp_failure_events"] += 1

    if is_safety_relevant(row):
        item["safety_relevant_failure_count"] += 1

    item["failure_type_score"] += RISK_WEIGHT.get(risk, 0.5)

    if score is not None:
        item["failure_type_score"] += min(score, 1.0) * 0.25


def get_box_values(row, prefix):
    return [
        row.get(f"{prefix}_x1", ""),
        row.get(f"{prefix}_y1", ""),
        row.get(f"{prefix}_x2", ""),
        row.get(f"{prefix}_y2", ""),
    ]


def push_candidate(heap, serial, candidate):
    priority = candidate["priority"]

    heapq.heappush(heap, (priority, serial, candidate))

    if len(heap) > TOP_CANDIDATE_LIMIT:
        heapq.heappop(heap)


def make_candidate(row, candidate_type):
    risk = row.get("risk_level", "unknown")
    score = safe_float(row.get("score")) or 0.0
    iou = safe_float(row.get("iou")) or 0.0
    best_same = safe_float(row.get("best_same_class_iou")) or 0.0
    best_wrong = safe_float(row.get("best_wrong_class_iou")) or 0.0

    priority = RISK_WEIGHT.get(risk, 0.5)

    if candidate_type == "false_positive":
        priority += score
        priority += 0.25 if is_safety_relevant(row) else 0.0

    elif candidate_type == "localization_error":
        priority += score
        priority += max(0.0, 0.50 - iou)
        priority += 0.50 if is_safety_relevant(row) else 0.0

    elif candidate_type == "class_confusion":
        priority += score
        priority += iou
        priority += 0.75 if is_safety_relevant(row) else 0.0

    elif candidate_type == "duplicate_detection":
        priority += score
        priority += iou

    else:
        priority += score

    gt_box = get_box_values(row, "gt")
    pred_box = get_box_values(row, "pred")

    return {
        "candidate_type": candidate_type,
        "dataset": row.get("dataset", ""),
        "detector": row.get("detector", ""),
        "image_id": row.get("image_id", ""),
        "image_path": row.get("image_path", ""),
        "risk_level": risk,

        "analysis_class_name": row.get("analysis_class_name", ""),
        "gt_class_name": row.get("gt_class_name", ""),
        "pred_class_name": row.get("pred_class_name", ""),
        "object_size_bin": row.get("object_size_bin", ""),

        "score": rounded(score, 8),
        "iou": rounded(iou, 6),
        "best_same_class_iou": rounded(best_same, 6),
        "best_wrong_class_iou": rounded(best_wrong, 6),
        "priority": rounded(priority, 6),

        "gt_x1": gt_box[0],
        "gt_y1": gt_box[1],
        "gt_x2": gt_box[2],
        "gt_y2": gt_box[3],
        "pred_x1": pred_box[0],
        "pred_y1": pred_box[1],
        "pred_x2": pred_box[2],
        "pred_y2": pred_box[3],

        "image_width": row.get("image_width", ""),
        "image_height": row.get("image_height", ""),
        "total_gt_in_image": row.get("total_gt_in_image", ""),
        "total_predictions_in_image": row.get("total_predictions_in_image", ""),
    }


def sort_failure_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            FAILURE_ORDER.get(r.get("failure_type", ""), 99),
            CLASS_ORDER.get(r.get("analysis_class_name", ""), 99),
            SIZE_ORDER.get(r.get("object_size_bin", ""), 99),
            RISK_ORDER.get(r.get("risk_level", ""), 99),
        ),
    )


def main():
    print("=" * 100)
    print("STEP 6/10 - Failure type analysis")
    print("=" * 100)

    if not INPUT_CSV.exists():
        print("ERROR: Missing input:", INPUT_CSV)
        print("STEP 6/10 FAILED ❌")
        raise SystemExit(1)

    failure_summary = defaultdict(init_agg)
    fp_summary = defaultdict(init_agg)
    localization_summary = defaultdict(init_agg)
    confusion_summary = defaultdict(init_agg)
    duplicate_summary = defaultdict(init_agg)
    image_failures = defaultdict(top_image_init)

    candidate_heap = []
    serial = 0

    rows_read = 0
    failure_rows_used = 0
    true_positive_rows = 0

    print("Reading detection error index:")
    print(INPUT_CSV)

    with INPUT_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows_read += 1

            failure_type = row.get("failure_type", "")
            dataset = row.get("dataset", "")
            detector = row.get("detector", "")

            analysis_class_name = normalize_blank(row.get("analysis_class_name"))
            object_size_bin = normalize_blank(row.get("object_size_bin"))
            risk_level = normalize_blank(row.get("risk_level"), "unknown")

            if failure_type == "true_positive":
                true_positive_rows += 1

            failure_key = (
                dataset,
                detector,
                failure_type,
                analysis_class_name,
                object_size_bin,
                risk_level,
            )
            add_agg(failure_summary[failure_key], row)

            if failure_type != "true_positive":
                failure_rows_used += 1

                image_key = (
                    dataset,
                    detector,
                    row.get("image_id", ""),
                    row.get("image_path", ""),
                )
                update_image_failure(image_failures[image_key], row)

            if failure_type == "false_positive":
                pred_class_name = normalize_blank(row.get("pred_class_name"))
                key = (dataset, detector, pred_class_name, risk_level)
                add_agg(fp_summary[key], row)

                serial += 1
                push_candidate(candidate_heap, serial, make_candidate(row, "false_positive"))

            elif failure_type == "localization_error":
                class_name = normalize_blank(row.get("gt_class_name"))
                key = (dataset, detector, class_name, object_size_bin, risk_level)
                add_agg(localization_summary[key], row)

                serial += 1
                push_candidate(candidate_heap, serial, make_candidate(row, "localization_error"))

            elif failure_type == "class_confusion":
                gt_class_name = normalize_blank(row.get("gt_class_name"))
                pred_class_name = normalize_blank(row.get("pred_class_name"))
                key = (dataset, detector, gt_class_name, pred_class_name, object_size_bin, risk_level)
                add_agg(confusion_summary[key], row)

                serial += 1
                push_candidate(candidate_heap, serial, make_candidate(row, "class_confusion"))

            elif failure_type == "duplicate_detection":
                class_name = normalize_blank(row.get("analysis_class_name"))
                key = (dataset, detector, class_name, object_size_bin, risk_level)
                add_agg(duplicate_summary[key], row)

                serial += 1
                push_candidate(candidate_heap, serial, make_candidate(row, "duplicate_detection"))

            if rows_read % 250000 == 0:
                print(f"  read {rows_read:,} rows")

    failure_rows = [
        failure_summary_row(key, agg)
        for key, agg in failure_summary.items()
    ]

    failure_rows = sorted(
        failure_rows,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            FAILURE_ORDER.get(r["failure_type"], 99),
            CLASS_ORDER.get(r["analysis_class_name"], 99),
            SIZE_ORDER.get(r["object_size_bin"], 99),
            RISK_ORDER.get(r["risk_level"], 99),
        ),
    )

    failure_fieldnames = [
        "dataset",
        "detector",
        "failure_type",
        "analysis_class_name",
        "object_size_bin",
        "risk_level",
        "count",
        "high_confidence_count",
        "safety_relevant_count",
        "mean_score",
        "mean_iou",
        "mean_best_same_class_iou",
        "mean_best_wrong_class_iou",
    ]

    write_csv(FAILURE_TYPE_SUMMARY_CSV, failure_rows, failure_fieldnames)
    write_json(FAILURE_TYPE_SUMMARY_JSON, failure_rows)

    fp_rows = [
        fp_summary_row(key, agg)
        for key, agg in fp_summary.items()
    ]

    fp_rows = sorted(
        fp_rows,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            CLASS_ORDER.get(r["pred_class_name"], 99),
            RISK_ORDER.get(r["risk_level"], 99),
        ),
    )

    fp_fieldnames = [
        "dataset",
        "detector",
        "pred_class_name",
        "risk_level",
        "false_positive_count",
        "high_confidence_fp_count",
        "safety_relevant_count",
        "mean_score",
        "mean_iou_to_any_gt",
    ]

    write_csv(FALSE_POSITIVE_SUMMARY_CSV, fp_rows, fp_fieldnames)
    write_json(FALSE_POSITIVE_SUMMARY_JSON, fp_rows)

    loc_rows = [
        localization_summary_row(key, agg)
        for key, agg in localization_summary.items()
    ]

    loc_rows = sorted(
        loc_rows,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            CLASS_ORDER.get(r["class_name"], 99),
            SIZE_ORDER.get(r["object_size_bin"], 99),
            RISK_ORDER.get(r["risk_level"], 99),
        ),
    )

    loc_fieldnames = [
        "dataset",
        "detector",
        "class_name",
        "object_size_bin",
        "risk_level",
        "localization_error_count",
        "high_confidence_count",
        "safety_relevant_count",
        "mean_score",
        "mean_iou",
    ]

    write_csv(LOCALIZATION_ERROR_SUMMARY_CSV, loc_rows, loc_fieldnames)
    write_json(LOCALIZATION_ERROR_SUMMARY_JSON, loc_rows)

    confusion_rows = [
        confusion_summary_row(key, agg)
        for key, agg in confusion_summary.items()
    ]

    confusion_rows = sorted(
        confusion_rows,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            CLASS_ORDER.get(r["gt_class_name"], 99),
            CLASS_ORDER.get(r["pred_class_name"], 99),
            SIZE_ORDER.get(r["object_size_bin"], 99),
            RISK_ORDER.get(r["risk_level"], 99),
        ),
    )

    confusion_fieldnames = [
        "dataset",
        "detector",
        "gt_class_name",
        "pred_class_name",
        "object_size_bin",
        "risk_level",
        "class_confusion_count",
        "high_confidence_count",
        "safety_relevant_count",
        "mean_score",
        "mean_iou",
    ]

    write_csv(CLASS_CONFUSION_SUMMARY_CSV, confusion_rows, confusion_fieldnames)
    write_json(CLASS_CONFUSION_SUMMARY_JSON, confusion_rows)

    duplicate_rows = [
        duplicate_summary_row(key, agg)
        for key, agg in duplicate_summary.items()
    ]

    duplicate_rows = sorted(
        duplicate_rows,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            CLASS_ORDER.get(r["class_name"], 99),
            SIZE_ORDER.get(r["object_size_bin"], 99),
            RISK_ORDER.get(r["risk_level"], 99),
        ),
    )

    duplicate_fieldnames = [
        "dataset",
        "detector",
        "class_name",
        "object_size_bin",
        "risk_level",
        "duplicate_detection_count",
        "high_confidence_count",
        "safety_relevant_count",
        "mean_score",
        "mean_iou",
    ]

    write_csv(DUPLICATE_DETECTION_SUMMARY_CSV, duplicate_rows, duplicate_fieldnames)
    write_json(DUPLICATE_DETECTION_SUMMARY_JSON, duplicate_rows)

    image_rows = list(image_failures.values())

    for item in image_rows:
        item["failure_type_score"] = round(float(item["failure_type_score"]), 6)

    image_rows = sorted(
        image_rows,
        key=lambda r: (
            -r["failure_type_score"],
            -r["safety_relevant_failure_count"],
            -r["class_confusion_count"],
            -r["localization_error_count"],
            -r["high_confidence_false_positive_count"],
            DETECTOR_ORDER.get(r["detector"], 99),
        ),
    )

    top_image_fieldnames = [
        "dataset",
        "detector",
        "image_id",
        "image_path",
        "false_negative_count",
        "false_positive_count",
        "localization_error_count",
        "class_confusion_count",
        "duplicate_detection_count",
        "high_confidence_false_positive_count",
        "safety_relevant_failure_count",
        "total_non_tp_failure_events",
        "failure_type_score",
        "image_width",
        "image_height",
        "total_gt_in_image",
        "total_predictions_in_image",
    ]

    write_csv(TOP_FAILURE_IMAGES_CSV, image_rows[:500], top_image_fieldnames)
    write_json(TOP_FAILURE_IMAGES_JSON, image_rows[:500])

    candidate_rows = [
        item[2]
        for item in sorted(candidate_heap, key=lambda x: x[0], reverse=True)
    ]

    candidate_fieldnames = [
        "candidate_type",
        "dataset",
        "detector",
        "image_id",
        "image_path",
        "risk_level",
        "analysis_class_name",
        "gt_class_name",
        "pred_class_name",
        "object_size_bin",
        "score",
        "iou",
        "best_same_class_iou",
        "best_wrong_class_iou",
        "priority",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "image_width",
        "image_height",
        "total_gt_in_image",
        "total_predictions_in_image",
    ]

    write_csv(FAILURE_CASE_CANDIDATES_CSV, candidate_rows, candidate_fieldnames)
    write_json(FAILURE_CASE_CANDIDATES_JSON, candidate_rows)

    # Console/report highlights.
    failure_type_totals = defaultdict(int)
    for r in failure_rows:
        if r["failure_type"] != "true_positive":
            failure_type_totals[(r["dataset"], r["detector"], r["failure_type"])] += r["count"]

    total_failure_highlights = [
        {
            "dataset": dataset,
            "detector": detector,
            "failure_type": failure_type,
            "count": count,
        }
        for (dataset, detector, failure_type), count in failure_type_totals.items()
    ]

    total_failure_highlights = sorted(
        total_failure_highlights,
        key=lambda r: (
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
            FAILURE_ORDER.get(r["failure_type"], 99),
        ),
    )

    top_confusions = sorted(
        confusion_rows,
        key=lambda r: (
            -r["class_confusion_count"],
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
        ),
    )[:12]

    top_localization = sorted(
        loc_rows,
        key=lambda r: (
            -r["localization_error_count"],
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
        ),
    )[:12]

    top_fp = sorted(
        fp_rows,
        key=lambda r: (
            -r["false_positive_count"],
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
        ),
    )[:12]

    high_conf_fp = sorted(
        fp_rows,
        key=lambda r: (
            -r["high_confidence_fp_count"],
            r["dataset"],
            DETECTOR_ORDER.get(r["detector"], 99),
        ),
    )[:12]

    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "input": str(INPUT_CSV.relative_to(PROJECT)),
        "rows_read": rows_read,
        "true_positive_rows": true_positive_rows,
        "non_true_positive_failure_rows": failure_rows_used,
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "top_failure_type_totals": total_failure_highlights,
        "top_class_confusions": top_confusions,
        "top_localization_errors": top_localization,
        "top_false_positives": top_fp,
        "top_high_confidence_false_positives": high_conf_fp,
        "top_failure_case_candidates": candidate_rows[:30],
        "outputs": {
            "failure_type_summary_csv": str(FAILURE_TYPE_SUMMARY_CSV.relative_to(PROJECT)),
            "false_positive_summary_csv": str(FALSE_POSITIVE_SUMMARY_CSV.relative_to(PROJECT)),
            "localization_error_summary_csv": str(LOCALIZATION_ERROR_SUMMARY_CSV.relative_to(PROJECT)),
            "class_confusion_summary_csv": str(CLASS_CONFUSION_SUMMARY_CSV.relative_to(PROJECT)),
            "duplicate_detection_summary_csv": str(DUPLICATE_DETECTION_SUMMARY_CSV.relative_to(PROJECT)),
            "top_failure_type_images_csv": str(TOP_FAILURE_IMAGES_CSV.relative_to(PROJECT)),
            "failure_case_candidates_csv": str(FAILURE_CASE_CANDIDATES_CSV.relative_to(PROJECT)),
            "summary_md": str(SUMMARY_MD.relative_to(PROJECT)),
        },
    }

    write_json(SUMMARY_JSON, summary_payload)

    md = []
    md.append("# Milestone 7 Failure Type Analysis")
    md.append("")
    md.append(f"Created at: `{summary_payload['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append(
        "This analysis summarizes false positives, localization errors, class confusion, "
        "and duplicate detections using the Milestone 7 detection error index."
    )
    md.append("")
    md.append("## Processing Summary")
    md.append("")
    md.append(f"- Rows read: `{rows_read}`")
    md.append(f"- True positive rows: `{true_positive_rows}`")
    md.append(f"- Non-true-positive failure rows: `{failure_rows_used}`")
    md.append(f"- High-confidence threshold for false positives: `{HIGH_CONFIDENCE_THRESHOLD}`")
    md.append("")
    md.append("## Failure Type Totals")
    md.append("")
    md.append("| Dataset | Detector | Failure Type | Count |")
    md.append("|---|---|---|---:|")
    for r in total_failure_highlights:
        md.append(
            f"| {r['dataset']} | {r['detector']} | {r['failure_type']} | {r['count']} |"
        )
    md.append("")
    md.append("## Top Class Confusions")
    md.append("")
    md.append("| Dataset | Detector | GT Class | Pred Class | Size | Count | Mean IoU | Mean Score |")
    md.append("|---|---|---|---|---|---:|---:|---:|")
    for r in top_confusions:
        md.append(
            f"| {r['dataset']} | {r['detector']} | {r['gt_class_name']} | "
            f"{r['pred_class_name']} | {r['object_size_bin']} | "
            f"{r['class_confusion_count']} | {r['mean_iou']} | {r['mean_score']} |"
        )
    md.append("")
    md.append("## Top Localization Error Groups")
    md.append("")
    md.append("| Dataset | Detector | Class | Size | Count | Mean IoU | Mean Score |")
    md.append("|---|---|---|---|---:|---:|---:|")
    for r in top_localization:
        md.append(
            f"| {r['dataset']} | {r['detector']} | {r['class_name']} | "
            f"{r['object_size_bin']} | {r['localization_error_count']} | "
            f"{r['mean_iou']} | {r['mean_score']} |"
        )
    md.append("")
    md.append("## Top False Positive Groups")
    md.append("")
    md.append("| Dataset | Detector | Pred Class | Count | High-Confidence Count | Mean Score |")
    md.append("|---|---|---|---:|---:|---:|")
    for r in top_fp:
        md.append(
            f"| {r['dataset']} | {r['detector']} | {r['pred_class_name']} | "
            f"{r['false_positive_count']} | {r['high_confidence_fp_count']} | {r['mean_score']} |"
        )
    md.append("")
    md.append("## Outputs")
    md.append("")
    for _, rel_path in summary_payload["outputs"].items():
        md.append(f"- `{rel_path}`")
    md.append("")

    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Failure type analysis created")
    print("=" * 100)
    print("Rows read:", rows_read)
    print("True positive rows:", true_positive_rows)
    print("Non-true-positive failure rows:", failure_rows_used)
    print("High-confidence threshold:", HIGH_CONFIDENCE_THRESHOLD)

    print()
    print("Created:", FAILURE_TYPE_SUMMARY_CSV)
    print("Created:", FALSE_POSITIVE_SUMMARY_CSV)
    print("Created:", LOCALIZATION_ERROR_SUMMARY_CSV)
    print("Created:", CLASS_CONFUSION_SUMMARY_CSV)
    print("Created:", DUPLICATE_DETECTION_SUMMARY_CSV)
    print("Created:", TOP_FAILURE_IMAGES_CSV)
    print("Created:", FAILURE_CASE_CANDIDATES_CSV)
    print("Created:", SUMMARY_JSON)
    print("Created:", SUMMARY_MD)

    print()
    print("Failure type totals:")
    for r in total_failure_highlights:
        print(
            f"  {r['dataset']} / {r['detector']} / {r['failure_type']}: {r['count']}"
        )

    print()
    print("Top class confusions:")
    for r in top_confusions[:8]:
        print(
            f"  {r['dataset']} / {r['detector']} | "
            f"{r['gt_class_name']} -> {r['pred_class_name']} | "
            f"size={r['object_size_bin']} | count={r['class_confusion_count']} | "
            f"mean_iou={r['mean_iou']} | mean_score={r['mean_score']}"
        )

    print()
    print("Top localization errors:")
    for r in top_localization[:8]:
        print(
            f"  {r['dataset']} / {r['detector']} | "
            f"{r['class_name']} | size={r['object_size_bin']} | "
            f"count={r['localization_error_count']} | mean_iou={r['mean_iou']} | "
            f"mean_score={r['mean_score']}"
        )

    print()
    print("Top high-confidence false positives:")
    for r in high_conf_fp[:8]:
        print(
            f"  {r['dataset']} / {r['detector']} | "
            f"pred={r['pred_class_name']} | FP={r['false_positive_count']} | "
            f"high_conf_FP={r['high_confidence_fp_count']} | mean_score={r['mean_score']}"
        )

    print()
    print("STEP 6/10 COMPLETE ✅")
    print("False positive, localization, class-confusion, and duplicate-detection analysis is ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()