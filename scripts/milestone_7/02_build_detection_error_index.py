from pathlib import Path
from datetime import datetime
import json
import csv
from collections import Counter, defaultdict

from PIL import Image


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

OUTPUT_DIR = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_INDEX_CSV = OUTPUT_DIR / "detection_error_index.csv"
ERROR_INDEX_JSON = OUTPUT_DIR / "detection_error_index.json"
ERROR_SUMMARY_CSV = OUTPUT_DIR / "detection_error_summary.csv"
CORE_SUMMARY_CSV = OUTPUT_DIR / "detection_core_tp_fp_fn_summary.csv"
ERROR_INDEX_MD = OUTPUT_DIR / "DETECTION_ERROR_INDEX.md"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DETECTORS = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]

CLASS_NAMES = {
    0: "Vehicle",
    1: "Pedestrian",
    2: "Cyclist",
}

DATASETS = {
    "kitti": {
        "image_dir": PROJECT / "data" / "processed" / "milestone_3" / "images" / "kitti" / "val",
        "label_dir": PROJECT / "data" / "processed" / "milestone_3" / "labels" / "kitti" / "val",
        "prediction_dir": PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "predictions",
        "expected_images": 1496,
    },
    "waymo": {
        "image_dir": PROJECT / "data" / "processed" / "milestone_3" / "images" / "waymo" / "external",
        "label_dir": PROJECT / "data" / "processed" / "milestone_3" / "labels" / "waymo" / "external",
        "prediction_dir": PROJECT / "outputs" / "milestone_6" / "waymo_external_validation" / "predictions",
        "expected_images": 996,
    },
}

PRIMARY_IOU_THRESHOLD = 0.50
LOCALIZATION_IOU_MIN = 0.10
LOCALIZATION_IOU_MAX = 0.50
CLASS_CONFUSION_IOU_MIN = 0.30
DUPLICATE_IOU_MIN = 0.50

PREDICTION_CONFIDENCE_MIN = 0.001
SMALL_MAX_NORMALIZED_AREA = 0.004157071390230814
MEDIUM_MAX_NORMALIZED_AREA = 0.015589650670960799
TARGET_BOXES_USED_FOR_SIZE_BINS = 39086


FIELDNAMES = [
    "dataset",
    "detector",
    "image_id",
    "image_path",
    "failure_type",
    "risk_level",

    "analysis_class_id",
    "analysis_class_name",
    "gt_class_id",
    "gt_class_name",
    "pred_class_id",
    "pred_class_name",

    "gt_index",
    "pred_index",
    "score",
    "iou",
    "best_same_class_iou",
    "best_wrong_class_iou",

    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "pred_x1",
    "pred_y1",
    "pred_x2",
    "pred_y2",

    "gt_width_px",
    "gt_height_px",
    "gt_area_px2",
    "gt_width_percent",
    "gt_height_percent",
    "object_size_score",
    "object_size_bin",

    "image_width",
    "image_height",
    "total_gt_in_image",
    "total_predictions_in_image",
]


def rel(path: Path):
    return str(path.relative_to(PROJECT))


def collect_images(image_dir: Path):
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def clamp(value, low, high):
    return max(low, min(value, high))


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def object_size_bin(normalized_area):
    """
    Project-specific target-box normalized-area quantile bins:
    small  = normalized_area <= 0.004157071390230814
    medium = 0.004157071390230814 < normalized_area <= 0.015589650670960799
    large  = normalized_area > 0.015589650670960799
    """
    if normalized_area <= SMALL_MAX_NORMALIZED_AREA:
        return "small"
    if normalized_area <= MEDIUM_MAX_NORMALIZED_AREA:
        return "medium"
    return "large"


def yolo_to_xyxy(class_id, xc, yc, w, h, image_width, image_height):
    x1 = (xc - w / 2.0) * image_width
    y1 = (yc - h / 2.0) * image_height
    x2 = (xc + w / 2.0) * image_width
    y2 = (yc + h / 2.0) * image_height

    x1 = clamp(float(x1), 0.0, float(image_width))
    y1 = clamp(float(y1), 0.0, float(image_height))
    x2 = clamp(float(x2), 0.0, float(image_width))
    y2 = clamp(float(y2), 0.0, float(image_height))

    width_px = max(0.0, x2 - x1)
    height_px = max(0.0, y2 - y1)
    area_px2 = width_px * height_px

    width_percent = 100.0 * width_px / float(image_width)
    height_percent = 100.0 * height_px / float(image_height)

    image_area_px2 = float(image_width) * float(image_height)
    normalized_area = area_px2 / image_area_px2 if image_area_px2 > 0 else 0.0
    size_bin = object_size_bin(normalized_area)

    return {
        "class_id": int(class_id),
        "class_name": CLASS_NAMES[int(class_id)],
        "bbox": [x1, y1, x2, y2],
        "width_px": width_px,
        "height_px": height_px,
        "area_px2": area_px2,
        "width_percent": width_percent,
        "height_percent": height_percent,
        "object_size_score": normalized_area,
        "object_size_bin": size_bin,
    }


def load_ground_truth(dataset_name, dataset_cfg):
    image_dir = dataset_cfg["image_dir"]
    label_dir = dataset_cfg["label_dir"]

    image_files = collect_images(image_dir)

    gt_by_image = {}
    image_meta = {}

    total_annotations = 0
    class_counts = Counter()
    size_counts = Counter()

    print(f"Loading ground truth for {dataset_name}: {len(image_files)} images")

    for idx, image_path in enumerate(image_files, start=1):
        image_id = image_path.stem
        label_path = label_dir / f"{image_id}.txt"

        with Image.open(image_path) as img:
            width, height = img.size

        image_meta[image_id] = {
            "image_path": rel(image_path),
            "width": width,
            "height": height,
        }

        objects = []

        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

            for line_number, line in enumerate(lines, start=1):
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO line in {label_path}, line {line_number}: {line}")

                class_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])

                if class_id not in CLASS_NAMES:
                    raise ValueError(f"Unexpected class id {class_id} in {label_path}")

                obj = yolo_to_xyxy(class_id, xc, yc, bw, bh, width, height)
                obj["gt_index"] = len(objects)
                objects.append(obj)

                total_annotations += 1
                class_counts[obj["class_name"]] += 1
                size_counts[obj["object_size_bin"]] += 1

        gt_by_image[image_id] = objects

        if idx % 300 == 0 or idx == len(image_files):
            print(f"  {dataset_name}: loaded {idx}/{len(image_files)}")

    return {
        "image_files": image_files,
        "gt_by_image": gt_by_image,
        "image_meta": image_meta,
        "total_annotations": total_annotations,
        "class_counts": dict(class_counts),
        "size_counts": dict(size_counts),
    }


def find_prediction_file(dataset_name, detector_name):
    prediction_dir = DATASETS[dataset_name]["prediction_dir"]

    if not prediction_dir.exists():
        return None

    files = sorted(prediction_dir.glob("*.jsonl"))

    detector_files = [
        p for p in files
        if detector_name.lower() in p.name.lower()
    ]

    if not detector_files:
        return None

    if dataset_name == "kitti":
        preferred = [
            p for p in detector_files
            if "full" in p.name.lower()
        ]
        return preferred[0] if preferred else detector_files[0]

    if dataset_name == "waymo":
        preferred = [
            p for p in detector_files
            if "waymo" in p.name.lower()
        ]
        return preferred[0] if preferred else detector_files[0]

    return detector_files[0]


def normalize_prediction(raw_pred):
    try:
        class_id = int(raw_pred.get("class_id"))
        score = float(raw_pred.get("score", 0.0))

        # Milestone 5 KITTI predictions use "box".
        # Milestone 6 Waymo predictions use "bbox".
        bbox_value = raw_pred.get("bbox", None)
        if bbox_value is None:
            bbox_value = raw_pred.get("box", None)

        bbox = [float(x) for x in bbox_value]
    except Exception:
        return None

    if class_id not in CLASS_NAMES:
        return None

    if score < PREDICTION_CONFIDENCE_MIN:
        return None

    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "score": score,
        "bbox": bbox,
    }

def load_predictions_by_image(prediction_file: Path):
    predictions_by_image = {}
    total_predictions = 0
    skipped_predictions = 0

    with prediction_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Could not parse {prediction_file}, line {line_number}: {exc}")

            image_id = obj.get("image_id")

            if not image_id:
                raise ValueError(f"Missing image_id in {prediction_file}, line {line_number}")

            raw_predictions = obj.get("predictions", [])

            normalized = []

            if isinstance(raw_predictions, list):
                for pred_index, raw_pred in enumerate(raw_predictions):
                    pred = normalize_prediction(raw_pred)

                    if pred is None:
                        skipped_predictions += 1
                        continue

                    pred["pred_index"] = pred_index
                    normalized.append(pred)
                    total_predictions += 1

            predictions_by_image[image_id] = normalized

    return {
        "predictions_by_image": predictions_by_image,
        "total_predictions": total_predictions,
        "skipped_predictions": skipped_predictions,
    }


def risk_level(failure_type, gt_class_id=None, pred_class_id=None):
    safety_classes = {1, 2}

    if failure_type == "true_positive":
        return "none"

    if failure_type == "false_negative":
        if gt_class_id in safety_classes:
            return "critical"
        return "medium"

    if failure_type == "class_confusion":
        if gt_class_id in safety_classes or pred_class_id in safety_classes:
            return "high"
        return "medium"

    if failure_type == "localization_error":
        if gt_class_id in safety_classes:
            return "medium"
        return "low"

    if failure_type == "false_positive":
        if pred_class_id in safety_classes:
            return "medium"
        return "low"

    if failure_type == "duplicate_detection":
        return "low"

    return "unknown"


def empty_row():
    return {field: "" for field in FIELDNAMES}


def fill_bbox(prefix, row, bbox):
    if bbox is None:
        return

    row[f"{prefix}_x1"] = round(float(bbox[0]), 3)
    row[f"{prefix}_y1"] = round(float(bbox[1]), 3)
    row[f"{prefix}_x2"] = round(float(bbox[2]), 3)
    row[f"{prefix}_y2"] = round(float(bbox[3]), 3)


def write_event(
    writer,
    summary_counter,
    core_counter,
    dataset_name,
    detector_name,
    image_id,
    image_path,
    image_width,
    image_height,
    total_gt,
    total_predictions,
    failure_type,
    gt=None,
    pred=None,
    iou=None,
    best_same_class_iou=None,
    best_wrong_class_iou=None,
):
    row = empty_row()

    gt_class_id = gt["class_id"] if gt else None
    pred_class_id = pred["class_id"] if pred else None

    if gt is not None:
        analysis_class_id = gt_class_id
        analysis_class_name = gt["class_name"]
    elif pred is not None:
        analysis_class_id = pred_class_id
        analysis_class_name = pred["class_name"]
    else:
        analysis_class_id = ""
        analysis_class_name = ""

    row["dataset"] = dataset_name
    row["detector"] = detector_name
    row["image_id"] = image_id
    row["image_path"] = image_path
    row["failure_type"] = failure_type
    row["risk_level"] = risk_level(failure_type, gt_class_id, pred_class_id)

    row["analysis_class_id"] = analysis_class_id
    row["analysis_class_name"] = analysis_class_name

    if gt is not None:
        row["gt_class_id"] = gt["class_id"]
        row["gt_class_name"] = gt["class_name"]
        row["gt_index"] = gt["gt_index"]
        fill_bbox("gt", row, gt["bbox"])

        row["gt_width_px"] = round(gt["width_px"], 3)
        row["gt_height_px"] = round(gt["height_px"], 3)
        row["gt_area_px2"] = round(gt["area_px2"], 3)
        row["gt_width_percent"] = round(gt["width_percent"], 6)
        row["gt_height_percent"] = round(gt["height_percent"], 6)
        row["object_size_score"] = round(gt["object_size_score"], 6)
        row["object_size_bin"] = gt["object_size_bin"]

    if pred is not None:
        row["pred_class_id"] = pred["class_id"]
        row["pred_class_name"] = pred["class_name"]
        row["pred_index"] = pred["pred_index"]
        row["score"] = round(pred["score"], 8)
        fill_bbox("pred", row, pred["bbox"])

    if iou is not None:
        row["iou"] = round(float(iou), 6)

    if best_same_class_iou is not None:
        row["best_same_class_iou"] = round(float(best_same_class_iou), 6)

    if best_wrong_class_iou is not None:
        row["best_wrong_class_iou"] = round(float(best_wrong_class_iou), 6)

    row["image_width"] = image_width
    row["image_height"] = image_height
    row["total_gt_in_image"] = total_gt
    row["total_predictions_in_image"] = total_predictions

    writer.writerow(row)

    summary_key = (
        dataset_name,
        detector_name,
        failure_type,
        analysis_class_name,
        row["object_size_bin"] if row["object_size_bin"] else "not_applicable",
        row["risk_level"],
    )
    summary_counter[summary_key] += 1

    core_key = (
        dataset_name,
        detector_name,
        analysis_class_name,
    )

    if failure_type == "true_positive":
        core_counter[core_key]["tp"] += 1
    elif failure_type == "false_negative":
        core_counter[core_key]["fn"] += 1
    elif failure_type == "false_positive":
        core_counter[core_key]["fp"] += 1


def best_iou_to_gt(pred, gt_objects, gt_filter=None):
    best_iou = 0.0
    best_gt_idx = None

    for idx, gt in enumerate(gt_objects):
        if gt_filter is not None and not gt_filter(idx, gt):
            continue

        iou = bbox_iou(pred["bbox"], gt["bbox"])

        if iou > best_iou:
            best_iou = iou
            best_gt_idx = idx

    return best_iou, best_gt_idx


def best_iou_to_pred(gt, predictions, pred_filter=None):
    best_iou = 0.0
    best_pred_idx = None

    for idx, pred in enumerate(predictions):
        if pred_filter is not None and not pred_filter(idx, pred):
            continue

        iou = bbox_iou(gt["bbox"], pred["bbox"])

        if iou > best_iou:
            best_iou = iou
            best_pred_idx = idx

    return best_iou, best_pred_idx


def process_image(
    writer,
    summary_counter,
    core_counter,
    dataset_name,
    detector_name,
    image_id,
    image_path,
    image_width,
    image_height,
    gt_objects,
    predictions,
):
    matched_gt = set()
    matched_pred = set()

    total_gt = len(gt_objects)
    total_predictions = len(predictions)

    # Primary true-positive matching: class-specific greedy matching by confidence.
    for class_id in CLASS_NAMES:
        pred_indices = [
            idx for idx, pred in enumerate(predictions)
            if pred["class_id"] == class_id
        ]

        pred_indices = sorted(
            pred_indices,
            key=lambda idx: predictions[idx]["score"],
            reverse=True,
        )

        for pred_idx in pred_indices:
            if pred_idx in matched_pred:
                continue

            pred = predictions[pred_idx]

            best_iou = 0.0
            best_gt_idx = None

            for gt_idx, gt in enumerate(gt_objects):
                if gt_idx in matched_gt:
                    continue

                if gt["class_id"] != class_id:
                    continue

                iou = bbox_iou(pred["bbox"], gt["bbox"])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx is not None and best_iou >= PRIMARY_IOU_THRESHOLD:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

                write_event(
                    writer=writer,
                    summary_counter=summary_counter,
                    core_counter=core_counter,
                    dataset_name=dataset_name,
                    detector_name=detector_name,
                    image_id=image_id,
                    image_path=image_path,
                    image_width=image_width,
                    image_height=image_height,
                    total_gt=total_gt,
                    total_predictions=total_predictions,
                    failure_type="true_positive",
                    gt=gt_objects[best_gt_idx],
                    pred=pred,
                    iou=best_iou,
                    best_same_class_iou=best_iou,
                    best_wrong_class_iou=None,
                )

    # False negatives and GT-centered failure explanations.
    for gt_idx, gt in enumerate(gt_objects):
        if gt_idx in matched_gt:
            continue

        best_same_iou, best_same_pred_idx = best_iou_to_pred(
            gt,
            predictions,
            pred_filter=lambda idx, pred: (
                idx not in matched_pred and pred["class_id"] == gt["class_id"]
            ),
        )

        best_wrong_iou, best_wrong_pred_idx = best_iou_to_pred(
            gt,
            predictions,
            pred_filter=lambda idx, pred: (
                idx not in matched_pred and pred["class_id"] != gt["class_id"]
            ),
        )

        write_event(
            writer=writer,
            summary_counter=summary_counter,
            core_counter=core_counter,
            dataset_name=dataset_name,
            detector_name=detector_name,
            image_id=image_id,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            total_gt=total_gt,
            total_predictions=total_predictions,
            failure_type="false_negative",
            gt=gt,
            pred=None,
            iou=None,
            best_same_class_iou=best_same_iou,
            best_wrong_class_iou=best_wrong_iou,
        )

        if LOCALIZATION_IOU_MIN <= best_same_iou < LOCALIZATION_IOU_MAX and best_same_pred_idx is not None:
            write_event(
                writer=writer,
                summary_counter=summary_counter,
                core_counter=core_counter,
                dataset_name=dataset_name,
                detector_name=detector_name,
                image_id=image_id,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                total_gt=total_gt,
                total_predictions=total_predictions,
                failure_type="localization_error",
                gt=gt,
                pred=predictions[best_same_pred_idx],
                iou=best_same_iou,
                best_same_class_iou=best_same_iou,
                best_wrong_class_iou=best_wrong_iou,
            )

        if best_wrong_iou >= CLASS_CONFUSION_IOU_MIN and best_wrong_pred_idx is not None:
            write_event(
                writer=writer,
                summary_counter=summary_counter,
                core_counter=core_counter,
                dataset_name=dataset_name,
                detector_name=detector_name,
                image_id=image_id,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                total_gt=total_gt,
                total_predictions=total_predictions,
                failure_type="class_confusion",
                gt=gt,
                pred=predictions[best_wrong_pred_idx],
                iou=best_wrong_iou,
                best_same_class_iou=best_same_iou,
                best_wrong_class_iou=best_wrong_iou,
            )

    # False positives and duplicate detections.
    for pred_idx, pred in enumerate(predictions):
        if pred_idx in matched_pred:
            continue

        best_same_iou, best_same_gt_idx = best_iou_to_gt(
            pred,
            gt_objects,
            gt_filter=lambda idx, gt: gt["class_id"] == pred["class_id"],
        )

        best_any_iou, best_any_gt_idx = best_iou_to_gt(
            pred,
            gt_objects,
            gt_filter=None,
        )

        write_event(
            writer=writer,
            summary_counter=summary_counter,
            core_counter=core_counter,
            dataset_name=dataset_name,
            detector_name=detector_name,
            image_id=image_id,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            total_gt=total_gt,
            total_predictions=total_predictions,
            failure_type="false_positive",
            gt=None,
            pred=pred,
            iou=best_any_iou,
            best_same_class_iou=best_same_iou,
            best_wrong_class_iou=None,
        )

        if best_same_gt_idx is not None and best_same_gt_idx in matched_gt and best_same_iou >= DUPLICATE_IOU_MIN:
            write_event(
                writer=writer,
                summary_counter=summary_counter,
                core_counter=core_counter,
                dataset_name=dataset_name,
                detector_name=detector_name,
                image_id=image_id,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                total_gt=total_gt,
                total_predictions=total_predictions,
                failure_type="duplicate_detection",
                gt=gt_objects[best_same_gt_idx],
                pred=pred,
                iou=best_same_iou,
                best_same_class_iou=best_same_iou,
                best_wrong_class_iou=None,
            )


def write_summary_files(summary_counter, core_counter):
    with ERROR_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "detector",
                "failure_type",
                "analysis_class_name",
                "object_size_bin",
                "risk_level",
                "count",
            ],
        )
        writer.writeheader()

        for key, count in sorted(summary_counter.items()):
            dataset, detector, failure_type, class_name, size_bin, risk = key
            writer.writerow({
                "dataset": dataset,
                "detector": detector,
                "failure_type": failure_type,
                "analysis_class_name": class_name,
                "object_size_bin": size_bin,
                "risk_level": risk,
                "count": count,
            })

    with CORE_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "detector",
                "class_name",
                "tp",
                "fn",
                "fp",
                "gt_objects",
                "predictions",
                "recall",
                "false_negative_rate",
                "precision",
            ],
        )
        writer.writeheader()

        for key, counts in sorted(core_counter.items()):
            dataset, detector, class_name = key

            tp = counts["tp"]
            fn = counts["fn"]
            fp = counts["fp"]

            gt_objects = tp + fn
            predictions = tp + fp

            recall = tp / gt_objects if gt_objects else None
            false_negative_rate = fn / gt_objects if gt_objects else None
            precision = tp / predictions if predictions else None

            writer.writerow({
                "dataset": dataset,
                "detector": detector,
                "class_name": class_name,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "gt_objects": gt_objects,
                "predictions": predictions,
                "recall": round(recall, 6) if recall is not None else "",
                "false_negative_rate": round(false_negative_rate, 6) if false_negative_rate is not None else "",
                "precision": round(precision, 6) if precision is not None else "",
            })


def main():
    print("=" * 100)
    print("STEP 3/10 - Build detection error index")
    print("=" * 100)
    print("Object-size bins: target-box normalized-area quantiles")
    print("Small: normalized_area <=", SMALL_MAX_NORMALIZED_AREA)
    print("Medium:", SMALL_MAX_NORMALIZED_AREA, "< normalized_area <=",       MEDIUM_MAX_NORMALIZED_AREA)
    print("Large: normalized_area >", MEDIUM_MAX_NORMALIZED_AREA)
    print("Target boxes used for bin thresholds:", TARGET_BOXES_USED_FOR_SIZE_BINS)
    print("Primary matching IoU:", PRIMARY_IOU_THRESHOLD)
    print()

    errors = []
    dataset_cache = {}
    prediction_files = {}

    for dataset_name, cfg in DATASETS.items():
        for path_name in ["image_dir", "label_dir", "prediction_dir"]:
            if not cfg[path_name].exists():
                errors.append(f"Missing {dataset_name} {path_name}: {cfg[path_name]}")

        if cfg["image_dir"].exists():
            image_count = len(collect_images(cfg["image_dir"]))
            if image_count != cfg["expected_images"]:
                print(
                    f"WARNING: {dataset_name} expected {cfg['expected_images']} images, "
                    f"found {image_count}"
                )

    for dataset_name in DATASETS:
        for detector in DETECTORS:
            pred_file = find_prediction_file(dataset_name, detector)
            prediction_files[(dataset_name, detector)] = pred_file

            if pred_file is None:
                errors.append(f"Missing prediction file for {dataset_name}/{detector}")
            else:
                print(f"Prediction file: {dataset_name}/{detector} -> {rel(pred_file)}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        print("STEP 3/10 FAILED ❌")
        raise SystemExit(1)

    print()
    print("Loading datasets...")

    for dataset_name, cfg in DATASETS.items():
        dataset_cache[dataset_name] = load_ground_truth(dataset_name, cfg)

    summary_counter = Counter()
    core_counter = defaultdict(Counter)

    total_rows_written = 0
    prediction_stats = {}

    with ERROR_INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for dataset_name in DATASETS:
            gt_by_image = dataset_cache[dataset_name]["gt_by_image"]
            image_meta = dataset_cache[dataset_name]["image_meta"]
            image_files = dataset_cache[dataset_name]["image_files"]

            image_ids = [p.stem for p in image_files]

            for detector in DETECTORS:
                print()
                print("-" * 100)
                print(f"Processing {dataset_name} / {detector}")
                print("-" * 100)

                prediction_file = prediction_files[(dataset_name, detector)]
                pred_payload = load_predictions_by_image(prediction_file)
                predictions_by_image = pred_payload["predictions_by_image"]

                prediction_stats[f"{dataset_name}_{detector}"] = {
                    "prediction_file": rel(prediction_file),
                    "total_predictions": pred_payload["total_predictions"],
                    "skipped_predictions": pred_payload["skipped_predictions"],
                    "images_with_prediction_records": len(predictions_by_image),
                }

                print("Prediction records:", len(predictions_by_image))
                print("Total predictions:", pred_payload["total_predictions"])
                print("Skipped predictions:", pred_payload["skipped_predictions"])

                detector_rows_before = sum(summary_counter.values())

                for idx, image_id in enumerate(image_ids, start=1):
                    meta = image_meta[image_id]
                    gt_objects = gt_by_image.get(image_id, [])
                    predictions = predictions_by_image.get(image_id, [])

                    process_image(
                        writer=writer,
                        summary_counter=summary_counter,
                        core_counter=core_counter,
                        dataset_name=dataset_name,
                        detector_name=detector,
                        image_id=image_id,
                        image_path=meta["image_path"],
                        image_width=meta["width"],
                        image_height=meta["height"],
                        gt_objects=gt_objects,
                        predictions=predictions,
                    )

                    if idx % 300 == 0 or idx == len(image_ids):
                        print(f"  {dataset_name}/{detector}: processed {idx}/{len(image_ids)} images")

                detector_rows_after = sum(summary_counter.values())
                detector_rows = detector_rows_after - detector_rows_before
                print(f"Rows added for {dataset_name}/{detector}: {detector_rows}")

    total_rows_written = sum(summary_counter.values())

    write_summary_files(summary_counter, core_counter)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "milestone": 7,
        "step": "Build detection error index",
        "matching_policy": {
            "primary_iou_threshold": PRIMARY_IOU_THRESHOLD,
            "localization_iou_range": [LOCALIZATION_IOU_MIN, LOCALIZATION_IOU_MAX],
            "class_confusion_iou_min": CLASS_CONFUSION_IOU_MIN,
            "duplicate_iou_min": DUPLICATE_IOU_MIN,
            "prediction_confidence_min": PREDICTION_CONFIDENCE_MIN,
        },
        "object_size_policy": {
    		"method": "target_box_normalized_area_quantiles",
    		"definition": "normalized_area = bbox_area / image_area",
    		"small": f"normalized_area <= {SMALL_MAX_NORMALIZED_AREA}",
    		"medium": f"{SMALL_MAX_NORMALIZED_AREA} < normalized_area <= {MEDIUM_MAX_NORMALIZED_AREA}",
    		"large": f"normalized_area > {MEDIUM_MAX_NORMALIZED_AREA}",
    		"target_boxes_used": TARGET_BOXES_USED_FOR_SIZE_BINS,
	},
        "dataset_summary": {
            dataset_name: {
                "images": len(payload["image_files"]),
                "annotations": payload["total_annotations"],
                "class_counts": payload["class_counts"],
                "size_counts": payload["size_counts"],
            }
            for dataset_name, payload in dataset_cache.items()
        },
        "prediction_stats": prediction_stats,
        "total_error_index_rows": total_rows_written,
        "outputs": {
            "detection_error_index_csv": rel(ERROR_INDEX_CSV),
            "detection_error_index_json": rel(ERROR_INDEX_JSON),
            "detection_error_summary_csv": rel(ERROR_SUMMARY_CSV),
            "detection_core_tp_fp_fn_summary_csv": rel(CORE_SUMMARY_CSV),
            "detection_error_index_md": rel(ERROR_INDEX_MD),
        },
        "notes": [
            "The CSV contains the full event-level detection error index.",
            "The JSON is a compact manifest/summary, not a duplicate of the full CSV.",
            "False positives are counted using the same low confidence threshold used for AP-style ranking.",
            "Object size bins follow target-box normalized-area quantiles computed from 39086 target boxes.",
        ],
    }

    ERROR_INDEX_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 7 Detection Error Index")
    md.append("")
    md.append(f"Created at: `{manifest['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append(
        "This artifact matches model predictions against ground-truth objects for KITTI and Waymo, "
        "then labels true positives, false negatives, false positives, localization errors, "
        "class confusion, and duplicate detections."
    )
    md.append("")
    md.append("## Matching Policy")
    md.append("")
    md.append(f"- Primary IoU threshold: `{PRIMARY_IOU_THRESHOLD}`")
    md.append(f"- Localization error IoU range: `{LOCALIZATION_IOU_MIN} <= IoU < {LOCALIZATION_IOU_MAX}`")
    md.append(f"- Class confusion IoU minimum: `{CLASS_CONFUSION_IOU_MIN}`")
    md.append(f"- Duplicate detection IoU minimum: `{DUPLICATE_IOU_MIN}`")
    md.append("")
    md.append("## Object Size Policy")
    md.append("")
    md.append("- Method: `target_box_normalized_area_quantiles`")
    md.append("- Normalized area: `bbox_area / image_area`")
    md.append(f"- Small: `normalized_area <= {SMALL_MAX_NORMALIZED_AREA}`")
    md.append(f"- Medium: `{SMALL_MAX_NORMALIZED_AREA} < normalized_area <= {MEDIUM_MAX_NORMALIZED_AREA}`")
    md.append(f"- Large: `normalized_area > {MEDIUM_MAX_NORMALIZED_AREA}`")
    md.append(f"- Target boxes used: `{TARGET_BOXES_USED_FOR_SIZE_BINS}`")
    md.append("")
    md.append("## Dataset Summary")
    md.append("")
    md.append("| Dataset | Images | Annotations | Vehicle | Pedestrian | Cyclist | Small | Medium | Large |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset_name, payload in dataset_cache.items():
        class_counts = payload["class_counts"]
        size_counts = payload["size_counts"]
        md.append(
            f"| {dataset_name} | {len(payload['image_files'])} | {payload['total_annotations']} | "
            f"{class_counts.get('Vehicle', 0)} | {class_counts.get('Pedestrian', 0)} | {class_counts.get('Cyclist', 0)} | "
            f"{size_counts.get('small', 0)} | {size_counts.get('medium', 0)} | {size_counts.get('large', 0)} |"
        )
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append(f"- `{rel(ERROR_INDEX_CSV)}`")
    md.append(f"- `{rel(ERROR_INDEX_JSON)}`")
    md.append(f"- `{rel(ERROR_SUMMARY_CSV)}`")
    md.append(f"- `{rel(CORE_SUMMARY_CSV)}`")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- The full event-level index is stored in CSV format.")
    md.append("- The JSON file stores the compact manifest and summary.")
    md.append("- Later Milestone 7 scripts use this index to produce object-size, safety, and failure-case analyses.")
    md.append("")

    ERROR_INDEX_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Detection error index created")
    print("=" * 100)
    print("Rows written:", total_rows_written)
    print("Created:", ERROR_INDEX_CSV)
    print("Created:", ERROR_INDEX_JSON)
    print("Created:", ERROR_SUMMARY_CSV)
    print("Created:", CORE_SUMMARY_CSV)
    print("Created:", ERROR_INDEX_MD)

    print()
    print("Dataset object-size distribution:")
    for dataset_name, payload in dataset_cache.items():
        print(dataset_name, payload["size_counts"])

    print()
    print("STEP 3/10 COMPLETE ✅")
    print("Detection error index is ready for object-size and safety-oriented analysis.")
    print("=" * 100)


if __name__ == "__main__":
    main()