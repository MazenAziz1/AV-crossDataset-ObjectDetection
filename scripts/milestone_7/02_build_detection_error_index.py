import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common
from scripts.milestone_5.evaluation.ignore_region_suppression import compute_iou

TP_IOU = 0.50
LOC_IOU_LOW = 0.10
CONF = 0.25

CSV_COLUMNS = [
    "dataset", "detector", "image_id", "file_name", "record_type", "error_type",
    "category_id", "class_name", "confidence", "iou",
    "gt_category_id", "gt_class_name",
    "x1", "y1", "x2", "y2", "area", "size_category",
]


def _i2f(v):
    return "" if v is None else v


def build_index(dataset, detector, bins):
    gt = common.load_coco(common.gt_path(dataset))
    id_to_file = {img["id"]: img["file_name"] for img in gt["images"]}

    gt_by_image = {}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    preds = common.load_predictions(common.pred_path(dataset, detector), CONF)
    preds.sort(key=lambda p: -p["confidence"])
    pred_by_image = {}
    for p in preds:
        pred_by_image.setdefault(p["image_id"], []).append(p)

    rows = []
    all_image_ids = set(gt_by_image) | set(pred_by_image)

    for image_id in all_image_ids:
        fname = id_to_file.get(image_id, "")
        gts = gt_by_image.get(image_id, [])
        im_preds = pred_by_image.get(image_id, [])

        gt_matched = [False] * len(gts)
        pred_matched = [False] * len(im_preds)

        # 1) greedy TP matching (confidence-descending)
        for pi, p in enumerate(im_preds):
            best_i = -1
            best_iou = 0.0
            for gi, a in enumerate(gts):
                if gt_matched[gi]:
                    continue
                if a["category_id"] != p["category_id"]:
                    continue
                iou = compute_iou(p["bbox_xywh"], a["bbox"])
                if iou >= TP_IOU and iou > best_iou:
                    best_iou = iou
                    best_i = gi
            if best_i >= 0:
                gt_matched[best_i] = True
                pred_matched[pi] = True
                a = gts[best_i]
                x, y, w, h = a["bbox"]
                rows.append({
                    "dataset": dataset, "detector": detector, "image_id": image_id, "file_name": fname,
                    "record_type": "detection", "error_type": "true_positive",
                    "category_id": p["category_id"], "class_name": common.CLASS_NAMES[p["category_id"]],
                    "confidence": p["confidence"], "iou": best_iou,
                    "gt_category_id": a["category_id"], "gt_class_name": common.CLASS_NAMES[a["category_id"]],
                    "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                    "area": a["area"], "size_category": common.size_category(a["area"], bins),
                })

        # 2) classify unmatched predictions
        for pi, p in enumerate(im_preds):
            if pred_matched[pi]:
                continue
            iou_same = 0.0
            iou_diff = 0.0
            diff_gt_cat = None
            for gi, a in enumerate(gts):
                iou = compute_iou(p["bbox_xywh"], a["bbox"])
                if a["category_id"] == p["category_id"]:
                    iou_same = max(iou_same, iou)
                else:
                    if iou > iou_diff:
                        iou_diff = iou
                        diff_gt_cat = a["category_id"]

            det_area = p["bbox_xywh"][2] * p["bbox_xywh"][3]
            if iou_same >= TP_IOU:
                err, iou_val, gt_cat = "over_detection", iou_same, p["category_id"]
            elif iou_same >= LOC_IOU_LOW:
                err, iou_val, gt_cat = "localization_error", iou_same, p["category_id"]
            elif iou_diff >= LOC_IOU_LOW:
                err, iou_val, gt_cat = "class_confusion", iou_diff, diff_gt_cat
            else:
                err, iou_val, gt_cat = "false_positive", 0.0, None

            x1, y1, x2, y2 = p["bbox_xyxy"]
            rows.append({
                "dataset": dataset, "detector": detector, "image_id": image_id, "file_name": fname,
                "record_type": "detection", "error_type": err,
                "category_id": p["category_id"], "class_name": common.CLASS_NAMES[p["category_id"]],
                "confidence": p["confidence"], "iou": iou_val,
                "gt_category_id": gt_cat, "gt_class_name": common.CLASS_NAMES.get(gt_cat, "") if gt_cat else "",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": det_area, "size_category": common.size_category(det_area, bins),
            })

        # 3) missed GT (false negatives)
        for gi, a in enumerate(gts):
            if gt_matched[gi]:
                continue
            x, y, w, h = a["bbox"]
            rows.append({
                "dataset": dataset, "detector": detector, "image_id": image_id, "file_name": fname,
                "record_type": "miss", "error_type": "false_negative",
                "category_id": a["category_id"], "class_name": common.CLASS_NAMES[a["category_id"]],
                "confidence": None, "iou": 0.0,
                "gt_category_id": a["category_id"], "gt_class_name": common.CLASS_NAMES[a["category_id"]],
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "area": a["area"], "size_category": common.size_category(a["area"], bins),
            })

    return rows


def main():
    print("=" * 79)
    print("Milestone 7 - Step 2: Build detection error index")
    print("=" * 79)

    bins = common.load_bins()
    out_dir = common.M7_OUT / "safety_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset in common.DATASETS:
        for detector in common.DETECTORS:
            print(f"[{dataset}/{detector}] matching ...")
            rows = build_index(dataset, detector, bins)
            all_rows.extend(rows)
            print(f"  -> {len(rows)} records")

    csv_path = out_dir / "detection_error_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: _i2f(r[k]) for k in CSV_COLUMNS})

    summary = {
        "milestone": 7,
        "step": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "confidence_threshold": CONF,
        "tp_iou_threshold": TP_IOU,
        "localization_iou_low": LOC_IOU_LOW,
        "total_records": len(all_rows),
        "counts_by_error_type": dict(Counter(r["error_type"] for r in all_rows)),
        "counts_by_dataset_detector_error": {},
    }
    for dataset in common.DATASETS:
        for detector in common.DETECTORS:
            key = f"{dataset}__{detector}"
            sub = [r for r in all_rows if r["dataset"] == dataset and r["detector"] == detector]
            summary["counts_by_dataset_detector_error"][key] = dict(Counter(r["error_type"] for r in sub))

    json_path = out_dir / "detection_error_index.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nTotal records: {len(all_rows)}")
    print("Error-type counts:", dict(summary["counts_by_error_type"]))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
