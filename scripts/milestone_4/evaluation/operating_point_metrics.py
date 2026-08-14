import json

from scripts.milestone_4.evaluation.ignore_region_suppression import compute_iou


def compute_operating_point_metrics(gt_json_path, predictions, conf_threshold=0.25, iou_threshold=0.50):
    """
    Compute fixed operating-point metrics (evaluation_protocol.md section 2):
        - Precision, Recall, F1 at confidence >= conf_threshold and IoU >= iou_threshold
        - Detections per image
        - False positives per image

    Predictions are expected to already be DontCare-suppressed.
    Matching follows COCO semantics: per-image, per-category greedy matching of
    confidence-sorted detections against unmatched ground-truth boxes.
    """
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    gt_anns = gt_data["annotations"]
    num_images = len(gt_data["images"])

    dets = [p for p in predictions if p.get("score", 0.0) >= conf_threshold]
    dets = sorted(dets, key=lambda p: -p.get("score", 0.0))

    # Index GT boxes by (image_id, category_id)
    gt_by_image_cat = {}
    for idx, ann in enumerate(gt_anns):
        key = (ann["image_id"], ann["category_id"])
        gt_by_image_cat.setdefault(key, []).append((idx, ann))

    matched_gt_indices = set()
    tp = 0
    fp = 0

    for p in dets:
        candidates = gt_by_image_cat.get((p["image_id"], p["category_id"]), [])
        best_iou = 0.0
        best_idx = -1
        for idx, ann in candidates:
            if idx in matched_gt_indices:
                continue
            iou = compute_iou(p["bbox"], ann["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt_indices.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(gt_anns) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    detections_per_image = len(dets) / num_images if num_images > 0 else 0.0
    fp_per_image = fp / num_images if num_images > 0 else 0.0

    return {
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "detections_per_image": round(detections_per_image, 4),
        "false_positives_per_image": round(fp_per_image, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
