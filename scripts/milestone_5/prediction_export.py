"""Deterministic prediction export for Milestone 7 robustness analysis.

This module serializes the exact predictions produced by the Milestone 5 (KITTI)
and Milestone 6 (Waymo) evaluators to JSONL so they can be reused by the
Milestone 7 failure-case analysis without re-running inference.

The serializer is read-only: it never mutates the prediction list, so evaluation
metrics are unchanged. Predictions are exported after ignore-region suppression,
i.e. they are identical to the set of detections the evaluator consumes.
"""

import json
from pathlib import Path

# Frozen three-class contract (COCO category ids).
CLASS_NAMES = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}


def _to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def save_predictions_jsonl(predictions, image_id_to_file, detector, dataset, out_path):
    """Write predictions to a deterministic JSONL file.

    Args:
        predictions: list of dicts with keys image_id, category_id, bbox [x,y,w,h], score.
        image_id_to_file: dict mapping COCO image id -> file_name.
        detector: detector key (yolo/rtdetr/retinanet/faster_rcnn).
        dataset: dataset key (kitti/waymo).
        out_path: destination .jsonl path.

    Returns:
        number of records written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in predictions:
            image_id = int(p["image_id"])
            cat_id = int(p["category_id"])
            bbox_xywh = [float(v) for v in p["bbox"]]

            record = {
                "dataset": dataset,
                "detector": detector,
                "image_id": image_id,
                "file_name": image_id_to_file.get(image_id, ""),
                "category_id": cat_id,
                "class_name": CLASS_NAMES.get(cat_id, "unknown"),
                "confidence": float(p["score"]),
                "bbox_xywh": bbox_xywh,
                "bbox_xyxy": _to_xyxy(bbox_xywh),
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1

    return written
