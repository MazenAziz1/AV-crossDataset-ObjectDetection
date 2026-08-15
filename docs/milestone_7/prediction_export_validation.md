# Milestone 5/6 Prediction Export — Validation Report

This document records how raw per-detection predictions are persisted during the
existing Milestone 5 (KITTI) and Milestone 6 (Waymo) evaluation runs, so that
Milestone 7 can build a prediction-vs-ground-truth error index without re-running
inference.

## What changed

| File | Change |
|------|--------|
| `scripts/milestone_5/prediction_export.py` | New shared JSONL serializer (read-only) |
| `scripts/milestone_5/evaluate.py` | Saves predictions after DontCare suppression; added optional `--limit` |
| `scripts/milestone_6/03_run_waymo_external_validation.py` | Saves predictions after ignore suppression |
| `.gitignore` | Ignores the two prediction directories (JSONL not committed) |

No retraining, no change to model loading, preprocessing, class mapping, thresholds,
or evaluation methodology. The serializer never mutates the prediction list.

## Prediction schema (one JSON object per line, `sort_keys=True`)

| Field | Type | Meaning |
|-------|------|---------|
| `dataset` | str | `kitti` or `waymo` |
| `detector` | str | `yolo` \| `rtdetr` \| `retinanet` \| `faster_rcnn` |
| `image_id` | int | COCO image id |
| `file_name` | str | image file name |
| `category_id` | int | COCO category id: 1=Vehicle, 2=Pedestrian, 3=Cyclist |
| `class_name` | str | Vehicle / Pedestrian / Cyclist |
| `confidence` | float | detection score in [0, 1] |
| `bbox_xywh` | [x,y,w,h] | absolute pixels (COCO convention), 640x640 |
| `bbox_xyxy` | [x1,y1,x2,y2] | absolute pixels (top-left / bottom-right) |

Coordinates are absolute pixels on the 640x640 images. Class ids follow the frozen
Milestone 3 COCO contract. Predictions are exported after ignore-region suppression,
so they are exactly the set the evaluator consumes.

## Output locations

| Dataset | Detector | Path |
|---------|----------|------|
| KITTI | yolo/rtdetr/retinanet/faster_rcnn | `outputs/milestone_5/predictions/kitti_validation/{detector}_predictions.jsonl` |
| Waymo | yolo/rtdetr/retinanet/faster_rcnn | `outputs/milestone_6/waymo_external_validation/predictions/{detector}_waymo_predictions.jsonl` |

## Record counts (equal to committed `num_predictions`)

| Detector | KITTI | Waymo |
|----------|-------|-------|
| yolo | 19,212 | 15,341 |
| rtdetr | 445,663 | 298,032 |
| retinanet | 20,728 | 28,036 |
| faster_rcnn | 9,178 | 9,090 |

## Validation

- All 8 files parse as valid JSONL; every line has valid category id, confidence in
  [0,1], positive-area boxes, a non-empty `file_name`, and an `image_id` present in the
  corresponding COCO ground truth.
- The regenerated aggregate accuracy metrics (mAP@0.5, mAP@0.5:0.95, per-class AP,
  operating point) are **bit-identical** to the committed Milestone 5 and Milestone 6
  results for all four detectors on both datasets.
- Only non-accuracy fields differ across runs and are therefore not re-committed:
  the M5 `checkpoint` path separator (cosmetic) and the M6 `mean/median_inference_ms`
  (wall-clock timing). The committed result files were left unchanged.

## Git safety

The prediction JSONL files live under git-ignored directories and are not committed.
