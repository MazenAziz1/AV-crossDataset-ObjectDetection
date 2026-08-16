# Milestone 7 Input Audit

Created at: `2026-08-15T09:02:12`

Status: **PASSED**

## Dataset Summary

| Dataset | Images | Labels | Matched Pairs | Annotations | Vehicle | Pedestrian | Cyclist |
|---|---:|---:|---:|---:|---:|---:|---:|
| KITTI val | 1496 | 1496 | 1496 | 7792 | 6472 | 980 | 340 |
| Waymo external | 996 | 996 | 996 | 24819 | 16928 | 7127 | 764 |

## Prediction Inventory

| Dataset | Detector | Exists | Records | Size MB | Path |
|---|---|---:|---:|---:|---|
| kitti | yolo | True | 1496 | 3.43 | outputs\milestone_5\final_kitti_validation\predictions\yolo_predictions_full.jsonl |
| kitti | rtdetr | True | 1496 | 57.045 | outputs\milestone_5\final_kitti_validation\predictions\rtdetr_predictions_full.jsonl |
| kitti | retinanet | True | 1496 | 50.582 | outputs\milestone_5\final_kitti_validation\predictions\retinanet_predictions_full.jsonl |
| kitti | faster_rcnn | True | 1496 | 4.772 | outputs\milestone_5\final_kitti_validation\predictions\faster_rcnn_predictions_full.jsonl |
| waymo | yolo | True | 996 | 3.648 | outputs\milestone_6\waymo_external_validation\predictions\yolo_waymo_predictions.jsonl |
| waymo | rtdetr | True | 996 | 103.168 | outputs\milestone_6\waymo_external_validation\predictions\rtdetr_waymo_predictions.jsonl |
| waymo | retinanet | True | 996 | 3.245 | outputs\milestone_6\waymo_external_validation\predictions\retinanet_waymo_predictions.jsonl |
| waymo | faster_rcnn | True | 996 | 3.929 | outputs\milestone_6\waymo_external_validation\predictions\faster_rcnn_waymo_predictions.jsonl |

## Errors

- None

## Warnings

- None

## Recovery Commands If Prediction JSONL Files Are Missing

```cmd
python scripts\milestone_5\03_run_final_kitti_validation.py
python scripts\milestone_6\03_run_waymo_external_validation.py
```
