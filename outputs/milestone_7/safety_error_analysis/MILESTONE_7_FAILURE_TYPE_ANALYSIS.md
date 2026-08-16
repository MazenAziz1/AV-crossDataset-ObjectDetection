# Milestone 7 Failure Type Analysis

Created at: `2026-08-15T12:44:33`

Status: **PASSED**

## Purpose

This analysis summarizes false positives, localization errors, class confusion, and duplicate detections using the Milestone 7 detection error index.

## Processing Summary

- Rows read: `1401946`
- True positive rows: `50385`
- Non-true-positive failure rows: `1351561`
- High-confidence threshold for false positives: `0.25`

## Failure Type Totals

| Dataset | Detector | Failure Type | Count |
|---|---|---|---:|
| kitti | yolo | false_negative | 96 |
| kitti | yolo | false_positive | 18558 |
| kitti | yolo | localization_error | 61 |
| kitti | yolo | class_confusion | 9 |
| kitti | yolo | duplicate_detection | 7253 |
| kitti | rtdetr | false_negative | 39 |
| kitti | rtdetr | false_positive | 441047 |
| kitti | rtdetr | localization_error | 33 |
| kitti | rtdetr | class_confusion | 27 |
| kitti | rtdetr | duplicate_detection | 20400 |
| kitti | retinanet | false_negative | 309 |
| kitti | retinanet | false_positive | 411624 |
| kitti | retinanet | localization_error | 266 |
| kitti | retinanet | class_confusion | 203 |
| kitti | retinanet | duplicate_detection | 1804 |
| kitti | faster_rcnn | false_negative | 279 |
| kitti | faster_rcnn | false_positive | 29225 |
| kitti | faster_rcnn | localization_error | 199 |
| kitti | faster_rcnn | class_confusion | 114 |
| kitti | faster_rcnn | duplicate_detection | 958 |
| waymo | yolo | false_negative | 21298 |
| waymo | yolo | false_positive | 6561 |
| waymo | yolo | localization_error | 888 |
| waymo | yolo | class_confusion | 159 |
| waymo | yolo | duplicate_detection | 2457 |
| waymo | rtdetr | false_negative | 17687 |
| waymo | rtdetr | false_positive | 291668 |
| waymo | rtdetr | localization_error | 8030 |
| waymo | rtdetr | class_confusion | 1774 |
| waymo | rtdetr | duplicate_detection | 13185 |
| waymo | retinanet | false_negative | 19779 |
| waymo | retinanet | false_positive | 3851 |
| waymo | retinanet | localization_error | 1660 |
| waymo | retinanet | class_confusion | 241 |
| waymo | retinanet | duplicate_detection | 260 |
| waymo | faster_rcnn | false_negative | 20572 |
| waymo | faster_rcnn | false_positive | 6666 |
| waymo | faster_rcnn | localization_error | 1765 |
| waymo | faster_rcnn | class_confusion | 319 |
| waymo | faster_rcnn | duplicate_detection | 237 |

## Top Class Confusions

| Dataset | Detector | GT Class | Pred Class | Size | Count | Mean IoU | Mean Score |
|---|---|---|---|---|---:|---:|---:|
| waymo | rtdetr | Vehicle | Pedestrian | small | 631 | 0.395988 | 0.01037318 |
| waymo | rtdetr | Pedestrian | Cyclist | small | 422 | 0.392849 | 0.02701849 |
| waymo | rtdetr | Pedestrian | Vehicle | small | 277 | 0.404672 | 0.01256005 |
| waymo | rtdetr | Vehicle | Cyclist | small | 205 | 0.397294 | 0.01036747 |
| waymo | faster_rcnn | Pedestrian | Cyclist | small | 124 | 0.450443 | 0.37733559 |
| waymo | rtdetr | Cyclist | Pedestrian | small | 107 | 0.416214 | 0.02648006 |
| waymo | retinanet | Pedestrian | Cyclist | small | 80 | 0.466301 | 0.32383507 |
| waymo | faster_rcnn | Vehicle | Pedestrian | small | 71 | 0.379834 | 0.2793508 |
| waymo | faster_rcnn | Cyclist | Pedestrian | small | 66 | 0.472971 | 0.31832112 |
| waymo | yolo | Vehicle | Pedestrian | small | 55 | 0.441466 | 0.00466163 |
| waymo | retinanet | Vehicle | Pedestrian | small | 50 | 0.423215 | 0.17200716 |
| waymo | rtdetr | Vehicle | Cyclist | medium | 46 | 0.402983 | 0.00918763 |

## Top Localization Error Groups

| Dataset | Detector | Class | Size | Count | Mean IoU | Mean Score |
|---|---|---|---|---:|---:|---:|
| waymo | rtdetr | Vehicle | small | 5099 | 0.257989 | 0.0257363 |
| waymo | rtdetr | Pedestrian | small | 2336 | 0.275697 | 0.02196548 |
| waymo | retinanet | Vehicle | small | 878 | 0.287099 | 0.42169265 |
| waymo | faster_rcnn | Pedestrian | small | 747 | 0.270552 | 0.42861587 |
| waymo | faster_rcnn | Vehicle | small | 712 | 0.283832 | 0.50851312 |
| waymo | retinanet | Pedestrian | small | 553 | 0.302599 | 0.35609439 |
| waymo | yolo | Vehicle | small | 417 | 0.278134 | 0.07217 |
| waymo | rtdetr | Vehicle | medium | 294 | 0.309262 | 0.02078946 |
| waymo | yolo | Pedestrian | small | 289 | 0.268248 | 0.01274702 |
| waymo | rtdetr | Cyclist | small | 178 | 0.294557 | 0.02996739 |
| waymo | retinanet | Vehicle | medium | 115 | 0.328869 | 0.42330241 |
| kitti | retinanet | Vehicle | small | 107 | 0.388727 | 0.10980915 |

## Top False Positive Groups

| Dataset | Detector | Pred Class | Count | High-Confidence Count | Mean Score |
|---|---|---|---:|---:|---:|
| kitti | retinanet | Vehicle | 316615 | 852 | 0.00591063 |
| kitti | rtdetr | Vehicle | 288025 | 2569 | 0.01536476 |
| waymo | rtdetr | Vehicle | 162755 | 539 | 0.01194197 |
| kitti | rtdetr | Pedestrian | 101941 | 879 | 0.01574145 |
| waymo | rtdetr | Pedestrian | 74632 | 391 | 0.01560204 |
| kitti | retinanet | Pedestrian | 67298 | 331 | 0.00848547 |
| waymo | rtdetr | Cyclist | 54281 | 92 | 0.01161264 |
| kitti | rtdetr | Cyclist | 51081 | 210 | 0.01526746 |
| kitti | retinanet | Cyclist | 27711 | 115 | 0.00739373 |
| kitti | faster_rcnn | Pedestrian | 10359 | 667 | 0.05682064 |
| kitti | faster_rcnn | Vehicle | 9915 | 1222 | 0.1058491 |
| kitti | yolo | Vehicle | 9210 | 495 | 0.04528711 |

## Outputs

- `outputs\milestone_7\safety_error_analysis\failure_type_summary.csv`
- `outputs\milestone_7\safety_error_analysis\false_positive_summary.csv`
- `outputs\milestone_7\safety_error_analysis\localization_error_summary.csv`
- `outputs\milestone_7\safety_error_analysis\class_confusion_summary.csv`
- `outputs\milestone_7\safety_error_analysis\duplicate_detection_summary.csv`
- `outputs\milestone_7\safety_error_analysis\top_failure_type_images.csv`
- `outputs\milestone_7\safety_error_analysis\failure_case_candidate_rows.csv`
- `outputs\milestone_7\safety_error_analysis\MILESTONE_7_FAILURE_TYPE_ANALYSIS.md`
