# Milestone 7 Object-Size Robustness Analysis

Created at: `2026-08-15T12:39:34`

Status: **PASSED**

## Object Size Policy

- Method: `target_box_normalized_area_quantiles`
- Normalized area: `bbox_area / image_area`
- Small: `normalized_area <= 0.004157071390230814`
- Medium: `0.004157071390230814 < normalized_area <= 0.015589650670960799`
- Large: `normalized_area > 0.015589650670960799`
- Target boxes used to define thresholds: `39086`

## Processing Summary

- Rows read from detection error index: `1401946`
- GT-centered TP/FN events used: `130444`
- Non-object-size rows ignored: `1271502`

## Best Small-Object Recall by Dataset, All Classes

| Dataset | Detector | Recall | False Negative Rate | GT Objects |
|---|---|---:|---:|---:|
| kitti | rtdetr | 0.993174 | 0.006826 | 4981 |
| waymo | rtdetr | 0.217936 | 0.782064 | 21933 |

## Worst Waymo Small Vulnerable-Road-User Rows

| Detector | Class | Recall | False Negative Rate | GT Objects | TP | FN |
|---|---|---:|---:|---:|---:|---:|
| yolo | Pedestrian | 0.05633 | 0.94367 | 6746 | 380 | 6366 |
| yolo | Vulnerable_Road_Users | 0.058546 | 0.941454 | 7430 | 435 | 6995 |
| yolo | Cyclist | 0.080409 | 0.919591 | 684 | 55 | 629 |
| faster_rcnn | Cyclist | 0.100877 | 0.899123 | 684 | 69 | 615 |
| faster_rcnn | Vulnerable_Road_Users | 0.10821 | 0.89179 | 7430 | 804 | 6626 |
| faster_rcnn | Pedestrian | 0.108953 | 0.891047 | 6746 | 735 | 6011 |
| retinanet | Cyclist | 0.119883 | 0.880117 | 684 | 82 | 602 |
| retinanet | Vulnerable_Road_Users | 0.127456 | 0.872544 | 7430 | 947 | 6483 |
| retinanet | Pedestrian | 0.128224 | 0.871776 | 6746 | 865 | 5881 |
| rtdetr | Pedestrian | 0.223095 | 0.776905 | 6746 | 1505 | 5241 |

## Outputs

- `outputs\milestone_7\object_size_analysis\object_size_summary.csv`
- `outputs\milestone_7\object_size_analysis\small_object_failure_summary.csv`
- `outputs\milestone_7\object_size_analysis\object_size_dataset_comparison.csv`
- `outputs\milestone_7\object_size_analysis\object_size_analysis_summary.json`
