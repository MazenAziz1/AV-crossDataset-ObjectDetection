# Milestone 7 Detection Error Index

Created at: `2026-08-15T12:37:34`

Status: **PASSED**

## Purpose

This artifact matches model predictions against ground-truth objects for KITTI and Waymo, then labels true positives, false negatives, false positives, localization errors, class confusion, and duplicate detections.

## Matching Policy

- Primary IoU threshold: `0.5`
- Localization error IoU range: `0.1 <= IoU < 0.5`
- Class confusion IoU minimum: `0.3`
- Duplicate detection IoU minimum: `0.5`

## Object Size Policy

- Method: `target_box_normalized_area_quantiles`
- Normalized area: `bbox_area / image_area`
- Small: `normalized_area <= 0.004157071390230814`
- Medium: `0.004157071390230814 < normalized_area <= 0.015589650670960799`
- Large: `normalized_area > 0.015589650670960799`
- Target boxes used: `39086`

## Dataset Summary

| Dataset | Images | Annotations | Vehicle | Pedestrian | Cyclist | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kitti | 1496 | 7792 | 6472 | 980 | 340 | 4981 | 1740 | 1071 |
| waymo | 996 | 24819 | 16928 | 7127 | 764 | 21933 | 1978 | 908 |

## Outputs

- `outputs\milestone_7\safety_error_analysis\detection_error_index.csv`
- `outputs\milestone_7\safety_error_analysis\detection_error_index.json`
- `outputs\milestone_7\safety_error_analysis\detection_error_summary.csv`
- `outputs\milestone_7\safety_error_analysis\detection_core_tp_fp_fn_summary.csv`

## Notes

- The full event-level index is stored in CSV format.
- The JSON file stores the compact manifest and summary.
- Later Milestone 7 scripts use this index to produce object-size, safety, and failure-case analyses.
