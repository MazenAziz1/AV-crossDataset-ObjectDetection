# Milestone 7 Safety-Oriented False Negative Analysis

Created at: `2026-08-15T12:41:41`

Status: **PASSED**

## Purpose

This analysis focuses on false negatives for pedestrians and cyclists, because missed vulnerable road users are safety-critical for intelligent-vehicle perception.

## Processing Summary

- Rows read: `1401946`
- GT-centered safety events used: `36844`
- Safety false-negative rows: `26558`
- Ignored rows: `1365102`

## Waymo Vulnerable Road User False Negative Rate, All Sizes

| Detector | TP | FN | GT Objects | Recall | FNR |
|---|---:|---:|---:|---:|---:|
| yolo | 782 | 7109 | 7891 | 0.0991 | 0.9009 |
| faster_rcnn | 1144 | 6747 | 7891 | 0.144975 | 0.855025 |
| retinanet | 1317 | 6574 | 7891 | 0.166899 | 0.833101 |
| rtdetr | 2071 | 5820 | 7891 | 0.262451 | 0.737549 |

## Waymo Small Vulnerable Road User False Negative Rate

| Detector | TP | FN | GT Objects | Recall | FNR |
|---|---:|---:|---:|---:|---:|
| yolo | 435 | 6995 | 7430 | 0.058546 | 0.941454 |
| faster_rcnn | 804 | 6626 | 7430 | 0.10821 | 0.89179 |
| retinanet | 947 | 6483 | 7430 | 0.127456 | 0.872544 |
| rtdetr | 1673 | 5757 | 7430 | 0.225168 | 0.774832 |

## Top Safety-Critical Images

| Dataset | Detector | Image ID | Missed Pedestrians | Missed Cyclists | Small Misses | Score |
|---|---|---|---:|---:|---:|---:|
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857366637459 | 36 | 2 | 38 | 57.5 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857363137470 | 35 | 2 | 37 | 56.0 |
| waymo | faster_rcnn | 8506432817378693815_4860_000_4880_000_1557857366637459 | 35 | 2 | 37 | 56.0 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857358135995 | 36 | 1 | 37 | 55.75 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857358636513 | 36 | 1 | 37 | 55.75 |
| waymo | retinanet | 8506432817378693815_4860_000_4880_000_1557857358636513 | 36 | 1 | 37 | 55.75 |
| waymo | faster_rcnn | 8506432817378693815_4860_000_4880_000_1557857358636513 | 36 | 1 | 37 | 55.75 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857366137460 | 34 | 2 | 36 | 54.5 |
| waymo | retinanet | 8506432817378693815_4860_000_4880_000_1557857366137460 | 34 | 2 | 36 | 54.5 |
| waymo | faster_rcnn | 8506432817378693815_4860_000_4880_000_1557857366137460 | 34 | 2 | 36 | 54.5 |
| waymo | faster_rcnn | 8506432817378693815_4860_000_4880_000_1557857358135995 | 35 | 1 | 36 | 54.25 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857362637395 | 33 | 2 | 35 | 53.0 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857365637432 | 33 | 2 | 35 | 53.0 |
| waymo | retinanet | 8506432817378693815_4860_000_4880_000_1557857366637459 | 33 | 2 | 35 | 53.0 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857356634947 | 34 | 1 | 35 | 52.75 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857357635506 | 34 | 1 | 35 | 52.75 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857359136948 | 34 | 1 | 35 | 52.75 |
| waymo | yolo | 8506432817378693815_4860_000_4880_000_1557857359637183 | 34 | 1 | 35 | 52.75 |
| waymo | retinanet | 8506432817378693815_4860_000_4880_000_1557857357635506 | 34 | 1 | 35 | 52.75 |
| waymo | retinanet | 8506432817378693815_4860_000_4880_000_1557857358135995 | 34 | 1 | 35 | 52.75 |

## Outputs

- `outputs\milestone_7\safety_error_analysis\safety_false_negative_summary.csv`
- `outputs\milestone_7\safety_error_analysis\top_safety_critical_images.csv`
- `outputs\milestone_7\safety_error_analysis\safety_dataset_comparison.csv`
- `outputs\milestone_7\safety_error_analysis\safety_false_negative_analysis_summary.json`
