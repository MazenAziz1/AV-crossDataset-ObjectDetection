# Milestone 7 Deployment Suitability and Trade-off Analysis

Created at: `2026-08-15T12:55:21`

Status: **PASSED**

## Purpose

This analysis compares detector suitability using KITTI accuracy, Waymo external-domain accuracy, generalization ratio, inference speed, vulnerable-road-user false-negative rate, small-object safety, and failure-case evidence.

## Scoring Weights

- `waymo_mAP50_95_score`: `0.25`
- `generalization_ratio_score`: `0.2`
- `vru_safety_score`: `0.3`
- `small_vru_safety_score`: `0.15`
- `speed_score`: `0.1`

## Deployment Ranking

| Rank | Detector | Score | Waymo mAP50-95 | Gen. Ratio | Waymo FPS | VRU FNR | Small VRU FNR | Tier |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | rtdetr | 0.197339 | 0.076835 | 0.121684 | 20.615 | 0.737549 | 0.774832 | research only / not deployment-ready |
| 2 | yolo | 0.1676 | 0.053924 | 0.078036 | 49.935 | 0.9009 | 0.941454 | research only / not deployment-ready |
| 3 | retinanet | 0.15844 | 0.078764 | 0.155011 | 19.254 | 0.833101 | 0.872544 | research only / not deployment-ready |
| 4 | faster_rcnn | 0.127558 | 0.054493 | 0.100982 | 16.985 | 0.855025 | 0.89179 | research only / not deployment-ready |

## Main Practical Interpretation

- Best overall trade-off by this safety-weighted score: `rtdetr`.
- Fastest Waymo inference: `yolo` with `49.935` FPS.
- Best Waymo mAP50-95: `retinanet` with `0.078764`.
- Best generalization ratio: `retinanet` with `0.155011`.
- Lowest Waymo vulnerable-road-user FNR: `rtdetr` with `0.737549`.
- Lowest Waymo small vulnerable-road-user FNR: `rtdetr` with `0.774832`.

## Detector Notes

### rtdetr

- Deployment tier: `research only / not deployment-ready`
- Note: Best vulnerable-road-user recall and strongest safety-oriented result, but slower and has many low-confidence proposals. Best candidate for further robustness improvement.

### yolo

- Deployment tier: `research only / not deployment-ready`
- Note: Fastest external inference, but weakest Waymo vulnerable-road-user safety recall. Useful as a speed baseline, not sufficient alone for safety-critical deployment.

### retinanet

- Deployment tier: `research only / not deployment-ready`
- Note: Strongest Waymo mAP50-95 and generalization ratio, but safety false negatives remain high. Useful as an external-generalization reference.

### faster_rcnn

- Deployment tier: `research only / not deployment-ready`
- Note: Moderate safety behavior but slowest external inference. Less attractive for real-time deployment compared with RT-DETR or YOLO.

## Recommendations

- **Use RT-DETR as the main robustness-improvement candidate.** It gives the best Waymo vulnerable-road-user recall and the strongest safety-oriented score, despite slower speed.
- **Use YOLO as the real-time speed baseline.** It is the fastest model on Waymo, but its vulnerable-road-user false-negative rate is too high for safety-critical use.
- **Use RetinaNet as the external-generalization reference.** It has the strongest Waymo mAP50-95 and generalization ratio, but still misses many vulnerable road users.
- **Do not claim any detector is deployment-ready.** The external-domain safety failure rates, especially small pedestrian/cyclist false negatives, remain high.
- **Improve with domain adaptation, threshold calibration, small-object augmentation, and safety-focused loss/reweighting.** Milestone 6 and Milestone 7 both show that cross-dataset generalization and small vulnerable-road-user detection are the main weaknesses.

## Important Warning

The deployment score is a decision-support summary, not an official benchmark. All detectors still show high external-domain vulnerable-road-user false-negative rates, so the results should be framed as research evidence and not deployment readiness.

## Outputs

- `outputs\milestone_7\deployment_tradeoff\deployment_suitability_summary.csv`
- `outputs\milestone_7\deployment_tradeoff\deployment_tradeoff_ranking.csv`
- `outputs\milestone_7\deployment_tradeoff\deployment_recommendations.json`
- `outputs\milestone_7\deployment_tradeoff\MILESTONE_7_DEPLOYMENT_TRADEOFF_ANALYSIS.md`
