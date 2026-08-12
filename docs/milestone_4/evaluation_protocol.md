# KITTI Evaluation Protocol — Milestone 4 + 5

## Purpose

This document defines the internal validation metrics used to compare all four detectors on KITTI validation.

## Final Evaluation Partition

| Item | Value |
|---|---|
| Partition | KITTI validation |
| Expected images | 1,496 |
| Image size | 640 x 640 |
| Classes | Vehicle, Pedestrian, Cyclist |

## Required Metrics

- COCO mAP@[0.50:0.95]
- AP50
- AP75
- Vehicle AP
- Pedestrian AP
- Cyclist AP
- Precision
- Recall
- F1 score
- Small-object AP
- Medium-object AP
- Large-object AP
- Detections per image
- False positives per image
- DontCare suppression count
- Confusion summary
- Precision-recall curve data

## Confidence Policy

AP and PR-curve computation use a low confidence threshold to preserve the full curve. Precision, recall, F1, and visual examples use a fixed common operating threshold of 0.25.

## Ignore Region Policy

KITTI DontCare regions may suppress eligible false positives. Excluded non-target objects such as Tram and Misc do not automatically suppress false positives.

## Unified Evaluation Command

    python -m scripts.milestone_4.evaluate --detector all --partition kitti_val --checkpoint-registry outputs\milestone_4\manifests\final_checkpoint_registry.csv --evaluation-policy configs\models\milestone_4\evaluation_policy.yaml --output-dir outputs\milestone_4

## Completion Gate

All four detectors must be evaluated using the same command-line module and the same metric engine.
