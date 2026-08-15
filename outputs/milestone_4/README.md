# Milestone 4: Detector Training (Kaggle)

## Overview

All four detectors — YOLOv8s, RT-DETR-L, RetinaNet, and Faster R-CNN — were trained on KITTI
(5,985 train images) on Kaggle GPU (Tesla T4) under a frozen experimental protocol. Training
outputs were imported locally and the final checkpoints locked with SHA-256 hashes.

In-domain (KITTI) evaluation and benchmarking results live in
[`outputs/milestone_5/README.md`](../milestone_5/README.md).

## Training Notes

- **YOLOv8s**: completed all 200 epochs in 6.33 hours (single Kaggle session).
- **RT-DETR-L**: 200-epoch schedule, early-stopped at epoch 131 (patience 20, best at epoch 111)
  across a multi-session resume chain; final session covered 29 epochs in 4.73 hours.
- **RetinaNet**: 200-epoch schedule, early-stopped at epoch 101 (patience 20, best at epoch 81)
  across a multi-session resume chain; final session covered epochs 64-101 in ~10.0 hours.
- **Faster R-CNN**: 200-epoch schedule, early-stopped at epoch 96 (patience 20) across a
  multi-session resume chain; final session covered epochs 77-96.

## Artifacts

- Checkpoints: `outputs/milestone_4/checkpoints/{yolo,rtdetr,retinanet,faster_rcnn}/final/best.pt`
- Training-val metrics: `outputs/milestone_4/metrics/training_metrics.json`
- Locked checkpoint registry: `outputs/milestone_4/manifests/final_checkpoint_registry.csv`
- Imported checkpoints manifest: `outputs/milestone_4/manifests/imported_checkpoints.csv`
- Session manifests: `outputs/milestone_4/manifests/kaggle_session_manifest.csv`
- Reports: `outputs/milestone_4/reports/`

## Evaluation (Milestone 5)

Final KITTI in-domain evaluation metrics, benchmarks, and comparison tables are under
`outputs/milestone_5/`.
