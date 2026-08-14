# Milestone 4 + 5: Detector Training & Evaluation

## Overview

All four detectors — YOLOv8s, RT-DETR-L, RetinaNet, and Faster R-CNN — were trained on KITTI (5,985 train images) and evaluated on the KITTI validation partition (1,496 images) using a frozen experimental protocol. Training ran on Kaggle GPU (Tesla T4); evaluation, benchmarking, and checkpoint locking ran locally.

## Results (KITTI validation, local frozen protocol)

| Metric | YOLOv8s | RT-DETR-L | RetinaNet | Faster R-CNN |
|--------|---------|-----------|-----------|--------------|
| mAP@0.50:0.95 | **0.690** | 0.606 | 0.538 | 0.589 |
| mAP@0.50 | **0.925** | 0.906 | 0.839 | 0.879 |
| mAP@0.75 | **0.780** | 0.679 | 0.577 | 0.667 |
| Vehicle AP@0.50:0.95 | **0.849** | 0.730 | 0.703 | 0.741 |
| Pedestrian AP@0.50:0.95 | **0.530** | 0.490 | 0.392 | 0.432 |
| Cyclist AP@0.50:0.95 | **0.691** | 0.597 | 0.518 | 0.594 |
| FPS (local GPU) | **94.77** | 18.73 | 13.70 | 11.24 |
| Parameters | **11.1M** | 32.8M | 36.4M | 41.3M |
| Checkpoint size | 21.5 MB | 63.15 MB | 278.0 MB | 315.68 MB |

## Training vs Local Validation

Best training-validation metric (during training, drives early stopping / best-checkpoint selection, no DontCare suppression) compared against the local frozen-protocol validation:

| Detector | Train-val mAP@0.50:0.95 | Local-val mAP@0.50:0.95 | Δ | Train-val mAP@0.50 | Local-val mAP@0.50 |
|----------|------------------------:|-------------------------:|--:|-------------------:|-------------------:|
| YOLOv8s | 0.690 | 0.690 | -0.0001 | 0.927 | 0.925 |
| RT-DETR-L | 0.603 | 0.606 | +0.0027 | 0.904 | 0.906 |
| RetinaNet | 0.531 | 0.538 | +0.0064 | 0.841 | 0.839 |
| Faster R-CNN | 0.590 | 0.589 | -0.0013 | 0.870 | 0.879 |

Training-val and local-val agree to within ±0.006 mAP for every detector, confirming the imported checkpoints match their training-time performance (DontCare suppression accounts for the small positive shift in RetinaNet and RT-DETR).

## Operating Point (conf ≥ 0.25, IoU ≥ 0.50)

| Detector | Precision | Recall | F1 | Detections/image | FP/image |
|----------|----------:|-------:|---:|-----------------:|---------:|
| YOLOv8s | **0.926** | **0.947** | **0.936** | 5.330 | **0.396** |
| RT-DETR-L | 0.482 | 0.969 | 0.644 | 10.479 | 5.431 |
| RetinaNet | 0.777 | 0.907 | 0.837 | 6.079 | 1.354 |
| Faster R-CNN | 0.876 | 0.928 | 0.901 | 5.520 | 0.687 |

At the fixed operating point, RT-DETR-L achieves the highest recall but at much lower precision (0.482) — its decoder emits many low-confidence detections (10.5 detections/image, 5.4 false positives/image), so a fixed 0.25 confidence threshold is a poor operating point for it. YOLOv8s dominates precision, recall, and F1.

## Training Notes

- **YOLOv8s**: completed all 200 epochs in 6.33 hours (single Kaggle session).
- **RT-DETR-L**: 200-epoch schedule, early-stopped at epoch 131 (patience 20, best at epoch 111) across a multi-session resume chain; final session covered 29 epochs in 4.73 hours.
- **RetinaNet**: 200-epoch schedule, early-stopped at epoch 101 (patience 20, best at epoch 81) across a multi-session resume chain; final session covered epochs 64-101 in ~10.0 hours.
- **Faster R-CNN**: 200-epoch schedule, early-stopped at epoch 96 (patience 20, best mAP 0.5905 in the pre-resume session) across a multi-session resume chain; final session covered epochs 77-96.
- DontCare suppression at evaluation: 351 detections (YOLOv8s), 2,372 (RT-DETR-L), 326 (RetinaNet), 140 (Faster R-CNN).

## Evaluation Notes

- **Per-model preprocessing**: For the torchvision-based detectors, the training pipeline applied ImageNet normalization through Albumentations in addition to the normalization performed internally by the torchvision detection models. Consequently, evaluation was performed using the same preprocessing configuration employed during training to avoid a train–evaluation distribution mismatch. The Ultralytics-based YOLOv8s and RT-DETR models did not exhibit this issue. (Evidence: evaluating the torchvision models on raw inputs instead drops mAP@0.50:0.95 substantially — e.g., RetinaNet 0.538 → 0.384.)
- **Benchmark timing**: torchvision models perform NMS inside `forward()` during eval, so their inference latency includes postprocessing; Ultralytics numbers are raw forward passes.

## Artifacts

- Checkpoints: `outputs/milestone_4/checkpoints/{yolo,rtdetr,retinanet,faster_rcnn}/final/best.pt`
- Metrics: `outputs/milestone_4/metrics/kitti_validation/{yolo,rtdetr,retinanet,faster_rcnn}_metrics.json`
- Training-val metrics: `outputs/milestone_4/metrics/training_metrics.json`
- Benchmarks: `outputs/milestone_4/benchmarks/{yolo,rtdetr,retinanet,faster_rcnn}_benchmark.json`
- Comparisons: `outputs/milestone_4/figures/{accuracy,efficiency,training_vs_validation}_comparison.csv`
- Manifest: `outputs/milestone_4/manifests/imported_checkpoints.csv`
- Registry: `outputs/milestone_4/manifests/final_checkpoint_registry.csv`
