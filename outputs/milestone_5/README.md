# Milestone 5: KITTI In-Domain Evaluation & Benchmarking

## Overview

The four detectors — YOLOv8s, RT-DETR-L, RetinaNet, and Faster R-CNN — were evaluated on the
KITTI validation partition (1,496 images) under a frozen evaluation protocol, using the locked
checkpoints produced in Milestone 4. This directory holds the in-domain results that serve as
the KITTI baseline for the Milestone 6 cross-dataset (Waymo) analysis.

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

Best training-validation metric (during training, drives early stopping / best-checkpoint
selection, no DontCare suppression) compared against the local frozen-protocol validation:

| Detector | Train-val mAP@0.50:0.95 | Local-val mAP@0.50:0.95 | Δ | Train-val mAP@0.50 | Local-val mAP@0.50 |
|----------|------------------------:|-------------------------:|--:|-------------------:|-------------------:|
| YOLOv8s | 0.690 | 0.690 | -0.0001 | 0.927 | 0.925 |
| RT-DETR-L | 0.603 | 0.606 | +0.0027 | 0.904 | 0.906 |
| RetinaNet | 0.531 | 0.538 | +0.0064 | 0.841 | 0.839 |
| Faster R-CNN | 0.590 | 0.589 | -0.0013 | 0.870 | 0.879 |

## Operating Point (conf ≥ 0.25, IoU ≥ 0.50)

| Detector | Precision | Recall | F1 | Detections/image | FP/image |
|----------|----------:|-------:|---:|-----------------:|---------:|
| YOLOv8s | **0.926** | **0.947** | **0.936** | 5.330 | **0.396** |
| RT-DETR-L | 0.482 | 0.969 | 0.644 | 10.479 | 5.431 |
| RetinaNet | 0.777 | 0.907 | 0.837 | 6.079 | 1.354 |
| Faster R-CNN | 0.876 | 0.928 | 0.901 | 5.520 | 0.687 |

## Evaluation Notes

- **Per-model preprocessing**: For the torchvision-based detectors, the training pipeline applied
  ImageNet normalization through Albumentations in addition to the normalization performed
  internally by the torchvision detection models. Evaluation used the same preprocessing to avoid a
  train-evaluation distribution mismatch (evaluating on raw inputs instead drops mAP@0.50:0.95
  substantially — e.g., RetinaNet 0.538 → 0.384).
- **Benchmark timing**: torchvision models perform NMS inside `forward()` during eval, so their
  inference latency includes postprocessing; Ultralytics numbers are raw forward passes.

## Artifacts

- Metrics: `outputs/milestone_5/metrics/kitti_validation/{yolo,rtdetr,retinanet,faster_rcnn}_metrics.json`
- Benchmarks: `outputs/milestone_5/benchmarks/{yolo,rtdetr,retinanet,faster_rcnn}_benchmark.json`
- Comparisons: `outputs/milestone_5/figures/{accuracy,efficiency,operating_point,training_vs_validation}_comparison.csv`
- Locked checkpoints (input from Milestone 4): `outputs/milestone_4/checkpoints/{model}/final/best.pt`
- Checkpoint registry: `outputs/milestone_4/manifests/final_checkpoint_registry.csv`
