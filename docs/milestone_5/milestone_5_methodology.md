# Milestone 4 + 5: Local KITTI Evaluation Methodology

**Status**: Complete
**Scope**: All four detectors (YOLOv8s, RT-DETR-L, RetinaNet, Faster R-CNN) trained on Kaggle, imported, evaluated, benchmarked, and locked locally.

---

## 1. Purpose

This document records the local in-domain validation protocol for models trained on Kaggle. The local machine is the source of truth for evaluation, checkpoint integrity, benchmarking, and documentation. Waymo external validation is deferred to Milestone 6.

---

## 2. Checkpoint Import & Integrity

### 2.1 Imported Checkpoints

| Detector | Checkpoint | Size (MiB) | SHA-256 (best.pt) | Status |
|----------|-----------|-----------|-------------------|--------|
| YOLOv8s | best.pt / last.pt | 21.5 | `b2f4a3066c767edeee90e02b54ed71022319b30196585046697677cca8ff146c` | ✅ locked |
| RT-DETR-L | best.pt / last.pt | 63.15 | `d79a9d88b5c249b6757372be825b635d096eebb5cf322ce26b664c9c44475aee` | ✅ locked |
| RetinaNet | best.pt / last.pt | 278.0 | `dc44dfcf5a743980b86c5e4deb0f9e53efe7b648c94aeadfa8c28212552fa633` | ✅ locked |
| Faster R-CNN | best.pt / last.pt | 315.68 | `1d0bebabe7f171f060e48b47a361c7207ff7f700f74bcb45c36b4d9d938be223` | ✅ locked |

All four checkpoints passed the load test and are recorded in `outputs/milestone_4/manifests/final_checkpoint_registry.csv`.

---

## 3. Local KITTI Validation Protocol

- **Partition**: `kitti_val` (1,496 images)
- **Ground truth**: COCO JSON `data/processed/milestone_3/annotations/coco/kitti_val.json`
- **Class contract**: Vehicle (1), Pedestrian (2), Cyclist (3)
- **Metric engine**: pycocotools `COCOeval` (bbox), identical across all four models
- **Confidence threshold**: 0.001 (low, for full AP curve)
- **DontCare suppression**: enabled, IoU ≥ 0.50 against ignore regions
- **Operating point**: confidence ≥ 0.25 and IoU ≥ 0.50 → precision, recall, F1, detections/image, false-positives/image

### 3.1 Per-Model Preprocessing Note

For the torchvision-based detectors, the training pipeline applied ImageNet normalization through Albumentations in addition to the normalization performed internally by the torchvision detection models. Consequently, evaluation was performed using the same preprocessing configuration employed during training to avoid a train–evaluation distribution mismatch. The Ultralytics-based YOLOv8s and RT-DETR models did not exhibit this issue.

---

## 4. Results (local frozen protocol)

### 4.1 Cross-Model Comparison

| Detector | mAP@0.50:0.95 | mAP@0.50 | mAP@0.75 | Vehicle AP | Pedestrian AP | Cyclist AP | FPS | Params |
|----------|---------------|----------|----------|------------|---------------|------------|-----|--------|
| YOLOv8s | **0.690** | **0.925** | **0.780** | **0.849** | **0.530** | **0.691** | **94.77** | **11.1M** |
| RT-DETR-L | 0.606 | 0.906 | 0.679 | 0.730 | 0.490 | 0.597 | 18.73 | 32.8M |
| Faster R-CNN | 0.589 | 0.879 | 0.667 | 0.741 | 0.432 | 0.594 | 11.24 | 41.3M |
| RetinaNet | 0.538 | 0.839 | 0.577 | 0.703 | 0.392 | 0.518 | 13.70 | 36.4M |

### 4.2 YOLOv8s

| Metric | Value |
|--------|-------|
| mAP@0.50:0.95 | 0.690 |
| mAP@0.50 | 0.925 |
| mAP@0.75 | 0.780 |
| AP small / medium / large | 0.645 / 0.752 / 0.718 |
| Vehicle / Pedestrian / Cyclist AP@0.50:0.95 | 0.849 / 0.530 / 0.691 |
| FPS (local GPU) | 94.77 |
| Parameters | 11,136,761 |

### 4.3 RT-DETR-L

| Metric | Value |
|--------|-------|
| mAP@0.50:0.95 | 0.606 |
| mAP@0.50 | 0.906 |
| mAP@0.75 | 0.679 |
| AP small / medium / large | 0.515 / 0.700 / 0.714 |
| Vehicle / Pedestrian / Cyclist AP@0.50:0.95 | 0.730 / 0.490 / 0.597 |
| FPS (local GPU) | 18.73 |
| Parameters | 32,812,241 |

### 4.4 RetinaNet

| Metric | Value |
|--------|-------|
| mAP@0.50:0.95 | 0.538 |
| mAP@0.50 | 0.839 |
| mAP@0.75 | 0.577 |
| AP small / medium / large | 0.467 / 0.622 / 0.624 |
| Vehicle / Pedestrian / Cyclist AP@0.50:0.95 | 0.703 / 0.392 / 0.518 |
| FPS (local GPU) | 13.70 |
| Parameters | 36,392,072 |

### 4.5 Faster R-CNN

| Metric | Value |
|--------|-------|
| mAP@0.50:0.95 | 0.589 |
| mAP@0.50 | 0.879 |
| mAP@0.75 | 0.667 |
| AP small / medium / large | 0.558 / 0.628 / 0.606 |
| Vehicle / Pedestrian / Cyclist AP@0.50:0.95 | 0.741 / 0.432 / 0.594 |
| FPS (local GPU) | 11.24 |
| Parameters | 41,309,411 |

### 4.6 Operating Point (confidence ≥ 0.25, IoU ≥ 0.50)

| Detector | Precision | Recall | F1 | Detections/image | FP/image |
|----------|----------:|-------:|---:|-----------------:|---------:|
| YOLOv8s | 0.926 | 0.947 | 0.936 | 5.330 | 0.396 |
| RT-DETR-L | 0.482 | 0.969 | 0.644 | 10.479 | 5.431 |
| RetinaNet | 0.777 | 0.907 | 0.837 | 6.079 | 1.354 |
| Faster R-CNN | 0.876 | 0.928 | 0.901 | 5.520 | 0.687 |

RT-DETR-L emits many low-confidence detections (DETR-style 300 queries), so at a fixed 0.25 threshold it achieves the highest recall but the lowest precision (0.482) and most false positives per image (5.431).

---

## 5. Training Configuration (Recap)

| Detector | Framework | Epochs (target / run) | Early stop | Best epoch |
|----------|-----------|----------------------|------------|------------|
| YOLOv8s | Ultralytics | 200 / 200 | — | final |
| RT-DETR-L | Ultralytics | 200 / 131 | patience 20 | 111 |
| RetinaNet | Torchvision | 200 / 101 | patience 20 | 81 |
| Faster R-CNN | Torchvision | 200 / 96 | patience 20 | pre-resume (≤76) |

- Input 640×640, seed 42 (deterministic), effective batch 16, Kaggle Tesla T4.
- RT-DETR, RetinaNet, and Faster R-CNN used multi-session resume chains (validated contiguous, no gaps).

---

## 6. Completion Checklist

- [x] Four checkpoints imported and integrity-verified (SHA-256)
- [x] RT-DETR multi-session resume chain validated (3 sessions, contiguous)
- [x] Four checkpoints locked (load test PASSED)
- [x] Four KITTI evaluations complete (shared COCO metric engine)
- [x] Four efficiency benchmarks complete
- [x] Cross-model comparison outputs generated (accuracy, efficiency, training-vs-validation)
- [x] Final audit PASSED (13 checks, 0 issues)
- [x] Waymo external validation deferred to Milestone 6
