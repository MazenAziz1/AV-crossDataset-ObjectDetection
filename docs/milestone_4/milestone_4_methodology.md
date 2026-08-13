# Milestone 4 + 5: Local KITTI Evaluation Methodology

**Status**: In Progress
**Scope**: YOLOv8s validated locally; Faster R-CNN, RetinaNet, RT-DETR pending Kaggle import

---

## 1. Purpose

This document records the local in-domain validation protocol for models trained on Kaggle. Local machine is the source of truth for evaluation, checkpoint integrity, benchmarking, and documentation.

---

## 2. Checkpoint Import & Integrity

### 2.1 Imported Checkpoints

| Detector | Checkpoint | Size (MB) | SHA-256 | Status |
|----------|-----------|-----------|---------|--------|
| YOLOv8s | best.pt | 21.5 | `b2f4a3066c767edeee90e02b54ed71022319b30196585046697677cca8ff146c` | ✅ locked |
| YOLOv8s | last.pt | 21.5 | `26af53006191ec427056beea0a450f631d20a9bf9d3bc9629291ed2421a50fb8` | ✅ locked |
| Faster R-CNN | — | — | — | pending |
| RetinaNet | — | — | — | pending |
| RT-DETR-L | — | — | — | pending |

---

## 3. Local KITTI Validation Protocol

- **Partition**: `kitti_val` (1,496 images)
- **Ground truth**: COCO JSON `data/processed/milestone_3/annotations/coco/kitti_val.json`
- **Class contract**: Vehicle (1), Pedestrian (2), Cyclist (3)
- **Metric engine**: pycocotools `COCOeval` (bbox), identical across all four models
- **Confidence threshold**: 0.001 (low, for full AP curve)
- **DontCare suppression**: enabled, IoU ≥ 0.50 against ignore regions

---

## 4. YOLOv8s Results

### 4.1 Detection Accuracy

| Metric | Value |
|--------|-------|
| mAP@0.50:0.95 | 0.690 |
| mAP@0.50 | 0.925 |
| mAP@0.75 | 0.780 |
| AP small | 0.645 |
| AP medium | 0.752 |
| AP large | 0.718 |
| AR@100 | 0.744 |

### 4.2 Per-Class Performance

| Class | AP@0.50:0.95 | AP@0.50 |
|-------|--------------|---------|
| Vehicle | 0.849 | 0.975 |
| Pedestrian | 0.530 | 0.869 |
| Cyclist | 0.691 | 0.931 |

**Observation**: Pedestrian is the hardest class (0.530), consistent with KITTI's small, occluded pedestrian instances. Vehicle dominates due to abundant, large training instances.

### 4.3 Efficiency

| Metric | Value |
|--------|-------|
| Parameters | 11,136,761 |
| Checkpoint size | 21.5 MB |
| GPU memory peak | 84.9 MB |
| Preprocessing | 1.745 ms |
| Inference | 10.551 ms |
| Postprocessing | 0.058 ms |
| Total latency | 12.355 ms |
| FPS | 94.77 |

---

## 5. Training Configuration (Recap)

- **Framework**: Ultralytics 8.4.x
- **Pretrained**: COCO `yolov8s.pt`
- **Epochs**: 200 (early stopping patience 20)
- **Input**: 640×640
- **Optimizer**: auto (AdamW/MuSGD), seed 42
- **GPU**: Kaggle Tesla T4 (14.6 GB)

---

## 6. Remaining Work

- [ ] Import and validate Faster R-CNN, RetinaNet, RT-DETR checkpoints
- [ ] Validate RT-DETR multi-session resume chain
- [ ] Generate cross-model comparison tables and figures
- [ ] Final audit and Git closure
