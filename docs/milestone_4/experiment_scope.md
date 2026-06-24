# Milestone 4 + 5: Experiment Scope & Protocol
**Status**: `FROZEN`
**Date Frozen**: 2026-06-24

---

## 1. Objectives & Boundary Scope
The objective of Milestones 4 and 5 is to integrate, train, evaluate, and benchmark four object detection architectures on the **KITTI** dataset under a unified, rigorous experimental protocol.

> [!IMPORTANT]
> **Waymo Exclusion Boundary Rule**:
> Under no circumstances will any Waymo dataset files, labels, or paths be used for model training, validation, or evaluation during this phase. All evaluations in Milestones 4 & 5 are internal validations on the KITTI validation partition. External validation, domain degradation, and cross-dataset testing are strictly deferred to Milestone 6.

---

## 2. Dataset Allocations
We use the preprocessed KITTI dataset created in Milestone 3, stored under `data/processed/milestone_3/`.

* **Training Set**: `kitti_train`
  - **Count**: 5,985 images
  - **Resolution**: 640x640 (letterboxed)
  - **Role**: Backpropagation and parameter updates.
* **Validation Set**: `kitti_val`
  - **Count**: 1,496 images
  - **Resolution**: 640x640 (letterboxed)
  - **Role**: Model selection, hyperparameter tuning, and final internal validation.

---

## 3. Class Representation Contract
We detect three object classes: **Vehicle**, **Pedestrian**, and **Cyclist**. Due to differences in framework requirements, we freeze the class mapping contracts:

1. **YOLO / RT-DETR Contract (Ultralytics)**:
   - `0` -> Vehicle
   - `1` -> Pedestrian
   - `2` -> Cyclist
2. **Faster R-CNN / RetinaNet Contract (Torchvision)**:
   - `0` -> background
   - `1` -> Vehicle
   - `2` -> Pedestrian
   - `3` -> Cyclist

---

## 4. Training Policies
* **Seed Policy**: Seed is locked to `42` across Python, NumPy, PyTorch, and DataLoader worker seeds.
* **Input Size**: All models are trained with `640x640` input dimensions.
* **Weights Policy**: Pretrained weights (pre-trained on COCO/ImageNet) are loaded to start training.
* **Early Stopping**: Early stopping is enabled, triggered when the validation `mAP@0.50:0.95` fails to improve for `20` consecutive epochs.

---

## 5. Evaluation & Benchmarking Policies
* **Primary Metric**: COCO `mAP@0.50:0.95` (using identical evaluator matching engine).
* **Operating Point Metric**: Precision, Recall, and F1-score evaluated at a confidence threshold of `0.25` and IoU threshold of `0.50`.
* **DontCare Suppression**: Detections falling inside KITTI `DontCare` ignore regions must be matched to prevent penalizing models as false positives.
* **Efficiency Metrics**: Parameters count, FPS, and latencies (pre-processing, inference, post-processing) measured on a standard hardware environment using `CUDA`.
