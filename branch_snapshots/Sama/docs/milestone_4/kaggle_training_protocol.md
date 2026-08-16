# Milestone 4 + 5: Kaggle Training Protocol
**Status**: `FROZEN`
**Date Frozen**: 2026-06-26
**Target Platform**: Kaggle GPU Notebooks (P100 / T4 x2)

---

## 1. Purpose

This protocol defines the shared training configuration used by both Kaggle compute slots (Slot A and Slot B). All four detectors share the same training policies for comparability; only per-model batch sizes and VRAM mitigations differ (see `model_registry.yaml`).

---

## 2. Shared Training Configuration

### 2.1 Epochs and Early Stopping

| Parameter | Value |
|-----------|-------|
| Target epochs | 200 |
| Early stopping patience | 20 epochs |
| Early stopping metric | `mAP@0.50:0.95` |
| Min delta for improvement | 0.001 |

### 2.2 Effective Batch Size

All detectors are trained with an **effective batch size of 32**, except RT-DETR-L which uses **16**:

| Detector | Physical Batch | Gradient Accumulation | Effective Batch |
|----------|---------------|----------------------|-----------------|
| YOLOv8s | 32 | 1 | 32 |
| Faster R-CNN | 4 | 8 | 32 |
| RetinaNet | 4 | 8 | 32 |
| RT-DETR-L | 16 | 1 | 16 |

### 2.3 Dataloader Configuration

- **Workers**: 4 per detector.
- **Seed locked**: DataLoader worker seed enforced for reproducibility.
- **Resolution**: 640 x 640 (letterboxed).
- **Shuffle**: Enabled for training, disabled for validation.

---

## 3. Framework-Specific Optimizer Settings

### 3.1 Ultralytics (YOLOv8s, RT-DETR-L)

| Parameter | YOLOv8s | RT-DETR-L |
|-----------|---------|-----------|
| Optimizer | auto (SGD) | auto (AdamW) |
| Initial LR (`lr0`) | 0.01 | 0.0001 |
| Final LR factor (`lrf`) | 0.01 | 0.01 |
| Momentum | 0.937 | 0.9 |
| Weight decay | 0.0005 | 0.0001 |
| Warmup epochs | 3 | 3 |
| Warmup momentum | 0.8 | 0.8 |
| Warmup bias LR | 0.1 | 0.1 |

### 3.2 Torchvision (Faster R-CNN, RetinaNet)

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning rate | 0.02 |
| Momentum | 0.9 |
| Weight decay | 0.0001 |
| LR scheduler | CosineAnnealingLR (T_max = 200) |
| Warmup epochs | 5 |

---

## 4. Augmentation Policy

Training augmentation follows the frozen policy defined in:
```
configs/datasets/milestone_3/augmentation.yaml
```

- **Training**: Augmentation enabled (Albumentations).
- **Validation**: Augmentation disabled (raw 640x640 letterboxed images).
- Standard augmentations: random horizontal flip, brightness/contrast, HSV adjustment, Gaussian blur.

---

## 5. Checkpointing Policy

| Rule | Value |
|------|-------|
| Save best checkpoint | Yes (by `mAP@0.50:0.95`) |
| Save last checkpoint | Yes (every epoch) |
| Save frequency | Every epoch |
| Keep last N | 3 most recent checkpoints |
| Save resume state | Yes (`resume_state.json`) |
| Checkpoint format | `.pt` (PyTorch) |

---

## 6. Runtime Guard

To prevent forced Kaggle session termination from corrupting checkpoints:

- **Max runtime**: 10.5 hours (safe margin before ~12h hard limit).
- **Grace period**: 5 minutes for saving and packaging before exit.
- **Check interval**: Every 300 seconds (5 minutes).
- **On timeout**: Finish current epoch, save checkpoint, write `resume_state.json`, package outputs, exit cleanly.
- **Exit status recorded** in session manifest as `runtime_guard_stop`.

---

## 7. Kaggle Dataset Paths

All paths are relative to the Kaggle working directory after unzipping the training package:

```
/kaggle/working/project/milestone4_kaggle_training_package/
```

| Resource | Relative Path |
|----------|--------------|
| KITTI train images | `data/processed/milestone_3/images/kitti/train` |
| KITTI val images | `data/processed/milestone_3/images/kitti/val` |
| COCO train annotations | `data/processed/milestone_3/annotations/coco/kitti_train.json` |
| COCO val annotations | `data/processed/milestone_3/annotations/coco/kitti_val.json` |
| YOLO train labels | `data/processed/milestone_3/labels/kitti/train` |
| YOLO val labels | `data/processed/milestone_3/labels/kitti/val` |
| Outputs | `outputs/milestone_4/` |

---

## 8. Waymo Exclusion

> **Hard boundary**: No Waymo images, labels, or paths are present in the Kaggle training package. Waymo is excluded from Milestones 4 and 5 and deferred to Milestone 6.

---

## 9. Completion Gate

- [x] Training epochs, early stopping frozen
- [x] Effective batch size = 32 (RT-DETR-L = 16)
- [x] Optimizer settings frozen per framework
- [x] Augmentation policy linked
- [x] Checkpointing rules frozen
- [x] Runtime guard configured (10.5h limit)
- [x] Kaggle dataset paths documented
- [x] Waymo exclusion enforced
- [x] Config file frozen: `configs/models/milestone_4/kaggle_training_policy.yaml`
