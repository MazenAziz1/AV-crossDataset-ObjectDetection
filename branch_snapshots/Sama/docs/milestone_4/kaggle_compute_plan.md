# Milestone 4 + 5: Kaggle Compute-Allocation Plan
**Status**: `FROZEN`
**Date Frozen**: 2026-06-26

---

## 1. Purpose

This document defines how two legitimate Kaggle GPU compute slots are allocated across the four detector families and how the RT-DETR multi-session resume chain is managed. The policy ensures:

- Reproducible training across separate Kaggle sessions.
- No local GPU training (local is the source of truth for code, configs, evaluation, and Git).
- RT-DETR interruption-resilience via checkpoint/resume packaging.

---

## 2. Compute Slot Assignment

### Slot A: Primary Compute Slot

| Detector | Architecture | Target Epochs | Early Stopping Patience |
|----------|-------------|---------------|-------------------------|
| YOLOv8s | Single-stage CNN | 200 | 20 |
| Faster R-CNN | Two-stage CNN (ResNet-50-FPN) | 200 | 20 |
| RetinaNet | Single-stage CNN (ResNet-50-FPN V2) | 200 | 20 |

### Slot B: Secondary Compute Slot

| Detector | Architecture | Target Epochs | Resume Strategy |
|----------|-------------|---------------|-----------------|
| RT-DETR-L | Vision Transformer (DETR) | 200 | Multi-session resume |

> **Note**: RT-DETR-L is isolated to Slot B for dedicated multi-session resume-chain tracking, since it is the most likely detector to require multiple sessions.

---

## 3. Session Limit Policy

- **Hard Kaggle limit**: ~12 hours per GPU session.
- **Safe stop margin**: Training stops gracefully at **10.5 hours**.
- **On timeout**: The runtime guard finishes the current epoch, saves the checkpoint, writes `resume_state.json`, packages all outputs, and exits cleanly.

---

## 4. RT-DETR Resume Policy

RT-DETR is treated as an **interruption-expected** training run. The resume chain works as follows:

1. **Session N starts** with `--resume latest` and reads `resume_state.json` to determine the last completed epoch.
2. **Training continues** from epoch `last_epoch + 1` toward the target of 200.
3. **Every 5 epochs**: A hard checkpoint is saved and validation is run.
4. **Before session timeout**: The runtime guard triggers a graceful save:
   - `last.pt` (latest checkpoint)
   - `best.pt` (best mAP checkpoint)
   - `resume_state.json` (epoch, optimizer, scheduler state)
   - `session_manifest.csv` (session metadata)
5. **The resume package** is exported as a ZIP for the next session.
6. **Session N+1** unzips the resume package and continues training.

### Epoch Continuity Validation

Between sessions, epoch numbering must be contiguous with no gaps. The resume chain audit validates:
- `end_epoch` of Session N == `start_epoch - 1` of Session N+1.
- Identical config, dataset, and seed hashes across all sessions.

---

## 5. Reproducibility Guarantees

- Random seed locked to `42` (Python, NumPy, PyTorch, DataLoader workers).
- Deterministic algorithm mode enabled.
- All training configs frozen before any training starts.
- No Waymo data is used in Milestones 4 or 5.

---

## 6. Waymo Exclusion Boundary

> **Rule**: Zero training, validation, or evaluation on Waymo dataset files during Milestones 4 and 5. Waymo cross-domain evaluation is strictly deferred to Milestone 6.

---

## 7. Completion Gate

- [x] Slot assignments frozen (Slot A: YOLO, Faster R-CNN, RetinaNet / Slot B: RT-DETR)
- [x] Session limits defined (10.5 hours safe stop)
- [x] RT-DETR resume policy defined (target 200 epochs, checkpoint every 5 epochs)
- [x] Waymo exclusion rule enforced
- [x] Config file frozen: `configs/models/milestone_4/kaggle_compute_plan.yaml`
