# Experiment Scope — Milestone 4 + 5

## Purpose

Milestone 4 + 5 trains and compares four object detector families under a controlled KITTI-only training and internal validation protocol.

## Detector Families

- YOLO
- Faster R-CNN
- RetinaNet
- RT-DETR

## Dataset Roles

| Dataset Partition | Role | Allowed in Milestone 4 + 5 |
|---|---|---|
| KITTI train | Model optimization | Yes |
| KITTI validation | Internal validation, checkpoint selection, final in-domain evaluation | Yes |
| Waymo external | External validation in Milestone 6 | No |

## Scope Boundary

Waymo is not allowed during this milestone. It must not be used for training, hyperparameter tuning, checkpoint selection, threshold tuning, or early analysis.

## Execution Boundary

The local machine remains the source of truth. Kaggle is used only for GPU training.

## Completion Gate

This scope is complete when the experiment protocol, class contract, evaluation policy, and Kaggle training policy are all frozen.
