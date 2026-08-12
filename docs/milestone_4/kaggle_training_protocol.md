# Kaggle Training Protocol — Milestone 4 + 5

## Purpose

This document defines how Kaggle is used for model training while keeping the local machine as the source of truth.

## Roles

| Environment | Role |
|---|---|
| Local machine | Source of truth, packaging, import, validation, documentation, Git |
| Kaggle | GPU training only |

## Kaggle Slot Allocation

| Slot | Detectors |
|---|---|
| Slot A | YOLO, Faster R-CNN, RetinaNet |
| Slot B | RT-DETR |

## Runtime Policy

Kaggle sessions are treated as limited sessions. Training scripts must use a runtime guard of 10.5 hours, save checkpoints, save session manifests, and package outputs before shutdown.

## RT-DETR Resume Policy

RT-DETR may exceed a single session. It must support resume from the latest checkpoint. The resume chain must be validated locally after importing Kaggle outputs.

## Prohibited Actions

- Do not upload Waymo to Kaggle for this milestone.
- Do not use Waymo for training or checkpoint selection.
- Do not commit checkpoints or Kaggle ZIP packages to Git.

## Completion Gate

Kaggle training policy is complete when Slot A, Slot B, checkpointing, resume, output packaging, and local import responsibilities are defined.
