# Kaggle Upload Set — Milestone 4 + 5

## Purpose

This document defines what is allowed to enter the Kaggle training package.

Kaggle is used for GPU training only. The local machine remains the source of truth for code, configuration, validation, documentation, imported outputs, and Git commits.

## Included Groups

| Group | Included |
|---|---|
| Source code | `scripts/milestone_4` |
| Model configs | `configs/models/milestone_4` |
| Dataset configs | `configs/datasets/milestone_3` |
| Processed images | `data/processed/milestone_3/images` |
| COCO annotations | `data/processed/milestone_3/annotations/coco` |
| YOLO labels | `data/processed/milestone_3/labels` |
| Ignore regions | `data/processed/milestone_3/annotations/ignore_regions` |
| Excluded objects | `data/processed/milestone_3/annotations/excluded_objects` |
| Manifests | `data/processed/milestone_3/manifests` |
| Reports | `data/processed/milestone_3/reports` |
| Documentation | `docs/milestone_4` |

## Dataset Boundary

| Dataset | Included? |
|---|---|
| KITTI train | Yes |
| KITTI validation | Yes |
| Waymo external validation | No |

Waymo remains excluded from Milestone 4 + 5 and is deferred to Milestone 6.

## Prohibited Uploads

The Kaggle package must not include:

- model checkpoints;
- pretrained weights;
- Kaggle output ZIP packages;
- credentials or secrets;
- raw Waymo data;
- Waymo external validation data;
- Milestone 6 evaluation outputs.

## Completion Gate

This step is complete when the upload-set YAML and manifest exist and the manifest confirms all required groups are present.
