# Milestone 6 Configuration Summary

Created at: `2026-08-15T00:19:50`

## Purpose

Milestone 6 evaluates the locked KITTI-trained object detectors on the Waymo external subset without retraining.

## Waymo Handoff

- Images: `996`
- Labels: `996`
- Matched image-label pairs: `996`
- Total annotations: `24819`

## Class Mapping

| Class ID | Class Name | Waymo Annotation Count |
|---:|---|---:|
| 0 | Vehicle | 16928 |
| 1 | Pedestrian | 7127 |
| 2 | Cyclist | 764 |

## Policy

- No retraining
- No fine-tuning
- No architecture changes
- No threshold tuning on Waymo
- Same class mapping as Milestone 5
- Same metric definitions as Milestone 5
- Compare Waymo external validation against KITTI in-domain validation

## Generated Files

- `configs/datasets/milestone_6/waymo_external_subset.yaml`
- `configs/evaluation/milestone_6/external_validation_policy.yaml`
- `configs/evaluation/milestone_6/generalization_metrics.yaml`
