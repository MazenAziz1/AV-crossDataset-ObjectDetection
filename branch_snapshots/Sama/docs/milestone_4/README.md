# Milestone 4 — Documentation

This directory contains the frozen protocol documents and reports for **Milestone 4**
(model implementation, fair training protocol, and Kaggle training infrastructure).

KITTI in-domain evaluation and benchmarking documentation lives in
[`docs/milestone_5/`](../milestone_5/README.md).

## Frozen Protocol Documents

| Document | Purpose | Status |
|----------|---------|--------|
| `experiment_scope.md` | Experiment scope, dataset roles, Waymo exclusion rule | FROZEN |
| `kaggle_compute_plan.md` | Slot A/B allocation, session limits, RT-DETR resume policy | FROZEN |
| `kaggle_training_protocol.md` | Shared training config, effective batch, optimizer, checkpointing, runtime guard | FROZEN |
| `kaggle_package_spec.md` | Kaggle training package contents | FROZEN |
| `model_selection_rationale.md` | Rationale for the four selected detectors | FROZEN |

## Related

- Milestone 5 (KITTI in-domain evaluation): `docs/milestone_5/`
- Milestone 6 (Waymo external validation): `docs/milestone_6/`

## Results

Training artifacts (checkpoints, training manifests, training-time metrics, and the locked
checkpoint registry) live under `outputs/milestone_4/`. Final KITTI evaluation metrics,
benchmarks, and comparison tables live under `outputs/milestone_5/`.
