# Milestone 4 + 5: Kaggle Training Package Specification
**Status**: `FROZEN`
**Date Frozen**: 2026-06-26

---

## 1. Purpose

Define the exact file inclusion and exclusion rules for the Kaggle training package. The package is a self-contained ZIP that provides everything Kaggle notebooks need for KITTI-only training -- no local outputs, no pretrained weights (downloaded on Kaggle), and zero Waymo data.

---

## 2. Inclusion Rules

### 2.1 Configuration Files

| Source | Files |
|--------|-------|
| `configs/models/milestone_4/` | All 7 YAML files (protocol, registry, evaluation, class contract, compute plan, training policy, kaggle training policy) |
| `configs/datasets/milestone_3/` | All 6 YAML files (dataset registry, KITTI/Waymo config, class mapping, COCO paths, preprocessing, augmentation) |

### 2.2 Scripts

| Source | Files |
|--------|-------|
| `scripts/milestone_4/` | All Python files (training, utilities, adapters, evaluation, kaggle) |
| `scripts/milestone_4/kaggle/` | Kaggle environment validators and package scripts |
| `scripts/__init__.py` | Root package init |

### 2.3 Dataset: KITTI Only

| Source | Contents | Count | Approx. Size |
|--------|----------|-------|-------------|
| `data/processed/milestone_3/images/kitti/train/` | 640x640 PNG images | 5,985 | ~2.5 GB |
| `data/processed/milestone_3/images/kitti/val/` | 640x640 PNG images | 1,496 | ~650 MB |
| `data/processed/milestone_3/labels/kitti/train/` | YOLO-format TXT labels | 5,985 | ~2 MB |
| `data/processed/milestone_3/labels/kitti/val/` | YOLO-format TXT labels | 1,496 | ~0.5 MB |
| `data/processed/milestone_3/annotations/coco/` | COCO JSON annotations (kitti_train.json, kitti_val.json) | 2 | ~15 MB |
| `data/processed/milestone_3/annotations/ignore_regions/` | DontCare ignore sidecars (kitti_train_ignore.json, kitti_val_ignore.json) | 2 | ~2 MB |
| `data/processed/milestone_3/annotations/excluded_objects/` | Excluded non-targets (kitti_train_excluded.json, kitti_val_excluded.json) | 2 | ~1 MB |

### 2.4 Requirements

| Source | Purpose |
|--------|---------|
| `requirements.txt` or `kaggle_requirements.txt` | Python dependencies for Kaggle notebook |

### 2.5 Documentation

| Source | Purpose |
|--------|---------|
| `docs/milestone_4/*.md` | Training protocol documentation (for reference on Kaggle) |

---

## 3. Exclusion Rules (Hard Blocklist)

### 3.1 Waymo Data -- ZERO Tolerance

All files under these paths are excluded:

```
data/processed/milestone_3/images/waymo/
data/processed/milestone_3/labels/waymo/
data/processed/milestone_3/annotations/*waymo*
data/processed/milestone_3/visual_checks/*/waymo/
data/processed/milestone_3/reports/waymo*
data/waymo/               (entire directory)
```

Verification: 3,008 Waymo files exist in `data/processed/milestone_3/` and must ALL be excluded.

### 3.2 Pretrained Weights

Weights are downloaded fresh on Kaggle to ensure integrity and avoid large uploads:

```
outputs/milestone_4/pretrained/   (exclude entirely)
*.pt files in outputs/            (exclude entirely)
*.pth files in outputs/           (exclude entirely)
```

### 3.3 Local Artifacts

```
.git/
.venv/
__pycache__/
*.pyc
outputs/milestone_4/checkpoints/
outputs/milestone_4/logs/
outputs/milestone_4/predictions/
outputs/milestone_4/metrics/
outputs/milestone_4/benchmarks/
outputs/milestone_4/kaggle_packages/   (avoid nesting)
```

### 3.4 Binary/Model Files

```
*.pt      (except for scripts, which don't contain .pt)
*.pth
*.ckpt
*.onnx
*.engine
*.zip
*.tar.gz
```

---

## 4. Package Structure (Inside ZIP)

```
milestone4_kaggle_training_package/
├── configs/
│   ├── models/milestone_4/          (all YAML files)
│   └── datasets/milestone_3/        (all YAML files)
├── scripts/
│   ├── __init__.py
│   └── milestone_4/
│       ├── __init__.py
│       ├── *.py                     (training, evaluation scripts)
│       ├── kaggle/                  (Kaggle-specific scripts)
│       ├── adapters/                (prediction adapters)
│       ├── trainers/                (training core)
│       ├── evaluation/              (evaluation core)
│       └── utilities/               (helpers)
├── data/processed/milestone_3/
│   ├── images/kitti/train/          (5,985 PNG)
│   ├── images/kitti/val/            (1,496 PNG)
│   ├── labels/kitti/train/          (5,985 TXT)
│   ├── labels/kitti/val/            (1,496 TXT)
│   ├── annotations/coco/            (2 JSON)
│   ├── annotations/ignore_regions/  (2 JSON)
│   └── annotations/excluded_objects/(2 JSON)
├── docs/milestone_4/                (protocol docs)
└── kaggle_requirements.txt
```

---

## 5. Waymo Exclusion Audit

| Check | Status |
|-------|--------|
| `images/waymo/` in package | Must be 0 |
| `labels/waymo/` in package | Must be 0 |
| `annotations/*waymo*` in package | Must be 0 |
| `visual_checks/*/waymo/` in package | Must be 0 |
| `reports/waymo*` in package | Must be 0 |
| Waymo path references in configs | Must be non-active (exclusion rule only) |

---

## 6. Completion Gate

- [ ] All KITTI train images (5,985) included
- [ ] All KITTI val images (1,496) included
- [ ] All KITTI labels included
- [ ] All COCO annotations included
- [ ] All ignore/excluded sidecars included
- [ ] All configs included
- [ ] All scripts included
- [ ] Waymo file count in package = 0
- [ ] Pretrained weights excluded
- [ ] Package specification approved and frozen
