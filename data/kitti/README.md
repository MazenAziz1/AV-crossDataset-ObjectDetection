# KITTI Object-Detection Dataset Preparation

This directory documents the complete, reproducible workflow used to prepare the **KITTI Object Detection** dataset for Milestone 2 of this research project.

KITTI is used for model training and in-domain validation. Waymo is prepared separately and used only for external validation.

---

## 1. Final Result

| Item | Final count |
|---|---:|
| Labeled KITTI images | 7,481 |
| Label files | 7,481 |
| Calibration files | 7,481 |
| Original annotation rows (boxes) | 51,865 |
| Harmonized target boxes | 39,086 |
| Ignored boxes | 12,779 |
| Training images | 5,985 |
| Validation images | 1,496 |
| Random seed | 42 |
| Corrupted images | 0 |
| Missing matching files | 0 |
| Invalid bounding boxes | 0 |
| Calibration issues | 0 |
| Target-empty images | 0 |
| Ignored-only images | 0 |

The harmonized target classes are:

```text
0 = Vehicle
1 = Pedestrian
2 = Cyclist
```

| Harmonized class | Total | Train | Validation |
|---|---:|---:|---:|
| Vehicle | 32,750 | 26,278 | 6,472 |
| Pedestrian | 4,709 | 3,729 | 980 |
| Cyclist | 1,627 | 1,287 | 340 |
| Ignored | 12,779 | — | — |

The official KITTI testing images are not used for local evaluation because their public ground-truth labels are unavailable.

---

## 2. What `image_2` Means

The name `image_2` does **not** mean image number 2, a second dataset version, or a processed image set.

In KITTI camera naming:

```text
image_0 = left grayscale camera
image_1 = right grayscale camera
image_2 = left color camera
image_3 = right color camera
```

The KITTI object-detection benchmark distributes the **left color images** inside:

```text
image_2/
```

The corresponding object labels are stored in:

```text
label_2/
```

The suffix `_2` means that the images and annotations correspond to the left color camera stream. Its calibration projection matrix is:

```text
P2
```

This project uses only `image_2` because the study performs monocular RGB object detection. `image_3` is needed only for stereo experiments.

---

## 3. Required KITTI Downloads

Download these four official KITTI object-detection packages:

```text
data_object_image_2.zip
data_object_label_2.zip
data_object_calib.zip
devkit_object.zip
```

After extraction:

```text
data_object_image_2/
data_object_label_2/
data_object_calib/
devkit_object/
```

### `data_object_image_2`

Contains left color images:

```text
data_object_image_2/
├── training/
│   └── image_2/
└── testing/
    └── image_2/
```

The labeled training portion contains 7,481 PNG images.

### `data_object_label_2`

Contains labels for the training images:

```text
data_object_label_2/
└── training/
    └── label_2/
```

Each label file matches one image by basename:

```text
000000.png
000000.txt
```

### `data_object_calib`

Contains calibration files:

```text
data_object_calib/
├── training/
│   └── calib/
└── testing/
    └── calib/
```

The `P2` entry corresponds to the left color camera used by `image_2`.


## 4. Why the Official Testing Set Is Excluded

Only the official KITTI `training/` portion has public labels.

The official `testing/` images cannot be used locally to calculate:

- mAP,
- AP50,
- precision,
- recall,
- F1-score,
- false positives,
- class-specific metrics,
- or confusion matrices.

This project therefore uses:

```text
KITTI official labeled training set
├── project training split
└── project validation split
```

The split is created once, frozen, and reused for every detector.

---

## 5. Requirements

This workflow was developed on Windows with:

- Python 3.12
- virtual environment `AVenv`

Activate the environment:

```cmd
cd <your_project_root>
AVenv\Scripts\activate
```

## 6. Directory Structure

```text
data/
└── kitti/
    ├── README.md
    ├── raw/
    │   ├── training/
    │   │   ├── image_2/
    │   │   ├── label_2/
    │   │   └── calib/
    │   └── devkit_object/
    ├── selection/
    ├── statistics/
    └── visual_checks/
```

Create it:

```cmd
mkdir data\kitti\raw\training\image_2
mkdir data\kitti\raw\training\label_2
mkdir data\kitti\raw\training\calib
mkdir data\kitti\raw\devkit_object
mkdir data\kitti\selection
mkdir data\kitti\statistics
mkdir data\kitti\visual_checks
type nul > data\kitti\README.md
```

---

# Reproduction Workflow

## Step 1 — Extract the Four Packages

Extract the downloaded ZIP files into a local folder, for example:

```text
%USERPROFILE%\Downloads\
```

Expected folders:

```text
%USERPROFILE%\Downloads\data_object_image_2
%USERPROFILE%\Downloads\data_object_label_2
%USERPROFILE%\Downloads\data_object_calib
%USERPROFILE%\Downloads\devkit_object
```

Do not copy:

```text
data_object_image_2/testing/
data_object_calib/testing/
```

because the corresponding public test labels are unavailable.

---

## Step 2 — Copy the Labeled Data

Run from the project root.

### Images

```cmd
robocopy "%USERPROFILE%\Downloads\data_object_image_2\training\image_2" "data\kitti\raw\training\image_2" *.png /E
```

Observed result:

```text
Files: 7481
Copied: 7481
Failed: 0
Size: approximately 5.793 GB
```

### Labels

```cmd
robocopy "%USERPROFILE%\Downloads\data_object_label_2\training\label_2" "data\kitti\raw\training\label_2" *.txt /E
```

### Calibration

```cmd
robocopy "%USERPROFILE%\Downloads\data_object_calib\training\calib" "data\kitti\raw\training\calib" *.txt /E
```

### Devkit

```cmd
robocopy "%USERPROFILE%\Downloads\devkit_object" "data\kitti\raw\devkit_object" /E
```

Observed result:

```text
Devkit files copied: 18
Failed: 0
```

`robocopy` exit codes from 0 through 7 normally indicate success or a non-critical copy result.

---

## Step 3 — Verify Raw Counts

```cmd
dir /b "data\kitti\raw\training\image_2\*.png" | find /c /v ""
dir /b "data\kitti\raw\training\label_2\*.txt" | find /c /v ""
dir /b "data\kitti\raw\training\calib\*.txt" | find /c /v ""
```

Expected:

```text
Images:       7481
Labels:       7481
Calibration:  7481
```

Matching examples:

```text
image_2/000018.png
label_2/000018.txt
calib/000018.txt
```

---

## Step 4 — Verify Dataset Integrity

Script:

```text
scripts/milestone_2/kitti/01_verify_kitti_integrity.py
```

Run:

```cmd
python scripts\milestone_2\kitti\01_verify_kitti_integrity.py
```

The script verifies:

- matching image, label, and calibration IDs,
- readable PNG files,
- valid annotation field counts,
- numeric annotation values,
- valid bounding-box dimensions,
- known KITTI classes,
- truncation and occlusion values,
- presence of `P2`,
- valid calibration values,
- and out-of-image box warnings.

Generated files:

```text
data/kitti/statistics/
├── dataset_integrity_report.json
├── filename_mismatches.csv
├── image_issues.csv
├── label_issues.csv
├── calibration_issues.csv
└── box_warnings.csv
```

### KITTI label-row format

A normal ground-truth row contains:

```text
type
truncated
occluded
alpha
bbox_left
bbox_top
bbox_right
bbox_bottom
height_3d
width_3d
length_3d
location_x
location_y
location_z
rotation_y
```

2D box coordinates:

```text
xmin = bbox_left
ymin = bbox_top
xmax = bbox_right
ymax = bbox_bottom
```

### `DontCare` placeholder values

`DontCare` rows may use:

```text
truncation = -1
occlusion = -1
```

These are valid placeholders and must not be treated as corrupted values.

The validator skips normal truncation and occlusion checks when:

```python
class_name == "DontCare"
```

### Final integrity result

```text
Images found: 7481
Labels found: 7481
Calibration files found: 7481
Matching sample IDs: 7481
Total annotation rows: 51865
Valid annotation rows: 51865
Empty label files: 0
Filename mismatches: 0
Unreadable images: 0
Label issues: 0
Invalid bounding boxes: 0
Out-of-image box warnings: 0
Calibration issues: 0

Integrity status: PASSED
```

---

## Step 5 — Original Class Distribution

| Original class | Count |
|---|---:|
| Car | 28,742 |
| Van | 2,914 |
| Truck | 1,094 |
| Pedestrian | 4,487 |
| Person_sitting | 222 |
| Cyclist | 1,627 |
| Tram | 511 |
| Misc | 973 |
| DontCare | 11,295 |
| **Total** | **51,865** |

---

## Step 6 — Define the Class Mapping

File:

```text
data/kitti/selection/class_mapping.yaml
```

Mapping:

| KITTI class | Treatment | Final class |
|---|---|---|
| Car | Map | Vehicle |
| Van | Map | Vehicle |
| Truck | Map | Vehicle |
| Pedestrian | Map | Pedestrian |
| Person_sitting | Map | Pedestrian |
| Cyclist | Map | Cyclist |
| Tram | Ignore | — |
| Misc | Ignore | — |
| DontCare | Ignore region | — |

Complete mapping:

```yaml
dataset: KITTI Object Detection

final_classes:
  0: Vehicle
  1: Pedestrian
  2: Cyclist

kitti_mapping:
  Car:
    action: map
    mapped_class_id: 0
    mapped_class_name: Vehicle

  Van:
    action: map
    mapped_class_id: 0
    mapped_class_name: Vehicle

  Truck:
    action: map
    mapped_class_id: 0
    mapped_class_name: Vehicle

  Pedestrian:
    action: map
    mapped_class_id: 1
    mapped_class_name: Pedestrian

  Person_sitting:
    action: map
    mapped_class_id: 1
    mapped_class_name: Pedestrian

  Cyclist:
    action: map
    mapped_class_id: 2
    mapped_class_name: Cyclist

  Tram:
    action: ignore
    mapped_class_id: null
    mapped_class_name: null

  Misc:
    action: ignore
    mapped_class_id: null
    mapped_class_name: null

  DontCare:
    action: ignore
    mapped_class_id: null
    mapped_class_name: null

notes:
  - Car, Van, and Truck are harmonized into Vehicle.
  - Pedestrian and Person_sitting are harmonized into Pedestrian.
  - Cyclist remains Cyclist.
  - Tram and Misc are outside the unified three-class task.
  - DontCare is preserved only as an ignored evaluation region where needed.
  - The official KITTI testing split is not used because public labels are unavailable.
  - This mapping must remain fixed for all models and both datasets.
```

### Why `DontCare` is not a class

`DontCare` is an ignored region, not an object class the detector should learn.

Therefore:

```text
Ignore annotation ≠ remove image
```

The image remains in the dataset, while the region is retained for later evaluation handling.

---

## Step 7 — Validate the Mapping

Script:

```text
scripts/milestone_2/kitti/02_validate_class_mapping.py
```

Run:

```cmd
python scripts\milestone_2\kitti\02_validate_class_mapping.py
```

Generated report:

```text
data/kitti/statistics/class_mapping_validation.json
```

Mapped counts:

```text
Vehicle = Car + Van + Truck
        = 28,742 + 2,914 + 1,094
        = 32,750

Pedestrian = Pedestrian + Person_sitting
           = 4,487 + 222
           = 4,709

Cyclist = 1,627

Ignored = Tram + Misc + DontCare
        = 511 + 973 + 11,295
        = 12,779
```

Result:

```text
Unmapped source classes: None
Mapping status: PASSED
```

---

## Step 8 — Create the Fixed Train/Validation Split

Script:

```text
scripts/milestone_2/kitti/03_create_train_val_split.py
```

Run:

```cmd
python scripts\milestone_2\kitti\03_create_train_val_split.py
```

Split:

```text
Training:   5,985 images
Validation: 1,496 images
Total:      7,481 images
Random seed: 42
```

The split is stratified by:

- Vehicle presence,
- Pedestrian presence,
- Cyclist presence,
- and target-object density.

Generated files:

```text
data/kitti/selection/
├── train.txt
├── val.txt
├── split_assignments.csv
└── split_summary.json
```

No image is duplicated or moved. The text files contain six-digit image IDs.

### Target-empty and ignored-only images

The split code supports images with no mapped target objects and preserves them for background and false-positive evaluation.

Actual KITTI result:

```text
Target-empty images: 0
Ignored-only images: 0
```

Every labeled image contains at least one mapped Vehicle, Pedestrian, or Cyclist annotation.

### Final split counts

Training boxes:

```text
Vehicle: 26278
Pedestrian: 3729
Cyclist: 1287
```

Validation boxes:

```text
Vehicle: 6472
Pedestrian: 980
Cyclist: 340
```

Validation density:

```text
low: 594
high: 459
medium: 443
```

Result:

```text
Split status: PASSED
```

Do not regenerate this split after training starts.

---

## Step 9 — Generate Dataset Statistics

Script:

```text
scripts/milestone_2/kitti/04_generate_kitti_statistics.py
```

Run:

```cmd
python scripts\milestone_2\kitti\04_generate_kitti_statistics.py
```

Generated files:

```text
data/kitti/statistics/
├── dataset_summary.json
├── image_level_statistics.csv
├── object_level_statistics.csv
├── original_class_distribution.csv
├── mapped_class_distribution.csv
├── train_val_distribution.csv
├── bbox_size_statistics.csv
├── bbox_size_thresholds.json
├── occlusion_statistics.csv
├── truncation_statistics.csv
└── difficulty_statistics.csv
```

### Image-level statistics

Includes:

```text
image_id
split
image_width
image_height
original_annotation_count
target_box_count
ignored_box_count
vehicle_count
pedestrian_count
cyclist_count
contains_vehicle
contains_pedestrian
contains_cyclist
target_empty
ignored_only
presence_signature
density_group
stratification_group
```

### Object-level statistics

Includes:

```text
image_id
split
line_number
original_class
mapping_action
is_target_class
mapped_class_id
mapped_class_name
truncation
occlusion
alpha
xmin
ymin
xmax
ymax
bbox_width
bbox_height
bbox_area
normalized_bbox_area
image_width
image_height
3D dimensions
3D location
rotation_y
difficulty_group
bbox_size_group
```

### Size groups

Target boxes are grouped by normalized area:

```text
normalized_bbox_area = bbox_area / image_area
```

The lower and upper thirds define:

```text
small
medium
large
```

These are project-specific descriptive categories, not official KITTI categories.

### Difficulty groups

Descriptive groups:

```text
easy
moderate
hard
outside_standard_difficulty
not_applicable
```

### Final statistics result

```text
Image-level rows: 7481
Object-level rows: 51865

Complete dataset:
  Target boxes: 39086
  Ignored boxes: 12779
  Target-empty images: 0
  Ignored-only images: 0

Training split:
  Images: 5985
  Vehicle boxes: 26278
  Pedestrian boxes: 3729
  Cyclist boxes: 1287

Validation split:
  Images: 1496
  Vehicle boxes: 6472
  Pedestrian boxes: 980
  Cyclist boxes: 340

Statistics status: PASSED
```

---

## Step 10 — Create Statistical Figures

Script:

```text
scripts/milestone_2/kitti/05_create_kitti_figures.py
```

Run:

```cmd
python scripts\milestone_2\kitti\05_create_kitti_figures.py
```

Expected output:

```text
data/kitti/statistics/figures/
├── original_class_distribution.png
├── mapped_class_distribution.png
├── train_val_class_distribution.png
├── bbox_area_distribution.png
├── objects_per_image_distribution.png
├── occlusion_distribution.png
├── truncation_distribution.png
├── difficulty_distribution.png
└── figures_summary.json
```

---

## Step 11 — Create Annotated Visual Checks

Script:

```text
scripts/milestone_2/kitti/06_create_kitti_visual_checks.py
```

Run:

```cmd
python scripts\milestone_2\kitti\06_create_kitti_visual_checks.py
```

Generated files:

```text
data/kitti/visual_checks/
├── sample_vehicle.jpg
├── sample_pedestrian.jpg
├── sample_cyclist.jpg
├── sample_mixed_scene.jpg
├── sample_crowded_scene.jpg
├── sample_occluded_scene.jpg
├── sample_truncated_scene.jpg
├── sample_dontcare_scene.jpg
├── visual_checks_manifest.csv
└── visual_checks_summary.json
```

Selected source IDs:

| Check | Image ID |
|---|---:|
| Vehicle | `002972` |
| Pedestrian | `002707` |
| Cyclist | `006682` |
| Mixed | `004139` |
| Crowded | `004139` |
| Occluded | `002050` |
| Truncated | `002979` |
| DontCare | `000018` |

Colors:

```text
Vehicle = red
Pedestrian = green
Cyclist = blue
Ignored = yellow
```

The visualizer displays mappings such as:

```text
Car -> Vehicle
Van -> Vehicle
Pedestrian -> Pedestrian
Cyclist -> Cyclist
DontCare -> Ignore
```

### Visualizer behavior

This script does not run a detector. It draws only the original ground-truth annotations.

A visible object without a drawn box means that object is not labeled in the corresponding KITTI text file.

### Parked bicycles

A parked bicycle without a rider may not be labeled as `Cyclist`.

Typical behavior:

```text
parked bicycle without rider -> may be unlabeled
person near bicycle -> may be Pedestrian
person riding bicycle -> Cyclist
```

Verify a label file:

```cmd
type data\kitti\raw\training\label_2\002979.txt
```

Search for Cyclist:

```cmd
findstr /B /I "Cyclist" data\kitti\raw\training\label_2\002979.txt
```

---

# Final Directory Structure

```text
data/
└── kitti/
    ├── README.md
    ├── raw/
    │   ├── training/
    │   │   ├── image_2/
    │   │   ├── label_2/
    │   │   └── calib/
    │   └── devkit_object/
    ├── selection/
    │   ├── class_mapping.yaml
    │   ├── train.txt
    │   ├── val.txt
    │   ├── split_assignments.csv
    │   └── split_summary.json
    ├── statistics/
    │   ├── dataset_integrity_report.json
    │   ├── filename_mismatches.csv
    │   ├── image_issues.csv
    │   ├── label_issues.csv
    │   ├── calibration_issues.csv
    │   ├── box_warnings.csv
    │   ├── class_mapping_validation.json
    │   ├── dataset_summary.json
    │   ├── image_level_statistics.csv
    │   ├── object_level_statistics.csv
    │   ├── original_class_distribution.csv
    │   ├── mapped_class_distribution.csv
    │   ├── train_val_distribution.csv
    │   ├── bbox_size_statistics.csv
    │   ├── bbox_size_thresholds.json
    │   ├── occlusion_statistics.csv
    │   ├── truncation_statistics.csv
    │   ├── difficulty_statistics.csv
    │   └── figures/
    │       ├── original_class_distribution.png
    │       ├── mapped_class_distribution.png
    │       ├── train_val_class_distribution.png
    │       ├── bbox_area_distribution.png
    │       ├── objects_per_image_distribution.png
    │       ├── occlusion_distribution.png
    │       ├── truncation_distribution.png
    │       ├── difficulty_distribution.png
    │       └── figures_summary.json
    └── visual_checks/
        ├── sample_vehicle.jpg
        ├── sample_pedestrian.jpg
        ├── sample_cyclist.jpg
        ├── sample_mixed_scene.jpg
        ├── sample_crowded_scene.jpg
        ├── sample_occluded_scene.jpg
        ├── sample_truncated_scene.jpg
        ├── sample_dontcare_scene.jpg
        ├── visual_checks_manifest.csv
        └── visual_checks_summary.json
```

Scripts:

```text
scripts/milestone_2/kitti/
├── 01_verify_kitti_integrity.py
├── 02_validate_class_mapping.py
├── 03_create_train_val_split.py
├── 04_generate_kitti_statistics.py
├── 05_create_kitti_figures.py
└── 06_create_kitti_visual_checks.py
```

---

# Key Files for Later Milestones

Frozen split:

```text
data/kitti/selection/train.txt
data/kitti/selection/val.txt
data/kitti/selection/split_assignments.csv
data/kitti/selection/split_summary.json
```

Frozen mapping:

```text
data/kitti/selection/class_mapping.yaml
```

Raw data:

```text
data/kitti/raw/training/image_2/
data/kitti/raw/training/label_2/
data/kitti/raw/training/calib/
```

Main statistics:

```text
data/kitti/statistics/dataset_summary.json
data/kitti/statistics/image_level_statistics.csv
data/kitti/statistics/object_level_statistics.csv
```

Validation evidence:

```text
data/kitti/statistics/dataset_integrity_report.json
data/kitti/statistics/class_mapping_validation.json
data/kitti/visual_checks/
```

Milestone 3 will use these outputs to create COCO, YOLO, and framework-specific dataset formats.

---

# Reproducibility Rules

1. Do not manually modify `data/kitti/raw/`.
2. Do not change the class mapping after training begins.
3. Do not regenerate the split with another seed.
4. Use the same train and validation IDs for all models.
5. Do not use the unlabeled official test images for local metrics.
6. Do not train `DontCare` as a class.
7. Preserve ignored regions for later evaluation handling.
8. Keep all images even when they contain ignored annotations.
9. Preserve integrity, split, and mapping reports.
10. Rerun validation after copying or storage changes.
11. Inspect visual checks before annotation conversion.
12. Keep the class order fixed:
    - `0 Vehicle`
    - `1 Pedestrian`
    - `2 Cyclist`

---

# Git and Storage Notes

Do not commit raw KITTI data:

```gitignore
data/kitti/raw/
```

Optionally ignore generated image artifacts:

```gitignore
data/kitti/visual_checks/*.jpg
data/kitti/statistics/figures/*.png
```

Useful files to commit:

```text
data/kitti/README.md
data/kitti/selection/
data/kitti/statistics/*.json
data/kitti/statistics/*.csv
scripts/milestone_2/kitti/*.py
```

Check dataset terms before publishing visual samples or other dataset-derived image artifacts.

---

# Troubleshooting

## Counts are not 7,481

Confirm the source directories use `training/`, not `testing/`.

## Exactly 22,590 label issues appear

This happens when 11,295 `DontCare` rows are each flagged for both truncation and occlusion placeholder values.

Skip normal truncation and occlusion checks for `DontCare`.

## A visible object has no box

The visualizer only draws official labels. Inspect the matching `.txt` file.

## `DontCare` is shown as Ignore

Correct behavior. It is not a detector class.

## Split counts are wrong

Rerun the split script and confirm:

```text
seed = 42
train = 5985
val = 1496
```

## Leading zeros disappear

Read IDs as strings and use:

```python
.str.zfill(6)
```

## Missing Python package

```cmd
python -m pip install PyYAML scikit-learn matplotlib tqdm
```

---

# Completion Checklist

```text
[✓] Four official packages downloaded
[✓] Four packages extracted
[✓] 7,481 image_2 training images copied
[✓] 7,481 label_2 files copied
[✓] 7,481 calibration files copied
[✓] Devkit copied
[✓] File correspondence verified
[✓] PNG readability verified
[✓] 51,865 rows validated
[✓] DontCare placeholders handled
[✓] Zero invalid boxes
[✓] Zero calibration issues
[✓] Class mapping frozen
[✓] Mapping validation passed
[✓] 5,985 training IDs created
[✓] 1,496 validation IDs created
[✓] Seed fixed to 42
[✓] Split validation passed
[✓] Image-level statistics created
[✓] Object-level statistics created
[✓] Class distributions created
[✓] Box statistics created
[✓] Occlusion statistics created
[✓] Truncation statistics created
[✓] Difficulty statistics created
[✓] Statistical figures generated
[✓] Eight visual checks generated
[✓] README completed
```

---

# Completion Status

The frozen experimental setup is:

```text
KITTI labeled set
├── Train: 5,985 images
└── Validation: 1,496 images

Waymo representative subset
└── External validation: 996 FRONT-camera images
```

All models must use the same:

- KITTI training split,
- KITTI validation split,
- three-class mapping,
- and Waymo external-validation subset.
