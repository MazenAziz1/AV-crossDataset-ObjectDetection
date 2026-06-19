# \# Milestone 3 — Unified Dataset Preprocessing and Validation

#

# \## Overview

#

# Milestone 3 converts the frozen KITTI and Waymo data prepared during Milestone 2 into a unified, model-ready object-detection dataset.

#

# The milestone standardizes:

#

# \* image dimensions and aspect-ratio handling;

# \* class definitions;

# \* bounding-box coordinates;

# \* COCO annotations;

# \* YOLO annotations;

# \* evaluation-ignore regions;

# \* excluded non-target objects;

# \* training augmentation;

# \* framework configuration files;

# \* PyTorch dataset loading;

# \* validation and reproducibility procedures.

#

# No detector is trained during this milestone.

#

# Model implementation, detector integration, training, hyperparameter tuning, and evaluation belong to Milestone 4.

#

# \---

#

# \## Experimental Roles

#

# The two datasets have different experimental purposes.

#

# | Dataset partition | Experimental role                 | Training permitted | Model selection permitted |

# | ----------------- | --------------------------------- | -----------------: | ------------------------: |

# | KITTI train       | Model training                    |                Yes |                        No |

# | KITTI validation  | In-domain validation              |                 No |                       Yes |

# | Waymo external    | Cross-dataset external validation |                 No |                        No |

#

# Waymo is never used for:

#

# \* training;

# \* fine-tuning;

# \* augmentation-source sampling;

# \* hyperparameter selection;

# \* checkpoint selection;

# \* early stopping;

# \* threshold tuning.

#

# This separation is enforced by the dataset registry and the DataLoader validation checks.

#

# \---

#

# \## Source Data

#

# \### KITTI

#

# The complete official labeled KITTI object-detection training set is used.

#

# \* Images: `7,481`

# \* Camera: left color camera, stored by KITTI as `image\_2`

# \* Labels: `label\_2`

# \* Calibration: `calib`

# \* Official testing data: not used because its labels are not publicly available

#

# In KITTI terminology, `image\_2` means the images captured by the left color camera. It does not mean that the dataset contains only two images or that the images are a second copy.

#

# The KITTI images were divided once into a frozen project split:

#

# | Partition  | Images |

# | ---------- | -----: |

# | Train      |  5,985 |

# | Validation |  1,496 |

# | Total      |  7,481 |

#

# The split seed is `42`.

#

# The split must not be changed after model training begins.

#

# \### Waymo

#

# A representative subset of the official Waymo validation split is used only for external validation.

#

# \* Images: `996`

# \* Driving segments: `25`

# \* Camera: `FRONT`

# \* Sampling rule: every fifth FRONT frame, starting with the first selected frame

# \* Target-negative images: `12`

#

# The sampling policy reduces temporal redundancy while preserving variation in traffic density, location, time of day, and weather.

#

# \---

#

# \## Final Dataset Totals

#

# | Partition        |    Images | Target boxes |    Vehicle | Pedestrian |   Cyclist | Negative images |

# | ---------------- | --------: | -----------: | ---------: | ---------: | --------: | --------------: |

# | KITTI train      |     5,985 |       31,294 |     26,278 |      3,729 |     1,287 |               0 |

# | KITTI validation |     1,496 |        7,792 |      6,472 |        980 |       340 |               0 |

# | Waymo external   |       996 |       24,819 |     16,928 |      7,127 |       764 |              12 |

# | \*\*Combined\*\*     | \*\*8,477\*\* |   \*\*63,905\*\* | \*\*49,678\*\* | \*\*11,836\*\* | \*\*2,391\*\* |          \*\*12\*\* |

#

# The 12 target-negative Waymo images are intentionally preserved for false-positive analysis.

#

# \---

#

# \## Harmonized Target Classes

#

# The project uses three semantic classes.

#

# | Internal ID | COCO ID | YOLO ID | Class      |

# | ----------: | ------: | ------: | ---------- |

# |           0 |       1 |       0 | Vehicle    |

# |           1 |       2 |       1 | Pedestrian |

# |           2 |       3 |       2 | Cyclist    |

#

# \### KITTI Mapping

#

# | KITTI source class | Unified treatment          |

# | ------------------ | -------------------------- |

# | Car                | Vehicle                    |

# | Van                | Vehicle                    |

# | Truck              | Vehicle                    |

# | Pedestrian         | Pedestrian                 |

# | Person\_sitting     | Pedestrian                 |

# | Cyclist            | Cyclist                    |

# | DontCare           | Evaluation-ignore region   |

# | Tram               | Excluded non-target object |

# | Misc               | Excluded non-target object |

#

# \### Waymo Mapping

#

# | Waymo source class | Unified treatment          |

# | ------------------ | -------------------------- |

# | Vehicle            | Vehicle                    |

# | Pedestrian         | Pedestrian                 |

# | Cyclist            | Cyclist                    |

# | Sign               | Excluded non-target object |

# | Unknown            | Excluded if encountered    |

#

# The selected Waymo subset contains no Sign or Unknown boxes, but the policy remains explicitly defined.

#

# \---

#

# \## Region Policy

#

# Target annotations, evaluation-ignore regions, and excluded objects are stored separately.

#

# \### Evaluation-ignore regions

#

# KITTI `DontCare` regions are retained to suppress detections that should not count as false positives during evaluation.

#

# Total evaluation-ignore regions:

#

# ```text

# 11,295

# ```

#

# \### Excluded non-target objects

#

# KITTI `Tram` and `Misc` objects are preserved in audit sidecars but are not targets and do not suppress false positives.

#

# Total excluded non-target regions:

#

# ```text

# 1,484

# ```

#

# Waymo `Sign` is also defined as an excluded class, although none occurs in the selected external subset.

#

# \---

#

# \## Image Preprocessing

#

# All images are converted to a shared size of:

#

# ```text

# 640 × 640 pixels

# ```

#

# The preprocessing procedure is:

#

# 1\. Read the original RGB camera image.

# 2\. Preserve the original aspect ratio.

# 3\. Resize the image so its longest side fits within `640 × 640`.

# 4\. Center the resized image inside a `640 × 640` canvas.

# 5\. Fill the remaining letterbox area with pixel value `114`.

# 6\. Transform every bounding box using the same scale and padding offsets.

# 7\. Save the processed image as PNG.

#

# No image is stretched.

#

# No object box is intentionally removed during preprocessing.

#

# \### Typical transformations

#

# KITTI images are wide and receive larger top and bottom padding.

#

# Example:

#

# ```text

# Original: 1224 × 370

# Resized:  640 × 193

# Padding:  left=0, top=223, right=0, bottom=224

# ```

#

# Waymo FRONT images are less wide relative to height.

#

# Example:

#

# ```text

# Original: 1920 × 1280

# Resized:  640 × 427

# Padding:  left=0, top=106, right=0, bottom=107

# ```

#

# \---

#

# \## Canonical Annotation Format

#

# COCO is the canonical annotation representation.

#

# Files:

#

# ```text

# annotations/coco/kitti\_train.json

# annotations/coco/kitti\_val.json

# annotations/coco/waymo\_external.json

# ```

#

# Each COCO annotation uses:

#

# ```text

# \[x, y, width, height]

# ```

#

# in absolute processed-image pixel coordinates.

#

# COCO category IDs are:

#

# ```text

# 1 = Vehicle

# 2 = Pedestrian

# 3 = Cyclist

# ```

#

# \---

#

# \## Derived YOLO Annotation Format

#

# YOLO labels are derived from the canonical COCO annotations.

#

# Files:

#

# ```text

# annotations/yolo/kitti/train/

# annotations/yolo/kitti/val/

# annotations/yolo/waymo/external/

# ```

#

# Each YOLO row uses:

#

# ```text

# class\_id center\_x center\_y width height

# ```

#

# All coordinates are normalized to the processed `640 × 640` image.

#

# YOLO class IDs are:

#

# ```text

# 0 = Vehicle

# 1 = Pedestrian

# 2 = Cyclist

# ```

#

# YOLO coordinates are serialized with ten decimal places.

#

# The numerical equivalence validation proved that all `63,905` COCO boxes match their YOLO representations.

#

# Observed combined maximum errors:

#

# ```text

# Maximum normalized error: approximately 5.0 × 10^-11

# Maximum pixel error: approximately 4.8 × 10^-8 pixels

# Minimum reconstructed IoU: 0.999999785696

# ```

#

# These differences are caused only by decimal serialization.

#

# \---

#

# \## Framework Label Adapter

#

# Framework-ready YOLO labels are mirrored under:

#

# ```text

# labels/kitti/train/

# labels/kitti/val/

# labels/waymo/external/

# ```

#

# The adapter labels are byte-identical to the canonical YOLO labels.

#

# The duplicated layout exists because frameworks such as Ultralytics normally expect sibling `images` and `labels` directory structures.

#

# \---

#

# \## Augmentation Policy

#

# Augmentation is applied online during training.

#

# No permanent augmented image dataset is generated.

#

# Augmentation is enabled only for:

#

# ```text

# kitti\_train

# ```

#

# It is disabled for:

#

# ```text

# kitti\_val

# waymo\_external

# ```

#

# \### Enabled transformations

#

# | Transformation          | Probability |

# | ----------------------- | ----------: |

# | Horizontal flip         |        0.50 |

# | Brightness and contrast |        0.50 |

# | HSV adjustment          |        0.30 |

# | Mild Gaussian blur      |        0.10 |

#

# \### Disabled transformations

#

# \* vertical flipping;

# \* random cropping;

# \* arbitrary rotation;

# \* perspective distortion;

# \* random scaling;

# \* shear;

# \* Mosaic;

# \* MixUp;

# \* Copy-Paste;

# \* Cutout.

#

# These transformations are disabled to maintain a conservative and consistent policy across detector families.

#

# The same image, image ID, epoch, and base seed produce the same augmentation.

#

# The base seed is:

#

# ```text

# 42

# ```

#

# \---

#

# \## Directory Structure

#

# ```text

# data/processed/milestone\_3/

# ├── README.md

# ├── images/

# │   ├── kitti/

# │   │   ├── train/

# │   │   └── val/

# │   └── waymo/

# │       └── external/

# ├── labels/

# │   ├── kitti/

# │   │   ├── train/

# │   │   └── val/

# │   └── waymo/

# │       └── external/

# ├── annotations/

# │   ├── coco/

# │   ├── yolo/

# │   ├── ignore\_regions/

# │   └── excluded\_objects/

# ├── manifests/

# ├── reports/

# └── visual\_checks/

# ```

#

# Large processed-image and generated-label directories may be excluded from Git because they can be regenerated from the frozen sources and scripts.

#

# Configurations, scripts, reports, manifests, documentation, and selected visual-quality artifacts should remain version-controlled.

#

# \---

#

# \## Dataset Configurations

#

# Configuration files are stored under:

#

# ```text

# configs/datasets/milestone\_3/

# ```

#

# Important files:

#

# ```text

# preprocessing.yaml

# class\_mapping.yaml

# augmentation.yaml

# kitti\_waymo\_yolo.yaml

# coco\_paths.yaml

# dataset\_registry.yaml

# ```

#

# \### Ultralytics configuration

#

# ```text

# configs/datasets/milestone\_3/kitti\_waymo\_yolo.yaml

# ```

#

# The configuration defines:

#

# ```text

# train = KITTI train

# val   = KITTI validation

# test  = Waymo external

# ```

#

# The `test` path is external validation only. It must not influence model development.

#

# \---

#

# \## PyTorch Dataset Loader

#

# The framework-neutral PyTorch loader is implemented in:

#

# ```text

# scripts/milestone\_3/dataset\_core.py

# ```

#

# Available partitions:

#

# ```python

# kitti\_train

# kitti\_val

# waymo\_external

# ```

#

# Example:

#

# ```python

# from scripts.milestone\_3.dataset\_core import (

# &#x20;   Milestone3DetectionDataset,

# &#x20;   detection\_collate\_fn,

# )

#

# dataset = Milestone3DetectionDataset(

# &#x20;   partition\_name="kitti\_val",

# &#x20;   enable\_augmentation=False,

# )

#

# image, target = dataset\[0]

# ```

#

# The returned image is a float tensor with shape:

#

# ```text

# \[3, 640, 640]

# ```

#

# and range:

#

# ```text

# \[0, 1]

# ```

#

# Targets contain:

#

# ```text

# boxes

# labels

# image\_id

# area

# iscrowd

# ignore\_boxes

# excluded\_boxes

# size

# image\_path

# file\_name

# source\_dataset

# source\_image\_id

# partition

# role

# augmentation\_trace

# ```

#

# Canonical target labels preserve COCO and Torchvision-compatible IDs:

#

# ```text

# 1 = Vehicle

# 2 = Pedestrian

# 3 = Cyclist

# ```

#

# \---

#

# \## Validation Results

#

# The complete pipeline passed all validation stages.

#

# \### Final audit

#

# ```text

# Previous validation reports passed: 14 / 14

# Frozen configurations passed:      6 / 6

# Generated manifests passed:        9 / 9

# Unresolved issue rows:              0

# Final audit status:                 PASSED

# ```

#

# \### Verified properties

#

# \* `8,477` processed images exist.

# \* Every processed image is `640 × 640`.

# \* `63,905` target boxes are preserved.

# \* COCO and YOLO totals match.

# \* All COCO and YOLO boxes are valid and in bounds.

# \* Canonical and framework YOLO files match.

# \* KITTI train and validation contain no overlapping image IDs.

# \* Waymo external does not overlap with KITTI.

# \* The 12 Waymo negative images contain empty target labels.

# \* Visual COCO and YOLO overlays coincide.

# \* Online augmentation preserves box counts and class IDs.

# \* Validation and external loaders are deterministic.

# \* Augmentation cannot be enabled for validation partitions.

# \* No unresolved validation issue remains.

#

# \---

#

# \## Reproducibility Runner

#

# The complete pipeline can be listed with:

#

# ```cmd

# python scripts\\milestone\_3\\run\_milestone\_3.py --list

# ```

#

# \### Safe validation-only run

#

# ```cmd

# python scripts\\milestone\_3\\run\_milestone\_3.py --validate-only

# ```

#

# This reruns the non-destructive validation stages without rebuilding the complete processed image dataset.

#

# \### Validation dry run

#

# ```cmd

# python scripts\\milestone\_3\\run\_milestone\_3.py --validate-only --dry-run

# ```

#

# \### Full pipeline

#

# The full pipeline is protected by an explicit confirmation token:

#

# ```cmd

# python scripts\\milestone\_3\\run\_milestone\_3.py --full --confirm RUN\_MILESTONE\_3\_FULL

# ```

#

# \### Clean regeneration

#

# A clean regeneration may delete and recreate generated outputs. It requires a stronger token:

#

# ```cmd

# python scripts\\milestone\_3\\run\_milestone\_3.py --full --clean-generated --confirm REGENERATE\_MILESTONE\_3

# ```

#

# Clean regeneration should be used only when:

#

# \* all frozen source data is available;

# \* sufficient disk space is available;

# \* regeneration is intentional;

# \* existing generated outputs may safely be replaced.

#

# \---

#

# \## Reproducibility Result

#

# The validation-only runner completed successfully:

#

# ```text

# Selected stages:     9

# Passed stages:       9

# Failed stages:       0

# Final dataset audit: PASSED

# Overall status:      PASSED

# ```

#

# Generated reproducibility records:

#

# ```text

# reports/reproducibility\_report.json

# manifests/reproducibility\_run\_manifest.csv

# ```

#

# \---

#

# \## Visual Quality Checks

#

# The main visual artifacts are:

#

# ```text

# visual\_checks/preprocessing\_dry\_run/

# visual\_checks/coco\_yolo\_comparison/

# visual\_checks/augmentation\_policy/

# ```

#

# Important contact sheets:

#

# ```text

# visual\_checks/coco\_yolo\_comparison/annotation\_comparison\_contact\_sheet.png

# visual\_checks/augmentation\_policy/augmentation\_policy\_contact\_sheet.png

# ```

#

# The visual checks confirmed:

#

# \* correct letterboxing;

# \* correct transformed coordinates;

# \* COCO–YOLO visual equivalence;

# \* correct cyclist and pedestrian annotations;

# \* preserved negative scenes;

# \* correct horizontal-flip behavior;

# \* unchanged letterbox padding;

# \* realistic photometric augmentation.

#

# \---

#

# \## Generated Reports

#

# Reports are stored under:

#

# ```text

# data/processed/milestone\_3/reports/

# ```

#

# Important reports include:

#

# ```text

# source\_input\_validation.json

# source\_manifest\_summary.json

# preprocessing\_dry\_run.json

# image\_preprocessing\_report.json

# coco\_creation\_report.json

# region\_policy\_report.json

# yolo\_conversion\_report.json

# config\_validation.json

# coco\_validation\_report.json

# yolo\_validation\_report.json

# coco\_yolo\_equivalence\_report.json

# visual\_annotation\_checks\_report.json

# augmentation\_policy\_report.json

# dataloader\_validation\_report.json

# final\_dataset\_audit.json

# reproducibility\_report.json

# ```

#

# Each validation stage also generates an issue CSV. A successful stage leaves its issue CSV with a header and zero issue rows.

#

# \---

#

# \## Generated Manifests

#

# Important manifests include:

#

# ```text

# source\_manifest.csv

# transform\_manifest.csv

# region\_policy\_manifest.csv

# yolo\_label\_manifest.csv

# framework\_label\_adapter\_manifest.csv

# coco\_yolo\_equivalence\_manifest.csv

# visual\_annotation\_checks\_manifest.csv

# augmentation\_policy\_manifest.csv

# dataloader\_smoke\_test\_manifest.csv

# reproducibility\_run\_manifest.csv

# ```

#

# The region-policy manifest contains one row per image, not one row per region.

#

# Therefore:

#

# ```text

# region\_policy\_manifest.csv rows = 8,477

# ```

#

# The combined region totals are validated separately through the sidecar files:

#

# ```text

# Evaluation-ignore regions = 11,295

# Excluded non-target regions = 1,484

# ```

#

# \---

#

# \## Tested Environment

#

# The final validation was executed using:

#

# ```text

# PyTorch:      2.11.0+cu128

# Torchvision:  0.26.0+cu128

# Ultralytics:  8.4.60

# CUDA:         available

# GPU:          NVIDIA GeForce RTX 3060 Laptop GPU

# ```

#

# Milestone 3 does not execute or train the four detector families.

#

# Ultralytics is present because the project will use it during the model implementation stage and because its dataset configuration format is prepared here.

#

# \---

#

# \## Scope Boundary

#

# Milestone 3 ends with a validated, reproducible, framework-ready dataset.

#

# The following tasks are intentionally deferred to Milestone 4:

#

# \* initializing detector architectures;

# \* downloading pretrained weights;

# \* detector-interface smoke tests;

# \* selecting model variants;

# \* training;

# \* checkpoint creation;

# \* hyperparameter tuning;

# \* in-domain evaluation;

# \* external Waymo evaluation;

# \* inference-speed benchmarking;

# \* comparison of YOLO, Faster R-CNN, RetinaNet, and RT-DETR.

#

# No model-performance claim is made by Milestone 3.

#

# \---

#

# \## Completion Status

#

# ```text

# Dataset preprocessing:       COMPLETE

# Annotation harmonization:    COMPLETE

# COCO generation:             COMPLETE

# YOLO conversion:             COMPLETE

# Region-policy preservation:  COMPLETE

# Augmentation policy:         COMPLETE

# DataLoader validation:       COMPLETE

# Final dataset audit:         PASSED

# Reproducibility validation:  PASSED

# Model training:              NOT PART OF THIS MILESTONE

# ```
