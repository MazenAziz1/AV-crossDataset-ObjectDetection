# AV Cross-Dataset Object Detection for Autonomous-Driving Perception

This repository contains a milestone-driven autonomous-driving object-detection study focused on **cross-dataset generalization**, **safety-oriented failure analysis**, and **deployment-aware evaluation**. The final manuscript integrates two coordinated implementation tracks into an **eight-instance** comparison: two independent implementations of four detector families evaluated under a shared KITTI-to-Waymo protocol.

The work asks a practical question: **does strong in-domain KITTI performance remain reliable when the same locked detector is evaluated directly on Waymo without retraining or target-domain tuning?**

---

## 1. Project Summary

### Core idea

- Train object detectors using a harmonized KITTI source-domain pipeline.
- Evaluate in-domain on a frozen KITTI validation split.
- Evaluate externally on a representative Waymo FRONT-camera subset.
- Keep Waymo fully isolated from training, threshold tuning, hyperparameter selection, checkpoint selection, or model reselection.
- Analyze results beyond mAP using class-wise degradation, generalization ratios, vulnerable-road-user false negatives, object-size robustness, localization errors, class confusion, duplicate detections, latency, FPS, and deployment-support scoring.

### Shared dataset contract

| Partition | Role | Images | Target boxes | Use |
|---|---:|---:|---:|---|
| KITTI train | Source-domain training | 5,985 | 31,294 | Parameter learning only |
| KITTI validation | In-domain validation | 1,496 | 7,792 | In-domain evaluation |
| Waymo external subset | External validation | 996 | 24,819 | Direct transfer and safety stress test |

### Unified classes

| Source label group | Unified class | Action |
|---|---|---|
| KITTI Car, Van, Truck; Waymo Vehicle | Vehicle | Mapped to common target class |
| KITTI Pedestrian, Person_sitting; Waymo Pedestrian | Pedestrian | Mapped to common target class |
| KITTI Cyclist; Waymo Cyclist | Cyclist | Mapped to common target class |
| KITTI Tram, Misc, DontCare; Waymo Sign | Ignored / ignore region | Excluded from the three-class target task |

---

## 2. Eight Detector Instances

The final integrated manuscript treats the two coordinated tracks as **eight detector instances**, not as two separate four-model studies.

| Instance | Track | Family | Model/setup label |
|---|---|---|---|
| A-YOLOv8s | A | Real-time one-stage CNN | YOLOv8s |
| A-RT-DETR-L | A | Real-time transformer | RT-DETR-L |
| A-Faster R-CNN | A | Two-stage proposal-based CNN | Faster R-CNN |
| A-RetinaNet | A | Dense one-stage focal-loss CNN | RetinaNet |
| B-YOLO26s | B | Real-time one-stage CNN | YOLO26s |
| B-RT-DETR-L | B | Real-time transformer | RT-DETR-L |
| B-RetinaNet | B | Dense one-stage focal-loss CNN | RetinaNet |
| B-Faster R-CNN | B | Two-stage proposal-based CNN | Faster R-CNN |

### Track A training setup

| Detector | Physical batch | Gradient accumulation | Effective batch | Epochs | Image size |
|---|---:|---:|---:|---:|---|
| A-YOLOv8s | 32 | 1 | 32 | 200 | 640 x 640 |
| A-Faster R-CNN | 4 | 8 | 32 | 200 | 640 x 640 |
| A-RetinaNet | 4 | 8 | 32 | 200 | 640 x 640 |
| A-RT-DETR-L | 16 | 1 | 16 | 200 | 640 x 640 |

Effective batch size is:

```text
B_effective = B_batch × N_accumulation
```

### Track B training setup

Track B uses a 640 x 640 input size, 120 epochs, patience 20, seed 42, and detector-specific optimizer/batch settings.

| Detector | Model | Optimizer | LR | Momentum | Weight decay | Physical batch | Grad. accum. | Effective batch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B-YOLO26s | `yolo26s.pt` | auto | Ultralytics auto | - | - | 16 | 1 | 16 |
| B-RT-DETR-L | `rtdetr-l.pt` | auto | Ultralytics auto | - | - | 8 | 1 | 8 |
| B-Faster R-CNN | `fasterrcnn_resnet50_fpn` | SGD | 0.005 | 0.9 | 0.0005 | 16 | 1 | 16 |
| B-RetinaNet | `retinanet_resnet50_fpn` | AdamW | 0.0001 | - | 0.0001 | 16 | 1 | 16 |

### Shared class contract

```yaml
torchvision_num_classes: 4
torchvision_background_id: 0
yolo_num_classes: 3
class_names:
  - Vehicle
  - Pedestrian
  - Cyclist
```

---

## 3. Repository Layout (close enough but not similar)

```text
.
├── configs/
│   ├── datasets/
│   │   └── milestone_3/
|   |   └── milestone_6/
│   ├── models/
│   │   └── milestone_4/
│   └── analysis/
│   |    └── milestone_7/
|   └── evaluation/
│       └── milestone_6/
├── data/
│   ├── kitti raw/                      # not tracked in Git
│   ├── waymo/raw/                      # not tracked in Git
│   └── processed/milestone_3/          # generated dataset handoff
├── docs/
│   ├── milestone_1/                    # literature and review material
│   ├── milestone_2/                    # dataset selection/preparation reports
│   ├── milestone_3/                    # preprocessing and annotation reports
│   ├── milestone_4/                    # training package and model reports
│   ├── milestone_5/                    # KITTI validation reports
│   ├── milestone_6/                    # Waymo external validation reports
│   ├── milestone_7/                    # robustness/failure/safety reports
│   └── milestone_8/                    # paper drafts, algorithms, final manuscript
├── outputs/
│   ├── milestone_4/                    # checkpoints, registries, Kaggle packages
│   ├── milestone_5/                    # KITTI metrics/predictions/tables
│   ├── milestone_6/                    # Waymo metrics/predictions/tables/figures
│   └── milestone_7/                    # robustness, failure, safety, figures
├── scripts/
│   ├── milestone_2/
│   ├── milestone_3/
│   ├── milestone_4/
│   ├── milestone_5/
│   ├── milestone_6/
│   └── milestone_7/
├── requirements.txt
└── README.md
```

> **Important path note:** some later scripts contain a hard-coded Windows project root. If your clone is in another location, update the `PROJECT = Path(...)` line inside the relevant scripts before running them.

---

## 4. Main Results Snapshot

| Result type | Main takeaway |
|---|---|
| In-domain KITTI | YOLO-family instances dominate mAP50-95. |
| External Waymo | RetinaNet variants retain the strongest external mAP/generalization stability. |
| Generalization | All detectors degrade severely under direct KITTI-to-Waymo transfer. |
| Safety | Vulnerable-road-user false negatives remain high for every instance. |
| Deployment | B-RT-DETR-L gives the strongest safety-weighted decision-support score, while B-YOLO26s is fastest. |
| Interpretation | Results are robustness evidence and decision support, **not deployment certification**. |

---

## 5. Paper Figures

All paper figures are included below so the root README gives a visual overview of the full project.

### Figure 1. Dataset preparation and evaluation protocol

![Figure 1](C:\Users\Mazen\Desktop\AAST\Research\Autonomous research\docs\figures\figure_01_dataset_preparation_evaluation_protocol.png)

### Figure 2. Unified preprocessing and validation pipeline

![Figure 2](docs\figures\figure_02_unified_preprocessing_validation_pipeline.png)

### Figure 3. KITTI in-domain mAP50-95 across eight detector instances

![Figure 3](docs\figures\figure_03_kitti_map50_95_eight_instances.png)

### Figure 4. Waymo external mAP50-95 across eight detector instances

![Figure 4](docs\figures\figure_04_waymo_map50_95_eight_instances.png)

### Figure 5. Waymo class-wise AP50-95 across eight detector instances

![Figure 5](docs\figures\figure_05_waymo_classwise_ap50_95.png)

### Figure 6. KITTI-to-Waymo mAP50-95 generalization ratio

![Figure 6](docs\figures\figure_06_generalization_ratio.png)

### Figure 7. Waymo vulnerable-road-user false-negative rate

![Figure 7](docs\figures\figure_07_waymo_vru_fnr.png)

### Figure 8. Waymo small vulnerable-road-user recall

![Figure 8](docs\figures\figure_08_waymo_small_vru_recall.png)

### Figure 9. Waymo failure-event burden

![Figure 9](docs\figures\figure_09_waymo_failure_event_burden.png)

### Figure 10. Representative visual safety check

![Figure 10](docs\figures\figure_10_visual_safety_check.png)

### Figure 11. Safety-weighted deployment-support score

![Figure 11](docs\figures\figure_11_deployment_support_score.png)

### Figure 12. Waymo inference latency

![Figure 12](docs\figures\figure_12_waymo_latency.png)

### Figure 13. Waymo accuracy-latency trade-off

![Figure 13](docs\figures\figure_13_accuracy_latency_tradeoff.png)

---

## 6. Environment Setup Commands 

The commands below use **Windows CMD** and assume they are executed from the repository root unless otherwise stated.

> Go and checkout the **setUP_Steps.md** file in the root, as it will provide all the environment configuration steps in detail.

## 7. Milestone 2 Commands — Dataset Inspection and Selection

### 7.1 Verify KITTI integrity

Needed files/folders before this command:

- `scripts\milestone_2\kitti\01_verify_kitti_integrity.py`
- `data\kitti\raw\training\image_2`
- `data\kitti\raw\training\label_2`
- `data\kitti\raw\training\calib`
- Optional: `data\kitti\raw\devkit_object`

Command:

```cmd
python scripts\milestone_2\kitti\01_verify_kitti_integrity.py
```

### 7.2 Validate KITTI class mapping

Needed files/folders before this command:

- `scripts\milestone_2\kitti\02_validate_class_mapping.py`
- KITTI integrity outputs in `data\kitti\statistics`
- KITTI raw labels in `data\kitti\raw\training\label_2`

Command:

```cmd
python scripts\milestone_2\kitti\02_validate_class_mapping.py
```

### 7.3 Create the frozen KITTI train/validation split

Needed files/folders before this command:

- `scripts\milestone_2\kitti\03_create_train_val_split.py`
- KITTI raw data under `data\kitti\raw\training`
- Class-mapping validation output from the previous step

Command:

```cmd
python scripts\milestone_2\kitti\03_create_train_val_split.py
```

### 7.4 Generate KITTI statistics

Needed files/folders before this command:

- `scripts\milestone_2\kitti\04_generate_kitti_statistics.py`
- `data\kitti\selection\train.txt`
- `data\kitti\selection\val.txt`
- KITTI raw labels in `data\kitti\raw\training\label_2`

Command:

```cmd
python scripts\milestone_2\kitti\04_generate_kitti_statistics.py
```

### 7.5 Create KITTI figures

Needed files/folders before this command:

- `scripts\milestone_2\kitti\05_create_kitti_figures.py`
- KITTI statistics outputs in `data\kitti\statistics`

Command:

```cmd
python scripts\milestone_2\kitti\05_create_kitti_figures.py
```

### 7.6 Create KITTI visual checks

Needed files/folders before this command:

- `scripts\milestone_2\kitti\06_create_kitti_visual_checks.py`
- KITTI raw images in `data\kitti\raw\training\image_2`
- KITTI raw labels in `data\kitti\raw\training\label_2`
- KITTI split files in `data\kitti\selection`

Command:

```cmd
python scripts\milestone_2\kitti\06_create_kitti_visual_checks.py
```

### 7.7 Inspect Waymo validation statistics

Needed files/folders before this command:

- `scripts\milestone_2\waymo\01_inspect_waymo_stats.py`
- Waymo validation stats Parquet files in `data\waymo\raw\validation\stats`
- `pyarrow` installed

Command:

```cmd
python scripts\milestone_2\waymo\01_inspect_waymo_stats.py
```

### 7.8 Build Waymo segment catalog

Needed files/folders before this command:

- `scripts\milestone_2\waymo\02_build_waymo_segment_catalog.py`
- Waymo validation stats Parquet files in `data\waymo\raw\validation\stats`
- Output from the stats schema inspection is recommended

Command:

```cmd
python scripts\milestone_2\waymo\02_build_waymo_segment_catalog.py
```

### 7.9 Select Waymo candidate segments

Needed files/folders before this command:

- `scripts\milestone_2\waymo\03_select_candidate_segments.py`
- Waymo segment catalog generated by the previous command

Command:

```cmd
python scripts\milestone_2\waymo\03_select_candidate_segments.py
```

### 7.10 Download candidate camera boxes

Needed files/folders before this command:

- `scripts\milestone_2\waymo\04_download_candidate_camera_boxes.py`
- Selected candidate segment list from the previous command
- Waymo Open Dataset access configured locally
- Internet/storage access as required by the Waymo download path

Command:

```cmd
python scripts\milestone_2\waymo\04_download_candidate_camera_boxes.py
```

### 7.11 Inspect Waymo camera-box schema

Needed files/folders before this command:

- `scripts\milestone_2\waymo\05_inspect_camera_box_schema.py`
- Downloaded candidate camera-box files

Command:

```cmd
python scripts\milestone_2\waymo\05_inspect_camera_box_schema.py
```

### 7.12 Analyze candidate FRONT-camera boxes

Needed files/folders before this command:

- `scripts\milestone_2\waymo\06_analyze_candidate_front_boxes.py`
- Downloaded candidate camera-box files
- Schema inspection output is recommended

Command:

```cmd
python scripts\milestone_2\waymo\06_analyze_candidate_front_boxes.py
```

### 7.13 Select final Waymo segments

Needed files/folders before this command:

- `scripts\milestone_2\waymo\07_select_final_segments.py`
- Candidate segment metadata and FRONT-camera box analysis outputs

Command:

```cmd
python scripts\milestone_2\waymo\07_select_final_segments.py
```

### 7.14 Download final Waymo images and calibration

Needed files/folders before this command:

- `scripts\milestone_2\waymo\08_download_final_images_and_calibration.py`
- Final selected segment list from the previous command
- Waymo Open Dataset access configured locally

Command:

```cmd
python scripts\milestone_2\waymo\08_download_final_images_and_calibration.py
```

### 7.15 Extract representative Waymo subset

Needed files/folders before this command:

- `scripts\milestone_2\waymo\10_extract_representative_subset.py`
- Downloaded final Waymo images/calibration files
- Final Waymo 2D camera boxes

Command:

```cmd
python scripts\milestone_2\waymo\10_extract_representative_subset.py
```

### 7.16 Validate and visualize the Waymo subset

Needed files/folders before this command:

- `scripts\milestone_2\waymo\11_validate_and_visualize_subset.py`
- Extracted representative Waymo images
- Extracted representative Waymo labels/boxes

Command:

```cmd
python scripts\milestone_2\waymo\11_validate_and_visualize_subset.py
```

---

## 8. Milestone 3 Commands — Unified Preprocessing and Annotation Pipeline

### 8.1 List pipeline stages

Needed files/folders before this command:

- `scripts\milestone_3\run_milestone_3.py`
- All individual Milestone 3 stage scripts under `scripts\milestone_3`

Command:

```cmd
python scripts\milestone_3\run_milestone_3.py --list
```

### 8.2 Run validation-only Milestone 3 checks

Needed files/folders before this command:

- `scripts\milestone_3\run_milestone_3.py`
- Milestone 2 outputs: KITTI split files, KITTI statistics, Waymo representative subset, and class mapping files
- Dataset configs in `configs\datasets\milestone_3`

Command:

```cmd
python scripts\milestone_3\run_milestone_3.py --validate-only
```

### 8.3 Run the full Milestone 3 generation pipeline

Needed files/folders before this command:

- `scripts\milestone_3\run_milestone_3.py`
- Raw KITTI data in `data\kitti\raw\training`
- Milestone 2 KITTI split files in `data\kitti\selection`
- Waymo representative subset inputs from Milestone 2
- Dataset configs in `configs\datasets\milestone_3`

Command:

```cmd
python scripts\milestone_3\run_milestone_3.py --full --confirm RUN_MILESTONE_3_FULL
```

### 8.4 Regenerate Milestone 3 outputs from a clean state

Needed files/folders before this command:

- Same files as the full Milestone 3 generation command
- Existing generated outputs may be overwritten/regenerated

Command:

```cmd
python scripts\milestone_3\run_milestone_3.py --full --clean-generated --confirm REGENERATE_MILESTONE_3
```

---

## 9. Milestone 4 Commands — Kaggle Training Package and Final Training

### 9.1 Validate the Milestone 3 handoff

Needed files/folders before this command:

- `scripts\milestone_4\01_validate_milestone_3_handoff.py`
- Processed Milestone 3 dataset under `data\processed\milestone_3`
- Dataset configs under `configs\datasets\milestone_3`
- Milestone 4 model configs under `configs\models\milestone_4`

Command:

```cmd
python scripts\milestone_4\01_validate_milestone_3_handoff.py
```

### 9.2 Define the Kaggle upload set

Needed files/folders before this command:

- `scripts\milestone_4\02_define_kaggle_upload_set.py`
- Validated Milestone 3 processed data
- Milestone 4 config files under `configs\models\milestone_4`

Command:

```cmd
python scripts\milestone_4\02_define_kaggle_upload_set.py
```

### 9.3 Prepare the Kaggle training package

Needed files/folders before this command:

- `scripts\milestone_4\kaggle\02_prepare_kaggle_training_package.py`
- Upload-set manifest at `outputs\milestone_4\manifests\kaggle_upload_set_manifest.json`
- All source files listed in the upload-set manifest

Command:

```cmd
python scripts\milestone_4\kaggle\02_prepare_kaggle_training_package.py
```

### 9.4 Install Ultralytics inside Kaggle

Needed files/folders before this command:

- Kaggle notebook kernel is running.
- Internet/package installation is enabled or package cache is available.

Command:

```python
import sys
!{sys.executable} -m pip install -q ultralytics==8.4.117
```

### 9.5 Kaggle dry-run before training

Needed files/folders before this command:

- Kaggle dataset package is attached.
- `/kaggle/working/project/milestone4_kaggle_training_package` exists.
- `scripts/milestone_4/train.py` exists inside the Kaggle working package.
- `configs/models/milestone_4/training_runs.yaml` exists.
- `configs/models/milestone_4/detector_hyperparameters.yaml` exists.

Command:

```python
%cd /kaggle/working/project/milestone4_kaggle_training_package
!python scripts/milestone_4/train.py --detector retinanet --run-type final --slot slot_a --device 0 --dry-run
```

### 9.6 Kaggle resume-aware RetinaNet final training

Needed files/folders before this command:

- Kaggle working package exists at `/kaggle/working/project/milestone4_kaggle_training_package`.
- `scripts/milestone_4/train.py` exists inside the working package.
- Processed KITTI train/val data exists inside the working package.
- The checkpoint passed to `--resume` exists.
- GPU device `0` is available.

Command:

```python
%cd /kaggle/working/project/milestone4_kaggle_training_package
!python scripts/milestone_4/train.py \
  --detector retinanet \
  --run-type final_resume_if_needed \
  --slot slot_a \
  --device 0 \
  --resume "/kaggle/working/project/milestone4_kaggle_training_package/outputs/milestone_4/checkpoints/retinanet/retinanet_final_resume_if_needed_20260814_075644/last.pth" \
  --max-runtime-hours 10.5 \
  --package-on-exit
```

### 9.7 Generic Kaggle training command template

Needed files/folders before this command:

- Kaggle working package exists.
- `scripts/milestone_4/train.py` exists.
- Model configs and processed data exist inside the package.
- For `--resume latest`, at least one previous final checkpoint exists under `outputs/milestone_4/checkpoints/<detector>`.

Command:

```python
%cd /kaggle/working/project/milestone4_kaggle_training_package
!python scripts/milestone_4/train.py --detector <yolo|rtdetr|retinanet|faster_rcnn> --run-type final_resume_if_needed --slot <slot_a|slot_b> --device 0 --resume latest --package-on-exit
```

---

## 10. Milestone 5 Commands — Final KITTI Validation

### 10.1 Check local evaluation inputs

Needed files/folders before this command:

- `scripts\milestone_5\00_check_local_eval_inputs.py`
- `outputs\milestone_4\locked_final_checkpoints\final_checkpoint_registry.json`
- Processed KITTI validation images in `data\processed\milestone_3\images\kitti\val`
- Processed KITTI validation labels in `data\processed\milestone_3\labels\kitti\val`
- Final checkpoints referenced by the registry

Command:

```cmd
python scripts\milestone_5\00_check_local_eval_inputs.py
```

### 10.2 Inspect evaluation formats

Needed files/folders before this command:

- `scripts\milestone_5\01_inspect_eval_formats.py`
- Final checkpoint registry
- KITTI validation images and labels

Command:

```cmd
python scripts\milestone_5\01_inspect_eval_formats.py
```

### 10.3 Smoke-test model loading

Needed files/folders before this command:

- `scripts\milestone_5\02_smoke_load_final_models.py`
- Final checkpoint registry
- At least one KITTI validation image
- Final model checkpoint files referenced by the registry

Command:

```cmd
python scripts\milestone_5\02_smoke_load_final_models.py
```

### 10.4 Run final KITTI validation for all detectors

Needed files/folders before this command:

- `scripts\milestone_5\03_run_final_kitti_validation.py`
- Final checkpoint registry
- KITTI validation images and labels
- CUDA-capable environment recommended for speed

Command:

```cmd
python scripts\milestone_5\03_run_final_kitti_validation.py --detector all --device auto --conf 0.001 --imgsz 640
```

### 10.5 Create benchmark comparison outputs

Needed files/folders before this command:

- `scripts\milestone_5\04_create_benchmark_outputs.py`
- KITTI validation summary table at `outputs\milestone_5\final_kitti_validation\tables\comparison_summary_full.csv`

Command:

```cmd
python scripts\milestone_5\04_create_benchmark_outputs.py
```

### 10.6 Run Milestone 5 final audit

Needed files/folders before this command:

- `scripts\milestone_5\06_final_audit.py`
- KITTI validation metrics, tables, benchmark outputs, and checkpoint registry

Command:

```cmd
python scripts\milestone_5\06_final_audit.py
```

---

## 11. Milestone 6 Commands — Waymo External Validation

### 11.1 Validate Waymo handoff

Needed files/folders before this command:

- `scripts\milestone_6\00_validate_waymo_handoff.py`
- Waymo external images in `data\processed\milestone_3\images\waymo\external`
- Waymo external labels in `data\processed\milestone_3\labels\waymo\external`

Command:

```cmd
python scripts\milestone_6\00_validate_waymo_handoff.py
```

### 11.2 Create Milestone 6 configs

Needed files/folders before this command:

- `scripts\milestone_6\01_create_milestone_6_configs.py`
- Milestone 3 processed Waymo external subset exists

Command:

```cmd
python scripts\milestone_6\01_create_milestone_6_configs.py
```

### 11.3 Smoke-test Waymo models

Needed files/folders before this command:

- `scripts\milestone_6\02_smoke_test_waymo_models.py`
- Final checkpoint registry
- Waymo external images
- Final model checkpoints

Command:

```cmd
python scripts\milestone_6\02_smoke_test_waymo_models.py
```

### 11.4 Run Waymo external validation

Needed files/folders before this command:

- `scripts\milestone_6\03_run_waymo_external_validation.py`
- Final checkpoint registry
- Waymo external images and labels
- CUDA-capable environment recommended

Command:

```cmd
python scripts\milestone_6\03_run_waymo_external_validation.py
```

### 11.5 Create generalization analysis

Needed files/folders before this command:

- `scripts\milestone_6\04_create_generalization_analysis.py`
- KITTI validation summary table from Milestone 5
- Waymo external summary table from Milestone 6

Command:

```cmd
python scripts\milestone_6\04_create_generalization_analysis.py
```

### 11.6 Create Waymo/generalization figures

Needed files/folders before this command:

- `scripts\milestone_6\05_create_waymo_generalization_figures.py`
- Milestone 6 generalization-analysis tables
- Waymo external summary table

Command:

```cmd
python scripts\milestone_6\05_create_waymo_generalization_figures.py
```

### 11.7 Run Milestone 6 final audit

Needed files/folders before this command:

- `scripts\milestone_6\07_final_audit.py`
- Milestone 6 configs, metrics, tables, figures, and report outputs

Command:

```cmd
python scripts\milestone_6\07_final_audit.py
```

---

## 12. Milestone 7 Commands — Robustness, Safety, and Failure Analysis

### 12.1 Validate Milestone 7 inputs

Needed files/folders before this command:

- `scripts\milestone_7\00_validate_milestone_7_inputs.py`
- KITTI validation images and labels
- Waymo external images and labels
- Milestone 5 KITTI predictions and summaries
- Milestone 6 Waymo predictions and summaries
- Final checkpoint registry

Command:

```cmd
python scripts\milestone_7\00_validate_milestone_7_inputs.py
```

### 12.2 Create Milestone 7 configs

Needed files/folders before this command:

- `scripts\milestone_7\01_create_milestone_7_configs.py`
- Milestone 7 workspace can be created under `configs\analysis\milestone_7`
- Milestone 5 and 6 summary tables are recommended for consistency

Command:

```cmd
python scripts\milestone_7\01_create_milestone_7_configs.py
```

### 12.3 Build the detection-error index

Needed files/folders before this command:

- `scripts\milestone_7\02_build_detection_error_index.py`
- KITTI validation images and labels
- Waymo external images and labels
- KITTI prediction JSONL files from Milestone 5
- Waymo prediction JSONL files from Milestone 6

Command:

```cmd
python scripts\milestone_7\02_build_detection_error_index.py
```

### 12.4 Run object-size analysis

Needed files/folders before this command:

- `scripts\milestone_7\03_object_size_analysis.py`
- Detection-error index at `outputs\milestone_7\safety_error_analysis\detection_error_index.csv`

Command:

```cmd
python scripts\milestone_7\03_object_size_analysis.py
```

### 12.5 Run safety false-negative analysis

Needed files/folders before this command:

- `scripts\milestone_7\04_safety_false_negative_analysis.py`
- Detection-error index from Step 12.3

Command:

```cmd
python scripts\milestone_7\04_safety_false_negative_analysis.py
```

### 12.6 Run failure-type analysis

Needed files/folders before this command:

- `scripts\milestone_7\05_failure_type_analysis.py`
- Detection-error index from Step 12.3

Command:

```cmd
python scripts\milestone_7\05_failure_type_analysis.py
```

### 12.7 Create the failure-case gallery

Needed files/folders before this command:

- `scripts\milestone_7\06_create_failure_case_gallery.py`
- Detection-error index
- Failure-case candidate rows
- KITTI and Waymo validation images

Command:

```cmd
python scripts\milestone_7\06_create_failure_case_gallery.py
```

### 12.8 Create deployment trade-off analysis

Needed files/folders before this command:

- `scripts\milestone_7\07_deployment_tradeoff_analysis.py`
- Milestone 5 KITTI comparison summary
- Milestone 6 Waymo external summary
- Milestone 7 object-size, safety, and failure-type summaries

Command:

```cmd
python scripts\milestone_7\07_deployment_tradeoff_analysis.py
```

### 12.9 Create figures and report bundle

Needed files/folders before this command:

- `scripts\milestone_7\08_create_figures_and_report_bundle.py`
- Milestone 5, 6, and 7 summary outputs
- Failure-case panels
- Deployment trade-off summary

Command:

```cmd
python scripts\milestone_7\08_create_figures_and_report_bundle.py
```

### 12.10 Run final Milestone 7 audit

Needed files/folders before this command:

- `scripts\milestone_7\09_final_audit.py`
- Milestone 7 configs, summaries, figures, panels, report, and manifests

Command:

```cmd
python scripts\milestone_7\09_final_audit.py
```

---

## 13. Git Safety Commands Used in This Project

### 13.1 Check the working tree

Needed files/folders before this command:

- Repository has been initialized as a Git repository

Command:

```cmd
git status --short
```

### 13.2 Check staged files for forbidden large artifacts

Needed files/folders before this command:

- Files have already been staged with `git add`
- Git is available in the terminal

Command:

```cmd
git diff --cached --name-only | findstr /R "\.pt$ \.pth$ \.zip$ \.jsonl$ detection_error_index.csv failure_cases\\images"
```

Expected output:

```text
No output means the staged set passed this safety check.
```

### 13.3 Commit safe generated analysis artifacts

Needed files/folders before this command:

- Safe outputs are staged
- Forbidden files are not staged
- Safety check from the previous command returns no output

Command:

```cmd
git commit -m "Add Milestone 7 generated analysis artifacts"
```

### 13.4 Push to GitHub

Needed files/folders before this command:

- A local commit exists
- Remote `origin` points to the GitHub repository
- GitHub authentication is configured

Command:

```cmd
git push origin main
```

---

## 14. Files That Should Usually Stay Out of Git

These files are large, regenerated, or not suitable for direct repository tracking:

```text
*.pt
*.pth
*.zip
*.jsonl
data/
outputs/milestone_7/safety_error_analysis/detection_error_index.csv
outputs/milestone_7/failure_cases/images/*.png
```

Recommended tracked alternatives:

```text
configs/
scripts/
docs/
outputs/*/tables/
outputs/*/figures/
outputs/*/manifests/
outputs/*/final_audit/
README.md
```

---

## 15. How to Reproduce the Main Experimental Flow

At a high level, the reproducible route is:

1. Place KITTI raw data under `data\kitti\raw\training`.
2. Place Waymo validation metadata/statistics under `data\waymo\raw\validation`.
3. Run Milestone 2 KITTI and Waymo selection scripts.
4. Run Milestone 3 full preprocessing and annotation generation.
5. Validate the Milestone 3 handoff.
6. Prepare the Milestone 4 Kaggle package.
7. Train/resume detector families on Kaggle using the resume-aware notebook/training runner.
8. Register locked final checkpoints.
9. Run Milestone 5 KITTI validation.
10. Run Milestone 6 Waymo external validation.
11. Run Milestone 7 safety/failure/deployment analysis.
12. Update Milestone 8 paper assets.

---

## 16. Citation / Manuscript

working title:

```text
Beyond In-Domain Accuracy: Eight-Instance Cross-Dataset Generalization, Safety-Oriented Failure Analysis, and Deployment Trade-offs for Autonomous-Driving Object Detection
```

---

## 17. Practical Interpretation

This repository should not be read as claiming that any evaluated detector is deployment-ready. The main contribution is a reproducible, safety-aware, target-free evaluation framework showing that in-domain accuracy alone is not enough to support autonomous-driving perception robustness claims.
