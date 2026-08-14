from pathlib import Path
import json
from datetime import datetime

PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

HANDOFF_JSON = PROJECT / "outputs" / "milestone_6" / "handoff_validation" / "waymo_handoff_summary.json"
CHECKPOINT_REGISTRY = PROJECT / "outputs" / "milestone_4" / "locked_final_checkpoints" / "final_checkpoint_registry.json"
KITTI_BASELINE_CSV = PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "tables" / "comparison_summary_full.csv"

DATASET_CONFIG_DIR = PROJECT / "configs" / "datasets" / "milestone_6"
EVAL_CONFIG_DIR = PROJECT / "configs" / "evaluation" / "milestone_6"

DATASET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main():
    print("=" * 100)
    print("STEP 2/10 - Create Milestone 6 configuration and policy files")
    print("=" * 100)

    errors = []

    if not HANDOFF_JSON.exists():
        errors.append(f"Missing Waymo handoff summary: {HANDOFF_JSON}")

    if not CHECKPOINT_REGISTRY.exists():
        errors.append(f"Missing final checkpoint registry: {CHECKPOINT_REGISTRY}")

    if not KITTI_BASELINE_CSV.exists():
        errors.append(f"Missing KITTI baseline table: {KITTI_BASELINE_CSV}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        print("STEP 2/10 FAILED ❌")
        raise SystemExit(1)

    handoff = json.loads(HANDOFF_JSON.read_text(encoding="utf-8"))

    waymo_images = handoff["num_images"]
    waymo_labels = handoff["num_labels"]
    waymo_pairs = handoff["num_matched_pairs"]
    waymo_annotations = handoff["total_annotations"]
    class_counts = handoff["class_counts"]

    created_at = datetime.now().isoformat(timespec="seconds")

    dataset_yaml = f"""
# Milestone 6 dataset configuration
# Purpose: Waymo external validation subset for cross-dataset generalization analysis.
# Created automatically by scripts/milestone_6/01_create_milestone_6_configs.py

milestone: 6
dataset_name: waymo_external_subset
task: external_validation
created_at: "{created_at}"

source:
  dataset_family: Waymo
  role: external_validation_only
  training_allowed: false
  retraining_allowed: false
  fine_tuning_allowed: false
  purpose: >
    Evaluate KITTI-trained object detectors directly on an external Waymo subset
    to measure cross-dataset generalization and domain-shift degradation.

paths:
  image_dir: "data/processed/milestone_3/images/waymo/external"
  label_dir: "data/processed/milestone_3/labels/waymo/external"

handoff_counts:
  images: {waymo_images}
  labels: {waymo_labels}
  matched_image_label_pairs: {waymo_pairs}
  total_annotations: {waymo_annotations}

label_format:
  format: yolo_normalized
  fields:
    - class_id
    - x_center
    - y_center
    - width
    - height
  coordinate_range: "[0, 1]"
  bbox_interpretation: center_x_center_y_width_height_normalized_by_image_size

class_mapping:
  0: Vehicle
  1: Pedestrian
  2: Cyclist

class_distribution:
  Vehicle: {class_counts.get("0", 0)}
  Pedestrian: {class_counts.get("1", 0)}
  Cyclist: {class_counts.get("2", 0)}

notes:
  - This subset is not used for training.
  - This subset is not used for hyperparameter selection.
  - Empty-label files are allowed only if the corresponding image contains no target objects.
  - The class mapping must remain identical to the Milestone 5 KITTI evaluation.
"""

    external_policy_yaml = f"""
# Milestone 6 external validation policy
# Purpose: ensure fair cross-dataset comparison without retraining.

milestone: 6
policy_name: waymo_external_validation_policy
created_at: "{created_at}"

core_rule:
  description: >
    All models evaluated in Milestone 6 must be the locked KITTI-trained checkpoints
    produced in Milestone 4 and evaluated in Milestone 5.
  retraining: forbidden
  fine_tuning: forbidden
  architecture_changes: forbidden
  threshold_tuning_on_waymo: forbidden
  data_leakage_from_waymo_to_training: forbidden

models:
  - yolo
  - rtdetr
  - retinanet
  - faster_rcnn

checkpoint_source:
  registry: "outputs/milestone_4/locked_final_checkpoints/final_checkpoint_registry.json"
  checkpoint_selection: best_checkpoint
  selection_basis: Milestone 4/5 KITTI validation performance and locked checkpoint registry

evaluation_dataset:
  name: waymo_external_subset
  config: "configs/datasets/milestone_6/waymo_external_subset.yaml"

baseline_dataset:
  name: kitti_validation
  baseline_results: "outputs/milestone_5/final_kitti_validation/tables/comparison_summary_full.csv"

input_policy:
  image_size: 640
  preprocessing: same_as_milestone_5_evaluator
  class_mapping:
    0: Vehicle
    1: Pedestrian
    2: Cyclist

metrics:
  primary:
    - mAP50_95
  secondary:
    - mAP50
    - per_class_AP50
    - per_class_AP50_95
    - mean_inference_ms

fairness_rules:
  - Use the same Waymo image-label subset for all detectors.
  - Use the same IoU thresholds as the Milestone 5 KITTI evaluator.
  - Use the same class mapping as Milestone 5.
  - Do not retrain, fine-tune, calibrate, or tune thresholds on Waymo.
  - Use one locked checkpoint per detector.
  - Report both aggregate and class-wise results.
  - Compare against Milestone 5 KITTI validation metrics using the same metric names.
"""

    generalization_yaml = f"""
# Milestone 6 generalization metric configuration
# Purpose: compare KITTI in-domain results against Waymo external validation results.

milestone: 6
config_name: generalization_metrics
created_at: "{created_at}"

baseline:
  dataset: KITTI
  source_table: "outputs/milestone_5/final_kitti_validation/tables/comparison_summary_full.csv"

external:
  dataset: Waymo
  expected_output_table: "outputs/milestone_6/waymo_external_validation/tables/waymo_external_summary.csv"

primary_generalization_formula:
  name: Generalization Ratio
  formula: "mAP_Waymo / mAP_KITTI"
  interpretation:
    near_1: stronger cross-dataset stability
    below_1: degradation under domain shift
    above_1: possible improvement or dataset-composition effect requiring discussion

computed_fields:
  aggregate:
    - kitti_mAP50
    - waymo_mAP50
    - mAP50_drop
    - mAP50_generalization_ratio
    - kitti_mAP50_95
    - waymo_mAP50_95
    - mAP50_95_drop
    - mAP50_95_generalization_ratio
  class_wise:
    - Vehicle_AP50_95_drop
    - Pedestrian_AP50_95_drop
    - Cyclist_AP50_95_drop
    - largest_degraded_class

paper_outputs:
  - KITTI_vs_Waymo_comparison_table
  - Generalization_Ratio_table
  - Class_wise_degradation_chart
  - Domain_shift_analysis_subsection
  - External_validation_discussion_draft
"""

    readme_md = f"""
# Milestone 6 Configuration Summary

Created at: `{created_at}`

## Purpose

Milestone 6 evaluates the locked KITTI-trained object detectors on the Waymo external subset without retraining.

## Waymo Handoff

- Images: `{waymo_images}`
- Labels: `{waymo_labels}`
- Matched image-label pairs: `{waymo_pairs}`
- Total annotations: `{waymo_annotations}`

## Class Mapping

| Class ID | Class Name | Waymo Annotation Count |
|---:|---|---:|
| 0 | Vehicle | {class_counts.get("0", 0)} |
| 1 | Pedestrian | {class_counts.get("1", 0)} |
| 2 | Cyclist | {class_counts.get("2", 0)} |

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
"""

    dataset_path = DATASET_CONFIG_DIR / "waymo_external_subset.yaml"
    policy_path = EVAL_CONFIG_DIR / "external_validation_policy.yaml"
    generalization_path = EVAL_CONFIG_DIR / "generalization_metrics.yaml"
    readme_path = EVAL_CONFIG_DIR / "README.md"

    write_text(dataset_path, dataset_yaml)
    write_text(policy_path, external_policy_yaml)
    write_text(generalization_path, generalization_yaml)
    write_text(readme_path, readme_md)

    print("Created:", dataset_path)
    print("Created:", policy_path)
    print("Created:", generalization_path)
    print("Created:", readme_path)

    print()
    print("STEP 2/10 COMPLETE ✅")
    print("Milestone 6 dataset, external-validation policy, and generalization metric configs are ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()