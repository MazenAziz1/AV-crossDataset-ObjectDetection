from pathlib import Path
from datetime import datetime
import json


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

CONFIG_DIR = PROJECT / "configs" / "analysis" / "milestone_7"
OUTPUT_DIR = PROJECT / "outputs" / "milestone_7" / "data_audit"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "failure_case_policy": CONFIG_DIR / "failure_case_policy.yaml",
    "safety_error_policy": CONFIG_DIR / "safety_error_policy.yaml",
    "object_size_bins": CONFIG_DIR / "object_size_bins.yaml",
    "readme": CONFIG_DIR / "README.md",
}

MANIFEST_JSON = OUTPUT_DIR / "milestone_7_config_manifest.json"
MANIFEST_MD = OUTPUT_DIR / "MILESTONE_7_CONFIG_MANIFEST.md"


FAILURE_CASE_POLICY = """# Milestone 7 Failure-Case Policy
# Robustness, Failure-Case & Safety-Oriented Analysis

milestone: 7
name: "Robustness, Failure-Case & Safety-Oriented Analysis"

purpose:
  - "Analyze difficult perception cases."
  - "Prioritize safety-relevant failures."
  - "Create failure-case galleries."
  - "Connect numerical errors to intelligent-vehicle deployment recommendations."

datasets:
  kitti:
    split: "validation"
    image_dir: "data/processed/milestone_3/images/kitti/val"
    label_dir: "data/processed/milestone_3/labels/kitti/val"
    prediction_dir: "outputs/milestone_5/final_kitti_validation/predictions"
  waymo:
    split: "external"
    image_dir: "data/processed/milestone_3/images/waymo/external"
    label_dir: "data/processed/milestone_3/labels/waymo/external"
    prediction_dir: "outputs/milestone_6/waymo_external_validation/predictions"

detectors:
  - yolo
  - rtdetr
  - retinanet
  - faster_rcnn

classes:
  0: "Vehicle"
  1: "Pedestrian"
  2: "Cyclist"

matching_policy:
  primary_iou_threshold: 0.50
  localization_error_iou_min: 0.10
  localization_error_iou_max: 0.50
  class_confusion_iou_min: 0.30
  duplicate_detection_iou_min: 0.50
  prediction_confidence_min_for_matching: 0.001
  prediction_confidence_min_for_gallery: 0.25

failure_types:
  true_positive:
    description: "Prediction has correct class and IoU >= primary_iou_threshold with an unmatched ground-truth object."
  false_negative:
    description: "Ground-truth object has no matched prediction of the correct class."
  false_positive:
    description: "Prediction does not match any ground-truth object of the correct class."
  localization_error:
    description: "Prediction has the correct class and overlaps the object, but IoU is below the primary threshold."
  class_confusion:
    description: "Prediction overlaps a ground-truth object but predicts the wrong class."
  duplicate_detection:
    description: "Multiple predictions match or overlap the same ground-truth object."

gallery_policy:
  priority_classes:
    - "Pedestrian"
    - "Cyclist"
  secondary_class:
    - "Vehicle"
  priority_failure_types:
    - false_negative
    - class_confusion
    - localization_error
    - false_positive
  examples_per_detector_dataset_failure: 5
  max_total_gallery_images: 120
  prefer_examples:
    - "high confidence false positives"
    - "small missed pedestrians"
    - "small missed cyclists"
    - "crowded scenes when visible from labels/predictions"
    - "largest localization errors"
    - "class confusion involving pedestrians or cyclists"

annotation_colors:
  ground_truth: "green"
  true_positive: "blue"
  false_negative: "red"
  false_positive: "orange"
  localization_error: "yellow"
  class_confusion: "purple"

output_targets:
  error_index_csv: "outputs/milestone_7/safety_error_analysis/detection_error_index.csv"
  error_index_json: "outputs/milestone_7/safety_error_analysis/detection_error_index.json"
  gallery_dir: "outputs/milestone_7/failure_cases/images"
  panel_dir: "outputs/milestone_7/failure_cases/panels"
"""


SAFETY_ERROR_POLICY = """# Milestone 7 Safety Error Policy

milestone: 7
name: "Safety-Oriented Error Analysis"

safety_focus:
  primary_classes:
    - "Pedestrian"
    - "Cyclist"
  rationale:
    - "False negatives for vulnerable road users are more safety-critical than vehicle-only AP."
    - "A missed pedestrian or cyclist can be more important than a small change in average mAP."

core_error_metrics:
  - false_negative_count
  - false_negative_rate
  - false_positive_count
  - localization_error_count
  - class_confusion_count
  - recall_by_class
  - recall_by_object_size
  - safety_critical_miss_rate

matching:
  iou_threshold: 0.50
  prediction_confidence_min: 0.001

risk_levels:
  critical:
    condition: "false_negative and class in [Pedestrian, Cyclist]"
    description: "Missed vulnerable road user."
  high:
    condition: "class_confusion involving Pedestrian or Cyclist"
    description: "Vulnerable road user confused with another class."
  medium:
    condition: "localization_error involving Pedestrian or Cyclist"
    description: "Detected but poorly localized vulnerable road user."
  low:
    condition: "vehicle false positive or duplicate detection"
    description: "Less safety-critical than missed vulnerable road users, but relevant for reliability."

analysis_dimensions:
  - dataset
  - detector
  - class_name
  - object_size_bin
  - failure_type
  - risk_level

dataset_comparison:
  baseline_dataset: "kitti"
  external_dataset: "waymo"
  compare:
    - "false_negative_rate"
    - "safety_critical_miss_rate"
    - "recall_by_class"
    - "recall_by_object_size"

output_targets:
  safety_false_negative_summary: "outputs/milestone_7/safety_error_analysis/safety_false_negative_summary.csv"
  top_safety_critical_images: "outputs/milestone_7/safety_error_analysis/top_safety_critical_images.csv"
  failure_type_summary: "outputs/milestone_7/safety_error_analysis/failure_type_summary.csv"
  class_confusion_summary: "outputs/milestone_7/safety_error_analysis/class_confusion_summary.csv"
"""


OBJECT_SIZE_BINS = """# Milestone 7 Object Size Bins
# Quantile-based object-size bins computed from target-box normalized area.

milestone: 7
name: "Object Size Analysis"

method: "target_box_normalized_area_quantiles"
normalized_area_definition: "bbox_area / image_area"
target_boxes_used: 39086

bbox_size_definition:
  formula: "normalized_area = bbox_area_pixels / image_area_pixels"
  bbox_area_pixels: "bbox_width_pixels * bbox_height_pixels"
  image_area_pixels: "image_width_pixels * image_height_pixels"
  source: "ground-truth box after converting YOLO normalized labels to pixel coordinates"

size_bins:
  small:
    definition: "normalized_area <= lower_33_percent_quantile"
    max_normalized_area: 0.004157071390230814
  medium:
    definition: "lower_33_percent_quantile < normalized_area <= upper_67_percent_quantile"
    min_normalized_area_exclusive: 0.004157071390230814
    max_normalized_area: 0.015589650670960799
  large:
    definition: "normalized_area > upper_67_percent_quantile"
    min_normalized_area_exclusive: 0.015589650670960799

notes:
  - "These bins follow the predefined project method based on target-box normalized-area quantiles."
  - "The bins are computed from 39086 target boxes."
  - "Object size is based on bounding-box area divided by image area, not IoU and not the larger box side."
  - "Small object analysis should be reported separately for Vehicle, Pedestrian, and Cyclist."
  - "IoU matching thresholds are separate and remain defined in the failure-case and safety-error policies."

outputs:
  object_size_summary: "outputs/milestone_7/object_size_analysis/object_size_summary.csv"
  small_object_failure_summary: "outputs/milestone_7/object_size_analysis/small_object_failure_summary.csv"
  object_size_dataset_comparison: "outputs/milestone_7/object_size_analysis/object_size_dataset_comparison.csv"
"""

README = """# Milestone 7 Analysis Configs

Milestone 7 focuses on robustness, failure-case analysis, and safety-oriented interpretation.

## Config Files

- `failure_case_policy.yaml`  
  Defines failure types, IoU thresholds, gallery selection rules, and visualization colors.

- `safety_error_policy.yaml`  
  Defines safety-critical error priorities, especially false negatives for pedestrians and cyclists.

- `object_size_bins.yaml`  
  Defines small, medium, and large object bins using ground-truth bounding-box pixel area.

## Main Principle

Milestone 7 should not be treated as another training or benchmark-only milestone.  
It translates KITTI and Waymo results into safety-relevant insight for intelligent-vehicle perception.

## Expected Downstream Scripts

- `01_build_detection_error_index.py`
- `02_object_size_analysis.py`
- `03_safety_false_negative_analysis.py`
- `04_failure_type_analysis.py`
- `05_create_failure_case_gallery.py`
- `06_deployment_tradeoff_analysis.py`
- `07_create_milestone_7_figures.py`
- `08_final_audit.py`
"""


def main():
    print("=" * 100)
    print("STEP 2/10 - Create Milestone 7 analysis config files")
    print("=" * 100)

    write_map = {
        "failure_case_policy": FAILURE_CASE_POLICY,
        "safety_error_policy": SAFETY_ERROR_POLICY,
        "object_size_bins": OBJECT_SIZE_BINS,
        "readme": README,
    }

    created = []

    for key, content in write_map.items():
        path = FILES[key]
        path.write_text(content, encoding="utf-8")
        created.append(str(path.relative_to(PROJECT)))
        print("Created:", path)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "milestone": 7,
        "purpose": "Create fixed analysis rules for robustness, failure-case, and safety-oriented evaluation.",
        "created_files": created,
        "policy_summary": {
            "primary_iou_threshold": 0.50,
            "localization_error_iou_range": [0.10, 0.50],
            "class_confusion_iou_min": 0.30,
            "priority_classes": ["Pedestrian", "Cyclist"],
            "size_bins": {
    		"method": "target_box_normalized_area_quantiles",
    		"normalized_area_definition": "bbox_area / image_area",
    		"small": "normalized_area <= 0.004157071390230814",
    		"medium": "0.004157071390230814 < normalized_area <= 0.015589650670960799",
    		"large": "normalized_area > 0.015589650670960799",
    		"target_boxes_used": 39086,
	    },
        },
    }

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 7 Config Manifest")
    md.append("")
    md.append(f"Created at: `{manifest['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append("## Created Files")
    md.append("")
    for file in created:
        md.append(f"- `{file}`")
    md.append("")
    md.append("## Key Analysis Rules")
    md.append("")
    md.append("- Primary IoU threshold: `0.50`")
    md.append("- Localization error IoU range: `0.10 <= IoU < 0.50`")
    md.append("- Class confusion IoU minimum: `0.30`")
    md.append("- Priority safety classes: `Pedestrian`, `Cyclist`")
    md.append("- Method: `target_box_normalized_area_quantiles`")
    md.append("- Normalized area: `bbox_area / image_area`")
    md.append("- Small objects: `normalized_area <= 0.004157071390230814`")
    md.append("- Medium objects: `0.004157071390230814 < normalized_area <= 0.015589650670960799`")
    md.append("- Large objects: `normalized_area > 0.015589650670960799`")
    md.append("- Target boxes used to define bins: `39086`")
    md.append("")

    MANIFEST_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("Created:", MANIFEST_JSON)
    print("Created:", MANIFEST_MD)

    print()
    print("STEP 2/10 COMPLETE ✅")
    print("Milestone 7 analysis policies and configs are ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()