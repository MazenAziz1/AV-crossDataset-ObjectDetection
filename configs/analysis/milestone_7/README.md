# Milestone 7 Analysis Configs

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
