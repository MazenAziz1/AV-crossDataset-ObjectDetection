"""Create and lock the Milestone 7 analysis policy configs.

Object-size bins are reused verbatim from the frozen Milestone 5 evaluation
policy (configs/evaluation/milestone_5/evaluation_policy.yaml scale_ranges),
not redefined.
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common


FAILURE_CASE_POLICY = {
    "metadata": {
        "milestone": 7,
        "status": "frozen",
        "date_frozen": "2026-08-15",
    },
    "matching": {
        "tp_iou_threshold": 0.50,
        "confidence_threshold_operating_point": 0.25,
        "note": "Confidence threshold matches the frozen Milestone 5/6 operating point.",
    },
    "failure_types": {
        "missed_detection": "ground-truth object with no matched prediction (false negative)",
        "false_positive": "prediction with no ground-truth overlap (IoU < 0.10 against any GT)",
        "localization_error": "correct class but 0.10 <= IoU < 0.50 against an unmatched same-class GT",
        "class_confusion": "prediction overlaps a ground-truth object of a different class (IoU >= 0.10)",
        "over_detection": "extra prediction overlapping an already-matched GT of the same class (IoU >= 0.50)",
    },
    "thresholds": {
        "localization_ioi_low": 0.10,
        "localization_iou_high": 0.50,
        "confusion_iou_low": 0.10,
        "over_detection_iou": 0.50,
    },
    "datasets": ["kitti", "waymo"],
    "detectors": common.DETECTORS,
    "classes": {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"},
}

SAFETY_ERROR_POLICY = {
    "metadata": {
        "milestone": 7,
        "status": "frozen",
        "date_frozen": "2026-08-15",
    },
    "safety_priority_classes": {2: "Pedestrian", 3: "Cyclist"},
    "fn_rate_definition": "false_negatives / (true_positives + false_negatives) per (detector, dataset, class)",
    "top_safety_critical_images": 10,
    "top_image_criterion": "highest count of pedestrian + cyclist missed detections per image",
    "datasets": ["kitti", "waymo"],
    "detectors": common.DETECTORS,
}

README = """# Milestone 7 Analysis Configs

Frozen analysis policy for the robustness/failure-case/safety analysis.

- `failure_case_policy.yaml` - TP/FP/FN matching rules, failure-type definitions, thresholds.
- `safety_error_policy.yaml` - safety-priority classes (Pedestrian, Cyclist) and FN-rate rules.
- `object_size_bins.yaml` - object-size bins, reused verbatim from the frozen Milestone 5
  evaluation policy (`configs/evaluation/milestone_5/evaluation_policy.yaml` scale_ranges).

These policies are locked before any result is generated.
"""


def main():
    print("=" * 79)
    print("Milestone 7 - Step 1: Create analysis policy configs")
    print("=" * 79)

    out_dir = common.M7_CFG
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write failure case policy
    (out_dir / "failure_case_policy.yaml").write_text(
        yaml.safe_dump(FAILURE_CASE_POLICY, sort_keys=False), encoding="utf-8")

    # Write safety error policy
    (out_dir / "safety_error_policy.yaml").write_text(
        yaml.safe_dump(SAFETY_ERROR_POLICY, sort_keys=False), encoding="utf-8")

    # Reuse frozen M5 object-size bins (read from the frozen source, not redefined)
    m5_policy_path = common.PROJECT_ROOT / "configs" / "evaluation" / "milestone_5" / "evaluation_policy.yaml"
    with open(m5_policy_path, encoding="utf-8") as f:
        m5_policy = yaml.safe_load(f)
    scale_ranges = m5_policy["metrics"]["scale_ranges"]

    bins_config = {
        "metadata": {
            "milestone": 7,
            "status": "frozen",
            "source": "reused verbatim from configs/evaluation/milestone_5/evaluation_policy.yaml scale_ranges",
            "source_date_frozen": "2026-06-24",
        },
        "unit": "absolute_pixels_squared",
        "bins": scale_ranges,
    }
    (out_dir / "object_size_bins.yaml").write_text(
        yaml.safe_dump(bins_config, sort_keys=False), encoding="utf-8")

    # Write README
    (out_dir / "README.md").write_text(README, encoding="utf-8")

    # Also mirror the locked policies into data_audit for traceability
    audit_dir = common.M7_OUT / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "m7_config_snapshot.json").write_text(json.dumps({
        "failure_case_policy": FAILURE_CASE_POLICY,
        "safety_error_policy": SAFETY_ERROR_POLICY,
        "object_size_bins": bins_config,
    }, indent=2), encoding="utf-8")

    print(f"Wrote failure_case_policy.yaml")
    print(f"Wrote safety_error_policy.yaml")
    print(f"Wrote object_size_bins.yaml (reused M5 scale_ranges: {scale_ranges})")
    print(f"Wrote README.md")


if __name__ == "__main__":
    main()
