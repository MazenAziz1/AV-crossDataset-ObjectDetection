from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any

import yaml


# ============================================================
# PATHS
# ============================================================

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

CONFIG_ROOT = Path(
    "configs/datasets/milestone_3"
)

REPORT_DIR = (
    PROCESSED_ROOT
    / "reports"
)

MANIFEST_DIR = (
    PROCESSED_ROOT
    / "manifests"
)

ANNOTATION_ROOT = (
    PROCESSED_ROOT
    / "annotations"
)

IMAGE_ROOT = (
    PROCESSED_ROOT
    / "images"
)

FRAMEWORK_LABEL_ROOT = (
    PROCESSED_ROOT
    / "labels"
)

FINAL_REPORT_FILE = (
    REPORT_DIR
    / "final_dataset_audit.json"
)

FINAL_ISSUES_FILE = (
    REPORT_DIR
    / "final_dataset_audit_issues.csv"
)


# ============================================================
# EXPECTED TOTALS
# ============================================================

EXPECTED_PARTITIONS = {
    "kitti_train": {
        "image_dir": (
            IMAGE_ROOT
            / "kitti/train"
        ),
        "canonical_yolo_dir": (
            ANNOTATION_ROOT
            / "yolo/kitti/train"
        ),
        "framework_yolo_dir": (
            FRAMEWORK_LABEL_ROOT
            / "kitti/train"
        ),
        "coco_file": (
            ANNOTATION_ROOT
            / "coco/kitti_train.json"
        ),
        "images": 5985,
        "annotations": 31294,
        "negative_images": 0,
        "empty_labels": 0,
        "class_counts": {
            "Vehicle": 26278,
            "Pedestrian": 3729,
            "Cyclist": 1287,
        },
    },

    "kitti_val": {
        "image_dir": (
            IMAGE_ROOT
            / "kitti/val"
        ),
        "canonical_yolo_dir": (
            ANNOTATION_ROOT
            / "yolo/kitti/val"
        ),
        "framework_yolo_dir": (
            FRAMEWORK_LABEL_ROOT
            / "kitti/val"
        ),
        "coco_file": (
            ANNOTATION_ROOT
            / "coco/kitti_val.json"
        ),
        "images": 1496,
        "annotations": 7792,
        "negative_images": 0,
        "empty_labels": 0,
        "class_counts": {
            "Vehicle": 6472,
            "Pedestrian": 980,
            "Cyclist": 340,
        },
    },

    "waymo_external": {
        "image_dir": (
            IMAGE_ROOT
            / "waymo/external"
        ),
        "canonical_yolo_dir": (
            ANNOTATION_ROOT
            / "yolo/waymo/external"
        ),
        "framework_yolo_dir": (
            FRAMEWORK_LABEL_ROOT
            / "waymo/external"
        ),
        "coco_file": (
            ANNOTATION_ROOT
            / "coco/waymo_external.json"
        ),
        "images": 996,
        "annotations": 24819,
        "negative_images": 12,
        "empty_labels": 12,
        "class_counts": {
            "Vehicle": 16928,
            "Pedestrian": 7127,
            "Cyclist": 764,
        },
    },
}


EXPECTED_COMBINED = {
    "images": 8477,
    "annotations": 63905,
    "negative_images": 12,
    "canonical_yolo_labels": 8477,
    "framework_yolo_labels": 8477,
    "empty_labels": 12,
    "class_counts": {
        "Vehicle": 49678,
        "Pedestrian": 11836,
        "Cyclist": 2391,
    },
    "evaluation_ignore_regions": 11295,
    "excluded_non_target_regions": 1484,
}


EXPECTED_COCO_CATEGORIES = {
    1: "Vehicle",
    2: "Pedestrian",
    3: "Cyclist",
}


# ============================================================
# PREVIOUS REPORTS
# ============================================================

REPORT_SPECIFICATIONS = {
    "source_input_validation": {
        "path": (
            REPORT_DIR
            / "source_input_validation.json"
        ),
        "pass_key": (
            "source_input_validation_passed"
        ),
    },

    "source_manifest": {
        "path": (
            REPORT_DIR
            / "source_manifest_summary.json"
        ),
        "pass_key": (
            "source_manifest_passed"
        ),
    },

    "preprocessing_dry_run": {
        "path": (
            REPORT_DIR
            / "preprocessing_dry_run.json"
        ),
        "pass_key": (
            "preprocessing_dry_run_passed"
        ),
    },

    "image_preprocessing": {
        "path": (
            REPORT_DIR
            / "image_preprocessing_report.json"
        ),
        "pass_key": (
            "image_preprocessing_passed"
        ),
    },

    "coco_creation": {
        "path": (
            REPORT_DIR
            / "coco_creation_report.json"
        ),
        "pass_key": (
            "coco_creation_passed"
        ),
    },

    "region_policy": {
        "path": (
            REPORT_DIR
            / "region_policy_report.json"
        ),
        "pass_key": (
            "region_policy_passed"
        ),
    },

    "yolo_conversion": {
        "path": (
            REPORT_DIR
            / "yolo_conversion_report.json"
        ),
        "pass_key": (
            "yolo_conversion_passed"
        ),
    },

    "dataset_configuration": {
        "path": (
            REPORT_DIR
            / "config_validation.json"
        ),
        "pass_key": (
            "config_validation_passed"
        ),
    },

    "coco_validation": {
        "path": (
            REPORT_DIR
            / "coco_validation_report.json"
        ),
        "pass_key": (
            "coco_validation_passed"
        ),
    },

    "yolo_validation": {
        "path": (
            REPORT_DIR
            / "yolo_validation_report.json"
        ),
        "pass_key": (
            "yolo_validation_passed"
        ),
    },

    "coco_yolo_equivalence": {
        "path": (
            REPORT_DIR
            / "coco_yolo_equivalence_report.json"
        ),
        "pass_key": (
            "coco_yolo_equivalence_passed"
        ),
    },

    "visual_annotation_checks": {
        "path": (
            REPORT_DIR
            / "visual_annotation_checks_report.json"
        ),
        "pass_key": (
            "visual_annotation_checks_passed"
        ),
    },

    "augmentation_policy": {
        "path": (
            REPORT_DIR
            / "augmentation_policy_report.json"
        ),
        "pass_key": (
            "augmentation_policy_passed"
        ),
    },

    "dataloader_validation": {
        "path": (
            REPORT_DIR
            / "dataloader_validation_report.json"
        ),
        "pass_key": (
            "dataloader_validation_passed"
        ),
    },
}


# ============================================================
# EXPECTED MANIFESTS
# ============================================================

EXPECTED_MANIFEST_ROWS = {
    "source_manifest.csv": 8477,
    "transform_manifest.csv": 8477,
    "yolo_label_manifest.csv": 8477,
    "framework_label_adapter_manifest.csv": 8477,
    "coco_yolo_equivalence_manifest.csv": 8477,
    "region_policy_manifest.csv": 8477,
    "visual_annotation_checks_manifest.csv": 8,
    "augmentation_policy_manifest.csv": 8,
    "dataloader_smoke_test_manifest.csv": 24,
}


# ============================================================
# REGION SIDECARS
# ============================================================

IGNORE_SIDECARS = [
    (
        ANNOTATION_ROOT
        / "ignore_regions/"
        "kitti_train_ignore.json"
    ),
    (
        ANNOTATION_ROOT
        / "ignore_regions/"
        "kitti_val_ignore.json"
    ),
    (
        ANNOTATION_ROOT
        / "ignore_regions/"
        "waymo_external_ignore.json"
    ),
]


EXCLUDED_SIDECARS = [
    (
        ANNOTATION_ROOT
        / "excluded_objects/"
        "kitti_train_excluded.json"
    ),
    (
        ANNOTATION_ROOT
        / "excluded_objects/"
        "kitti_val_excluded.json"
    ),
    (
        ANNOTATION_ROOT
        / "excluded_objects/"
        "waymo_external_excluded.json"
    ),
]


# ============================================================
# REQUIRED VISUAL ARTIFACTS
# ============================================================

REQUIRED_VISUAL_ARTIFACTS = [
    (
        PROCESSED_ROOT
        / "visual_checks/"
        "preprocessing_dry_run"
    ),
    (
        PROCESSED_ROOT
        / "visual_checks/"
        "coco_yolo_comparison/"
        "annotation_comparison_contact_sheet.png"
    ),
    (
        PROCESSED_ROOT
        / "visual_checks/"
        "augmentation_policy/"
        "augmentation_policy_contact_sheet.png"
    ),
]


# ============================================================
# HELPERS
# ============================================================

def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found:\n"
            f"{path.resolve()}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"JSON root must be an object:\n"
            f"{path.resolve()}"
        )

    return data


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"YAML file not found:\n"
            f"{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"YAML root must be a mapping:\n"
            f"{path.resolve()}"
        )

    return data


def add_issue(
    issues: list[dict[str, str]],
    section: str,
    category: str,
    identifier: str,
    details: str,
) -> None:
    issues.append(
        {
            "section": section,
            "category": category,
            "identifier": identifier,
            "details": details,
        }
    )


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def count_csv_rows(
    path: Path,
) -> int:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n"
            f"{path.resolve()}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        return sum(
            1
            for _ in reader
        )


def count_files(
    directory: Path,
    pattern: str,
) -> int:
    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.glob(
            pattern
        )
        if path.is_file()
    )


def file_names(
    directory: Path,
    pattern: str,
) -> set[str]:
    if not directory.exists():
        return set()

    return {
        path.name
        for path in directory.glob(
            pattern
        )
        if path.is_file()
    }


def file_stems(
    directory: Path,
    pattern: str,
) -> set[str]:
    if not directory.exists():
        return set()

    return {
        path.stem
        for path in directory.glob(
            pattern
        )
        if path.is_file()
    }


def count_empty_files(
    directory: Path,
    pattern: str,
) -> int:
    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.glob(
            pattern
        )
        if (
            path.is_file()
            and path.stat().st_size == 0
        )
    )


def report_pass_status(
    report: dict[str, Any],
    preferred_key: str,
) -> tuple[
    bool | None,
    str,
]:
    """
    Resolve pass/fail from a generated report while remaining
    compatible with earlier report naming conventions.
    """

    if preferred_key in report:
        return (
            bool(
                report[
                    preferred_key
                ]
            ),
            preferred_key,
        )

    top_level_pass_keys = [
        key
        for key, value
        in report.items()
        if (
            key.endswith(
                "_passed"
            )
            and isinstance(
                value,
                bool,
            )
        )
    ]

    if top_level_pass_keys:
        return (
            all(
                bool(
                    report[key]
                )
                for key
                in top_level_pass_keys
            ),
            "|".join(
                top_level_pass_keys
            ),
        )

    for status_key in [
        "final_status",
        "status",
    ]:
        if status_key in report:
            status = str(
                report[
                    status_key
                ]
            ).strip().upper()

            if status in {
                "PASSED",
                "PASS",
                "SUCCESS",
                "COMPLETED",
            }:
                return True, status_key

            if status in {
                "FAILED",
                "FAIL",
                "ERROR",
            }:
                return False, status_key

    for count_key in [
        "issue_count",
        "issues_found",
    ]:
        if count_key in report:
            try:
                return (
                    int(
                        report[
                            count_key
                        ]
                    )
                    == 0,
                    count_key,
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

    return None, "unresolved"


def get_nested(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(
            current,
            dict,
        ):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


# ============================================================
# PREVIOUS REPORT AUDIT
# ============================================================

def audit_previous_reports(
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for report_name, specification in (
        REPORT_SPECIFICATIONS.items()
    ):
        path = Path(
            specification["path"]
        )

        preferred_key = str(
            specification["pass_key"]
        )

        if not path.exists():
            add_issue(
                issues,
                "previous_reports",
                "missing_report",
                report_name,
                str(path),
            )

            results[
                report_name
            ] = {
                "path": path.as_posix(),
                "exists": False,
                "passed": False,
                "status_source": (
                    "missing"
                ),
            }

            continue

        try:
            report = load_json(path)

            passed, status_source = (
                report_pass_status(
                    report,
                    preferred_key,
                )
            )

        except Exception as error:
            add_issue(
                issues,
                "previous_reports",
                "report_read_failed",
                report_name,
                str(error),
            )

            results[
                report_name
            ] = {
                "path": path.as_posix(),
                "exists": True,
                "passed": False,
                "status_source": (
                    "read_failed"
                ),
            }

            continue

        if passed is None:
            add_issue(
                issues,
                "previous_reports",
                "report_status_unresolved",
                report_name,
                (
                    f"No recognized pass flag "
                    f"was found in {path.name}."
                ),
            )

            passed = False

        elif not passed:
            add_issue(
                issues,
                "previous_reports",
                "previous_step_failed",
                report_name,
                (
                    f"Report status source: "
                    f"{status_source}"
                ),
            )

        results[
            report_name
        ] = {
            "path": path.as_posix(),
            "exists": True,
            "passed": bool(passed),
            "status_source": (
                status_source
            ),
        }

    return results


# ============================================================
# CONFIGURATION AUDIT
# ============================================================

def audit_configurations(
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    preprocessing_path = (
        CONFIG_ROOT
        / "preprocessing.yaml"
    )

    class_mapping_path = (
        CONFIG_ROOT
        / "class_mapping.yaml"
    )

    augmentation_path = (
        CONFIG_ROOT
        / "augmentation.yaml"
    )

    yolo_path = (
        CONFIG_ROOT
        / "kitti_waymo_yolo.yaml"
    )

    coco_paths_file = (
        CONFIG_ROOT
        / "coco_paths.yaml"
    )

    registry_path = (
        CONFIG_ROOT
        / "dataset_registry.yaml"
    )

    required_paths = [
        preprocessing_path,
        class_mapping_path,
        augmentation_path,
        yolo_path,
        coco_paths_file,
        registry_path,
    ]

    for path in required_paths:
        if not path.exists():
            add_issue(
                issues,
                "configurations",
                "missing_configuration",
                path.name,
                str(path),
            )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessing_passed = False

    if preprocessing_path.exists():
        try:
            preprocessing = load_yaml(
                preprocessing_path
            )

            preprocessing_checks = {
                "status_frozen": (
                    preprocessing.get(
                        "status"
                    )
                    == "frozen"
                ),

                "target_width": (
                    get_nested(
                        preprocessing,
                        "image_preprocessing",
                        "target_width",
                    )
                    == 640
                ),

                "target_height": (
                    get_nested(
                        preprocessing,
                        "image_preprocessing",
                        "target_height",
                    )
                    == 640
                ),

                "expected_images": (
                    get_nested(
                        preprocessing,
                        "expected_totals",
                        "images",
                    )
                    == 8477
                ),

                "expected_boxes": (
                    get_nested(
                        preprocessing,
                        "expected_totals",
                        "target_boxes",
                    )
                    == 63905
                ),
            }

            preprocessing_passed = all(
                preprocessing_checks.values()
            )

            if not preprocessing_passed:
                add_issue(
                    issues,
                    "configurations",
                    "preprocessing_policy_mismatch",
                    preprocessing_path.name,
                    str(
                        preprocessing_checks
                    ),
                )

        except Exception as error:
            preprocessing_checks = {}
            add_issue(
                issues,
                "configurations",
                "configuration_read_failed",
                preprocessing_path.name,
                str(error),
            )

    else:
        preprocessing_checks = {}

    results["preprocessing"] = {
        "path": (
            preprocessing_path
            .as_posix()
        ),
        "checks": (
            preprocessing_checks
        ),
        "passed": (
            preprocessing_passed
        ),
    }

    # --------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------

    class_mapping_passed = False

    if class_mapping_path.exists():
        try:
            class_mapping = load_yaml(
                class_mapping_path
            )

            actual_classes = sorted(
                [
                    (
                        int(
                            entry[
                                "internal_id"
                            ]
                        ),
                        int(
                            entry[
                                "yolo_id"
                            ]
                        ),
                        int(
                            entry[
                                "coco_id"
                            ]
                        ),
                        str(
                            entry["name"]
                        ),
                    )
                    for entry
                    in class_mapping.get(
                        "target_classes",
                        [],
                    )
                ],
                key=lambda item: (
                    item[0]
                ),
            )

            expected_classes = [
                (
                    0,
                    0,
                    1,
                    "Vehicle",
                ),
                (
                    1,
                    1,
                    2,
                    "Pedestrian",
                ),
                (
                    2,
                    2,
                    3,
                    "Cyclist",
                ),
            ]

            class_mapping_checks = {
                "status_frozen": (
                    class_mapping.get(
                        "status"
                    )
                    == "frozen"
                ),

                "target_classes": (
                    actual_classes
                    == expected_classes
                ),

                "kitti_ignore": (
                    get_nested(
                        class_mapping,
                        "source_mappings",
                        "kitti",
                        "evaluation_ignore_classes",
                    )
                    == ["DontCare"]
                ),

                "kitti_excluded": (
                    get_nested(
                        class_mapping,
                        "source_mappings",
                        "kitti",
                        "excluded_non_target_classes",
                    )
                    == [
                        "Tram",
                        "Misc",
                    ]
                ),

                "waymo_ignore": (
                    get_nested(
                        class_mapping,
                        "source_mappings",
                        "waymo",
                        "evaluation_ignore_classes",
                    )
                    == []
                ),

                "waymo_excluded": (
                    get_nested(
                        class_mapping,
                        "source_mappings",
                        "waymo",
                        "excluded_non_target_classes",
                    )
                    == ["Sign"]
                ),
            }

            class_mapping_passed = all(
                class_mapping_checks
                .values()
            )

            if not class_mapping_passed:
                add_issue(
                    issues,
                    "configurations",
                    "class_mapping_mismatch",
                    class_mapping_path.name,
                    str(
                        class_mapping_checks
                    ),
                )

        except Exception as error:
            class_mapping_checks = {}
            add_issue(
                issues,
                "configurations",
                "configuration_read_failed",
                class_mapping_path.name,
                str(error),
            )

    else:
        class_mapping_checks = {}

    results["class_mapping"] = {
        "path": (
            class_mapping_path
            .as_posix()
        ),
        "checks": (
            class_mapping_checks
        ),
        "passed": (
            class_mapping_passed
        ),
    }

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    augmentation_passed = False

    if augmentation_path.exists():
        try:
            augmentation = load_yaml(
                augmentation_path
            )

            augmentation_checks = {
                "status_frozen": (
                    augmentation.get(
                        "status"
                    )
                    == "frozen"
                ),

                "online_only": (
                    get_nested(
                        augmentation,
                        "execution",
                        "mode",
                    )
                    == "online_during_training"
                    and not bool(
                        get_nested(
                            augmentation,
                            "execution",
                            "generate_permanent_augmented_dataset",
                            default=True,
                        )
                    )
                ),

                "enabled_partition": (
                    get_nested(
                        augmentation,
                        "partition_policy",
                        "enabled",
                    )
                    == ["kitti_train"]
                ),

                "disabled_partitions": (
                    get_nested(
                        augmentation,
                        "partition_policy",
                        "disabled",
                    )
                    == [
                        "kitti_val",
                        "waymo_external",
                    ]
                ),
            }

            augmentation_passed = all(
                augmentation_checks.values()
            )

            if not augmentation_passed:
                add_issue(
                    issues,
                    "configurations",
                    "augmentation_policy_mismatch",
                    augmentation_path.name,
                    str(
                        augmentation_checks
                    ),
                )

        except Exception as error:
            augmentation_checks = {}
            add_issue(
                issues,
                "configurations",
                "configuration_read_failed",
                augmentation_path.name,
                str(error),
            )

    else:
        augmentation_checks = {}

    results["augmentation"] = {
        "path": (
            augmentation_path
            .as_posix()
        ),
        "checks": (
            augmentation_checks
        ),
        "passed": (
            augmentation_passed
        ),
    }

    # --------------------------------------------------------
    # YOLO dataset configuration
    # --------------------------------------------------------

    yolo_passed = False

    if yolo_path.exists():
        try:
            yolo_configuration = (
                load_yaml(
                    yolo_path
                )
            )

            yolo_names = {
                int(key): str(value)
                for key, value
                in yolo_configuration.get(
                    "names",
                    {},
                ).items()
            }

            yolo_checks = {
                "dataset_root": (
                    str(
                        yolo_configuration.get(
                            "path",
                            "",
                        )
                    )
                    == (
                        "data/processed/"
                        "milestone_3"
                    )
                ),

                "train": (
                    str(
                        yolo_configuration.get(
                            "train",
                            "",
                        )
                    )
                    == "images/kitti/train"
                ),

                "validation": (
                    str(
                        yolo_configuration.get(
                            "val",
                            "",
                        )
                    )
                    == "images/kitti/val"
                ),

                "external": (
                    str(
                        yolo_configuration.get(
                            "test",
                            "",
                        )
                    )
                    == (
                        "images/waymo/"
                        "external"
                    )
                ),

                "class_count": (
                    int(
                        yolo_configuration.get(
                            "nc",
                            -1,
                        )
                    )
                    == 3
                ),

                "class_names": (
                    yolo_names
                    == {
                        0: "Vehicle",
                        1: "Pedestrian",
                        2: "Cyclist",
                    }
                ),
            }

            yolo_passed = all(
                yolo_checks.values()
            )

            if not yolo_passed:
                add_issue(
                    issues,
                    "configurations",
                    "yolo_configuration_mismatch",
                    yolo_path.name,
                    str(yolo_checks),
                )

        except Exception as error:
            yolo_checks = {}
            add_issue(
                issues,
                "configurations",
                "configuration_read_failed",
                yolo_path.name,
                str(error),
            )

    else:
        yolo_checks = {}

    results["yolo"] = {
        "path": yolo_path.as_posix(),
        "checks": yolo_checks,
        "passed": yolo_passed,
    }

    # --------------------------------------------------------
    # COCO paths and registry existence
    # --------------------------------------------------------

    coco_paths_passed = (
        coco_paths_file.exists()
    )

    registry_passed = False

    if registry_path.exists():
        try:
            registry = load_yaml(
                registry_path
            )

            usage_policy = (
                registry.get(
                    "waymo_usage_policy",
                    {}
                )
            )

            registry_checks = {
                "training_disabled": (
                    usage_policy.get(
                        "training"
                    )
                    is False
                ),

                "fine_tuning_disabled": (
                    usage_policy.get(
                        "fine_tuning"
                    )
                    is False
                ),

                "hyperparameter_selection_disabled": (
                    usage_policy.get(
                        "hyperparameter_selection"
                    )
                    is False
                ),

                "checkpoint_selection_disabled": (
                    usage_policy.get(
                        "checkpoint_selection"
                    )
                    is False
                ),

                "external_evaluation_enabled": (
                    usage_policy.get(
                        "external_evaluation_only"
                    )
                    is True
                ),
            }

            registry_passed = all(
                registry_checks.values()
            )

            if not registry_passed:
                add_issue(
                    issues,
                    "configurations",
                    "registry_policy_mismatch",
                    registry_path.name,
                    str(registry_checks),
                )

        except Exception as error:
            registry_checks = {}
            add_issue(
                issues,
                "configurations",
                "configuration_read_failed",
                registry_path.name,
                str(error),
            )

    else:
        registry_checks = {}

    results["coco_paths"] = {
        "path": (
            coco_paths_file
            .as_posix()
        ),
        "passed": (
            coco_paths_passed
        ),
    }

    results["dataset_registry"] = {
        "path": (
            registry_path
            .as_posix()
        ),
        "checks": (
            registry_checks
        ),
        "passed": (
            registry_passed
        ),
    }

    return results


# ============================================================
# PHYSICAL FILE AND COCO AUDIT
# ============================================================

def audit_partition(
    partition_name: str,
    specification: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    image_dir = Path(
        specification["image_dir"]
    )

    canonical_yolo_dir = Path(
        specification[
            "canonical_yolo_dir"
        ]
    )

    framework_yolo_dir = Path(
        specification[
            "framework_yolo_dir"
        ]
    )

    coco_file = Path(
        specification["coco_file"]
    )

    image_count = count_files(
        image_dir,
        "*.png",
    )

    canonical_label_count = (
        count_files(
            canonical_yolo_dir,
            "*.txt",
        )
    )

    framework_label_count = (
        count_files(
            framework_yolo_dir,
            "*.txt",
        )
    )

    canonical_empty_count = (
        count_empty_files(
            canonical_yolo_dir,
            "*.txt",
        )
    )

    framework_empty_count = (
        count_empty_files(
            framework_yolo_dir,
            "*.txt",
        )
    )

    image_stem_set = file_stems(
        image_dir,
        "*.png",
    )

    canonical_stem_set = file_stems(
        canonical_yolo_dir,
        "*.txt",
    )

    framework_stem_set = file_stems(
        framework_yolo_dir,
        "*.txt",
    )

    file_set_checks = {
        "image_canonical_stems": (
            image_stem_set
            == canonical_stem_set
        ),

        "image_framework_stems": (
            image_stem_set
            == framework_stem_set
        ),

        "canonical_framework_stems": (
            canonical_stem_set
            == framework_stem_set
        ),
    }

    if not all(
        file_set_checks.values()
    ):
        add_issue(
            issues,
            "physical_dataset",
            "filename_set_mismatch",
            partition_name,
            str(file_set_checks),
        )

    physical_checks = {
        "image_count": (
            image_count
            == specification["images"]
        ),

        "canonical_label_count": (
            canonical_label_count
            == specification["images"]
        ),

        "framework_label_count": (
            framework_label_count
            == specification["images"]
        ),

        "canonical_empty_labels": (
            canonical_empty_count
            == specification[
                "empty_labels"
            ]
        ),

        "framework_empty_labels": (
            framework_empty_count
            == specification[
                "empty_labels"
            ]
        ),

        **file_set_checks,
    }

    if not all(
        physical_checks.values()
    ):
        add_issue(
            issues,
            "physical_dataset",
            "physical_partition_mismatch",
            partition_name,
            str(physical_checks),
        )

    # --------------------------------------------------------
    # COCO validation
    # --------------------------------------------------------

    coco_checks: dict[str, bool] = {}
    coco_summary: dict[str, Any] = {}

    if not coco_file.exists():
        add_issue(
            issues,
            "physical_dataset",
            "missing_coco_file",
            partition_name,
            str(coco_file),
        )

        coco_checks = {
            "exists": False,
        }

    else:
        try:
            coco_data = load_json(
                coco_file
            )

            images = coco_data.get(
                "images",
                [],
            )

            annotations = coco_data.get(
                "annotations",
                [],
            )

            categories = {
                int(
                    category["id"]
                ): str(
                    category["name"]
                )
                for category
                in coco_data.get(
                    "categories",
                    [],
                )
            }

            image_ids = {
                int(image["id"])
                for image in images
            }

            annotation_counts_by_image: (
                Counter
            ) = Counter()

            class_counts: Counter = (
                Counter()
            )

            for annotation in annotations:
                image_id = int(
                    annotation[
                        "image_id"
                    ]
                )

                category_id = int(
                    annotation[
                        "category_id"
                    ]
                )

                annotation_counts_by_image[
                    image_id
                ] += 1

                if category_id in (
                    EXPECTED_COCO_CATEGORIES
                ):
                    class_counts[
                        EXPECTED_COCO_CATEGORIES[
                            category_id
                        ]
                    ] += 1

            negative_images = sum(
                annotation_counts_by_image[
                    image_id
                ]
                == 0
                for image_id in image_ids
            )

            coco_summary = {
                "images": len(images),
                "annotations": (
                    len(annotations)
                ),
                "negative_images": (
                    negative_images
                ),
                "class_counts": {
                    class_name: int(
                        class_counts[
                            class_name
                        ]
                    )
                    for class_name in [
                        "Vehicle",
                        "Pedestrian",
                        "Cyclist",
                    ]
                },
                "categories": categories,
            }

            coco_checks = {
                "categories": (
                    categories
                    == (
                        EXPECTED_COCO_CATEGORIES
                    )
                ),

                "image_count": (
                    len(images)
                    == specification[
                        "images"
                    ]
                ),

                "annotation_count": (
                    len(annotations)
                    == specification[
                        "annotations"
                    ]
                ),

                "negative_image_count": (
                    negative_images
                    == specification[
                        "negative_images"
                    ]
                ),

                "vehicle_count": (
                    class_counts[
                        "Vehicle"
                    ]
                    == specification[
                        "class_counts"
                    ]["Vehicle"]
                ),

                "pedestrian_count": (
                    class_counts[
                        "Pedestrian"
                    ]
                    == specification[
                        "class_counts"
                    ]["Pedestrian"]
                ),

                "cyclist_count": (
                    class_counts[
                        "Cyclist"
                    ]
                    == specification[
                        "class_counts"
                    ]["Cyclist"]
                ),

                "coco_physical_filename_set": (
                    {
                        str(
                            image[
                                "file_name"
                            ]
                        )
                        for image in images
                    }
                    == file_names(
                        image_dir,
                        "*.png",
                    )
                ),
            }

            if not all(
                coco_checks.values()
            ):
                add_issue(
                    issues,
                    "physical_dataset",
                    "coco_partition_mismatch",
                    partition_name,
                    str(coco_checks),
                )

        except Exception as error:
            add_issue(
                issues,
                "physical_dataset",
                "coco_read_failed",
                partition_name,
                str(error),
            )

            coco_checks = {
                "readable": False,
            }

    partition_passed = (
        all(
            physical_checks.values()
        )
        and all(
            coco_checks.values()
        )
    )

    return {
        "image_count": image_count,
        "canonical_label_count": (
            canonical_label_count
        ),
        "framework_label_count": (
            framework_label_count
        ),
        "canonical_empty_labels": (
            canonical_empty_count
        ),
        "framework_empty_labels": (
            framework_empty_count
        ),
        "physical_checks": (
            physical_checks
        ),
        "coco_summary": coco_summary,
        "coco_checks": coco_checks,
        "passed": partition_passed,
    }


# ============================================================
# REGION POLICY AUDIT
# ============================================================

def region_count(
    path: Path,
) -> int:
    data = load_json(path)

    regions = data.get(
        "regions",
        [],
    )

    if not isinstance(
        regions,
        list,
    ):
        raise ValueError(
            f"'regions' must be a list:\n"
            f"{path.resolve()}"
        )

    return len(regions)


def audit_region_sidecars(
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    ignore_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}

    for path in IGNORE_SIDECARS:
        if not path.exists():
            add_issue(
                issues,
                "region_policy",
                "missing_ignore_sidecar",
                path.name,
                str(path),
            )

            ignore_counts[
                path.name
            ] = -1

            continue

        try:
            ignore_counts[
                path.name
            ] = region_count(path)

        except Exception as error:
            add_issue(
                issues,
                "region_policy",
                "ignore_sidecar_read_failed",
                path.name,
                str(error),
            )

            ignore_counts[
                path.name
            ] = -1

    for path in EXCLUDED_SIDECARS:
        if not path.exists():
            add_issue(
                issues,
                "region_policy",
                "missing_excluded_sidecar",
                path.name,
                str(path),
            )

            excluded_counts[
                path.name
            ] = -1

            continue

        try:
            excluded_counts[
                path.name
            ] = region_count(path)

        except Exception as error:
            add_issue(
                issues,
                "region_policy",
                "excluded_sidecar_read_failed",
                path.name,
                str(error),
            )

            excluded_counts[
                path.name
            ] = -1

    total_ignore = sum(
        value
        for value
        in ignore_counts.values()
        if value >= 0
    )

    total_excluded = sum(
        value
        for value
        in excluded_counts.values()
        if value >= 0
    )

    checks = {
        "evaluation_ignore_total": (
            total_ignore
            == EXPECTED_COMBINED[
                "evaluation_ignore_regions"
            ]
        ),

        "excluded_non_target_total": (
            total_excluded
            == EXPECTED_COMBINED[
                "excluded_non_target_regions"
            ]
        ),

        "waymo_ignore_empty": (
            ignore_counts.get(
                "waymo_external_ignore.json"
            )
            == 0
        ),

        "waymo_excluded_empty": (
            excluded_counts.get(
                "waymo_external_excluded.json"
            )
            == 0
        ),
    }

    if not all(checks.values()):
        add_issue(
            issues,
            "region_policy",
            "region_total_mismatch",
            "combined",
            str(checks),
        )

    return {
        "ignore_sidecars": (
            ignore_counts
        ),
        "excluded_sidecars": (
            excluded_counts
        ),
        "total_evaluation_ignore": (
            total_ignore
        ),
        "total_excluded_non_target": (
            total_excluded
        ),
        "checks": checks,
        "passed": all(
            checks.values()
        ),
    }


# ============================================================
# MANIFEST AUDIT
# ============================================================

def audit_manifests(
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for file_name, expected_rows in (
        EXPECTED_MANIFEST_ROWS.items()
    ):
        path = (
            MANIFEST_DIR
            / file_name
        )

        if not path.exists():
            add_issue(
                issues,
                "manifests",
                "missing_manifest",
                file_name,
                str(path),
            )

            results[file_name] = {
                "path": path.as_posix(),
                "expected_rows": (
                    expected_rows
                ),
                "actual_rows": None,
                "passed": False,
            }

            continue

        try:
            actual_rows = count_csv_rows(
                path
            )

            passed = (
                actual_rows
                == expected_rows
            )

            if not passed:
                add_issue(
                    issues,
                    "manifests",
                    "manifest_row_mismatch",
                    file_name,
                    (
                        f"Expected "
                        f"{expected_rows}, "
                        f"found {actual_rows}."
                    ),
                )

        except Exception as error:
            actual_rows = None
            passed = False

            add_issue(
                issues,
                "manifests",
                "manifest_read_failed",
                file_name,
                str(error),
            )

        results[file_name] = {
            "path": path.as_posix(),
            "expected_rows": (
                expected_rows
            ),
            "actual_rows": (
                actual_rows
            ),
            "passed": passed,
        }

    return results


# ============================================================
# ISSUE CSV AUDIT
# ============================================================

def audit_issue_csvs(
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    issue_files = sorted(
        path
        for path
        in REPORT_DIR.glob(
            "*_issues.csv"
        )
        if (
            path.is_file()
            and path
            != FINAL_ISSUES_FILE
        )
    )

    for path in issue_files:
        try:
            row_count = count_csv_rows(
                path
            )

        except Exception as error:
            add_issue(
                issues,
                "issue_files",
                "issue_csv_read_failed",
                path.name,
                str(error),
            )

            results[
                path.name
            ] = {
                "rows": None,
                "passed": False,
            }

            continue

        passed = row_count == 0

        if not passed:
            add_issue(
                issues,
                "issue_files",
                "unresolved_issue_rows",
                path.name,
                (
                    f"Found {row_count} "
                    f"unresolved issue rows."
                ),
            )

        results[path.name] = {
            "rows": row_count,
            "passed": passed,
        }

    if not issue_files:
        add_issue(
            issues,
            "issue_files",
            "no_issue_csv_files_found",
            "reports",
            (
                "No generated issue CSV files "
                "were found for auditing."
            ),
        )

    return results


# ============================================================
# VISUAL ARTIFACT AUDIT
# ============================================================

def audit_visual_artifacts(
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for path in (
        REQUIRED_VISUAL_ARTIFACTS
    ):
        exists = path.exists()

        if path.is_dir():
            nonempty = (
                exists
                and any(
                    child.is_file()
                    for child in path.iterdir()
                )
            )
        else:
            nonempty = (
                exists
                and path.is_file()
                and path.stat().st_size > 0
            )

        passed = (
            exists
            and nonempty
        )

        if not passed:
            add_issue(
                issues,
                "visual_artifacts",
                "missing_or_empty_visual_artifact",
                path.name,
                str(path),
            )

        results[
            path.as_posix()
        ] = {
            "exists": exists,
            "nonempty": nonempty,
            "passed": passed,
        }

    return results


# ============================================================
# CROSS-REPORT SEMANTIC AUDIT
# ============================================================

def audit_semantic_reports(
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # --------------------------------------------------------
    # COCO validation overlap checks
    # --------------------------------------------------------

    coco_validation_path = (
        REPORT_DIR
        / "coco_validation_report.json"
    )

    overlap_checks = {}

    if coco_validation_path.exists():
        try:
            coco_validation = load_json(
                coco_validation_path
            )

            overlaps = (
                coco_validation.get(
                    "cross_partition_overlap_counts",
                    {},
                )
            )

            overlap_checks = {
                str(key): (
                    int(value) == 0
                )
                for key, value
                in overlaps.items()
            }

            if (
                not overlap_checks
                or not all(
                    overlap_checks.values()
                )
            ):
                add_issue(
                    issues,
                    "semantic_reports",
                    "partition_overlap_detected",
                    coco_validation_path.name,
                    str(overlaps),
                )

        except Exception as error:
            add_issue(
                issues,
                "semantic_reports",
                "coco_validation_read_failed",
                coco_validation_path.name,
                str(error),
            )

    else:
        add_issue(
            issues,
            "semantic_reports",
            "missing_report",
            coco_validation_path.name,
            str(coco_validation_path),
        )

    results[
        "partition_overlap_checks"
    ] = overlap_checks

    # --------------------------------------------------------
    # COCO–YOLO equivalence
    # --------------------------------------------------------

    equivalence_path = (
        REPORT_DIR
        / "coco_yolo_equivalence_report.json"
    )

    equivalence_checks = {}

    if equivalence_path.exists():
        try:
            equivalence = load_json(
                equivalence_path
            )

            combined = (
                equivalence.get(
                    "combined",
                    {},
                )
            )

            equivalence_checks = {
                "coco_annotations": (
                    combined.get(
                        "coco_annotations"
                    )
                    == 63905
                ),

                "yolo_rows": (
                    combined.get(
                        "yolo_rows"
                    )
                    == 63905
                ),

                "matched_boxes": (
                    combined.get(
                        "matched_boxes"
                    )
                    == 63905
                ),

                "equivalent_boxes": (
                    combined.get(
                        "equivalent_boxes"
                    )
                    == 63905
                ),

                "mismatched_images": (
                    combined.get(
                        "mismatched_images"
                    )
                    == 0
                ),
            }

            if not all(
                equivalence_checks.values()
            ):
                add_issue(
                    issues,
                    "semantic_reports",
                    "equivalence_summary_mismatch",
                    equivalence_path.name,
                    str(
                        equivalence_checks
                    ),
                )

        except Exception as error:
            add_issue(
                issues,
                "semantic_reports",
                "equivalence_report_read_failed",
                equivalence_path.name,
                str(error),
            )

    else:
        add_issue(
            issues,
            "semantic_reports",
            "missing_report",
            equivalence_path.name,
            str(equivalence_path),
        )

    results[
        "coco_yolo_equivalence_checks"
    ] = equivalence_checks

    # --------------------------------------------------------
    # DataLoader partition totals
    # --------------------------------------------------------

    dataloader_path = (
        REPORT_DIR
        / "dataloader_validation_report.json"
    )

    dataloader_checks = {}

    if dataloader_path.exists():
        try:
            dataloader = load_json(
                dataloader_path
            )

            lengths = dataloader.get(
                "dataset_lengths",
                {},
            )

            dataloader_checks = {
                "kitti_train": (
                    lengths.get(
                        "kitti_train"
                    )
                    == 5985
                ),

                "kitti_val": (
                    lengths.get(
                        "kitti_val"
                    )
                    == 1496
                ),

                "waymo_external": (
                    lengths.get(
                        "waymo_external"
                    )
                    == 996
                ),

                "negative_external": (
                    get_nested(
                        dataloader,
                        "negative_external_test",
                        "passed",
                    )
                    is True
                ),

                "validation_deterministic": (
                    get_nested(
                        dataloader,
                        "deterministic_evaluation",
                        "kitti_val",
                    )
                    is True
                ),

                "external_deterministic": (
                    get_nested(
                        dataloader,
                        "deterministic_evaluation",
                        "waymo_external",
                    )
                    is True
                ),
            }

            if not all(
                dataloader_checks.values()
            ):
                add_issue(
                    issues,
                    "semantic_reports",
                    "dataloader_summary_mismatch",
                    dataloader_path.name,
                    str(
                        dataloader_checks
                    ),
                )

        except Exception as error:
            add_issue(
                issues,
                "semantic_reports",
                "dataloader_report_read_failed",
                dataloader_path.name,
                str(error),
            )

    else:
        add_issue(
            issues,
            "semantic_reports",
            "missing_report",
            dataloader_path.name,
            str(dataloader_path),
        )

    results[
        "dataloader_checks"
    ] = dataloader_checks

    return results


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[
        dict[str, str]
    ] = []

    print("=" * 76)
    print("MILESTONE 3 FINAL END-TO-END DATASET AUDIT")
    print("=" * 76)

    # --------------------------------------------------------
    # Previous reports
    # --------------------------------------------------------

    print(
        "\nAuditing previous validation reports..."
    )

    previous_reports = (
        audit_previous_reports(
            issues
        )
    )

    previous_reports_passed = sum(
        bool(
            result["passed"]
        )
        for result
        in previous_reports.values()
    )

    # --------------------------------------------------------
    # Configurations
    # --------------------------------------------------------

    print(
        "Auditing frozen configurations..."
    )

    configurations = (
        audit_configurations(
            issues
        )
    )

    configuration_passed_count = sum(
        bool(
            result["passed"]
        )
        for result
        in configurations.values()
    )

    # --------------------------------------------------------
    # Physical files and COCO
    # --------------------------------------------------------

    print(
        "Auditing physical images, labels, "
        "and COCO files..."
    )

    partition_results: dict[
        str,
        Any,
    ] = {}

    for partition_name, specification in (
        EXPECTED_PARTITIONS.items()
    ):
        partition_results[
            partition_name
        ] = audit_partition(
            partition_name,
            specification,
            issues,
        )

    # --------------------------------------------------------
    # Combined physical totals
    # --------------------------------------------------------

    combined_physical = {
        "images": sum(
            result[
                "image_count"
            ]
            for result
            in partition_results.values()
        ),

        "canonical_yolo_labels": sum(
            result[
                "canonical_label_count"
            ]
            for result
            in partition_results.values()
        ),

        "framework_yolo_labels": sum(
            result[
                "framework_label_count"
            ]
            for result
            in partition_results.values()
        ),

        "canonical_empty_labels": sum(
            result[
                "canonical_empty_labels"
            ]
            for result
            in partition_results.values()
        ),

        "framework_empty_labels": sum(
            result[
                "framework_empty_labels"
            ]
            for result
            in partition_results.values()
        ),

        "coco_annotations": sum(
            int(
                result[
                    "coco_summary"
                ].get(
                    "annotations",
                    0,
                )
            )
            for result
            in partition_results.values()
        ),

        "negative_images": sum(
            int(
                result[
                    "coco_summary"
                ].get(
                    "negative_images",
                    0,
                )
            )
            for result
            in partition_results.values()
        ),

        "class_counts": {
            class_name: sum(
                int(
                    result[
                        "coco_summary"
                    ].get(
                        "class_counts",
                        {},
                    ).get(
                        class_name,
                        0,
                    )
                )
                for result
                in partition_results.values()
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },
    }

    combined_physical_checks = {
        "images": (
            combined_physical[
                "images"
            ]
            == EXPECTED_COMBINED[
                "images"
            ]
        ),

        "canonical_yolo_labels": (
            combined_physical[
                "canonical_yolo_labels"
            ]
            == EXPECTED_COMBINED[
                "canonical_yolo_labels"
            ]
        ),

        "framework_yolo_labels": (
            combined_physical[
                "framework_yolo_labels"
            ]
            == EXPECTED_COMBINED[
                "framework_yolo_labels"
            ]
        ),

        "canonical_empty_labels": (
            combined_physical[
                "canonical_empty_labels"
            ]
            == EXPECTED_COMBINED[
                "empty_labels"
            ]
        ),

        "framework_empty_labels": (
            combined_physical[
                "framework_empty_labels"
            ]
            == EXPECTED_COMBINED[
                "empty_labels"
            ]
        ),

        "annotations": (
            combined_physical[
                "coco_annotations"
            ]
            == EXPECTED_COMBINED[
                "annotations"
            ]
        ),

        "negative_images": (
            combined_physical[
                "negative_images"
            ]
            == EXPECTED_COMBINED[
                "negative_images"
            ]
        ),

        "vehicle_count": (
            combined_physical[
                "class_counts"
            ]["Vehicle"]
            == EXPECTED_COMBINED[
                "class_counts"
            ]["Vehicle"]
        ),

        "pedestrian_count": (
            combined_physical[
                "class_counts"
            ]["Pedestrian"]
            == EXPECTED_COMBINED[
                "class_counts"
            ]["Pedestrian"]
        ),

        "cyclist_count": (
            combined_physical[
                "class_counts"
            ]["Cyclist"]
            == EXPECTED_COMBINED[
                "class_counts"
            ]["Cyclist"]
        ),
    }

    if not all(
        combined_physical_checks.values()
    ):
        add_issue(
            issues,
            "physical_dataset",
            "combined_total_mismatch",
            "combined",
            str(
                combined_physical_checks
            ),
        )

    # --------------------------------------------------------
    # Region policies
    # --------------------------------------------------------

    print(
        "Auditing ignore and excluded-object "
        "sidecars..."
    )

    region_policy = (
        audit_region_sidecars(
            issues
        )
    )

    # --------------------------------------------------------
    # Manifests
    # --------------------------------------------------------

    print(
        "Auditing generated manifests..."
    )

    manifests = audit_manifests(
        issues
    )

    # --------------------------------------------------------
    # Existing issue files
    # --------------------------------------------------------

    print(
        "Checking unresolved issue CSV files..."
    )

    issue_csvs = audit_issue_csvs(
        issues
    )

    # --------------------------------------------------------
    # Visual outputs
    # --------------------------------------------------------

    print(
        "Auditing required visual artifacts..."
    )

    visual_artifacts = (
        audit_visual_artifacts(
            issues
        )
    )

    # --------------------------------------------------------
    # Semantic cross-report checks
    # --------------------------------------------------------

    print(
        "Auditing partition separation and "
        "cross-format equivalence..."
    )

    semantic_reports = (
        audit_semantic_reports(
            issues
        )
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    checks = {
        "all_previous_reports_passed": all(
            result["passed"]
            for result
            in previous_reports.values()
        ),

        "all_configurations_passed": all(
            result["passed"]
            for result
            in configurations.values()
        ),

        "all_partitions_passed": all(
            result["passed"]
            for result
            in partition_results.values()
        ),

        "combined_physical_totals": all(
            combined_physical_checks
            .values()
        ),

        "region_policy_passed": (
            region_policy[
                "passed"
            ]
        ),

        "all_manifests_passed": all(
            result["passed"]
            for result
            in manifests.values()
        ),

        "all_issue_csvs_empty": (
            bool(issue_csvs)
            and all(
                result["passed"]
                for result
                in issue_csvs.values()
            )
        ),

        "all_visual_artifacts_present": all(
            result["passed"]
            for result
            in visual_artifacts.values()
        ),

        "no_partition_overlap": (
            bool(
                semantic_reports[
                    "partition_overlap_checks"
                ]
            )
            and all(
                semantic_reports[
                    "partition_overlap_checks"
                ].values()
            )
        ),

        "coco_yolo_equivalence": (
            bool(
                semantic_reports[
                    "coco_yolo_equivalence_checks"
                ]
            )
            and all(
                semantic_reports[
                    "coco_yolo_equivalence_checks"
                ].values()
            )
        ),

        "dataloader_contract": (
            bool(
                semantic_reports[
                    "dataloader_checks"
                ]
            )
            and all(
                semantic_reports[
                    "dataloader_checks"
                ].values()
            )
        ),
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "final_audit",
                "final_check_failed",
                check_name,
                (
                    "The final Milestone 3 "
                    "dataset audit check "
                    "returned false."
                ),
            )

    overall_passed = (
        all(
            checks.values()
        )
        and len(issues) == 0
    )

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    report = {
        "milestone": 3,
        "step": 16,

        "purpose": (
            "Perform the final read-only "
            "end-to-end audit of the complete "
            "Milestone 3 dataset pipeline."
        ),

        "scope": {
            "read_only": True,
            "regenerates_images": False,
            "regenerates_annotations": False,
            "trains_models": False,
            "executes_detectors": False,
        },

        "expected_combined": (
            EXPECTED_COMBINED
        ),

        "previous_reports": (
            previous_reports
        ),

        "configurations": (
            configurations
        ),

        "partitions": (
            partition_results
        ),

        "combined_physical": (
            combined_physical
        ),

        "combined_physical_checks": (
            combined_physical_checks
        ),

        "region_policy": (
            region_policy
        ),

        "manifests": manifests,

        "issue_csvs": (
            issue_csvs
        ),

        "visual_artifacts": (
            visual_artifacts
        ),

        "semantic_reports": (
            semantic_reports
        ),

        "checks": checks,

        "issue_count": len(
            issues
        ),

        "final_dataset_audit_passed": (
            overall_passed
        ),
    }

    FINAL_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(
        FINAL_ISSUES_FILE,
        issues,
        [
            "section",
            "category",
            "identifier",
            "details",
        ],
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("FINAL DATASET AUDIT SUMMARY")
    print("=" * 76)

    print(
        "\nPrevious reports:"
    )

    print(
        f"  Passed: "
        f"{previous_reports_passed}/"
        f"{len(previous_reports)}"
    )

    print(
        "\nFrozen configurations:"
    )

    print(
        f"  Passed: "
        f"{configuration_passed_count}/"
        f"{len(configurations)}"
    )

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        result = partition_results[
            partition_name
        ]

        print(
            f"\n{partition_name}:"
        )

        print(
            f"  Images: "
            f"{result['image_count']}"
        )

        print(
            f"  COCO annotations: "
            f"{result['coco_summary'].get('annotations', 0)}"
        )

        print(
            f"  Canonical YOLO labels: "
            f"{result['canonical_label_count']}"
        )

        print(
            f"  Framework YOLO labels: "
            f"{result['framework_label_count']}"
        )

        print(
            f"  Empty labels: "
            f"{result['canonical_empty_labels']}"
        )

        print(
            f"  Status: "
            f"{'PASSED' if result['passed'] else 'FAILED'}"
        )

    print(
        "\nCombined dataset:"
    )

    print(
        f"  Images: "
        f"{combined_physical['images']}"
    )

    print(
        f"  Target annotations: "
        f"{combined_physical['coco_annotations']}"
    )

    print(
        f"  Vehicle: "
        f"{combined_physical['class_counts']['Vehicle']}"
    )

    print(
        f"  Pedestrian: "
        f"{combined_physical['class_counts']['Pedestrian']}"
    )

    print(
        f"  Cyclist: "
        f"{combined_physical['class_counts']['Cyclist']}"
    )

    print(
        f"  Negative images: "
        f"{combined_physical['negative_images']}"
    )

    print(
        f"  Evaluation-ignore regions: "
        f"{region_policy['total_evaluation_ignore']}"
    )

    print(
        f"  Excluded non-target regions: "
        f"{region_policy['total_excluded_non_target']}"
    )

    passed_manifests = sum(
        result["passed"]
        for result
        in manifests.values()
    )

    print(
        "\nGenerated manifests:"
    )

    print(
        f"  Passed: "
        f"{passed_manifests}/"
        f"{len(manifests)}"
    )

    unresolved_issue_rows = sum(
        int(
            result["rows"]
            or 0
        )
        for result
        in issue_csvs.values()
    )

    print(
        "\nIssue files:"
    )

    print(
        f"  Files checked: "
        f"{len(issue_csvs)}"
    )

    print(
        f"  Unresolved rows: "
        f"{unresolved_issue_rows}"
    )

    print(
        "\nCross-dataset safeguards:"
    )

    print(
        f"  No partition overlap: "
        f"{checks['no_partition_overlap']}"
    )

    print(
        f"  COCO–YOLO equivalence: "
        f"{checks['coco_yolo_equivalence']}"
    )

    print(
        f"  DataLoader contract: "
        f"{checks['dataloader_contract']}"
    )

    print(
        f"\nIssues found: "
        f"{len(issues)}"
    )

    print(
        "\nFinal status: "
        + (
            "PASSED"
            if overall_passed
            else "FAILED"
        )
    )

    print(
        f"\nFinal audit report:\n"
        f"{FINAL_REPORT_FILE.resolve()}"
    )

    print(
        f"\nFinal audit issues:\n"
        f"{FINAL_ISSUES_FILE.resolve()}"
    )

    if not overall_passed:
        print(
            "\nDo not continue to Milestone 3 "
            "packaging until every final-audit "
            "issue is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 16 completed successfully. "
        "The finalized dataset pipeline is "
        "internally consistent and ready "
        "for reproducibility packaging."
    )


if __name__ == "__main__":
    main()