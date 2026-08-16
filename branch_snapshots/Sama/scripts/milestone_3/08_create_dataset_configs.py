from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import yaml
from tqdm import tqdm

from ultralytics.data.utils import (
    check_det_dataset,
    img2label_paths,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path.cwd()

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

CLASS_MAPPING_CONFIG = Path(
    "configs/datasets/milestone_3/class_mapping.yaml"
)

PREPROCESSING_CONFIG = Path(
    "configs/datasets/milestone_3/preprocessing.yaml"
)

YOLO_CONVERSION_REPORT = Path(
    "data/processed/milestone_3/reports/"
    "yolo_conversion_report.json"
)

COCO_CREATION_REPORT = Path(
    "data/processed/milestone_3/reports/"
    "coco_creation_report.json"
)

REGION_POLICY_REPORT = Path(
    "data/processed/milestone_3/reports/"
    "region_policy_report.json"
)


# Framework-ready configurations
YOLO_CONFIG_FILE = Path(
    "configs/datasets/milestone_3/"
    "kitti_waymo_yolo.yaml"
)

COCO_CONFIG_FILE = Path(
    "configs/datasets/milestone_3/"
    "coco_paths.yaml"
)

DATASET_REGISTRY_FILE = Path(
    "configs/datasets/milestone_3/"
    "dataset_registry.yaml"
)


# Reports
REPORT_FILE = Path(
    "data/processed/milestone_3/reports/"
    "config_validation.json"
)

ISSUES_FILE = Path(
    "data/processed/milestone_3/reports/"
    "config_validation_issues.csv"
)

ADAPTER_MANIFEST_FILE = Path(
    "data/processed/milestone_3/manifests/"
    "framework_label_adapter_manifest.csv"
)


# ============================================================
# PARTITIONS
# ============================================================

PARTITIONS = {
    "kitti_train": {
        "role": "model_training",

        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/train"
        ),

        "canonical_yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/kitti/train"
        ),

        "framework_label_dir": (
            PROCESSED_ROOT
            / "labels/kitti/train"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_train.json"
        ),

        "ignore_file": (
            PROCESSED_ROOT
            / "annotations/ignore_regions/"
            "kitti_train_ignore.json"
        ),

        "excluded_file": (
            PROCESSED_ROOT
            / "annotations/excluded_objects/"
            "kitti_train_excluded.json"
        ),
    },

    "kitti_val": {
        "role": "in_domain_validation",

        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/val"
        ),

        "canonical_yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/kitti/val"
        ),

        "framework_label_dir": (
            PROCESSED_ROOT
            / "labels/kitti/val"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_val.json"
        ),

        "ignore_file": (
            PROCESSED_ROOT
            / "annotations/ignore_regions/"
            "kitti_val_ignore.json"
        ),

        "excluded_file": (
            PROCESSED_ROOT
            / "annotations/excluded_objects/"
            "kitti_val_excluded.json"
        ),
    },

    "waymo_external": {
        "role": "external_validation_only",

        "image_dir": (
            PROCESSED_ROOT
            / "images/waymo/external"
        ),

        "canonical_yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/waymo/external"
        ),

        "framework_label_dir": (
            PROCESSED_ROOT
            / "labels/waymo/external"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/waymo_external.json"
        ),

        "ignore_file": (
            PROCESSED_ROOT
            / "annotations/ignore_regions/"
            "waymo_external_ignore.json"
        ),

        "excluded_file": (
            PROCESSED_ROOT
            / "annotations/excluded_objects/"
            "waymo_external_excluded.json"
        ),
    },
}


EXPECTED = {
    "kitti_train": {
        "images": 5985,
        "labels": 5985,
        "empty_labels": 0,
    },

    "kitti_val": {
        "images": 1496,
        "labels": 1496,
        "empty_labels": 0,
    },

    "waymo_external": {
        "images": 996,
        "labels": 996,
        "empty_labels": 12,
    },

    "combined": {
        "images": 8477,
        "labels": 8477,
        "empty_labels": 12,
    },
}


ADAPTER_MANIFEST_COLUMNS = [
    "partition",
    "image_filename",
    "image_path",
    "canonical_label_path",
    "framework_label_path",
    "is_empty",
    "size_bytes",
    "canonical_sha256",
    "framework_sha256",
    "hashes_match",
    "status",
]


# ============================================================
# HELPERS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create framework-ready Milestone 3 "
            "dataset configurations and label adapters."
        )
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete existing generated framework "
            "label mirrors before recreating them."
        ),
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found:\n"
            f"{path.resolve()}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required YAML file not found:\n"
            f"{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(
            f"YAML root must be a mapping:\n"
            f"{path.resolve()}"
        )

    return content


def save_yaml(
    path: Path,
    content: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            yaml.safe_dump(
                content,
                file,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_csv(
    path: Path,
    rows: list[dict],
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


def add_issue(
    issues: list[dict],
    partition: str,
    category: str,
    identifier: str,
    details: str,
) -> None:
    issues.append(
        {
            "partition": partition,
            "category": category,
            "identifier": identifier,
            "details": details,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def project_relative_path(
    path: Path,
) -> str:
    try:
        return (
            path.resolve()
            .relative_to(
                PROJECT_ROOT.resolve()
            )
            .as_posix()
        )

    except ValueError:
        return path.resolve().as_posix()


def copy_file_atomic(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_name(
        destination.name + ".tmp"
    )

    try:
        shutil.copyfile(
            source,
            temporary_path,
        )

        os.replace(
            temporary_path,
            destination,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def safely_clean_framework_labels() -> int:
    removed = 0

    labels_root = (
        PROCESSED_ROOT
        / "labels"
    ).resolve()

    for specification in (
        PARTITIONS.values()
    ):
        directory = (
            specification[
                "framework_label_dir"
            ]
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        resolved = directory.resolve()

        if not resolved.is_relative_to(
            labels_root
        ):
            raise RuntimeError(
                f"Unsafe label directory:\n"
                f"{resolved}"
            )

        for pattern in (
            "*.txt",
            "*.tmp",
        ):
            for path in directory.glob(
                pattern
            ):
                if path.is_file():
                    path.unlink()
                    removed += 1

    return removed


def build_class_maps(
    mapping_config: dict,
) -> tuple[
    dict[int, str],
    dict[int, str],
]:
    target_classes = (
        mapping_config.get(
            "target_classes",
            []
        )
    )

    yolo_names: dict[int, str] = {}
    coco_names: dict[int, str] = {}

    for entry in target_classes:
        class_name = str(
            entry["name"]
        )

        yolo_names[
            int(entry["yolo_id"])
        ] = class_name

        coco_names[
            int(entry["coco_id"])
        ] = class_name

    expected_yolo = {
        0: "Vehicle",
        1: "Pedestrian",
        2: "Cyclist",
    }

    expected_coco = {
        1: "Vehicle",
        2: "Pedestrian",
        3: "Cyclist",
    }

    if yolo_names != expected_yolo:
        raise ValueError(
            "Unexpected frozen YOLO class mapping."
        )

    if coco_names != expected_coco:
        raise ValueError(
            "Unexpected frozen COCO class mapping."
        )

    return yolo_names, coco_names


# ============================================================
# LABEL ADAPTER
# ============================================================

def build_framework_label_adapter(
    partition_name: str,
    specification: dict,
    issues: list[dict],
) -> tuple[dict, list[dict]]:
    image_dir = Path(
        specification["image_dir"]
    )

    canonical_dir = Path(
        specification[
            "canonical_yolo_dir"
        ]
    )

    framework_dir = Path(
        specification[
            "framework_label_dir"
        ]
    )

    framework_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        path
        for path in image_dir.glob(
            "*.png"
        )
        if path.is_file()
    )

    canonical_files = sorted(
        path
        for path in canonical_dir.glob(
            "*.txt"
        )
        if path.is_file()
    )

    image_stems = {
        path.stem
        for path in image_files
    }

    canonical_stems = {
        path.stem
        for path in canonical_files
    }

    missing_canonical_stems = sorted(
        image_stems - canonical_stems
    )

    unknown_canonical_stems = sorted(
        canonical_stems - image_stems
    )

    for stem in missing_canonical_stems:
        add_issue(
            issues,
            partition_name,
            "missing_canonical_label",
            stem,
            (
                "Processed image has no canonical "
                "YOLO label file."
            ),
        )

    for stem in unknown_canonical_stems:
        add_issue(
            issues,
            partition_name,
            "canonical_label_without_image",
            stem,
            (
                "Canonical YOLO label has no "
                "processed image."
            ),
        )

    manifest_rows: list[dict] = []

    status_counts: Counter = Counter()

    print(
        f"\nCreating framework adapter for "
        f"{partition_name}..."
    )

    for image_path in tqdm(
        image_files,
        unit="label",
    ):
        canonical_path = (
            canonical_dir
            / f"{image_path.stem}.txt"
        )

        framework_path = (
            framework_dir
            / f"{image_path.stem}.txt"
        )

        if not canonical_path.exists():
            continue

        canonical_hash = (
            sha256_file(
                canonical_path
            )
        )

        status = "copied"

        if framework_path.exists():
            framework_hash_before = (
                sha256_file(
                    framework_path
                )
            )

            if (
                framework_hash_before
                == canonical_hash
            ):
                status = "reused_existing"

            else:
                copy_file_atomic(
                    canonical_path,
                    framework_path,
                )

                status = (
                    "replaced_hash_mismatch"
                )

        else:
            copy_file_atomic(
                canonical_path,
                framework_path,
            )

        framework_hash = (
            sha256_file(
                framework_path
            )
        )

        hashes_match = (
            canonical_hash
            == framework_hash
        )

        if not hashes_match:
            add_issue(
                issues,
                partition_name,
                "adapter_hash_mismatch",
                image_path.stem,
                (
                    "Framework label differs from "
                    "the canonical YOLO label."
                ),
            )

        is_empty = (
            framework_path.stat().st_size
            == 0
        )

        status_counts[status] += 1

        manifest_rows.append(
            {
                "partition": (
                    partition_name
                ),
                "image_filename": (
                    image_path.name
                ),
                "image_path": (
                    project_relative_path(
                        image_path
                    )
                ),
                "canonical_label_path": (
                    project_relative_path(
                        canonical_path
                    )
                ),
                "framework_label_path": (
                    project_relative_path(
                        framework_path
                    )
                ),
                "is_empty": is_empty,
                "size_bytes": (
                    framework_path.stat().st_size
                ),
                "canonical_sha256": (
                    canonical_hash
                ),
                "framework_sha256": (
                    framework_hash
                ),
                "hashes_match": (
                    hashes_match
                ),
                "status": status,
            }
        )

    actual_framework_files = {
        path
        for path in framework_dir.glob(
            "*.txt"
        )
        if path.is_file()
    }

    expected_framework_files = {
        framework_dir
        / f"{stem}.txt"
        for stem in image_stems
    }

    missing_framework_files = (
        expected_framework_files
        - actual_framework_files
    )

    extra_framework_files = (
        actual_framework_files
        - expected_framework_files
    )

    for path in sorted(
        missing_framework_files
    ):
        add_issue(
            issues,
            partition_name,
            "missing_framework_label",
            path.name,
            str(path),
        )

    for path in sorted(
        extra_framework_files
    ):
        add_issue(
            issues,
            partition_name,
            "unexpected_framework_label",
            path.name,
            str(path),
        )

    # Validate using Ultralytics' own path mapping.
    mapped_label_paths = {
        Path(path).resolve()
        for path in img2label_paths(
            [
                str(path.resolve())
                for path in image_files
            ]
        )
    }

    expected_resolved_paths = {
        path.resolve()
        for path in expected_framework_files
    }

    ultralytics_mapping_matches = (
        mapped_label_paths
        == expected_resolved_paths
    )

    if not ultralytics_mapping_matches:
        add_issue(
            issues,
            partition_name,
            "ultralytics_path_mapping_failed",
            partition_name,
            (
                "img2label_paths did not map the "
                "image directory to the generated "
                "labels directory."
            ),
        )

    empty_labels = sum(
        bool(row["is_empty"])
        for row in manifest_rows
    )

    expected = EXPECTED[
        partition_name
    ]

    checks = {
        "image_count": (
            len(image_files)
            == expected["images"]
        ),

        "canonical_label_count": (
            len(canonical_files)
            == expected["labels"]
        ),

        "framework_label_count": (
            len(actual_framework_files)
            == expected["labels"]
        ),

        "manifest_row_count": (
            len(manifest_rows)
            == expected["labels"]
        ),

        "empty_label_count": (
            empty_labels
            == expected["empty_labels"]
        ),

        "image_canonical_stem_match": (
            len(missing_canonical_stems) == 0
            and len(
                unknown_canonical_stems
            ) == 0
        ),

        "no_missing_framework_files": (
            len(missing_framework_files)
            == 0
        ),

        "no_extra_framework_files": (
            len(extra_framework_files)
            == 0
        ),

        "all_hashes_match": all(
            bool(row["hashes_match"])
            for row in manifest_rows
        ),

        "ultralytics_mapping_matches": (
            ultralytics_mapping_matches
        ),
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                partition_name,
                "adapter_check_failed",
                check_name,
                (
                    "Framework label adapter "
                    "check returned false."
                ),
            )

    return {
        "images": int(
            len(image_files)
        ),
        "canonical_labels": int(
            len(canonical_files)
        ),
        "framework_labels": int(
            len(actual_framework_files)
        ),
        "empty_labels": int(
            empty_labels
        ),
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "checks": checks,
        "validation_passed": all(
            checks.values()
        ),
    }, manifest_rows


# ============================================================
# CONFIG GENERATION
# ============================================================

def create_yolo_config(
    yolo_names: dict[int, str],
) -> dict:
    return {
        "path": (
            "data/processed/milestone_3"
        ),

        "train": (
            "images/kitti/train"
        ),

        "val": (
            "images/kitti/val"
        ),

        "test": (
            "images/waymo/external"
        ),

        "nc": len(yolo_names),

        "names": {
            int(class_id): class_name
            for class_id, class_name
            in sorted(
                yolo_names.items()
            )
        },
    }


def create_coco_config(
    coco_names: dict[int, str],
) -> dict:
    return {
        "schema_version": 1,
        "milestone": 3,

        "dataset_root": (
            "data/processed/milestone_3"
        ),

        "canonical_annotation_format": (
            "COCO"
        ),

        "image_preprocessing": {
            "width": 640,
            "height": 640,
            "method": (
                "centered_letterbox"
            ),
        },

        "categories": {
            int(class_id): class_name
            for class_id, class_name
            in sorted(
                coco_names.items()
            )
        },

        "partitions": {
            "kitti_train": {
                "role": (
                    "model_training"
                ),

                "images": (
                    "data/processed/milestone_3/"
                    "images/kitti/train"
                ),

                "annotations": (
                    "data/processed/milestone_3/"
                    "annotations/coco/"
                    "kitti_train.json"
                ),

                "evaluation_ignore": (
                    "data/processed/milestone_3/"
                    "annotations/ignore_regions/"
                    "kitti_train_ignore.json"
                ),

                "excluded_non_target": (
                    "data/processed/milestone_3/"
                    "annotations/excluded_objects/"
                    "kitti_train_excluded.json"
                ),
            },

            "kitti_validation": {
                "role": (
                    "in_domain_validation"
                ),

                "images": (
                    "data/processed/milestone_3/"
                    "images/kitti/val"
                ),

                "annotations": (
                    "data/processed/milestone_3/"
                    "annotations/coco/"
                    "kitti_val.json"
                ),

                "evaluation_ignore": (
                    "data/processed/milestone_3/"
                    "annotations/ignore_regions/"
                    "kitti_val_ignore.json"
                ),

                "excluded_non_target": (
                    "data/processed/milestone_3/"
                    "annotations/excluded_objects/"
                    "kitti_val_excluded.json"
                ),
            },

            "waymo_external": {
                "role": (
                    "external_validation_only"
                ),

                "images": (
                    "data/processed/milestone_3/"
                    "images/waymo/external"
                ),

                "annotations": (
                    "data/processed/milestone_3/"
                    "annotations/coco/"
                    "waymo_external.json"
                ),

                "evaluation_ignore": (
                    "data/processed/milestone_3/"
                    "annotations/ignore_regions/"
                    "waymo_external_ignore.json"
                ),

                "excluded_non_target": (
                    "data/processed/milestone_3/"
                    "annotations/excluded_objects/"
                    "waymo_external_excluded.json"
                ),
            },
        },
    }


def create_dataset_registry() -> dict:
    return {
        "schema_version": 1,
        "milestone": 3,

        "task": (
            "three_class_object_detection"
        ),

        "processed_image_size": [
            640,
            640,
        ],

        "training_source": (
            "KITTI project training split"
        ),

        "in_domain_validation": (
            "KITTI project validation split"
        ),

        "external_validation": (
            "Waymo frozen representative subset"
        ),

        "waymo_usage_policy": {
            "training": False,
            "fine_tuning": False,
            "hyperparameter_selection": False,
            "checkpoint_selection": False,
            "external_evaluation_only": True,
        },

        "configurations": {
            "yolo": (
                "configs/datasets/milestone_3/"
                "kitti_waymo_yolo.yaml"
            ),

            "coco": (
                "configs/datasets/milestone_3/"
                "coco_paths.yaml"
            ),
        },

        "canonical_annotations": {
            "kitti_train": (
                "data/processed/milestone_3/"
                "annotations/coco/"
                "kitti_train.json"
            ),

            "kitti_validation": (
                "data/processed/milestone_3/"
                "annotations/coco/"
                "kitti_val.json"
            ),

            "waymo_external": (
                "data/processed/milestone_3/"
                "annotations/coco/"
                "waymo_external.json"
            ),
        },

        "derived_annotations": {
            "format": "YOLO",

            "canonical_storage": (
                "data/processed/milestone_3/"
                "annotations/yolo"
            ),

            "framework_adapter": (
                "data/processed/milestone_3/"
                "labels"
            ),
        },
    }


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_coco_config(
    coco_config: dict,
    issues: list[dict],
) -> dict:
    checks: dict[str, bool] = {}

    categories = (
        coco_config.get(
            "categories",
            {}
        )
    )

    checks["categories"] = (
        categories
        == {
            1: "Vehicle",
            2: "Pedestrian",
            3: "Cyclist",
        }
    )

    partitions = (
        coco_config.get(
            "partitions",
            {}
        )
    )

    expected_partition_names = {
        "kitti_train",
        "kitti_validation",
        "waymo_external",
    }

    checks["partition_names"] = (
        set(partitions.keys())
        == expected_partition_names
    )

    path_checks = {}

    for partition_name, values in (
        partitions.items()
    ):
        for field in [
            "images",
            "annotations",
            "evaluation_ignore",
            "excluded_non_target",
        ]:
            path = Path(
                str(values[field])
            )

            path_exists = path.exists()

            path_checks[
                f"{partition_name}:{field}"
            ] = path_exists

            if not path_exists:
                add_issue(
                    issues,
                    partition_name,
                    "missing_coco_config_path",
                    field,
                    str(path),
                )

    checks["all_paths_exist"] = all(
        path_checks.values()
    )

    expected_roles = {
        "kitti_train": (
            "model_training"
        ),

        "kitti_validation": (
            "in_domain_validation"
        ),

        "waymo_external": (
            "external_validation_only"
        ),
    }

    checks["partition_roles"] = all(
        partitions[name]["role"]
        == role
        for name, role
        in expected_roles.items()
    )

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "COCO",
                "coco_config_check_failed",
                check_name,
                (
                    "COCO configuration check "
                    "returned false."
                ),
            )

    return {
        "checks": checks,
        "path_checks": path_checks,
        "validation_passed": all(
            checks.values()
        ),
    }


def validate_yolo_config(
    issues: list[dict],
) -> dict:
    resolved = check_det_dataset(
        str(YOLO_CONFIG_FILE),
        autodownload=False,
    )

    resolved_train = Path(
        str(resolved["train"])
    ).resolve()

    resolved_val = Path(
        str(resolved["val"])
    ).resolve()

    resolved_test = Path(
        str(resolved["test"])
    ).resolve()

    expected_train = (
        PARTITIONS[
            "kitti_train"
        ]["image_dir"].resolve()
    )

    expected_val = (
        PARTITIONS[
            "kitti_val"
        ]["image_dir"].resolve()
    )

    expected_test = (
        PARTITIONS[
            "waymo_external"
        ]["image_dir"].resolve()
    )

    names = {
        int(key): str(value)
        for key, value
        in resolved["names"].items()
    }

    checks = {
        "train_path": (
            resolved_train
            == expected_train
        ),

        "validation_path": (
            resolved_val
            == expected_val
        ),

        "test_path": (
            resolved_test
            == expected_test
        ),

        "class_count": (
            int(resolved["nc"])
            == 3
        ),

        "class_names": (
            names
            == {
                0: "Vehicle",
                1: "Pedestrian",
                2: "Cyclist",
            }
        ),
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "YOLO",
                "yolo_config_check_failed",
                check_name,
                (
                    "Ultralytics-resolved dataset "
                    "configuration differs from "
                    "the frozen paths."
                ),
            )

    return {
        "resolved_path": str(
            Path(
                resolved["path"]
            ).resolve()
        ),

        "resolved_train": str(
            resolved_train
        ),

        "resolved_val": str(
            resolved_val
        ),

        "resolved_test": str(
            resolved_test
        ),

        "names": names,
        "checks": checks,
        "validation_passed": all(
            checks.values()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    arguments = parse_arguments()

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ADAPTER_MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []

    print("=" * 76)
    print("CREATING FRAMEWORK-READY DATASET CONFIGURATIONS")
    print("=" * 76)

    preprocessing_config = load_yaml(
        PREPROCESSING_CONFIG
    )

    mapping_config = load_yaml(
        CLASS_MAPPING_CONFIG
    )

    yolo_report = load_json(
        YOLO_CONVERSION_REPORT
    )

    coco_report = load_json(
        COCO_CREATION_REPORT
    )

    region_report = load_json(
        REGION_POLICY_REPORT
    )

    if not yolo_report.get(
        "yolo_conversion_passed",
        False,
    ):
        raise RuntimeError(
            "Step 8 YOLO conversion has not passed."
        )

    if not coco_report.get(
        "coco_creation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 6 COCO creation has not passed."
        )

    if not region_report.get(
        "region_policy_passed",
        False,
    ):
        raise RuntimeError(
            "Step 7 region policy has not passed."
        )

    if (
        preprocessing_config[
            "annotation_policy"
        ]["canonical_format"]
        != "COCO"
    ):
        raise ValueError(
            "COCO must remain the canonical format."
        )

    yolo_names, coco_names = (
        build_class_maps(
            mapping_config
        )
    )

    if arguments.clean:
        removed = (
            safely_clean_framework_labels()
        )

        print(
            f"Existing framework labels removed: "
            f"{removed}"
        )

    adapter_results = {}
    all_manifest_rows: list[dict] = []

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        result, manifest_rows = (
            build_framework_label_adapter(
                partition_name=(
                    partition_name
                ),
                specification=(
                    PARTITIONS[
                        partition_name
                    ]
                ),
                issues=issues,
            )
        )

        adapter_results[
            partition_name
        ] = result

        all_manifest_rows.extend(
            manifest_rows
        )

    all_manifest_rows.sort(
        key=lambda row: (
            row["partition"],
            row["image_filename"],
        )
    )

    write_csv(
        ADAPTER_MANIFEST_FILE,
        all_manifest_rows,
        ADAPTER_MANIFEST_COLUMNS,
    )

    yolo_config = create_yolo_config(
        yolo_names
    )

    coco_config = create_coco_config(
        coco_names
    )

    dataset_registry = (
        create_dataset_registry()
    )

    save_yaml(
        YOLO_CONFIG_FILE,
        yolo_config,
    )

    save_yaml(
        COCO_CONFIG_FILE,
        coco_config,
    )

    save_yaml(
        DATASET_REGISTRY_FILE,
        dataset_registry,
    )

    # Re-read to ensure serialization is valid.
    saved_yolo_config = load_yaml(
        YOLO_CONFIG_FILE
    )

    saved_coco_config = load_yaml(
        COCO_CONFIG_FILE
    )

    saved_registry = load_yaml(
        DATASET_REGISTRY_FILE
    )

    if saved_yolo_config != yolo_config:
        add_issue(
            issues,
            "YOLO",
            "serialization_mismatch",
            str(YOLO_CONFIG_FILE),
            (
                "Saved YOLO YAML differs from "
                "the generated dictionary."
            ),
        )

    if saved_coco_config != coco_config:
        add_issue(
            issues,
            "COCO",
            "serialization_mismatch",
            str(COCO_CONFIG_FILE),
            (
                "Saved COCO YAML differs from "
                "the generated dictionary."
            ),
        )

    if saved_registry != dataset_registry:
        add_issue(
            issues,
            "Registry",
            "serialization_mismatch",
            str(DATASET_REGISTRY_FILE),
            (
                "Saved registry YAML differs from "
                "the generated dictionary."
            ),
        )

    yolo_validation = (
        validate_yolo_config(
            issues
        )
    )

    coco_validation = (
        validate_coco_config(
            saved_coco_config,
            issues,
        )
    )

    combined_adapter = {
        "images": int(
            sum(
                result["images"]
                for result
                in adapter_results.values()
            )
        ),

        "canonical_labels": int(
            sum(
                result[
                    "canonical_labels"
                ]
                for result
                in adapter_results.values()
            )
        ),

        "framework_labels": int(
            sum(
                result[
                    "framework_labels"
                ]
                for result
                in adapter_results.values()
            )
        ),

        "empty_labels": int(
            sum(
                result[
                    "empty_labels"
                ]
                for result
                in adapter_results.values()
            )
        ),
    }

    combined_checks = {
        "image_count": (
            combined_adapter["images"]
            == EXPECTED[
                "combined"
            ]["images"]
        ),

        "canonical_label_count": (
            combined_adapter[
                "canonical_labels"
            ]
            == EXPECTED[
                "combined"
            ]["labels"]
        ),

        "framework_label_count": (
            combined_adapter[
                "framework_labels"
            ]
            == EXPECTED[
                "combined"
            ]["labels"]
        ),

        "empty_label_count": (
            combined_adapter[
                "empty_labels"
            ]
            == EXPECTED[
                "combined"
            ]["empty_labels"]
        ),

        "adapter_manifest_rows": (
            len(all_manifest_rows)
            == EXPECTED[
                "combined"
            ]["labels"]
        ),

        "all_partition_adapters_passed": all(
            result[
                "validation_passed"
            ]
            for result
            in adapter_results.values()
        ),

        "yolo_config_passed": (
            yolo_validation[
                "validation_passed"
            ]
        ),

        "coco_config_passed": (
            coco_validation[
                "validation_passed"
            ]
        ),
    }

    for check_name, passed in (
        combined_checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "Combined",
                "combined_config_check_failed",
                check_name,
                (
                    "Combined configuration check "
                    "returned false."
                ),
            )

    overall_passed = (
        all(
            combined_checks.values()
        )
        and len(issues) == 0
    )

    report = {
        "milestone": 3,
        "step": 9,

        "purpose": (
            "Create framework-ready YOLO and "
            "COCO dataset configurations."
        ),

        "canonical_yolo_storage": (
            "data/processed/milestone_3/"
            "annotations/yolo"
        ),

        "framework_yolo_label_adapter": (
            "data/processed/milestone_3/"
            "labels"
        ),

        "adapter_partitions": (
            adapter_results
        ),

        "combined_adapter": (
            combined_adapter
        ),

        "yolo_configuration": {
            "path": (
                YOLO_CONFIG_FILE.as_posix()
            ),

            "sha256": (
                sha256_file(
                    YOLO_CONFIG_FILE
                )
            ),

            "validation": (
                yolo_validation
            ),
        },

        "coco_configuration": {
            "path": (
                COCO_CONFIG_FILE.as_posix()
            ),

            "sha256": (
                sha256_file(
                    COCO_CONFIG_FILE
                )
            ),

            "validation": (
                coco_validation
            ),
        },

        "dataset_registry": {
            "path": (
                DATASET_REGISTRY_FILE
                .as_posix()
            ),

            "sha256": (
                sha256_file(
                    DATASET_REGISTRY_FILE
                )
            ),
        },

        "adapter_manifest": {
            "path": (
                ADAPTER_MANIFEST_FILE
                .as_posix()
            ),

            "rows": int(
                len(all_manifest_rows)
            ),

            "sha256": (
                sha256_file(
                    ADAPTER_MANIFEST_FILE
                )
            ),
        },

        "combined_checks": (
            combined_checks
        ),

        "issue_count": len(issues),

        "config_validation_passed": (
            overall_passed
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(
        ISSUES_FILE,
        issues,
        [
            "partition",
            "category",
            "identifier",
            "details",
        ],
    )

    print("\n" + "=" * 76)
    print("FRAMEWORK CONFIGURATION SUMMARY")
    print("=" * 76)

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        result = adapter_results[
            partition_name
        ]

        print(f"\n{partition_name}:")

        print(
            f"  Images: "
            f"{result['images']}"
        )

        print(
            f"  Canonical labels: "
            f"{result['canonical_labels']}"
        )

        print(
            f"  Framework labels: "
            f"{result['framework_labels']}"
        )

        print(
            f"  Empty labels: "
            f"{result['empty_labels']}"
        )

        print(
            f"  Status: "
            f"{'PASSED' if result['validation_passed'] else 'FAILED'}"
        )

    print("\nCombined:")

    print(
        f"  Images: "
        f"{combined_adapter['images']}"
    )

    print(
        f"  Canonical labels: "
        f"{combined_adapter['canonical_labels']}"
    )

    print(
        f"  Framework labels: "
        f"{combined_adapter['framework_labels']}"
    )

    print(
        f"  Empty labels: "
        f"{combined_adapter['empty_labels']}"
    )

    print(
        f"\nYOLO configuration: "
        f"{'PASSED' if yolo_validation['validation_passed'] else 'FAILED'}"
    )

    print(
        f"COCO configuration: "
        f"{'PASSED' if coco_validation['validation_passed'] else 'FAILED'}"
    )

    print(
        f"Issues found: "
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

    print("\nCreated configuration files:")

    print(
        f"  {YOLO_CONFIG_FILE.resolve()}"
    )

    print(
        f"  {COCO_CONFIG_FILE.resolve()}"
    )

    print(
        f"  {DATASET_REGISTRY_FILE.resolve()}"
    )

    print(
        f"\nAdapter manifest:\n"
        f"{ADAPTER_MANIFEST_FILE.resolve()}"
    )

    print(
        f"\nReport:\n"
        f"{REPORT_FILE.resolve()}"
    )

    print(
        f"\nIssues:\n"
        f"{ISSUES_FILE.resolve()}"
    )

    if not overall_passed:
        print(
            "\nDo not continue until every "
            "framework configuration issue "
            "is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 9 completed successfully."
    )


if __name__ == "__main__":
    main()