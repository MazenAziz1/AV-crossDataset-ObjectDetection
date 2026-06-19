from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import pandas as pd
import yaml
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

SOURCE_MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/source_manifest.csv"
)

YOLO_CONFIG_FILE = Path(
    "configs/datasets/milestone_3/"
    "kitti_waymo_yolo.yaml"
)

CLASS_MAPPING_FILE = Path(
    "configs/datasets/milestone_3/"
    "class_mapping.yaml"
)

CONFIG_VALIDATION_REPORT = (
    PROCESSED_ROOT
    / "reports/config_validation.json"
)

COCO_VALIDATION_REPORT = (
    PROCESSED_ROOT
    / "reports/coco_validation_report.json"
)

REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/yolo_validation_report.json"
)

ISSUES_FILE = (
    PROCESSED_ROOT
    / "reports/yolo_validation_issues.csv"
)


# ============================================================
# PARTITIONS
# ============================================================

PARTITIONS = {
    "kitti_train": {
        "dataset": "KITTI",
        "partition": "train",

        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/train"
        ),

        "canonical_label_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/kitti/train"
        ),

        "framework_label_dir": (
            PROCESSED_ROOT
            / "labels/kitti/train"
        ),

        "expected": {
            "images": 5985,
            "labels": 5985,
            "annotation_rows": 31294,
            "empty_labels": 0,
            "Vehicle": 26278,
            "Pedestrian": 3729,
            "Cyclist": 1287,
        },
    },

    "kitti_val": {
        "dataset": "KITTI",
        "partition": "val",

        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/val"
        ),

        "canonical_label_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/kitti/val"
        ),

        "framework_label_dir": (
            PROCESSED_ROOT
            / "labels/kitti/val"
        ),

        "expected": {
            "images": 1496,
            "labels": 1496,
            "annotation_rows": 7792,
            "empty_labels": 0,
            "Vehicle": 6472,
            "Pedestrian": 980,
            "Cyclist": 340,
        },
    },

    "waymo_external": {
        "dataset": "Waymo",
        "partition": "external",

        "image_dir": (
            PROCESSED_ROOT
            / "images/waymo/external"
        ),

        "canonical_label_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/waymo/external"
        ),

        "framework_label_dir": (
            PROCESSED_ROOT
            / "labels/waymo/external"
        ),

        "expected": {
            "images": 996,
            "labels": 996,
            "annotation_rows": 24819,
            "empty_labels": 12,
            "Vehicle": 16928,
            "Pedestrian": 7127,
            "Cyclist": 764,
        },
    },
}


EXPECTED_COMBINED = {
    "images": 8477,
    "labels": 8477,
    "annotation_rows": 63905,
    "empty_labels": 12,
    "Vehicle": 49678,
    "Pedestrian": 11836,
    "Cyclist": 2391,
}


# Small tolerance for values rounded to ten decimal places.
NORMALIZED_TOLERANCE = 1e-9


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found:\n"
            f"{path.resolve()}"
        )

    content = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            f"JSON root must be an object:\n"
            f"{path.resolve()}"
        )

    return content


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


def build_class_mapping(
    mapping_config: dict,
) -> dict[int, str]:
    target_classes = mapping_config.get(
        "target_classes",
        [],
    )

    mapping = {
        int(entry["yolo_id"]): str(
            entry["name"]
        )
        for entry in target_classes
    }

    expected = {
        0: "Vehicle",
        1: "Pedestrian",
        2: "Cyclist",
    }

    if mapping != expected:
        raise ValueError(
            "The frozen YOLO mapping must be "
            "0=Vehicle, 1=Pedestrian, 2=Cyclist."
        )

    return mapping


def validate_yolo_config(
    configuration: dict,
    issues: list[dict],
) -> dict:
    configured_names = {
        int(key): str(value)
        for key, value in (
            configuration.get(
                "names",
                {}
            ).items()
        )
    }

    checks = {
        "dataset_root": (
            str(
                configuration.get(
                    "path",
                    "",
                )
            )
            == "data/processed/milestone_3"
        ),

        "train_path": (
            str(
                configuration.get(
                    "train",
                    "",
                )
            )
            == "images/kitti/train"
        ),

        "validation_path": (
            str(
                configuration.get(
                    "val",
                    "",
                )
            )
            == "images/kitti/val"
        ),

        "external_path": (
            str(
                configuration.get(
                    "test",
                    "",
                )
            )
            == "images/waymo/external"
        ),

        "class_count": (
            int(
                configuration.get(
                    "nc",
                    -1,
                )
            )
            == 3
        ),

        "class_names": (
            configured_names
            == {
                0: "Vehicle",
                1: "Pedestrian",
                2: "Cyclist",
            }
        ),
    }

    for check_name, passed in checks.items():
        if not passed:
            add_issue(
                issues,
                "configuration",
                "yolo_config_check_failed",
                check_name,
                (
                    "The YOLO dataset configuration "
                    "does not match the frozen policy."
                ),
            )

    return {
        "checks": checks,
        "validation_passed": all(
            checks.values()
        ),
    }


def check_normalized_box(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[bool, dict]:
    values = {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
    }

    finite = all(
        math.isfinite(value)
        for value in values.values()
    )

    if not finite:
        return False, {
            "finite": False,
            "center_range": False,
            "size_range": False,
            "box_in_bounds": False,
        }

    center_range = (
        -NORMALIZED_TOLERANCE
        <= center_x
        <= 1.0 + NORMALIZED_TOLERANCE
        and
        -NORMALIZED_TOLERANCE
        <= center_y
        <= 1.0 + NORMALIZED_TOLERANCE
    )

    size_range = (
        width > 0.0
        and height > 0.0
        and width
        <= 1.0 + NORMALIZED_TOLERANCE
        and height
        <= 1.0 + NORMALIZED_TOLERANCE
    )

    xmin = (
        center_x - width / 2.0
    )

    ymin = (
        center_y - height / 2.0
    )

    xmax = (
        center_x + width / 2.0
    )

    ymax = (
        center_y + height / 2.0
    )

    box_in_bounds = (
        xmin >= -NORMALIZED_TOLERANCE
        and ymin >= -NORMALIZED_TOLERANCE
        and xmax
        <= 1.0 + NORMALIZED_TOLERANCE
        and ymax
        <= 1.0 + NORMALIZED_TOLERANCE
    )

    checks = {
        "finite": finite,
        "center_range": center_range,
        "size_range": size_range,
        "box_in_bounds": box_in_bounds,
    }

    return all(
        checks.values()
    ), checks


# ============================================================
# PARTITION VALIDATION
# ============================================================

def validate_partition(
    partition_name: str,
    specification: dict,
    source_manifest: pd.DataFrame,
    class_mapping: dict[int, str],
    issues: list[dict],
) -> dict:
    image_dir = Path(
        specification["image_dir"]
    )

    canonical_label_dir = Path(
        specification[
            "canonical_label_dir"
        ]
    )

    framework_label_dir = Path(
        specification[
            "framework_label_dir"
        ]
    )

    expected = specification[
        "expected"
    ]

    partition_rows = (
        source_manifest[
            (
                source_manifest["dataset"]
                == specification["dataset"]
            )
            & (
                source_manifest["partition"]
                == specification["partition"]
            )
        ]
        .copy()
        .sort_values(
            "global_image_id"
        )
        .reset_index(drop=True)
    )

    expected_image_names = {
        str(filename)
        for filename in (
            partition_rows[
                "output_filename"
            ].tolist()
        )
    }

    expected_label_names = {
        f"{Path(filename).stem}.txt"
        for filename
        in expected_image_names
    }

    physical_image_names = {
        path.name
        for path in image_dir.glob(
            "*.png"
        )
        if path.is_file()
    }

    canonical_label_names = {
        path.name
        for path
        in canonical_label_dir.glob(
            "*.txt"
        )
        if path.is_file()
    }

    framework_label_names = {
        path.name
        for path
        in framework_label_dir.glob(
            "*.txt"
        )
        if path.is_file()
    }

    file_set_differences = {
        "missing_images": sorted(
            expected_image_names
            - physical_image_names
        ),

        "extra_images": sorted(
            physical_image_names
            - expected_image_names
        ),

        "missing_canonical_labels": sorted(
            expected_label_names
            - canonical_label_names
        ),

        "extra_canonical_labels": sorted(
            canonical_label_names
            - expected_label_names
        ),

        "missing_framework_labels": sorted(
            expected_label_names
            - framework_label_names
        ),

        "extra_framework_labels": sorted(
            framework_label_names
            - expected_label_names
        ),
    }

    for category, values in (
        file_set_differences.items()
    ):
        for value in values:
            add_issue(
                issues,
                partition_name,
                category,
                value,
                (
                    "The physical filename set "
                    "does not match the source manifest."
                ),
            )

    total_rows = 0
    empty_labels = 0

    class_counts: Counter = Counter()

    invalid_field_rows = 0
    invalid_class_rows = 0
    non_numeric_rows = 0
    invalid_coordinate_rows = 0
    hash_mismatches = 0
    per_image_count_mismatches = 0
    per_image_class_mismatches = 0
    unexpected_empty_labels = 0
    unexpected_nonempty_labels = 0

    minimum_coordinates = {
        "center_x": None,
        "center_y": None,
        "width": None,
        "height": None,
    }

    maximum_coordinates = {
        "center_x": None,
        "center_y": None,
        "width": None,
        "height": None,
    }

    print(
        f"\nValidating {partition_name} "
        f"YOLO labels..."
    )

    for _, manifest_row in tqdm(
        partition_rows.iterrows(),
        total=len(partition_rows),
        unit="label",
    ):
        global_image_id = int(
            manifest_row[
                "global_image_id"
            ]
        )

        image_filename = str(
            manifest_row[
                "output_filename"
            ]
        )

        label_filename = (
            f"{Path(image_filename).stem}.txt"
        )

        canonical_path = (
            canonical_label_dir
            / label_filename
        )

        framework_path = (
            framework_label_dir
            / label_filename
        )

        if (
            not canonical_path.exists()
            or not framework_path.exists()
        ):
            continue

        canonical_hash = sha256_file(
            canonical_path
        )

        framework_hash = sha256_file(
            framework_path
        )

        if canonical_hash != framework_hash:
            hash_mismatches += 1

            add_issue(
                issues,
                partition_name,
                "framework_hash_mismatch",
                label_filename,
                (
                    "Framework mirror differs from "
                    "the canonical YOLO label."
                ),
            )

        try:
            label_text = (
                canonical_path.read_text(
                    encoding="utf-8"
                )
            )

        except UnicodeDecodeError as error:
            add_issue(
                issues,
                partition_name,
                "invalid_utf8_label",
                label_filename,
                str(error),
            )
            continue

        lines = [
            line.strip()
            for line
            in label_text.splitlines()
            if line.strip()
        ]

        expected_target_count = int(
            manifest_row[
                "target_box_count"
            ]
        )

        if not lines:
            empty_labels += 1

        if (
            not lines
            and expected_target_count > 0
        ):
            unexpected_empty_labels += 1

            add_issue(
                issues,
                partition_name,
                "unexpected_empty_label",
                label_filename,
                (
                    f"Source manifest expects "
                    f"{expected_target_count} objects."
                ),
            )

        if (
            lines
            and expected_target_count == 0
        ):
            unexpected_nonempty_labels += 1

            add_issue(
                issues,
                partition_name,
                "unexpected_nonempty_label",
                label_filename,
                (
                    "Source manifest defines this "
                    "image as target-negative."
                ),
            )

        if (
            len(lines)
            != expected_target_count
        ):
            per_image_count_mismatches += 1

            add_issue(
                issues,
                partition_name,
                "per_image_count_mismatch",
                str(global_image_id),
                (
                    f"Expected "
                    f"{expected_target_count}, "
                    f"found {len(lines)}."
                ),
            )

        image_class_counts: Counter = (
            Counter()
        )

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            tokens = line.split()

            identifier = (
                f"{label_filename}:"
                f"{line_number}"
            )

            if len(tokens) != 5:
                invalid_field_rows += 1

                add_issue(
                    issues,
                    partition_name,
                    "invalid_field_count",
                    identifier,
                    (
                        f"Expected 5 tokens, "
                        f"found {len(tokens)}."
                    ),
                )
                continue

            try:
                class_id = int(
                    tokens[0]
                )

            except ValueError:
                invalid_class_rows += 1

                add_issue(
                    issues,
                    partition_name,
                    "non_integer_class_id",
                    identifier,
                    tokens[0],
                )
                continue

            if class_id not in class_mapping:
                invalid_class_rows += 1

                add_issue(
                    issues,
                    partition_name,
                    "unknown_class_id",
                    identifier,
                    str(class_id),
                )
                continue

            try:
                (
                    center_x,
                    center_y,
                    width,
                    height,
                ) = [
                    float(value)
                    for value in tokens[1:]
                ]

            except ValueError as error:
                non_numeric_rows += 1

                add_issue(
                    issues,
                    partition_name,
                    "non_numeric_coordinate",
                    identifier,
                    str(error),
                )
                continue

            box_valid, box_checks = (
                check_normalized_box(
                    center_x=center_x,
                    center_y=center_y,
                    width=width,
                    height=height,
                )
            )

            if not box_valid:
                invalid_coordinate_rows += 1

                add_issue(
                    issues,
                    partition_name,
                    "invalid_normalized_box",
                    identifier,
                    (
                        f"values="
                        f"{[center_x, center_y, width, height]}; "
                        f"checks={box_checks}"
                    ),
                )
                continue

            coordinate_values = {
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
            }

            for name, value in (
                coordinate_values.items()
            ):
                current_minimum = (
                    minimum_coordinates[
                        name
                    ]
                )

                current_maximum = (
                    maximum_coordinates[
                        name
                    ]
                )

                if (
                    current_minimum is None
                    or value < current_minimum
                ):
                    minimum_coordinates[
                        name
                    ] = value

                if (
                    current_maximum is None
                    or value > current_maximum
                ):
                    maximum_coordinates[
                        name
                    ] = value

            class_name = (
                class_mapping[
                    class_id
                ]
            )

            image_class_counts[
                class_name
            ] += 1

            class_counts[
                class_name
            ] += 1

            total_rows += 1

        expected_class_counts = {
            "Vehicle": int(
                manifest_row[
                    "vehicle_count"
                ]
            ),

            "Pedestrian": int(
                manifest_row[
                    "pedestrian_count"
                ]
            ),

            "Cyclist": int(
                manifest_row[
                    "cyclist_count"
                ]
            ),
        }

        actual_class_counts = {
            class_name: int(
                image_class_counts[
                    class_name
                ]
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        }

        if (
            actual_class_counts
            != expected_class_counts
        ):
            per_image_class_mismatches += 1

            add_issue(
                issues,
                partition_name,
                "per_image_class_mismatch",
                str(global_image_id),
                (
                    f"expected="
                    f"{expected_class_counts}; "
                    f"actual="
                    f"{actual_class_counts}"
                ),
            )

    summary = {
        "images": int(
            len(partition_rows)
        ),

        "physical_images": int(
            len(physical_image_names)
        ),

        "canonical_labels": int(
            len(canonical_label_names)
        ),

        "framework_labels": int(
            len(framework_label_names)
        ),

        "annotation_rows": int(
            total_rows
        ),

        "empty_labels": int(
            empty_labels
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

        "invalid_field_rows": int(
            invalid_field_rows
        ),

        "invalid_class_rows": int(
            invalid_class_rows
        ),

        "non_numeric_rows": int(
            non_numeric_rows
        ),

        "invalid_coordinate_rows": int(
            invalid_coordinate_rows
        ),

        "hash_mismatches": int(
            hash_mismatches
        ),

        "per_image_count_mismatches": int(
            per_image_count_mismatches
        ),

        "per_image_class_mismatches": int(
            per_image_class_mismatches
        ),

        "unexpected_empty_labels": int(
            unexpected_empty_labels
        ),

        "unexpected_nonempty_labels": int(
            unexpected_nonempty_labels
        ),

        "minimum_normalized_coordinates": (
            minimum_coordinates
        ),

        "maximum_normalized_coordinates": (
            maximum_coordinates
        ),

        "file_set_differences": {
            category: int(
                len(values)
            )
            for category, values
            in file_set_differences.items()
        },
    }

    checks = {
        "image_count": (
            summary["images"]
            == expected["images"]
        ),

        "physical_image_count": (
            summary[
                "physical_images"
            ]
            == expected["images"]
        ),

        "canonical_label_count": (
            summary[
                "canonical_labels"
            ]
            == expected["labels"]
        ),

        "framework_label_count": (
            summary[
                "framework_labels"
            ]
            == expected["labels"]
        ),

        "annotation_row_count": (
            summary[
                "annotation_rows"
            ]
            == expected[
                "annotation_rows"
            ]
        ),

        "empty_label_count": (
            summary[
                "empty_labels"
            ]
            == expected[
                "empty_labels"
            ]
        ),

        "vehicle_count": (
            summary[
                "class_counts"
            ]["Vehicle"]
            == expected["Vehicle"]
        ),

        "pedestrian_count": (
            summary[
                "class_counts"
            ]["Pedestrian"]
            == expected["Pedestrian"]
        ),

        "cyclist_count": (
            summary[
                "class_counts"
            ]["Cyclist"]
            == expected["Cyclist"]
        ),

        "exact_image_set": (
            len(
                file_set_differences[
                    "missing_images"
                ]
            )
            == 0
            and len(
                file_set_differences[
                    "extra_images"
                ]
            )
            == 0
        ),

        "exact_canonical_label_set": (
            len(
                file_set_differences[
                    "missing_canonical_labels"
                ]
            )
            == 0
            and len(
                file_set_differences[
                    "extra_canonical_labels"
                ]
            )
            == 0
        ),

        "exact_framework_label_set": (
            len(
                file_set_differences[
                    "missing_framework_labels"
                ]
            )
            == 0
            and len(
                file_set_differences[
                    "extra_framework_labels"
                ]
            )
            == 0
        ),

        "canonical_framework_hashes": (
            hash_mismatches == 0
        ),

        "valid_field_counts": (
            invalid_field_rows == 0
        ),

        "valid_class_ids": (
            invalid_class_rows == 0
        ),

        "numeric_coordinates": (
            non_numeric_rows == 0
        ),

        "valid_normalized_boxes": (
            invalid_coordinate_rows == 0
        ),

        "per_image_counts": (
            per_image_count_mismatches
            == 0
        ),

        "per_image_class_counts": (
            per_image_class_mismatches
            == 0
        ),

        "negative_image_policy": (
            unexpected_empty_labels == 0
            and unexpected_nonempty_labels
            == 0
        ),
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                partition_name,
                "partition_validation_failed",
                check_name,
                (
                    "The independent YOLO "
                    "validation check returned false."
                ),
            )

    return {
        **summary,
        "checks": checks,
        "validation_passed": all(
            checks.values()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []

    print("=" * 76)
    print("INDEPENDENT YOLO ANNOTATION VALIDATION")
    print("=" * 76)

    configuration_report = load_json(
        CONFIG_VALIDATION_REPORT
    )

    coco_validation_report = load_json(
        COCO_VALIDATION_REPORT
    )

    if not configuration_report.get(
        "config_validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 9 configuration validation "
            "has not passed."
        )

    if not coco_validation_report.get(
        "coco_validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 10 COCO validation has not passed."
        )

    yolo_configuration = load_yaml(
        YOLO_CONFIG_FILE
    )

    mapping_configuration = load_yaml(
        CLASS_MAPPING_FILE
    )

    class_mapping = build_class_mapping(
        mapping_configuration
    )

    configuration_validation = (
        validate_yolo_config(
            yolo_configuration,
            issues,
        )
    )

    if not SOURCE_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Source manifest not found:\n"
            f"{SOURCE_MANIFEST_FILE.resolve()}"
        )

    source_manifest = pd.read_csv(
        SOURCE_MANIFEST_FILE,
        dtype={
            "source_image_id": str,
            "output_filename": str,
        },
    )

    source_manifest[
        "global_image_id"
    ] = pd.to_numeric(
        source_manifest[
            "global_image_id"
        ],
        errors="raise",
    ).astype(int)

    if (
        source_manifest[
            "global_image_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "Source manifest contains duplicate "
            "global image IDs."
        )

    results = {}

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        results[
            partition_name
        ] = validate_partition(
            partition_name=(
                partition_name
            ),
            specification=(
                PARTITIONS[
                    partition_name
                ]
            ),
            source_manifest=(
                source_manifest
            ),
            class_mapping=(
                class_mapping
            ),
            issues=issues,
        )

    combined = {
        "images": int(
            sum(
                result["images"]
                for result
                in results.values()
            )
        ),

        "canonical_labels": int(
            sum(
                result[
                    "canonical_labels"
                ]
                for result
                in results.values()
            )
        ),

        "framework_labels": int(
            sum(
                result[
                    "framework_labels"
                ]
                for result
                in results.values()
            )
        ),

        "annotation_rows": int(
            sum(
                result[
                    "annotation_rows"
                ]
                for result
                in results.values()
            )
        ),

        "empty_labels": int(
            sum(
                result[
                    "empty_labels"
                ]
                for result
                in results.values()
            )
        ),

        "class_counts": {
            class_name: int(
                sum(
                    result[
                        "class_counts"
                    ][class_name]
                    for result
                    in results.values()
                )
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },

        "invalid_rows": int(
            sum(
                result[
                    "invalid_field_rows"
                ]
                + result[
                    "invalid_class_rows"
                ]
                + result[
                    "non_numeric_rows"
                ]
                + result[
                    "invalid_coordinate_rows"
                ]
                for result
                in results.values()
            )
        ),

        "hash_mismatches": int(
            sum(
                result[
                    "hash_mismatches"
                ]
                for result
                in results.values()
            )
        ),

        "per_image_count_mismatches": int(
            sum(
                result[
                    "per_image_count_mismatches"
                ]
                for result
                in results.values()
            )
        ),

        "per_image_class_mismatches": int(
            sum(
                result[
                    "per_image_class_mismatches"
                ]
                for result
                in results.values()
            )
        ),
    }

    combined_checks = {
        "image_count": (
            combined["images"]
            == EXPECTED_COMBINED[
                "images"
            ]
        ),

        "canonical_label_count": (
            combined[
                "canonical_labels"
            ]
            == EXPECTED_COMBINED[
                "labels"
            ]
        ),

        "framework_label_count": (
            combined[
                "framework_labels"
            ]
            == EXPECTED_COMBINED[
                "labels"
            ]
        ),

        "annotation_row_count": (
            combined[
                "annotation_rows"
            ]
            == EXPECTED_COMBINED[
                "annotation_rows"
            ]
        ),

        "empty_label_count": (
            combined[
                "empty_labels"
            ]
            == EXPECTED_COMBINED[
                "empty_labels"
            ]
        ),

        "vehicle_count": (
            combined[
                "class_counts"
            ]["Vehicle"]
            == EXPECTED_COMBINED[
                "Vehicle"
            ]
        ),

        "pedestrian_count": (
            combined[
                "class_counts"
            ]["Pedestrian"]
            == EXPECTED_COMBINED[
                "Pedestrian"
            ]
        ),

        "cyclist_count": (
            combined[
                "class_counts"
            ]["Cyclist"]
            == EXPECTED_COMBINED[
                "Cyclist"
            ]
        ),

        "no_invalid_rows": (
            combined[
                "invalid_rows"
            ]
            == 0
        ),

        "no_hash_mismatches": (
            combined[
                "hash_mismatches"
            ]
            == 0
        ),

        "no_per_image_count_mismatches": (
            combined[
                "per_image_count_mismatches"
            ]
            == 0
        ),

        "no_per_image_class_mismatches": (
            combined[
                "per_image_class_mismatches"
            ]
            == 0
        ),

        "all_partitions_passed": all(
            result[
                "validation_passed"
            ]
            for result
            in results.values()
        ),

        "configuration_passed": (
            configuration_validation[
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
                "combined",
                "combined_validation_failed",
                check_name,
                (
                    "The combined YOLO validation "
                    "check returned false."
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
        "step": 11,

        "purpose": (
            "Independently validate all canonical "
            "and framework YOLO label files."
        ),

        "class_mapping": {
            str(class_id): class_name
            for class_id, class_name
            in class_mapping.items()
        },

        "normalized_coordinate_tolerance": (
            NORMALIZED_TOLERANCE
        ),

        "configuration_validation": (
            configuration_validation
        ),

        "partitions": results,

        "combined": combined,

        "combined_checks": (
            combined_checks
        ),

        "issue_count": len(issues),

        "yolo_validation_passed": (
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
    print("YOLO VALIDATION SUMMARY")
    print("=" * 76)

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        result = results[
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
            f"  Annotation rows: "
            f"{result['annotation_rows']}"
        )

        print(
            f"  Empty labels: "
            f"{result['empty_labels']}"
        )

        print(
            f"  Invalid coordinate rows: "
            f"{result['invalid_coordinate_rows']}"
        )

        print(
            f"  Hash mismatches: "
            f"{result['hash_mismatches']}"
        )

        print(
            f"  Per-image count mismatches: "
            f"{result['per_image_count_mismatches']}"
        )

        print(
            f"  Per-image class mismatches: "
            f"{result['per_image_class_mismatches']}"
        )

        print(
            f"  Status: "
            f"{'PASSED' if result['validation_passed'] else 'FAILED'}"
        )

    print("\nCombined:")

    print(
        f"  Images: "
        f"{combined['images']}"
    )

    print(
        f"  Canonical labels: "
        f"{combined['canonical_labels']}"
    )

    print(
        f"  Framework labels: "
        f"{combined['framework_labels']}"
    )

    print(
        f"  Annotation rows: "
        f"{combined['annotation_rows']}"
    )

    print(
        f"  Vehicle: "
        f"{combined['class_counts']['Vehicle']}"
    )

    print(
        f"  Pedestrian: "
        f"{combined['class_counts']['Pedestrian']}"
    )

    print(
        f"  Cyclist: "
        f"{combined['class_counts']['Cyclist']}"
    )

    print(
        f"  Empty labels: "
        f"{combined['empty_labels']}"
    )

    print(
        f"  Invalid rows: "
        f"{combined['invalid_rows']}"
    )

    print(
        f"  Hash mismatches: "
        f"{combined['hash_mismatches']}"
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
        f"\nReport:\n"
        f"{REPORT_FILE.resolve()}"
    )

    print(
        f"\nIssues:\n"
        f"{ISSUES_FILE.resolve()}"
    )

    if not overall_passed:
        print(
            "\nDo not continue to COCO–YOLO "
            "equivalence validation until every "
            "YOLO issue is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 11 completed successfully."
    )


if __name__ == "__main__":
    main()