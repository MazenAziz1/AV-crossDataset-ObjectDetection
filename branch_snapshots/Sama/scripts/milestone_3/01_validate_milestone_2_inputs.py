from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import math
import sys
from typing import Any

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PREPROCESSING_CONFIG = Path(
    "configs/datasets/milestone_3/preprocessing.yaml"
)

MILESTONE_3_MAPPING_CONFIG = Path(
    "configs/datasets/milestone_3/class_mapping.yaml"
)

OUTPUT_DIR = Path(
    "data/processed/milestone_3/reports"
)

REPORT_FILE = (
    OUTPUT_DIR / "source_input_validation.json"
)

ISSUES_FILE = (
    OUTPUT_DIR / "source_input_validation_issues.csv"
)


# KITTI inputs
KITTI_IMAGE_DIR = Path(
    "data/kitti/raw/training/image_2"
)

KITTI_LABEL_DIR = Path(
    "data/kitti/raw/training/label_2"
)

KITTI_CALIB_DIR = Path(
    "data/kitti/raw/training/calib"
)

KITTI_TRAIN_IDS_FILE = Path(
    "data/kitti/selection/train.txt"
)

KITTI_VAL_IDS_FILE = Path(
    "data/kitti/selection/val.txt"
)

KITTI_MAPPING_FILE = Path(
    "data/kitti/selection/class_mapping.yaml"
)

KITTI_SUMMARY_FILE = Path(
    "data/kitti/statistics/dataset_summary.json"
)

KITTI_INTEGRITY_REPORT = Path(
    "data/kitti/statistics/dataset_integrity_report.json"
)

KITTI_MAPPING_REPORT = Path(
    "data/kitti/statistics/class_mapping_validation.json"
)


# Waymo inputs
WAYMO_SUBSET_ROOT = Path(
    "data/waymo/representative_subset"
)

WAYMO_IMAGE_DIR = Path(
    "data/waymo/representative_subset/images/front"
)

WAYMO_BOXES_FILE = Path(
    "data/waymo/representative_subset/annotations/boxes.csv"
)

WAYMO_MAPPING_FILE = Path(
    "data/waymo/representative_subset/annotations/class_mapping.yaml"
)

WAYMO_MANIFEST_FILE = Path(
    "data/waymo/representative_subset/metadata/manifest.csv"
)

WAYMO_SUMMARY_FILE = Path(
    "data/waymo/representative_subset/metadata/subset_summary.json"
)

WAYMO_VALIDATION_REPORT = Path(
    "data/waymo/representative_subset/metadata/"
    "subset_validation_report.json"
)


TARGET_CLASSES = {
    "Vehicle",
    "Pedestrian",
    "Cyclist",
}

EXPECTED = {
    "kitti_train_images": 5985,
    "kitti_val_images": 1496,
    "kitti_total_images": 7481,

    "kitti_train_target_boxes": 31294,
    "kitti_val_target_boxes": 7792,
    "kitti_total_target_boxes": 39086,

    "kitti_vehicle_boxes": 32750,
    "kitti_pedestrian_boxes": 4709,
    "kitti_cyclist_boxes": 1627,

    "waymo_images": 996,
    "waymo_segments": 25,
    "waymo_target_boxes": 24819,
    "waymo_vehicle_boxes": 16928,
    "waymo_pedestrian_boxes": 7127,
    "waymo_cyclist_boxes": 764,
    "waymo_negative_images": 12,

    "combined_images": 8477,
    "combined_target_boxes": 63905,
    "combined_vehicle_boxes": 49678,
    "combined_pedestrian_boxes": 11836,
    "combined_cyclist_boxes": 2391,
}


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found:\n{path.resolve()}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required YAML file not found:\n{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(
            f"YAML root must be a mapping:\n{path.resolve()}"
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


def read_id_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"ID file not found:\n{path.resolve()}"
        )

    ids = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    return [
        str(image_id).zfill(6)
        for image_id in ids
    ]


def resolve_column(
    dataframe: pd.DataFrame,
    aliases: list[str],
    description: str,
    required: bool = True,
) -> str | None:
    exact_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for alias in aliases:
        match = exact_lookup.get(
            alias.strip().lower()
        )

        if match is not None:
            return str(match)

    if required:
        raise KeyError(
            f"Could not find the {description} column.\n"
            f"Accepted names: {aliases}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def add_issue(
    issues: list[dict],
    dataset: str,
    category: str,
    identifier: str,
    details: str,
) -> None:
    issues.append(
        {
            "dataset": dataset,
            "category": category,
            "identifier": identifier,
            "details": details,
        }
    )


def inspect_image(
    image_path: Path,
) -> tuple[int, int, str | None]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()

        if width <= 0 or height <= 0:
            return (
                int(width),
                int(height),
                "Image has non-positive dimensions",
            )

        return int(width), int(height), None

    except Exception as error:
        return 0, 0, str(error)


def is_finite_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bool:
    return all(
        math.isfinite(value)
        for value in [
            xmin,
            ymin,
            xmax,
            ymax,
        ]
    )


def validate_expected(
    actual: int,
    expected: int,
    dataset: str,
    label: str,
    issues: list[dict],
) -> bool:
    passed = actual == expected

    if not passed:
        add_issue(
            issues=issues,
            dataset=dataset,
            category="count_mismatch",
            identifier=label,
            details=(
                f"Expected {expected}, found {actual}"
            ),
        )

    return passed


def value_from_possible_keys(
    dictionary: dict,
    keys: list[str],
) -> Any:
    for key in keys:
        if key in dictionary:
            return dictionary[key]

    return None


# ============================================================
# KITTI VALIDATION
# ============================================================

def validate_kitti(
    issues: list[dict],
) -> dict:
    required_paths = [
        KITTI_IMAGE_DIR,
        KITTI_LABEL_DIR,
        KITTI_CALIB_DIR,
        KITTI_TRAIN_IDS_FILE,
        KITTI_VAL_IDS_FILE,
        KITTI_MAPPING_FILE,
        KITTI_SUMMARY_FILE,
        KITTI_INTEGRITY_REPORT,
        KITTI_MAPPING_REPORT,
    ]

    for path in required_paths:
        if not path.exists():
            add_issue(
                issues,
                "KITTI",
                "missing_required_path",
                str(path),
                "Required Milestone 2 input does not exist.",
            )

    if any(
        not path.exists()
        for path in required_paths
    ):
        return {
            "validation_passed": False,
            "reason": "One or more required paths are missing.",
        }

    mapping_config = load_yaml(
        KITTI_MAPPING_FILE
    )

    kitti_mapping = mapping_config.get(
        "kitti_mapping",
        {}
    )

    if not kitti_mapping:
        add_issue(
            issues,
            "KITTI",
            "invalid_mapping",
            str(KITTI_MAPPING_FILE),
            "No kitti_mapping section was found.",
        )

    train_ids = read_id_file(
        KITTI_TRAIN_IDS_FILE
    )

    val_ids = read_id_file(
        KITTI_VAL_IDS_FILE
    )

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    duplicate_train_ids = (
        len(train_ids) - len(train_id_set)
    )

    duplicate_val_ids = (
        len(val_ids) - len(val_id_set)
    )

    overlap = sorted(
        train_id_set & val_id_set
    )

    if duplicate_train_ids:
        add_issue(
            issues,
            "KITTI",
            "duplicate_train_ids",
            "train.txt",
            f"Duplicate IDs: {duplicate_train_ids}",
        )

    if duplicate_val_ids:
        add_issue(
            issues,
            "KITTI",
            "duplicate_validation_ids",
            "val.txt",
            f"Duplicate IDs: {duplicate_val_ids}",
        )

    if overlap:
        add_issue(
            issues,
            "KITTI",
            "partition_overlap",
            "train.txt and val.txt",
            (
                f"{len(overlap)} IDs occur in both splits. "
                f"First examples: {overlap[:10]}"
            ),
        )

    all_ids = sorted(
        train_id_set | val_id_set
    )

    file_issues_before = len(issues)

    image_dimensions: dict[
        str,
        tuple[int, int],
    ] = {}

    print("\nChecking KITTI source files...")

    for image_id in tqdm(
        all_ids,
        unit="image",
    ):
        image_file = (
            KITTI_IMAGE_DIR
            / f"{image_id}.png"
        )

        label_file = (
            KITTI_LABEL_DIR
            / f"{image_id}.txt"
        )

        calibration_file = (
            KITTI_CALIB_DIR
            / f"{image_id}.txt"
        )

        for file_type, path in [
            ("image", image_file),
            ("label", label_file),
            ("calibration", calibration_file),
        ]:
            if not path.exists():
                add_issue(
                    issues,
                    "KITTI",
                    f"missing_{file_type}",
                    image_id,
                    str(path),
                )

        if image_file.exists():
            width, height, error = inspect_image(
                image_file
            )

            image_dimensions[image_id] = (
                width,
                height,
            )

            if error is not None:
                add_issue(
                    issues,
                    "KITTI",
                    "unreadable_image",
                    image_id,
                    error,
                )

        if (
            calibration_file.exists()
            and "P2:" not in calibration_file.read_text(
                encoding="utf-8"
            )
        ):
            add_issue(
                issues,
                "KITTI",
                "missing_p2_calibration",
                image_id,
                str(calibration_file),
            )

    class_counts: Counter = Counter()
    split_target_counts = {
        "train": 0,
        "val": 0,
    }

    ignored_count = 0
    invalid_box_count = 0
    unknown_classes: set[str] = set()

    split_lookup = {
        **{
            image_id: "train"
            for image_id in train_ids
        },
        **{
            image_id: "val"
            for image_id in val_ids
        },
    }

    print("\nChecking KITTI source annotations...")

    for image_id in tqdm(
        all_ids,
        unit="label",
    ):
        label_file = (
            KITTI_LABEL_DIR
            / f"{image_id}.txt"
        )

        if not label_file.exists():
            continue

        lines = label_file.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            if not line.strip():
                continue

            fields = line.split()

            if len(fields) not in {15, 16}:
                add_issue(
                    issues,
                    "KITTI",
                    "invalid_label_field_count",
                    f"{image_id}:{line_number}",
                    (
                        f"Expected 15 or 16 fields, "
                        f"found {len(fields)}"
                    ),
                )
                continue

            source_class = fields[0]

            mapping_entry = kitti_mapping.get(
                source_class
            )

            if mapping_entry is None:
                unknown_classes.add(
                    source_class
                )

                add_issue(
                    issues,
                    "KITTI",
                    "unmapped_source_class",
                    f"{image_id}:{line_number}",
                    source_class,
                )
                continue

            try:
                xmin = float(fields[4])
                ymin = float(fields[5])
                xmax = float(fields[6])
                ymax = float(fields[7])

            except ValueError as error:
                add_issue(
                    issues,
                    "KITTI",
                    "non_numeric_box",
                    f"{image_id}:{line_number}",
                    str(error),
                )
                continue

            if (
                not is_finite_box(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                )
                or xmax <= xmin
                or ymax <= ymin
            ):
                invalid_box_count += 1

                add_issue(
                    issues,
                    "KITTI",
                    "invalid_box",
                    f"{image_id}:{line_number}",
                    (
                        f"xmin={xmin}, ymin={ymin}, "
                        f"xmax={xmax}, ymax={ymax}"
                    ),
                )
                continue

            action = mapping_entry.get(
                "action"
            )

            if action == "map":
                mapped_name = mapping_entry.get(
                    "mapped_class_name"
                )

                if mapped_name not in TARGET_CLASSES:
                    add_issue(
                        issues,
                        "KITTI",
                        "invalid_target_mapping",
                        source_class,
                        str(mapped_name),
                    )
                    continue

                class_counts[mapped_name] += 1

                split_target_counts[
                    split_lookup[image_id]
                ] += 1

            elif action == "ignore":
                ignored_count += 1

            else:
                add_issue(
                    issues,
                    "KITTI",
                    "unknown_mapping_action",
                    source_class,
                    str(action),
                )

    integrity_report = load_json(
        KITTI_INTEGRITY_REPORT
    )

    mapping_report = load_json(
        KITTI_MAPPING_REPORT
    )

    summary = load_json(
        KITTI_SUMMARY_FILE
    )

    integrity_passed = bool(
        integrity_report.get(
            "integrity_passed",
            False,
        )
    )

    mapping_passed = bool(
        mapping_report.get(
            "mapping_validation_passed",
            False,
        )
    )

    statistics_passed = bool(
        summary.get(
            "statistics_validation_passed",
            False,
        )
    )

    if not integrity_passed:
        add_issue(
            issues,
            "KITTI",
            "milestone_2_report_failed",
            str(KITTI_INTEGRITY_REPORT),
            "integrity_passed is not true.",
        )

    if not mapping_passed:
        add_issue(
            issues,
            "KITTI",
            "milestone_2_report_failed",
            str(KITTI_MAPPING_REPORT),
            "mapping_validation_passed is not true.",
        )

    if not statistics_passed:
        add_issue(
            issues,
            "KITTI",
            "milestone_2_report_failed",
            str(KITTI_SUMMARY_FILE),
            "statistics_validation_passed is not true.",
        )

    checks = {
        "train_image_count": validate_expected(
            len(train_ids),
            EXPECTED["kitti_train_images"],
            "KITTI",
            "train_images",
            issues,
        ),
        "validation_image_count": validate_expected(
            len(val_ids),
            EXPECTED["kitti_val_images"],
            "KITTI",
            "validation_images",
            issues,
        ),
        "total_image_count": validate_expected(
            len(all_ids),
            EXPECTED["kitti_total_images"],
            "KITTI",
            "total_images",
            issues,
        ),
        "train_target_box_count": validate_expected(
            split_target_counts["train"],
            EXPECTED[
                "kitti_train_target_boxes"
            ],
            "KITTI",
            "train_target_boxes",
            issues,
        ),
        "validation_target_box_count": validate_expected(
            split_target_counts["val"],
            EXPECTED[
                "kitti_val_target_boxes"
            ],
            "KITTI",
            "validation_target_boxes",
            issues,
        ),
        "total_target_box_count": validate_expected(
            sum(class_counts.values()),
            EXPECTED[
                "kitti_total_target_boxes"
            ],
            "KITTI",
            "total_target_boxes",
            issues,
        ),
        "vehicle_box_count": validate_expected(
            class_counts["Vehicle"],
            EXPECTED["kitti_vehicle_boxes"],
            "KITTI",
            "vehicle_boxes",
            issues,
        ),
        "pedestrian_box_count": validate_expected(
            class_counts["Pedestrian"],
            EXPECTED[
                "kitti_pedestrian_boxes"
            ],
            "KITTI",
            "pedestrian_boxes",
            issues,
        ),
        "cyclist_box_count": validate_expected(
            class_counts["Cyclist"],
            EXPECTED["kitti_cyclist_boxes"],
            "KITTI",
            "cyclist_boxes",
            issues,
        ),
        "no_partition_overlap": len(overlap) == 0,
        "no_duplicate_train_ids": (
            duplicate_train_ids == 0
        ),
        "no_duplicate_validation_ids": (
            duplicate_val_ids == 0
        ),
        "no_unknown_classes": (
            len(unknown_classes) == 0
        ),
        "no_invalid_boxes": (
            invalid_box_count == 0
        ),
        "milestone_2_integrity_passed": (
            integrity_passed
        ),
        "milestone_2_mapping_passed": (
            mapping_passed
        ),
        "milestone_2_statistics_passed": (
            statistics_passed
        ),
    }

    passed = (
        all(checks.values())
        and len(issues) == file_issues_before
    )

    return {
        "validation_passed": passed,
        "train_images": len(train_ids),
        "validation_images": len(val_ids),
        "total_images": len(all_ids),
        "train_target_boxes": (
            split_target_counts["train"]
        ),
        "validation_target_boxes": (
            split_target_counts["val"]
        ),
        "total_target_boxes": int(
            sum(class_counts.values())
        ),
        "ignored_boxes": int(
            ignored_count
        ),
        "class_counts": dict(
            sorted(class_counts.items())
        ),
        "duplicate_train_ids": (
            duplicate_train_ids
        ),
        "duplicate_validation_ids": (
            duplicate_val_ids
        ),
        "train_validation_overlap": (
            len(overlap)
        ),
        "invalid_boxes": (
            invalid_box_count
        ),
        "unknown_classes": sorted(
            unknown_classes
        ),
        "checks": checks,
    }


# ============================================================
# WAYMO VALIDATION
# ============================================================
def resolve_waymo_image_path(
    row: pd.Series,
    image_id: str,
    image_path_column: str | None,
) -> Path | None:
    """
    Resolve a Waymo image using the exact paths stored in the
    Milestone 2 manifest.

    Expected structure:
    data/waymo/representative_subset/
        images/front/<segment_id>/<timestamp>.jpg
    """
    candidates: list[Path] = []

    # --------------------------------------------------------
    # 1. Use the manifest relative path.
    # Example:
    # images/front/<segment_id>/<timestamp>.jpg
    # --------------------------------------------------------
    if image_path_column is not None:
        raw_value = row.get(
            image_path_column
        )

        if pd.notna(raw_value):
            path_text = (
                str(raw_value)
                .strip()
                .replace("\\", "/")
            )

            relative_path = Path(
                path_text
            )

            if relative_path.is_absolute():
                candidates.append(
                    relative_path
                )
            else:
                candidates.append(
                    WAYMO_SUBSET_ROOT
                    / relative_path
                )

                candidates.append(
                    Path.cwd()
                    / relative_path
                )

    # --------------------------------------------------------
    # 2. Construct the path from the manifest segment and
    # image filename fields.
    # --------------------------------------------------------
    segment_id = str(
        row.get(
            "segment_id",
            "",
        )
    ).strip()

    image_filename = str(
        row.get(
            "image_filename",
            "",
        )
    ).strip()

    frame_timestamp = str(
        row.get(
            "frame_timestamp_micros",
            "",
        )
    ).strip()

    if (
        segment_id
        and segment_id.lower() != "nan"
        and image_filename
        and image_filename.lower() != "nan"
    ):
        candidates.append(
            WAYMO_IMAGE_DIR
            / segment_id
            / image_filename
        )

    # --------------------------------------------------------
    # 3. Timestamp-based fallback.
    # --------------------------------------------------------
    if (
        segment_id
        and segment_id.lower() != "nan"
        and frame_timestamp
        and frame_timestamp.lower() != "nan"
    ):
        for extension in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            candidates.append(
                WAYMO_IMAGE_DIR
                / segment_id
                / f"{frame_timestamp}{extension}"
            )

    # --------------------------------------------------------
    # 4. Flat filename fallback, included for compatibility
    # with any future manifest version.
    # --------------------------------------------------------
    if (
        image_filename
        and image_filename.lower() != "nan"
    ):
        candidates.append(
            WAYMO_IMAGE_DIR
            / image_filename
        )

    for extension in [
        ".jpg",
        ".jpeg",
        ".png",
    ]:
        candidates.append(
            WAYMO_IMAGE_DIR
            / f"{image_id}{extension}"
        )

    # Remove duplicate candidates while preserving order.
    unique_candidates: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        candidate_key = str(
            candidate.resolve(
                strict=False
            )
        ).lower()

        if candidate_key not in seen:
            seen.add(candidate_key)
            unique_candidates.append(
                candidate
            )

    for candidate in unique_candidates:
        if (
            candidate.exists()
            and candidate.is_file()
        ):
            return candidate

    return None


def validate_waymo(
    issues: list[dict],
) -> dict:
    required_paths = [
        WAYMO_IMAGE_DIR,
        WAYMO_BOXES_FILE,
        WAYMO_MAPPING_FILE,
        WAYMO_MANIFEST_FILE,
        WAYMO_SUMMARY_FILE,
        WAYMO_VALIDATION_REPORT,
    ]

    for path in required_paths:
        if not path.exists():
            add_issue(
                issues,
                "Waymo",
                "missing_required_path",
                str(path),
                "Required Milestone 2 input does not exist.",
            )

    if any(
        not path.exists()
        for path in required_paths
    ):
        return {
            "validation_passed": False,
            "reason": "One or more required paths are missing.",
        }

    load_yaml(
        WAYMO_MAPPING_FILE
    )

    manifest = pd.read_csv(
        WAYMO_MANIFEST_FILE,
        dtype=str,
    )

    boxes = pd.read_csv(
        WAYMO_BOXES_FILE,
        dtype=str,
    )

    manifest_image_id_column = resolve_column(
        manifest,
        [
            "image_id",
            "global_image_id",
            "filename",
            "file_name",
        ],
        "Waymo manifest image ID",
    )

    manifest_segment_column = resolve_column(
        manifest,
        [
            "segment_id",
            "segment_context_name",
            "key.segment_context_name",
        ],
        "Waymo segment ID",
    )

    image_path_column = resolve_column(
        manifest,
        [
            "image_path",
            "relative_image_path",
            "file_path",
            "path",
            "filename",
            "file_name",
        ],
        "Waymo image path",
        required=False,
    )

    boxes_image_id_column = resolve_column(
        boxes,
        [
            "image_id",
            "global_image_id",
            "filename",
            "file_name",
        ],
        "Waymo boxes image ID",
    )

    boxes_class_column = resolve_column(
        boxes,
        [
            "class_name",
            "mapped_class_name",
            "category_name",
            "label",
        ],
        "Waymo class name",
    )

    xmin_column = resolve_column(
        boxes,
        ["xmin", "x_min"],
        "Waymo xmin",
    )

    ymin_column = resolve_column(
        boxes,
        ["ymin", "y_min"],
        "Waymo ymin",
    )

    xmax_column = resolve_column(
        boxes,
        ["xmax", "x_max"],
        "Waymo xmax",
    )

    ymax_column = resolve_column(
        boxes,
        ["ymax", "y_max"],
        "Waymo ymax",
    )

    manifest[
        manifest_image_id_column
    ] = (
        manifest[
            manifest_image_id_column
        ]
        .astype(str)
        .str.strip()
    )

    boxes[
        boxes_image_id_column
    ] = (
        boxes[
            boxes_image_id_column
        ]
        .astype(str)
        .str.strip()
    )

    duplicate_manifest_ids = int(
        manifest[
            manifest_image_id_column
        ].duplicated().sum()
    )

    if duplicate_manifest_ids:
        add_issue(
            issues,
            "Waymo",
            "duplicate_manifest_image_ids",
            str(WAYMO_MANIFEST_FILE),
            str(duplicate_manifest_ids),
        )

    manifest_ids = set(
        manifest[
            manifest_image_id_column
        ].tolist()
    )

    annotation_ids = set(
        boxes[
            boxes_image_id_column
        ].tolist()
    )

    unknown_annotation_ids = sorted(
        annotation_ids - manifest_ids
    )

    if unknown_annotation_ids:
        add_issue(
            issues,
            "Waymo",
            "annotations_for_unknown_images",
            str(WAYMO_BOXES_FILE),
            (
                f"{len(unknown_annotation_ids)} unknown IDs. "
                f"First examples: {unknown_annotation_ids[:10]}"
            ),
        )

    print("\nChecking Waymo source images...")

    missing_images = 0
    unreadable_images = 0

    resolved_paths: dict[str, str] = {}

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        unit="image",
    ):
        image_id = str(
            row[manifest_image_id_column]
        )

        image_path = resolve_waymo_image_path(
            row=row,
            image_id=image_id,
            image_path_column=image_path_column,
        )

        if image_path is None:
            missing_images += 1

            add_issue(
                issues,
                "Waymo",
                "missing_image",
                image_id,
                "Could not resolve source image path.",
            )
            continue

        resolved_paths[image_id] = str(
            image_path
        )

        _, _, error = inspect_image(
            image_path
        )

        if error is not None:
            unreadable_images += 1

            add_issue(
                issues,
                "Waymo",
                "unreadable_image",
                image_id,
                error,
            )

    class_counts: Counter = Counter()
    image_box_counts: Counter = Counter()

    invalid_boxes = 0
    unknown_classes: set[str] = set()

    print("\nChecking Waymo source annotations...")

    for row_number, row in tqdm(
        boxes.iterrows(),
        total=len(boxes),
        unit="box",
    ):
        image_id = str(
            row[boxes_image_id_column]
        )

        class_name = str(
            row[boxes_class_column]
        ).strip()

        if class_name not in TARGET_CLASSES:
            unknown_classes.add(
                class_name
            )

            add_issue(
                issues,
                "Waymo",
                "unexpected_target_class",
                f"row_{row_number}",
                class_name,
            )
            continue

        try:
            xmin = float(row[xmin_column])
            ymin = float(row[ymin_column])
            xmax = float(row[xmax_column])
            ymax = float(row[ymax_column])

        except (TypeError, ValueError) as error:
            invalid_boxes += 1

            add_issue(
                issues,
                "Waymo",
                "non_numeric_box",
                f"row_{row_number}",
                str(error),
            )
            continue

        if (
            not is_finite_box(
                xmin,
                ymin,
                xmax,
                ymax,
            )
            or xmax <= xmin
            or ymax <= ymin
        ):
            invalid_boxes += 1

            add_issue(
                issues,
                "Waymo",
                "invalid_box",
                f"row_{row_number}",
                (
                    f"xmin={xmin}, ymin={ymin}, "
                    f"xmax={xmax}, ymax={ymax}"
                ),
            )
            continue

        class_counts[class_name] += 1
        image_box_counts[image_id] += 1

    negative_images = sum(
        image_box_counts.get(
            image_id,
            0,
        )
        == 0
        for image_id in manifest_ids
    )

    segment_count = int(
        manifest[
            manifest_segment_column
        ].nunique()
    )

    summary = load_json(
        WAYMO_SUMMARY_FILE
    )

    validation_report = load_json(
        WAYMO_VALIDATION_REPORT
    )

    validation_status = value_from_possible_keys(
        validation_report,
        [
            "validation_passed",
            "subset_validation_passed",
            "passed",
        ],
    )

    if validation_status is None:
        # Older report version may store text.
        status_text = str(
            validation_report.get(
                "validation_status",
                "",
            )
        ).strip().upper()

        validation_status = (
            status_text == "PASSED"
        )

    validation_status = bool(
        validation_status
    )

    if not validation_status:
        add_issue(
            issues,
            "Waymo",
            "milestone_2_report_failed",
            str(WAYMO_VALIDATION_REPORT),
            "Waymo subset validation is not recorded as passed.",
        )

    checks = {
        "image_count": validate_expected(
            len(manifest),
            EXPECTED["waymo_images"],
            "Waymo",
            "images",
            issues,
        ),
        "segment_count": validate_expected(
            segment_count,
            EXPECTED["waymo_segments"],
            "Waymo",
            "segments",
            issues,
        ),
        "target_box_count": validate_expected(
            len(boxes),
            EXPECTED["waymo_target_boxes"],
            "Waymo",
            "target_boxes",
            issues,
        ),
        "vehicle_box_count": validate_expected(
            class_counts["Vehicle"],
            EXPECTED["waymo_vehicle_boxes"],
            "Waymo",
            "vehicle_boxes",
            issues,
        ),
        "pedestrian_box_count": validate_expected(
            class_counts["Pedestrian"],
            EXPECTED[
                "waymo_pedestrian_boxes"
            ],
            "Waymo",
            "pedestrian_boxes",
            issues,
        ),
        "cyclist_box_count": validate_expected(
            class_counts["Cyclist"],
            EXPECTED["waymo_cyclist_boxes"],
            "Waymo",
            "cyclist_boxes",
            issues,
        ),
        "negative_image_count": validate_expected(
            negative_images,
            EXPECTED[
                "waymo_negative_images"
            ],
            "Waymo",
            "negative_images",
            issues,
        ),
        "no_duplicate_manifest_ids": (
            duplicate_manifest_ids == 0
        ),
        "all_annotation_ids_known": (
            len(unknown_annotation_ids) == 0
        ),
        "no_missing_images": (
            missing_images == 0
        ),
        "no_unreadable_images": (
            unreadable_images == 0
        ),
        "no_invalid_boxes": (
            invalid_boxes == 0
        ),
        "no_unknown_target_classes": (
            len(unknown_classes) == 0
        ),
        "milestone_2_validation_passed": (
            validation_status
        ),
    }

    passed = all(
        checks.values()
    )

    return {
        "validation_passed": passed,
        "images": int(len(manifest)),
        "segments": segment_count,
        "target_boxes": int(len(boxes)),
        "negative_images": int(
            negative_images
        ),
        "class_counts": dict(
            sorted(class_counts.items())
        ),
        "duplicate_manifest_image_ids": (
            duplicate_manifest_ids
        ),
        "unknown_annotation_image_ids": (
            len(unknown_annotation_ids)
        ),
        "missing_images": (
            missing_images
        ),
        "unreadable_images": (
            unreadable_images
        ),
        "invalid_boxes": (
            invalid_boxes
        ),
        "unknown_target_classes": sorted(
            unknown_classes
        ),
        "checks": checks,
    }


# ============================================================
# CONFIGURATION CONSISTENCY
# ============================================================

def validate_configuration_consistency(
    preprocessing_config: dict,
    mapping_config: dict,
    kitti_result: dict,
    waymo_result: dict,
    issues: list[dict],
) -> dict:
    expected_totals = (
        preprocessing_config.get(
            "expected_totals",
            {}
        )
    )

    target_classes = (
        mapping_config.get(
            "target_classes",
            []
        )
    )

    internal_mapping = {
        str(entry.get("name")): int(
            entry.get("internal_id")
        )
        for entry in target_classes
    }

    yolo_mapping = {
        str(entry.get("name")): int(
            entry.get("yolo_id")
        )
        for entry in target_classes
    }

    coco_mapping = {
        str(entry.get("name")): int(
            entry.get("coco_id")
        )
        for entry in target_classes
    }

    computed_totals = {
        "images": (
            int(kitti_result.get(
                "total_images",
                0,
            ))
            + int(waymo_result.get(
                "images",
                0,
            ))
        ),
        "target_boxes": (
            int(kitti_result.get(
                "total_target_boxes",
                0,
            ))
            + int(waymo_result.get(
                "target_boxes",
                0,
            ))
        ),
        "vehicle_boxes": (
            int(
                kitti_result.get(
                    "class_counts",
                    {},
                ).get(
                    "Vehicle",
                    0,
                )
            )
            + int(
                waymo_result.get(
                    "class_counts",
                    {},
                ).get(
                    "Vehicle",
                    0,
                )
            )
        ),
        "pedestrian_boxes": (
            int(
                kitti_result.get(
                    "class_counts",
                    {},
                ).get(
                    "Pedestrian",
                    0,
                )
            )
            + int(
                waymo_result.get(
                    "class_counts",
                    {},
                ).get(
                    "Pedestrian",
                    0,
                )
            )
        ),
        "cyclist_boxes": (
            int(
                kitti_result.get(
                    "class_counts",
                    {},
                ).get(
                    "Cyclist",
                    0,
                )
            )
            + int(
                waymo_result.get(
                    "class_counts",
                    {},
                ).get(
                    "Cyclist",
                    0,
                )
            )
        ),
    }

    checks = {}

    for key, actual in computed_totals.items():
        configured = expected_totals.get(
            key
        )

        expected_value = EXPECTED[
            f"combined_{key}"
        ]

        checks[
            f"computed_{key}_matches_expected"
        ] = validate_expected(
            actual,
            expected_value,
            "Combined",
            key,
            issues,
        )

        if configured != expected_value:
            add_issue(
                issues,
                "Configuration",
                "configured_total_mismatch",
                key,
                (
                    f"preprocessing.yaml contains "
                    f"{configured}; expected "
                    f"{expected_value}"
                ),
            )

            checks[
                f"configured_{key}_matches_expected"
            ] = False
        else:
            checks[
                f"configured_{key}_matches_expected"
            ] = True

    expected_internal = {
        "Vehicle": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
    }

    expected_yolo = {
        "Vehicle": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
    }

    expected_coco = {
        "Vehicle": 1,
        "Pedestrian": 2,
        "Cyclist": 3,
    }

    checks["internal_mapping_correct"] = (
        internal_mapping
        == expected_internal
    )

    checks["yolo_mapping_correct"] = (
        yolo_mapping
        == expected_yolo
    )

    checks["coco_mapping_correct"] = (
        coco_mapping
        == expected_coco
    )

    if not checks[
        "internal_mapping_correct"
    ]:
        add_issue(
            issues,
            "Configuration",
            "invalid_internal_mapping",
            str(MILESTONE_3_MAPPING_CONFIG),
            str(internal_mapping),
        )

    if not checks[
        "yolo_mapping_correct"
    ]:
        add_issue(
            issues,
            "Configuration",
            "invalid_yolo_mapping",
            str(MILESTONE_3_MAPPING_CONFIG),
            str(yolo_mapping),
        )

    if not checks[
        "coco_mapping_correct"
    ]:
        add_issue(
            issues,
            "Configuration",
            "invalid_coco_mapping",
            str(MILESTONE_3_MAPPING_CONFIG),
            str(coco_mapping),
        )

    return {
        "validation_passed": all(
            checks.values()
        ),
        "computed_totals": (
            computed_totals
        ),
        "configured_totals": (
            expected_totals
        ),
        "internal_class_mapping": (
            internal_mapping
        ),
        "yolo_class_mapping": (
            yolo_mapping
        ),
        "coco_class_mapping": (
            coco_mapping
        ),
        "checks": checks,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []

    print("=" * 76)
    print("MILESTONE 3 SOURCE INPUT VALIDATION")
    print("=" * 76)

    preprocessing_config = load_yaml(
        PREPROCESSING_CONFIG
    )

    mapping_config = load_yaml(
        MILESTONE_3_MAPPING_CONFIG
    )

    kitti_result = validate_kitti(
        issues
    )

    waymo_result = validate_waymo(
        issues
    )

    configuration_result = (
        validate_configuration_consistency(
            preprocessing_config=(
                preprocessing_config
            ),
            mapping_config=(
                mapping_config
            ),
            kitti_result=kitti_result,
            waymo_result=waymo_result,
            issues=issues,
        )
    )

    overall_passed = (
        bool(
            kitti_result.get(
                "validation_passed",
                False,
            )
        )
        and bool(
            waymo_result.get(
                "validation_passed",
                False,
            )
        )
        and bool(
            configuration_result.get(
                "validation_passed",
                False,
            )
        )
        and len(issues) == 0
    )

    report = {
        "milestone": 3,
        "step": 2,
        "purpose": (
            "Validate all frozen Milestone 2 inputs "
            "before preprocessing or conversion."
        ),
        "kitti": kitti_result,
        "waymo": waymo_result,
        "configuration_consistency": (
            configuration_result
        ),
        "issue_count": len(issues),
        "validation_passed": (
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
            "dataset",
            "category",
            "identifier",
            "details",
        ],
    )

    print("\n" + "=" * 76)
    print("SOURCE INPUT VALIDATION SUMMARY")
    print("=" * 76)

    print("\nKITTI:")

    print(
        f"  Train images: "
        f"{kitti_result.get('train_images', 0)}"
    )

    print(
        f"  Validation images: "
        f"{kitti_result.get('validation_images', 0)}"
    )

    print(
        f"  Target boxes: "
        f"{kitti_result.get('total_target_boxes', 0)}"
    )

    print(
        f"  Status: "
        f"{'PASSED' if kitti_result.get('validation_passed') else 'FAILED'}"
    )

    print("\nWaymo:")

    print(
        f"  Images: "
        f"{waymo_result.get('images', 0)}"
    )

    print(
        f"  Segments: "
        f"{waymo_result.get('segments', 0)}"
    )

    print(
        f"  Target boxes: "
        f"{waymo_result.get('target_boxes', 0)}"
    )

    print(
        f"  Negative images: "
        f"{waymo_result.get('negative_images', 0)}"
    )

    print(
        f"  Status: "
        f"{'PASSED' if waymo_result.get('validation_passed') else 'FAILED'}"
    )

    print("\nCombined:")

    computed_totals = (
        configuration_result.get(
            "computed_totals",
            {},
        )
    )

    print(
        f"  Images: "
        f"{computed_totals.get('images', 0)}"
    )

    print(
        f"  Target boxes: "
        f"{computed_totals.get('target_boxes', 0)}"
    )

    print(
        f"  Vehicle boxes: "
        f"{computed_totals.get('vehicle_boxes', 0)}"
    )

    print(
        f"  Pedestrian boxes: "
        f"{computed_totals.get('pedestrian_boxes', 0)}"
    )

    print(
        f"  Cyclist boxes: "
        f"{computed_totals.get('cyclist_boxes', 0)}"
    )

    print(
        f"\nIssues found: {len(issues)}"
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
            "\nDo not continue to image preprocessing "
            "until every source-input issue is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 2 completed successfully. "
        "The frozen Milestone 2 inputs are safe "
        "to use for Milestone 3."
    )


if __name__ == "__main__":
    main()