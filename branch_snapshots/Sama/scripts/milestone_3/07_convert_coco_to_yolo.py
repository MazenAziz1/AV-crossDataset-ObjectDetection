from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import yaml
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PREPROCESSING_CONFIG = Path(
    "configs/datasets/milestone_3/preprocessing.yaml"
)

CLASS_MAPPING_CONFIG = Path(
    "configs/datasets/milestone_3/class_mapping.yaml"
)

COCO_CREATION_REPORT = Path(
    "data/processed/milestone_3/reports/"
    "coco_creation_report.json"
)

REGION_POLICY_REPORT = Path(
    "data/processed/milestone_3/reports/"
    "region_policy_report.json"
)

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

COCO_DIR = (
    PROCESSED_ROOT
    / "annotations/coco"
)

YOLO_ROOT = (
    PROCESSED_ROOT
    / "annotations/yolo"
)

REPORT_DIR = (
    PROCESSED_ROOT
    / "reports"
)

MANIFEST_DIR = (
    PROCESSED_ROOT
    / "manifests"
)

REPORT_FILE = (
    REPORT_DIR
    / "yolo_conversion_report.json"
)

ISSUES_FILE = (
    REPORT_DIR
    / "yolo_conversion_issues.csv"
)

LABEL_MANIFEST_FILE = (
    MANIFEST_DIR
    / "yolo_label_manifest.csv"
)


# ============================================================
# PARTITION DEFINITIONS
# ============================================================

PARTITIONS = {
    "kitti_train": {
        "coco_file": (
            COCO_DIR / "kitti_train.json"
        ),
        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/train"
        ),
        "label_dir": (
            YOLO_ROOT
            / "kitti/train"
        ),
    },
    "kitti_val": {
        "coco_file": (
            COCO_DIR / "kitti_val.json"
        ),
        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/val"
        ),
        "label_dir": (
            YOLO_ROOT
            / "kitti/val"
        ),
    },
    "waymo_external": {
        "coco_file": (
            COCO_DIR / "waymo_external.json"
        ),
        "image_dir": (
            PROCESSED_ROOT
            / "images/waymo/external"
        ),
        "label_dir": (
            YOLO_ROOT
            / "waymo/external"
        ),
    },
}


EXPECTED = {
    "kitti_train": {
        "images": 5985,
        "labels": 5985,
        "annotation_rows": 31294,
        "empty_labels": 0,
        "Vehicle": 26278,
        "Pedestrian": 3729,
        "Cyclist": 1287,
    },
    "kitti_val": {
        "images": 1496,
        "labels": 1496,
        "annotation_rows": 7792,
        "empty_labels": 0,
        "Vehicle": 6472,
        "Pedestrian": 980,
        "Cyclist": 340,
    },
    "waymo_external": {
        "images": 996,
        "labels": 996,
        "annotation_rows": 24819,
        "empty_labels": 12,
        "Vehicle": 16928,
        "Pedestrian": 7127,
        "Cyclist": 764,
    },
    "combined": {
        "images": 8477,
        "labels": 8477,
        "annotation_rows": 63905,
        "empty_labels": 12,
        "Vehicle": 49678,
        "Pedestrian": 11836,
        "Cyclist": 2391,
    },
}


MANIFEST_COLUMNS = [
    "partition",
    "global_image_id",
    "image_filename",
    "image_relative_path",
    "label_filename",
    "label_relative_path",
    "annotation_count",
    "is_empty",
    "label_size_bytes",
    "label_sha256",
]


UNIT_TOLERANCE = 1e-9


# ============================================================
# HELPERS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the canonical Milestone 3 "
            "COCO annotations to YOLO labels."
        )
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Remove existing TXT labels from the "
            "three expected YOLO directories."
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
                Path.cwd().resolve()
            )
            .as_posix()
        )

    except ValueError:
        return path.resolve().as_posix()


def write_text_atomic(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    try:
        temporary_path.write_text(
            content,
            encoding="utf-8",
            newline="\n",
        )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def safely_clean_label_directories() -> int:
    removed = 0

    root = YOLO_ROOT.resolve()

    for specification in (
        PARTITIONS.values()
    ):
        label_dir = (
            specification[
                "label_dir"
            ]
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        resolved = label_dir.resolve()

        if not resolved.is_relative_to(
            root
        ):
            raise RuntimeError(
                f"Unsafe YOLO label path:\n"
                f"{resolved}"
            )

        for path in label_dir.glob(
            "*.txt"
        ):
            if path.is_file():
                path.unlink()
                removed += 1

        for path in label_dir.glob(
            "*.tmp"
        ):
            if path.is_file():
                path.unlink()
                removed += 1

    return removed


def build_class_mappings(
    mapping_config: dict,
) -> tuple[
    dict[int, int],
    dict[int, str],
]:
    target_classes = mapping_config.get(
        "target_classes",
        [],
    )

    coco_to_yolo: dict[int, int] = {}
    yolo_to_name: dict[int, str] = {}

    for entry in target_classes:
        class_name = str(
            entry["name"]
        )

        coco_id = int(
            entry["coco_id"]
        )

        yolo_id = int(
            entry["yolo_id"]
        )

        coco_to_yolo[
            coco_id
        ] = yolo_id

        yolo_to_name[
            yolo_id
        ] = class_name

    expected_coco_to_yolo = {
        1: 0,
        2: 1,
        3: 2,
    }

    expected_yolo_to_name = {
        0: "Vehicle",
        1: "Pedestrian",
        2: "Cyclist",
    }

    if (
        coco_to_yolo
        != expected_coco_to_yolo
    ):
        raise ValueError(
            "COCO-to-YOLO class mapping differs "
            "from the frozen policy."
        )

    if (
        yolo_to_name
        != expected_yolo_to_name
    ):
        raise ValueError(
            "YOLO class ordering differs from "
            "the frozen policy."
        )

    return (
        coco_to_yolo,
        yolo_to_name,
    )


def clamp_unit_with_tolerance(
    value: float,
    coordinate_name: str,
) -> float:
    if not math.isfinite(value):
        raise ValueError(
            f"{coordinate_name} is non-finite."
        )

    if (
        value < -UNIT_TOLERANCE
        or value > 1.0 + UNIT_TOLERANCE
    ):
        raise ValueError(
            f"{coordinate_name}={value} lies "
            "outside the normalized range."
        )

    return min(
        max(float(value), 0.0),
        1.0,
    )


def coco_bbox_to_yolo(
    bbox: list,
    image_width: int,
    image_height: int,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    if len(bbox) != 4:
        raise ValueError(
            "COCO bbox must contain four values."
        )

    x, y, width, height = [
        float(value)
        for value in bbox
    ]

    if not all(
        math.isfinite(value)
        for value in [
            x,
            y,
            width,
            height,
        ]
    ):
        raise ValueError(
            "COCO bbox contains non-finite values."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "COCO bbox has non-positive dimensions."
        )

    if (
        x < -UNIT_TOLERANCE
        or y < -UNIT_TOLERANCE
        or x + width
        > image_width + UNIT_TOLERANCE
        or y + height
        > image_height + UNIT_TOLERANCE
    ):
        raise ValueError(
            "COCO bbox lies outside the image bounds."
        )

    center_x = (
        x + width / 2.0
    ) / image_width

    center_y = (
        y + height / 2.0
    ) / image_height

    normalized_width = (
        width / image_width
    )

    normalized_height = (
        height / image_height
    )

    center_x = (
        clamp_unit_with_tolerance(
            center_x,
            "center_x",
        )
    )

    center_y = (
        clamp_unit_with_tolerance(
            center_y,
            "center_y",
        )
    )

    normalized_width = (
        clamp_unit_with_tolerance(
            normalized_width,
            "width",
        )
    )

    normalized_height = (
        clamp_unit_with_tolerance(
            normalized_height,
            "height",
        )
    )

    if (
        normalized_width <= 0
        or normalized_height <= 0
    ):
        raise ValueError(
            "Normalized bbox has non-positive size."
        )

    return (
        center_x,
        center_y,
        normalized_width,
        normalized_height,
    )


def validate_coco_categories(
    coco_data: dict,
    coco_to_yolo: dict[int, int],
) -> None:
    actual_ids = {
        int(category["id"])
        for category
        in coco_data.get(
            "categories",
            [],
        )
    }

    expected_ids = set(
        coco_to_yolo.keys()
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "COCO categories do not match the "
            "frozen class mapping."
        )


# ============================================================
# CONVERSION
# ============================================================

def convert_partition(
    partition_name: str,
    specification: dict,
    coco_to_yolo: dict[int, int],
    yolo_to_name: dict[int, str],
    decimal_precision: int,
    issues: list[dict],
) -> tuple[dict, list[dict]]:
    coco_file = Path(
        specification["coco_file"]
    )

    image_dir = Path(
        specification["image_dir"]
    )

    label_dir = Path(
        specification["label_dir"]
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coco_data = load_json(
        coco_file
    )

    validate_coco_categories(
        coco_data,
        coco_to_yolo,
    )

    images = coco_data.get(
        "images",
        [],
    )

    annotations = coco_data.get(
        "annotations",
        [],
    )

    image_ids = [
        int(image["id"])
        for image in images
    ]

    annotation_ids = [
        int(annotation["id"])
        for annotation in annotations
    ]

    if len(image_ids) != len(
        set(image_ids)
    ):
        raise ValueError(
            f"{partition_name} contains "
            "duplicate COCO image IDs."
        )

    if len(annotation_ids) != len(
        set(annotation_ids)
    ):
        raise ValueError(
            f"{partition_name} contains "
            "duplicate COCO annotation IDs."
        )

    image_id_set = set(
        image_ids
    )

    annotations_by_image: dict[
        int,
        list[dict],
    ] = defaultdict(list)

    for annotation in annotations:
        image_id = int(
            annotation["image_id"]
        )

        if image_id not in image_id_set:
            add_issue(
                issues,
                partition_name,
                "annotation_for_unknown_image",
                str(annotation["id"]),
                f"image_id={image_id}",
            )
            continue

        annotations_by_image[
            image_id
        ].append(annotation)

    for image_id in (
        annotations_by_image
    ):
        annotations_by_image[
            image_id
        ].sort(
            key=lambda item: int(
                item["id"]
            )
        )

    class_counts: Counter = Counter()
    manifest_rows: list[dict] = []

    empty_label_count = 0
    generated_row_count = 0

    expected_label_paths: set[Path] = set()

    images = sorted(
        images,
        key=lambda item: int(
            item["id"]
        ),
    )

    print(
        f"\nConverting {partition_name}..."
    )

    for image in tqdm(
        images,
        unit="image",
    ):
        image_id = int(
            image["id"]
        )

        image_filename = str(
            image["file_name"]
        )

        image_width = int(
            image["width"]
        )

        image_height = int(
            image["height"]
        )

        if (
            image_width <= 0
            or image_height <= 0
        ):
            add_issue(
                issues,
                partition_name,
                "invalid_image_dimensions",
                str(image_id),
                (
                    f"{image_width}x"
                    f"{image_height}"
                ),
            )
            continue

        image_path = (
            image_dir / image_filename
        )

        if not image_path.exists():
            add_issue(
                issues,
                partition_name,
                "missing_processed_image",
                str(image_id),
                str(image_path),
            )

        label_filename = (
            Path(image_filename).stem
            + ".txt"
        )

        label_path = (
            label_dir
            / label_filename
        )

        if label_path in (
            expected_label_paths
        ):
            add_issue(
                issues,
                partition_name,
                "duplicate_label_filename",
                label_filename,
                (
                    "Two COCO images resolve to "
                    "the same YOLO label."
                ),
            )
            continue

        expected_label_paths.add(
            label_path
        )

        lines: list[str] = []

        image_annotations = (
            annotations_by_image.get(
                image_id,
                [],
            )
        )

        for annotation in (
            image_annotations
        ):
            category_id = int(
                annotation[
                    "category_id"
                ]
            )

            if category_id not in (
                coco_to_yolo
            ):
                add_issue(
                    issues,
                    partition_name,
                    "unknown_coco_category",
                    str(annotation["id"]),
                    str(category_id),
                )
                continue

            yolo_id = int(
                coco_to_yolo[
                    category_id
                ]
            )

            try:
                (
                    center_x,
                    center_y,
                    normalized_width,
                    normalized_height,
                ) = coco_bbox_to_yolo(
                    bbox=annotation["bbox"],
                    image_width=image_width,
                    image_height=image_height,
                )

            except Exception as error:
                add_issue(
                    issues,
                    partition_name,
                    "bbox_conversion_failed",
                    str(annotation["id"]),
                    str(error),
                )
                continue

            line = (
                f"{yolo_id} "
                f"{center_x:.{decimal_precision}f} "
                f"{center_y:.{decimal_precision}f} "
                f"{normalized_width:.{decimal_precision}f} "
                f"{normalized_height:.{decimal_precision}f}"
            )

            lines.append(line)

            class_name = (
                yolo_to_name[
                    yolo_id
                ]
            )

            class_counts[
                class_name
            ] += 1

            generated_row_count += 1

        content = "\n".join(
            lines
        )

        if lines:
            content += "\n"
        else:
            empty_label_count += 1

        write_text_atomic(
            label_path,
            content,
        )

        manifest_rows.append(
            {
                "partition": (
                    partition_name
                ),
                "global_image_id": (
                    image_id
                ),
                "image_filename": (
                    image_filename
                ),
                "image_relative_path": (
                    project_relative_path(
                        image_path
                    )
                ),
                "label_filename": (
                    label_filename
                ),
                "label_relative_path": (
                    project_relative_path(
                        label_path
                    )
                ),
                "annotation_count": (
                    len(lines)
                ),
                "is_empty": (
                    len(lines) == 0
                ),
                "label_size_bytes": (
                    label_path.stat().st_size
                ),
                "label_sha256": (
                    sha256_file(
                        label_path
                    )
                ),
            }
        )

    actual_label_paths = {
        path
        for path in label_dir.glob(
            "*.txt"
        )
        if path.is_file()
    }

    missing_label_paths = (
        expected_label_paths
        - actual_label_paths
    )

    extra_label_paths = (
        actual_label_paths
        - expected_label_paths
    )

    for path in sorted(
        missing_label_paths
    ):
        add_issue(
            issues,
            partition_name,
            "missing_label_file",
            path.name,
            str(path),
        )

    for path in sorted(
        extra_label_paths
    ):
        add_issue(
            issues,
            partition_name,
            "unexpected_label_file",
            path.name,
            str(path),
        )

    summary = {
        "images": int(
            len(images)
        ),
        "coco_annotations": int(
            len(annotations)
        ),
        "label_files": int(
            len(actual_label_paths)
        ),
        "annotation_rows": int(
            generated_row_count
        ),
        "empty_labels": int(
            empty_label_count
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
        "missing_labels": int(
            len(missing_label_paths)
        ),
        "unexpected_labels": int(
            len(extra_label_paths)
        ),
    }

    expected = EXPECTED[
        partition_name
    ]

    checks = {
        "image_count": (
            summary["images"]
            == expected["images"]
        ),
        "label_file_count": (
            summary["label_files"]
            == expected["labels"]
        ),
        "coco_annotation_count": (
            summary[
                "coco_annotations"
            ]
            == expected[
                "annotation_rows"
            ]
        ),
        "yolo_annotation_row_count": (
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
        "no_missing_labels": (
            summary["missing_labels"]
            == 0
        ),
        "no_unexpected_labels": (
            summary[
                "unexpected_labels"
            ]
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
                "partition_check_failed",
                check_name,
                (
                    "The YOLO conversion check "
                    "returned false."
                ),
            )

    summary["checks"] = checks
    summary["validation_passed"] = all(
        checks.values()
    )

    return summary, manifest_rows


# ============================================================
# INDEPENDENT LABEL VALIDATION
# ============================================================

def independently_validate_labels(
    manifest_rows: list[dict],
    yolo_to_name: dict[int, str],
    issues: list[dict],
) -> dict:
    class_counts: Counter = Counter()

    total_rows = 0
    empty_files = 0
    invalid_rows = 0

    print(
        "\nIndependently validating "
        "generated YOLO labels..."
    )

    for row in tqdm(
        manifest_rows,
        unit="label",
    ):
        partition = str(
            row["partition"]
        )

        label_path = Path(
            row["label_relative_path"]
        )

        if not label_path.exists():
            add_issue(
                issues,
                partition,
                "validation_missing_label",
                str(
                    row["global_image_id"]
                ),
                str(label_path),
            )
            continue

        text = label_path.read_text(
            encoding="utf-8"
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            empty_files += 1

        expected_count = int(
            row["annotation_count"]
        )

        if len(lines) != expected_count:
            add_issue(
                issues,
                partition,
                "label_line_count_mismatch",
                str(
                    row["global_image_id"]
                ),
                (
                    f"Expected {expected_count}, "
                    f"found {len(lines)}."
                ),
            )

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            tokens = line.split()

            if len(tokens) != 5:
                invalid_rows += 1

                add_issue(
                    issues,
                    partition,
                    "invalid_yolo_field_count",
                    (
                        f"{row['label_filename']}:"
                        f"{line_number}"
                    ),
                    (
                        f"Expected 5 values, "
                        f"found {len(tokens)}."
                    ),
                )
                continue

            try:
                class_id = int(
                    tokens[0]
                )

                coordinates = [
                    float(value)
                    for value in tokens[1:]
                ]

            except ValueError as error:
                invalid_rows += 1

                add_issue(
                    issues,
                    partition,
                    "invalid_yolo_numeric_value",
                    (
                        f"{row['label_filename']}:"
                        f"{line_number}"
                    ),
                    str(error),
                )
                continue

            if class_id not in (
                yolo_to_name
            ):
                invalid_rows += 1

                add_issue(
                    issues,
                    partition,
                    "invalid_yolo_class_id",
                    (
                        f"{row['label_filename']}:"
                        f"{line_number}"
                    ),
                    str(class_id),
                )
                continue

            if not all(
                math.isfinite(value)
                for value in coordinates
            ):
                invalid_rows += 1

                add_issue(
                    issues,
                    partition,
                    "non_finite_yolo_coordinate",
                    (
                        f"{row['label_filename']}:"
                        f"{line_number}"
                    ),
                    str(coordinates),
                )
                continue

            (
                center_x,
                center_y,
                width,
                height,
            ) = coordinates

            if not (
                0.0 <= center_x <= 1.0
                and 0.0 <= center_y <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            ):
                invalid_rows += 1

                add_issue(
                    issues,
                    partition,
                    "out_of_range_yolo_coordinate",
                    (
                        f"{row['label_filename']}:"
                        f"{line_number}"
                    ),
                    str(coordinates),
                )
                continue

            class_counts[
                yolo_to_name[
                    class_id
                ]
            ] += 1

            total_rows += 1

    return {
        "label_files_checked": int(
            len(manifest_rows)
        ),
        "annotation_rows_checked": int(
            total_rows
        ),
        "empty_files": int(
            empty_files
        ),
        "invalid_rows": int(
            invalid_rows
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
        "validation_passed": (
            invalid_rows == 0
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    arguments = parse_arguments()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []

    print("=" * 76)
    print("CONVERTING CANONICAL COCO ANNOTATIONS TO YOLO")
    print("=" * 76)

    preprocessing_config = load_yaml(
        PREPROCESSING_CONFIG
    )

    mapping_config = load_yaml(
        CLASS_MAPPING_CONFIG
    )

    coco_report = load_json(
        COCO_CREATION_REPORT
    )

    region_report = load_json(
        REGION_POLICY_REPORT
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

    annotation_policy = (
        preprocessing_config[
            "annotation_policy"
        ]
    )

    if (
        annotation_policy[
            "canonical_format"
        ]
        != "COCO"
    ):
        raise ValueError(
            "COCO is not configured as the "
            "canonical annotation format."
        )

    derived_formats = {
        str(value).upper()
        for value in (
            annotation_policy[
                "derived_formats"
            ]
        )
    }

    if "YOLO" not in derived_formats:
        raise ValueError(
            "YOLO is not listed as a derived format."
        )

    decimal_precision = int(
        annotation_policy[
            "yolo"
        ]["decimal_precision"]
    )

    create_empty_labels = bool(
        annotation_policy[
            "yolo"
        ][
            "create_empty_label_file_for_negative_images"
        ]
    )

    if not create_empty_labels:
        raise ValueError(
            "The frozen policy requires empty "
            "YOLO label files for negative images."
        )

    if decimal_precision < 6:
        raise ValueError(
            "YOLO decimal precision is too low."
        )

    (
        coco_to_yolo,
        yolo_to_name,
    ) = build_class_mappings(
        mapping_config
    )

    if arguments.clean:
        removed = (
            safely_clean_label_directories()
        )

        print(
            f"Existing YOLO labels removed: "
            f"{removed}"
        )

    partition_results = {}
    all_manifest_rows: list[dict] = []

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        result, manifest_rows = (
            convert_partition(
                partition_name=(
                    partition_name
                ),
                specification=(
                    PARTITIONS[
                        partition_name
                    ]
                ),
                coco_to_yolo=(
                    coco_to_yolo
                ),
                yolo_to_name=(
                    yolo_to_name
                ),
                decimal_precision=(
                    decimal_precision
                ),
                issues=issues,
            )
        )

        partition_results[
            partition_name
        ] = result

        all_manifest_rows.extend(
            manifest_rows
        )

    all_manifest_rows.sort(
        key=lambda row: (
            row["partition"],
            int(
                row["global_image_id"]
            ),
        )
    )

    write_csv(
        LABEL_MANIFEST_FILE,
        all_manifest_rows,
        MANIFEST_COLUMNS,
    )

    independent_validation = (
        independently_validate_labels(
            manifest_rows=(
                all_manifest_rows
            ),
            yolo_to_name=(
                yolo_to_name
            ),
            issues=issues,
        )
    )

    combined = {
        "images": int(
            sum(
                result["images"]
                for result
                in partition_results.values()
            )
        ),
        "label_files": int(
            sum(
                result["label_files"]
                for result
                in partition_results.values()
            )
        ),
        "annotation_rows": int(
            sum(
                result[
                    "annotation_rows"
                ]
                for result
                in partition_results.values()
            )
        ),
        "empty_labels": int(
            sum(
                result["empty_labels"]
                for result
                in partition_results.values()
            )
        ),
        "class_counts": {
            class_name: int(
                sum(
                    result[
                        "class_counts"
                    ][class_name]
                    for result
                    in partition_results.values()
                )
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },
    }

    expected_combined = EXPECTED[
        "combined"
    ]

    combined_checks = {
        "image_count": (
            combined["images"]
            == expected_combined[
                "images"
            ]
        ),
        "label_file_count": (
            combined[
                "label_files"
            ]
            == expected_combined[
                "labels"
            ]
        ),
        "annotation_row_count": (
            combined[
                "annotation_rows"
            ]
            == expected_combined[
                "annotation_rows"
            ]
        ),
        "empty_label_count": (
            combined[
                "empty_labels"
            ]
            == expected_combined[
                "empty_labels"
            ]
        ),
        "vehicle_count": (
            combined[
                "class_counts"
            ]["Vehicle"]
            == expected_combined[
                "Vehicle"
            ]
        ),
        "pedestrian_count": (
            combined[
                "class_counts"
            ]["Pedestrian"]
            == expected_combined[
                "Pedestrian"
            ]
        ),
        "cyclist_count": (
            combined[
                "class_counts"
            ]["Cyclist"]
            == expected_combined[
                "Cyclist"
            ]
        ),
        "manifest_row_count": (
            len(all_manifest_rows)
            == expected_combined[
                "labels"
            ]
        ),
        "independent_validation": (
            independent_validation[
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
                "combined_check_failed",
                check_name,
                (
                    "The combined YOLO conversion "
                    "check returned false."
                ),
            )

    partition_passed = all(
        result[
            "validation_passed"
        ]
        for result
        in partition_results.values()
    )

    overall_passed = (
        partition_passed
        and all(
            combined_checks.values()
        )
        and len(issues) == 0
    )

    report = {
        "milestone": 3,
        "step": 8,
        "purpose": (
            "Generate YOLO labels directly from "
            "the canonical COCO annotations."
        ),
        "decimal_precision": (
            decimal_precision
        ),
        "class_mapping": {
            str(coco_id): yolo_id
            for coco_id, yolo_id
            in sorted(
                coco_to_yolo.items()
            )
        },
        "partitions": (
            partition_results
        ),
        "combined": combined,
        "combined_checks": (
            combined_checks
        ),
        "independent_validation": (
            independent_validation
        ),
        "label_manifest": {
            "path": (
                LABEL_MANIFEST_FILE.as_posix()
            ),
            "rows": int(
                len(all_manifest_rows)
            ),
            "sha256": (
                sha256_file(
                    LABEL_MANIFEST_FILE
                )
            ),
        },
        "issue_count": len(issues),
        "yolo_conversion_passed": (
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
    print("YOLO CONVERSION SUMMARY")
    print("=" * 76)

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        result = partition_results[
            partition_name
        ]

        print(f"\n{partition_name}:")

        print(
            f"  Images: "
            f"{result['images']}"
        )

        print(
            f"  Label files: "
            f"{result['label_files']}"
        )

        print(
            f"  Annotation rows: "
            f"{result['annotation_rows']}"
        )

        print(
            f"  Vehicle: "
            f"{result['class_counts']['Vehicle']}"
        )

        print(
            f"  Pedestrian: "
            f"{result['class_counts']['Pedestrian']}"
        )

        print(
            f"  Cyclist: "
            f"{result['class_counts']['Cyclist']}"
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
        f"{combined['images']}"
    )

    print(
        f"  Label files: "
        f"{combined['label_files']}"
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
        f"\nIndependent invalid rows: "
        f"{independent_validation['invalid_rows']}"
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

    print(
        f"\nLabel manifest:\n"
        f"{LABEL_MANIFEST_FILE.resolve()}"
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
            "\nDo not continue to dataset "
            "configuration generation until all "
            "YOLO conversion issues are resolved."
        )

        sys.exit(1)

    print(
        "\nStep 8 completed successfully."
    )


if __name__ == "__main__":
    main()