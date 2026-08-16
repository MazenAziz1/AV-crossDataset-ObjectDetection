from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import yaml


# ============================================================
# PATHS
# ============================================================

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

PREPROCESSING_CONFIG = Path(
    "configs/datasets/milestone_3/preprocessing.yaml"
)

CLASS_MAPPING_CONFIG = Path(
    "configs/datasets/milestone_3/class_mapping.yaml"
)

YOLO_VALIDATION_REPORT = (
    PROCESSED_ROOT
    / "reports/yolo_validation_report.json"
)

COCO_VALIDATION_REPORT = (
    PROCESSED_ROOT
    / "reports/coco_validation_report.json"
)

REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/coco_yolo_equivalence_report.json"
)

ISSUES_FILE = (
    PROCESSED_ROOT
    / "reports/coco_yolo_equivalence_issues.csv"
)

IMAGE_MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/coco_yolo_equivalence_manifest.csv"
)


# ============================================================
# PARTITIONS
# ============================================================

PARTITIONS = {
    "kitti_train": {
        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_train.json"
        ),

        "yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/kitti/train"
        ),

        "expected": {
            "images": 5985,
            "annotations": 31294,
            "Vehicle": 26278,
            "Pedestrian": 3729,
            "Cyclist": 1287,
            "empty_labels": 0,
        },
    },

    "kitti_val": {
        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_val.json"
        ),

        "yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/kitti/val"
        ),

        "expected": {
            "images": 1496,
            "annotations": 7792,
            "Vehicle": 6472,
            "Pedestrian": 980,
            "Cyclist": 340,
            "empty_labels": 0,
        },
    },

    "waymo_external": {
        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/waymo_external.json"
        ),

        "yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/waymo/external"
        ),

        "expected": {
            "images": 996,
            "annotations": 24819,
            "Vehicle": 16928,
            "Pedestrian": 7127,
            "Cyclist": 764,
            "empty_labels": 12,
        },
    },
}


EXPECTED_COMBINED = {
    "images": 8477,
    "annotations": 63905,
    "matched_boxes": 63905,
    "empty_labels": 12,
    "Vehicle": 49678,
    "Pedestrian": 11836,
    "Cyclist": 2391,
}


IMAGE_MANIFEST_COLUMNS = [
    "partition",
    "global_image_id",
    "image_filename",
    "coco_annotation_count",
    "yolo_row_count",
    "matched_box_count",
    "class_count_mismatch",
    "maximum_normalized_error",
    "maximum_pixel_error",
    "minimum_iou",
    "image_passed",
]


# ============================================================
# GENERAL HELPERS
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

    coco_to_yolo = {
        int(entry["coco_id"]): int(
            entry["yolo_id"]
        )
        for entry in target_classes
    }

    yolo_to_name = {
        int(entry["yolo_id"]): str(
            entry["name"]
        )
        for entry in target_classes
    }

    if coco_to_yolo != {
        1: 0,
        2: 1,
        3: 2,
    }:
        raise ValueError(
            "Unexpected COCO-to-YOLO mapping."
        )

    if yolo_to_name != {
        0: "Vehicle",
        1: "Pedestrian",
        2: "Cyclist",
    }:
        raise ValueError(
            "Unexpected YOLO class mapping."
        )

    return coco_to_yolo, yolo_to_name


# ============================================================
# BOX CONVERSION
# ============================================================

def coco_xywh_to_normalized_yolo(
    bbox: list,
    image_width: int,
    image_height: int,
) -> np.ndarray:
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
            "COCO bbox contains a non-finite value."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "COCO bbox has non-positive dimensions."
        )

    return np.asarray(
        [
            (x + width / 2.0)
            / image_width,

            (y + height / 2.0)
            / image_height,

            width / image_width,
            height / image_height,
        ],
        dtype=np.float64,
    )


def normalized_yolo_to_coco_xywh(
    values: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    center_x, center_y, width, height = [
        float(value)
        for value in values
    ]

    pixel_width = (
        width * image_width
    )

    pixel_height = (
        height * image_height
    )

    pixel_x = (
        center_x * image_width
        - pixel_width / 2.0
    )

    pixel_y = (
        center_y * image_height
        - pixel_height / 2.0
    )

    return np.asarray(
        [
            pixel_x,
            pixel_y,
            pixel_width,
            pixel_height,
        ],
        dtype=np.float64,
    )


def xywh_iou(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_x, first_y, (
        first_width
    ), first_height = [
        float(value)
        for value in first
    ]

    second_x, second_y, (
        second_width
    ), second_height = [
        float(value)
        for value in second
    ]

    first_xmax = (
        first_x + first_width
    )

    first_ymax = (
        first_y + first_height
    )

    second_xmax = (
        second_x + second_width
    )

    second_ymax = (
        second_y + second_height
    )

    intersection_xmin = max(
        first_x,
        second_x,
    )

    intersection_ymin = max(
        first_y,
        second_y,
    )

    intersection_xmax = min(
        first_xmax,
        second_xmax,
    )

    intersection_ymax = min(
        first_ymax,
        second_ymax,
    )

    intersection_width = max(
        0.0,
        intersection_xmax
        - intersection_xmin,
    )

    intersection_height = max(
        0.0,
        intersection_ymax
        - intersection_ymin,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    first_area = (
        first_width
        * first_height
    )

    second_area = (
        second_width
        * second_height
    )

    union_area = (
        first_area
        + second_area
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return float(
        intersection_area
        / union_area
    )


# ============================================================
# YOLO PARSING
# ============================================================

def read_yolo_file(
    path: Path,
    partition: str,
    image_id: int,
    issues: list[dict],
) -> list[dict]:
    if not path.exists():
        add_issue(
            issues,
            partition,
            "missing_yolo_label",
            str(image_id),
            str(path),
        )

        return []

    try:
        text = path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError as error:
        add_issue(
            issues,
            partition,
            "invalid_yolo_encoding",
            str(image_id),
            str(error),
        )

        return []

    records: list[dict] = []

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        tokens = line.split()

        identifier = (
            f"{path.name}:{line_number}"
        )

        if len(tokens) != 5:
            add_issue(
                issues,
                partition,
                "invalid_yolo_field_count",
                identifier,
                str(tokens),
            )

            continue

        try:
            class_id = int(
                tokens[0]
            )

            values = np.asarray(
                [
                    float(value)
                    for value in tokens[1:]
                ],
                dtype=np.float64,
            )

        except ValueError as error:
            add_issue(
                issues,
                partition,
                "invalid_yolo_numeric_value",
                identifier,
                str(error),
            )

            continue

        if class_id not in {
            0,
            1,
            2,
        }:
            add_issue(
                issues,
                partition,
                "invalid_yolo_class_id",
                identifier,
                str(class_id),
            )

            continue

        if not np.isfinite(
            values
        ).all():
            add_issue(
                issues,
                partition,
                "non_finite_yolo_coordinates",
                identifier,
                str(values.tolist()),
            )

            continue

        records.append(
            {
                "class_id": class_id,
                "values": values,
                "line_number": line_number,
            }
        )

    return records


# ============================================================
# CLASS-LEVEL MATCHING
# ============================================================

def match_class_boxes(
    coco_boxes: list[dict],
    yolo_boxes: list[dict],
    image_width: int,
    image_height: int,
    normalized_tolerance: float,
    pixel_tolerance: float,
    minimum_iou: float,
    partition: str,
    image_id: int,
    class_name: str,
    issues: list[dict],
) -> dict:
    if len(coco_boxes) != len(
        yolo_boxes
    ):
        add_issue(
            issues,
            partition,
            "class_box_count_mismatch",
            (
                f"{image_id}:{class_name}"
            ),
            (
                f"COCO={len(coco_boxes)}, "
                f"YOLO={len(yolo_boxes)}"
            ),
        )

        return {
            "matched_boxes": 0,
            "equivalent_boxes": 0,
            "maximum_normalized_error": (
                None
            ),
            "maximum_pixel_error": None,
            "minimum_iou": None,
            "normalized_error_sum": 0.0,
            "normalized_coordinate_count": 0,
            "pixel_error_sum": 0.0,
            "pixel_coordinate_count": 0,
            "passed": False,
        }

    if not coco_boxes:
        return {
            "matched_boxes": 0,
            "equivalent_boxes": 0,
            "maximum_normalized_error": 0.0,
            "maximum_pixel_error": 0.0,
            "minimum_iou": 1.0,
            "normalized_error_sum": 0.0,
            "normalized_coordinate_count": 0,
            "pixel_error_sum": 0.0,
            "pixel_coordinate_count": 0,
            "passed": True,
        }

    coco_normalized = np.stack(
        [
            record["normalized"]
            for record in coco_boxes
        ]
    )

    yolo_normalized = np.stack(
        [
            record["values"]
            for record in yolo_boxes
        ]
    )

    # Each matrix cell is the largest normalized coordinate
    # difference between one COCO box and one YOLO box.
    cost_matrix = np.max(
        np.abs(
            coco_normalized[:, None, :]
            - yolo_normalized[None, :, :]
        ),
        axis=2,
    )

    coco_indices, yolo_indices = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    matched_boxes = 0
    equivalent_boxes = 0

    maximum_normalized_error = 0.0
    maximum_pixel_error = 0.0
    observed_minimum_iou = 1.0

    normalized_error_sum = 0.0
    normalized_coordinate_count = 0

    pixel_error_sum = 0.0
    pixel_coordinate_count = 0

    matching_passed = True

    for coco_index, yolo_index in zip(
        coco_indices,
        yolo_indices,
    ):
        matched_boxes += 1

        coco_record = (
            coco_boxes[coco_index]
        )

        yolo_record = (
            yolo_boxes[yolo_index]
        )

        normalized_errors = np.abs(
            coco_record["normalized"]
            - yolo_record["values"]
        )

        pair_maximum_normalized_error = float(
            normalized_errors.max()
        )

        reconstructed_pixel_box = (
            normalized_yolo_to_coco_xywh(
                values=(
                    yolo_record["values"]
                ),
                image_width=image_width,
                image_height=image_height,
            )
        )

        pixel_errors = np.abs(
            coco_record["pixel_bbox"]
            - reconstructed_pixel_box
        )

        pair_maximum_pixel_error = float(
            pixel_errors.max()
        )

        pair_iou = xywh_iou(
            coco_record["pixel_bbox"],
            reconstructed_pixel_box,
        )

        pair_passed = (
            pair_maximum_normalized_error
            <= normalized_tolerance
            and pair_maximum_pixel_error
            <= pixel_tolerance
            and pair_iou >= minimum_iou
        )

        if pair_passed:
            equivalent_boxes += 1

        else:
            matching_passed = False

            add_issue(
                issues,
                partition,
                "coco_yolo_box_mismatch",
                (
                    f"image={image_id};"
                    f"class={class_name};"
                    f"annotation="
                    f"{coco_record['annotation_id']};"
                    f"yolo_line="
                    f"{yolo_record['line_number']}"
                ),
                (
                    f"normalized_error="
                    f"{pair_maximum_normalized_error}; "
                    f"pixel_error="
                    f"{pair_maximum_pixel_error}; "
                    f"iou={pair_iou}"
                ),
            )

        maximum_normalized_error = max(
            maximum_normalized_error,
            pair_maximum_normalized_error,
        )

        maximum_pixel_error = max(
            maximum_pixel_error,
            pair_maximum_pixel_error,
        )

        observed_minimum_iou = min(
            observed_minimum_iou,
            pair_iou,
        )

        normalized_error_sum += float(
            normalized_errors.sum()
        )

        normalized_coordinate_count += int(
            normalized_errors.size
        )

        pixel_error_sum += float(
            pixel_errors.sum()
        )

        pixel_coordinate_count += int(
            pixel_errors.size
        )

    return {
        "matched_boxes": int(
            matched_boxes
        ),
        "equivalent_boxes": int(
            equivalent_boxes
        ),
        "maximum_normalized_error": float(
            maximum_normalized_error
        ),
        "maximum_pixel_error": float(
            maximum_pixel_error
        ),
        "minimum_iou": float(
            observed_minimum_iou
        ),
        "normalized_error_sum": float(
            normalized_error_sum
        ),
        "normalized_coordinate_count": int(
            normalized_coordinate_count
        ),
        "pixel_error_sum": float(
            pixel_error_sum
        ),
        "pixel_coordinate_count": int(
            pixel_coordinate_count
        ),
        "passed": bool(
            matching_passed
            and matched_boxes
            == len(coco_boxes)
        ),
    }


# ============================================================
# PARTITION VALIDATION
# ============================================================

def validate_partition(
    partition_name: str,
    specification: dict,
    coco_to_yolo: dict[int, int],
    yolo_to_name: dict[int, str],
    normalized_tolerance: float,
    pixel_tolerance: float,
    minimum_iou: float,
    issues: list[dict],
) -> tuple[dict, list[dict]]:
    coco_file = Path(
        specification["coco_file"]
    )

    yolo_dir = Path(
        specification["yolo_dir"]
    )

    expected = specification[
        "expected"
    ]

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

    annotations_by_image: dict[
        int,
        list[dict],
    ] = defaultdict(list)

    for annotation in annotations:
        annotations_by_image[
            int(annotation["image_id"])
        ].append(annotation)

    for image_id in annotations_by_image:
        annotations_by_image[
            image_id
        ].sort(
            key=lambda record: int(
                record["id"]
            )
        )

    coco_class_counts: Counter = Counter()
    yolo_class_counts: Counter = Counter()

    matched_boxes = 0
    equivalent_boxes = 0
    empty_labels = 0
    mismatched_images = 0
    class_count_mismatch_images = 0

    maximum_normalized_error = 0.0
    maximum_pixel_error = 0.0
    observed_minimum_iou = 1.0

    normalized_error_sum = 0.0
    normalized_coordinate_count = 0

    pixel_error_sum = 0.0
    pixel_coordinate_count = 0

    manifest_rows: list[dict] = []

    print(
        f"\nChecking {partition_name} "
        f"COCO–YOLO equivalence..."
    )

    for image in tqdm(
        sorted(
            images,
            key=lambda record: int(
                record["id"]
            ),
        ),
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

        label_path = (
            yolo_dir
            / f"{Path(image_filename).stem}.txt"
        )

        yolo_records = read_yolo_file(
            path=label_path,
            partition=partition_name,
            image_id=image_id,
            issues=issues,
        )

        if not yolo_records:
            empty_labels += 1

        coco_records = (
            annotations_by_image.get(
                image_id,
                [],
            )
        )

        coco_by_class: dict[
            int,
            list[dict],
        ] = defaultdict(list)

        for annotation in coco_records:
            category_id = int(
                annotation["category_id"]
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

            yolo_class_id = int(
                coco_to_yolo[
                    category_id
                ]
            )

            pixel_bbox = np.asarray(
                [
                    float(value)
                    for value in annotation[
                        "bbox"
                    ]
                ],
                dtype=np.float64,
            )

            try:
                normalized = (
                    coco_xywh_to_normalized_yolo(
                        bbox=annotation["bbox"],
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

            except Exception as error:
                add_issue(
                    issues,
                    partition_name,
                    "coco_normalization_failed",
                    str(annotation["id"]),
                    str(error),
                )

                continue

            coco_by_class[
                yolo_class_id
            ].append(
                {
                    "annotation_id": int(
                        annotation["id"]
                    ),
                    "pixel_bbox": (
                        pixel_bbox
                    ),
                    "normalized": (
                        normalized
                    ),
                }
            )

            coco_class_counts[
                yolo_to_name[
                    yolo_class_id
                ]
            ] += 1

        yolo_by_class: dict[
            int,
            list[dict],
        ] = defaultdict(list)

        for record in yolo_records:
            yolo_by_class[
                record["class_id"]
            ].append(record)

            yolo_class_counts[
                yolo_to_name[
                    record["class_id"]
                ]
            ] += 1

        image_class_count_mismatch = False
        image_passed = True

        image_matched_boxes = 0
        image_maximum_normalized_error = 0.0
        image_maximum_pixel_error = 0.0
        image_minimum_iou = 1.0

        for class_id in [
            0,
            1,
            2,
        ]:
            coco_class_boxes = (
                coco_by_class.get(
                    class_id,
                    [],
                )
            )

            yolo_class_boxes = (
                yolo_by_class.get(
                    class_id,
                    [],
                )
            )

            if (
                len(coco_class_boxes)
                != len(yolo_class_boxes)
            ):
                image_class_count_mismatch = (
                    True
                )

            class_result = (
                match_class_boxes(
                    coco_boxes=(
                        coco_class_boxes
                    ),
                    yolo_boxes=(
                        yolo_class_boxes
                    ),
                    image_width=image_width,
                    image_height=image_height,
                    normalized_tolerance=(
                        normalized_tolerance
                    ),
                    pixel_tolerance=(
                        pixel_tolerance
                    ),
                    minimum_iou=(
                        minimum_iou
                    ),
                    partition=(
                        partition_name
                    ),
                    image_id=image_id,
                    class_name=(
                        yolo_to_name[
                            class_id
                        ]
                    ),
                    issues=issues,
                )
            )

            image_passed = (
                image_passed
                and class_result["passed"]
            )

            image_matched_boxes += int(
                class_result[
                    "matched_boxes"
                ]
            )

            matched_boxes += int(
                class_result[
                    "matched_boxes"
                ]
            )

            equivalent_boxes += int(
                class_result[
                    "equivalent_boxes"
                ]
            )

            maximum_normalized_error = max(
                maximum_normalized_error,
                float(
                    class_result[
                        "maximum_normalized_error"
                    ]
                    or 0.0
                ),
            )

            maximum_pixel_error = max(
                maximum_pixel_error,
                float(
                    class_result[
                        "maximum_pixel_error"
                    ]
                    or 0.0
                ),
            )

            image_maximum_normalized_error = max(
                image_maximum_normalized_error,
                float(
                    class_result[
                        "maximum_normalized_error"
                    ]
                    or 0.0
                ),
            )

            image_maximum_pixel_error = max(
                image_maximum_pixel_error,
                float(
                    class_result[
                        "maximum_pixel_error"
                    ]
                    or 0.0
                ),
            )

            class_minimum_iou = (
                class_result[
                    "minimum_iou"
                ]
            )

            if class_minimum_iou is not None:
                observed_minimum_iou = min(
                    observed_minimum_iou,
                    float(
                        class_minimum_iou
                    ),
                )

                image_minimum_iou = min(
                    image_minimum_iou,
                    float(
                        class_minimum_iou
                    ),
                )

            normalized_error_sum += float(
                class_result[
                    "normalized_error_sum"
                ]
            )

            normalized_coordinate_count += int(
                class_result[
                    "normalized_coordinate_count"
                ]
            )

            pixel_error_sum += float(
                class_result[
                    "pixel_error_sum"
                ]
            )

            pixel_coordinate_count += int(
                class_result[
                    "pixel_coordinate_count"
                ]
            )

        if image_class_count_mismatch:
            class_count_mismatch_images += 1
            image_passed = False

        if not image_passed:
            mismatched_images += 1

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
                "coco_annotation_count": (
                    len(coco_records)
                ),
                "yolo_row_count": (
                    len(yolo_records)
                ),
                "matched_box_count": (
                    image_matched_boxes
                ),
                "class_count_mismatch": (
                    image_class_count_mismatch
                ),
                "maximum_normalized_error": (
                    image_maximum_normalized_error
                ),
                "maximum_pixel_error": (
                    image_maximum_pixel_error
                ),
                "minimum_iou": (
                    image_minimum_iou
                ),
                "image_passed": (
                    image_passed
                ),
            }
        )

    mean_normalized_error = (
        normalized_error_sum
        / normalized_coordinate_count
        if normalized_coordinate_count
        else 0.0
    )

    mean_pixel_error = (
        pixel_error_sum
        / pixel_coordinate_count
        if pixel_coordinate_count
        else 0.0
    )

    summary = {
        "images": int(
            len(images)
        ),
        "coco_annotations": int(
            len(annotations)
        ),
        "yolo_rows": int(
            sum(
                len(
                    read_yolo_file(
                        path=(
                            yolo_dir
                            / (
                                f"{Path(str(image['file_name'])).stem}"
                                ".txt"
                            )
                        ),
                        partition=partition_name,
                        image_id=int(
                            image["id"]
                        ),
                        issues=[],
                    )
                )
                for image in images
            )
        ),
        "matched_boxes": int(
            matched_boxes
        ),
        "equivalent_boxes": int(
            equivalent_boxes
        ),
        "empty_labels": int(
            empty_labels
        ),
        "mismatched_images": int(
            mismatched_images
        ),
        "class_count_mismatch_images": int(
            class_count_mismatch_images
        ),
        "coco_class_counts": {
            class_name: int(
                coco_class_counts[
                    class_name
                ]
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },
        "yolo_class_counts": {
            class_name: int(
                yolo_class_counts[
                    class_name
                ]
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },
        "maximum_normalized_error": float(
            maximum_normalized_error
        ),
        "mean_normalized_error": float(
            mean_normalized_error
        ),
        "maximum_pixel_error": float(
            maximum_pixel_error
        ),
        "mean_pixel_error": float(
            mean_pixel_error
        ),
        "minimum_iou": float(
            observed_minimum_iou
        ),
    }

    checks = {
        "image_count": (
            summary["images"]
            == expected["images"]
        ),
        "coco_annotation_count": (
            summary["coco_annotations"]
            == expected["annotations"]
        ),
        "yolo_row_count": (
            summary["yolo_rows"]
            == expected["annotations"]
        ),
        "matched_box_count": (
            summary["matched_boxes"]
            == expected["annotations"]
        ),
        "equivalent_box_count": (
            summary["equivalent_boxes"]
            == expected["annotations"]
        ),
        "empty_label_count": (
            summary["empty_labels"]
            == expected["empty_labels"]
        ),
        "vehicle_count": (
            summary[
                "coco_class_counts"
            ]["Vehicle"]
            == expected["Vehicle"]
            and summary[
                "yolo_class_counts"
            ]["Vehicle"]
            == expected["Vehicle"]
        ),
        "pedestrian_count": (
            summary[
                "coco_class_counts"
            ]["Pedestrian"]
            == expected["Pedestrian"]
            and summary[
                "yolo_class_counts"
            ]["Pedestrian"]
            == expected["Pedestrian"]
        ),
        "cyclist_count": (
            summary[
                "coco_class_counts"
            ]["Cyclist"]
            == expected["Cyclist"]
            and summary[
                "yolo_class_counts"
            ]["Cyclist"]
            == expected["Cyclist"]
        ),
        "no_mismatched_images": (
            summary["mismatched_images"]
            == 0
        ),
        "no_class_count_mismatches": (
            summary[
                "class_count_mismatch_images"
            ]
            == 0
        ),
        "normalized_error_within_tolerance": (
            summary[
                "maximum_normalized_error"
            ]
            <= normalized_tolerance
        ),
        "pixel_error_within_tolerance": (
            summary[
                "maximum_pixel_error"
            ]
            <= pixel_tolerance
        ),
        "minimum_iou_passed": (
            summary["minimum_iou"]
            >= minimum_iou
        ),
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                partition_name,
                "partition_equivalence_failed",
                check_name,
                (
                    "The COCO–YOLO equivalence "
                    "check returned false."
                ),
            )

    summary["checks"] = checks
    summary["validation_passed"] = all(
        checks.values()
    )

    return summary, manifest_rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []

    print("=" * 76)
    print("NUMERICAL COCO–YOLO EQUIVALENCE VALIDATION")
    print("=" * 76)

    yolo_validation = load_json(
        YOLO_VALIDATION_REPORT
    )

    coco_validation = load_json(
        COCO_VALIDATION_REPORT
    )

    if not yolo_validation.get(
        "yolo_validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 11 YOLO validation has not passed."
        )

    if not coco_validation.get(
        "coco_validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 10 COCO validation has not passed."
        )

    preprocessing_config = load_yaml(
        PREPROCESSING_CONFIG
    )

    mapping_config = load_yaml(
        CLASS_MAPPING_CONFIG
    )

    (
        coco_to_yolo,
        yolo_to_name,
    ) = build_class_mappings(
        mapping_config
    )

    decimal_precision = int(
        preprocessing_config[
            "annotation_policy"
        ]["yolo"]["decimal_precision"]
    )

    if decimal_precision != 10:
        raise ValueError(
            "The frozen YOLO decimal precision "
            "must remain 10."
        )

    serialization_unit = (
        10.0 ** (-decimal_precision)
    )

    theoretical_half_unit_error = (
        0.5 * serialization_unit
    )

    # Slightly larger than one complete serialization unit.
    normalized_tolerance = (
        1.1 * serialization_unit
    )

    # Center and size rounding can combine while reconstructing
    # pixel-space corners. One micro-pixel remains far looser
    # than the expected serialization error while still proving
    # numerical equivalence.
    pixel_tolerance = 1e-6

    minimum_iou = 0.99999

    results = {}
    all_manifest_rows: list[dict] = []

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        result, manifest_rows = (
            validate_partition(
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
                normalized_tolerance=(
                    normalized_tolerance
                ),
                pixel_tolerance=(
                    pixel_tolerance
                ),
                minimum_iou=(
                    minimum_iou
                ),
                issues=issues,
            )
        )

        results[
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
        IMAGE_MANIFEST_FILE,
        all_manifest_rows,
        IMAGE_MANIFEST_COLUMNS,
    )

    combined = {
        "images": int(
            sum(
                result["images"]
                for result in results.values()
            )
        ),
        "coco_annotations": int(
            sum(
                result[
                    "coco_annotations"
                ]
                for result in results.values()
            )
        ),
        "yolo_rows": int(
            sum(
                result["yolo_rows"]
                for result in results.values()
            )
        ),
        "matched_boxes": int(
            sum(
                result[
                    "matched_boxes"
                ]
                for result in results.values()
            )
        ),
        "equivalent_boxes": int(
            sum(
                result[
                    "equivalent_boxes"
                ]
                for result in results.values()
            )
        ),
        "empty_labels": int(
            sum(
                result["empty_labels"]
                for result in results.values()
            )
        ),
        "mismatched_images": int(
            sum(
                result[
                    "mismatched_images"
                ]
                for result in results.values()
            )
        ),
        "class_count_mismatch_images": int(
            sum(
                result[
                    "class_count_mismatch_images"
                ]
                for result in results.values()
            )
        ),
        "maximum_normalized_error": float(
            max(
                result[
                    "maximum_normalized_error"
                ]
                for result in results.values()
            )
        ),
        "maximum_pixel_error": float(
            max(
                result[
                    "maximum_pixel_error"
                ]
                for result in results.values()
            )
        ),
        "minimum_iou": float(
            min(
                result["minimum_iou"]
                for result in results.values()
            )
        ),
        "class_counts": {
            class_name: int(
                sum(
                    result[
                        "coco_class_counts"
                    ][class_name]
                    for result in results.values()
                )
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },
    }

    combined_checks = {
        "image_count": (
            combined["images"]
            == EXPECTED_COMBINED["images"]
        ),
        "coco_annotation_count": (
            combined["coco_annotations"]
            == EXPECTED_COMBINED[
                "annotations"
            ]
        ),
        "yolo_row_count": (
            combined["yolo_rows"]
            == EXPECTED_COMBINED[
                "annotations"
            ]
        ),
        "matched_box_count": (
            combined["matched_boxes"]
            == EXPECTED_COMBINED[
                "matched_boxes"
            ]
        ),
        "equivalent_box_count": (
            combined["equivalent_boxes"]
            == EXPECTED_COMBINED[
                "matched_boxes"
            ]
        ),
        "empty_label_count": (
            combined["empty_labels"]
            == EXPECTED_COMBINED[
                "empty_labels"
            ]
        ),
        "vehicle_count": (
            combined["class_counts"]["Vehicle"]
            == EXPECTED_COMBINED["Vehicle"]
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
            combined["class_counts"]["Cyclist"]
            == EXPECTED_COMBINED["Cyclist"]
        ),
        "no_mismatched_images": (
            combined["mismatched_images"]
            == 0
        ),
        "no_class_count_mismatches": (
            combined[
                "class_count_mismatch_images"
            ]
            == 0
        ),
        "normalized_error_within_tolerance": (
            combined[
                "maximum_normalized_error"
            ]
            <= normalized_tolerance
        ),
        "pixel_error_within_tolerance": (
            combined[
                "maximum_pixel_error"
            ]
            <= pixel_tolerance
        ),
        "minimum_iou_passed": (
            combined["minimum_iou"]
            >= minimum_iou
        ),
        "all_partitions_passed": all(
            result["validation_passed"]
            for result in results.values()
        ),
        "manifest_row_count": (
            len(all_manifest_rows)
            == EXPECTED_COMBINED["images"]
        ),
    }

    for check_name, passed in (
        combined_checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "combined",
                "combined_equivalence_failed",
                check_name,
                (
                    "The combined COCO–YOLO "
                    "equivalence check returned false."
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
        "step": 12,
        "purpose": (
            "Prove numerical equivalence between "
            "the canonical COCO boxes and serialized "
            "YOLO labels."
        ),
        "matching_method": (
            "per-image, per-class minimum-cost "
            "bipartite matching"
        ),
        "yolo_decimal_precision": (
            decimal_precision
        ),
        "serialization_unit": (
            serialization_unit
        ),
        "theoretical_half_unit_error": (
            theoretical_half_unit_error
        ),
        "tolerances": {
            "maximum_normalized_error": (
                normalized_tolerance
            ),
            "maximum_pixel_error": (
                pixel_tolerance
            ),
            "minimum_iou": minimum_iou,
        },
        "partitions": results,
        "combined": combined,
        "combined_checks": (
            combined_checks
        ),
        "image_manifest": {
            "path": (
                IMAGE_MANIFEST_FILE.as_posix()
            ),
            "rows": len(
                all_manifest_rows
            ),
        },
        "issue_count": len(issues),
        "coco_yolo_equivalence_passed": (
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
    print("COCO–YOLO EQUIVALENCE SUMMARY")
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
            f"  COCO annotations: "
            f"{result['coco_annotations']}"
        )

        print(
            f"  YOLO rows: "
            f"{result['yolo_rows']}"
        )

        print(
            f"  Matched boxes: "
            f"{result['matched_boxes']}"
        )

        print(
            f"  Equivalent boxes: "
            f"{result['equivalent_boxes']}"
        )

        print(
            f"  Mismatched images: "
            f"{result['mismatched_images']}"
        )

        print(
            f"  Maximum normalized error: "
            f"{result['maximum_normalized_error']:.12g}"
        )

        print(
            f"  Maximum pixel error: "
            f"{result['maximum_pixel_error']:.12g}"
        )

        print(
            f"  Minimum IoU: "
            f"{result['minimum_iou']:.12f}"
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
        f"  COCO annotations: "
        f"{combined['coco_annotations']}"
    )

    print(
        f"  YOLO rows: "
        f"{combined['yolo_rows']}"
    )

    print(
        f"  Matched boxes: "
        f"{combined['matched_boxes']}"
    )

    print(
        f"  Equivalent boxes: "
        f"{combined['equivalent_boxes']}"
    )

    print(
        f"  Mismatched images: "
        f"{combined['mismatched_images']}"
    )

    print(
        f"  Maximum normalized error: "
        f"{combined['maximum_normalized_error']:.12g}"
    )

    print(
        f"  Maximum pixel error: "
        f"{combined['maximum_pixel_error']:.12g}"
    )

    print(
        f"  Minimum IoU: "
        f"{combined['minimum_iou']:.12f}"
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
        f"\nEquivalence manifest:\n"
        f"{IMAGE_MANIFEST_FILE.resolve()}"
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
            "\nDo not continue to visual annotation "
            "validation until all COCO–YOLO "
            "equivalence issues are resolved."
        )

        sys.exit(1)

    print(
        "\nStep 12 completed successfully."
    )


if __name__ == "__main__":
    main()