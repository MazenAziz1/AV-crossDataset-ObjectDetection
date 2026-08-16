from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing_core import (
    LetterboxTransform,
    transform_xyxy,
)


# ============================================================
# PATHS
# ============================================================

CLASS_MAPPING_CONFIG = Path(
    "configs/datasets/milestone_3/class_mapping.yaml"
)

PREPROCESSING_CONFIG = Path(
    "configs/datasets/milestone_3/preprocessing.yaml"
)

SOURCE_MANIFEST_FILE = Path(
    "data/processed/milestone_3/manifests/source_manifest.csv"
)

TRANSFORM_MANIFEST_FILE = Path(
    "data/processed/milestone_3/manifests/transform_manifest.csv"
)

SOURCE_MANIFEST_SUMMARY = Path(
    "data/processed/milestone_3/reports/"
    "source_manifest_summary.json"
)

IMAGE_PREPROCESSING_REPORT = Path(
    "data/processed/milestone_3/reports/"
    "image_preprocessing_report.json"
)

KITTI_MAPPING_FILE = Path(
    "data/kitti/selection/class_mapping.yaml"
)

WAYMO_BOXES_FILE = Path(
    "data/waymo/representative_subset/"
    "annotations/boxes.csv"
)

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

OUTPUT_DIR = (
    PROCESSED_ROOT
    / "annotations/coco"
)

KITTI_TRAIN_OUTPUT = (
    OUTPUT_DIR / "kitti_train.json"
)

KITTI_VAL_OUTPUT = (
    OUTPUT_DIR / "kitti_val.json"
)

WAYMO_EXTERNAL_OUTPUT = (
    OUTPUT_DIR / "waymo_external.json"
)

REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/coco_creation_report.json"
)

ISSUES_FILE = (
    PROCESSED_ROOT
    / "reports/coco_creation_issues.csv"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED = {
    "kitti_train": {
        "images": 5985,
        "annotations": 31294,
        "negative_images": 0,
        "Vehicle": 26278,
        "Pedestrian": 3729,
        "Cyclist": 1287,
    },
    "kitti_val": {
        "images": 1496,
        "annotations": 7792,
        "negative_images": 0,
        "Vehicle": 6472,
        "Pedestrian": 980,
        "Cyclist": 340,
    },
    "waymo_external": {
        "images": 996,
        "annotations": 24819,
        "negative_images": 12,
        "Vehicle": 16928,
        "Pedestrian": 7127,
        "Cyclist": 764,
    },
    "combined": {
        "images": 8477,
        "annotations": 63905,
        "negative_images": 12,
        "Vehicle": 49678,
        "Pedestrian": 11836,
        "Cyclist": 2391,
    },
}


PARTITION_OUTPUTS = {
    "kitti_train": KITTI_TRAIN_OUTPUT,
    "kitti_val": KITTI_VAL_OUTPUT,
    "waymo_external": WAYMO_EXTERNAL_OUTPUT,
}


TARGET_CLASS_NAMES = {
    "Vehicle",
    "Pedestrian",
    "Cyclist",
}


COORDINATE_PRECISION = 10


# ============================================================
# GENERAL HELPERS
# ============================================================

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
    output_file: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
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


def resolve_column(
    dataframe: pd.DataFrame,
    aliases: list[str],
    description: str,
) -> str:
    lookup = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for alias in aliases:
        match = lookup.get(
            alias.strip().lower()
        )

        if match is not None:
            return match

    raise KeyError(
        f"Could not resolve {description} column.\n"
        f"Accepted aliases: {aliases}\n"
        f"Available columns: "
        f"{list(dataframe.columns)}"
    )


def partition_key(
    dataset: str,
    partition: str,
) -> str:
    if dataset == "KITTI":
        if partition == "train":
            return "kitti_train"

        if partition == "val":
            return "kitti_val"

    if (
        dataset == "Waymo"
        and partition == "external"
    ):
        return "waymo_external"

    raise ValueError(
        f"Unsupported dataset/partition: "
        f"{dataset}/{partition}"
    )


def rounded_float(
    value: float,
) -> float:
    return round(
        float(value),
        COORDINATE_PRECISION,
    )


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


def create_transform(
    row: pd.Series,
) -> LetterboxTransform:
    return LetterboxTransform(
        source_width=int(
            row["source_width"]
        ),
        source_height=int(
            row["source_height"]
        ),
        target_width=int(
            row["target_width"]
        ),
        target_height=int(
            row["target_height"]
        ),
        nominal_scale=float(
            row["nominal_scale"]
        ),
        actual_scale_x=float(
            row["actual_scale_x"]
        ),
        actual_scale_y=float(
            row["actual_scale_y"]
        ),
        resized_width=int(
            row["resized_width"]
        ),
        resized_height=int(
            row["resized_height"]
        ),
        padding_left=int(
            row["padding_left"]
        ),
        padding_top=int(
            row["padding_top"]
        ),
        padding_right=int(
            row["padding_right"]
        ),
        padding_bottom=int(
            row["padding_bottom"]
        ),
    )


def clip_transformed_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    target_width: int,
    target_height: int,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    clipped_xmin = min(
        max(float(xmin), 0.0),
        float(target_width),
    )

    clipped_ymin = min(
        max(float(ymin), 0.0),
        float(target_height),
    )

    clipped_xmax = min(
        max(float(xmax), 0.0),
        float(target_width),
    )

    clipped_ymax = min(
        max(float(ymax), 0.0),
        float(target_height),
    )

    return (
        clipped_xmin,
        clipped_ymin,
        clipped_xmax,
        clipped_ymax,
    )


def convert_source_box(
    source_box: tuple[
        float,
        float,
        float,
        float,
    ],
    transform: LetterboxTransform,
) -> dict:
    source_xmin, (
        source_ymin
    ), source_xmax, source_ymax = (
        source_box
    )

    if not is_finite_box(
        source_xmin,
        source_ymin,
        source_xmax,
        source_ymax,
    ):
        raise ValueError(
            "Source box contains a non-finite coordinate."
        )

    if (
        source_xmax <= source_xmin
        or source_ymax <= source_ymin
    ):
        raise ValueError(
            "Source box has non-positive dimensions."
        )

    (
        transformed_xmin,
        transformed_ymin,
        transformed_xmax,
        transformed_ymax,
    ) = transform_xyxy(
        xmin=source_xmin,
        ymin=source_ymin,
        xmax=source_xmax,
        ymax=source_ymax,
        transform=transform,
    )

    (
        clipped_xmin,
        clipped_ymin,
        clipped_xmax,
        clipped_ymax,
    ) = clip_transformed_box(
        xmin=transformed_xmin,
        ymin=transformed_ymin,
        xmax=transformed_xmax,
        ymax=transformed_ymax,
        target_width=transform.target_width,
        target_height=transform.target_height,
    )

    width = (
        clipped_xmax
        - clipped_xmin
    )

    height = (
        clipped_ymax
        - clipped_ymin
    )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Transformed box has non-positive dimensions."
        )

    bbox = [
        rounded_float(clipped_xmin),
        rounded_float(clipped_ymin),
        rounded_float(width),
        rounded_float(height),
    ]

    area = rounded_float(
        bbox[2] * bbox[3]
    )

    clipping_applied = any(
        not math.isclose(
            original,
            clipped,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for original, clipped in zip(
            [
                transformed_xmin,
                transformed_ymin,
                transformed_xmax,
                transformed_ymax,
            ],
            [
                clipped_xmin,
                clipped_ymin,
                clipped_xmax,
                clipped_ymax,
            ],
        )
    )

    return {
        "bbox": bbox,
        "area": area,
        "clipping_applied": (
            clipping_applied
        ),
    }


# ============================================================
# COCO STRUCTURE
# ============================================================

def build_categories(
    mapping_config: dict,
) -> tuple[
    list[dict],
    dict[str, int],
]:
    target_classes = (
        mapping_config.get(
            "target_classes",
            [],
        )
    )

    categories: list[dict] = []
    name_to_coco_id: dict[str, int] = {}

    for entry in target_classes:
        class_name = str(
            entry["name"]
        )

        coco_id = int(
            entry["coco_id"]
        )

        if class_name not in (
            TARGET_CLASS_NAMES
        ):
            raise ValueError(
                f"Unexpected target class: "
                f"{class_name}"
            )

        categories.append(
            {
                "id": coco_id,
                "name": class_name,
                "supercategory": (
                    "road_user"
                ),
            }
        )

        name_to_coco_id[
            class_name
        ] = coco_id

    categories.sort(
        key=lambda item: item["id"]
    )

    expected_mapping = {
        "Vehicle": 1,
        "Pedestrian": 2,
        "Cyclist": 3,
    }

    if name_to_coco_id != expected_mapping:
        raise ValueError(
            "COCO class mapping differs from "
            "the frozen Milestone 3 mapping."
        )

    return categories, name_to_coco_id


def build_coco_image_record(
    source_row: pd.Series,
    transform_row: pd.Series,
) -> dict:
    output_path = (
        PROCESSED_ROOT
        / Path(
            str(
                transform_row[
                    "output_relative_path"
                ]
            )
        )
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Processed image not found:\n"
            f"{output_path.resolve()}"
        )

    return {
        "id": int(
            source_row[
                "global_image_id"
            ]
        ),
        "file_name": str(
            source_row[
                "output_filename"
            ]
        ),
        "width": int(
            transform_row[
                "target_width"
            ]
        ),
        "height": int(
            transform_row[
                "target_height"
            ]
        ),
        "source_dataset": str(
            source_row["dataset"]
        ),
        "source_image_id": str(
            source_row[
                "source_image_id"
            ]
        ),
        "partition": str(
            source_row["partition"]
        ),
        "canonical_image_key": str(
            source_row[
                "canonical_image_key"
            ]
        ),
    }


def empty_partition_container(
    categories: list[dict],
    partition_name: str,
) -> dict:
    return {
        "info": {
            "description": (
                "Milestone 3 canonical COCO annotations "
                f"for {partition_name}"
            ),
            "version": "1.0",
            "milestone": 3,
            "canonical_format": True,
            "image_preprocessing": (
                "640x640 centered letterbox"
            ),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }


# ============================================================
# KITTI ANNOTATIONS
# ============================================================

def parse_kitti_target_boxes(
    label_file: Path,
    kitti_mapping: dict,
) -> list[dict]:
    boxes: list[dict] = []

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

        if len(fields) not in {
            15,
            16,
        }:
            raise ValueError(
                f"Invalid KITTI field count in "
                f"{label_file.name}, "
                f"line {line_number}."
            )

        source_class = fields[0]

        mapping_entry = kitti_mapping.get(
            source_class
        )

        if mapping_entry is None:
            raise KeyError(
                f"Unmapped KITTI source class "
                f"'{source_class}'."
            )

        action = str(
            mapping_entry.get(
                "action",
                "",
            )
        )

        if action == "ignore":
            continue

        if action != "map":
            raise ValueError(
                f"Unsupported KITTI mapping "
                f"action: {action}"
            )

        mapped_class_name = str(
            mapping_entry[
                "mapped_class_name"
            ]
        )

        if mapped_class_name not in (
            TARGET_CLASS_NAMES
        ):
            raise ValueError(
                f"Unexpected mapped KITTI class: "
                f"{mapped_class_name}"
            )

        xmin = float(fields[4])
        ymin = float(fields[5])
        xmax = float(fields[6])
        ymax = float(fields[7])

        boxes.append(
            {
                "source_class": (
                    source_class
                ),
                "mapped_class_name": (
                    mapped_class_name
                ),
                "source_box": (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ),
                "source_line_number": (
                    line_number
                ),
            }
        )

    return boxes


# ============================================================
# WAYMO ANNOTATIONS
# ============================================================

def prepare_waymo_groups(
    dataframe: pd.DataFrame,
) -> tuple[
    dict[str, list[dict]],
    dict[str, str],
]:
    image_id_column = resolve_column(
        dataframe,
        [
            "image_id",
            "global_image_id",
        ],
        "Waymo image ID",
    )

    class_column = resolve_column(
        dataframe,
        [
            "class_name",
            "mapped_class_name",
            "category_name",
            "label",
        ],
        "Waymo target class",
    )

    xmin_column = resolve_column(
        dataframe,
        [
            "xmin",
            "x_min",
        ],
        "Waymo xmin",
    )

    ymin_column = resolve_column(
        dataframe,
        [
            "ymin",
            "y_min",
        ],
        "Waymo ymin",
    )

    xmax_column = resolve_column(
        dataframe,
        [
            "xmax",
            "x_max",
        ],
        "Waymo xmax",
    )

    ymax_column = resolve_column(
        dataframe,
        [
            "ymax",
            "y_max",
        ],
        "Waymo ymax",
    )

    columns = {
        "image_id": image_id_column,
        "class_name": class_column,
        "xmin": xmin_column,
        "ymin": ymin_column,
        "xmax": xmax_column,
        "ymax": ymax_column,
    }

    grouped: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for row_number, row in dataframe.iterrows():
        image_id = str(
            row[image_id_column]
        ).strip()

        class_name = str(
            row[class_column]
        ).strip()

        if class_name not in (
            TARGET_CLASS_NAMES
        ):
            raise ValueError(
                f"Unexpected Waymo target class "
                f"'{class_name}' at row "
                f"{row_number}."
            )

        grouped[image_id].append(
            {
                "source_class": class_name,
                "mapped_class_name": class_name,
                "source_box": (
                    float(row[xmin_column]),
                    float(row[ymin_column]),
                    float(row[xmax_column]),
                    float(row[ymax_column]),
                ),
                "source_row_number": int(
                    row_number
                ),
            }
        )

    return dict(grouped), columns


# ============================================================
# PARTITION VALIDATION
# ============================================================

def summarize_partition(
    coco_data: dict,
) -> dict:
    class_id_to_name = {
        int(category["id"]): str(
            category["name"]
        )
        for category in coco_data[
            "categories"
        ]
    }

    class_counts: Counter = Counter()

    annotation_counts_by_image = (
        Counter()
    )

    clipping_count = 0

    for annotation in coco_data[
        "annotations"
    ]:
        class_name = (
            class_id_to_name[
                int(
                    annotation[
                        "category_id"
                    ]
                )
            ]
        )

        class_counts[class_name] += 1

        annotation_counts_by_image[
            int(annotation["image_id"])
        ] += 1

        if annotation.get(
            "clipping_applied",
            False,
        ):
            clipping_count += 1

    negative_images = sum(
        annotation_counts_by_image.get(
            int(image["id"]),
            0,
        )
        == 0
        for image in coco_data["images"]
    )

    return {
        "images": int(
            len(coco_data["images"])
        ),
        "annotations": int(
            len(coco_data["annotations"])
        ),
        "negative_images": int(
            negative_images
        ),
        "class_counts": {
            name: int(
                class_counts[name]
            )
            for name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },
        "clipping_applied_boxes": int(
            clipping_count
        ),
    }


def validate_partition(
    partition_name: str,
    coco_data: dict,
    expected_image_counts: dict[int, int],
    issues: list[dict],
) -> dict:
    expected = EXPECTED[
        partition_name
    ]

    summary = summarize_partition(
        coco_data
    )

    checks = {
        "image_count": (
            summary["images"]
            == expected["images"]
        ),
        "annotation_count": (
            summary["annotations"]
            == expected["annotations"]
        ),
        "negative_image_count": (
            summary["negative_images"]
            == expected[
                "negative_images"
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
            == expected[
                "Pedestrian"
            ]
        ),
        "cyclist_count": (
            summary[
                "class_counts"
            ]["Cyclist"]
            == expected["Cyclist"]
        ),
        "unique_image_ids": (
            len(
                {
                    int(image["id"])
                    for image
                    in coco_data["images"]
                }
            )
            == len(coco_data["images"])
        ),
        "unique_annotation_ids": (
            len(
                {
                    int(annotation["id"])
                    for annotation
                    in coco_data[
                        "annotations"
                    ]
                }
            )
            == len(
                coco_data["annotations"]
            )
        ),
    }

    actual_counts_by_image: Counter = (
        Counter(
            int(
                annotation["image_id"]
            )
            for annotation
            in coco_data["annotations"]
        )
    )

    count_mismatch_ids = []

    for image in coco_data["images"]:
        image_id = int(image["id"])

        expected_count = int(
            expected_image_counts.get(
                image_id,
                0,
            )
        )

        actual_count = int(
            actual_counts_by_image.get(
                image_id,
                0,
            )
        )

        if actual_count != expected_count:
            count_mismatch_ids.append(
                {
                    "image_id": image_id,
                    "expected": expected_count,
                    "actual": actual_count,
                }
            )

    checks[
        "per_image_annotation_counts"
    ] = (
        len(count_mismatch_ids) == 0
    )

    for check_name, passed in checks.items():
        if not passed:
            add_issue(
                issues,
                partition_name,
                "coco_creation_check_failed",
                check_name,
                (
                    "The partition creation check "
                    "returned false."
                ),
            )

    for mismatch in count_mismatch_ids:
        add_issue(
            issues,
            partition_name,
            "per_image_annotation_count_mismatch",
            str(mismatch["image_id"]),
            (
                f"Expected "
                f"{mismatch['expected']}, "
                f"generated "
                f"{mismatch['actual']}."
            ),
        )

    return {
        **summary,
        "count_mismatch_images": int(
            len(count_mismatch_ids)
        ),
        "checks": checks,
        "validation_passed": all(
            checks.values()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []

    print("=" * 76)
    print("CREATING CANONICAL COCO TARGET ANNOTATIONS")
    print("=" * 76)

    mapping_config = load_yaml(
        CLASS_MAPPING_CONFIG
    )

    preprocessing_config = load_yaml(
        PREPROCESSING_CONFIG
    )

    source_summary = load_json(
        SOURCE_MANIFEST_SUMMARY
    )

    preprocessing_report = load_json(
        IMAGE_PREPROCESSING_REPORT
    )

    if not source_summary.get(
        "source_manifest_passed",
        False,
    ):
        raise RuntimeError(
            "Step 3 source manifest has not passed."
        )

    if not preprocessing_report.get(
        "image_preprocessing_passed",
        False,
    ):
        raise RuntimeError(
            "Step 5 image preprocessing has not passed."
        )

    canonical_format = (
        preprocessing_config[
            "annotation_policy"
        ]["canonical_format"]
    )

    if canonical_format != "COCO":
        raise ValueError(
            "preprocessing.yaml does not define "
            "COCO as the canonical format."
        )

    categories, name_to_coco_id = (
        build_categories(
            mapping_config
        )
    )

    kitti_mapping_config = load_yaml(
        KITTI_MAPPING_FILE
    )

    kitti_mapping = (
        kitti_mapping_config.get(
            "kitti_mapping",
            {}
        )
    )

    if not kitti_mapping:
        raise ValueError(
            "KITTI mapping file does not contain "
            "kitti_mapping."
        )

    if not SOURCE_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Source manifest not found:\n"
            f"{SOURCE_MANIFEST_FILE.resolve()}"
        )

    if not TRANSFORM_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Transform manifest not found:\n"
            f"{TRANSFORM_MANIFEST_FILE.resolve()}"
        )

    source_manifest = pd.read_csv(
        SOURCE_MANIFEST_FILE,
        dtype={
            "source_image_id": str,
            "source_annotation_path": str,
            "output_filename": str,
        },
    )

    transform_manifest = pd.read_csv(
        TRANSFORM_MANIFEST_FILE,
        dtype={
            "source_image_id": str,
            "output_filename": str,
            "output_relative_path": str,
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

    transform_manifest[
        "global_image_id"
    ] = pd.to_numeric(
        transform_manifest[
            "global_image_id"
        ],
        errors="raise",
    ).astype(int)

    source_ids = set(
        source_manifest[
            "global_image_id"
        ].tolist()
    )

    transform_ids = set(
        transform_manifest[
            "global_image_id"
        ].tolist()
    )

    if source_ids != transform_ids:
        raise ValueError(
            "Source and transform manifests do not "
            "contain identical global image IDs."
        )

    if (
        source_manifest[
            "global_image_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate global image IDs exist in "
            "the source manifest."
        )

    if (
        transform_manifest[
            "global_image_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate global image IDs exist in "
            "the transform manifest."
        )

    transform_lookup = (
        transform_manifest.set_index(
            "global_image_id",
            drop=False,
        )
    )

    waymo_boxes = pd.read_csv(
        WAYMO_BOXES_FILE,
        dtype=str,
    )

    (
        waymo_groups,
        waymo_columns,
    ) = prepare_waymo_groups(
        waymo_boxes
    )

    partitions = {
        name: empty_partition_container(
            categories=categories,
            partition_name=name,
        )
        for name in PARTITION_OUTPUTS
    }

    expected_counts_by_partition: dict[
        str,
        dict[int, int],
    ] = {
        name: {}
        for name in PARTITION_OUTPUTS
    }

    annotation_id_counters = {
        name: 1
        for name in PARTITION_OUTPUTS
    }

    clipping_counts: Counter = Counter()

    print(
        f"Source images: "
        f"{len(source_manifest)}"
    )

    print(
        f"Waymo source boxes: "
        f"{len(waymo_boxes)}\n"
    )

    source_manifest = (
        source_manifest.sort_values(
            "global_image_id"
        )
        .reset_index(drop=True)
    )

    for _, source_row in tqdm(
        source_manifest.iterrows(),
        total=len(source_manifest),
        unit="image",
    ):
        global_image_id = int(
            source_row[
                "global_image_id"
            ]
        )

        dataset = str(
            source_row["dataset"]
        )

        partition = str(
            source_row["partition"]
        )

        partition_name = partition_key(
            dataset=dataset,
            partition=partition,
        )

        transform_row = (
            transform_lookup.loc[
                global_image_id
            ]
        )

        transform = create_transform(
            transform_row
        )

        image_record = (
            build_coco_image_record(
                source_row=source_row,
                transform_row=transform_row,
            )
        )

        partitions[
            partition_name
        ]["images"].append(
            image_record
        )

        expected_target_count = int(
            source_row[
                "target_box_count"
            ]
        )

        expected_counts_by_partition[
            partition_name
        ][global_image_id] = (
            expected_target_count
        )

        source_image_id = str(
            source_row[
                "source_image_id"
            ]
        )

        if dataset == "KITTI":
            label_file = Path(
                str(
                    source_row[
                        "source_annotation_path"
                    ]
                )
            )

            source_boxes = (
                parse_kitti_target_boxes(
                    label_file=label_file,
                    kitti_mapping=(
                        kitti_mapping
                    ),
                )
            )

        elif dataset == "Waymo":
            source_boxes = (
                waymo_groups.get(
                    source_image_id,
                    [],
                )
            )

        else:
            raise ValueError(
                f"Unsupported dataset: {dataset}"
            )

        if (
            len(source_boxes)
            != expected_target_count
        ):
            add_issue(
                issues,
                dataset,
                "source_box_count_mismatch",
                source_image_id,
                (
                    f"Manifest expected "
                    f"{expected_target_count}, "
                    f"source parser found "
                    f"{len(source_boxes)}."
                ),
            )

        for source_box_record in source_boxes:
            mapped_class_name = str(
                source_box_record[
                    "mapped_class_name"
                ]
            )

            category_id = int(
                name_to_coco_id[
                    mapped_class_name
                ]
            )

            try:
                converted = (
                    convert_source_box(
                        source_box=(
                            source_box_record[
                                "source_box"
                            ]
                        ),
                        transform=transform,
                    )
                )

            except Exception as error:
                add_issue(
                    issues,
                    dataset,
                    "box_transformation_failed",
                    source_image_id,
                    str(error),
                )
                continue

            annotation_id = (
                annotation_id_counters[
                    partition_name
                ]
            )

            annotation_id_counters[
                partition_name
            ] += 1

            clipping_applied = bool(
                converted[
                    "clipping_applied"
                ]
            )

            if clipping_applied:
                clipping_counts[
                    partition_name
                ] += 1

            annotation_record = {
                "id": int(
                    annotation_id
                ),
                "image_id": int(
                    global_image_id
                ),
                "category_id": int(
                    category_id
                ),
                "bbox": converted[
                    "bbox"
                ],
                "area": converted[
                    "area"
                ],
                "iscrowd": 0,

                "source_dataset": (
                    dataset
                ),
                "source_image_id": (
                    source_image_id
                ),
                "source_class": str(
                    source_box_record[
                        "source_class"
                    ]
                ),
                "clipping_applied": (
                    clipping_applied
                ),
            }

            partitions[
                partition_name
            ]["annotations"].append(
                annotation_record
            )

    validation_results = {}

    for partition_name, coco_data in (
        partitions.items()
    ):
        coco_data["images"].sort(
            key=lambda item: int(
                item["id"]
            )
        )

        coco_data[
            "annotations"
        ].sort(
            key=lambda item: int(
                item["id"]
            )
        )

        validation_results[
            partition_name
        ] = validate_partition(
            partition_name=(
                partition_name
            ),
            coco_data=coco_data,
            expected_image_counts=(
                expected_counts_by_partition[
                    partition_name
                ]
            ),
            issues=issues,
        )

    overall_passed = (
        all(
            result[
                "validation_passed"
            ]
            for result
            in validation_results.values()
        )
        and len(issues) == 0
    )

    output_hashes = {}

    for partition_name, output_file in (
        PARTITION_OUTPUTS.items()
    ):
        output_file.write_text(
            json.dumps(
                partitions[
                    partition_name
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        output_hashes[
            partition_name
        ] = sha256_file(
            output_file
        )

    combined_summary = {
        "images": int(
            sum(
                result["images"]
                for result
                in validation_results.values()
            )
        ),
        "annotations": int(
            sum(
                result["annotations"]
                for result
                in validation_results.values()
            )
        ),
        "negative_images": int(
            sum(
                result[
                    "negative_images"
                ]
                for result
                in validation_results.values()
            )
        ),
        "class_counts": {
            class_name: int(
                sum(
                    result[
                        "class_counts"
                    ][class_name]
                    for result
                    in validation_results.values()
                )
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },
        "clipping_applied_boxes": int(
            sum(clipping_counts.values())
        ),
    }

    combined_checks = {
        "image_count": (
            combined_summary["images"]
            == EXPECTED[
                "combined"
            ]["images"]
        ),
        "annotation_count": (
            combined_summary[
                "annotations"
            ]
            == EXPECTED[
                "combined"
            ]["annotations"]
        ),
        "negative_image_count": (
            combined_summary[
                "negative_images"
            ]
            == EXPECTED[
                "combined"
            ]["negative_images"]
        ),
        "vehicle_count": (
            combined_summary[
                "class_counts"
            ]["Vehicle"]
            == EXPECTED[
                "combined"
            ]["Vehicle"]
        ),
        "pedestrian_count": (
            combined_summary[
                "class_counts"
            ]["Pedestrian"]
            == EXPECTED[
                "combined"
            ]["Pedestrian"]
        ),
        "cyclist_count": (
            combined_summary[
                "class_counts"
            ]["Cyclist"]
            == EXPECTED[
                "combined"
            ]["Cyclist"]
        ),
    }

    for check_name, passed in (
        combined_checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "Combined",
                "combined_coco_check_failed",
                check_name,
                (
                    "The combined COCO creation "
                    "check returned false."
                ),
            )

    overall_passed = (
        overall_passed
        and all(
            combined_checks.values()
        )
        and len(issues) == 0
    )

    report = {
        "milestone": 3,
        "step": 6,
        "purpose": (
            "Create canonical COCO target "
            "annotations for all experimental "
            "partitions."
        ),
        "coordinate_precision": (
            COORDINATE_PRECISION
        ),
        "categories": categories,
        "waymo_resolved_columns": (
            waymo_columns
        ),
        "partitions": (
            validation_results
        ),
        "combined": (
            combined_summary
        ),
        "combined_checks": (
            combined_checks
        ),
        "output_files": {
            partition_name: {
                "path": (
                    output_file.as_posix()
                ),
                "sha256": (
                    output_hashes[
                        partition_name
                    ]
                ),
            }
            for partition_name, output_file
            in PARTITION_OUTPUTS.items()
        },
        "issue_count": len(issues),
        "coco_creation_passed": (
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
    print("CANONICAL COCO CREATION SUMMARY")
    print("=" * 76)

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        result = validation_results[
            partition_name
        ]

        print(f"\n{partition_name}:")

        print(
            f"  Images: "
            f"{result['images']}"
        )

        print(
            f"  Annotations: "
            f"{result['annotations']}"
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
            f"  Negative images: "
            f"{result['negative_images']}"
        )

        print(
            f"  Clipped boxes: "
            f"{result['clipping_applied_boxes']}"
        )

        print(
            f"  Status: "
            f"{'PASSED' if result['validation_passed'] else 'FAILED'}"
        )

    print("\nCombined:")

    print(
        f"  Images: "
        f"{combined_summary['images']}"
    )

    print(
        f"  Annotations: "
        f"{combined_summary['annotations']}"
    )

    print(
        f"  Vehicle: "
        f"{combined_summary['class_counts']['Vehicle']}"
    )

    print(
        f"  Pedestrian: "
        f"{combined_summary['class_counts']['Pedestrian']}"
    )

    print(
        f"  Cyclist: "
        f"{combined_summary['class_counts']['Cyclist']}"
    )

    print(
        f"  Negative images: "
        f"{combined_summary['negative_images']}"
    )

    print(
        f"  Clipped boxes: "
        f"{combined_summary['clipping_applied_boxes']}"
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

    print("\nCreated files:")

    for output_file in (
        PARTITION_OUTPUTS.values()
    ):
        print(
            f"  {output_file.resolve()}"
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
            "\nDo not continue to ignored-region "
            "generation until all COCO creation "
            "issues are resolved."
        )

        sys.exit(1)

    print(
        "\nStep 6 completed successfully."
    )


if __name__ == "__main__":
    main()