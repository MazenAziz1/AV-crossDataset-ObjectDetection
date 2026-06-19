from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml


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

TRANSFORM_MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/transform_manifest.csv"
)

COCO_CONFIG_FILE = Path(
    "configs/datasets/milestone_3/coco_paths.yaml"
)

CLASS_MAPPING_FILE = Path(
    "configs/datasets/milestone_3/class_mapping.yaml"
)

CONFIG_VALIDATION_REPORT = (
    PROCESSED_ROOT
    / "reports/config_validation.json"
)

REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/coco_validation_report.json"
)

ISSUES_FILE = (
    PROCESSED_ROOT
    / "reports/coco_validation_issues.csv"
)


PARTITIONS = {
    "kitti_train": {
        "dataset": "KITTI",
        "partition": "train",

        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/train"
        ),

        "annotation_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_train.json"
        ),

        "expected": {
            "images": 5985,
            "annotations": 31294,
            "negative_images": 0,
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

        "annotation_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_val.json"
        ),

        "expected": {
            "images": 1496,
            "annotations": 7792,
            "negative_images": 0,
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

        "annotation_file": (
            PROCESSED_ROOT
            / "annotations/coco/waymo_external.json"
        ),

        "expected": {
            "images": 996,
            "annotations": 24819,
            "negative_images": 12,
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
    "Vehicle": 49678,
    "Pedestrian": 11836,
    "Cyclist": 2391,
}


EXPECTED_CATEGORIES = {
    1: "Vehicle",
    2: "Pedestrian",
    3: "Cyclist",
}


BOUNDARY_TOLERANCE = 1e-6
AREA_ABSOLUTE_TOLERANCE = 1e-5
AREA_RELATIVE_TOLERANCE = 1e-9


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


def parse_bool(value) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def finite_number(value) -> bool:
    try:
        return math.isfinite(
            float(value)
        )

    except (
        TypeError,
        ValueError,
    ):
        return False


def verify_image_file(
    path: Path,
) -> tuple[
    bool,
    int,
    int,
    str,
    str,
]:
    if not path.exists():
        return (
            False,
            0,
            0,
            "",
            "File does not exist.",
        )

    if not path.is_file():
        return (
            False,
            0,
            0,
            "",
            "Path is not a file.",
        )

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            mode = str(image.mode)
            image_format = str(
                image.format
            )

        return (
            True,
            int(width),
            int(height),
            mode,
            image_format,
        )

    except Exception as error:
        return (
            False,
            0,
            0,
            "",
            str(error),
        )


def build_expected_class_mapping(
    mapping_config: dict,
) -> dict[int, str]:
    classes = (
        mapping_config.get(
            "target_classes",
            []
        )
    )

    mapping = {
        int(entry["coco_id"]): str(
            entry["name"]
        )
        for entry in classes
    }

    if mapping != EXPECTED_CATEGORIES:
        raise ValueError(
            "The frozen COCO category mapping "
            "does not match 1=Vehicle, "
            "2=Pedestrian, 3=Cyclist."
        )

    return mapping


def validate_coco_config_paths(
    coco_config: dict,
    issues: list[dict],
) -> dict:
    config_partitions = (
        coco_config.get(
            "partitions",
            {}
        )
    )

    expected_config_names = {
        "kitti_train",
        "kitti_validation",
        "waymo_external",
    }

    checks = {
        "partition_names": (
            set(
                config_partitions.keys()
            )
            == expected_config_names
        )
    }

    expected_paths = {
        "kitti_train": {
            "images": (
                PARTITIONS[
                    "kitti_train"
                ]["image_dir"]
            ),

            "annotations": (
                PARTITIONS[
                    "kitti_train"
                ]["annotation_file"]
            ),
        },

        "kitti_validation": {
            "images": (
                PARTITIONS[
                    "kitti_val"
                ]["image_dir"]
            ),

            "annotations": (
                PARTITIONS[
                    "kitti_val"
                ]["annotation_file"]
            ),
        },

        "waymo_external": {
            "images": (
                PARTITIONS[
                    "waymo_external"
                ]["image_dir"]
            ),

            "annotations": (
                PARTITIONS[
                    "waymo_external"
                ]["annotation_file"]
            ),
        },
    }

    path_checks: dict[str, bool] = {}

    for config_name, paths in (
        expected_paths.items()
    ):
        values = config_partitions.get(
            config_name,
            {}
        )

        for field, expected_path in (
            paths.items()
        ):
            configured_path = Path(
                str(
                    values.get(
                        field,
                        "",
                    )
                )
            )

            passed = (
                configured_path.resolve()
                == expected_path.resolve()
            )

            path_checks[
                f"{config_name}:{field}"
            ] = passed

            if not passed:
                add_issue(
                    issues,
                    "configuration",
                    "coco_config_path_mismatch",
                    (
                        f"{config_name}:"
                        f"{field}"
                    ),
                    (
                        f"Configured="
                        f"{configured_path}; "
                        f"expected="
                        f"{expected_path}"
                    ),
                )

    checks["all_paths_match"] = all(
        path_checks.values()
    )

    if not checks["partition_names"]:
        add_issue(
            issues,
            "configuration",
            "coco_config_partition_mismatch",
            "partitions",
            (
                f"Found "
                f"{list(config_partitions.keys())}"
            ),
        )

    return {
        "checks": checks,
        "path_checks": path_checks,
        "validation_passed": all(
            checks.values()
        ),
    }


# ============================================================
# PARTITION VALIDATION
# ============================================================

def validate_partition(
    partition_name: str,
    specification: dict,
    source_manifest: pd.DataFrame,
    transform_manifest: pd.DataFrame,
    class_mapping: dict[int, str],
    issues: list[dict],
) -> dict:
    annotation_file = Path(
        specification[
            "annotation_file"
        ]
    )

    image_dir = Path(
        specification["image_dir"]
    )

    expected = specification[
        "expected"
    ]

    coco_data = load_json(
        annotation_file
    )

    required_top_level_keys = {
        "info",
        "licenses",
        "images",
        "annotations",
        "categories",
    }

    missing_top_level_keys = (
        required_top_level_keys
        - set(coco_data.keys())
    )

    if missing_top_level_keys:
        add_issue(
            issues,
            partition_name,
            "missing_top_level_keys",
            annotation_file.name,
            str(
                sorted(
                    missing_top_level_keys
                )
            ),
        )

    images = coco_data.get(
        "images",
        [],
    )

    annotations = coco_data.get(
        "annotations",
        [],
    )

    categories = coco_data.get(
        "categories",
        [],
    )

    if not isinstance(images, list):
        raise ValueError(
            f"{annotation_file} images must be a list."
        )

    if not isinstance(
        annotations,
        list,
    ):
        raise ValueError(
            f"{annotation_file} annotations "
            "must be a list."
        )

    if not isinstance(
        categories,
        list,
    ):
        raise ValueError(
            f"{annotation_file} categories "
            "must be a list."
        )

    category_ids = [
        int(category["id"])
        for category in categories
    ]

    category_names = {
        int(category["id"]): str(
            category["name"]
        )
        for category in categories
    }

    category_supercategories = {
        int(category["id"]): str(
            category.get(
                "supercategory",
                "",
            )
        )
        for category in categories
    }

    category_checks = {
        "three_categories": (
            len(categories) == 3
        ),

        "unique_category_ids": (
            len(category_ids)
            == len(set(category_ids))
        ),

        "exact_category_mapping": (
            category_names
            == class_mapping
        ),

        "supercategories": all(
            value == "road_user"
            for value in (
                category_supercategories
                .values()
            )
        ),
    }

    for check_name, passed in (
        category_checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                partition_name,
                "category_check_failed",
                check_name,
                (
                    f"Categories={categories}"
                ),
            )

    expected_source_rows = (
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
    )

    expected_transform_rows = (
        transform_manifest[
            (
                transform_manifest["dataset"]
                == specification["dataset"]
            )
            & (
                transform_manifest["partition"]
                == specification["partition"]
            )
        ]
        .copy()
    )

    expected_source_rows = (
        expected_source_rows.set_index(
            "global_image_id",
            drop=False,
        )
    )

    expected_transform_rows = (
        expected_transform_rows.set_index(
            "global_image_id",
            drop=False,
        )
    )

    expected_image_ids = set(
        int(value)
        for value in (
            expected_source_rows.index
        )
    )

    transform_image_ids = set(
        int(value)
        for value in (
            expected_transform_rows.index
        )
    )

    if (
        expected_image_ids
        != transform_image_ids
    ):
        add_issue(
            issues,
            partition_name,
            "manifest_id_mismatch",
            partition_name,
            (
                "Source and transform manifests "
                "have different image IDs."
            ),
        )

    image_ids: list[int] = []
    file_names: list[str] = []

    coco_image_lookup: dict[
        int,
        dict,
    ] = {}

    unreadable_images = 0
    missing_images = 0
    image_metadata_mismatches = 0

    print(
        f"\nValidating {partition_name} "
        f"COCO images..."
    )

    for image in tqdm(
        images,
        unit="image",
    ):
        try:
            image_id = int(
                image["id"]
            )

            file_name = str(
                image["file_name"]
            )

            width = int(
                image["width"]
            )

            height = int(
                image["height"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            add_issue(
                issues,
                partition_name,
                "invalid_image_record",
                str(
                    image.get(
                        "id",
                        "unknown",
                    )
                ),
                str(error),
            )
            continue

        image_ids.append(image_id)
        file_names.append(file_name)

        if image_id in (
            coco_image_lookup
        ):
            add_issue(
                issues,
                partition_name,
                "duplicate_image_id",
                str(image_id),
                file_name,
            )
            continue

        coco_image_lookup[
            image_id
        ] = image

        if image_id not in (
            expected_image_ids
        ):
            add_issue(
                issues,
                partition_name,
                "unknown_coco_image_id",
                str(image_id),
                file_name,
            )
            continue

        source_row = (
            expected_source_rows.loc[
                image_id
            ]
        )

        transform_row = (
            expected_transform_rows.loc[
                image_id
            ]
        )

        expected_filename = str(
            source_row[
                "output_filename"
            ]
        )

        expected_width = int(
            transform_row[
                "target_width"
            ]
        )

        expected_height = int(
            transform_row[
                "target_height"
            ]
        )

        metadata_checks = {
            "file_name": (
                file_name
                == expected_filename
            ),

            "width": (
                width
                == expected_width
                == 640
            ),

            "height": (
                height
                == expected_height
                == 640
            ),

            "source_dataset": (
                str(
                    image.get(
                        "source_dataset",
                        "",
                    )
                )
                == str(
                    source_row["dataset"]
                )
            ),

            "source_image_id": (
                str(
                    image.get(
                        "source_image_id",
                        "",
                    )
                )
                == str(
                    source_row[
                        "source_image_id"
                    ]
                )
            ),

            "partition": (
                str(
                    image.get(
                        "partition",
                        "",
                    )
                )
                == str(
                    source_row["partition"]
                )
            ),

            "canonical_image_key": (
                str(
                    image.get(
                        "canonical_image_key",
                        "",
                    )
                )
                == str(
                    source_row[
                        "canonical_image_key"
                    ]
                )
            ),
        }

        if not all(
            metadata_checks.values()
        ):
            image_metadata_mismatches += 1

            add_issue(
                issues,
                partition_name,
                "image_metadata_mismatch",
                str(image_id),
                str(metadata_checks),
            )

        if (
            "/" in file_name
            or "\\" in file_name
        ):
            add_issue(
                issues,
                partition_name,
                "non_flat_coco_filename",
                str(image_id),
                file_name,
            )

        image_path = (
            image_dir / file_name
        )

        (
            readable,
            actual_width,
            actual_height,
            mode,
            image_format,
        ) = verify_image_file(
            image_path
        )

        if not image_path.exists():
            missing_images += 1

        if not readable:
            unreadable_images += 1

            add_issue(
                issues,
                partition_name,
                "unreadable_image",
                str(image_id),
                (
                    f"{image_path}: "
                    f"{image_format}"
                ),
            )
            continue

        physical_checks = {
            "width": (
                actual_width == width
            ),

            "height": (
                actual_height == height
            ),

            "mode": (
                mode == "RGB"
            ),

            "format": (
                image_format.upper()
                == "PNG"
            ),
        }

        if not all(
            physical_checks.values()
        ):
            add_issue(
                issues,
                partition_name,
                "physical_image_mismatch",
                str(image_id),
                (
                    f"path={image_path}; "
                    f"checks="
                    f"{physical_checks}"
                ),
            )

    coco_image_id_set = set(
        image_ids
    )

    missing_coco_image_ids = sorted(
        expected_image_ids
        - coco_image_id_set
    )

    extra_coco_image_ids = sorted(
        coco_image_id_set
        - expected_image_ids
    )

    for image_id in (
        missing_coco_image_ids
    ):
        add_issue(
            issues,
            partition_name,
            "missing_coco_image_record",
            str(image_id),
            "Expected image ID is absent.",
        )

    for image_id in (
        extra_coco_image_ids
    ):
        add_issue(
            issues,
            partition_name,
            "extra_coco_image_record",
            str(image_id),
            "Unexpected image ID is present.",
        )

    physical_png_names = {
        path.name
        for path in image_dir.glob(
            "*.png"
        )
        if path.is_file()
    }

    coco_file_name_set = set(
        file_names
    )

    missing_physical_files = sorted(
        coco_file_name_set
        - physical_png_names
    )

    extra_physical_files = sorted(
        physical_png_names
        - coco_file_name_set
    )

    for file_name in (
        missing_physical_files
    ):
        add_issue(
            issues,
            partition_name,
            "missing_physical_image",
            file_name,
            str(
                image_dir / file_name
            ),
        )

    for file_name in (
        extra_physical_files
    ):
        add_issue(
            issues,
            partition_name,
            "physical_image_not_in_coco",
            file_name,
            str(
                image_dir / file_name
            ),
        )

    # --------------------------------------------------------
    # Annotation validation
    # --------------------------------------------------------

    annotation_ids: list[int] = []

    annotation_counts_by_image: Counter = (
        Counter()
    )

    class_counts_by_image: dict[
        int,
        Counter,
    ] = defaultdict(Counter)

    total_class_counts: Counter = (
        Counter()
    )

    invalid_boxes = 0
    out_of_bounds_boxes = 0
    area_mismatches = 0
    unknown_image_references = 0
    unknown_categories = 0
    metadata_annotation_mismatches = 0

    print(
        f"Validating {partition_name} "
        f"COCO annotations..."
    )

    for annotation in tqdm(
        annotations,
        unit="annotation",
    ):
        try:
            annotation_id = int(
                annotation["id"]
            )

            image_id = int(
                annotation["image_id"]
            )

            category_id = int(
                annotation["category_id"]
            )

            bbox = annotation["bbox"]

            area = float(
                annotation["area"]
            )

            iscrowd = int(
                annotation["iscrowd"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            add_issue(
                issues,
                partition_name,
                "invalid_annotation_record",
                str(
                    annotation.get(
                        "id",
                        "unknown",
                    )
                ),
                str(error),
            )
            continue

        annotation_ids.append(
            annotation_id
        )

        if image_id not in (
            coco_image_lookup
        ):
            unknown_image_references += 1

            add_issue(
                issues,
                partition_name,
                "annotation_unknown_image",
                str(annotation_id),
                str(image_id),
            )
            continue

        if category_id not in (
            class_mapping
        ):
            unknown_categories += 1

            add_issue(
                issues,
                partition_name,
                "annotation_unknown_category",
                str(annotation_id),
                str(category_id),
            )
            continue

        if (
            not isinstance(
                bbox,
                list,
            )
            or len(bbox) != 4
            or not all(
                finite_number(value)
                for value in bbox
            )
        ):
            invalid_boxes += 1

            add_issue(
                issues,
                partition_name,
                "invalid_bbox_structure",
                str(annotation_id),
                str(bbox),
            )
            continue

        x, y, width, height = [
            float(value)
            for value in bbox
        ]

        if (
            width <= 0
            or height <= 0
        ):
            invalid_boxes += 1

            add_issue(
                issues,
                partition_name,
                "non_positive_bbox",
                str(annotation_id),
                str(bbox),
            )
            continue

        image_record = (
            coco_image_lookup[
                image_id
            ]
        )

        image_width = float(
            image_record["width"]
        )

        image_height = float(
            image_record["height"]
        )

        bounds_valid = (
            x >= -BOUNDARY_TOLERANCE
            and y >= -BOUNDARY_TOLERANCE
            and x + width
            <= image_width
            + BOUNDARY_TOLERANCE
            and y + height
            <= image_height
            + BOUNDARY_TOLERANCE
        )

        if not bounds_valid:
            out_of_bounds_boxes += 1

            add_issue(
                issues,
                partition_name,
                "out_of_bounds_bbox",
                str(annotation_id),
                (
                    f"bbox={bbox}; "
                    f"image="
                    f"{image_width}x"
                    f"{image_height}"
                ),
            )

        expected_area = (
            width * height
        )

        if (
            not math.isfinite(area)
            or area <= 0
            or not math.isclose(
                area,
                expected_area,
                rel_tol=(
                    AREA_RELATIVE_TOLERANCE
                ),
                abs_tol=(
                    AREA_ABSOLUTE_TOLERANCE
                ),
            )
        ):
            area_mismatches += 1

            add_issue(
                issues,
                partition_name,
                "bbox_area_mismatch",
                str(annotation_id),
                (
                    f"stored={area}; "
                    f"calculated="
                    f"{expected_area}"
                ),
            )

        if iscrowd != 0:
            add_issue(
                issues,
                partition_name,
                "invalid_iscrowd",
                str(annotation_id),
                str(iscrowd),
            )

        image_source_dataset = str(
            image_record.get(
                "source_dataset",
                "",
            )
        )

        image_source_id = str(
            image_record.get(
                "source_image_id",
                "",
            )
        )

        annotation_metadata_checks = {
            "source_dataset": (
                str(
                    annotation.get(
                        "source_dataset",
                        "",
                    )
                )
                == image_source_dataset
            ),

            "source_image_id": (
                str(
                    annotation.get(
                        "source_image_id",
                        "",
                    )
                )
                == image_source_id
            ),

            "source_class_present": (
                str(
                    annotation.get(
                        "source_class",
                        "",
                    )
                ).strip()
                != ""
            ),

            "clipping_flag_boolean": (
                isinstance(
                    annotation.get(
                        "clipping_applied",
                        None,
                    ),
                    bool,
                )
            ),
        }

        if not all(
            annotation_metadata_checks
            .values()
        ):
            (
                metadata_annotation_mismatches
            ) += 1

            add_issue(
                issues,
                partition_name,
                "annotation_metadata_mismatch",
                str(annotation_id),
                str(
                    annotation_metadata_checks
                ),
            )

        class_name = (
            class_mapping[
                category_id
            ]
        )

        annotation_counts_by_image[
            image_id
        ] += 1

        class_counts_by_image[
            image_id
        ][class_name] += 1

        total_class_counts[
            class_name
        ] += 1

    duplicate_annotation_ids = (
        len(annotation_ids)
        - len(set(annotation_ids))
    )

    if duplicate_annotation_ids:
        add_issue(
            issues,
            partition_name,
            "duplicate_annotation_ids",
            partition_name,
            str(
                duplicate_annotation_ids
            ),
        )

    # --------------------------------------------------------
    # Per-image agreement with source manifest
    # --------------------------------------------------------

    target_count_mismatches = 0
    class_count_mismatches = 0

    for image_id in sorted(
        expected_image_ids
    ):
        source_row = (
            expected_source_rows.loc[
                image_id
            ]
        )

        actual_target_count = int(
            annotation_counts_by_image.get(
                image_id,
                0,
            )
        )

        expected_target_count = int(
            source_row[
                "target_box_count"
            ]
        )

        if (
            actual_target_count
            != expected_target_count
        ):
            target_count_mismatches += 1

            add_issue(
                issues,
                partition_name,
                "per_image_target_count_mismatch",
                str(image_id),
                (
                    f"expected="
                    f"{expected_target_count}; "
                    f"actual="
                    f"{actual_target_count}"
                ),
            )

        expected_per_class = {
            "Vehicle": int(
                source_row[
                    "vehicle_count"
                ]
            ),

            "Pedestrian": int(
                source_row[
                    "pedestrian_count"
                ]
            ),

            "Cyclist": int(
                source_row[
                    "cyclist_count"
                ]
            ),
        }

        actual_per_class = {
            class_name: int(
                class_counts_by_image[
                    image_id
                ][class_name]
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        }

        if (
            actual_per_class
            != expected_per_class
        ):
            class_count_mismatches += 1

            add_issue(
                issues,
                partition_name,
                "per_image_class_count_mismatch",
                str(image_id),
                (
                    f"expected="
                    f"{expected_per_class}; "
                    f"actual="
                    f"{actual_per_class}"
                ),
            )

    negative_images = sum(
        annotation_counts_by_image.get(
            image_id,
            0,
        )
        == 0
        for image_id in expected_image_ids
    )

    summary = {
        "images": int(
            len(images)
        ),

        "annotations": int(
            len(annotations)
        ),

        "negative_images": int(
            negative_images
        ),

        "class_counts": {
            class_name: int(
                total_class_counts[
                    class_name
                ]
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        },

        "missing_images": int(
            missing_images
        ),

        "unreadable_images": int(
            unreadable_images
        ),

        "image_metadata_mismatches": int(
            image_metadata_mismatches
        ),

        "invalid_boxes": int(
            invalid_boxes
        ),

        "out_of_bounds_boxes": int(
            out_of_bounds_boxes
        ),

        "area_mismatches": int(
            area_mismatches
        ),

        "unknown_image_references": int(
            unknown_image_references
        ),

        "unknown_categories": int(
            unknown_categories
        ),

        "annotation_metadata_mismatches": int(
            metadata_annotation_mismatches
        ),

        "duplicate_image_ids": int(
            len(image_ids)
            - len(set(image_ids))
        ),

        "duplicate_file_names": int(
            len(file_names)
            - len(set(file_names))
        ),

        "duplicate_annotation_ids": int(
            duplicate_annotation_ids
        ),

        "missing_coco_image_records": int(
            len(missing_coco_image_ids)
        ),

        "extra_coco_image_records": int(
            len(extra_coco_image_ids)
        ),

        "missing_physical_files": int(
            len(missing_physical_files)
        ),

        "extra_physical_files": int(
            len(extra_physical_files)
        ),

        "target_count_mismatch_images": int(
            target_count_mismatches
        ),

        "class_count_mismatch_images": int(
            class_count_mismatches
        ),
    }

    checks = {
        "top_level_structure": (
            len(missing_top_level_keys)
            == 0
        ),

        "categories": all(
            category_checks.values()
        ),

        "image_count": (
            summary["images"]
            == expected["images"]
        ),

        "annotation_count": (
            summary["annotations"]
            == expected[
                "annotations"
            ]
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
            == expected["Pedestrian"]
        ),

        "cyclist_count": (
            summary[
                "class_counts"
            ]["Cyclist"]
            == expected["Cyclist"]
        ),

        "unique_image_ids": (
            summary[
                "duplicate_image_ids"
            ]
            == 0
        ),

        "unique_file_names": (
            summary[
                "duplicate_file_names"
            ]
            == 0
        ),

        "unique_annotation_ids": (
            summary[
                "duplicate_annotation_ids"
            ]
            == 0
        ),

        "exact_image_id_set": (
            summary[
                "missing_coco_image_records"
            ]
            == 0
            and summary[
                "extra_coco_image_records"
            ]
            == 0
        ),

        "exact_physical_file_set": (
            summary[
                "missing_physical_files"
            ]
            == 0
            and summary[
                "extra_physical_files"
            ]
            == 0
        ),

        "all_images_readable": (
            summary[
                "unreadable_images"
            ]
            == 0
        ),

        "image_metadata": (
            summary[
                "image_metadata_mismatches"
            ]
            == 0
        ),

        "valid_boxes": (
            summary["invalid_boxes"]
            == 0
        ),

        "boxes_in_bounds": (
            summary[
                "out_of_bounds_boxes"
            ]
            == 0
        ),

        "areas_consistent": (
            summary[
                "area_mismatches"
            ]
            == 0
        ),

        "valid_references": (
            summary[
                "unknown_image_references"
            ]
            == 0
            and summary[
                "unknown_categories"
            ]
            == 0
        ),

        "annotation_metadata": (
            summary[
                "annotation_metadata_mismatches"
            ]
            == 0
        ),

        "per_image_target_counts": (
            summary[
                "target_count_mismatch_images"
            ]
            == 0
        ),

        "per_image_class_counts": (
            summary[
                "class_count_mismatch_images"
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
                "partition_validation_failed",
                check_name,
                (
                    "The independent COCO "
                    "validation check returned false."
                ),
            )

    return {
        **summary,
        "category_checks": (
            category_checks
        ),
        "checks": checks,
        "image_ids": sorted(
            expected_image_ids
        ),
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
    print("INDEPENDENT CANONICAL COCO VALIDATION")
    print("=" * 76)

    configuration_report = load_json(
        CONFIG_VALIDATION_REPORT
    )

    if not configuration_report.get(
        "config_validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 9 configuration validation "
            "has not passed."
        )

    class_mapping_config = load_yaml(
        CLASS_MAPPING_FILE
    )

    coco_config = load_yaml(
        COCO_CONFIG_FILE
    )

    class_mapping = (
        build_expected_class_mapping(
            class_mapping_config
        )
    )

    config_validation = (
        validate_coco_config_paths(
            coco_config,
            issues,
        )
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
            "output_filename": str,
        },
    )

    transform_manifest = pd.read_csv(
        TRANSFORM_MANIFEST_FILE,
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

    transform_manifest[
        "global_image_id"
    ] = pd.to_numeric(
        transform_manifest[
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

    if (
        transform_manifest[
            "global_image_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "Transform manifest contains duplicate "
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
            transform_manifest=(
                transform_manifest
            ),
            class_mapping=(
                class_mapping
            ),
            issues=issues,
        )

    # --------------------------------------------------------
    # Cross-partition checks
    # --------------------------------------------------------

    image_id_sets = {
        name: set(
            result["image_ids"]
        )
        for name, result in (
            results.items()
        )
    }

    pairwise_overlaps = {
        "kitti_train__kitti_val": (
            image_id_sets[
                "kitti_train"
            ]
            & image_id_sets[
                "kitti_val"
            ]
        ),

        "kitti_train__waymo_external": (
            image_id_sets[
                "kitti_train"
            ]
            & image_id_sets[
                "waymo_external"
            ]
        ),

        "kitti_val__waymo_external": (
            image_id_sets[
                "kitti_val"
            ]
            & image_id_sets[
                "waymo_external"
            ]
        ),
    }

    overlap_counts = {
        name: int(len(values))
        for name, values
        in pairwise_overlaps.items()
    }

    for name, overlap in (
        pairwise_overlaps.items()
    ):
        if overlap:
            add_issue(
                issues,
                "combined",
                "cross_partition_image_id_overlap",
                name,
                str(
                    sorted(overlap)[:20]
                ),
            )

    combined = {
        "images": int(
            sum(
                result["images"]
                for result
                in results.values()
            )
        ),

        "annotations": int(
            sum(
                result["annotations"]
                for result
                in results.values()
            )
        ),

        "negative_images": int(
            sum(
                result[
                    "negative_images"
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
    }

    combined_checks = {
        "image_count": (
            combined["images"]
            == EXPECTED_COMBINED[
                "images"
            ]
        ),

        "annotation_count": (
            combined[
                "annotations"
            ]
            == EXPECTED_COMBINED[
                "annotations"
            ]
        ),

        "negative_image_count": (
            combined[
                "negative_images"
            ]
            == EXPECTED_COMBINED[
                "negative_images"
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

        "no_partition_overlap": all(
            count == 0
            for count in (
                overlap_counts.values()
            )
        ),

        "all_partitions_passed": all(
            result[
                "validation_passed"
            ]
            for result
            in results.values()
        ),

        "configuration_passed": (
            config_validation[
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
                    "The combined COCO "
                    "validation check returned false."
                ),
            )

    overall_passed = (
        all(
            combined_checks.values()
        )
        and len(issues) == 0
    )

    report_results = {}

    for name, result in (
        results.items()
    ):
        report_results[name] = {
            key: value
            for key, value
            in result.items()
            if key != "image_ids"
        }

    report = {
        "milestone": 3,
        "step": 10,

        "purpose": (
            "Independently validate the saved "
            "canonical COCO datasets."
        ),

        "category_mapping": {
            str(key): value
            for key, value
            in class_mapping.items()
        },

        "configuration_validation": (
            config_validation
        ),

        "partitions": (
            report_results
        ),

        "cross_partition_overlap_counts": (
            overlap_counts
        ),

        "combined": combined,

        "combined_checks": (
            combined_checks
        ),

        "issue_count": len(issues),

        "coco_validation_passed": (
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
    print("COCO VALIDATION SUMMARY")
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
            f"  Annotations: "
            f"{result['annotations']}"
        )

        print(
            f"  Negative images: "
            f"{result['negative_images']}"
        )

        print(
            f"  Invalid boxes: "
            f"{result['invalid_boxes']}"
        )

        print(
            f"  Out-of-bounds boxes: "
            f"{result['out_of_bounds_boxes']}"
        )

        print(
            f"  Area mismatches: "
            f"{result['area_mismatches']}"
        )

        print(
            f"  Missing images: "
            f"{result['missing_physical_files']}"
        )

        print(
            f"  Per-image count mismatches: "
            f"{result['target_count_mismatch_images']}"
        )

        print(
            f"  Status: "
            f"{'PASSED' if result['validation_passed'] else 'FAILED'}"
        )

    print("\nCross-partition overlaps:")

    for name, count in (
        overlap_counts.items()
    ):
        print(
            f"  {name}: {count}"
        )

    print("\nCombined:")

    print(
        f"  Images: "
        f"{combined['images']}"
    )

    print(
        f"  Annotations: "
        f"{combined['annotations']}"
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
        f"  Negative images: "
        f"{combined['negative_images']}"
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
            "\nDo not continue to YOLO "
            "validation until all COCO issues "
            "are resolved."
        )

        sys.exit(1)

    print(
        "\nStep 10 completed successfully."
    )


if __name__ == "__main__":
    main()