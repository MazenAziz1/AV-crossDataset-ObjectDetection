from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import sys

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

SOURCE_VALIDATION_REPORT = Path(
    "data/processed/milestone_3/reports/"
    "source_input_validation.json"
)

OUTPUT_DIR = Path(
    "data/processed/milestone_3/manifests"
)

REPORT_DIR = Path(
    "data/processed/milestone_3/reports"
)

SOURCE_MANIFEST_FILE = (
    OUTPUT_DIR / "source_manifest.csv"
)

SUMMARY_FILE = (
    REPORT_DIR / "source_manifest_summary.json"
)

ISSUES_FILE = (
    REPORT_DIR / "source_manifest_issues.csv"
)


# ------------------------------------------------------------
# KITTI
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# WAYMO
# ------------------------------------------------------------

WAYMO_SUBSET_ROOT = Path(
    "data/waymo/representative_subset"
)

WAYMO_IMAGE_DIR = (
    WAYMO_SUBSET_ROOT / "images/front"
)

WAYMO_MANIFEST_FILE = (
    WAYMO_SUBSET_ROOT / "metadata/manifest.csv"
)

WAYMO_BOXES_FILE = (
    WAYMO_SUBSET_ROOT / "annotations/boxes.csv"
)

WAYMO_RAW_CAMERA_BOX_DIR = Path(
    "data/waymo/raw/validation/"
    "camera_box/candidates"
)


# Waymo CameraBox enum values.
WAYMO_FRONT_CAMERA_ID = 1
WAYMO_SIGN_TYPE_ID = 3


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

    "waymo_images": 996,
    "waymo_target_boxes": 24819,
    "waymo_negative_images": 12,
    "waymo_segments": 25,

    "combined_images": 8477,
    "combined_target_boxes": 63905,

    "vehicle_boxes": 49678,
    "pedestrian_boxes": 11836,
    "cyclist_boxes": 2391,
}


MANIFEST_COLUMNS = [
    "global_image_id",
    "canonical_image_key",
    "dataset",
    "partition",
    "experimental_role",

    "source_image_id",
    "source_image_path",
    "source_annotation_path",
    "source_annotation_key",
    "source_calibration_path",
    "source_raw_annotation_path",

    "source_width",
    "source_height",
    "source_extension",

    "output_filename",
    "output_relative_path",

    "target_box_count",
    "vehicle_count",
    "pedestrian_count",
    "cyclist_count",
    "ignored_box_count",
    "is_negative",

    "segment_id",
    "frame_timestamp_micros",
    "camera_name",
    "time_of_day",
    "weather",
    "location",
    "density_group",
    "sampling_rule",
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


def project_relative_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(
            Path.cwd().resolve()
        )

        return relative.as_posix()

    except ValueError:
        return path.resolve().as_posix()


def read_id_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"ID file not found:\n"
            f"{path.resolve()}"
        )

    return [
        line.strip().zfill(6)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


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
        f"Could not find {description} column.\n"
        f"Accepted aliases: {aliases}\n"
        f"Available columns: "
        f"{list(dataframe.columns)}"
    )


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
) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size

    return int(width), int(height)


def boolean_text(value: bool) -> str:
    return "true" if value else "false"


# ============================================================
# KITTI MANIFEST
# ============================================================

def inspect_kitti_label(
    label_file: Path,
    mapping: dict,
) -> dict:
    counts = Counter()
    ignored_count = 0

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
            raise ValueError(
                f"Invalid KITTI annotation row in "
                f"{label_file.name}, line {line_number}."
            )

        source_class = fields[0]

        entry = mapping.get(
            source_class
        )

        if entry is None:
            raise KeyError(
                f"Unmapped KITTI class "
                f"'{source_class}' in {label_file.name}."
            )

        action = entry.get("action")

        if action == "map":
            mapped_name = entry.get(
                "mapped_class_name"
            )

            if mapped_name not in TARGET_CLASSES:
                raise ValueError(
                    f"Invalid mapped KITTI class: "
                    f"{mapped_name}"
                )

            counts[mapped_name] += 1

        elif action == "ignore":
            ignored_count += 1

        else:
            raise ValueError(
                f"Unknown KITTI mapping action "
                f"'{action}' for {source_class}."
            )

    target_count = int(
        sum(counts.values())
    )

    return {
        "target_box_count": target_count,
        "vehicle_count": int(
            counts["Vehicle"]
        ),
        "pedestrian_count": int(
            counts["Pedestrian"]
        ),
        "cyclist_count": int(
            counts["Cyclist"]
        ),
        "ignored_box_count": int(
            ignored_count
        ),
        "is_negative": (
            target_count == 0
        ),
    }


def build_kitti_records(
    issues: list[dict],
) -> list[dict]:
    mapping_config = load_yaml(
        KITTI_MAPPING_FILE
    )

    mapping = mapping_config.get(
        "kitti_mapping",
        {}
    )

    if not mapping:
        raise ValueError(
            "KITTI mapping file contains no "
            "kitti_mapping section."
        )

    train_ids = read_id_file(
        KITTI_TRAIN_IDS_FILE
    )

    val_ids = read_id_file(
        KITTI_VAL_IDS_FILE
    )

    partition_lookup = {
        **{
            image_id: "train"
            for image_id in train_ids
        },
        **{
            image_id: "val"
            for image_id in val_ids
        },
    }

    all_ids = sorted(
        partition_lookup.keys()
    )

    records: list[dict] = []

    print("\nBuilding KITTI source records...")

    for image_id in tqdm(
        all_ids,
        unit="image",
    ):
        partition = partition_lookup[
            image_id
        ]

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

        if not image_file.exists():
            add_issue(
                issues,
                "KITTI",
                "missing_image",
                image_id,
                str(image_file),
            )
            continue

        if not label_file.exists():
            add_issue(
                issues,
                "KITTI",
                "missing_annotation",
                image_id,
                str(label_file),
            )
            continue

        if not calibration_file.exists():
            add_issue(
                issues,
                "KITTI",
                "missing_calibration",
                image_id,
                str(calibration_file),
            )
            continue

        width, height = inspect_image(
            image_file
        )

        counts = inspect_kitti_label(
            label_file=label_file,
            mapping=mapping,
        )

        global_image_id = (
            int(image_id) + 1
        )

        output_filename = (
            f"{image_id}.png"
        )

        output_relative_path = (
            Path("images")
            / "kitti"
            / partition
            / output_filename
        )

        records.append(
            {
                "global_image_id": (
                    global_image_id
                ),
                "canonical_image_key": (
                    f"kitti_{image_id}"
                ),
                "dataset": "KITTI",
                "partition": partition,
                "experimental_role": (
                    "model_training"
                    if partition == "train"
                    else "in_domain_validation"
                ),

                "source_image_id": image_id,
                "source_image_path": (
                    project_relative_path(
                        image_file
                    )
                ),
                "source_annotation_path": (
                    project_relative_path(
                        label_file
                    )
                ),
                "source_annotation_key": (
                    image_id
                ),
                "source_calibration_path": (
                    project_relative_path(
                        calibration_file
                    )
                ),
                "source_raw_annotation_path": (
                    project_relative_path(
                        label_file
                    )
                ),

                "source_width": width,
                "source_height": height,
                "source_extension": (
                    image_file.suffix.lower()
                ),

                "output_filename": (
                    output_filename
                ),
                "output_relative_path": (
                    output_relative_path.as_posix()
                ),

                "target_box_count": (
                    counts[
                        "target_box_count"
                    ]
                ),
                "vehicle_count": (
                    counts["vehicle_count"]
                ),
                "pedestrian_count": (
                    counts[
                        "pedestrian_count"
                    ]
                ),
                "cyclist_count": (
                    counts["cyclist_count"]
                ),
                "ignored_box_count": (
                    counts[
                        "ignored_box_count"
                    ]
                ),
                "is_negative": boolean_text(
                    counts["is_negative"]
                ),

                "segment_id": "",
                "frame_timestamp_micros": "",
                "camera_name": (
                    "LEFT_COLOR_IMAGE_2"
                ),
                "time_of_day": "",
                "weather": "",
                "location": "",
                "density_group": "",
                "sampling_rule": (
                    "official_labeled_image"
                ),
            }
        )

    return records


# ============================================================
# WAYMO IGNORED SIGN COUNTS
# ============================================================

def build_waymo_sign_counts(
    manifest: pd.DataFrame,
    issues: list[dict],
) -> dict[str, int]:
    selected_by_segment: dict[
        str,
        set[int],
    ] = {}

    for segment_id, group in manifest.groupby(
        "segment_id"
    ):
        selected_by_segment[
            str(segment_id)
        ] = {
            int(value)
            for value in group[
                "frame_timestamp_micros"
            ].astype(str)
        }

    sign_counts: Counter = Counter()

    required_columns = [
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "[CameraBoxComponent].type",
    ]

    print(
        "\nCounting ignored Waymo Sign regions..."
    )

    for segment_id in tqdm(
        sorted(selected_by_segment),
        unit="segment",
    ):
        parquet_file = (
            WAYMO_RAW_CAMERA_BOX_DIR
            / f"{segment_id}.parquet"
        )

        if not parquet_file.exists():
            add_issue(
                issues,
                "Waymo",
                "missing_raw_camera_box_file",
                segment_id,
                str(parquet_file),
            )
            continue

        dataframe = pd.read_parquet(
            parquet_file,
            columns=required_columns,
        )

        selected_timestamps = (
            selected_by_segment[
                segment_id
            ]
        )

        filtered = dataframe[
            (
                dataframe[
                    "key.camera_name"
                ]
                == WAYMO_FRONT_CAMERA_ID
            )
            & (
                dataframe[
                    "key.frame_timestamp_micros"
                ].isin(
                    selected_timestamps
                )
            )
            & (
                dataframe[
                    "[CameraBoxComponent].type"
                ]
                == WAYMO_SIGN_TYPE_ID
            )
        ]

        for timestamp, count in (
            filtered.groupby(
                "key.frame_timestamp_micros"
            )
            .size()
            .items()
        ):
            image_id = (
                f"{segment_id}_{int(timestamp)}"
            )

            sign_counts[image_id] = int(
                count
            )

    return dict(sign_counts)


# ============================================================
# WAYMO MANIFEST
# ============================================================

def build_waymo_records(
    issues: list[dict],
) -> list[dict]:
    manifest = pd.read_csv(
        WAYMO_MANIFEST_FILE,
        dtype=str,
    )

    boxes = pd.read_csv(
        WAYMO_BOXES_FILE,
        dtype=str,
    )

    required_manifest_columns = [
        "image_id",
        "final_segment_number",
        "segment_id",
        "frame_timestamp_micros",
        "camera_name",
        "selected_frame_number_in_segment",
        "relative_image_path",
        "image_filename",
        "image_width",
        "image_height",
        "time_of_day",
        "weather",
        "location",
        "density_group",
        "sampling_rule",
    ]

    missing_manifest_columns = [
        column
        for column in required_manifest_columns
        if column not in manifest.columns
    ]

    if missing_manifest_columns:
        raise KeyError(
            "Waymo manifest is missing columns:\n"
            f"{missing_manifest_columns}"
        )

    boxes_image_id_column = resolve_column(
        boxes,
        [
            "image_id",
            "global_image_id",
        ],
        "Waymo box image ID",
    )

    boxes_class_column = resolve_column(
        boxes,
        [
            "class_name",
            "mapped_class_name",
            "category_name",
            "label",
        ],
        "Waymo target class",
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

    boxes[
        boxes_class_column
    ] = (
        boxes[
            boxes_class_column
        ]
        .astype(str)
        .str.strip()
    )

    grouped_counts = (
        boxes.groupby(
            [
                boxes_image_id_column,
                boxes_class_column,
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    manifest["final_segment_number"] = (
        pd.to_numeric(
            manifest[
                "final_segment_number"
            ],
            errors="raise",
        )
    )

    manifest[
        "selected_frame_number_in_segment"
    ] = pd.to_numeric(
        manifest[
            "selected_frame_number_in_segment"
        ],
        errors="raise",
    )

    manifest = (
        manifest.sort_values(
            [
                "final_segment_number",
                "selected_frame_number_in_segment",
                "image_id",
            ]
        )
        .reset_index(drop=True)
    )

    sign_counts = build_waymo_sign_counts(
        manifest=manifest,
        issues=issues,
    )

    records: list[dict] = []

    print("\nBuilding Waymo source records...")

    for row_index, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        unit="image",
    ):
        image_id = str(
            row["image_id"]
        ).strip()

        segment_id = str(
            row["segment_id"]
        ).strip()

        timestamp = str(
            row[
                "frame_timestamp_micros"
            ]
        ).strip()

        source_image_path = (
            WAYMO_SUBSET_ROOT
            / Path(
                str(
                    row[
                        "relative_image_path"
                    ]
                ).replace("\\", "/")
            )
        )

        raw_annotation_file = (
            WAYMO_RAW_CAMERA_BOX_DIR
            / f"{segment_id}.parquet"
        )

        if not source_image_path.exists():
            add_issue(
                issues,
                "Waymo",
                "missing_image",
                image_id,
                str(source_image_path),
            )
            continue

        width, height = inspect_image(
            source_image_path
        )

        recorded_width = int(
            row["image_width"]
        )

        recorded_height = int(
            row["image_height"]
        )

        if (
            width != recorded_width
            or height != recorded_height
        ):
            add_issue(
                issues,
                "Waymo",
                "dimension_mismatch",
                image_id,
                (
                    f"Manifest={recorded_width}x"
                    f"{recorded_height}, "
                    f"actual={width}x{height}"
                ),
            )

        if image_id in grouped_counts.index:
            class_row = grouped_counts.loc[
                image_id
            ]

            vehicle_count = int(
                class_row.get(
                    "Vehicle",
                    0,
                )
            )

            pedestrian_count = int(
                class_row.get(
                    "Pedestrian",
                    0,
                )
            )

            cyclist_count = int(
                class_row.get(
                    "Cyclist",
                    0,
                )
            )

        else:
            vehicle_count = 0
            pedestrian_count = 0
            cyclist_count = 0

        target_count = (
            vehicle_count
            + pedestrian_count
            + cyclist_count
        )

        ignored_count = int(
            sign_counts.get(
                image_id,
                0,
            )
        )

        global_image_id = (
            1_000_001 + row_index
        )

        output_filename = (
            f"{image_id}.png"
        )

        output_relative_path = (
            Path("images")
            / "waymo"
            / "external"
            / output_filename
        )

        records.append(
            {
                "global_image_id": (
                    global_image_id
                ),
                "canonical_image_key": (
                    f"waymo_{image_id}"
                ),
                "dataset": "Waymo",
                "partition": "external",
                "experimental_role": (
                    "external_validation_only"
                ),

                "source_image_id": image_id,
                "source_image_path": (
                    project_relative_path(
                        source_image_path
                    )
                ),
                "source_annotation_path": (
                    project_relative_path(
                        WAYMO_BOXES_FILE
                    )
                ),
                "source_annotation_key": (
                    image_id
                ),
                "source_calibration_path": "",
                "source_raw_annotation_path": (
                    project_relative_path(
                        raw_annotation_file
                    )
                ),

                "source_width": width,
                "source_height": height,
                "source_extension": (
                    source_image_path
                    .suffix
                    .lower()
                ),

                "output_filename": (
                    output_filename
                ),
                "output_relative_path": (
                    output_relative_path
                    .as_posix()
                ),

                "target_box_count": (
                    target_count
                ),
                "vehicle_count": (
                    vehicle_count
                ),
                "pedestrian_count": (
                    pedestrian_count
                ),
                "cyclist_count": (
                    cyclist_count
                ),
                "ignored_box_count": (
                    ignored_count
                ),
                "is_negative": boolean_text(
                    target_count == 0
                ),

                "segment_id": segment_id,
                "frame_timestamp_micros": (
                    timestamp
                ),
                "camera_name": str(
                    row["camera_name"]
                ),
                "time_of_day": str(
                    row["time_of_day"]
                ),
                "weather": str(
                    row["weather"]
                ),
                "location": str(
                    row["location"]
                ),
                "density_group": str(
                    row["density_group"]
                ),
                "sampling_rule": str(
                    row["sampling_rule"]
                ),
            }
        )

    return records


# ============================================================
# SUMMARY AND VALIDATION
# ============================================================

def partition_summary(
    dataframe: pd.DataFrame,
) -> dict:
    return {
        "images": int(
            len(dataframe)
        ),
        "target_boxes": int(
            dataframe[
                "target_box_count"
            ].sum()
        ),
        "vehicle_boxes": int(
            dataframe[
                "vehicle_count"
            ].sum()
        ),
        "pedestrian_boxes": int(
            dataframe[
                "pedestrian_count"
            ].sum()
        ),
        "cyclist_boxes": int(
            dataframe[
                "cyclist_count"
            ].sum()
        ),
        "ignored_boxes": int(
            dataframe[
                "ignored_box_count"
            ].sum()
        ),
        "negative_images": int(
            dataframe[
                "is_negative"
            ].astype(str)
            .str.lower()
            .eq("true")
            .sum()
        ),
    }


def validate_manifest(
    dataframe: pd.DataFrame,
    issues: list[dict],
) -> dict:
    checks: dict[str, bool] = {}

    checks["row_count"] = (
        len(dataframe)
        == EXPECTED["combined_images"]
    )

    checks["unique_global_image_ids"] = (
        not dataframe[
            "global_image_id"
        ].duplicated().any()
    )

    checks["unique_canonical_keys"] = (
        not dataframe[
            "canonical_image_key"
        ].duplicated().any()
    )

    checks["unique_output_paths"] = (
        not dataframe[
            "output_relative_path"
        ].duplicated().any()
    )

    checks["source_paths_exist"] = all(
        Path(path).exists()
        for path in dataframe[
            "source_image_path"
        ].tolist()
    )

    kitti_train = dataframe[
        (
            dataframe["dataset"]
            == "KITTI"
        )
        & (
            dataframe["partition"]
            == "train"
        )
    ]

    kitti_val = dataframe[
        (
            dataframe["dataset"]
            == "KITTI"
        )
        & (
            dataframe["partition"]
            == "val"
        )
    ]

    waymo_external = dataframe[
        dataframe["dataset"]
        == "Waymo"
    ]

    summaries = {
        "kitti_train": partition_summary(
            kitti_train
        ),
        "kitti_validation": partition_summary(
            kitti_val
        ),
        "waymo_external": partition_summary(
            waymo_external
        ),
        "combined": partition_summary(
            dataframe
        ),
    }

    expected_checks = {
        "kitti_train_images": (
            summaries[
                "kitti_train"
            ]["images"]
            == EXPECTED[
                "kitti_train_images"
            ]
        ),
        "kitti_val_images": (
            summaries[
                "kitti_validation"
            ]["images"]
            == EXPECTED[
                "kitti_val_images"
            ]
        ),
        "kitti_train_boxes": (
            summaries[
                "kitti_train"
            ]["target_boxes"]
            == EXPECTED[
                "kitti_train_target_boxes"
            ]
        ),
        "kitti_val_boxes": (
            summaries[
                "kitti_validation"
            ]["target_boxes"]
            == EXPECTED[
                "kitti_val_target_boxes"
            ]
        ),
        "waymo_images": (
            summaries[
                "waymo_external"
            ]["images"]
            == EXPECTED[
                "waymo_images"
            ]
        ),
        "waymo_boxes": (
            summaries[
                "waymo_external"
            ]["target_boxes"]
            == EXPECTED[
                "waymo_target_boxes"
            ]
        ),
        "waymo_negative_images": (
            summaries[
                "waymo_external"
            ]["negative_images"]
            == EXPECTED[
                "waymo_negative_images"
            ]
        ),
        "combined_target_boxes": (
            summaries[
                "combined"
            ]["target_boxes"]
            == EXPECTED[
                "combined_target_boxes"
            ]
        ),
        "combined_vehicle_boxes": (
            summaries[
                "combined"
            ]["vehicle_boxes"]
            == EXPECTED[
                "vehicle_boxes"
            ]
        ),
        "combined_pedestrian_boxes": (
            summaries[
                "combined"
            ]["pedestrian_boxes"]
            == EXPECTED[
                "pedestrian_boxes"
            ]
        ),
        "combined_cyclist_boxes": (
            summaries[
                "combined"
            ]["cyclist_boxes"]
            == EXPECTED[
                "cyclist_boxes"
            ]
        ),
    }

    checks.update(
        expected_checks
    )

    for check_name, passed in checks.items():
        if not passed:
            add_issue(
                issues,
                "Combined",
                "manifest_validation_failed",
                check_name,
                "The manifest check returned false.",
            )

    return {
        "validation_passed": all(
            checks.values()
        ),
        "checks": checks,
        "summaries": summaries,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []

    print("=" * 76)
    print("BUILDING MILESTONE 3 SOURCE MANIFEST")
    print("=" * 76)

    preprocessing_config = load_yaml(
        PREPROCESSING_CONFIG
    )

    load_yaml(
        MILESTONE_3_MAPPING_CONFIG
    )

    validation_report = load_json(
        SOURCE_VALIDATION_REPORT
    )

    if not validation_report.get(
        "validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 2 source validation has not passed. "
            "Do not build the source manifest."
        )

    filename_policy = (
        preprocessing_config[
            "image_preprocessing"
        ]["filename_policy"]
    )

    if (
        filename_policy[
            "kitti"
        ]["method"]
        != "preserve_source_stem"
    ):
        raise ValueError(
            "Unexpected KITTI filename policy."
        )

    if (
        filename_policy[
            "waymo"
        ]["method"]
        != "use_manifest_image_id"
    ):
        raise ValueError(
            "Unexpected Waymo filename policy."
        )

    kitti_records = build_kitti_records(
        issues
    )

    waymo_records = build_waymo_records(
        issues
    )

    all_records = (
        kitti_records
        + waymo_records
    )

    dataframe = pd.DataFrame(
        all_records,
        columns=MANIFEST_COLUMNS,
    )

    numeric_columns = [
        "global_image_id",
        "source_width",
        "source_height",
        "target_box_count",
        "vehicle_count",
        "pedestrian_count",
        "cyclist_count",
        "ignored_box_count",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe = (
        dataframe.sort_values(
            "global_image_id"
        )
        .reset_index(drop=True)
    )

    validation = validate_manifest(
        dataframe=dataframe,
        issues=issues,
    )

    overall_passed = (
        validation[
            "validation_passed"
        ]
        and len(issues) == 0
    )

    dataframe.to_csv(
        SOURCE_MANIFEST_FILE,
        index=False,
    )

    summary = {
        "milestone": 3,
        "step": 3,
        "purpose": (
            "Create one canonical source record "
            "for every KITTI and Waymo image."
        ),
        "global_id_policy": {
            "KITTI": (
                "integer KITTI image ID plus one"
            ),
            "Waymo": (
                "1000001 plus deterministic "
                "sorted manifest row index"
            ),
        },
        "filename_policy": {
            "KITTI": (
                "preserve six-digit source stem"
            ),
            "Waymo": (
                "use full manifest image_id"
            ),
        },
        "manifest_rows": int(
            len(dataframe)
        ),
        "issues": len(issues),
        **validation,
        "source_manifest_passed": (
            overall_passed
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
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
    print("SOURCE MANIFEST SUMMARY")
    print("=" * 76)

    summaries = validation[
        "summaries"
    ]

    for name in [
        "kitti_train",
        "kitti_validation",
        "waymo_external",
        "combined",
    ]:
        values = summaries[name]

        print(f"\n{name}:")

        print(
            f"  Images: "
            f"{values['images']}"
        )

        print(
            f"  Target boxes: "
            f"{values['target_boxes']}"
        )

        print(
            f"  Ignored boxes: "
            f"{values['ignored_boxes']}"
        )

        print(
            f"  Negative images: "
            f"{values['negative_images']}"
        )

    print(
        f"\nManifest rows: "
        f"{len(dataframe)}"
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
        f"\nManifest:\n"
        f"{SOURCE_MANIFEST_FILE.resolve()}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_FILE.resolve()}"
    )

    print(
        f"\nIssues:\n"
        f"{ISSUES_FILE.resolve()}"
    )

    if not overall_passed:
        print(
            "\nDo not continue to image "
            "preprocessing until the source "
            "manifest issues are resolved."
        )

        sys.exit(1)

    print(
        "\nStep 3 completed successfully."
    )


if __name__ == "__main__":
    main()