from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import sys

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

PREPROCESSING_CONFIG = Path(
    "configs/datasets/milestone_3/preprocessing.yaml"
)

CLASS_MAPPING_CONFIG = Path(
    "configs/datasets/milestone_3/class_mapping.yaml"
)

SOURCE_MANIFEST_FILE = Path(
    "data/processed/milestone_3/manifests/source_manifest.csv"
)

TRANSFORM_MANIFEST_FILE = Path(
    "data/processed/milestone_3/manifests/transform_manifest.csv"
)

COCO_CREATION_REPORT = Path(
    "data/processed/milestone_3/reports/"
    "coco_creation_report.json"
)

WAYMO_MANIFEST_FILE = Path(
    "data/waymo/representative_subset/"
    "metadata/manifest.csv"
)

WAYMO_RAW_BOX_DIR = Path(
    "data/waymo/raw/validation/"
    "camera_box/candidates"
)

IGNORE_OUTPUT_DIR = Path(
    "data/processed/milestone_3/"
    "annotations/ignore_regions"
)

EXCLUDED_OUTPUT_DIR = Path(
    "data/processed/milestone_3/"
    "annotations/excluded_objects"
)

REPORT_DIR = Path(
    "data/processed/milestone_3/reports"
)

MANIFEST_DIR = Path(
    "data/processed/milestone_3/manifests"
)

REPORT_FILE = (
    REPORT_DIR / "region_policy_report.json"
)

ISSUES_FILE = (
    REPORT_DIR / "region_policy_issues.csv"
)

WAYMO_TYPE_AUDIT_FILE = (
    REPORT_DIR
    / "waymo_selected_type_distribution.csv"
)

REGION_POLICY_MANIFEST_FILE = (
    MANIFEST_DIR
    / "region_policy_manifest.csv"
)


PARTITIONS = [
    "kitti_train",
    "kitti_val",
    "waymo_external",
]


IGNORE_OUTPUTS = {
    "kitti_train": (
        IGNORE_OUTPUT_DIR
        / "kitti_train_ignore.json"
    ),
    "kitti_val": (
        IGNORE_OUTPUT_DIR
        / "kitti_val_ignore.json"
    ),
    "waymo_external": (
        IGNORE_OUTPUT_DIR
        / "waymo_external_ignore.json"
    ),
}


EXCLUDED_OUTPUTS = {
    "kitti_train": (
        EXCLUDED_OUTPUT_DIR
        / "kitti_train_excluded.json"
    ),
    "kitti_val": (
        EXCLUDED_OUTPUT_DIR
        / "kitti_val_excluded.json"
    ),
    "waymo_external": (
        EXCLUDED_OUTPUT_DIR
        / "waymo_external_excluded.json"
    ),
}


# Waymo CameraBox enum.
WAYMO_TYPE_NAMES = {
    0: "Unknown",
    1: "Vehicle",
    2: "Pedestrian",
    3: "Sign",
    4: "Cyclist",
}

WAYMO_FRONT_CAMERA_ID = 1


EXPECTED = {
    "total_images": 8477,

    "kitti_dontcare": 11295,
    "kitti_tram": 511,
    "kitti_misc": 973,

    "waymo_vehicle": 16928,
    "waymo_pedestrian": 7127,
    "waymo_cyclist": 764,

    "waymo_target_total": 24819,
}


COORDINATE_PRECISION = 10


# ============================================================
# HELPERS
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
        f"Unsupported partition: "
        f"{dataset}/{partition}"
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


def convert_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    transform: LetterboxTransform,
) -> tuple[list[float], float, bool]:
    values = [
        xmin,
        ymin,
        xmax,
        ymax,
    ]

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Box contains non-finite coordinates."
        )

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            "Box has non-positive dimensions."
        )

    (
        transformed_xmin,
        transformed_ymin,
        transformed_xmax,
        transformed_ymax,
    ) = transform_xyxy(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        transform=transform,
    )

    clipped_xmin = min(
        max(transformed_xmin, 0.0),
        float(transform.target_width),
    )

    clipped_ymin = min(
        max(transformed_ymin, 0.0),
        float(transform.target_height),
    )

    clipped_xmax = min(
        max(transformed_xmax, 0.0),
        float(transform.target_width),
    )

    clipped_ymax = min(
        max(transformed_ymax, 0.0),
        float(transform.target_height),
    )

    width = (
        clipped_xmax - clipped_xmin
    )

    height = (
        clipped_ymax - clipped_ymin
    )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Transformed region has "
            "non-positive dimensions."
        )

    bbox = [
        round(
            clipped_xmin,
            COORDINATE_PRECISION,
        ),
        round(
            clipped_ymin,
            COORDINATE_PRECISION,
        ),
        round(
            width,
            COORDINATE_PRECISION,
        ),
        round(
            height,
            COORDINATE_PRECISION,
        ),
    ]

    area = round(
        bbox[2] * bbox[3],
        COORDINATE_PRECISION,
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

    return (
        bbox,
        area,
        clipping_applied,
    )


def empty_sidecar(
    partition_name: str,
    policy_name: str,
    classes: list[str],
) -> dict:
    return {
        "info": {
            "milestone": 3,
            "partition": partition_name,
            "policy": policy_name,
            "classes": classes,
            "coordinate_format": (
                "xywh_absolute_float"
            ),
            "image_size": [
                640,
                640,
            ],
        },
        "images": [],
        "regions": [],
    }


def image_record(
    source_row: pd.Series,
    transform_row: pd.Series,
) -> dict:
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
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    IGNORE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXCLUDED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print("CREATING REGION-POLICY SIDECARS")
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

    if not coco_report.get(
        "coco_creation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 6 COCO creation has not passed."
        )

    policy = (
        preprocessing_config[
            "annotation_policy"
        ]["region_policy"]
    )

    expected_ignore_kitti = set(
        policy[
            "evaluation_ignore"
        ]["kitti_classes"]
    )

    expected_ignore_waymo = set(
        policy[
            "evaluation_ignore"
        ]["waymo_classes"]
    )

    expected_excluded_kitti = set(
        policy[
            "excluded_non_target_objects"
        ]["kitti_classes"]
    )

    expected_excluded_waymo = set(
        policy[
            "excluded_non_target_objects"
        ]["waymo_classes"]
    )

    if expected_ignore_kitti != {
        "DontCare"
    }:
        raise ValueError(
            "Frozen KITTI evaluation-ignore "
            "policy must contain only DontCare."
        )

    if expected_ignore_waymo:
        raise ValueError(
            "Waymo evaluation-ignore policy "
            "must be empty."
        )

    if expected_excluded_kitti != {
        "Tram",
        "Misc",
    }:
        raise ValueError(
            "Unexpected KITTI excluded-object policy."
        )

    if expected_excluded_waymo != {
        "Sign"
    }:
        raise ValueError(
            "Unexpected Waymo excluded-object policy."
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

    transform_lookup = (
        transform_manifest.set_index(
            "global_image_id",
            drop=False,
        )
    )

    source_lookup_by_source_id = {
        str(row["source_image_id"]): row
        for _, row
        in source_manifest.iterrows()
    }

    sidecars = {}

    for partition_name in PARTITIONS:
        sidecars[
            (
                partition_name,
                "evaluation_ignore",
            )
        ] = empty_sidecar(
            partition_name,
            "evaluation_ignore",
            (
                ["DontCare"]
                if partition_name.startswith(
                    "kitti"
                )
                else []
            ),
        )

        sidecars[
            (
                partition_name,
                "excluded_non_target",
            )
        ] = empty_sidecar(
            partition_name,
            "excluded_non_target",
            (
                [
                    "Tram",
                    "Misc",
                ]
                if partition_name.startswith(
                    "kitti"
                )
                else ["Sign"]
            ),
        )

    region_id_counters = defaultdict(
        lambda: 1
    )

    per_image_counts = defaultdict(
        lambda: {
            "evaluation_ignore_count": 0,
            "excluded_non_target_count": 0,
        }
    )

    for _, source_row in (
        source_manifest.sort_values(
            "global_image_id"
        ).iterrows()
    ):
        global_image_id = int(
            source_row[
                "global_image_id"
            ]
        )

        partition_name = partition_key(
            str(source_row["dataset"]),
            str(source_row["partition"]),
        )

        transform_row = (
            transform_lookup.loc[
                global_image_id
            ]
        )

        record = image_record(
            source_row,
            transform_row,
        )

        sidecars[
            (
                partition_name,
                "evaluation_ignore",
            )
        ]["images"].append(record)

        sidecars[
            (
                partition_name,
                "excluded_non_target",
            )
        ]["images"].append(record)

    class_counts = Counter()
    clipping_counts = Counter()

    # --------------------------------------------------------
    # KITTI regions
    # --------------------------------------------------------

    kitti_rows = source_manifest[
        source_manifest["dataset"]
        == "KITTI"
    ]

    print("\nExtracting KITTI region policies...")

    for _, source_row in tqdm(
        kitti_rows.iterrows(),
        total=len(kitti_rows),
        unit="image",
    ):
        global_image_id = int(
            source_row[
                "global_image_id"
            ]
        )

        partition_name = partition_key(
            "KITTI",
            str(source_row["partition"]),
        )

        transform = create_transform(
            transform_lookup.loc[
                global_image_id
            ]
        )

        label_file = Path(
            str(
                source_row[
                    "source_annotation_path"
                ]
            )
        )

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
                add_issue(
                    issues,
                    "KITTI",
                    "invalid_label_row",
                    (
                        f"{source_row['source_image_id']}:"
                        f"{line_number}"
                    ),
                    (
                        f"Expected 15 or 16 fields, "
                        f"found {len(fields)}."
                    ),
                )
                continue

            source_class = fields[0]

            if source_class == "DontCare":
                policy_name = (
                    "evaluation_ignore"
                )

            elif source_class in {
                "Tram",
                "Misc",
            }:
                policy_name = (
                    "excluded_non_target"
                )

            else:
                continue

            try:
                xmin = float(fields[4])
                ymin = float(fields[5])
                xmax = float(fields[6])
                ymax = float(fields[7])

                (
                    bbox,
                    area,
                    clipping_applied,
                ) = convert_box(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    transform,
                )

            except Exception as error:
                add_issue(
                    issues,
                    "KITTI",
                    "region_conversion_failed",
                    (
                        f"{source_row['source_image_id']}:"
                        f"{line_number}"
                    ),
                    str(error),
                )
                continue

            key = (
                partition_name,
                policy_name,
            )

            region_id = (
                region_id_counters[key]
            )

            region_id_counters[key] += 1

            sidecars[key]["regions"].append(
                {
                    "id": region_id,
                    "image_id": (
                        global_image_id
                    ),
                    "source_class": (
                        source_class
                    ),
                    "bbox": bbox,
                    "area": area,
                    "clipping_applied": (
                        clipping_applied
                    ),
                }
            )

            class_counts[
                (
                    partition_name,
                    source_class,
                )
            ] += 1

            clipping_counts[key] += int(
                clipping_applied
            )

            per_image_counts[
                global_image_id
            ][
                (
                    "evaluation_ignore_count"
                    if policy_name
                    == "evaluation_ignore"
                    else
                    "excluded_non_target_count"
                )
            ] += 1

    # --------------------------------------------------------
    # Waymo raw type audit and Sign sidecars
    # --------------------------------------------------------

    waymo_manifest = pd.read_csv(
        WAYMO_MANIFEST_FILE,
        dtype=str,
    )

    selected_timestamps = {}

    for segment_id, group in (
        waymo_manifest.groupby(
            "segment_id"
        )
    ):
        selected_timestamps[
            str(segment_id)
        ] = {
            int(value)
            for value in group[
                "frame_timestamp_micros"
            ].astype(str)
        }

    waymo_type_counts = Counter()

    required_columns = [
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "key.camera_object_id",
        "[CameraBoxComponent].box.center.x",
        "[CameraBoxComponent].box.center.y",
        "[CameraBoxComponent].box.size.x",
        "[CameraBoxComponent].box.size.y",
        "[CameraBoxComponent].type",
    ]

    print("\nAuditing Waymo selected raw types...")

    for segment_id in tqdm(
        sorted(selected_timestamps),
        unit="segment",
    ):
        parquet_file = (
            WAYMO_RAW_BOX_DIR
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

        dataframe = dataframe[
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
                    selected_timestamps[
                        segment_id
                    ]
                )
            )
        ]

        for row_number, row in (
            dataframe.iterrows()
        ):
            type_id = int(
                row[
                    "[CameraBoxComponent].type"
                ]
            )

            type_name = (
                WAYMO_TYPE_NAMES.get(
                    type_id
                )
            )

            if type_name is None:
                add_issue(
                    issues,
                    "Waymo",
                    "unknown_camera_box_type",
                    str(type_id),
                    (
                        f"Segment={segment_id}, "
                        f"row={row_number}"
                    ),
                )
                continue

            waymo_type_counts[
                type_name
            ] += 1

            if type_name != "Sign":
                continue

            timestamp = int(
                row[
                    "key.frame_timestamp_micros"
                ]
            )

            source_image_id = (
                f"{segment_id}_{timestamp}"
            )

            source_row = (
                source_lookup_by_source_id.get(
                    source_image_id
                )
            )

            if source_row is None:
                add_issue(
                    issues,
                    "Waymo",
                    "sign_for_unknown_image",
                    source_image_id,
                    (
                        "Selected raw Sign row has no "
                        "source-manifest image."
                    ),
                )
                continue

            global_image_id = int(
                source_row[
                    "global_image_id"
                ]
            )

            transform = create_transform(
                transform_lookup.loc[
                    global_image_id
                ]
            )

            center_x = float(
                row[
                    "[CameraBoxComponent].box.center.x"
                ]
            )

            center_y = float(
                row[
                    "[CameraBoxComponent].box.center.y"
                ]
            )

            size_x = float(
                row[
                    "[CameraBoxComponent].box.size.x"
                ]
            )

            size_y = float(
                row[
                    "[CameraBoxComponent].box.size.y"
                ]
            )

            xmin = center_x - size_x / 2.0
            ymin = center_y - size_y / 2.0
            xmax = center_x + size_x / 2.0
            ymax = center_y + size_y / 2.0

            try:
                (
                    bbox,
                    area,
                    clipping_applied,
                ) = convert_box(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    transform,
                )

            except Exception as error:
                add_issue(
                    issues,
                    "Waymo",
                    "sign_region_conversion_failed",
                    source_image_id,
                    str(error),
                )
                continue

            key = (
                "waymo_external",
                "excluded_non_target",
            )

            region_id = (
                region_id_counters[key]
            )

            region_id_counters[key] += 1

            sidecars[key]["regions"].append(
                {
                    "id": region_id,
                    "image_id": (
                        global_image_id
                    ),
                    "source_class": "Sign",
                    "source_object_id": str(
                        row[
                            "key.camera_object_id"
                        ]
                    ),
                    "bbox": bbox,
                    "area": area,
                    "clipping_applied": (
                        clipping_applied
                    ),
                }
            )

            class_counts[
                (
                    "waymo_external",
                    "Sign",
                )
            ] += 1

            clipping_counts[key] += int(
                clipping_applied
            )

            per_image_counts[
                global_image_id
            ][
                "excluded_non_target_count"
            ] += 1

    # --------------------------------------------------------
    # Validate raw Waymo target totals
    # --------------------------------------------------------

    waymo_target_total = (
        waymo_type_counts["Vehicle"]
        + waymo_type_counts["Pedestrian"]
        + waymo_type_counts["Cyclist"]
    )

    checks = {
        "source_image_count": (
            len(source_manifest)
            == EXPECTED["total_images"]
        ),

        "kitti_dontcare_count": (
            sum(
                count
                for (
                    partition,
                    source_class,
                ), count
                in class_counts.items()
                if source_class == "DontCare"
            )
            == EXPECTED["kitti_dontcare"]
        ),

        "kitti_tram_count": (
            sum(
                count
                for (
                    partition,
                    source_class,
                ), count
                in class_counts.items()
                if source_class == "Tram"
            )
            == EXPECTED["kitti_tram"]
        ),

        "kitti_misc_count": (
            sum(
                count
                for (
                    partition,
                    source_class,
                ), count
                in class_counts.items()
                if source_class == "Misc"
            )
            == EXPECTED["kitti_misc"]
        ),

        "waymo_vehicle_audit": (
            waymo_type_counts["Vehicle"]
            == EXPECTED["waymo_vehicle"]
        ),

        "waymo_pedestrian_audit": (
            waymo_type_counts["Pedestrian"]
            == EXPECTED["waymo_pedestrian"]
        ),

        "waymo_cyclist_audit": (
            waymo_type_counts["Cyclist"]
            == EXPECTED["waymo_cyclist"]
        ),

        "waymo_target_total_audit": (
            waymo_target_total
            == EXPECTED["waymo_target_total"]
        ),

        "waymo_evaluation_ignore_empty": (
            len(
                sidecars[
                    (
                        "waymo_external",
                        "evaluation_ignore",
                    )
                ]["regions"]
            )
            == 0
        ),
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "Combined",
                "region_policy_check_failed",
                check_name,
                "The region-policy check returned false.",
            )

    # --------------------------------------------------------
    # Save sidecars
    # --------------------------------------------------------

    for (
        partition_name,
        policy_name,
    ), sidecar in sidecars.items():
        sidecar["images"].sort(
            key=lambda item: int(
                item["id"]
            )
        )

        sidecar["regions"].sort(
            key=lambda item: int(
                item["id"]
            )
        )

        sidecar["summary"] = {
            "images": len(
                sidecar["images"]
            ),
            "regions": len(
                sidecar["regions"]
            ),
            "clipped_regions": int(
                clipping_counts[
                    (
                        partition_name,
                        policy_name,
                    )
                ]
            ),
        }

        if policy_name == (
            "evaluation_ignore"
        ):
            output_file = (
                IGNORE_OUTPUTS[
                    partition_name
                ]
            )
        else:
            output_file = (
                EXCLUDED_OUTPUTS[
                    partition_name
                ]
            )

        output_file.write_text(
            json.dumps(
                sidecar,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # --------------------------------------------------------
    # Per-image region-policy manifest
    # --------------------------------------------------------

    policy_manifest_rows = []

    for _, source_row in (
        source_manifest.sort_values(
            "global_image_id"
        ).iterrows()
    ):
        global_image_id = int(
            source_row[
                "global_image_id"
            ]
        )

        counts = per_image_counts[
            global_image_id
        ]

        policy_manifest_rows.append(
            {
                "global_image_id": (
                    global_image_id
                ),
                "canonical_image_key": (
                    source_row[
                        "canonical_image_key"
                    ]
                ),
                "dataset": (
                    source_row["dataset"]
                ),
                "partition": (
                    source_row["partition"]
                ),
                "source_image_id": (
                    source_row[
                        "source_image_id"
                    ]
                ),
                "evaluation_ignore_count": (
                    counts[
                        "evaluation_ignore_count"
                    ]
                ),
                "excluded_non_target_count": (
                    counts[
                        "excluded_non_target_count"
                    ]
                ),
                "total_region_policy_count": (
                    counts[
                        "evaluation_ignore_count"
                    ]
                    + counts[
                        "excluded_non_target_count"
                    ]
                ),
            }
        )

    write_csv(
        REGION_POLICY_MANIFEST_FILE,
        policy_manifest_rows,
        [
            "global_image_id",
            "canonical_image_key",
            "dataset",
            "partition",
            "source_image_id",
            "evaluation_ignore_count",
            "excluded_non_target_count",
            "total_region_policy_count",
        ],
    )

    type_audit_rows = []

    for type_id, type_name in (
        WAYMO_TYPE_NAMES.items()
    ):
        type_audit_rows.append(
            {
                "type_id": type_id,
                "type_name": type_name,
                "selected_front_box_count": int(
                    waymo_type_counts[
                        type_name
                    ]
                ),
                "policy": (
                    "target"
                    if type_name in {
                        "Vehicle",
                        "Pedestrian",
                        "Cyclist",
                    }
                    else (
                        "excluded_non_target"
                        if type_name == "Sign"
                        else "unknown"
                    )
                ),
            }
        )

    write_csv(
        WAYMO_TYPE_AUDIT_FILE,
        type_audit_rows,
        [
            "type_id",
            "type_name",
            "selected_front_box_count",
            "policy",
        ],
    )

    total_evaluation_ignore = sum(
        len(
            sidecars[
                (
                    partition,
                    "evaluation_ignore",
                )
            ]["regions"]
        )
        for partition in PARTITIONS
    )

    total_excluded = sum(
        len(
            sidecars[
                (
                    partition,
                    "excluded_non_target",
                )
            ]["regions"]
        )
        for partition in PARTITIONS
    )

    overall_passed = (
        all(checks.values())
        and len(issues) == 0
    )

    report = {
        "milestone": 3,
        "step": 7,
        "purpose": (
            "Separate true evaluation-ignore "
            "regions from excluded non-target "
            "objects."
        ),
        "policy": {
            "evaluation_ignore": {
                "KITTI": [
                    "DontCare"
                ],
                "Waymo": [],
                "suppresses_false_positives": (
                    True
                ),
            },
            "excluded_non_target": {
                "KITTI": [
                    "Tram",
                    "Misc",
                ],
                "Waymo": [
                    "Sign",
                ],
                "suppresses_false_positives": (
                    False
                ),
            },
        },
        "class_counts": {
            (
                f"{partition}:{source_class}"
            ): int(count)
            for (
                partition,
                source_class,
            ), count
            in sorted(
                class_counts.items()
            )
        },
        "waymo_selected_type_counts": {
            type_name: int(
                waymo_type_counts[
                    type_name
                ]
            )
            for type_name in (
                WAYMO_TYPE_NAMES.values()
            )
        },
        "waymo_target_total": int(
            waymo_target_total
        ),
        "total_evaluation_ignore_regions": int(
            total_evaluation_ignore
        ),
        "total_excluded_non_target_regions": int(
            total_excluded
        ),
        "checks": checks,
        "issue_count": len(issues),
        "region_policy_passed": (
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
    print("REGION-POLICY SUMMARY")
    print("=" * 76)

    print("\nKITTI source classes:")

    for source_class in [
        "DontCare",
        "Tram",
        "Misc",
    ]:
        count = sum(
            value
            for (
                partition,
                class_name,
            ), value
            in class_counts.items()
            if class_name == source_class
        )

        print(
            f"  {source_class}: {count}"
        )

    print("\nWaymo selected FRONT types:")

    for type_name in [
        "Vehicle",
        "Pedestrian",
        "Cyclist",
        "Sign",
        "Unknown",
    ]:
        print(
            f"  {type_name}: "
            f"{waymo_type_counts[type_name]}"
        )

    print(
        f"\nEvaluation-ignore regions: "
        f"{total_evaluation_ignore}"
    )

    print(
        f"Excluded non-target regions: "
        f"{total_excluded}"
    )

    print(
        f"Issues found: {len(issues)}"
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
        f"\nRegion-policy manifest:\n"
        f"{REGION_POLICY_MANIFEST_FILE.resolve()}"
    )

    print(
        f"\nWaymo type audit:\n"
        f"{WAYMO_TYPE_AUDIT_FILE.resolve()}"
    )

    if not overall_passed:
        print(
            "\nDo not continue until every "
            "region-policy issue is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 7 completed successfully."
    )


if __name__ == "__main__":
    main()