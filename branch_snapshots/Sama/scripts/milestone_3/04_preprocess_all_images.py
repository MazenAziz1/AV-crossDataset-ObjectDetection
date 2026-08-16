from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing_core import (
    apply_letterbox_bgr,
    calculate_aspect_ratio_error,
    calculate_letterbox_transform,
    maximum_allowed_aspect_ratio_error,
    verify_padding,
)


# ============================================================
# PATHS
# ============================================================

PREPROCESSING_CONFIG = Path(
    "configs/datasets/milestone_3/preprocessing.yaml"
)

SOURCE_MANIFEST_FILE = Path(
    "data/processed/milestone_3/manifests/source_manifest.csv"
)

SOURCE_MANIFEST_SUMMARY = Path(
    "data/processed/milestone_3/reports/source_manifest_summary.json"
)

DRY_RUN_REPORT = Path(
    "data/processed/milestone_3/reports/preprocessing_dry_run.json"
)

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

TRANSFORM_MANIFEST_FILE = Path(
    "data/processed/milestone_3/manifests/transform_manifest.csv"
)

PARTIAL_MANIFEST_FILE = Path(
    "data/processed/milestone_3/manifests/"
    "transform_manifest.partial.csv"
)

REPORT_FILE = Path(
    "data/processed/milestone_3/reports/"
    "image_preprocessing_report.json"
)

ISSUES_FILE = Path(
    "data/processed/milestone_3/reports/"
    "image_preprocessing_issues.csv"
)


OUTPUT_DIRECTORIES = [
    PROCESSED_ROOT
    / "images/kitti/train",

    PROCESSED_ROOT
    / "images/kitti/val",

    PROCESSED_ROOT
    / "images/waymo/external",
]


EXPECTED = {
    "total_images": 8477,
    "kitti_train": 5985,
    "kitti_val": 1496,
    "waymo_external": 996,
}


TRANSFORM_COLUMNS = [
    "global_image_id",
    "canonical_image_key",
    "dataset",
    "partition",
    "experimental_role",

    "source_image_id",
    "source_image_path",
    "output_filename",
    "output_relative_path",
    "output_absolute_path",

    "source_width",
    "source_height",
    "target_width",
    "target_height",

    "nominal_scale",
    "actual_scale_x",
    "actual_scale_y",

    "resized_width",
    "resized_height",

    "padding_left",
    "padding_top",
    "padding_right",
    "padding_bottom",
    "padding_value",

    "interpolation",
    "source_extension",
    "output_extension",
    "png_compression_level",

    "target_box_count",
    "vehicle_count",
    "pedestrian_count",
    "cyclist_count",
    "ignored_box_count",
    "is_negative",

    "output_size_bytes",

    "dimensions_valid",
    "padding_valid",
    "geometry_valid",
    "aspect_ratio_error",
    "aspect_ratio_valid",

    "processing_status",
]


# ============================================================
# HELPERS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the complete KITTI and Waymo "
            "Milestone 3 image collection."
        )
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete existing final processed PNG files "
            "before starting."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rewrite existing expected processed images."
        ),
    )

    return parser.parse_args()


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


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found:\n"
            f"{path.resolve()}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
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


def add_issue(
    issues: list[dict],
    category: str,
    identifier: str,
    details: str,
) -> None:
    issues.append(
        {
            "category": category,
            "identifier": identifier,
            "details": details,
        }
    )


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


def parse_boolean(value) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def project_relative_path(path: Path) -> str:
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


def safely_clean_output_directories() -> int:
    removed = 0
    processed_root_resolved = (
        PROCESSED_ROOT.resolve()
    )

    for directory in OUTPUT_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        directory_resolved = (
            directory.resolve()
        )

        if processed_root_resolved not in (
            directory_resolved,
            *directory_resolved.parents,
        ):
            raise RuntimeError(
                f"Unsafe output directory:\n"
                f"{directory_resolved}"
            )

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            if (
                path.suffix.lower() == ".png"
                or path.name.endswith(".tmp")
            ):
                path.unlink()
                removed += 1

    return removed


def encode_and_write_png_atomic(
    output_path: Path,
    image: np.ndarray,
    compression_level: int,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    succeeded, encoded = cv2.imencode(
        ".png",
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            int(compression_level),
        ],
    )

    if not succeeded:
        raise RuntimeError(
            "OpenCV failed to encode the PNG."
        )

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    try:
        temporary_path.write_bytes(
            encoded.tobytes()
        )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_saved_image(
    output_path: Path,
    target_width: int,
    target_height: int,
    transform,
    padding_value: int,
) -> tuple[bool, bool]:
    image = cv2.imread(
        str(output_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return False, False

    height, width = image.shape[:2]

    dimensions_valid = (
        width == target_width
        and height == target_height
        and image.ndim == 3
        and image.shape[2] == 3
    )

    padding_valid = (
        verify_padding(
            image=image,
            transform=transform,
            padding_value=padding_value,
        )
        if dimensions_valid
        else False
    )

    return (
        dimensions_valid,
        padding_valid,
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

    return f"{dataset}_{partition}"


def save_partial_manifest(
    records: list[dict],
) -> None:
    if not records:
        return

    dataframe = pd.DataFrame(
        records,
        columns=TRANSFORM_COLUMNS,
    )

    dataframe.to_csv(
        PARTIAL_MANIFEST_FILE,
        index=False,
    )


# ============================================================
# PROCESSING
# ============================================================

def process_one_image(
    row,
    target_width: int,
    target_height: int,
    padding_value: int,
    compression_level: int,
    overwrite: bool,
) -> dict:
    source_path = Path(
        str(row.source_image_path)
    )

    output_relative_path = Path(
        str(row.output_relative_path)
    )

    output_path = (
        PROCESSED_ROOT
        / output_relative_path
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source image not found: {source_path}"
        )

    source_image = cv2.imread(
        str(source_path),
        cv2.IMREAD_COLOR,
    )

    if source_image is None:
        raise ValueError(
            f"Could not decode source image: {source_path}"
        )

    actual_source_height, actual_source_width = (
        source_image.shape[:2]
    )

    expected_source_width = int(
        row.source_width
    )

    expected_source_height = int(
        row.source_height
    )

    if (
        actual_source_width
        != expected_source_width
        or actual_source_height
        != expected_source_height
    ):
        raise ValueError(
            "Source dimensions differ from the source manifest. "
            f"Expected {expected_source_width}x"
            f"{expected_source_height}, found "
            f"{actual_source_width}x"
            f"{actual_source_height}."
        )

    transform = (
        calculate_letterbox_transform(
            source_width=(
                actual_source_width
            ),
            source_height=(
                actual_source_height
            ),
            target_width=target_width,
            target_height=target_height,
        )
    )

    geometry_valid = (
        transform.resized_width
        + transform.padding_left
        + transform.padding_right
        == target_width
        and transform.resized_height
        + transform.padding_top
        + transform.padding_bottom
        == target_height
    )

    aspect_ratio_error = (
        calculate_aspect_ratio_error(
            transform
        )
    )

    aspect_ratio_valid = (
        aspect_ratio_error
        <= maximum_allowed_aspect_ratio_error(
            transform
        )
    )

    processing_status = "created"

    should_write = (
        overwrite
        or not output_path.exists()
    )

    if not should_write:
        (
            dimensions_valid,
            padding_valid,
        ) = validate_saved_image(
            output_path=output_path,
            target_width=target_width,
            target_height=target_height,
            transform=transform,
            padding_value=padding_value,
        )

        if (
            dimensions_valid
            and padding_valid
        ):
            processing_status = (
                "reused_existing"
            )

        else:
            should_write = True
            processing_status = (
                "rewritten_invalid_existing"
            )

    if should_write:
        (
            processed_image,
            interpolation_name,
        ) = apply_letterbox_bgr(
            source_image=source_image,
            transform=transform,
            padding_value=padding_value,
        )

        encode_and_write_png_atomic(
            output_path=output_path,
            image=processed_image,
            compression_level=(
                compression_level
            ),
        )

        (
            dimensions_valid,
            padding_valid,
        ) = validate_saved_image(
            output_path=output_path,
            target_width=target_width,
            target_height=target_height,
            transform=transform,
            padding_value=padding_value,
        )

    else:
        if (
            transform.resized_width
            == transform.source_width
            and transform.resized_height
            == transform.source_height
        ):
            interpolation_name = (
                "identity"
            )

        elif (
            transform.resized_width
            < transform.source_width
            or transform.resized_height
            < transform.source_height
        ):
            interpolation_name = (
                "opencv_inter_area"
            )

        else:
            interpolation_name = (
                "opencv_inter_linear"
            )

    if not output_path.exists():
        raise FileNotFoundError(
            "Processed output was not created."
        )

    if not dimensions_valid:
        raise ValueError(
            "Saved image dimensions are invalid."
        )

    if not padding_valid:
        raise ValueError(
            "Saved image padding is invalid."
        )

    if not geometry_valid:
        raise ValueError(
            "Transform geometry is invalid."
        )

    if not aspect_ratio_valid:
        raise ValueError(
            "Aspect-ratio preservation check failed."
        )

    return {
        "global_image_id": int(
            row.global_image_id
        ),
        "canonical_image_key": (
            row.canonical_image_key
        ),
        "dataset": row.dataset,
        "partition": row.partition,
        "experimental_role": (
            row.experimental_role
        ),

        "source_image_id": (
            row.source_image_id
        ),
        "source_image_path": (
            source_path.as_posix()
        ),
        "output_filename": (
            row.output_filename
        ),
        "output_relative_path": (
            output_relative_path.as_posix()
        ),
        "output_absolute_path": (
            output_path.resolve().as_posix()
        ),

        "source_width": (
            transform.source_width
        ),
        "source_height": (
            transform.source_height
        ),
        "target_width": (
            transform.target_width
        ),
        "target_height": (
            transform.target_height
        ),

        "nominal_scale": (
            transform.nominal_scale
        ),
        "actual_scale_x": (
            transform.actual_scale_x
        ),
        "actual_scale_y": (
            transform.actual_scale_y
        ),

        "resized_width": (
            transform.resized_width
        ),
        "resized_height": (
            transform.resized_height
        ),

        "padding_left": (
            transform.padding_left
        ),
        "padding_top": (
            transform.padding_top
        ),
        "padding_right": (
            transform.padding_right
        ),
        "padding_bottom": (
            transform.padding_bottom
        ),
        "padding_value": (
            padding_value
        ),

        "interpolation": (
            interpolation_name
        ),
        "source_extension": (
            Path(source_path).suffix.lower()
        ),
        "output_extension": ".png",
        "png_compression_level": (
            compression_level
        ),

        "target_box_count": int(
            row.target_box_count
        ),
        "vehicle_count": int(
            row.vehicle_count
        ),
        "pedestrian_count": int(
            row.pedestrian_count
        ),
        "cyclist_count": int(
            row.cyclist_count
        ),
        "ignored_box_count": int(
            row.ignored_box_count
        ),
        "is_negative": parse_boolean(
            row.is_negative
        ),

        "output_size_bytes": (
            output_path.stat().st_size
        ),

        "dimensions_valid": (
            dimensions_valid
        ),
        "padding_valid": (
            padding_valid
        ),
        "geometry_valid": (
            geometry_valid
        ),
        "aspect_ratio_error": (
            aspect_ratio_error
        ),
        "aspect_ratio_valid": (
            aspect_ratio_valid
        ),

        "processing_status": (
            processing_status
        ),
    }


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_complete_output(
    source_manifest: pd.DataFrame,
    transform_manifest: pd.DataFrame,
    issues: list[dict],
) -> dict:
    checks: dict[str, bool] = {}

    checks["source_manifest_rows"] = (
        len(source_manifest)
        == EXPECTED["total_images"]
    )

    checks["transform_manifest_rows"] = (
        len(transform_manifest)
        == EXPECTED["total_images"]
    )

    checks["unique_global_image_ids"] = (
        not transform_manifest[
            "global_image_id"
        ].duplicated().any()
    )

    checks["unique_output_paths"] = (
        not transform_manifest[
            "output_relative_path"
        ].duplicated().any()
    )

    checks["all_dimensions_valid"] = bool(
        transform_manifest[
            "dimensions_valid"
        ].all()
    )

    checks["all_padding_valid"] = bool(
        transform_manifest[
            "padding_valid"
        ].all()
    )

    checks["all_geometry_valid"] = bool(
        transform_manifest[
            "geometry_valid"
        ].all()
    )

    checks["all_aspect_ratios_valid"] = bool(
        transform_manifest[
            "aspect_ratio_valid"
        ].all()
    )

    checks["all_outputs_nonempty"] = bool(
        (
            transform_manifest[
                "output_size_bytes"
            ]
            > 0
        ).all()
    )

    expected_paths = {
        str(
            (
                PROCESSED_ROOT
                / Path(relative_path)
            ).resolve()
        ).lower()
        for relative_path
        in source_manifest[
            "output_relative_path"
        ].tolist()
    }

    actual_paths: set[str] = set()

    for directory in OUTPUT_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for path in directory.rglob(
            "*.png"
        ):
            if path.is_file():
                actual_paths.add(
                    str(
                        path.resolve()
                    ).lower()
                )

    missing_paths = sorted(
        expected_paths - actual_paths
    )

    extra_paths = sorted(
        actual_paths - expected_paths
    )

    checks["no_missing_output_files"] = (
        len(missing_paths) == 0
    )

    checks["no_extra_output_files"] = (
        len(extra_paths) == 0
    )

    checks["output_file_count"] = (
        len(actual_paths)
        == EXPECTED["total_images"]
    )

    partition_counts = {}

    for (
        dataset,
        partition,
    ), group in transform_manifest.groupby(
        [
            "dataset",
            "partition",
        ]
    ):
        key = partition_key(
            str(dataset),
            str(partition),
        )

        partition_counts[key] = int(
            len(group)
        )

    checks["kitti_train_count"] = (
        partition_counts.get(
            "kitti_train",
            0,
        )
        == EXPECTED["kitti_train"]
    )

    checks["kitti_val_count"] = (
        partition_counts.get(
            "kitti_val",
            0,
        )
        == EXPECTED["kitti_val"]
    )

    checks["waymo_external_count"] = (
        partition_counts.get(
            "waymo_external",
            0,
        )
        == EXPECTED["waymo_external"]
    )

    for missing_path in missing_paths:
        add_issue(
            issues,
            "missing_processed_image",
            missing_path,
            "Expected output file was not found.",
        )

    for extra_path in extra_paths:
        add_issue(
            issues,
            "unexpected_processed_image",
            extra_path,
            "File is not listed in the source manifest.",
        )

    for check_name, passed in checks.items():
        if not passed:
            add_issue(
                issues,
                "final_validation_failed",
                check_name,
                "The final preprocessing check returned false.",
            )

    return {
        "validation_passed": all(
            checks.values()
        ),
        "checks": checks,
        "partition_counts": (
            partition_counts
        ),
        "missing_output_files": (
            len(missing_paths)
        ),
        "unexpected_output_files": (
            len(extra_paths)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    arguments = parse_arguments()

    start_time = time.perf_counter()

    cv2.setNumThreads(1)

    issues: list[dict] = []
    records: list[dict] = []

    print("=" * 76)
    print("PREPROCESSING COMPLETE MILESTONE 3 IMAGE SET")
    print("=" * 76)

    configuration = load_yaml(
        PREPROCESSING_CONFIG
    )

    source_summary = load_json(
        SOURCE_MANIFEST_SUMMARY
    )

    dry_run_report = load_json(
        DRY_RUN_REPORT
    )

    if not source_summary.get(
        "source_manifest_passed",
        False,
    ):
        raise RuntimeError(
            "Step 3 source manifest has not passed."
        )

    if not dry_run_report.get(
        "dry_run_passed",
        False,
    ):
        raise RuntimeError(
            "Step 4 preprocessing dry run has not passed."
        )

    if not SOURCE_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Source manifest not found:\n"
            f"{SOURCE_MANIFEST_FILE.resolve()}"
        )

    image_configuration = (
        configuration[
            "image_preprocessing"
        ]
    )

    target_width = int(
        image_configuration[
            "target_width"
        ]
    )

    target_height = int(
        image_configuration[
            "target_height"
        ]
    )

    padding_values = (
        image_configuration[
            "padding"
        ]["value_rgb"]
    )

    if (
        len(padding_values) != 3
        or len(set(padding_values)) != 1
    ):
        raise ValueError(
            "The implementation requires equal "
            "padding values for all channels."
        )

    padding_value = int(
        padding_values[0]
    )

    compression_level = int(
        image_configuration[
            "output"
        ]["png_compression_level"]
    )

    if not 0 <= compression_level <= 9:
        raise ValueError(
            "PNG compression level must be between 0 and 9."
        )

    source_manifest = pd.read_csv(
        SOURCE_MANIFEST_FILE,
        dtype={
            "source_image_id": str,
            "source_image_path": str,
            "output_filename": str,
            "output_relative_path": str,
        },
    )

    if len(source_manifest) != (
        EXPECTED["total_images"]
    ):
        raise ValueError(
            f"Expected {EXPECTED['total_images']} "
            f"source rows, found "
            f"{len(source_manifest)}."
        )

    source_manifest = (
        source_manifest.sort_values(
            "global_image_id"
        )
        .reset_index(drop=True)
    )

    for directory in OUTPUT_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    overwrite = bool(
        arguments.overwrite
    )

    if arguments.clean:
        removed_count = (
            safely_clean_output_directories()
        )

        overwrite = True

        print(
            f"Existing processed files removed: "
            f"{removed_count}"
        )

    print(
        f"Images to process: "
        f"{len(source_manifest)}"
    )

    print(
        f"Target size: "
        f"{target_width}x{target_height}"
    )

    print(
        f"Padding value: "
        f"{padding_value}"
    )

    print(
        f"PNG compression level: "
        f"{compression_level}"
    )

    print(
        f"Overwrite existing: "
        f"{overwrite}\n"
    )

    iterator = source_manifest.itertuples(
        index=False
    )

    for number, row in enumerate(
        tqdm(
            iterator,
            total=len(source_manifest),
            unit="image",
        ),
        start=1,
    ):
        identifier = (
            f"{row.dataset}:"
            f"{row.source_image_id}"
        )

        try:
            record = process_one_image(
                row=row,
                target_width=target_width,
                target_height=target_height,
                padding_value=padding_value,
                compression_level=(
                    compression_level
                ),
                overwrite=overwrite,
            )

            records.append(record)

        except Exception as error:
            add_issue(
                issues,
                "image_processing_failed",
                identifier,
                str(error),
            )

        if number % 250 == 0:
            save_partial_manifest(
                records
            )

    transform_manifest = pd.DataFrame(
        records,
        columns=TRANSFORM_COLUMNS,
    )

    if not transform_manifest.empty:
        transform_manifest = (
            transform_manifest.sort_values(
                "global_image_id"
            )
            .reset_index(drop=True)
        )

    validation = validate_complete_output(
        source_manifest=source_manifest,
        transform_manifest=(
            transform_manifest
        ),
        issues=issues,
    )

    overall_passed = (
        validation[
            "validation_passed"
        ]
        and len(issues) == 0
    )

    transform_manifest.to_csv(
        TRANSFORM_MANIFEST_FILE,
        index=False,
    )

    write_csv(
        ISSUES_FILE,
        issues,
        [
            "category",
            "identifier",
            "details",
        ],
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    processing_status_counts = dict(
        Counter(
            transform_manifest[
                "processing_status"
            ].tolist()
        )
    ) if not transform_manifest.empty else {}

    interpolation_counts = dict(
        Counter(
            transform_manifest[
                "interpolation"
            ].tolist()
        )
    ) if not transform_manifest.empty else {}

    source_extension_counts = dict(
        Counter(
            transform_manifest[
                "source_extension"
            ].tolist()
        )
    ) if not transform_manifest.empty else {}

    padding_pattern_counts = {}

    if not transform_manifest.empty:
        padding_patterns = (
            transform_manifest[
                [
                    "padding_left",
                    "padding_top",
                    "padding_right",
                    "padding_bottom",
                ]
            ]
            .astype(str)
            .agg("/".join, axis=1)
            .value_counts()
        )

        padding_pattern_counts = {
            str(key): int(value)
            for key, value
            in padding_patterns.items()
        }

    total_output_size_bytes = int(
        transform_manifest[
            "output_size_bytes"
        ].sum()
    ) if not transform_manifest.empty else 0

    maximum_aspect_ratio_error = float(
        transform_manifest[
            "aspect_ratio_error"
        ].max()
    ) if not transform_manifest.empty else None

    report = {
        "milestone": 3,
        "step": 5,
        "purpose": (
            "Create one deterministic 640x640 "
            "letterboxed image for every source sample."
        ),
        "configuration_sha256": (
            sha256_file(
                PREPROCESSING_CONFIG
            )
        ),
        "source_manifest_sha256": (
            sha256_file(
                SOURCE_MANIFEST_FILE
            )
        ),
        "target_width": target_width,
        "target_height": target_height,
        "padding_value": padding_value,
        "png_compression_level": (
            compression_level
        ),
        "source_manifest_rows": int(
            len(source_manifest)
        ),
        "transform_manifest_rows": int(
            len(transform_manifest)
        ),
        "partition_counts": validation[
            "partition_counts"
        ],
        "processing_status_counts": (
            processing_status_counts
        ),
        "interpolation_counts": (
            interpolation_counts
        ),
        "source_extension_counts": (
            source_extension_counts
        ),
        "padding_pattern_counts": (
            padding_pattern_counts
        ),
        "total_output_size_bytes": (
            total_output_size_bytes
        ),
        "total_output_size_gib": round(
            total_output_size_bytes
            / (1024 ** 3),
            6,
        ),
        "maximum_aspect_ratio_error": (
            maximum_aspect_ratio_error
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            3,
        ),
        "software": {
            "python": (
                platform.python_version()
            ),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "issue_count": len(issues),
        **validation,
        "image_preprocessing_passed": (
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

    if (
        overall_passed
        and PARTIAL_MANIFEST_FILE.exists()
    ):
        PARTIAL_MANIFEST_FILE.unlink()

    print("\n" + "=" * 76)
    print("COMPLETE IMAGE PREPROCESSING SUMMARY")
    print("=" * 76)

    print(
        f"Source manifest rows: "
        f"{len(source_manifest)}"
    )

    print(
        f"Transform manifest rows: "
        f"{len(transform_manifest)}"
    )

    print("\nPartition counts:")

    for name, count in (
        validation[
            "partition_counts"
        ].items()
    ):
        print(
            f"  {name}: {count}"
        )

    print("\nProcessing status:")

    for status, count in (
        processing_status_counts.items()
    ):
        print(
            f"  {status}: {count}"
        )

    print(
        f"\nMissing outputs: "
        f"{validation['missing_output_files']}"
    )

    print(
        f"Unexpected outputs: "
        f"{validation['unexpected_output_files']}"
    )

    print(
        f"Issues found: "
        f"{len(issues)}"
    )

    print(
        f"Output size: "
        f"{report['total_output_size_gib']} GiB"
    )

    print(
        f"Elapsed time: "
        f"{elapsed_seconds:.2f} seconds"
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
        f"\nTransform manifest:\n"
        f"{TRANSFORM_MANIFEST_FILE.resolve()}"
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
            "\nDo not continue to annotation conversion "
            "until all preprocessing issues are resolved."
        )

        sys.exit(1)

    print(
        "\nStep 5 completed successfully."
    )


if __name__ == "__main__":
    main()