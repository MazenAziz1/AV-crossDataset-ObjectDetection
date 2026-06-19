from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import math
import re
import sys

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


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

OUTPUT_DIR = Path(
    "data/processed/milestone_3/visual_checks/"
    "preprocessing_dry_run"
)

REPORT_DIR = Path(
    "data/processed/milestone_3/reports"
)

DRY_RUN_CSV = (
    REPORT_DIR / "preprocessing_dry_run.csv"
)

DRY_RUN_JSON = (
    REPORT_DIR / "preprocessing_dry_run.json"
)

ISSUES_FILE = (
    REPORT_DIR / "preprocessing_dry_run_issues.csv"
)


EXPECTED_SAMPLE_COUNT = 8


# ============================================================
# HELPERS
# ============================================================

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


def round_half_up(value: float) -> int:
    return int(
        math.floor(value + 0.5)
    )


def parse_boolean(value) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
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


def safe_filename_component(
    value: str,
) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    )

    return cleaned.strip("_")


def verify_padding(
    image: np.ndarray,
    padding_left: int,
    padding_top: int,
    padding_right: int,
    padding_bottom: int,
    padding_value: int,
) -> bool:
    expected = np.array(
        [
            padding_value,
            padding_value,
            padding_value,
        ],
        dtype=np.uint8,
    )

    checks: list[bool] = []

    if padding_top > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        :padding_top,
                        :,
                    ]
                    == expected
                )
            )
        )

    if padding_bottom > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        image.shape[0]
                        - padding_bottom:,
                        :,
                    ]
                    == expected
                )
            )
        )

    if padding_left > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        :,
                        :padding_left,
                    ]
                    == expected
                )
            )
        )

    if padding_right > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        :,
                        image.shape[1]
                        - padding_right:,
                    ]
                    == expected
                )
            )
        )

    return all(checks) if checks else True


# ============================================================
# SAMPLE SELECTION
# ============================================================

def choose_unique_row(
    dataframe: pd.DataFrame,
    mask: pd.Series,
    sort_column: str,
    ascending: bool,
    selected_global_ids: set[int],
) -> pd.Series | None:
    candidates = dataframe[
        mask
    ].copy()

    candidates = candidates[
        ~candidates[
            "global_image_id"
        ].isin(
            selected_global_ids
        )
    ]

    if candidates.empty:
        return None

    candidates = candidates.sort_values(
        [
            sort_column,
            "global_image_id",
        ],
        ascending=[
            ascending,
            True,
        ],
    )

    return candidates.iloc[0]


def select_dry_run_samples(
    dataframe: pd.DataFrame,
    issues: list[dict],
) -> list[tuple[str, pd.Series]]:
    selected: list[
        tuple[str, pd.Series]
    ] = []

    selected_global_ids: set[int] = set()

    criteria = [
        {
            "reason": "kitti_train_crowded",
            "mask": (
                (dataframe["dataset"] == "KITTI")
                & (dataframe["partition"] == "train")
            ),
            "sort_column": "target_box_count",
            "ascending": False,
        },
        {
            "reason": "kitti_val_crowded",
            "mask": (
                (dataframe["dataset"] == "KITTI")
                & (dataframe["partition"] == "val")
            ),
            "sort_column": "target_box_count",
            "ascending": False,
        },
        {
            "reason": "kitti_train_cyclist_rich",
            "mask": (
                (dataframe["dataset"] == "KITTI")
                & (dataframe["partition"] == "train")
                & (dataframe["cyclist_count"] > 0)
            ),
            "sort_column": "cyclist_count",
            "ascending": False,
        },
        {
            "reason": "kitti_val_pedestrian_rich",
            "mask": (
                (dataframe["dataset"] == "KITTI")
                & (dataframe["partition"] == "val")
                & (dataframe["pedestrian_count"] > 0)
            ),
            "sort_column": "pedestrian_count",
            "ascending": False,
        },
        {
            "reason": "waymo_crowded",
            "mask": (
                (dataframe["dataset"] == "Waymo")
                & (dataframe["partition"] == "external")
                & (~dataframe["is_negative_bool"])
            ),
            "sort_column": "target_box_count",
            "ascending": False,
        },
        {
            "reason": "waymo_cyclist_rich",
            "mask": (
                (dataframe["dataset"] == "Waymo")
                & (dataframe["partition"] == "external")
                & (dataframe["cyclist_count"] > 0)
            ),
            "sort_column": "cyclist_count",
            "ascending": False,
        },
        {
            "reason": "waymo_pedestrian_rich",
            "mask": (
                (dataframe["dataset"] == "Waymo")
                & (dataframe["partition"] == "external")
                & (dataframe["pedestrian_count"] > 0)
            ),
            "sort_column": "pedestrian_count",
            "ascending": False,
        },
        {
            "reason": "waymo_negative",
            "mask": (
                (dataframe["dataset"] == "Waymo")
                & (dataframe["partition"] == "external")
                & dataframe["is_negative_bool"]
            ),
            "sort_column": "global_image_id",
            "ascending": True,
        },
    ]

    for criterion in criteria:
        row = choose_unique_row(
            dataframe=dataframe,
            mask=criterion["mask"],
            sort_column=criterion[
                "sort_column"
            ],
            ascending=criterion[
                "ascending"
            ],
            selected_global_ids=(
                selected_global_ids
            ),
        )

        if row is None:
            add_issue(
                issues,
                "sample_selection_failed",
                criterion["reason"],
                "No suitable unique image was found.",
            )
            continue

        global_image_id = int(
            row["global_image_id"]
        )

        selected_global_ids.add(
            global_image_id
        )

        selected.append(
            (
                criterion["reason"],
                row,
            )
        )

    return selected


# ============================================================
# LETTERBOX PREPROCESSING
# ============================================================

def preprocess_image(
    source_image: np.ndarray,
    target_width: int,
    target_height: int,
    padding_value: int,
) -> tuple[np.ndarray, dict]:
    source_height, source_width = (
        source_image.shape[:2]
    )

    nominal_scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = min(
        target_width,
        round_half_up(
            source_width
            * nominal_scale
        ),
    )

    resized_height = min(
        target_height,
        round_half_up(
            source_height
            * nominal_scale
        ),
    )

    if (
        resized_width <= 0
        or resized_height <= 0
    ):
        raise ValueError(
            "Calculated resized dimensions "
            "are not positive."
        )

    actual_scale_x = (
        resized_width / source_width
    )

    actual_scale_y = (
        resized_height / source_height
    )

    if (
        resized_width == source_width
        and resized_height == source_height
    ):
        resized = source_image.copy()
        interpolation_name = "identity"

    else:
        is_downscale = (
            resized_width < source_width
            or resized_height < source_height
        )

        if is_downscale:
            interpolation = cv2.INTER_AREA
            interpolation_name = (
                "opencv_inter_area"
            )
        else:
            interpolation = cv2.INTER_LINEAR
            interpolation_name = (
                "opencv_inter_linear"
            )

        resized = cv2.resize(
            source_image,
            (
                resized_width,
                resized_height,
            ),
            interpolation=interpolation,
        )

    padding_left = (
        target_width - resized_width
    ) // 2

    padding_top = (
        target_height - resized_height
    ) // 2

    padding_right = (
        target_width
        - resized_width
        - padding_left
    )

    padding_bottom = (
        target_height
        - resized_height
        - padding_top
    )

    canvas = np.full(
        (
            target_height,
            target_width,
            3,
        ),
        fill_value=padding_value,
        dtype=np.uint8,
    )

    canvas[
        padding_top:
        padding_top + resized_height,
        padding_left:
        padding_left + resized_width,
    ] = resized

    transform = {
        "source_width": int(
            source_width
        ),
        "source_height": int(
            source_height
        ),
        "target_width": int(
            target_width
        ),
        "target_height": int(
            target_height
        ),
        "nominal_scale": float(
            nominal_scale
        ),
        "actual_scale_x": float(
            actual_scale_x
        ),
        "actual_scale_y": float(
            actual_scale_y
        ),
        "resized_width": int(
            resized_width
        ),
        "resized_height": int(
            resized_height
        ),
        "padding_left": int(
            padding_left
        ),
        "padding_top": int(
            padding_top
        ),
        "padding_right": int(
            padding_right
        ),
        "padding_bottom": int(
            padding_bottom
        ),
        "interpolation": (
            interpolation_name
        ),
    }

    return canvas, transform


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
    records: list[dict] = []

    print("=" * 76)
    print("MILESTONE 3 PREPROCESSING DRY RUN")
    print("=" * 76)

    configuration = load_yaml(
        PREPROCESSING_CONFIG
    )

    manifest_summary = load_json(
        SOURCE_MANIFEST_SUMMARY
    )

    if not manifest_summary.get(
        "source_manifest_passed",
        False,
    ):
        raise RuntimeError(
            "Step 3 source manifest has not passed."
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
            "This implementation requires one "
            "equal padding value for all channels."
        )

    padding_value = int(
        padding_values[0]
    )

    compression_level = int(
        image_configuration[
            "output"
        ]["png_compression_level"]
    )

    manifest = pd.read_csv(
        SOURCE_MANIFEST_FILE,
        dtype={
            "source_image_id": str,
            "source_image_path": str,
            "output_filename": str,
        },
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
        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="raise",
        )

    manifest["is_negative_bool"] = (
        manifest["is_negative"]
        .apply(parse_boolean)
    )

    selected_samples = (
        select_dry_run_samples(
            dataframe=manifest,
            issues=issues,
        )
    )

    print(
        f"\nSelected samples: "
        f"{len(selected_samples)}"
    )

    for sample_number, (
        selection_reason,
        row,
    ) in enumerate(
        tqdm(
            selected_samples,
            unit="image",
        ),
        start=1,
    ):
        global_image_id = int(
            row["global_image_id"]
        )

        source_path = Path(
            str(
                row["source_image_path"]
            )
        )

        identifier = (
            f"{row['dataset']}:"
            f"{row['source_image_id']}"
        )

        if not source_path.exists():
            add_issue(
                issues,
                "missing_source_image",
                identifier,
                str(source_path),
            )
            continue

        source_image = cv2.imread(
            str(source_path),
            cv2.IMREAD_COLOR,
        )

        if source_image is None:
            add_issue(
                issues,
                "image_decode_failed",
                identifier,
                str(source_path),
            )
            continue

        actual_source_height, (
            actual_source_width
        ) = source_image.shape[:2]

        if (
            actual_source_width
            != int(row["source_width"])
            or actual_source_height
            != int(row["source_height"])
        ):
            add_issue(
                issues,
                "source_dimension_mismatch",
                identifier,
                (
                    f"Manifest="
                    f"{row['source_width']}x"
                    f"{row['source_height']}; "
                    f"actual="
                    f"{actual_source_width}x"
                    f"{actual_source_height}"
                ),
            )
            continue

        processed_image, transform = (
            preprocess_image(
                source_image=source_image,
                target_width=target_width,
                target_height=target_height,
                padding_value=padding_value,
            )
        )

        safe_reason = (
            safe_filename_component(
                selection_reason
            )
        )

        original_output_name = Path(
            str(row["output_filename"])
        )

        dry_run_filename = (
            f"{sample_number:02d}_"
            f"{safe_reason}__"
            f"{original_output_name.name}"
        )

        output_path = (
            OUTPUT_DIR
            / dry_run_filename
        )

        write_succeeded = cv2.imwrite(
            str(output_path),
            processed_image,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                compression_level,
            ],
        )

        if not write_succeeded:
            add_issue(
                issues,
                "image_write_failed",
                identifier,
                str(output_path),
            )
            continue

        saved_image = cv2.imread(
            str(output_path),
            cv2.IMREAD_COLOR,
        )

        if saved_image is None:
            add_issue(
                issues,
                "saved_image_decode_failed",
                identifier,
                str(output_path),
            )
            continue

        saved_height, saved_width = (
            saved_image.shape[:2]
        )

        dimensions_valid = (
            saved_width == target_width
            and saved_height == target_height
        )

        padding_valid = verify_padding(
            image=saved_image,
            padding_left=transform[
                "padding_left"
            ],
            padding_top=transform[
                "padding_top"
            ],
            padding_right=transform[
                "padding_right"
            ],
            padding_bottom=transform[
                "padding_bottom"
            ],
            padding_value=padding_value,
        )

        geometry_valid = (
            transform["resized_width"]
            + transform["padding_left"]
            + transform["padding_right"]
            == target_width
            and transform["resized_height"]
            + transform["padding_top"]
            + transform["padding_bottom"]
            == target_height
        )

        aspect_ratio_error = abs(
            (
                transform["resized_width"]
                / transform["resized_height"]
            )
            - (
                transform["source_width"]
                / transform["source_height"]
            )
        )

        maximum_expected_ratio_error = (
            max(
                1.0 / transform[
                    "resized_height"
                ],
                1.0 / transform[
                    "resized_width"
                ],
            )
            * 2.0
        )

        aspect_ratio_valid = (
            aspect_ratio_error
            <= maximum_expected_ratio_error
        )

        sample_passed = all(
            [
                dimensions_valid,
                padding_valid,
                geometry_valid,
                aspect_ratio_valid,
            ]
        )

        if not sample_passed:
            add_issue(
                issues,
                "dry_run_sample_failed",
                identifier,
                (
                    f"dimensions_valid="
                    f"{dimensions_valid}; "
                    f"padding_valid="
                    f"{padding_valid}; "
                    f"geometry_valid="
                    f"{geometry_valid}; "
                    f"aspect_ratio_valid="
                    f"{aspect_ratio_valid}"
                ),
            )

        record = {
            "sample_number": (
                sample_number
            ),
            "selection_reason": (
                selection_reason
            ),
            "global_image_id": (
                global_image_id
            ),
            "dataset": row["dataset"],
            "partition": row["partition"],
            "source_image_id": (
                row["source_image_id"]
            ),
            "source_image_path": (
                source_path.as_posix()
            ),
            "output_image_path": (
                output_path.as_posix()
            ),
            "source_width": (
                transform["source_width"]
            ),
            "source_height": (
                transform["source_height"]
            ),
            "target_width": (
                transform["target_width"]
            ),
            "target_height": (
                transform["target_height"]
            ),
            "nominal_scale": (
                transform["nominal_scale"]
            ),
            "actual_scale_x": (
                transform["actual_scale_x"]
            ),
            "actual_scale_y": (
                transform["actual_scale_y"]
            ),
            "resized_width": (
                transform["resized_width"]
            ),
            "resized_height": (
                transform["resized_height"]
            ),
            "padding_left": (
                transform["padding_left"]
            ),
            "padding_top": (
                transform["padding_top"]
            ),
            "padding_right": (
                transform["padding_right"]
            ),
            "padding_bottom": (
                transform["padding_bottom"]
            ),
            "padding_value": (
                padding_value
            ),
            "interpolation": (
                transform["interpolation"]
            ),
            "target_box_count": int(
                row["target_box_count"]
            ),
            "vehicle_count": int(
                row["vehicle_count"]
            ),
            "pedestrian_count": int(
                row["pedestrian_count"]
            ),
            "cyclist_count": int(
                row["cyclist_count"]
            ),
            "is_negative": bool(
                row["is_negative_bool"]
            ),
            "source_sha256": (
                sha256_file(source_path)
            ),
            "output_sha256": (
                sha256_file(output_path)
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
            "sample_passed": (
                sample_passed
            ),
        }

        records.append(record)

    fieldnames = [
        "sample_number",
        "selection_reason",
        "global_image_id",
        "dataset",
        "partition",
        "source_image_id",
        "source_image_path",
        "output_image_path",
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
        "target_box_count",
        "vehicle_count",
        "pedestrian_count",
        "cyclist_count",
        "is_negative",
        "source_sha256",
        "output_sha256",
        "dimensions_valid",
        "padding_valid",
        "geometry_valid",
        "aspect_ratio_error",
        "aspect_ratio_valid",
        "sample_passed",
    ]

    write_csv(
        DRY_RUN_CSV,
        records,
        fieldnames,
    )

    all_samples_passed = (
        len(records)
        == EXPECTED_SAMPLE_COUNT
        and all(
            bool(record["sample_passed"])
            for record in records
        )
    )

    overall_passed = (
        all_samples_passed
        and len(issues) == 0
    )

    report = {
        "milestone": 3,
        "step": 4,
        "purpose": (
            "Validate deterministic 640x640 "
            "letterbox preprocessing on a "
            "representative sample."
        ),
        "target_size": {
            "width": target_width,
            "height": target_height,
        },
        "padding_value": (
            padding_value
        ),
        "expected_samples": (
            EXPECTED_SAMPLE_COUNT
        ),
        "created_samples": len(
            records
        ),
        "selection_reasons": [
            record[
                "selection_reason"
            ]
            for record in records
        ],
        "all_dimensions_valid": all(
            bool(
                record[
                    "dimensions_valid"
                ]
            )
            for record in records
        ) if records else False,
        "all_padding_valid": all(
            bool(
                record[
                    "padding_valid"
                ]
            )
            for record in records
        ) if records else False,
        "all_geometry_valid": all(
            bool(
                record[
                    "geometry_valid"
                ]
            )
            for record in records
        ) if records else False,
        "all_aspect_ratios_valid": all(
            bool(
                record[
                    "aspect_ratio_valid"
                ]
            )
            for record in records
        ) if records else False,
        "issue_count": len(issues),
        "dry_run_passed": (
            overall_passed
        ),
    }

    DRY_RUN_JSON.write_text(
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
            "category",
            "identifier",
            "details",
        ],
    )

    print("\n" + "=" * 76)
    print("PREPROCESSING DRY-RUN SUMMARY")
    print("=" * 76)

    for record in records:
        print(
            f"\n[{record['sample_number']}] "
            f"{record['selection_reason']}"
        )

        print(
            f"  Dataset: "
            f"{record['dataset']} / "
            f"{record['partition']}"
        )

        print(
            f"  Source: "
            f"{record['source_width']}x"
            f"{record['source_height']}"
        )

        print(
            f"  Resized: "
            f"{record['resized_width']}x"
            f"{record['resized_height']}"
        )

        print(
            f"  Padding L/T/R/B: "
            f"{record['padding_left']}/"
            f"{record['padding_top']}/"
            f"{record['padding_right']}/"
            f"{record['padding_bottom']}"
        )

        print(
            f"  Status: "
            f"{'PASSED' if record['sample_passed'] else 'FAILED'}"
        )

    print(
        f"\nSamples created: "
        f"{len(records)}"
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
        f"\nDry-run images:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        f"\nCSV report:\n"
        f"{DRY_RUN_CSV.resolve()}"
    )

    print(
        f"\nJSON report:\n"
        f"{DRY_RUN_JSON.resolve()}"
    )

    if not overall_passed:
        print(
            "\nDo not process the complete dataset "
            "until every dry-run issue is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 4 completed successfully. "
        "Inspect the eight generated images "
        "before proceeding."
    )


if __name__ == "__main__":
    main()