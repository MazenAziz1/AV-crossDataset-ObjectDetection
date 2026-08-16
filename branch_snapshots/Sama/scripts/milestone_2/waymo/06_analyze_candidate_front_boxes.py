from pathlib import Path
import json

import numpy as np
import pandas as pd


CAMERA_BOX_DIR = Path(
    "data/waymo/raw/validation/camera_box/candidates"
)

CANDIDATES_FILE = Path(
    "data/waymo/selection/candidate_segments.csv"
)

OUTPUT_FILE = Path(
    "data/waymo/selection/candidate_front_camera_stats.csv"
)

SIZE_THRESHOLDS_FILE = Path(
    "data/waymo/selection/front_box_size_thresholds.json"
)


# Exact columns found in Step 7.
SEGMENT_COLUMN = "key.segment_context_name"
TIMESTAMP_COLUMN = "key.frame_timestamp_micros"
CAMERA_COLUMN = "key.camera_name"
OBJECT_ID_COLUMN = "key.camera_object_id"

CENTER_X_COLUMN = "[CameraBoxComponent].box.center.x"
CENTER_Y_COLUMN = "[CameraBoxComponent].box.center.y"
WIDTH_COLUMN = "[CameraBoxComponent].box.size.x"
HEIGHT_COLUMN = "[CameraBoxComponent].box.size.y"

TYPE_COLUMN = "[CameraBoxComponent].type"

DETECTION_DIFFICULTY_COLUMN = (
    "[CameraBoxComponent].difficulty_level.detection"
)


# Waymo identifiers.
FRONT_CAMERA_ID = 1

VEHICLE_TYPE = 1
PEDESTRIAN_TYPE = 2
SIGN_TYPE = 3
CYCLIST_TYPE = 4

TARGET_TYPES = {
    VEHICLE_TYPE,
    PEDESTRIAN_TYPE,
    CYCLIST_TYPE,
}


REQUIRED_COLUMNS = [
    SEGMENT_COLUMN,
    TIMESTAMP_COLUMN,
    CAMERA_COLUMN,
    OBJECT_ID_COLUMN,
    CENTER_X_COLUMN,
    CENTER_Y_COLUMN,
    WIDTH_COLUMN,
    HEIGHT_COLUMN,
    TYPE_COLUMN,
    DETECTION_DIFFICULTY_COLUMN,
]


def safe_percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return round(100.0 * numerator / denominator, 2)


def count_frames_with_type(
    dataframe: pd.DataFrame,
    object_type: int,
) -> int:
    return int(
        dataframe.loc[
            dataframe[TYPE_COLUMN] == object_type,
            TIMESTAMP_COLUMN,
        ].nunique()
    )


def count_difficulty(
    difficulty_series: pd.Series,
    target_level: int,
) -> int:
    numeric = pd.to_numeric(
        difficulty_series,
        errors="coerce",
    )

    return int((numeric == target_level).sum())


def count_unspecified_difficulty(
    difficulty_series: pd.Series,
) -> int:
    numeric = pd.to_numeric(
        difficulty_series,
        errors="coerce",
    )

    # Both missing values and level 0 are treated as unspecified.
    return int(
        (
            numeric.isna()
            | (numeric == 0)
        ).sum()
    )


def main() -> None:
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidate list not found:\n"
            f"{CANDIDATES_FILE.resolve()}"
        )

    parquet_files = sorted(
        CAMERA_BOX_DIR.glob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No candidate camera-box files found in:\n"
            f"{CAMERA_BOX_DIR.resolve()}"
        )

    candidates = pd.read_csv(CANDIDATES_FILE)

    if len(parquet_files) != len(candidates):
        print(
            "Warning: number of camera-box files does not "
            "match the number of candidate segments."
        )
        print(f"Candidate rows: {len(candidates)}")
        print(f"Parquet files: {len(parquet_files)}")

    # ---------------------------------------------------------
    # First pass:
    # collect all target box areas to determine dataset-relative
    # small, medium and large thresholds.
    # ---------------------------------------------------------
    all_target_box_areas: list[np.ndarray] = []

    print("=" * 70)
    print("FIRST PASS: CALCULATING BOX-SIZE THRESHOLDS")
    print("=" * 70)

    for file_number, parquet_file in enumerate(
        parquet_files,
        start=1,
    ):
        dataframe = pd.read_parquet(
            parquet_file,
            columns=[
                CAMERA_COLUMN,
                WIDTH_COLUMN,
                HEIGHT_COLUMN,
                TYPE_COLUMN,
            ],
        )

        front_targets = dataframe[
            (dataframe[CAMERA_COLUMN] == FRONT_CAMERA_ID)
            & (dataframe[TYPE_COLUMN].isin(TARGET_TYPES))
        ].copy()

        if not front_targets.empty:
            areas = (
                front_targets[WIDTH_COLUMN].astype(float)
                * front_targets[HEIGHT_COLUMN].astype(float)
            ).to_numpy()

            valid_areas = areas[
                np.isfinite(areas) & (areas > 0)
            ]

            if len(valid_areas) > 0:
                all_target_box_areas.append(valid_areas)

        print(
            f"Scanned {file_number}/{len(parquet_files)}: "
            f"{parquet_file.name}"
        )

    if not all_target_box_areas:
        raise ValueError(
            "No valid FRONT-camera target bounding boxes were found."
        )

    combined_areas = np.concatenate(
        all_target_box_areas
    )

    small_threshold = float(
        np.quantile(combined_areas, 1 / 3)
    )

    large_threshold = float(
        np.quantile(combined_areas, 2 / 3)
    )

    size_thresholds = {
        "method": "candidate_front_camera_area_quantiles",
        "area_unit": "square_pixels",
        "small_definition": (
            "box_area <= lower_33_percent_quantile"
        ),
        "medium_definition": (
            "lower_33_percent_quantile < box_area "
            "<= upper_67_percent_quantile"
        ),
        "large_definition": (
            "box_area > upper_67_percent_quantile"
        ),
        "small_max_area": round(
            small_threshold,
            6,
        ),
        "medium_max_area": round(
            large_threshold,
            6,
        ),
        "total_target_boxes_used": int(
            len(combined_areas)
        ),
    }

    SIZE_THRESHOLDS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SIZE_THRESHOLDS_FILE.write_text(
        json.dumps(
            size_thresholds,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nBox-size thresholds:")
    print(
        f"Small: area <= {small_threshold:.2f}"
    )
    print(
        f"Medium: {small_threshold:.2f} < area "
        f"<= {large_threshold:.2f}"
    )
    print(
        f"Large: area > {large_threshold:.2f}"
    )

    # ---------------------------------------------------------
    # Second pass:
    # calculate one FRONT-camera summary row per segment.
    # ---------------------------------------------------------
    segment_records: list[dict] = []

    print("\n" + "=" * 70)
    print("SECOND PASS: ANALYZING FRONT-CAMERA BOXES")
    print("=" * 70)

    candidate_frame_lookup = (
        candidates
        .set_index("segment_id")["number_of_frames"]
        .to_dict()
    )

    for file_number, parquet_file in enumerate(
        parquet_files,
        start=1,
    ):
        dataframe = pd.read_parquet(
            parquet_file,
            columns=REQUIRED_COLUMNS,
        )

        front_boxes = dataframe[
            dataframe[CAMERA_COLUMN] == FRONT_CAMERA_ID
        ].copy()

        if front_boxes.empty:
            segment_id = parquet_file.stem

            segment_records.append(
                {
                    "segment_id": segment_id,
                    "source_camera_box_file": parquet_file.name,
                    "source_frame_count": int(
                        candidate_frame_lookup.get(
                            segment_id,
                            0,
                        )
                    ),
                    "front_labeled_frame_count": 0,
                    "front_target_frame_count": 0,
                    "front_target_frame_coverage_percent": 0.0,
                    "vehicle_box_count": 0,
                    "vehicle_frame_count": 0,
                    "vehicle_frame_coverage_percent": 0.0,
                    "pedestrian_box_count": 0,
                    "pedestrian_frame_count": 0,
                    "pedestrian_frame_coverage_percent": 0.0,
                    "cyclist_box_count": 0,
                    "cyclist_frame_count": 0,
                    "cyclist_frame_coverage_percent": 0.0,
                    "ignored_sign_box_count": 0,
                    "total_target_box_count": 0,
                    "average_target_boxes_per_source_frame": 0.0,
                    "maximum_target_boxes_in_one_frame": 0,
                    "small_box_count": 0,
                    "medium_box_count": 0,
                    "large_box_count": 0,
                    "mean_box_area": 0.0,
                    "median_box_area": 0.0,
                    "level_1_box_count": 0,
                    "level_2_box_count": 0,
                    "unspecified_difficulty_box_count": 0,
                }
            )

            continue

        segment_values = (
            front_boxes[SEGMENT_COLUMN]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(segment_values) == 0:
            segment_id = parquet_file.stem
        else:
            segment_id = segment_values[0]

        source_frame_count = int(
            candidate_frame_lookup.get(
                segment_id,
                front_boxes[TIMESTAMP_COLUMN].nunique(),
            )
        )

        front_labeled_frame_count = int(
            front_boxes[TIMESTAMP_COLUMN].nunique()
        )

        sign_box_count = int(
            (front_boxes[TYPE_COLUMN] == SIGN_TYPE).sum()
        )

        target_boxes = front_boxes[
            front_boxes[TYPE_COLUMN].isin(TARGET_TYPES)
        ].copy()

        target_boxes["box_area"] = (
            target_boxes[WIDTH_COLUMN].astype(float)
            * target_boxes[HEIGHT_COLUMN].astype(float)
        )

        target_boxes = target_boxes[
            np.isfinite(target_boxes["box_area"])
            & (target_boxes["box_area"] > 0)
        ].copy()

        front_target_frame_count = int(
            target_boxes[TIMESTAMP_COLUMN].nunique()
        )

        vehicle_box_count = int(
            (target_boxes[TYPE_COLUMN] == VEHICLE_TYPE).sum()
        )

        pedestrian_box_count = int(
            (
                target_boxes[TYPE_COLUMN]
                == PEDESTRIAN_TYPE
            ).sum()
        )

        cyclist_box_count = int(
            (target_boxes[TYPE_COLUMN] == CYCLIST_TYPE).sum()
        )

        vehicle_frame_count = count_frames_with_type(
            target_boxes,
            VEHICLE_TYPE,
        )

        pedestrian_frame_count = count_frames_with_type(
            target_boxes,
            PEDESTRIAN_TYPE,
        )

        cyclist_frame_count = count_frames_with_type(
            target_boxes,
            CYCLIST_TYPE,
        )

        boxes_per_frame = (
            target_boxes
            .groupby(TIMESTAMP_COLUMN)
            .size()
        )

        maximum_target_boxes = (
            int(boxes_per_frame.max())
            if not boxes_per_frame.empty
            else 0
        )

        small_box_count = int(
            (
                target_boxes["box_area"]
                <= small_threshold
            ).sum()
        )

        medium_box_count = int(
            (
                (target_boxes["box_area"] > small_threshold)
                & (
                    target_boxes["box_area"]
                    <= large_threshold
                )
            ).sum()
        )

        large_box_count = int(
            (
                target_boxes["box_area"]
                > large_threshold
            ).sum()
        )

        if target_boxes.empty:
            mean_box_area = 0.0
            median_box_area = 0.0
        else:
            mean_box_area = float(
                target_boxes["box_area"].mean()
            )
            median_box_area = float(
                target_boxes["box_area"].median()
            )

        difficulty = target_boxes[
            DETECTION_DIFFICULTY_COLUMN
        ]

        segment_records.append(
            {
                "segment_id": segment_id,
                "source_camera_box_file": parquet_file.name,
                "source_frame_count": source_frame_count,
                "front_labeled_frame_count": (
                    front_labeled_frame_count
                ),
                "front_target_frame_count": (
                    front_target_frame_count
                ),
                "front_target_frame_coverage_percent": (
                    safe_percentage(
                        front_target_frame_count,
                        source_frame_count,
                    )
                ),
                "vehicle_box_count": vehicle_box_count,
                "vehicle_frame_count": vehicle_frame_count,
                "vehicle_frame_coverage_percent": (
                    safe_percentage(
                        vehicle_frame_count,
                        source_frame_count,
                    )
                ),
                "pedestrian_box_count": pedestrian_box_count,
                "pedestrian_frame_count": pedestrian_frame_count,
                "pedestrian_frame_coverage_percent": (
                    safe_percentage(
                        pedestrian_frame_count,
                        source_frame_count,
                    )
                ),
                "cyclist_box_count": cyclist_box_count,
                "cyclist_frame_count": cyclist_frame_count,
                "cyclist_frame_coverage_percent": (
                    safe_percentage(
                        cyclist_frame_count,
                        source_frame_count,
                    )
                ),
                "ignored_sign_box_count": sign_box_count,
                "total_target_box_count": int(
                    len(target_boxes)
                ),
                "average_target_boxes_per_source_frame": round(
                    (
                        len(target_boxes)
                        / source_frame_count
                    )
                    if source_frame_count > 0
                    else 0.0,
                    3,
                ),
                "maximum_target_boxes_in_one_frame": (
                    maximum_target_boxes
                ),
                "small_box_count": small_box_count,
                "medium_box_count": medium_box_count,
                "large_box_count": large_box_count,
                "mean_box_area": round(
                    mean_box_area,
                    3,
                ),
                "median_box_area": round(
                    median_box_area,
                    3,
                ),
                "level_1_box_count": count_difficulty(
                    difficulty,
                    1,
                ),
                "level_2_box_count": count_difficulty(
                    difficulty,
                    2,
                ),
                "unspecified_difficulty_box_count": (
                    count_unspecified_difficulty(
                        difficulty
                    )
                ),
            }
        )

        print(
            f"Analyzed {file_number}/{len(parquet_files)}: "
            f"{segment_id}"
        )

    front_stats = pd.DataFrame(segment_records)

    # Add the scene metadata and original candidate-selection reasons.
    candidate_metadata_columns = [
        "segment_id",
        "time_of_day",
        "weather",
        "location",
        "density_group",
        "selection_reason",
        "candidate_selection_seed",
    ]

    available_metadata_columns = [
        column
        for column in candidate_metadata_columns
        if column in candidates.columns
    ]

    final_stats = candidates[
        available_metadata_columns
    ].merge(
        front_stats,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    final_stats["contains_front_vehicle"] = (
        final_stats["vehicle_box_count"] > 0
    )

    final_stats["contains_front_pedestrian"] = (
        final_stats["pedestrian_box_count"] > 0
    )

    final_stats["contains_front_cyclist"] = (
        final_stats["cyclist_box_count"] > 0
    )

    final_stats = final_stats.sort_values(
        by=[
            "time_of_day",
            "weather",
            "location",
            "density_group",
            "segment_id",
        ]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_stats.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FRONT-CAMERA CANDIDATE ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"Candidate segments: {len(final_stats)}")

    print(
        "Segments containing FRONT vehicles: "
        f"{int(final_stats['contains_front_vehicle'].sum())}"
    )

    print(
        "Segments containing FRONT pedestrians: "
        f"{int(final_stats['contains_front_pedestrian'].sum())}"
    )

    print(
        "Segments containing FRONT cyclists: "
        f"{int(final_stats['contains_front_cyclist'].sum())}"
    )

    print(
        "Total FRONT vehicle boxes: "
        f"{int(final_stats['vehicle_box_count'].sum())}"
    )

    print(
        "Total FRONT pedestrian boxes: "
        f"{int(final_stats['pedestrian_box_count'].sum())}"
    )

    print(
        "Total FRONT cyclist boxes: "
        f"{int(final_stats['cyclist_box_count'].sum())}"
    )

    print(
        "\nSmall/medium/large FRONT target boxes:"
    )

    print(
        f"Small: "
        f"{int(final_stats['small_box_count'].sum())}"
    )

    print(
        f"Medium: "
        f"{int(final_stats['medium_box_count'].sum())}"
    )

    print(
        f"Large: "
        f"{int(final_stats['large_box_count'].sum())}"
    )

    print(
        f"\nStatistics saved to:\n"
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        f"\nSize thresholds saved to:\n"
        f"{SIZE_THRESHOLDS_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()