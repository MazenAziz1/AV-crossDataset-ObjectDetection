from io import BytesIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


FINAL_SEGMENTS_FILE = Path(
    "data/waymo/selection/final_segments.csv"
)

CAMERA_IMAGE_DIR = Path(
    "data/waymo/raw/validation/camera_image/final"
)

CAMERA_BOX_DIR = Path(
    "data/waymo/raw/validation/camera_box/candidates"
)

REPRESENTATIVE_ROOT = Path(
    "data/waymo/representative_subset"
)

OUTPUT_IMAGE_ROOT = (
    REPRESENTATIVE_ROOT / "images" / "front"
)

OUTPUT_ANNOTATIONS_DIR = (
    REPRESENTATIVE_ROOT / "annotations"
)

OUTPUT_METADATA_DIR = (
    REPRESENTATIVE_ROOT / "metadata"
)

MANIFEST_FILE = (
    OUTPUT_METADATA_DIR / "manifest.csv"
)

BOXES_FILE = (
    OUTPUT_ANNOTATIONS_DIR / "boxes.csv"
)

CLASS_MAPPING_FILE = (
    OUTPUT_ANNOTATIONS_DIR / "class_mapping.yaml"
)

SUMMARY_FILE = (
    OUTPUT_METADATA_DIR / "subset_summary.json"
)


# Uniform temporal sampling:
# select frame positions 0, 5, 10, 15, ...
FRAME_INTERVAL = 5

FRONT_CAMERA_ID = 1
FRONT_CAMERA_NAME = "FRONT"


# Camera-image columns discovered in Step 11.
SEGMENT_COLUMN = "key.segment_context_name"
TIMESTAMP_COLUMN = "key.frame_timestamp_micros"
CAMERA_COLUMN = "key.camera_name"

IMAGE_BYTES_COLUMN = (
    "[CameraImageComponent].image"
)


# Camera-box columns discovered in Step 7.
OBJECT_ID_COLUMN = "key.camera_object_id"

CENTER_X_COLUMN = (
    "[CameraBoxComponent].box.center.x"
)

CENTER_Y_COLUMN = (
    "[CameraBoxComponent].box.center.y"
)

BOX_WIDTH_COLUMN = (
    "[CameraBoxComponent].box.size.x"
)

BOX_HEIGHT_COLUMN = (
    "[CameraBoxComponent].box.size.y"
)

TYPE_COLUMN = (
    "[CameraBoxComponent].type"
)

DETECTION_DIFFICULTY_COLUMN = (
    "[CameraBoxComponent].difficulty_level.detection"
)


IMAGE_COLUMNS = [
    SEGMENT_COLUMN,
    TIMESTAMP_COLUMN,
    CAMERA_COLUMN,
    IMAGE_BYTES_COLUMN,
]

BOX_COLUMNS_TO_READ = [
    SEGMENT_COLUMN,
    TIMESTAMP_COLUMN,
    CAMERA_COLUMN,
    OBJECT_ID_COLUMN,
    CENTER_X_COLUMN,
    CENTER_Y_COLUMN,
    BOX_WIDTH_COLUMN,
    BOX_HEIGHT_COLUMN,
    TYPE_COLUMN,
    DETECTION_DIFFICULTY_COLUMN,
]


# Waymo type ID:
# 1 = Vehicle
# 2 = Pedestrian
# 3 = Sign, ignored
# 4 = Cyclist
TYPE_MAPPING = {
    1: {
        "original_name": "Vehicle",
        "mapped_class_id": 0,
        "mapped_class_name": "Vehicle",
    },
    2: {
        "original_name": "Pedestrian",
        "mapped_class_id": 1,
        "mapped_class_name": "Pedestrian",
    },
    4: {
        "original_name": "Cyclist",
        "mapped_class_id": 2,
        "mapped_class_name": "Cyclist",
    },
}

TARGET_TYPE_IDS = set(TYPE_MAPPING)


MANIFEST_COLUMNS = [
    "image_id",
    "final_segment_number",
    "segment_id",
    "frame_timestamp_micros",
    "camera_id",
    "camera_name",
    "source_front_frame_index",
    "selected_frame_number_in_segment",
    "image_filename",
    "relative_image_path",
    "image_width",
    "image_height",
    "number_of_target_boxes",
    "vehicle_count",
    "pedestrian_count",
    "cyclist_count",
    "time_of_day",
    "weather",
    "location",
    "density_group",
    "sampling_rule",
]


BOX_OUTPUT_COLUMNS = [
    "image_id",
    "segment_id",
    "frame_timestamp_micros",
    "camera_id",
    "camera_name",
    "object_id",
    "original_type_id",
    "original_type_name",
    "mapped_class_id",
    "mapped_class_name",
    "center_x",
    "center_y",
    "box_width",
    "box_height",
    "raw_xmin",
    "raw_ymin",
    "raw_xmax",
    "raw_ymax",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "clipped_box_width",
    "clipped_box_height",
    "clipped_box_area",
    "image_width",
    "image_height",
    "detection_difficulty_raw",
    "detection_difficulty_name",
]


def read_front_component(
    parquet_file: Path,
    columns: list[str],
) -> pd.DataFrame:
    """
    Read only FRONT-camera rows when predicate filtering is
    supported. Fall back to normal filtering if needed.
    """
    try:
        table = pq.read_table(
            parquet_file,
            columns=columns,
            filters=[
                (
                    CAMERA_COLUMN,
                    "=",
                    FRONT_CAMERA_ID,
                )
            ],
        )

        dataframe = table.to_pandas()

    except Exception:
        table = pq.read_table(
            parquet_file,
            columns=columns,
        )

        dataframe = table.to_pandas()

        dataframe = dataframe[
            dataframe[CAMERA_COLUMN]
            == FRONT_CAMERA_ID
        ].copy()

    return dataframe


def normalize_binary_image(value) -> bytes:
    """Convert Arrow/Pandas binary values to Python bytes."""
    if isinstance(value, bytes):
        return value

    if isinstance(value, bytearray):
        return bytes(value)

    if isinstance(value, memoryview):
        return value.tobytes()

    if hasattr(value, "tobytes"):
        return value.tobytes()

    raise TypeError(
        f"Unsupported image binary type: {type(value)}"
    )


def save_image_and_get_size(
    image_bytes: bytes,
    output_file: Path,
) -> tuple[int, int]:
    """
    Preserve original JPEG bytes whenever possible and return
    the image width and height.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        image_format = (
            image.format.upper()
            if image.format
            else ""
        )

        if image_format in {"JPEG", "JPG"}:
            output_file.write_bytes(image_bytes)
        else:
            image.convert("RGB").save(
                output_file,
                format="JPEG",
                quality=95,
            )

    return int(width), int(height)


def difficulty_name(value) -> str:
    """
    Missing and level-zero values are treated as unspecified.
    """
    if pd.isna(value):
        return "UNSPECIFIED"

    numeric = int(value)

    if numeric == 1:
        return "LEVEL_1"

    if numeric == 2:
        return "LEVEL_2"

    return "UNSPECIFIED"


def difficulty_raw_value(value):
    if pd.isna(value):
        return None

    return int(value)


def clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return max(
        lower,
        min(float(value), upper),
    )


def value_counts_dict(
    series: pd.Series,
) -> dict:
    counts = series.value_counts(
        dropna=False
    )

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def write_class_mapping() -> None:
    content = """final_classes:
  0: Vehicle
  1: Pedestrian
  2: Cyclist

waymo_mapping:
  1:
    original_name: Vehicle
    mapped_class_id: 0
    mapped_class_name: Vehicle

  2:
    original_name: Pedestrian
    mapped_class_id: 1
    mapped_class_name: Pedestrian

  3:
    original_name: Sign
    mapped_class_id: null
    mapped_class_name: ignore

  4:
    original_name: Cyclist
    mapped_class_id: 2
    mapped_class_name: Cyclist

camera:
  id: 1
  name: FRONT

notes:
  - Waymo validation is used only for external evaluation.
  - No Waymo images are used for model training or tuning.
  - Signs are excluded from the harmonized three-class task.
  - Frames are sampled uniformly every fifth FRONT-camera frame.
"""

    CLASS_MAPPING_FILE.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    if not FINAL_SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Final segment file not found:\n"
            f"{FINAL_SEGMENTS_FILE.resolve()}"
        )

    final_segments = pd.read_csv(
        FINAL_SEGMENTS_FILE
    )

    if len(final_segments) != 25:
        raise ValueError(
            "Expected exactly 25 frozen final segments, "
            f"but found {len(final_segments)}."
        )

    required_final_columns = [
        "segment_id",
        "final_segment_number",
        "time_of_day",
        "weather",
        "location",
        "density_group",
    ]

    missing_final_columns = [
        column
        for column in required_final_columns
        if column not in final_segments.columns
    ]

    if missing_final_columns:
        raise KeyError(
            "Missing columns in final_segments.csv:\n"
            + "\n".join(missing_final_columns)
        )

    OUTPUT_IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ANNOTATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_records: list[dict] = []
    box_records: list[dict] = []

    invalid_box_count = 0

    print("=" * 72)
    print("EXTRACTING WAYMO REPRESENTATIVE SUBSET")
    print("=" * 72)

    print(f"Final segments: {len(final_segments)}")
    print(f"Camera: {FRONT_CAMERA_NAME}")
    print(
        f"Sampling: every {FRAME_INTERVAL}th "
        "chronological FRONT frame"
    )

    final_segments = final_segments.sort_values(
        "final_segment_number"
    ).reset_index(drop=True)

    for segment_position, segment_row in final_segments.iterrows():
        segment_id = str(
            segment_row["segment_id"]
        )

        final_segment_number = int(
            segment_row["final_segment_number"]
        )

        image_parquet_file = (
            CAMERA_IMAGE_DIR
            / f"{segment_id}.parquet"
        )

        box_parquet_file = (
            CAMERA_BOX_DIR
            / f"{segment_id}.parquet"
        )

        if not image_parquet_file.exists():
            raise FileNotFoundError(
                f"Missing image Parquet file:\n"
                f"{image_parquet_file.resolve()}"
            )

        if not box_parquet_file.exists():
            raise FileNotFoundError(
                f"Missing camera-box Parquet file:\n"
                f"{box_parquet_file.resolve()}"
            )

        print(
            f"\n[{segment_position + 1}/"
            f"{len(final_segments)}] "
            f"{segment_id}"
        )

        front_images = read_front_component(
            image_parquet_file,
            IMAGE_COLUMNS,
        )

        if front_images.empty:
            raise ValueError(
                f"No FRONT-camera images found for "
                f"segment {segment_id}."
            )

        front_images = (
            front_images
            .sort_values(TIMESTAMP_COLUMN)
            .drop_duplicates(
                subset=[TIMESTAMP_COLUMN],
                keep="first",
            )
            .reset_index(drop=True)
        )

        selected_images = (
            front_images.iloc[::FRAME_INTERVAL]
            .copy()
        )

        front_boxes = read_front_component(
            box_parquet_file,
            BOX_COLUMNS_TO_READ,
        )

        target_boxes = front_boxes[
            front_boxes[TYPE_COLUMN]
            .isin(TARGET_TYPE_IDS)
        ].copy()

        target_boxes[TIMESTAMP_COLUMN] = (
            pd.to_numeric(
                target_boxes[TIMESTAMP_COLUMN],
                errors="raise",
            ).astype("int64")
        )

        box_groups = {
            int(timestamp): group.copy()
            for timestamp, group
            in target_boxes.groupby(
                TIMESTAMP_COLUMN
            )
        }

        segment_output_dir = (
            OUTPUT_IMAGE_ROOT / segment_id
        )

        segment_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected_count = 0

        for source_frame_index, image_row in selected_images.iterrows():
            timestamp = int(
                image_row[TIMESTAMP_COLUMN]
            )

            image_id = (
                f"{segment_id}_{timestamp}"
            )

            image_filename = (
                f"{timestamp}.jpg"
            )

            output_image_file = (
                segment_output_dir
                / image_filename
            )

            image_bytes = normalize_binary_image(
                image_row[IMAGE_BYTES_COLUMN]
            )

            image_width, image_height = (
                save_image_and_get_size(
                    image_bytes,
                    output_image_file,
                )
            )

            image_boxes = box_groups.get(
                timestamp,
                pd.DataFrame(
                    columns=BOX_COLUMNS_TO_READ
                ),
            )

            vehicle_count = 0
            pedestrian_count = 0
            cyclist_count = 0
            valid_target_box_count = 0

            for _, box_row in image_boxes.iterrows():
                type_id = int(
                    box_row[TYPE_COLUMN]
                )

                type_information = (
                    TYPE_MAPPING[type_id]
                )

                center_x = float(
                    box_row[CENTER_X_COLUMN]
                )

                center_y = float(
                    box_row[CENTER_Y_COLUMN]
                )

                box_width = float(
                    box_row[BOX_WIDTH_COLUMN]
                )

                box_height = float(
                    box_row[BOX_HEIGHT_COLUMN]
                )

                if (
                    not np.isfinite(center_x)
                    or not np.isfinite(center_y)
                    or not np.isfinite(box_width)
                    or not np.isfinite(box_height)
                    or box_width <= 0
                    or box_height <= 0
                ):
                    invalid_box_count += 1
                    continue

                raw_xmin = (
                    center_x - box_width / 2.0
                )

                raw_ymin = (
                    center_y - box_height / 2.0
                )

                raw_xmax = (
                    center_x + box_width / 2.0
                )

                raw_ymax = (
                    center_y + box_height / 2.0
                )

                xmin = clamp(
                    raw_xmin,
                    0.0,
                    float(image_width),
                )

                ymin = clamp(
                    raw_ymin,
                    0.0,
                    float(image_height),
                )

                xmax = clamp(
                    raw_xmax,
                    0.0,
                    float(image_width),
                )

                ymax = clamp(
                    raw_ymax,
                    0.0,
                    float(image_height),
                )

                clipped_width = max(
                    0.0,
                    xmax - xmin,
                )

                clipped_height = max(
                    0.0,
                    ymax - ymin,
                )

                clipped_area = (
                    clipped_width
                    * clipped_height
                )

                if clipped_area <= 0:
                    invalid_box_count += 1
                    continue

                mapped_class_name = (
                    type_information[
                        "mapped_class_name"
                    ]
                )

                if mapped_class_name == "Vehicle":
                    vehicle_count += 1
                elif mapped_class_name == "Pedestrian":
                    pedestrian_count += 1
                elif mapped_class_name == "Cyclist":
                    cyclist_count += 1

                valid_target_box_count += 1

                difficulty_value = box_row[
                    DETECTION_DIFFICULTY_COLUMN
                ]

                box_records.append(
                    {
                        "image_id": image_id,
                        "segment_id": segment_id,
                        "frame_timestamp_micros": timestamp,
                        "camera_id": FRONT_CAMERA_ID,
                        "camera_name": FRONT_CAMERA_NAME,
                        "object_id": str(
                            box_row[OBJECT_ID_COLUMN]
                        ),
                        "original_type_id": type_id,
                        "original_type_name": (
                            type_information[
                                "original_name"
                            ]
                        ),
                        "mapped_class_id": (
                            type_information[
                                "mapped_class_id"
                            ]
                        ),
                        "mapped_class_name": (
                            mapped_class_name
                        ),
                        "center_x": round(
                            center_x,
                            6,
                        ),
                        "center_y": round(
                            center_y,
                            6,
                        ),
                        "box_width": round(
                            box_width,
                            6,
                        ),
                        "box_height": round(
                            box_height,
                            6,
                        ),
                        "raw_xmin": round(
                            raw_xmin,
                            6,
                        ),
                        "raw_ymin": round(
                            raw_ymin,
                            6,
                        ),
                        "raw_xmax": round(
                            raw_xmax,
                            6,
                        ),
                        "raw_ymax": round(
                            raw_ymax,
                            6,
                        ),
                        "xmin": round(
                            xmin,
                            6,
                        ),
                        "ymin": round(
                            ymin,
                            6,
                        ),
                        "xmax": round(
                            xmax,
                            6,
                        ),
                        "ymax": round(
                            ymax,
                            6,
                        ),
                        "clipped_box_width": round(
                            clipped_width,
                            6,
                        ),
                        "clipped_box_height": round(
                            clipped_height,
                            6,
                        ),
                        "clipped_box_area": round(
                            clipped_area,
                            6,
                        ),
                        "image_width": image_width,
                        "image_height": image_height,
                        "detection_difficulty_raw": (
                            difficulty_raw_value(
                                difficulty_value
                            )
                        ),
                        "detection_difficulty_name": (
                            difficulty_name(
                                difficulty_value
                            )
                        ),
                    }
                )

            selected_count += 1

            relative_image_path = (
                output_image_file.relative_to(
                    REPRESENTATIVE_ROOT
                ).as_posix()
            )

            manifest_records.append(
                {
                    "image_id": image_id,
                    "final_segment_number": (
                        final_segment_number
                    ),
                    "segment_id": segment_id,
                    "frame_timestamp_micros": timestamp,
                    "camera_id": FRONT_CAMERA_ID,
                    "camera_name": FRONT_CAMERA_NAME,
                    "source_front_frame_index": int(
                        source_frame_index
                    ),
                    "selected_frame_number_in_segment": (
                        selected_count
                    ),
                    "image_filename": image_filename,
                    "relative_image_path": (
                        relative_image_path
                    ),
                    "image_width": image_width,
                    "image_height": image_height,
                    "number_of_target_boxes": (
                        valid_target_box_count
                    ),
                    "vehicle_count": vehicle_count,
                    "pedestrian_count": (
                        pedestrian_count
                    ),
                    "cyclist_count": cyclist_count,
                    "time_of_day": str(
                        segment_row[
                            "time_of_day"
                        ]
                    ),
                    "weather": str(
                        segment_row["weather"]
                    ),
                    "location": str(
                        segment_row["location"]
                    ),
                    "density_group": str(
                        segment_row[
                            "density_group"
                        ]
                    ),
                    "sampling_rule": (
                        "every_5th_front_frame_"
                        "starting_at_first"
                    ),
                }
            )

        print(
            f"  FRONT frames available: "
            f"{len(front_images)}"
        )

        print(
            f"  Frames selected: "
            f"{len(selected_images)}"
        )

    manifest = pd.DataFrame(
        manifest_records,
        columns=MANIFEST_COLUMNS,
    )

    boxes = pd.DataFrame(
        box_records,
        columns=BOX_OUTPUT_COLUMNS,
    )

    manifest = manifest.sort_values(
        by=[
            "final_segment_number",
            "frame_timestamp_micros",
        ]
    ).reset_index(drop=True)

    boxes = boxes.sort_values(
        by=[
            "segment_id",
            "frame_timestamp_micros",
            "mapped_class_id",
            "object_id",
        ]
    ).reset_index(drop=True)

    manifest.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    boxes.to_csv(
        BOXES_FILE,
        index=False,
    )

    write_class_mapping()

    images_without_target_boxes = int(
        (
            manifest["number_of_target_boxes"]
            == 0
        ).sum()
    )

    summary = {
        "source_dataset": (
            "Waymo Open Dataset v2.0.1"
        ),
        "source_split": "validation",
        "purpose": (
            "external_validation_without_retraining"
        ),
        "camera_id": FRONT_CAMERA_ID,
        "camera_name": FRONT_CAMERA_NAME,
        "number_of_segments": int(
            manifest["segment_id"].nunique()
        ),
        "frame_sampling_interval": (
            FRAME_INTERVAL
        ),
        "frame_sampling_rule": (
            "every_5th_front_frame_"
            "starting_at_first"
        ),
        "number_of_selected_images": int(
            len(manifest)
        ),
        "images_without_target_boxes": (
            images_without_target_boxes
        ),
        "number_of_target_boxes": int(
            len(boxes)
        ),
        "vehicle_box_count": int(
            (
                boxes["mapped_class_name"]
                == "Vehicle"
            ).sum()
        ),
        "pedestrian_box_count": int(
            (
                boxes["mapped_class_name"]
                == "Pedestrian"
            ).sum()
        ),
        "cyclist_box_count": int(
            (
                boxes["mapped_class_name"]
                == "Cyclist"
            ).sum()
        ),
        "invalid_or_zero_area_boxes_skipped": (
            int(invalid_box_count)
        ),
        "time_of_day_image_distribution": (
            value_counts_dict(
                manifest["time_of_day"]
            )
        ),
        "weather_image_distribution": (
            value_counts_dict(
                manifest["weather"]
            )
        ),
        "location_image_distribution": (
            value_counts_dict(
                manifest["location"]
            )
        ),
        "density_image_distribution": (
            value_counts_dict(
                manifest["density_group"]
            )
        ),
        "selection_frozen_before_model_evaluation": (
            True
        ),
        "manifest_file": (
            MANIFEST_FILE.as_posix()
        ),
        "boxes_file": BOXES_FILE.as_posix(),
        "class_mapping_file": (
            CLASS_MAPPING_FILE.as_posix()
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("REPRESENTATIVE SUBSET EXTRACTION COMPLETE")
    print("=" * 72)

    print(
        f"Segments represented: "
        f"{manifest['segment_id'].nunique()}"
    )

    print(
        f"Selected FRONT images: "
        f"{len(manifest)}"
    )

    print(
        f"Images without target boxes: "
        f"{images_without_target_boxes}"
    )

    print(
        f"Total retained boxes: "
        f"{len(boxes)}"
    )

    print(
        "Vehicle boxes: "
        f"{int((boxes['mapped_class_name'] == 'Vehicle').sum())}"
    )

    print(
        "Pedestrian boxes: "
        f"{int((boxes['mapped_class_name'] == 'Pedestrian').sum())}"
    )

    print(
        "Cyclist boxes: "
        f"{int((boxes['mapped_class_name'] == 'Cyclist').sum())}"
    )

    print(
        f"Invalid boxes skipped: "
        f"{invalid_box_count}"
    )

    print(
        f"\nManifest:\n"
        f"{MANIFEST_FILE.resolve()}"
    )

    print(
        f"\nBounding boxes:\n"
        f"{BOXES_FILE.resolve()}"
    )

    print(
        f"\nClass mapping:\n"
        f"{CLASS_MAPPING_FILE.resolve()}"
    )

    print(
        f"\nSubset summary:\n"
        f"{SUMMARY_FILE.resolve()}"
    )

    print("\nStep 12 completed successfully.")


if __name__ == "__main__":
    main()