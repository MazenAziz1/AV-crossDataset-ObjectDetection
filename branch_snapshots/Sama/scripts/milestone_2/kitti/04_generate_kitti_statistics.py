from pathlib import Path
import json
from typing import Iterator

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm


IMAGE_DIR = Path(
    "data/kitti/raw/training/image_2"
)

LABEL_DIR = Path(
    "data/kitti/raw/training/label_2"
)

MAPPING_FILE = Path(
    "data/kitti/selection/class_mapping.yaml"
)

ASSIGNMENTS_FILE = Path(
    "data/kitti/selection/split_assignments.csv"
)

OUTPUT_DIR = Path(
    "data/kitti/statistics"
)


DATASET_SUMMARY_FILE = (
    OUTPUT_DIR / "dataset_summary.json"
)

IMAGE_STATS_FILE = (
    OUTPUT_DIR / "image_level_statistics.csv"
)

OBJECT_STATS_FILE = (
    OUTPUT_DIR / "object_level_statistics.csv"
)

ORIGINAL_DISTRIBUTION_FILE = (
    OUTPUT_DIR / "original_class_distribution.csv"
)

MAPPED_DISTRIBUTION_FILE = (
    OUTPUT_DIR / "mapped_class_distribution.csv"
)

TRAIN_VAL_DISTRIBUTION_FILE = (
    OUTPUT_DIR / "train_val_distribution.csv"
)

BBOX_STATISTICS_FILE = (
    OUTPUT_DIR / "bbox_size_statistics.csv"
)

BBOX_THRESHOLDS_FILE = (
    OUTPUT_DIR / "bbox_size_thresholds.json"
)

OCCLUSION_STATISTICS_FILE = (
    OUTPUT_DIR / "occlusion_statistics.csv"
)

TRUNCATION_STATISTICS_FILE = (
    OUTPUT_DIR / "truncation_statistics.csv"
)

DIFFICULTY_STATISTICS_FILE = (
    OUTPUT_DIR / "difficulty_statistics.csv"
)


EXPECTED_IMAGE_COUNT = 7481
EXPECTED_ANNOTATION_COUNT = 51865

TARGET_CLASS_NAMES = {
    "Vehicle",
    "Pedestrian",
    "Cyclist",
}

OCCLUSION_NAMES = {
    0: "fully_visible",
    1: "partly_occluded",
    2: "largely_occluded",
    3: "unknown",
}


def load_class_mapping() -> dict:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Class mapping not found:\n"
            f"{MAPPING_FILE.resolve()}"
        )

    with MAPPING_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = yaml.safe_load(file)

    mapping = configuration.get(
        "kitti_mapping"
    )

    if not mapping:
        raise ValueError(
            "class_mapping.yaml has no "
            "kitti_mapping section."
        )

    return mapping


def dataset_partitions(
    dataframe: pd.DataFrame,
) -> Iterator[tuple[str, pd.DataFrame]]:
    yield "all", dataframe
    yield (
        "train",
        dataframe[
            dataframe["split"] == "train"
        ],
    )
    yield (
        "val",
        dataframe[
            dataframe["split"] == "val"
        ],
    )


def calculate_kitti_difficulty(
    class_name: str,
    truncation: float,
    occlusion: int,
    bbox_height: float,
) -> str:
    """
    Descriptive KITTI difficulty classification.

    This does not replace the final evaluation protocol.
    """
    if class_name == "DontCare":
        return "not_applicable"

    if truncation < 0 or occlusion < 0:
        return "not_applicable"

    if (
        bbox_height >= 40
        and occlusion == 0
        and truncation <= 0.15
    ):
        return "easy"

    if (
        bbox_height >= 25
        and occlusion <= 1
        and truncation <= 0.30
    ):
        return "moderate"

    if (
        bbox_height >= 25
        and occlusion <= 2
        and truncation <= 0.50
    ):
        return "hard"

    return "outside_standard_difficulty"


def truncation_group(
    truncation: float,
) -> str:
    if truncation < 0:
        return "not_applicable"

    if truncation == 0:
        return "none"

    if truncation <= 0.15:
        return "low_0_to_0.15"

    if truncation <= 0.30:
        return "moderate_0.15_to_0.30"

    if truncation <= 0.50:
        return "high_0.30_to_0.50"

    return "very_high_above_0.50"


def safe_percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        100.0 * numerator / denominator,
        4,
    )


def distribution_table(
    dataframe: pd.DataFrame,
    class_column: str,
) -> pd.DataFrame:
    records: list[dict] = []

    for partition_name, partition in dataset_partitions(
        dataframe
    ):
        counts = (
            partition[class_column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        total = int(len(partition))

        for class_name, count in counts.items():
            records.append(
                {
                    "partition": partition_name,
                    "class_name": class_name,
                    "count": int(count),
                    "percentage": safe_percentage(
                        int(count),
                        total,
                    ),
                }
            )

    return pd.DataFrame(records)


def summarize_partition(
    image_dataframe: pd.DataFrame,
    object_dataframe: pd.DataFrame,
    partition_name: str,
) -> dict:
    if partition_name == "all":
        images = image_dataframe
        objects = object_dataframe
    else:
        images = image_dataframe[
            image_dataframe["split"]
            == partition_name
        ]

        objects = object_dataframe[
            object_dataframe["split"]
            == partition_name
        ]

    target_objects = objects[
        objects["is_target_class"]
    ]

    ignored_objects = objects[
        ~objects["is_target_class"]
    ]

    return {
        "images": int(len(images)),
        "original_annotations": int(
            len(objects)
        ),
        "target_boxes": int(
            len(target_objects)
        ),
        "ignored_boxes": int(
            len(ignored_objects)
        ),
        "vehicle_boxes": int(
            (
                target_objects[
                    "mapped_class_name"
                ]
                == "Vehicle"
            ).sum()
        ),
        "pedestrian_boxes": int(
            (
                target_objects[
                    "mapped_class_name"
                ]
                == "Pedestrian"
            ).sum()
        ),
        "cyclist_boxes": int(
            (
                target_objects[
                    "mapped_class_name"
                ]
                == "Cyclist"
            ).sum()
        ),
        "images_containing_vehicle": int(
            images["contains_vehicle"].sum()
        ),
        "images_containing_pedestrian": int(
            images[
                "contains_pedestrian"
            ].sum()
        ),
        "images_containing_cyclist": int(
            images["contains_cyclist"].sum()
        ),
        "target_empty_images": int(
            images["target_empty"].sum()
        ),
        "ignored_only_images": int(
            images["ignored_only"].sum()
        ),
        "average_target_boxes_per_image": round(
            float(
                images[
                    "target_box_count"
                ].mean()
            ),
            6,
        ),
        "maximum_target_boxes_in_image": int(
            images[
                "target_box_count"
            ].max()
        ),
        "density_distribution": {
            str(key): int(value)
            for key, value in images[
                "density_group"
            ].value_counts().items()
        },
    }


def main() -> None:
    required_paths = [
        IMAGE_DIR,
        LABEL_DIR,
        MAPPING_FILE,
        ASSIGNMENTS_FILE,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n"
                f"{path.resolve()}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_mapping = load_class_mapping()

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={"image_id": str},
    )

    assignments["image_id"] = (
        assignments["image_id"]
        .astype(str)
        .str.zfill(6)
    )

    if len(assignments) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT} "
            f"split assignments, but found "
            f"{len(assignments)}."
        )

    if assignments["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate image IDs were found "
            "inside split_assignments.csv."
        )

    assignment_lookup = (
        assignments
        .set_index("image_id")
        .to_dict(orient="index")
    )

    image_records: list[dict] = []
    object_records: list[dict] = []

    label_files = sorted(
        LABEL_DIR.glob("*.txt")
    )

    print("=" * 72)
    print("GENERATING KITTI DATASET STATISTICS")
    print("=" * 72)

    for label_file in tqdm(
        label_files,
        unit="image",
    ):
        image_id = label_file.stem

        if image_id not in assignment_lookup:
            raise KeyError(
                f"No split assignment found "
                f"for image {image_id}."
            )

        assignment = assignment_lookup[
            image_id
        ]

        image_file = (
            IMAGE_DIR / f"{image_id}.png"
        )

        if not image_file.exists():
            raise FileNotFoundError(
                f"Image file not found:\n"
                f"{image_file.resolve()}"
            )

        with Image.open(image_file) as image:
            image_width, image_height = (
                image.size
            )

        vehicle_count = 0
        pedestrian_count = 0
        cyclist_count = 0
        ignored_count = 0
        original_annotation_count = 0

        lines = label_file.read_text(
            encoding="utf-8",
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
                    f"Invalid field count in "
                    f"{label_file.name}, "
                    f"line {line_number}."
                )

            original_annotation_count += 1

            original_class = fields[0]

            mapping_entry = class_mapping.get(
                original_class
            )

            if mapping_entry is None:
                raise KeyError(
                    f"Unmapped class "
                    f"{original_class} in "
                    f"{label_file.name}."
                )

            action = mapping_entry["action"]

            is_target_class = (
                action == "map"
            )

            if is_target_class:
                mapped_class_id = int(
                    mapping_entry[
                        "mapped_class_id"
                    ]
                )

                mapped_class_name = str(
                    mapping_entry[
                        "mapped_class_name"
                    ]
                )
            else:
                mapped_class_id = None
                mapped_class_name = None
                ignored_count += 1

            truncation = float(fields[1])
            occlusion = int(fields[2])
            alpha = float(fields[3])

            xmin = float(fields[4])
            ymin = float(fields[5])
            xmax = float(fields[6])
            ymax = float(fields[7])

            object_height_3d = float(
                fields[8]
            )
            object_width_3d = float(
                fields[9]
            )
            object_length_3d = float(
                fields[10]
            )

            location_x = float(fields[11])
            location_y = float(fields[12])
            location_z = float(fields[13])

            rotation_y = float(fields[14])

            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            bbox_area = (
                bbox_width * bbox_height
            )

            image_area = (
                image_width * image_height
            )

            normalized_bbox_area = (
                bbox_area / image_area
                if image_area > 0
                else 0.0
            )

            if mapped_class_name == "Vehicle":
                vehicle_count += 1
            elif mapped_class_name == "Pedestrian":
                pedestrian_count += 1
            elif mapped_class_name == "Cyclist":
                cyclist_count += 1

            if occlusion in OCCLUSION_NAMES:
                occlusion_name = (
                    OCCLUSION_NAMES[occlusion]
                )
            elif occlusion < 0:
                occlusion_name = (
                    "not_applicable"
                )
            else:
                occlusion_name = (
                    "unexpected"
                )

            mapped_class_for_stats = (
                mapped_class_name
                if is_target_class
                else "Ignored"
            )

            object_records.append(
                {
                    "image_id": image_id,
                    "split": assignment[
                        "split"
                    ],
                    "line_number": line_number,
                    "original_class": (
                        original_class
                    ),
                    "mapping_action": action,
                    "is_target_class": (
                        is_target_class
                    ),
                    "mapped_class_id": (
                        mapped_class_id
                    ),
                    "mapped_class_name": (
                        mapped_class_name
                    ),
                    "mapped_class_for_stats": (
                        mapped_class_for_stats
                    ),
                    "truncation": truncation,
                    "truncation_group": (
                        truncation_group(
                            truncation
                        )
                    ),
                    "occlusion": occlusion,
                    "occlusion_name": (
                        occlusion_name
                    ),
                    "alpha": alpha,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area": bbox_area,
                    "normalized_bbox_area": (
                        normalized_bbox_area
                    ),
                    "image_width": (
                        image_width
                    ),
                    "image_height": (
                        image_height
                    ),
                    "object_height_3d": (
                        object_height_3d
                    ),
                    "object_width_3d": (
                        object_width_3d
                    ),
                    "object_length_3d": (
                        object_length_3d
                    ),
                    "location_x": location_x,
                    "location_y": location_y,
                    "location_z": location_z,
                    "rotation_y": rotation_y,
                    "difficulty_group": (
                        calculate_kitti_difficulty(
                            original_class,
                            truncation,
                            occlusion,
                            bbox_height,
                        )
                    ),
                }
            )

        target_box_count = (
            vehicle_count
            + pedestrian_count
            + cyclist_count
        )

        image_records.append(
            {
                "image_id": image_id,
                "split": assignment[
                    "split"
                ],
                "image_width": (
                    image_width
                ),
                "image_height": (
                    image_height
                ),
                "original_annotation_count": (
                    original_annotation_count
                ),
                "target_box_count": (
                    target_box_count
                ),
                "ignored_box_count": (
                    ignored_count
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
                "contains_vehicle": (
                    vehicle_count > 0
                ),
                "contains_pedestrian": (
                    pedestrian_count > 0
                ),
                "contains_cyclist": (
                    cyclist_count > 0
                ),
                "target_empty": (
                    target_box_count == 0
                ),
                "ignored_only": (
                    target_box_count == 0
                    and ignored_count > 0
                ),
                "presence_signature": (
                    assignment[
                        "presence_signature"
                    ]
                ),
                "density_group": (
                    assignment[
                        "density_group"
                    ]
                ),
                "stratification_group": (
                    assignment[
                        "stratification_group"
                    ]
                ),
            }
        )

    image_dataframe = pd.DataFrame(
        image_records
    )

    object_dataframe = pd.DataFrame(
        object_records
    )

    if len(image_dataframe) != (
        EXPECTED_IMAGE_COUNT
    ):
        raise ValueError(
            "Generated image-level table "
            "has an unexpected row count."
        )

    if len(object_dataframe) != (
        EXPECTED_ANNOTATION_COUNT
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_ANNOTATION_COUNT} "
            f"object rows, but generated "
            f"{len(object_dataframe)}."
        )

    # -----------------------------------------------------
    # Define descriptive object-size groups using the
    # normalized area of mapped target boxes.
    # -----------------------------------------------------
    target_mask = (
        object_dataframe[
            "is_target_class"
        ]
    )

    target_normalized_areas = (
        object_dataframe.loc[
            target_mask,
            "normalized_bbox_area",
        ]
        .astype(float)
        .to_numpy()
    )

    lower_size_threshold = float(
        np.quantile(
            target_normalized_areas,
            1 / 3,
        )
    )

    upper_size_threshold = float(
        np.quantile(
            target_normalized_areas,
            2 / 3,
        )
    )

    def assign_size_group(
        row: pd.Series,
    ) -> str:
        if not bool(
            row["is_target_class"]
        ):
            return "not_applicable"

        area = float(
            row["normalized_bbox_area"]
        )

        if area <= lower_size_threshold:
            return "small"

        if area <= upper_size_threshold:
            return "medium"

        return "large"

    object_dataframe[
        "bbox_size_group"
    ] = object_dataframe.apply(
        assign_size_group,
        axis=1,
    )

    bbox_thresholds = {
        "method": (
            "target_box_normalized_area_quantiles"
        ),
        "normalized_area_definition": (
            "bbox_area / image_area"
        ),
        "small_definition": (
            "normalized_area <= "
            "lower_33_percent_quantile"
        ),
        "medium_definition": (
            "lower_33_percent_quantile < "
            "normalized_area <= "
            "upper_67_percent_quantile"
        ),
        "large_definition": (
            "normalized_area > "
            "upper_67_percent_quantile"
        ),
        "small_max_normalized_area": (
            lower_size_threshold
        ),
        "medium_max_normalized_area": (
            upper_size_threshold
        ),
        "target_boxes_used": int(
            target_mask.sum()
        ),
    }

    BBOX_THRESHOLDS_FILE.write_text(
        json.dumps(
            bbox_thresholds,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Save primary row-level tables.
    # -----------------------------------------------------
    image_dataframe = (
        image_dataframe
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    object_dataframe = (
        object_dataframe
        .sort_values(
            [
                "image_id",
                "line_number",
            ]
        )
        .reset_index(drop=True)
    )

    image_dataframe.to_csv(
        IMAGE_STATS_FILE,
        index=False,
    )

    object_dataframe.to_csv(
        OBJECT_STATS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Original and mapped class distributions.
    # -----------------------------------------------------
    original_distribution = (
        distribution_table(
            object_dataframe,
            "original_class",
        )
    )

    original_distribution.to_csv(
        ORIGINAL_DISTRIBUTION_FILE,
        index=False,
    )

    mapped_distribution = (
        distribution_table(
            object_dataframe,
            "mapped_class_for_stats",
        )
    )

    mapped_distribution.to_csv(
        MAPPED_DISTRIBUTION_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Train/validation summary table.
    # -----------------------------------------------------
    partition_records = []

    for partition_name in [
        "all",
        "train",
        "val",
    ]:
        partition_summary = (
            summarize_partition(
                image_dataframe,
                object_dataframe,
                partition_name,
            )
        )

        partition_records.append(
            {
                "partition": (
                    partition_name
                ),
                **{
                    key: value
                    for key, value
                    in partition_summary.items()
                    if not isinstance(
                        value,
                        dict,
                    )
                },
            }
        )

    train_val_distribution = (
        pd.DataFrame(
            partition_records
        )
    )

    train_val_distribution.to_csv(
        TRAIN_VAL_DISTRIBUTION_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Bounding-box statistics for mapped targets.
    # -----------------------------------------------------
    target_objects = object_dataframe[
        object_dataframe[
            "is_target_class"
        ]
    ].copy()

    bbox_records: list[dict] = []

    for partition_name, partition in dataset_partitions(
        target_objects
    ):
        for mapped_class, group in partition.groupby(
            "mapped_class_name"
        ):
            bbox_records.append(
                {
                    "partition": (
                        partition_name
                    ),
                    "mapped_class": (
                        mapped_class
                    ),
                    "box_count": int(
                        len(group)
                    ),
                    "mean_width": round(
                        float(
                            group[
                                "bbox_width"
                            ].mean()
                        ),
                        6,
                    ),
                    "median_width": round(
                        float(
                            group[
                                "bbox_width"
                            ].median()
                        ),
                        6,
                    ),
                    "mean_height": round(
                        float(
                            group[
                                "bbox_height"
                            ].mean()
                        ),
                        6,
                    ),
                    "median_height": round(
                        float(
                            group[
                                "bbox_height"
                            ].median()
                        ),
                        6,
                    ),
                    "mean_area": round(
                        float(
                            group[
                                "bbox_area"
                            ].mean()
                        ),
                        6,
                    ),
                    "median_area": round(
                        float(
                            group[
                                "bbox_area"
                            ].median()
                        ),
                        6,
                    ),
                    "mean_normalized_area": round(
                        float(
                            group[
                                "normalized_bbox_area"
                            ].mean()
                        ),
                        10,
                    ),
                    "median_normalized_area": round(
                        float(
                            group[
                                "normalized_bbox_area"
                            ].median()
                        ),
                        10,
                    ),
                    "small_box_count": int(
                        (
                            group[
                                "bbox_size_group"
                            ]
                            == "small"
                        ).sum()
                    ),
                    "medium_box_count": int(
                        (
                            group[
                                "bbox_size_group"
                            ]
                            == "medium"
                        ).sum()
                    ),
                    "large_box_count": int(
                        (
                            group[
                                "bbox_size_group"
                            ]
                            == "large"
                        ).sum()
                    ),
                }
            )

    pd.DataFrame(
        bbox_records
    ).to_csv(
        BBOX_STATISTICS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Occlusion, truncation, and difficulty tables.
    # -----------------------------------------------------
    occlusion_records = []
    truncation_records = []
    difficulty_records = []

    for partition_name, partition in dataset_partitions(
        target_objects
    ):
        occlusion_counts = (
            partition.groupby(
                [
                    "mapped_class_name",
                    "occlusion_name",
                ]
            )
            .size()
            .reset_index(name="count")
        )

        for _, row in occlusion_counts.iterrows():
            occlusion_records.append(
                {
                    "partition": (
                        partition_name
                    ),
                    "mapped_class": row[
                        "mapped_class_name"
                    ],
                    "occlusion_group": row[
                        "occlusion_name"
                    ],
                    "count": int(
                        row["count"]
                    ),
                }
            )

        truncation_counts = (
            partition.groupby(
                [
                    "mapped_class_name",
                    "truncation_group",
                ]
            )
            .size()
            .reset_index(name="count")
        )

        for _, row in truncation_counts.iterrows():
            truncation_records.append(
                {
                    "partition": (
                        partition_name
                    ),
                    "mapped_class": row[
                        "mapped_class_name"
                    ],
                    "truncation_group": row[
                        "truncation_group"
                    ],
                    "count": int(
                        row["count"]
                    ),
                }
            )

        difficulty_counts = (
            partition.groupby(
                [
                    "mapped_class_name",
                    "difficulty_group",
                ]
            )
            .size()
            .reset_index(name="count")
        )

        for _, row in difficulty_counts.iterrows():
            difficulty_records.append(
                {
                    "partition": (
                        partition_name
                    ),
                    "mapped_class": row[
                        "mapped_class_name"
                    ],
                    "difficulty_group": row[
                        "difficulty_group"
                    ],
                    "count": int(
                        row["count"]
                    ),
                }
            )

    pd.DataFrame(
        occlusion_records
    ).to_csv(
        OCCLUSION_STATISTICS_FILE,
        index=False,
    )

    pd.DataFrame(
        truncation_records
    ).to_csv(
        TRUNCATION_STATISTICS_FILE,
        index=False,
    )

    pd.DataFrame(
        difficulty_records
    ).to_csv(
        DIFFICULTY_STATISTICS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Final validation and JSON summary.
    # -----------------------------------------------------
    full_summary = summarize_partition(
        image_dataframe,
        object_dataframe,
        "all",
    )

    train_summary = summarize_partition(
        image_dataframe,
        object_dataframe,
        "train",
    )

    val_summary = summarize_partition(
        image_dataframe,
        object_dataframe,
        "val",
    )

    validation_checks = {
        "image_rows_equal_7481": (
            len(image_dataframe)
            == EXPECTED_IMAGE_COUNT
        ),
        "object_rows_equal_51865": (
            len(object_dataframe)
            == EXPECTED_ANNOTATION_COUNT
        ),
        "train_images_equal_5985": (
            train_summary["images"]
            == 5985
        ),
        "validation_images_equal_1496": (
            val_summary["images"]
            == 1496
        ),
        "overall_target_boxes_equal_39086": (
            full_summary[
                "target_boxes"
            ]
            == 39086
        ),
        "overall_ignored_boxes_equal_12779": (
            full_summary[
                "ignored_boxes"
            ]
            == 12779
        ),
        "all_target_classes_present_train": all(
            train_summary[
                f"{class_name.lower()}_boxes"
            ] > 0
            for class_name
            in TARGET_CLASS_NAMES
        ),
        "all_target_classes_present_validation": all(
            val_summary[
                f"{class_name.lower()}_boxes"
            ] > 0
            for class_name
            in TARGET_CLASS_NAMES
        ),
    }

    statistics_validation_passed = all(
        validation_checks.values()
    )

    summary = {
        "dataset": (
            "KITTI Object Detection"
        ),
        "source_subset": (
            "official labeled training set"
        ),
        "official_testing_set_used": False,
        "class_mapping": {
            "Vehicle": [
                "Car",
                "Van",
                "Truck",
            ],
            "Pedestrian": [
                "Pedestrian",
                "Person_sitting",
            ],
            "Cyclist": [
                "Cyclist",
            ],
            "Ignored": [
                "Tram",
                "Misc",
                "DontCare",
            ],
        },
        "all": full_summary,
        "train": train_summary,
        "validation": val_summary,
        "bbox_size_thresholds": (
            bbox_thresholds
        ),
        "validation_checks": (
            validation_checks
        ),
        "statistics_validation_passed": (
            statistics_validation_passed
        ),
        "notes": [
            (
                "Target-empty images are retained "
                "for background and false-positive "
                "evaluation."
            ),
            (
                "Ignored-only images are retained, "
                "but ignored annotations are not "
                "treated as target classes."
            ),
            (
                "DontCare regions remain available "
                "for later evaluation handling."
            ),
        ],
    }

    DATASET_SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("KITTI STATISTICS GENERATION COMPLETE")
    print("=" * 72)

    print(
        f"Image-level rows: "
        f"{len(image_dataframe)}"
    )

    print(
        f"Object-level rows: "
        f"{len(object_dataframe)}"
    )

    print("\nComplete dataset:")

    print(
        f"  Target boxes: "
        f"{full_summary['target_boxes']}"
    )

    print(
        f"  Ignored boxes: "
        f"{full_summary['ignored_boxes']}"
    )

    print(
        f"  Target-empty images: "
        f"{full_summary['target_empty_images']}"
    )

    print(
        f"  Ignored-only images: "
        f"{full_summary['ignored_only_images']}"
    )

    print("\nTraining split:")

    print(
        f"  Images: "
        f"{train_summary['images']}"
    )

    print(
        f"  Vehicle boxes: "
        f"{train_summary['vehicle_boxes']}"
    )

    print(
        f"  Pedestrian boxes: "
        f"{train_summary['pedestrian_boxes']}"
    )

    print(
        f"  Cyclist boxes: "
        f"{train_summary['cyclist_boxes']}"
    )

    print("\nValidation split:")

    print(
        f"  Images: "
        f"{val_summary['images']}"
    )

    print(
        f"  Vehicle boxes: "
        f"{val_summary['vehicle_boxes']}"
    )

    print(
        f"  Pedestrian boxes: "
        f"{val_summary['pedestrian_boxes']}"
    )

    print(
        f"  Cyclist boxes: "
        f"{val_summary['cyclist_boxes']}"
    )

    print(
        "\nStatistics status: "
        + (
            "PASSED"
            if statistics_validation_passed
            else "FAILED"
        )
    )

    print(
        f"\nDataset summary:\n"
        f"{DATASET_SUMMARY_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()