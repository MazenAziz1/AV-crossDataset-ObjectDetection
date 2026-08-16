from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit


LABEL_DIR = Path(
    "data/kitti/raw/training/label_2"
)

MAPPING_FILE = Path(
    "data/kitti/selection/class_mapping.yaml"
)

OUTPUT_DIR = Path(
    "data/kitti/selection"
)

TRAIN_FILE = OUTPUT_DIR / "train.txt"
VAL_FILE = OUTPUT_DIR / "val.txt"
ASSIGNMENTS_FILE = OUTPUT_DIR / "split_assignments.csv"
SUMMARY_FILE = OUTPUT_DIR / "split_summary.json"


RANDOM_SEED = 42
TOTAL_EXPECTED_IMAGES = 7481
VALIDATION_IMAGE_COUNT = 1496
TRAIN_IMAGE_COUNT = (
    TOTAL_EXPECTED_IMAGES
    - VALIDATION_IMAGE_COUNT
)


TARGET_CLASSES = [
    "Vehicle",
    "Pedestrian",
    "Cyclist",
]


def load_mapping() -> dict:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Mapping file not found:\n"
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
            "No kitti_mapping section found "
            "inside class_mapping.yaml."
        )

    return mapping


def inspect_label_file(
    label_file: Path,
    class_mapping: dict,
) -> dict:
    counts = {
        "Vehicle": 0,
        "Pedestrian": 0,
        "Cyclist": 0,
    }

    ignored_count = 0
    original_annotation_count = 0

    lines = label_file.read_text(
        encoding="utf-8",
    ).splitlines()

    for line in lines:
        if not line.strip():
            continue

        original_annotation_count += 1
        original_class = line.split()[0]

        mapping_entry = class_mapping.get(
            original_class
        )

        if mapping_entry is None:
            raise KeyError(
                f"Unmapped KITTI class "
                f"'{original_class}' in "
                f"{label_file.name}"
            )

        action = mapping_entry.get("action")

        if action == "ignore":
            ignored_count += 1
            continue

        if action != "map":
            raise ValueError(
                f"Unknown mapping action "
                f"'{action}' for "
                f"{original_class}"
            )

        mapped_class = mapping_entry.get(
            "mapped_class_name"
        )

        if mapped_class not in counts:
            raise ValueError(
                f"Unexpected mapped class: "
                f"{mapped_class}"
            )

        counts[mapped_class] += 1

    target_count = sum(counts.values())

    presence_signature = (
        f"V{int(counts['Vehicle'] > 0)}_"
        f"P{int(counts['Pedestrian'] > 0)}_"
        f"C{int(counts['Cyclist'] > 0)}"
    )

    return {
        "image_id": label_file.stem,
        "vehicle_count": counts["Vehicle"],
        "pedestrian_count": (
            counts["Pedestrian"]
        ),
        "cyclist_count": counts["Cyclist"],
        "target_count": target_count,
        "ignored_count": ignored_count,
        "original_annotation_count": (
            original_annotation_count
        ),
        "contains_vehicle": (
            counts["Vehicle"] > 0
        ),
        "contains_pedestrian": (
            counts["Pedestrian"] > 0
        ),
        "contains_cyclist": (
            counts["Cyclist"] > 0
        ),
        "presence_signature": (
            presence_signature
        ),
    }


def assign_density_groups(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    dataframe = dataframe.copy()

    positive_counts = dataframe.loc[
        dataframe["target_count"] > 0,
        "target_count",
    ].to_numpy(dtype=float)

    if len(positive_counts) == 0:
        raise ValueError(
            "No mapped target objects were found."
        )

    lower_threshold = float(
        np.quantile(
            positive_counts,
            1 / 3,
        )
    )

    upper_threshold = float(
        np.quantile(
            positive_counts,
            2 / 3,
        )
    )

    def density_group(
        target_count: int,
    ) -> str:
        if target_count == 0:
            return "target_empty"

        if target_count <= lower_threshold:
            return "low"

        if target_count <= upper_threshold:
            return "medium"

        return "high"

    dataframe["density_group"] = (
        dataframe["target_count"]
        .apply(density_group)
    )

    thresholds = {
        "method": (
            "positive_target_count_quantiles"
        ),
        "target_empty_definition": (
            "target_count == 0"
        ),
        "low_max_target_count": (
            lower_threshold
        ),
        "medium_max_target_count": (
            upper_threshold
        ),
    }

    return dataframe, thresholds


def create_stratification_labels(
    dataframe: pd.DataFrame,
) -> pd.Series:
    combined = (
        dataframe["presence_signature"]
        + "__"
        + dataframe["density_group"]
    )

    combined_counts = (
        combined.value_counts()
    )

    # Very rare combined groups are reduced to
    # class-presence signatures.
    labels = combined.copy()

    rare_combined = set(
        combined_counts[
            combined_counts < 2
        ].index
    )

    labels.loc[
        labels.isin(rare_combined)
    ] = dataframe.loc[
        labels.isin(rare_combined),
        "presence_signature",
    ]

    # Final fallback in the unlikely event that
    # a presence signature occurs only once.
    label_counts = labels.value_counts()

    rare_labels = set(
        label_counts[
            label_counts < 2
        ].index
    )

    labels.loc[
        labels.isin(rare_labels)
    ] = "other_rare"

    if (
        labels.value_counts().min()
        < 2
    ):
        raise ValueError(
            "At least one stratification group "
            "still contains fewer than two images."
        )

    return labels


def summarize_split(
    dataframe: pd.DataFrame,
) -> dict:
    result = {
        "images": int(len(dataframe)),
        "target_boxes": int(
            dataframe["target_count"].sum()
        ),
        "ignored_boxes": int(
            dataframe["ignored_count"].sum()
        ),
        "vehicle_boxes": int(
            dataframe["vehicle_count"].sum()
        ),
        "pedestrian_boxes": int(
            dataframe[
                "pedestrian_count"
            ].sum()
        ),
        "cyclist_boxes": int(
            dataframe["cyclist_count"].sum()
        ),
        "images_containing_vehicle": int(
            dataframe[
                "contains_vehicle"
            ].sum()
        ),
        "images_containing_pedestrian": int(
            dataframe[
                "contains_pedestrian"
            ].sum()
        ),
        "images_containing_cyclist": int(
            dataframe[
                "contains_cyclist"
            ].sum()
        ),
        "target_empty_images": int(
            (
                dataframe["target_count"]
                == 0
            ).sum()
        ),
        "density_distribution": {
            str(key): int(value)
            for key, value
            in dataframe[
                "density_group"
            ].value_counts().items()
        },
        "presence_distribution": {
            str(key): int(value)
            for key, value
            in dataframe[
                "presence_signature"
            ].value_counts().items()
        },
    }

    return result


def write_id_file(
    output_file: Path,
    image_ids: list[str],
) -> None:
    output_file.write_text(
        "\n".join(image_ids) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found:\n"
            f"{LABEL_DIR.resolve()}"
        )

    class_mapping = load_mapping()

    label_files = sorted(
        LABEL_DIR.glob("*.txt")
    )

    if len(label_files) != (
        TOTAL_EXPECTED_IMAGES
    ):
        raise ValueError(
            f"Expected "
            f"{TOTAL_EXPECTED_IMAGES} "
            f"label files, but found "
            f"{len(label_files)}."
        )

    print("=" * 72)
    print("CREATING FIXED KITTI TRAIN/VALIDATION SPLIT")
    print("=" * 72)

    records = []

    for label_file in label_files:
        records.append(
            inspect_label_file(
                label_file,
                class_mapping,
            )
        )

    dataframe = pd.DataFrame(records)

    dataframe, density_thresholds = (
        assign_density_groups(
            dataframe
        )
    )

    dataframe[
        "stratification_group"
    ] = create_stratification_labels(
        dataframe
    )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_IMAGE_COUNT,
        random_state=RANDOM_SEED,
    )

    train_indices, val_indices = next(
        splitter.split(
            dataframe,
            dataframe[
                "stratification_group"
            ],
        )
    )

    dataframe["split"] = ""

    dataframe.loc[
        train_indices,
        "split",
    ] = "train"

    dataframe.loc[
        val_indices,
        "split",
    ] = "val"

    train_dataframe = dataframe[
        dataframe["split"] == "train"
    ].copy()

    val_dataframe = dataframe[
        dataframe["split"] == "val"
    ].copy()

    train_ids = sorted(
        train_dataframe[
            "image_id"
        ].astype(str).tolist()
    )

    val_ids = sorted(
        val_dataframe[
            "image_id"
        ].astype(str).tolist()
    )

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    all_id_set = set(
        dataframe[
            "image_id"
        ].astype(str)
    )

    overlap = sorted(
        train_id_set
        & val_id_set
    )

    missing_ids = sorted(
        all_id_set
        - train_id_set
        - val_id_set
    )

    validation_checks = {
        "expected_total_images": (
            TOTAL_EXPECTED_IMAGES
        ),
        "actual_total_images": int(
            len(dataframe)
        ),
        "expected_train_images": (
            TRAIN_IMAGE_COUNT
        ),
        "actual_train_images": int(
            len(train_dataframe)
        ),
        "expected_validation_images": (
            VALIDATION_IMAGE_COUNT
        ),
        "actual_validation_images": int(
            len(val_dataframe)
        ),
        "train_validation_overlap": (
            len(overlap)
        ),
        "unassigned_images": (
            len(missing_ids)
        ),
        "all_target_classes_in_train": all(
            train_dataframe[
                f"{class_name.lower()}_count"
            ].sum() > 0
            for class_name
            in TARGET_CLASSES
        ),
        "all_target_classes_in_validation": all(
            val_dataframe[
                f"{class_name.lower()}_count"
            ].sum() > 0
            for class_name
            in TARGET_CLASSES
        ),
    }

    split_validation_passed = (
        validation_checks[
            "actual_total_images"
        ]
        == TOTAL_EXPECTED_IMAGES
        and validation_checks[
            "actual_train_images"
        ]
        == TRAIN_IMAGE_COUNT
        and validation_checks[
            "actual_validation_images"
        ]
        == VALIDATION_IMAGE_COUNT
        and validation_checks[
            "train_validation_overlap"
        ]
        == 0
        and validation_checks[
            "unassigned_images"
        ]
        == 0
        and validation_checks[
            "all_target_classes_in_train"
        ]
        and validation_checks[
            "all_target_classes_in_validation"
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_id_file(
        TRAIN_FILE,
        train_ids,
    )

    write_id_file(
        VAL_FILE,
        val_ids,
    )

    dataframe = dataframe.sort_values(
        "image_id"
    ).reset_index(drop=True)

    dataframe.to_csv(
        ASSIGNMENTS_FILE,
        index=False,
    )

    summary = {
        "dataset": (
            "KITTI Object Detection"
        ),
        "source_subset": (
            "official labeled training set"
        ),
        "official_testing_set_used": False,
        "split_method": (
            "stratified_shuffle_split"
        ),
        "stratification_features": [
            "Vehicle presence",
            "Pedestrian presence",
            "Cyclist presence",
            "Mapped target-object density",
        ],
        "random_seed": RANDOM_SEED,
        "total_images": int(
            len(dataframe)
        ),
        "train_ratio": (
            len(train_dataframe)
            / len(dataframe)
        ),
        "validation_ratio": (
            len(val_dataframe)
            / len(dataframe)
        ),
        "density_thresholds": (
            density_thresholds
        ),
        "train": summarize_split(
            train_dataframe
        ),
        "validation": summarize_split(
            val_dataframe
        ),
        "validation_checks": (
            validation_checks
        ),
        "split_validation_passed": (
            split_validation_passed
        ),
        "split_frozen_before_training": (
            True
        ),
        "notes": [
            (
                "The same split must be used "
                "for all detector families."
            ),
            (
                "The official KITTI testing "
                "set is excluded because its "
                "ground-truth labels are not "
                "publicly available."
            ),
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nTotal images: "
        f"{len(dataframe)}"
    )

    print(
        f"Training images: "
        f"{len(train_dataframe)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataframe)}"
    )

    print("\nTraining mapped box counts:")

    print(
        f"  Vehicle: "
        f"{train_dataframe['vehicle_count'].sum()}"
    )

    print(
        f"  Pedestrian: "
        f"{train_dataframe['pedestrian_count'].sum()}"
    )

    print(
        f"  Cyclist: "
        f"{train_dataframe['cyclist_count'].sum()}"
    )

    print("\nValidation mapped box counts:")

    print(
        f"  Vehicle: "
        f"{val_dataframe['vehicle_count'].sum()}"
    )

    print(
        f"  Pedestrian: "
        f"{val_dataframe['pedestrian_count'].sum()}"
    )

    print(
        f"  Cyclist: "
        f"{val_dataframe['cyclist_count'].sum()}"
    )

    print("\nValidation density distribution:")

    print(
        val_dataframe[
            "density_group"
        ].value_counts()
    )

    print(
        "\nSplit status: "
        + (
            "PASSED"
            if split_validation_passed
            else "FAILED"
        )
    )

    print(
        f"\nTrain IDs:\n"
        f"{TRAIN_FILE.resolve()}"
    )

    print(
        f"\nValidation IDs:\n"
        f"{VAL_FILE.resolve()}"
    )

    print(
        f"\nAssignments:\n"
        f"{ASSIGNMENTS_FILE.resolve()}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_FILE.resolve()}"
    )

    print(
        "\nThe split is now frozen and "
        "must not be changed after "
        "model training begins."
    )


if __name__ == "__main__":
    main()