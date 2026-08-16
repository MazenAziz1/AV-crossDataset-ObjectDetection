from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import sys

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from augmentation_core import (
    apply_training_augmentation,
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

AUGMENTATION_CONFIG = Path(
    "configs/datasets/milestone_3/"
    "augmentation.yaml"
)

SOURCE_MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/source_manifest.csv"
)

TRANSFORM_MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/transform_manifest.csv"
)

KITTI_TRAIN_COCO_FILE = (
    PROCESSED_ROOT
    / "annotations/coco/kitti_train.json"
)

VISUAL_CHECK_REPORT = (
    PROCESSED_ROOT
    / "reports/"
    "visual_annotation_checks_report.json"
)

VISUAL_DIR = (
    PROCESSED_ROOT
    / "visual_checks/"
    "augmentation_policy"
)

MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/"
    "augmentation_policy_manifest.csv"
)

REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/"
    "augmentation_policy_report.json"
)

ISSUES_FILE = (
    PROCESSED_ROOT
    / "reports/"
    "augmentation_policy_issues.csv"
)


EXPECTED_SAMPLE_COUNT = 8

CLASS_NAMES = {
    0: "Vehicle",
    1: "Pedestrian",
    2: "Cyclist",
}

COCO_TO_INTERNAL = {
    1: 0,
    2: 1,
    3: 2,
}

CLASS_COLORS = {
    0: (0, 210, 0),
    1: (0, 165, 255),
    2: (255, 120, 0),
}


MANIFEST_COLUMNS = [
    "sample_number",
    "selection_reason",
    "global_image_id",
    "source_image_id",
    "image_filename",
    "epoch",
    "derived_seed",
    "required_operation",
    "operations_applied",
    "original_box_count",
    "augmented_box_count",
    "class_counts_match",
    "dimensions_valid",
    "boxes_valid",
    "padding_valid",
    "deterministic_repeat",
    "comparison_path",
    "sample_passed",
]


# ============================================================
# HELPERS
# ============================================================

def load_yaml(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required YAML file not found:\n"
            f"{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "YAML root must be a mapping."
        )

    return data


def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found:\n"
            f"{path.resolve()}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            "JSON root must be an object."
        )

    return data


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


def parse_bool(
    value,
) -> bool:
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def verify_boxes(
    boxes: np.ndarray,
    width: int,
    height: int,
) -> bool:
    if boxes.ndim != 2:
        return False

    if boxes.shape[1] != 4:
        return False

    if not np.isfinite(boxes).all():
        return False

    if len(boxes) == 0:
        return True

    return bool(
        np.all(boxes[:, 0] >= 0.0)
        and np.all(boxes[:, 1] >= 0.0)
        and np.all(
            boxes[:, 2] <= width
        )
        and np.all(
            boxes[:, 3] <= height
        )
        and np.all(
            boxes[:, 2]
            > boxes[:, 0]
        )
        and np.all(
            boxes[:, 3]
            > boxes[:, 1]
        )
    )


def verify_padding(
    image: np.ndarray,
    transform_row: pd.Series,
    padding_value: int,
    horizontal_flip_applied: bool,
) -> bool:
    top = int(
        transform_row["padding_top"]
    )

    bottom = int(
        transform_row["padding_bottom"]
    )

    left = int(
        transform_row["padding_left"]
    )

    right = int(
        transform_row["padding_right"]
    )

    if horizontal_flip_applied:
        left, right = right, left

    expected = np.asarray(
        [
            padding_value,
            padding_value,
            padding_value,
        ],
        dtype=np.uint8,
    )

    checks = []

    if top > 0:
        checks.append(
            bool(
                np.all(
                    image[:top, :]
                    == expected
                )
            )
        )

    if bottom > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        image.shape[0]
                        - bottom:,
                        :,
                    ]
                    == expected
                )
            )
        )

    if left > 0:
        checks.append(
            bool(
                np.all(
                    image[:, :left]
                    == expected
                )
            )
        )

    if right > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        :,
                        image.shape[1]
                        - right:,
                    ]
                    == expected
                )
            )
        )

    return all(checks) if checks else True


def choose_unique(
    dataframe: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_ids: set[int],
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.Series | None:
    available = candidates[
        ~candidates[
            "global_image_id"
        ].isin(selected_ids)
    ].copy()

    if available.empty:
        return None

    available = available.sort_values(
        sort_columns,
        ascending=ascending,
    )

    return available.iloc[0]


def select_samples(
    dataframe: pd.DataFrame,
    issues: list[dict],
) -> list[
    tuple[str, pd.Series]
]:
    selected = []
    selected_ids: set[int] = set()

    train = dataframe[
        (
            dataframe["dataset"]
            == "KITTI"
        )
        & (
            dataframe["partition"]
            == "train"
        )
    ].copy()

    train[
        "all_three_classes"
    ] = (
        (train["vehicle_count"] > 0)
        & (
            train["pedestrian_count"]
            > 0
        )
        & (
            train["cyclist_count"] > 0
        )
    )

    median_count = float(
        train[
            "target_box_count"
        ].median()
    )

    train[
        "median_distance"
    ] = (
        train["target_box_count"]
        - median_count
    ).abs()

    criteria = [
        (
            "crowded",
            train,
            ["target_box_count", "global_image_id"],
            [False, True],
        ),
        (
            "vehicle_rich",
            train[
                train["vehicle_count"] > 0
            ],
            ["vehicle_count", "global_image_id"],
            [False, True],
        ),
        (
            "pedestrian_rich",
            train[
                train["pedestrian_count"] > 0
            ],
            [
                "pedestrian_count",
                "global_image_id",
            ],
            [False, True],
        ),
        (
            "cyclist_rich",
            train[
                train["cyclist_count"] > 0
            ],
            [
                "cyclist_count",
                "global_image_id",
            ],
            [False, True],
        ),
        (
            "mixed_three_classes",
            train[
                train[
                    "all_three_classes"
                ]
            ],
            ["target_box_count", "global_image_id"],
            [False, True],
        ),
        (
            "low_density",
            train,
            ["target_box_count", "global_image_id"],
            [True, True],
        ),
        (
            "median_density",
            train,
            ["median_distance", "global_image_id"],
            [True, True],
        ),
        (
            "deterministic_reference",
            train,
            ["global_image_id"],
            [True],
        ),
    ]

    for (
        reason,
        candidates,
        sort_columns,
        ascending,
    ) in criteria:
        row = choose_unique(
            dataframe=dataframe,
            candidates=candidates,
            selected_ids=selected_ids,
            sort_columns=sort_columns,
            ascending=ascending,
        )

        if row is None:
            add_issue(
                issues,
                "sample_selection_failed",
                reason,
                "No unique training image was available.",
            )
            continue

        image_id = int(
            row["global_image_id"]
        )

        selected_ids.add(image_id)
        selected.append((reason, row))

    return selected


def load_coco_records() -> tuple[
    dict[int, dict],
    dict[int, list[dict]],
]:
    data = load_json(
        KITTI_TRAIN_COCO_FILE
    )

    image_lookup = {
        int(image["id"]): image
        for image in data["images"]
    }

    annotations_by_image: dict[
        int,
        list[dict],
    ] = defaultdict(list)

    for annotation in data[
        "annotations"
    ]:
        annotations_by_image[
            int(annotation["image_id"])
        ].append(annotation)

    return (
        image_lookup,
        dict(annotations_by_image),
    )


def annotation_arrays(
    annotations: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    class_ids = []

    for annotation in annotations:
        x, y, width, height = [
            float(value)
            for value
            in annotation["bbox"]
        ]

        boxes.append(
            [
                x,
                y,
                x + width,
                y + height,
            ]
        )

        class_ids.append(
            COCO_TO_INTERNAL[
                int(
                    annotation[
                        "category_id"
                    ]
                )
            ]
        )

    if not boxes:
        return (
            np.empty(
                (0, 4),
                dtype=np.float64,
            ),
            np.empty(
                (0,),
                dtype=np.int64,
            ),
        )

    return (
        np.asarray(
            boxes,
            dtype=np.float64,
        ),
        np.asarray(
            class_ids,
            dtype=np.int64,
        ),
    )


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    title: str,
    operations: list[str],
) -> np.ndarray:
    output = image.copy()

    for box, class_id in zip(
        boxes,
        class_ids,
    ):
        xmin, ymin, xmax, ymax = [
            int(
                math.floor(
                    float(value) + 0.5
                )
            )
            for value in box
        ]

        class_id = int(class_id)
        class_name = CLASS_NAMES[
            class_id
        ]

        color = CLASS_COLORS[
            class_id
        ]

        cv2.rectangle(
            output,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            output,
            class_name,
            (
                xmin,
                max(18, ymin - 4),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 34),
        (25, 25, 25),
        -1,
    )

    operation_text = (
        ", ".join(operations)
        if operations
        else "none"
    )

    cv2.putText(
        output,
        (
            f"{title} | boxes={len(boxes)} "
            f"| ops={operation_text}"
        ),
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )

    return output


def make_comparison(
    original: np.ndarray,
    augmented: np.ndarray,
    reason: str,
    epoch: int,
    seed: int,
) -> np.ndarray:
    header_height = 58

    comparison = np.full(
        (
            original.shape[0]
            + header_height,
            original.shape[1] * 2,
            3,
        ),
        30,
        dtype=np.uint8,
    )

    comparison[
        header_height:,
        :original.shape[1],
    ] = original

    comparison[
        header_height:,
        original.shape[1]:,
    ] = augmented

    cv2.putText(
        comparison,
        "Original processed image",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        comparison,
        "Augmented training view",
        (
            original.shape[1] + 10,
            25,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    subtitle = (
        f"{reason} | epoch={epoch} "
        f"| derived seed={seed}"
    )

    cv2.putText(
        comparison,
        subtitle,
        (10, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        lineType=cv2.LINE_AA,
    )

    return comparison


def save_png(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not success:
        raise RuntimeError(
            f"Could not write image:\n"
            f"{path.resolve()}"
        )


def find_demonstration_epoch(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    configuration: dict,
    global_image_id: int,
    required_operation: str | None,
    starting_epoch: int,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    for offset in range(2000):
        epoch = (
            starting_epoch + offset
        )

        (
            augmented_image,
            augmented_boxes,
            augmented_classes,
            trace,
        ) = apply_training_augmentation(
            image=image,
            boxes_xyxy=boxes,
            class_ids=class_ids,
            configuration=configuration,
            global_image_id=(
                global_image_id
            ),
            epoch=epoch,
        )

        operations = set(
            trace[
                "operations_applied"
            ]
        )

        suitable = (
            required_operation
            in operations
            if required_operation
            is not None
            else len(operations) > 0
        )

        if suitable:
            return (
                epoch,
                augmented_image,
                augmented_boxes,
                augmented_classes,
                trace,
            )

    raise RuntimeError(
        "Could not find a deterministic preview seed "
        f"for operation: {required_operation}"
    )


def create_contact_sheet(
    comparison_paths: list[Path],
    output_path: Path,
) -> None:
    thumbnails = []

    for path in comparison_paths:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"Could not decode:\n{path}"
            )

        thumbnail = cv2.resize(
            image,
            (960, 524),
            interpolation=cv2.INTER_AREA,
        )

        thumbnails.append(thumbnail)

    if len(thumbnails) != 8:
        raise ValueError(
            "Exactly eight comparisons are required."
        )

    rows = []

    for index in range(0, 8, 2):
        rows.append(
            np.hstack(
                [
                    thumbnails[index],
                    thumbnails[index + 1],
                ]
            )
        )

    sheet = np.vstack(rows)

    save_png(
        output_path,
        sheet,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    VISUAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []
    manifest_rows: list[dict] = []
    comparison_paths: list[Path] = []

    print("=" * 76)
    print("VALIDATING FROZEN AUGMENTATION POLICY")
    print("=" * 76)

    visual_report = load_json(
        VISUAL_CHECK_REPORT
    )

    if not visual_report.get(
        "visual_annotation_checks_passed",
        False,
    ):
        raise RuntimeError(
            "Step 13 visual annotation checks "
            "have not passed."
        )

    configuration = load_yaml(
        AUGMENTATION_CONFIG
    )

    required_partition_policy = {
        "enabled": [
            "kitti_train",
        ],
        "disabled": [
            "kitti_val",
            "waymo_external",
        ],
    }

    partition_policy_valid = (
        configuration[
            "partition_policy"
        ]["enabled"]
        == required_partition_policy[
            "enabled"
        ]
        and configuration[
            "partition_policy"
        ]["disabled"]
        == required_partition_policy[
            "disabled"
        ]
    )

    if not partition_policy_valid:
        add_issue(
            issues,
            "partition_policy_invalid",
            "augmentation.yaml",
            str(
                configuration[
                    "partition_policy"
                ]
            ),
        )

    source_manifest = pd.read_csv(
        SOURCE_MANIFEST_FILE,
        dtype={
            "source_image_id": str,
            "output_filename": str,
            "output_relative_path": str,
        },
    )

    transform_manifest = pd.read_csv(
        TRANSFORM_MANIFEST_FILE,
        dtype={
            "source_image_id": str,
        },
    )

    numeric_columns = [
        "global_image_id",
        "target_box_count",
        "vehicle_count",
        "pedestrian_count",
        "cyclist_count",
    ]

    for column in numeric_columns:
        source_manifest[column] = (
            pd.to_numeric(
                source_manifest[column],
                errors="raise",
            )
        )

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

    (
        coco_image_lookup,
        annotations_by_image,
    ) = load_coco_records()

    selected_samples = select_samples(
        source_manifest,
        issues,
    )

    print(
        f"\nSelected KITTI training samples: "
        f"{len(selected_samples)}"
    )

    required_operations = [
        "horizontal_flip",
        "brightness_contrast",
        "hsv_adjustment",
        "gaussian_blur",
        None,
        None,
        None,
        None,
    ]

    operation_counts: Counter = Counter()

    for sample_number, (
        reason,
        source_row,
    ) in enumerate(
        tqdm(
            selected_samples,
            unit="image",
        ),
        start=1,
    ):
        global_image_id = int(
            source_row[
                "global_image_id"
            ]
        )

        image_filename = str(
            source_row[
                "output_filename"
            ]
        )

        image_path = (
            PROCESSED_ROOT
            / Path(
                str(
                    source_row[
                        "output_relative_path"
                    ]
                )
            )
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            add_issue(
                issues,
                "image_decode_failed",
                str(global_image_id),
                str(image_path),
            )
            continue

        if global_image_id not in (
            coco_image_lookup
        ):
            add_issue(
                issues,
                "missing_coco_image",
                str(global_image_id),
                image_filename,
            )
            continue

        annotations = (
            annotations_by_image.get(
                global_image_id,
                [],
            )
        )

        boxes, class_ids = (
            annotation_arrays(
                annotations
            )
        )

        required_operation = (
            required_operations[
                sample_number - 1
            ]
        )

        try:
            (
                epoch,
                augmented_image,
                augmented_boxes,
                augmented_class_ids,
                trace,
            ) = find_demonstration_epoch(
                image=image,
                boxes=boxes,
                class_ids=class_ids,
                configuration=configuration,
                global_image_id=(
                    global_image_id
                ),
                required_operation=(
                    required_operation
                ),
                starting_epoch=(
                    sample_number * 1000
                ),
            )

        except Exception as error:
            add_issue(
                issues,
                "augmentation_failed",
                str(global_image_id),
                str(error),
            )
            continue

        # Re-run the same seed to prove determinism.
        (
            repeat_image,
            repeat_boxes,
            repeat_classes,
            repeat_trace,
        ) = apply_training_augmentation(
            image=image,
            boxes_xyxy=boxes,
            class_ids=class_ids,
            configuration=configuration,
            global_image_id=(
                global_image_id
            ),
            epoch=epoch,
        )

        deterministic_repeat = (
            np.array_equal(
                augmented_image,
                repeat_image,
            )
            and np.array_equal(
                augmented_boxes,
                repeat_boxes,
            )
            and np.array_equal(
                augmented_class_ids,
                repeat_classes,
            )
            and trace[
                "derived_seed"
            ]
            == repeat_trace[
                "derived_seed"
            ]
        )

        dimensions_valid = (
            augmented_image.shape
            == image.shape
            == (640, 640, 3)
        )

        boxes_valid = verify_boxes(
            augmented_boxes,
            width=640,
            height=640,
        )

        original_class_counts = (
            Counter(
                int(value)
                for value in class_ids
            )
        )

        augmented_class_counts = (
            Counter(
                int(value)
                for value
                in augmented_class_ids
            )
        )

        class_counts_match = (
            original_class_counts
            == augmented_class_counts
        )

        horizontal_flip_applied = (
            "horizontal_flip"
            in trace[
                "operations_applied"
            ]
        )

        transform_row = (
            transform_lookup.loc[
                global_image_id
            ]
        )

        padding_valid = verify_padding(
            image=augmented_image,
            transform_row=transform_row,
            padding_value=int(
                configuration[
                    "letterbox_padding"
                ]["value"]
            ),
            horizontal_flip_applied=(
                horizontal_flip_applied
            ),
        )

        count_valid = (
            len(boxes)
            == len(augmented_boxes)
            == int(
                source_row[
                    "target_box_count"
                ]
            )
        )

        sample_passed = all(
            [
                dimensions_valid,
                boxes_valid,
                class_counts_match,
                padding_valid,
                deterministic_repeat,
                count_valid,
            ]
        )

        if not sample_passed:
            add_issue(
                issues,
                "augmentation_sample_failed",
                str(global_image_id),
                str(
                    {
                        "dimensions_valid": (
                            dimensions_valid
                        ),
                        "boxes_valid": (
                            boxes_valid
                        ),
                        "class_counts_match": (
                            class_counts_match
                        ),
                        "padding_valid": (
                            padding_valid
                        ),
                        "deterministic_repeat": (
                            deterministic_repeat
                        ),
                        "count_valid": (
                            count_valid
                        ),
                    }
                ),
            )

        for operation in trace[
            "operations_applied"
        ]:
            operation_counts[
                operation
            ] += 1

        original_visual = draw_boxes(
            image=image,
            boxes=boxes,
            class_ids=class_ids,
            title="Original",
            operations=[],
        )

        augmented_visual = draw_boxes(
            image=augmented_image,
            boxes=augmented_boxes,
            class_ids=(
                augmented_class_ids
            ),
            title="Augmented",
            operations=trace[
                "operations_applied"
            ],
        )

        comparison = make_comparison(
            original=original_visual,
            augmented=augmented_visual,
            reason=reason,
            epoch=epoch,
            seed=int(
                trace["derived_seed"]
            ),
        )

        comparison_path = (
            VISUAL_DIR
            / (
                f"{sample_number:02d}_"
                f"{reason}__"
                f"{Path(image_filename).stem}"
                ".png"
            )
        )

        save_png(
            comparison_path,
            comparison,
        )

        comparison_paths.append(
            comparison_path
        )

        manifest_rows.append(
            {
                "sample_number": (
                    sample_number
                ),
                "selection_reason": (
                    reason
                ),
                "global_image_id": (
                    global_image_id
                ),
                "source_image_id": (
                    source_row[
                        "source_image_id"
                    ]
                ),
                "image_filename": (
                    image_filename
                ),
                "epoch": epoch,
                "derived_seed": int(
                    trace[
                        "derived_seed"
                    ]
                ),
                "required_operation": (
                    required_operation
                    if required_operation
                    is not None
                    else "any"
                ),
                "operations_applied": (
                    "|".join(
                        trace[
                            "operations_applied"
                        ]
                    )
                ),
                "original_box_count": (
                    len(boxes)
                ),
                "augmented_box_count": (
                    len(augmented_boxes)
                ),
                "class_counts_match": (
                    class_counts_match
                ),
                "dimensions_valid": (
                    dimensions_valid
                ),
                "boxes_valid": (
                    boxes_valid
                ),
                "padding_valid": (
                    padding_valid
                ),
                "deterministic_repeat": (
                    deterministic_repeat
                ),
                "comparison_path": (
                    comparison_path
                    .as_posix()
                ),
                "sample_passed": (
                    sample_passed
                ),
            }
        )

    contact_sheet_path = (
        VISUAL_DIR
        / "augmentation_policy_contact_sheet.png"
    )

    if (
        len(comparison_paths)
        == EXPECTED_SAMPLE_COUNT
    ):
        try:
            create_contact_sheet(
                comparison_paths,
                contact_sheet_path,
            )

        except Exception as error:
            add_issue(
                issues,
                "contact_sheet_failed",
                "contact_sheet",
                str(error),
            )

    else:
        add_issue(
            issues,
            "contact_sheet_skipped",
            "contact_sheet",
            (
                f"Expected 8 comparisons, "
                f"found "
                f"{len(comparison_paths)}."
            ),
        )

    write_csv(
        MANIFEST_FILE,
        manifest_rows,
        MANIFEST_COLUMNS,
    )

    required_operation_coverage = {
        operation: (
            operation_counts[
                operation
            ] > 0
        )
        for operation in [
            "horizontal_flip",
            "brightness_contrast",
            "hsv_adjustment",
            "gaussian_blur",
        ]
    }

    checks = {
        "configuration_frozen": (
            configuration["status"]
            == "frozen"
        ),

        "online_only": (
            configuration[
                "execution"
            ]["mode"]
            == "online_during_training"
            and not bool(
                configuration[
                    "execution"
                ][
                    "generate_permanent_augmented_dataset"
                ]
            )
        ),

        "partition_policy": (
            partition_policy_valid
        ),

        "sample_count": (
            len(manifest_rows)
            == EXPECTED_SAMPLE_COUNT
        ),

        "all_samples_passed": all(
            bool(row["sample_passed"])
            for row in manifest_rows
        ),

        "all_operations_demonstrated": all(
            required_operation_coverage
            .values()
        ),

        "contact_sheet_exists": (
            contact_sheet_path.exists()
        ),

        "no_dataset_files_modified": True,
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "policy_check_failed",
                check_name,
                "The augmentation-policy check returned false.",
            )

    overall_passed = (
        all(checks.values())
        and len(issues) == 0
    )

    report = {
        "milestone": 3,
        "step": 14,
        "purpose": (
            "Freeze and validate the shared "
            "training augmentation and "
            "image-quality policy."
        ),
        "policy_status": (
            configuration["status"]
        ),
        "execution_mode": (
            configuration[
                "execution"
            ]["mode"]
        ),
        "enabled_partitions": (
            configuration[
                "partition_policy"
            ]["enabled"]
        ),
        "disabled_partitions": (
            configuration[
                "partition_policy"
            ]["disabled"]
        ),
        "created_samples": len(
            manifest_rows
        ),
        "operation_counts": {
            name: int(count)
            for name, count
            in sorted(
                operation_counts.items()
            )
        },
        "required_operation_coverage": (
            required_operation_coverage
        ),
        "checks": checks,
        "issue_count": len(issues),
        "augmentation_policy_passed": (
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
            "category",
            "identifier",
            "details",
        ],
    )

    print("\n" + "=" * 76)
    print("AUGMENTATION POLICY SUMMARY")
    print("=" * 76)

    print(
        f"Policy status: "
        f"{configuration['status']}"
    )

    print(
        "Enabled partition: "
        "kitti_train"
    )

    print(
        "Disabled partitions: "
        "kitti_val, waymo_external"
    )

    for row in manifest_rows:
        print(
            f"\n[{row['sample_number']}] "
            f"{row['selection_reason']}"
        )

        print(
            f"  Operations: "
            f"{row['operations_applied']}"
        )

        print(
            f"  Boxes: "
            f"{row['original_box_count']} "
            f"-> "
            f"{row['augmented_box_count']}"
        )

        print(
            f"  Dimensions valid: "
            f"{row['dimensions_valid']}"
        )

        print(
            f"  Padding valid: "
            f"{row['padding_valid']}"
        )

        print(
            f"  Deterministic repeat: "
            f"{row['deterministic_repeat']}"
        )

        print(
            f"  Status: "
            f"{'PASSED' if row['sample_passed'] else 'FAILED'}"
        )

    print("\nOperation coverage:")

    for operation, covered in (
        required_operation_coverage.items()
    ):
        print(
            f"  {operation}: "
            f"{'YES' if covered else 'NO'}"
        )

    print(
        f"\nSamples created: "
        f"{len(manifest_rows)}"
    )

    print(
        f"Contact sheet created: "
        f"{contact_sheet_path.exists()}"
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
        f"\nVisual checks:\n"
        f"{VISUAL_DIR.resolve()}"
    )

    print(
        f"\nManifest:\n"
        f"{MANIFEST_FILE.resolve()}"
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
            "\nDo not continue until every "
            "augmentation-policy issue is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 14 completed successfully. "
        "Manually inspect the augmentation "
        "contact sheet before continuing."
    )


if __name__ == "__main__":
    main()