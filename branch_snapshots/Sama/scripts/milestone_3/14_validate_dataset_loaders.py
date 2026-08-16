from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import platform
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dataset_core import (
    Milestone3DetectionDataset,
    detection_collate_fn,
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

AUGMENTATION_CONFIG_FILE = Path(
    "configs/datasets/milestone_3/"
    "augmentation.yaml"
)

AUGMENTATION_REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/augmentation_policy_report.json"
)

COCO_VALIDATION_REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/coco_validation_report.json"
)

YOLO_VALIDATION_REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/yolo_validation_report.json"
)

REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/dataloader_validation_report.json"
)

ISSUES_FILE = (
    PROCESSED_ROOT
    / "reports/dataloader_validation_issues.csv"
)

MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/dataloader_smoke_test_manifest.csv"
)


EXPECTED = {
    "kitti_train": {
        "images": 5985,
        "negative_images": 0,
    },

    "kitti_val": {
        "images": 1496,
        "negative_images": 0,
    },

    "waymo_external": {
        "images": 996,
        "negative_images": 12,
    },
}


MANIFEST_COLUMNS = [
    "partition",
    "sample_index",
    "global_image_id",
    "file_name",
    "tensor_shape",
    "tensor_dtype",
    "tensor_minimum",
    "tensor_maximum",
    "target_box_count",
    "ignore_box_count",
    "excluded_box_count",
    "vehicle_count",
    "pedestrian_count",
    "cyclist_count",
    "augmentation_enabled",
    "operations_applied",
    "sample_passed",
]


# ============================================================
# HELPERS
# ============================================================

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


def tensors_equal(
    first: torch.Tensor,
    second: torch.Tensor,
) -> bool:
    return bool(
        torch.equal(
            first,
            second,
        )
    )


def targets_equal(
    first: dict,
    second: dict,
) -> bool:
    tensor_keys = [
        "boxes",
        "labels",
        "image_id",
        "area",
        "iscrowd",
        "ignore_boxes",
        "excluded_boxes",
        "size",
    ]

    return all(
        tensors_equal(
            first[key],
            second[key],
        )
        for key in tensor_keys
    )


def choose_sample_indices(
    dataset_length: int,
) -> list[int]:
    candidates = [
        0,
        dataset_length // 7,
        dataset_length // 5,
        dataset_length // 3,
        dataset_length // 2,
        (dataset_length * 2) // 3,
        (dataset_length * 4) // 5,
        dataset_length - 1,
    ]

    return sorted(
        set(
            max(
                0,
                min(
                    dataset_length - 1,
                    int(index),
                ),
            )
            for index in candidates
        )
    )


def validate_sample(
    partition_name: str,
    dataset: Milestone3DetectionDataset,
    sample_index: int,
    issues: list[dict],
) -> dict:
    try:
        image, target = dataset[
            sample_index
        ]

    except Exception as error:
        add_issue(
            issues,
            partition_name,
            "sample_loading_failed",
            str(sample_index),
            str(error),
        )

        return {
            "partition": partition_name,
            "sample_index": sample_index,
            "global_image_id": "",
            "file_name": "",
            "tensor_shape": "",
            "tensor_dtype": "",
            "tensor_minimum": "",
            "tensor_maximum": "",
            "target_box_count": "",
            "ignore_box_count": "",
            "excluded_box_count": "",
            "vehicle_count": "",
            "pedestrian_count": "",
            "cyclist_count": "",
            "augmentation_enabled": (
                dataset.enable_augmentation
            ),
            "operations_applied": "",
            "sample_passed": False,
        }

    boxes = target["boxes"]
    labels = target["labels"]

    tensor_shape_valid = (
        tuple(image.shape)
        == (3, 640, 640)
    )

    tensor_dtype_valid = (
        image.dtype
        == torch.float32
    )

    tensor_finite = bool(
        torch.isfinite(
            image
        ).all()
    )

    tensor_range_valid = bool(
        image.min().item() >= 0.0
        and image.max().item() <= 1.0
    )

    boxes_shape_valid = (
        boxes.ndim == 2
        and boxes.shape[1] == 4
    )

    labels_shape_valid = (
        labels.ndim == 1
        and len(labels) == len(boxes)
    )

    valid_classes = bool(
        len(labels) == 0
        or torch.isin(
            labels,
            torch.tensor(
                [1, 2, 3],
                dtype=torch.int64,
            ),
        ).all()
    )

    boxes_finite = bool(
        torch.isfinite(
            boxes
        ).all()
    )

    if len(boxes) > 0:
        boxes_valid = bool(
            boxes_finite
            and (
                boxes[:, 0] >= 0.0
            ).all()
            and (
                boxes[:, 1] >= 0.0
            ).all()
            and (
                boxes[:, 2] <= 640.0
            ).all()
            and (
                boxes[:, 3] <= 640.0
            ).all()
            and (
                boxes[:, 2]
                > boxes[:, 0]
            ).all()
            and (
                boxes[:, 3]
                > boxes[:, 1]
            ).all()
        )

    else:
        boxes_valid = boxes_finite

    area_valid = bool(
        len(target["area"])
        == len(boxes)
        and (
            len(target["area"]) == 0
            or (
                target["area"] > 0.0
            ).all()
        )
    )

    expected_count = (
        dataset.annotation_count(
            sample_index
        )
    )

    count_valid = (
        len(boxes)
        == expected_count
    )

    class_counts = Counter(
        int(value)
        for value in labels.tolist()
    )

    trace = target[
        "augmentation_trace"
    ]

    operations = (
        trace.get(
            "operations_applied",
            [],
        )
        if isinstance(trace, dict)
        else []
    )

    sample_checks = {
        "tensor_shape": (
            tensor_shape_valid
        ),
        "tensor_dtype": (
            tensor_dtype_valid
        ),
        "tensor_finite": (
            tensor_finite
        ),
        "tensor_range": (
            tensor_range_valid
        ),
        "boxes_shape": (
            boxes_shape_valid
        ),
        "labels_shape": (
            labels_shape_valid
        ),
        "valid_classes": (
            valid_classes
        ),
        "valid_boxes": (
            boxes_valid
        ),
        "valid_area": (
            area_valid
        ),
        "annotation_count": (
            count_valid
        ),
    }

    sample_passed = all(
        sample_checks.values()
    )

    if not sample_passed:
        add_issue(
            issues,
            partition_name,
            "sample_validation_failed",
            str(
                target[
                    "image_id"
                ].item()
            ),
            str(sample_checks),
        )

    return {
        "partition": (
            partition_name
        ),
        "sample_index": (
            sample_index
        ),
        "global_image_id": int(
            target[
                "image_id"
            ].item()
        ),
        "file_name": (
            target["file_name"]
        ),
        "tensor_shape": (
            "x".join(
                str(value)
                for value
                in image.shape
            )
        ),
        "tensor_dtype": str(
            image.dtype
        ),
        "tensor_minimum": float(
            image.min().item()
        ),
        "tensor_maximum": float(
            image.max().item()
        ),
        "target_box_count": int(
            len(boxes)
        ),
        "ignore_box_count": int(
            len(
                target[
                    "ignore_boxes"
                ]
            )
        ),
        "excluded_box_count": int(
            len(
                target[
                    "excluded_boxes"
                ]
            )
        ),
        "vehicle_count": int(
            class_counts[1]
        ),
        "pedestrian_count": int(
            class_counts[2]
        ),
        "cyclist_count": int(
            class_counts[3]
        ),
        "augmentation_enabled": bool(
            dataset.enable_augmentation
        ),
        "operations_applied": (
            "|".join(operations)
        ),
        "sample_passed": bool(
            sample_passed
        ),
    }


def validate_loader_batch(
    partition_name: str,
    dataset: Milestone3DetectionDataset,
    issues: list[dict],
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=(
            detection_collate_fn
        ),
    )

    try:
        images, targets = next(
            iter(loader)
        )

    except Exception as error:
        add_issue(
            issues,
            partition_name,
            "batch_loading_failed",
            partition_name,
            str(error),
        )

        return {
            "batch_size": 0,
            "image_tensor_shapes": [],
            "target_box_counts": [],
            "batch_passed": False,
        }

    batch_checks = {
        "batch_size": (
            len(images)
            == len(targets)
            == 4
        ),

        "image_shapes": all(
            tuple(image.shape)
            == (3, 640, 640)
            for image in images
        ),

        "image_dtypes": all(
            image.dtype
            == torch.float32
            for image in images
        ),

        "target_dictionaries": all(
            isinstance(
                target,
                dict,
            )
            for target in targets
        ),

        "variable_targets_supported": all(
            target["boxes"].ndim == 2
            for target in targets
        ),
    }

    batch_passed = all(
        batch_checks.values()
    )

    if not batch_passed:
        add_issue(
            issues,
            partition_name,
            "batch_validation_failed",
            partition_name,
            str(batch_checks),
        )

    return {
        "batch_size": int(
            len(images)
        ),

        "image_tensor_shapes": [
            list(image.shape)
            for image in images
        ],

        "target_box_counts": [
            int(
                len(
                    target["boxes"]
                )
            )
            for target in targets
        ],

        "checks": batch_checks,

        "batch_passed": bool(
            batch_passed
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

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    issues: list[dict] = []
    manifest_rows: list[dict] = []

    print("=" * 76)
    print("VALIDATING MODEL-READY DETECTION DATA LOADERS")
    print("=" * 76)

    augmentation_report = load_json(
        AUGMENTATION_REPORT_FILE
    )

    coco_report = load_json(
        COCO_VALIDATION_REPORT_FILE
    )

    yolo_report = load_json(
        YOLO_VALIDATION_REPORT_FILE
    )

    if not augmentation_report.get(
        "augmentation_policy_passed",
        False,
    ):
        raise RuntimeError(
            "Step 14 augmentation validation "
            "has not passed."
        )

    if not coco_report.get(
        "coco_validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 10 COCO validation "
            "has not passed."
        )

    if not yolo_report.get(
        "yolo_validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 11 YOLO validation "
            "has not passed."
        )

    augmentation_configuration = (
        load_yaml(
            AUGMENTATION_CONFIG_FILE
        )
    )

    datasets = {
        "kitti_train": (
            Milestone3DetectionDataset(
                partition_name=(
                    "kitti_train"
                ),
                enable_augmentation=False,
            )
        ),

        "kitti_val": (
            Milestone3DetectionDataset(
                partition_name=(
                    "kitti_val"
                ),
                enable_augmentation=False,
            )
        ),

        "waymo_external": (
            Milestone3DetectionDataset(
                partition_name=(
                    "waymo_external"
                ),
                enable_augmentation=False,
            )
        ),
    }

    augmented_train_dataset = (
        Milestone3DetectionDataset(
            partition_name=(
                "kitti_train"
            ),
            augmentation_configuration=(
                augmentation_configuration
            ),
            enable_augmentation=True,
        )
    )

    # --------------------------------------------------------
    # Dataset lengths
    # --------------------------------------------------------

    length_checks = {
        partition_name: (
            len(dataset)
            == EXPECTED[
                partition_name
            ]["images"]
        )
        for partition_name, dataset
        in datasets.items()
    }

    for partition_name, passed in (
        length_checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                partition_name,
                "dataset_length_mismatch",
                partition_name,
                (
                    f"Expected "
                    f"{EXPECTED[partition_name]['images']}, "
                    f"found "
                    f"{len(datasets[partition_name])}."
                ),
            )

    # --------------------------------------------------------
    # Individual sample smoke tests
    # --------------------------------------------------------

    for partition_name, dataset in (
        datasets.items()
    ):
        print(
            f"\nChecking {partition_name} "
            f"samples..."
        )

        sample_indices = (
            choose_sample_indices(
                len(dataset)
            )
        )

        for sample_index in (
            sample_indices
        ):
            manifest_rows.append(
                validate_sample(
                    partition_name=(
                        partition_name
                    ),
                    dataset=dataset,
                    sample_index=(
                        sample_index
                    ),
                    issues=issues,
                )
            )

    # --------------------------------------------------------
    # Detection DataLoader batch tests
    # --------------------------------------------------------

    batch_results = {}

    for partition_name, dataset in (
        datasets.items()
    ):
        print(
            f"Checking {partition_name} "
            f"batch loading..."
        )

        batch_results[
            partition_name
        ] = validate_loader_batch(
            partition_name=(
                partition_name
            ),
            dataset=dataset,
            issues=issues,
        )

    # --------------------------------------------------------
    # Deterministic validation and external evaluation
    # --------------------------------------------------------

    deterministic_evaluation = {}

    for partition_name in [
        "kitti_val",
        "waymo_external",
    ]:
        dataset = datasets[
            partition_name
        ]

        sample_index = (
            len(dataset) // 2
        )

        first_image, first_target = (
            dataset[sample_index]
        )

        second_image, second_target = (
            dataset[sample_index]
        )

        passed = (
            tensors_equal(
                first_image,
                second_image,
            )
            and targets_equal(
                first_target,
                second_target,
            )
            and first_target[
                "augmentation_trace"
            ]
            is None
            and second_target[
                "augmentation_trace"
            ]
            is None
        )

        deterministic_evaluation[
            partition_name
        ] = passed

        if not passed:
            add_issue(
                issues,
                partition_name,
                "evaluation_not_deterministic",
                str(sample_index),
                (
                    "Repeated evaluation loads "
                    "produced different results."
                ),
            )

    # --------------------------------------------------------
    # Augmentation guard
    # --------------------------------------------------------

    augmentation_guard_passed = False

    try:
        Milestone3DetectionDataset(
            partition_name="kitti_val",
            augmentation_configuration=(
                augmentation_configuration
            ),
            enable_augmentation=True,
        )

    except ValueError:
        augmentation_guard_passed = True

    if not augmentation_guard_passed:
        add_issue(
            issues,
            "combined",
            "augmentation_partition_guard_failed",
            "kitti_val",
            (
                "The loader allowed augmentation "
                "for a validation partition."
            ),
        )

    # --------------------------------------------------------
    # Training augmentation determinism
    # --------------------------------------------------------

    augmentation_test_index = 0
    augmentation_epoch = None
    augmentation_operations = []

    augmentation_deterministic = False
    augmentation_executed = False
    augmentation_changes_valid = False

    search_limit = min(
        64,
        len(augmented_train_dataset),
    )

    for candidate_index in range(
        search_limit
    ):
        for epoch in range(40):
            augmented_train_dataset.set_epoch(
                epoch
            )

            first_image, first_target = (
                augmented_train_dataset[
                    candidate_index
                ]
            )

            trace = first_target[
                "augmentation_trace"
            ]

            operations = (
                trace.get(
                    "operations_applied",
                    [],
                )
                if isinstance(trace, dict)
                else []
            )

            if not operations:
                continue

            second_image, second_target = (
                augmented_train_dataset[
                    candidate_index
                ]
            )

            augmentation_deterministic = (
                tensors_equal(
                    first_image,
                    second_image,
                )
                and targets_equal(
                    first_target,
                    second_target,
                )
                and (
                    first_target[
                        "augmentation_trace"
                    ][
                        "derived_seed"
                    ]
                    == second_target[
                        "augmentation_trace"
                    ][
                        "derived_seed"
                    ]
                )
            )

            augmentation_executed = True

            augmentation_changes_valid = (
                len(
                    first_target[
                        "boxes"
                    ]
                )
                == augmented_train_dataset
                .annotation_count(
                    candidate_index
                )
                and tuple(
                    first_image.shape
                )
                == (3, 640, 640)
            )

            augmentation_test_index = (
                candidate_index
            )

            augmentation_epoch = epoch

            augmentation_operations = list(
                operations
            )

            break

        if augmentation_executed:
            break

    if not augmentation_executed:
        add_issue(
            issues,
            "kitti_train",
            "augmentation_not_executed",
            "search",
            (
                "No enabled augmentation operation "
                "was observed during the smoke test."
            ),
        )

    if not augmentation_deterministic:
        add_issue(
            issues,
            "kitti_train",
            "augmentation_not_deterministic",
            str(
                augmentation_test_index
            ),
            (
                "The same image, epoch, and seed did "
                "not reproduce identical augmentation."
            ),
        )

    if not augmentation_changes_valid:
        add_issue(
            issues,
            "kitti_train",
            "augmented_sample_invalid",
            str(
                augmentation_test_index
            ),
            (
                "Augmentation changed the image size "
                "or target box count."
            ),
        )

    # --------------------------------------------------------
    # Negative external image
    # --------------------------------------------------------

    waymo_dataset = datasets[
        "waymo_external"
    ]

    negative_index = (
        waymo_dataset
        .find_first_negative_index()
    )

    negative_sample_passed = False
    negative_image_id = None

    if negative_index is None:
        add_issue(
            issues,
            "waymo_external",
            "negative_sample_not_found",
            "waymo_external",
            (
                "No target-negative external image "
                "was available."
            ),
        )

    else:
        negative_image, negative_target = (
            waymo_dataset[
                negative_index
            ]
        )

        negative_image_id = int(
            negative_target[
                "image_id"
            ].item()
        )

        negative_sample_passed = (
            tuple(
                negative_image.shape
            )
            == (3, 640, 640)
            and len(
                negative_target[
                    "boxes"
                ]
            )
            == 0
            and len(
                negative_target[
                    "labels"
                ]
            )
            == 0
        )

        if not negative_sample_passed:
            add_issue(
                issues,
                "waymo_external",
                "negative_sample_invalid",
                str(
                    negative_image_id
                ),
                (
                    "The selected negative image "
                    "contains target annotations."
                ),
            )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    sample_checks_passed = all(
        bool(
            row["sample_passed"]
        )
        for row in manifest_rows
    )

    batch_checks_passed = all(
        result["batch_passed"]
        for result
        in batch_results.values()
    )

    checks = {
        "dataset_lengths": all(
            length_checks.values()
        ),

        "sample_smoke_tests": (
            sample_checks_passed
        ),

        "batch_loading": (
            batch_checks_passed
        ),

        "validation_deterministic": (
            deterministic_evaluation[
                "kitti_val"
            ]
        ),

        "external_deterministic": (
            deterministic_evaluation[
                "waymo_external"
            ]
        ),

        "augmentation_partition_guard": (
            augmentation_guard_passed
        ),

        "training_augmentation_executed": (
            augmentation_executed
        ),

        "training_augmentation_deterministic": (
            augmentation_deterministic
        ),

        "augmented_sample_valid": (
            augmentation_changes_valid
        ),

        "external_negative_sample": (
            negative_sample_passed
        ),

        "manifest_row_count": (
            len(manifest_rows)
            >= 20
        ),
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "combined",
                "dataloader_check_failed",
                check_name,
                (
                    "The model-ready loader check "
                    "returned false."
                ),
            )

    overall_passed = (
        all(
            checks.values()
        )
        and len(issues) == 0
    )

    write_csv(
        MANIFEST_FILE,
        manifest_rows,
        MANIFEST_COLUMNS,
    )

    report = {
        "milestone": 3,
        "step": 15,

        "purpose": (
            "Build and validate model-ready "
            "PyTorch object-detection datasets "
            "and DataLoaders."
        ),

        "environment": {
            "python": (
                platform.python_version()
            ),

            "pytorch": (
                torch.__version__
            ),

            "opencv": (
                cv2.__version__
            ),

            "numpy": (
                np.__version__
            ),

            "cuda_available": (
                torch.cuda.is_available()
            ),
        },

        "dataset_lengths": {
            partition_name: int(
                len(dataset)
            )
            for partition_name, dataset
            in datasets.items()
        },

        "length_checks": (
            length_checks
        ),

        "batch_results": (
            batch_results
        ),

        "deterministic_evaluation": (
            deterministic_evaluation
        ),

        "augmentation_guard_passed": (
            augmentation_guard_passed
        ),

        "augmentation_test": {
            "sample_index": (
                augmentation_test_index
            ),

            "epoch": (
                augmentation_epoch
            ),

            "operations_applied": (
                augmentation_operations
            ),

            "executed": (
                augmentation_executed
            ),

            "deterministic": (
                augmentation_deterministic
            ),

            "sample_valid": (
                augmentation_changes_valid
            ),
        },

        "negative_external_test": {
            "sample_index": (
                negative_index
            ),

            "global_image_id": (
                negative_image_id
            ),

            "passed": (
                negative_sample_passed
            ),
        },

        "sample_manifest": {
            "path": (
                MANIFEST_FILE.as_posix()
            ),

            "rows": int(
                len(manifest_rows)
            ),
        },

        "checks": checks,

        "issue_count": len(
            issues
        ),

        "dataloader_validation_passed": (
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
    print("MODEL-READY DATA LOADER SUMMARY")
    print("=" * 76)

    for partition_name in [
        "kitti_train",
        "kitti_val",
        "waymo_external",
    ]:
        print(
            f"\n{partition_name}:"
        )

        print(
            f"  Dataset images: "
            f"{len(datasets[partition_name])}"
        )

        print(
            f"  Smoke-test samples: "
            f"{sum(row['partition'] == partition_name for row in manifest_rows)}"
        )

        print(
            f"  Batch size loaded: "
            f"{batch_results[partition_name]['batch_size']}"
        )

        print(
            f"  Batch status: "
            f"{'PASSED' if batch_results[partition_name]['batch_passed'] else 'FAILED'}"
        )

    print(
        "\nTraining augmentation:"
    )

    print(
        f"  Executed: "
        f"{augmentation_executed}"
    )

    print(
        f"  Operations: "
        f"{augmentation_operations}"
    )

    print(
        f"  Deterministic repeat: "
        f"{augmentation_deterministic}"
    )

    print(
        f"  Sample valid: "
        f"{augmentation_changes_valid}"
    )

    print(
        "\nEvaluation behavior:"
    )

    print(
        f"  KITTI validation deterministic: "
        f"{deterministic_evaluation['kitti_val']}"
    )

    print(
        f"  Waymo external deterministic: "
        f"{deterministic_evaluation['waymo_external']}"
    )

    print(
        f"  Augmentation partition guard: "
        f"{augmentation_guard_passed}"
    )

    print(
        f"  External negative sample valid: "
        f"{negative_sample_passed}"
    )

    print(
        f"\nManifest rows: "
        f"{len(manifest_rows)}"
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
            "model-ready data-loader issue "
            "is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 15 completed successfully."
    )


if __name__ == "__main__":
    main()