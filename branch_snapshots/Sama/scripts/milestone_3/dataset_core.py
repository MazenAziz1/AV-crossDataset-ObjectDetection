from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentation_core import (
    apply_training_augmentation,
)


# ============================================================
# PATHS AND PARTITIONS
# ============================================================

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)


PARTITIONS = {
    "kitti_train": {
        "dataset": "KITTI",
        "partition": "train",
        "role": "model_training",

        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/train"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_train.json"
        ),

        "ignore_file": (
            PROCESSED_ROOT
            / "annotations/ignore_regions/"
            "kitti_train_ignore.json"
        ),

        "excluded_file": (
            PROCESSED_ROOT
            / "annotations/excluded_objects/"
            "kitti_train_excluded.json"
        ),
    },

    "kitti_val": {
        "dataset": "KITTI",
        "partition": "val",
        "role": "in_domain_validation",

        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/val"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_val.json"
        ),

        "ignore_file": (
            PROCESSED_ROOT
            / "annotations/ignore_regions/"
            "kitti_val_ignore.json"
        ),

        "excluded_file": (
            PROCESSED_ROOT
            / "annotations/excluded_objects/"
            "kitti_val_excluded.json"
        ),
    },

    "waymo_external": {
        "dataset": "Waymo",
        "partition": "external",
        "role": "external_validation_only",

        "image_dir": (
            PROCESSED_ROOT
            / "images/waymo/external"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/waymo_external.json"
        ),

        "ignore_file": (
            PROCESSED_ROOT
            / "annotations/ignore_regions/"
            "waymo_external_ignore.json"
        ),

        "excluded_file": (
            PROCESSED_ROOT
            / "annotations/excluded_objects/"
            "waymo_external_excluded.json"
        ),
    },
}


EXPECTED_CATEGORY_MAPPING = {
    1: "Vehicle",
    2: "Pedestrian",
    3: "Cyclist",
}


# Sentinels used only while transforming sidecar regions.
IGNORE_REGION_SENTINEL = -1
EXCLUDED_REGION_SENTINEL = -2


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(
    path: Path,
) -> dict:
    import json

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
            f"JSON root must be an object:\n"
            f"{path.resolve()}"
        )

    return data


def xywh_to_xyxy(
    bbox: list | tuple,
) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(
            "A bounding box must contain four values."
        )

    x, y, width, height = [
        float(value)
        for value in bbox
    ]

    if not np.isfinite(
        [x, y, width, height]
    ).all():
        raise ValueError(
            "Bounding box contains non-finite values."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Bounding box has non-positive dimensions."
        )

    return [
        x,
        y,
        x + width,
        y + height,
    ]


def records_to_boxes(
    records: list[dict],
) -> np.ndarray:
    boxes = [
        xywh_to_xyxy(
            record["bbox"]
        )
        for record in records
    ]

    if not boxes:
        return np.empty(
            (0, 4),
            dtype=np.float64,
        )

    return np.asarray(
        boxes,
        dtype=np.float64,
    )


def concatenate_boxes(
    arrays: list[np.ndarray],
) -> np.ndarray:
    nonempty = [
        array
        for array in arrays
        if len(array) > 0
    ]

    if not nonempty:
        return np.empty(
            (0, 4),
            dtype=np.float64,
        )

    return np.concatenate(
        nonempty,
        axis=0,
    ).astype(
        np.float64,
        copy=False,
    )


def detection_collate_fn(
    batch: list[
        tuple[
            torch.Tensor,
            dict[str, Any],
        ]
    ],
) -> tuple[
    list[torch.Tensor],
    list[dict[str, Any]],
]:
    """
    Object-detection images contain different numbers of boxes,
    so they cannot use PyTorch's default tensor stacking for targets.
    """
    images, targets = zip(*batch)

    return list(images), list(targets)


# ============================================================
# DATASET
# ============================================================

class Milestone3DetectionDataset(
    Dataset,
):
    """
    Framework-neutral PyTorch dataset based on the canonical COCO files.

    Returned target class IDs preserve the canonical COCO IDs:

        1 = Vehicle
        2 = Pedestrian
        3 = Cyclist

    This is compatible with Torchvision detectors, where class ID zero
    is reserved for background.
    """

    def __init__(
        self,
        partition_name: str,
        augmentation_configuration: (
            dict[str, Any] | None
        ) = None,
        enable_augmentation: bool = False,
        include_region_sidecars: bool = True,
    ) -> None:
        if partition_name not in PARTITIONS:
            raise KeyError(
                f"Unknown partition: {partition_name}"
            )

        if (
            enable_augmentation
            and partition_name
            != "kitti_train"
        ):
            raise ValueError(
                "Online augmentation is permitted only "
                "for the kitti_train partition."
            )

        if (
            enable_augmentation
            and augmentation_configuration
            is None
        ):
            raise ValueError(
                "Augmentation was enabled, but no "
                "augmentation configuration was supplied."
            )

        self.partition_name = (
            partition_name
        )

        self.specification = (
            PARTITIONS[
                partition_name
            ]
        )

        self.enable_augmentation = bool(
            enable_augmentation
        )

        self.augmentation_configuration = (
            augmentation_configuration
        )

        self.include_region_sidecars = bool(
            include_region_sidecars
        )

        self.epoch = 0

        self.image_dir = Path(
            self.specification[
                "image_dir"
            ]
        )

        self.coco_file = Path(
            self.specification[
                "coco_file"
            ]
        )

        self.ignore_file = Path(
            self.specification[
                "ignore_file"
            ]
        )

        self.excluded_file = Path(
            self.specification[
                "excluded_file"
            ]
        )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found:\n"
                f"{self.image_dir.resolve()}"
            )

        coco_data = load_json(
            self.coco_file
        )

        categories = {
            int(category["id"]): str(
                category["name"]
            )
            for category
            in coco_data.get(
                "categories",
                [],
            )
        }

        if (
            categories
            != EXPECTED_CATEGORY_MAPPING
        ):
            raise ValueError(
                "Canonical COCO class mapping differs "
                "from 1=Vehicle, 2=Pedestrian, "
                "3=Cyclist."
            )

        self.image_records = sorted(
            coco_data.get(
                "images",
                [],
            ),
            key=lambda record: int(
                record["id"]
            ),
        )

        self.annotations_by_image: dict[
            int,
            list[dict],
        ] = defaultdict(list)

        for annotation in coco_data.get(
            "annotations",
            [],
        ):
            image_id = int(
                annotation["image_id"]
            )

            self.annotations_by_image[
                image_id
            ].append(annotation)

        for image_id in (
            self.annotations_by_image
        ):
            self.annotations_by_image[
                image_id
            ].sort(
                key=lambda record: int(
                    record["id"]
                )
            )

        self.ignore_regions_by_image: dict[
            int,
            list[dict],
        ] = defaultdict(list)

        self.excluded_regions_by_image: dict[
            int,
            list[dict],
        ] = defaultdict(list)

        if self.include_region_sidecars:
            self.ignore_regions_by_image = (
                self._load_sidecar_regions(
                    self.ignore_file
                )
            )

            self.excluded_regions_by_image = (
                self._load_sidecar_regions(
                    self.excluded_file
                )
            )

        self.image_id_to_index = {
            int(record["id"]): index
            for index, record
            in enumerate(
                self.image_records
            )
        }

    def _load_sidecar_regions(
        self,
        path: Path,
    ) -> dict[
        int,
        list[dict],
    ]:
        data = load_json(path)

        grouped: dict[
            int,
            list[dict],
        ] = defaultdict(list)

        for region in data.get(
            "regions",
            [],
        ):
            grouped[
                int(region["image_id"])
            ].append(region)

        return grouped

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """
        Update the epoch used to derive deterministic augmentation seeds.
        """
        self.epoch = int(epoch)

    def get_index_by_image_id(
        self,
        image_id: int,
    ) -> int:
        return self.image_id_to_index[
            int(image_id)
        ]

    def find_first_negative_index(
        self,
    ) -> int | None:
        for index, record in enumerate(
            self.image_records
        ):
            image_id = int(
                record["id"]
            )

            if (
                len(
                    self.annotations_by_image.get(
                        image_id,
                        [],
                    )
                )
                == 0
            ):
                return index

        return None

    def annotation_count(
        self,
        index: int,
    ) -> int:
        record = self.image_records[
            index
        ]

        return len(
            self.annotations_by_image.get(
                int(record["id"]),
                [],
            )
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.image_records
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        dict[str, Any],
    ]:
        image_record = (
            self.image_records[index]
        )

        image_id = int(
            image_record["id"]
        )

        image_filename = str(
            image_record["file_name"]
        )

        image_path = (
            self.image_dir
            / image_filename
        )

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise ValueError(
                f"Could not decode image:\n"
                f"{image_path.resolve()}"
            )

        image_height, image_width = (
            image_bgr.shape[:2]
        )

        if (
            image_width != 640
            or image_height != 640
        ):
            raise ValueError(
                f"Expected 640x640 image, found "
                f"{image_width}x{image_height}:\n"
                f"{image_path.resolve()}"
            )

        target_annotations = (
            self.annotations_by_image.get(
                image_id,
                [],
            )
        )

        target_boxes = records_to_boxes(
            target_annotations
        )

        target_labels = np.asarray(
            [
                int(
                    annotation[
                        "category_id"
                    ]
                )
                for annotation
                in target_annotations
            ],
            dtype=np.int64,
        )

        ignore_regions = (
            self.ignore_regions_by_image.get(
                image_id,
                [],
            )
        )

        excluded_regions = (
            self.excluded_regions_by_image.get(
                image_id,
                [],
            )
        )

        ignore_boxes = records_to_boxes(
            ignore_regions
        )

        excluded_boxes = records_to_boxes(
            excluded_regions
        )

        augmentation_trace = None

        if self.enable_augmentation:
            combined_boxes = (
                concatenate_boxes(
                    [
                        target_boxes,
                        ignore_boxes,
                        excluded_boxes,
                    ]
                )
            )

            combined_identifiers = (
                np.concatenate(
                    [
                        target_labels,

                        np.full(
                            len(ignore_boxes),
                            IGNORE_REGION_SENTINEL,
                            dtype=np.int64,
                        ),

                        np.full(
                            len(excluded_boxes),
                            EXCLUDED_REGION_SENTINEL,
                            dtype=np.int64,
                        ),
                    ]
                )
                if len(combined_boxes) > 0
                else np.empty(
                    (0,),
                    dtype=np.int64,
                )
            )

            (
                image_bgr,
                transformed_boxes,
                transformed_identifiers,
                augmentation_trace,
            ) = apply_training_augmentation(
                image=image_bgr,
                boxes_xyxy=combined_boxes,
                class_ids=combined_identifiers,
                configuration=(
                    self.augmentation_configuration
                ),
                global_image_id=image_id,
                epoch=self.epoch,
            )

            target_mask = (
                transformed_identifiers
                > 0
            )

            ignore_mask = (
                transformed_identifiers
                == IGNORE_REGION_SENTINEL
            )

            excluded_mask = (
                transformed_identifiers
                == EXCLUDED_REGION_SENTINEL
            )

            target_boxes = (
                transformed_boxes[
                    target_mask
                ]
            )

            target_labels = (
                transformed_identifiers[
                    target_mask
                ]
            )

            ignore_boxes = (
                transformed_boxes[
                    ignore_mask
                ]
            )

            excluded_boxes = (
                transformed_boxes[
                    excluded_mask
                ]
            )

        target_boxes = np.asarray(
            target_boxes,
            dtype=np.float32,
        ).reshape(-1, 4)

        target_labels = np.asarray(
            target_labels,
            dtype=np.int64,
        ).reshape(-1)

        ignore_boxes = np.asarray(
            ignore_boxes,
            dtype=np.float32,
        ).reshape(-1, 4)

        excluded_boxes = np.asarray(
            excluded_boxes,
            dtype=np.float32,
        ).reshape(-1, 4)

        if len(target_boxes) > 0:
            target_areas = (
                (
                    target_boxes[:, 2]
                    - target_boxes[:, 0]
                )
                * (
                    target_boxes[:, 3]
                    - target_boxes[:, 1]
                )
            ).astype(
                np.float32
            )

        else:
            target_areas = np.empty(
                (0,),
                dtype=np.float32,
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    image_rgb.transpose(
                        2,
                        0,
                        1,
                    )
                )
            )
            .to(
                dtype=torch.float32
            )
            .div(255.0)
        )

        target: dict[str, Any] = {
            "boxes": torch.from_numpy(
                target_boxes
            ),

            "labels": torch.from_numpy(
                target_labels
            ),

            "image_id": torch.tensor(
                image_id,
                dtype=torch.int64,
            ),

            "area": torch.from_numpy(
                target_areas
            ),

            "iscrowd": torch.zeros(
                len(target_boxes),
                dtype=torch.int64,
            ),

            "ignore_boxes": torch.from_numpy(
                ignore_boxes
            ),

            "excluded_boxes": torch.from_numpy(
                excluded_boxes
            ),

            "size": torch.tensor(
                [
                    image_height,
                    image_width,
                ],
                dtype=torch.int64,
            ),

            "image_path": str(
                image_path
            ),

            "file_name": image_filename,

            "source_dataset": str(
                image_record.get(
                    "source_dataset",
                    self.specification[
                        "dataset"
                    ],
                )
            ),

            "source_image_id": str(
                image_record.get(
                    "source_image_id",
                    "",
                )
            ),

            "partition": str(
                image_record.get(
                    "partition",
                    self.specification[
                        "partition"
                    ],
                )
            ),

            "role": str(
                self.specification[
                    "role"
                ]
            ),

            "augmentation_trace": (
                augmentation_trace
            ),
        }

        return image_tensor, target