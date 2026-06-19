from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import re
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROCESSED_ROOT = Path(
    "data/processed/milestone_3"
)

SOURCE_MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/source_manifest.csv"
)

EQUIVALENCE_MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/coco_yolo_equivalence_manifest.csv"
)

EQUIVALENCE_REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/coco_yolo_equivalence_report.json"
)

VISUAL_ROOT = (
    PROCESSED_ROOT
    / "visual_checks"
)

COMPARISON_DIR = (
    VISUAL_ROOT
    / "coco_yolo_comparison"
)

REPORT_FILE = (
    PROCESSED_ROOT
    / "reports/visual_annotation_checks_report.json"
)

ISSUES_FILE = (
    PROCESSED_ROOT
    / "reports/visual_annotation_checks_issues.csv"
)

VISUAL_MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifests/visual_annotation_checks_manifest.csv"
)


# ============================================================
# PARTITIONS
# ============================================================

PARTITIONS = {
    "kitti_train": {
        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/train"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_train.json"
        ),

        "yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/kitti/train"
        ),

        "coco_visual_dir": (
            VISUAL_ROOT
            / "coco/kitti/train"
        ),

        "yolo_visual_dir": (
            VISUAL_ROOT
            / "yolo/kitti/train"
        ),
    },

    "kitti_val": {
        "image_dir": (
            PROCESSED_ROOT
            / "images/kitti/val"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/kitti_val.json"
        ),

        "yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/kitti/val"
        ),

        "coco_visual_dir": (
            VISUAL_ROOT
            / "coco/kitti/val"
        ),

        "yolo_visual_dir": (
            VISUAL_ROOT
            / "yolo/kitti/val"
        ),
    },

    "waymo_external": {
        "image_dir": (
            PROCESSED_ROOT
            / "images/waymo/external"
        ),

        "coco_file": (
            PROCESSED_ROOT
            / "annotations/coco/waymo_external.json"
        ),

        "yolo_dir": (
            PROCESSED_ROOT
            / "annotations/yolo/waymo/external"
        ),

        "coco_visual_dir": (
            VISUAL_ROOT
            / "coco/waymo/external"
        ),

        "yolo_visual_dir": (
            VISUAL_ROOT
            / "yolo/waymo/external"
        ),
    },
}


EXPECTED_SAMPLE_COUNT = 8

CLASS_NAMES = {
    0: "Vehicle",
    1: "Pedestrian",
    2: "Cyclist",
}

COCO_TO_YOLO = {
    1: 0,
    2: 1,
    3: 2,
}


# OpenCV uses BGR.
CLASS_COLORS = {
    "Vehicle": (
        0,
        210,
        0,
    ),

    "Pedestrian": (
        0,
        165,
        255,
    ),

    "Cyclist": (
        255,
        120,
        0,
    ),
}


VISUAL_MANIFEST_COLUMNS = [
    "sample_number",
    "selection_reason",
    "partition",
    "global_image_id",
    "source_image_id",
    "image_filename",
    "target_box_count",
    "vehicle_count",
    "pedestrian_count",
    "cyclist_count",
    "is_negative",
    "coco_box_count",
    "yolo_box_count",
    "class_counts_match",
    "equivalence_image_passed",
    "maximum_normalized_error",
    "maximum_pixel_error",
    "minimum_iou",
    "coco_visual_path",
    "yolo_visual_path",
    "comparison_visual_path",
    "sample_passed",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found:\n"
            f"{path.resolve()}"
        )

    content = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        content,
        dict,
    ):
        raise ValueError(
            f"JSON root must be an object:\n"
            f"{path.resolve()}"
        )

    return content


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


def parse_bool(
    value,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def safe_filename_component(
    value: str,
) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(value),
    )

    return cleaned.strip("_")


def project_relative_path(
    path: Path,
) -> str:
    try:
        return (
            path.resolve()
            .relative_to(
                Path.cwd().resolve()
            )
            .as_posix()
        )

    except ValueError:
        return (
            path.resolve()
            .as_posix()
        )


def partition_key(
    dataset: str,
    partition: str,
) -> str:
    if (
        dataset == "KITTI"
        and partition == "train"
    ):
        return "kitti_train"

    if (
        dataset == "KITTI"
        and partition == "val"
    ):
        return "kitti_val"

    if (
        dataset == "Waymo"
        and partition == "external"
    ):
        return "waymo_external"

    raise ValueError(
        f"Unsupported dataset partition: "
        f"{dataset}/{partition}"
    )


def round_half_up(
    value: float,
) -> int:
    return int(
        math.floor(
            float(value) + 0.5
        )
    )


def save_image(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    succeeded = cv2.imwrite(
        str(path),
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not succeeded:
        raise RuntimeError(
            f"Could not write image:\n"
            f"{path.resolve()}"
        )


# ============================================================
# SAMPLE SELECTION
# ============================================================

def choose_unique_row(
    dataframe: pd.DataFrame,
    mask: pd.Series,
    sort_column: str,
    ascending: bool,
    selected_ids: set[int],
) -> pd.Series | None:
    candidates = dataframe[
        mask
    ].copy()

    candidates = candidates[
        ~candidates[
            "global_image_id"
        ].isin(
            selected_ids
        )
    ]

    if candidates.empty:
        return None

    candidates = (
        candidates.sort_values(
            [
                sort_column,
                "global_image_id",
            ],
            ascending=[
                ascending,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return candidates.iloc[0]


def select_samples(
    dataframe: pd.DataFrame,
    issues: list[dict],
) -> list[
    tuple[
        str,
        pd.Series,
    ]
]:
    selected: list[
        tuple[
            str,
            pd.Series,
        ]
    ] = []

    selected_ids: set[int] = set()

    criteria = [
        {
            "reason": (
                "kitti_train_crowded"
            ),
            "mask": (
                (
                    dataframe["dataset"]
                    == "KITTI"
                )
                & (
                    dataframe["partition"]
                    == "train"
                )
            ),
            "sort_column": (
                "target_box_count"
            ),
            "ascending": False,
        },

        {
            "reason": (
                "kitti_train_cyclist_rich"
            ),
            "mask": (
                (
                    dataframe["dataset"]
                    == "KITTI"
                )
                & (
                    dataframe["partition"]
                    == "train"
                )
                & (
                    dataframe["cyclist_count"]
                    > 0
                )
            ),
            "sort_column": (
                "cyclist_count"
            ),
            "ascending": False,
        },

        {
            "reason": (
                "kitti_val_crowded"
            ),
            "mask": (
                (
                    dataframe["dataset"]
                    == "KITTI"
                )
                & (
                    dataframe["partition"]
                    == "val"
                )
            ),
            "sort_column": (
                "target_box_count"
            ),
            "ascending": False,
        },

        {
            "reason": (
                "kitti_val_pedestrian_rich"
            ),
            "mask": (
                (
                    dataframe["dataset"]
                    == "KITTI"
                )
                & (
                    dataframe["partition"]
                    == "val"
                )
                & (
                    dataframe[
                        "pedestrian_count"
                    ]
                    > 0
                )
            ),
            "sort_column": (
                "pedestrian_count"
            ),
            "ascending": False,
        },

        {
            "reason": (
                "waymo_crowded"
            ),
            "mask": (
                (
                    dataframe["dataset"]
                    == "Waymo"
                )
                & (
                    dataframe["partition"]
                    == "external"
                )
                & (
                    ~dataframe[
                        "is_negative_bool"
                    ]
                )
            ),
            "sort_column": (
                "target_box_count"
            ),
            "ascending": False,
        },

        {
            "reason": (
                "waymo_cyclist_rich"
            ),
            "mask": (
                (
                    dataframe["dataset"]
                    == "Waymo"
                )
                & (
                    dataframe["partition"]
                    == "external"
                )
                & (
                    dataframe["cyclist_count"]
                    > 0
                )
            ),
            "sort_column": (
                "cyclist_count"
            ),
            "ascending": False,
        },

        {
            "reason": (
                "waymo_pedestrian_rich"
            ),
            "mask": (
                (
                    dataframe["dataset"]
                    == "Waymo"
                )
                & (
                    dataframe["partition"]
                    == "external"
                )
                & (
                    dataframe[
                        "pedestrian_count"
                    ]
                    > 0
                )
            ),
            "sort_column": (
                "pedestrian_count"
            ),
            "ascending": False,
        },

        {
            "reason": (
                "waymo_negative"
            ),
            "mask": (
                (
                    dataframe["dataset"]
                    == "Waymo"
                )
                & (
                    dataframe["partition"]
                    == "external"
                )
                & (
                    dataframe[
                        "is_negative_bool"
                    ]
                )
            ),
            "sort_column": (
                "global_image_id"
            ),
            "ascending": True,
        },
    ]

    for criterion in criteria:
        row = choose_unique_row(
            dataframe=dataframe,
            mask=criterion["mask"],
            sort_column=(
                criterion[
                    "sort_column"
                ]
            ),
            ascending=(
                criterion["ascending"]
            ),
            selected_ids=selected_ids,
        )

        if row is None:
            add_issue(
                issues,
                "selection",
                "sample_selection_failed",
                criterion["reason"],
                (
                    "No suitable unique image "
                    "was available."
                ),
            )

            continue

        global_image_id = int(
            row["global_image_id"]
        )

        selected_ids.add(
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
# ANNOTATION LOADING
# ============================================================

def load_coco_partition(
    path: Path,
) -> tuple[
    dict[int, dict],
    dict[int, list[dict]],
]:
    data = load_json(path)

    category_lookup = {
        int(category["id"]): str(
            category["name"]
        )
        for category in data[
            "categories"
        ]
    }

    expected_categories = {
        1: "Vehicle",
        2: "Pedestrian",
        3: "Cyclist",
    }

    if (
        category_lookup
        != expected_categories
    ):
        raise ValueError(
            f"Unexpected categories in:\n"
            f"{path.resolve()}"
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

    for image_id in (
        annotations_by_image
    ):
        annotations_by_image[
            image_id
        ].sort(
            key=lambda item: int(
                item["id"]
            )
        )

    return (
        image_lookup,
        dict(
            annotations_by_image
        ),
    )


def coco_records(
    annotations: list[dict],
) -> list[dict]:
    records: list[dict] = []

    for annotation in annotations:
        category_id = int(
            annotation["category_id"]
        )

        yolo_id = COCO_TO_YOLO[
            category_id
        ]

        x, y, width, height = [
            float(value)
            for value in annotation[
                "bbox"
            ]
        ]

        records.append(
            {
                "class_id": yolo_id,
                "class_name": (
                    CLASS_NAMES[yolo_id]
                ),
                "xmin": x,
                "ymin": y,
                "xmax": x + width,
                "ymax": y + height,
            }
        )

    return records


def read_yolo_records(
    path: Path,
    image_width: int,
    image_height: int,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"YOLO label not found:\n"
            f"{path.resolve()}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    records: list[dict] = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        tokens = line.split()

        if len(tokens) != 5:
            raise ValueError(
                f"{path.name}, line "
                f"{line_number}: expected "
                f"five values."
            )

        class_id = int(tokens[0])

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"{path.name}, line "
                f"{line_number}: unknown "
                f"class ID {class_id}."
            )

        (
            center_x,
            center_y,
            normalized_width,
            normalized_height,
        ) = [
            float(value)
            for value in tokens[1:]
        ]

        pixel_center_x = (
            center_x * image_width
        )

        pixel_center_y = (
            center_y * image_height
        )

        pixel_width = (
            normalized_width
            * image_width
        )

        pixel_height = (
            normalized_height
            * image_height
        )

        xmin = (
            pixel_center_x
            - pixel_width / 2.0
        )

        ymin = (
            pixel_center_y
            - pixel_height / 2.0
        )

        xmax = (
            pixel_center_x
            + pixel_width / 2.0
        )

        ymax = (
            pixel_center_y
            + pixel_height / 2.0
        )

        records.append(
            {
                "class_id": class_id,
                "class_name": (
                    CLASS_NAMES[class_id]
                ),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return records


# ============================================================
# DRAWING
# ============================================================

def draw_text_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    x, y = origin

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    x1 = max(
        0,
        x,
    )

    y1 = max(
        0,
        y - text_height - baseline - 4,
    )

    x2 = min(
        image.shape[1] - 1,
        x + text_width + 6,
    )

    y2 = min(
        image.shape[0] - 1,
        y + 2,
    )

    cv2.rectangle(
        image,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        background,
        thickness=-1,
    )

    cv2.putText(
        image,
        text,
        (
            x1 + 3,
            y2 - baseline - 2,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        foreground,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_annotations(
    source_image: np.ndarray,
    records: list[dict],
    annotation_source: str,
) -> np.ndarray:
    output = source_image.copy()

    image_height, image_width = (
        output.shape[:2]
    )

    class_counts: Counter = Counter(
        record["class_name"]
        for record in records
    )

    for record in records:
        class_name = str(
            record["class_name"]
        )

        color = CLASS_COLORS[
            class_name
        ]

        xmin = max(
            0,
            min(
                image_width - 1,
                round_half_up(
                    record["xmin"]
                ),
            ),
        )

        ymin = max(
            0,
            min(
                image_height - 1,
                round_half_up(
                    record["ymin"]
                ),
            ),
        )

        xmax = max(
            0,
            min(
                image_width - 1,
                round_half_up(
                    record["xmax"]
                ),
            ),
        )

        ymax = max(
            0,
            min(
                image_height - 1,
                round_half_up(
                    record["ymax"]
                ),
            ),
        )

        if (
            xmax <= xmin
            or ymax <= ymin
        ):
            continue

        cv2.rectangle(
            output,
            (
                xmin,
                ymin,
            ),
            (
                xmax,
                ymax,
            ),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        label_y = (
            ymin
            if ymin >= 22
            else ymin + 22
        )

        draw_text_box(
            image=output,
            text=class_name,
            origin=(
                xmin,
                label_y,
            ),
            foreground=(
                255,
                255,
                255,
            ),
            background=color,
        )

    summary_text = (
        f"{annotation_source} | "
        f"Boxes: {len(records)} | "
        f"V: {class_counts['Vehicle']}  "
        f"P: {class_counts['Pedestrian']}  "
        f"C: {class_counts['Cyclist']}"
    )

    cv2.rectangle(
        output,
        (
            0,
            0,
        ),
        (
            image_width,
            30,
        ),
        (
            25,
            25,
            25,
        ),
        thickness=-1,
    )

    cv2.putText(
        output,
        summary_text,
        (
            8,
            21,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            255,
            255,
            255,
        ),
        1,
        lineType=cv2.LINE_AA,
    )

    if not records:
        cv2.putText(
            output,
            "NO TARGET BOXES",
            (
                190,
                330,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (
                255,
                255,
                255,
            ),
            2,
            lineType=cv2.LINE_AA,
        )

    return output


def make_panel(
    image: np.ndarray,
    title: str,
    subtitle: str,
) -> np.ndarray:
    header_height = 72

    panel = np.full(
        (
            image.shape[0]
            + header_height,
            image.shape[1],
            3,
        ),
        fill_value=30,
        dtype=np.uint8,
    )

    panel[
        header_height:,
        :,
    ] = image

    cv2.putText(
        panel,
        title,
        (
            12,
            28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (
            255,
            255,
            255,
        ),
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        subtitle[:90],
        (
            12,
            55,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (
            210,
            210,
            210,
        ),
        1,
        lineType=cv2.LINE_AA,
    )

    return panel


def make_comparison(
    coco_visual: np.ndarray,
    yolo_visual: np.ndarray,
    image_identifier: str,
    maximum_pixel_error: float,
    minimum_iou: float,
) -> np.ndarray:
    subtitle = (
        f"{image_identifier} | "
        f"max pixel error="
        f"{maximum_pixel_error:.3e} | "
        f"min IoU={minimum_iou:.12f}"
    )

    left_panel = make_panel(
        image=coco_visual,
        title=(
            "Canonical COCO annotations"
        ),
        subtitle=subtitle,
    )

    right_panel = make_panel(
        image=yolo_visual,
        title=(
            "Derived YOLO annotations"
        ),
        subtitle=subtitle,
    )

    return np.hstack(
        [
            left_panel,
            right_panel,
        ]
    )


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    comparison_paths: list[Path],
    output_path: Path,
) -> None:
    thumbnails: list[np.ndarray] = []

    for path in comparison_paths:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"Could not decode comparison image:\n"
                f"{path.resolve()}"
            )

        thumbnail = cv2.resize(
            image,
            (
                640,
                356,
            ),
            interpolation=cv2.INTER_AREA,
        )

        thumbnails.append(
            thumbnail
        )

    if len(thumbnails) != (
        EXPECTED_SAMPLE_COUNT
    ):
        raise ValueError(
            "The contact sheet requires "
            "exactly eight comparison images."
        )

    rows: list[np.ndarray] = []

    for index in range(
        0,
        len(thumbnails),
        2,
    ):
        rows.append(
            np.hstack(
                [
                    thumbnails[index],
                    thumbnails[index + 1],
                ]
            )
        )

    sheet = np.vstack(rows)

    save_image(
        output_path,
        sheet,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISUAL_MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for specification in (
        PARTITIONS.values()
    ):
        specification[
            "coco_visual_dir"
        ].mkdir(
            parents=True,
            exist_ok=True,
        )

        specification[
            "yolo_visual_dir"
        ].mkdir(
            parents=True,
            exist_ok=True,
        )

    issues: list[dict] = []
    manifest_rows: list[dict] = []
    comparison_paths: list[Path] = []

    print("=" * 76)
    print("CREATING VISUAL COCO–YOLO ANNOTATION CHECKS")
    print("=" * 76)

    equivalence_report = load_json(
        EQUIVALENCE_REPORT_FILE
    )

    if not equivalence_report.get(
        "coco_yolo_equivalence_passed",
        False,
    ):
        raise RuntimeError(
            "Step 12 COCO–YOLO equivalence "
            "validation has not passed."
        )

    if not SOURCE_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Source manifest not found:\n"
            f"{SOURCE_MANIFEST_FILE.resolve()}"
        )

    if not (
        EQUIVALENCE_MANIFEST_FILE.exists()
    ):
        raise FileNotFoundError(
            f"Equivalence manifest not found:\n"
            f"{EQUIVALENCE_MANIFEST_FILE.resolve()}"
        )

    source_manifest = pd.read_csv(
        SOURCE_MANIFEST_FILE,
        dtype={
            "source_image_id": str,
            "output_filename": str,
            "output_relative_path": str,
        },
    )

    equivalence_manifest = pd.read_csv(
        EQUIVALENCE_MANIFEST_FILE,
        dtype={
            "image_filename": str,
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

    equivalence_numeric_columns = [
        "global_image_id",
        "coco_annotation_count",
        "yolo_row_count",
        "matched_box_count",
        "maximum_normalized_error",
        "maximum_pixel_error",
        "minimum_iou",
    ]

    for column in (
        equivalence_numeric_columns
    ):
        equivalence_manifest[column] = (
            pd.to_numeric(
                equivalence_manifest[
                    column
                ],
                errors="raise",
            )
        )

    source_manifest[
        "is_negative_bool"
    ] = source_manifest[
        "is_negative"
    ].apply(parse_bool)

    equivalence_lookup = {
        (
            str(row["partition"]),
            int(row["global_image_id"]),
        ): row
        for _, row
        in equivalence_manifest.iterrows()
    }

    coco_lookups = {}

    for partition_name, specification in (
        PARTITIONS.items()
    ):
        coco_lookups[
            partition_name
        ] = load_coco_partition(
            specification[
                "coco_file"
            ]
        )

    selected_samples = select_samples(
        dataframe=source_manifest,
        issues=issues,
    )

    print(
        f"\nSelected samples: "
        f"{len(selected_samples)}"
    )

    for sample_number, (
        selection_reason,
        source_row,
    ) in enumerate(
        tqdm(
            selected_samples,
            unit="image",
        ),
        start=1,
    ):
        dataset = str(
            source_row["dataset"]
        )

        partition = str(
            source_row["partition"]
        )

        partition_name = partition_key(
            dataset,
            partition,
        )

        specification = PARTITIONS[
            partition_name
        ]

        global_image_id = int(
            source_row[
                "global_image_id"
            ]
        )

        source_image_id = str(
            source_row[
                "source_image_id"
            ]
        )

        image_filename = str(
            source_row[
                "output_filename"
            ]
        )

        image_path = (
            specification["image_dir"]
            / image_filename
        )

        identifier = (
            f"{partition_name}:"
            f"{global_image_id}"
        )

        source_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if source_image is None:
            add_issue(
                issues,
                partition_name,
                "image_decode_failed",
                identifier,
                str(image_path),
            )

            continue

        image_height, image_width = (
            source_image.shape[:2]
        )

        if (
            image_width != 640
            or image_height != 640
        ):
            add_issue(
                issues,
                partition_name,
                "invalid_processed_dimensions",
                identifier,
                (
                    f"Found "
                    f"{image_width}x"
                    f"{image_height}."
                ),
            )

            continue

        (
            image_lookup,
            annotations_by_image,
        ) = coco_lookups[
            partition_name
        ]

        if global_image_id not in (
            image_lookup
        ):
            add_issue(
                issues,
                partition_name,
                "missing_coco_image_record",
                identifier,
                image_filename,
            )

            continue

        coco_annotations = (
            annotations_by_image.get(
                global_image_id,
                [],
            )
        )

        coco_box_records = coco_records(
            coco_annotations
        )

        yolo_path = (
            specification["yolo_dir"]
            / (
                f"{Path(image_filename).stem}"
                ".txt"
            )
        )

        try:
            yolo_box_records = (
                read_yolo_records(
                    path=yolo_path,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        except Exception as error:
            add_issue(
                issues,
                partition_name,
                "yolo_read_failed",
                identifier,
                str(error),
            )

            continue

        equivalence_key = (
            partition_name,
            global_image_id,
        )

        equivalence_row = (
            equivalence_lookup.get(
                equivalence_key
            )
        )

        if equivalence_row is None:
            add_issue(
                issues,
                partition_name,
                "missing_equivalence_record",
                identifier,
                str(equivalence_key),
            )

            continue

        coco_counts = Counter(
            record["class_name"]
            for record
            in coco_box_records
        )

        yolo_counts = Counter(
            record["class_name"]
            for record
            in yolo_box_records
        )

        expected_counts = {
            "Vehicle": int(
                source_row[
                    "vehicle_count"
                ]
            ),

            "Pedestrian": int(
                source_row[
                    "pedestrian_count"
                ]
            ),

            "Cyclist": int(
                source_row[
                    "cyclist_count"
                ]
            ),
        }

        actual_coco_counts = {
            class_name: int(
                coco_counts[class_name]
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        }

        actual_yolo_counts = {
            class_name: int(
                yolo_counts[class_name]
            )
            for class_name in [
                "Vehicle",
                "Pedestrian",
                "Cyclist",
            ]
        }

        class_counts_match = (
            actual_coco_counts
            == expected_counts
            and actual_yolo_counts
            == expected_counts
        )

        equivalence_image_passed = (
            parse_bool(
                equivalence_row[
                    "image_passed"
                ]
            )
        )

        count_checks = {
            "source_target_count": (
                len(coco_box_records)
                == int(
                    source_row[
                        "target_box_count"
                    ]
                )
            ),

            "coco_yolo_count": (
                len(coco_box_records)
                == len(yolo_box_records)
            ),

            "equivalence_manifest_count": (
                len(coco_box_records)
                == int(
                    equivalence_row[
                        "matched_box_count"
                    ]
                )
            ),

            "class_counts": (
                class_counts_match
            ),

            "equivalence_passed": (
                equivalence_image_passed
            ),
        }

        sample_passed = all(
            count_checks.values()
        )

        if not sample_passed:
            add_issue(
                issues,
                partition_name,
                "visual_sample_check_failed",
                identifier,
                str(count_checks),
            )

        coco_visual = draw_annotations(
            source_image=source_image,
            records=coco_box_records,
            annotation_source=(
                "Canonical COCO"
            ),
        )

        yolo_visual = draw_annotations(
            source_image=source_image,
            records=yolo_box_records,
            annotation_source=(
                "Derived YOLO"
            ),
        )

        safe_reason = safe_filename_component(
            selection_reason
        )

        output_stem = (
            f"{sample_number:02d}_"
            f"{safe_reason}__"
            f"{Path(image_filename).stem}"
        )

        coco_visual_path = (
            specification[
                "coco_visual_dir"
            ]
            / f"{output_stem}.png"
        )

        yolo_visual_path = (
            specification[
                "yolo_visual_dir"
            ]
            / f"{output_stem}.png"
        )

        comparison_path = (
            COMPARISON_DIR
            / f"{output_stem}.png"
        )

        comparison_visual = (
            make_comparison(
                coco_visual=coco_visual,
                yolo_visual=yolo_visual,
                image_identifier=(
                    f"{selection_reason} | "
                    f"{source_image_id}"
                ),
                maximum_pixel_error=float(
                    equivalence_row[
                        "maximum_pixel_error"
                    ]
                ),
                minimum_iou=float(
                    equivalence_row[
                        "minimum_iou"
                    ]
                ),
            )
        )

        save_image(
            coco_visual_path,
            coco_visual,
        )

        save_image(
            yolo_visual_path,
            yolo_visual,
        )

        save_image(
            comparison_path,
            comparison_visual,
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
                    selection_reason
                ),
                "partition": (
                    partition_name
                ),
                "global_image_id": (
                    global_image_id
                ),
                "source_image_id": (
                    source_image_id
                ),
                "image_filename": (
                    image_filename
                ),
                "target_box_count": int(
                    source_row[
                        "target_box_count"
                    ]
                ),
                "vehicle_count": int(
                    source_row[
                        "vehicle_count"
                    ]
                ),
                "pedestrian_count": int(
                    source_row[
                        "pedestrian_count"
                    ]
                ),
                "cyclist_count": int(
                    source_row[
                        "cyclist_count"
                    ]
                ),
                "is_negative": bool(
                    source_row[
                        "is_negative_bool"
                    ]
                ),
                "coco_box_count": len(
                    coco_box_records
                ),
                "yolo_box_count": len(
                    yolo_box_records
                ),
                "class_counts_match": (
                    class_counts_match
                ),
                "equivalence_image_passed": (
                    equivalence_image_passed
                ),
                "maximum_normalized_error": float(
                    equivalence_row[
                        "maximum_normalized_error"
                    ]
                ),
                "maximum_pixel_error": float(
                    equivalence_row[
                        "maximum_pixel_error"
                    ]
                ),
                "minimum_iou": float(
                    equivalence_row[
                        "minimum_iou"
                    ]
                ),
                "coco_visual_path": (
                    project_relative_path(
                        coco_visual_path
                    )
                ),
                "yolo_visual_path": (
                    project_relative_path(
                        yolo_visual_path
                    )
                ),
                "comparison_visual_path": (
                    project_relative_path(
                        comparison_path
                    )
                ),
                "sample_passed": (
                    sample_passed
                ),
            }
        )

    contact_sheet_path = (
        COMPARISON_DIR
        / "annotation_comparison_contact_sheet.png"
    )

    if (
        len(comparison_paths)
        == EXPECTED_SAMPLE_COUNT
    ):
        try:
            create_contact_sheet(
                comparison_paths=(
                    comparison_paths
                ),
                output_path=(
                    contact_sheet_path
                ),
            )

        except Exception as error:
            add_issue(
                issues,
                "combined",
                "contact_sheet_failed",
                "contact_sheet",
                str(error),
            )

    else:
        add_issue(
            issues,
            "combined",
            "contact_sheet_skipped",
            "contact_sheet",
            (
                f"Expected "
                f"{EXPECTED_SAMPLE_COUNT} "
                f"comparisons, found "
                f"{len(comparison_paths)}."
            ),
        )

    write_csv(
        VISUAL_MANIFEST_FILE,
        manifest_rows,
        VISUAL_MANIFEST_COLUMNS,
    )

    all_samples_passed = (
        len(manifest_rows)
        == EXPECTED_SAMPLE_COUNT
        and all(
            bool(
                row["sample_passed"]
            )
            for row in manifest_rows
        )
    )

    all_output_files_exist = all(
        Path(
            row["coco_visual_path"]
        ).exists()
        and Path(
            row["yolo_visual_path"]
        ).exists()
        and Path(
            row[
                "comparison_visual_path"
            ]
        ).exists()
        for row in manifest_rows
    )

    checks = {
        "sample_count": (
            len(manifest_rows)
            == EXPECTED_SAMPLE_COUNT
        ),

        "all_samples_passed": (
            all_samples_passed
        ),

        "all_output_files_exist": (
            all_output_files_exist
        ),

        "contact_sheet_exists": (
            contact_sheet_path.exists()
        ),

        "negative_sample_present": any(
            bool(row["is_negative"])
            for row in manifest_rows
        ),

        "cyclist_samples_present": any(
            int(row["cyclist_count"])
            > 0
            for row in manifest_rows
        ),

        "pedestrian_samples_present": any(
            int(
                row["pedestrian_count"]
            )
            > 0
            for row in manifest_rows
        ),

        "all_equivalence_records_passed": all(
            bool(
                row[
                    "equivalence_image_passed"
                ]
            )
            for row in manifest_rows
        ),
    }

    for check_name, passed in (
        checks.items()
    ):
        if not passed:
            add_issue(
                issues,
                "combined",
                "visual_check_failed",
                check_name,
                (
                    "The visual annotation "
                    "check returned false."
                ),
            )

    overall_passed = (
        all(
            checks.values()
        )
        and len(issues) == 0
    )

    report = {
        "milestone": 3,
        "step": 13,
        "purpose": (
            "Create deterministic visual "
            "comparisons of canonical COCO and "
            "derived YOLO annotations."
        ),
        "expected_samples": (
            EXPECTED_SAMPLE_COUNT
        ),
        "created_samples": len(
            manifest_rows
        ),
        "selection_reasons": [
            row["selection_reason"]
            for row in manifest_rows
        ],
        "maximum_selected_normalized_error": (
            max(
                (
                    float(
                        row[
                            "maximum_normalized_error"
                        ]
                    )
                    for row
                    in manifest_rows
                ),
                default=None,
            )
        ),
        "maximum_selected_pixel_error": (
            max(
                (
                    float(
                        row[
                            "maximum_pixel_error"
                        ]
                    )
                    for row
                    in manifest_rows
                ),
                default=None,
            )
        ),
        "minimum_selected_iou": (
            min(
                (
                    float(
                        row[
                            "minimum_iou"
                        ]
                    )
                    for row
                    in manifest_rows
                ),
                default=None,
            )
        ),
        "contact_sheet": (
            project_relative_path(
                contact_sheet_path
            )
            if contact_sheet_path.exists()
            else None
        ),
        "checks": checks,
        "issue_count": len(issues),
        "visual_annotation_checks_passed": (
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
    print("VISUAL ANNOTATION CHECK SUMMARY")
    print("=" * 76)

    for row in manifest_rows:
        print(
            f"\n[{row['sample_number']}] "
            f"{row['selection_reason']}"
        )

        print(
            f"  Partition: "
            f"{row['partition']}"
        )

        print(
            f"  Target boxes: "
            f"{row['target_box_count']}"
        )

        print(
            f"  COCO boxes: "
            f"{row['coco_box_count']}"
        )

        print(
            f"  YOLO boxes: "
            f"{row['yolo_box_count']}"
        )

        print(
            f"  Maximum pixel error: "
            f"{row['maximum_pixel_error']:.3e}"
        )

        print(
            f"  Minimum IoU: "
            f"{row['minimum_iou']:.12f}"
        )

        print(
            f"  Status: "
            f"{'PASSED' if row['sample_passed'] else 'FAILED'}"
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
        f"\nComparison images:\n"
        f"{COMPARISON_DIR.resolve()}"
    )

    print(
        f"\nVisual manifest:\n"
        f"{VISUAL_MANIFEST_FILE.resolve()}"
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
            "visual annotation check issue "
            "is resolved."
        )

        sys.exit(1)

    print(
        "\nStep 13 completed successfully. "
        "Manually inspect the comparison "
        "contact sheet before continuing."
    )


if __name__ == "__main__":
    main()