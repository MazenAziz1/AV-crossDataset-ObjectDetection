from pathlib import Path
import json

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SUBSET_ROOT = Path(
    "data/waymo/representative_subset"
)

MANIFEST_FILE = (
    SUBSET_ROOT / "metadata" / "manifest.csv"
)

BOXES_FILE = (
    SUBSET_ROOT / "annotations" / "boxes.csv"
)

REPORT_FILE = (
    SUBSET_ROOT / "metadata"
    / "subset_validation_report.json"
)

VISUAL_DIR = (
    SUBSET_ROOT / "visual_checks"
)


CLASS_COLORS = {
    "Vehicle": (255, 80, 80),
    "Pedestrian": (80, 255, 80),
    "Cyclist": (80, 160, 255),
}


def select_first(
    dataframe: pd.DataFrame,
    condition: pd.Series,
):
    selected = dataframe.loc[condition]

    if selected.empty:
        return None

    return selected.iloc[0]


def draw_annotations(
    manifest_row: pd.Series,
    image_boxes: pd.DataFrame,
    output_name: str,
) -> None:
    image_path = (
        SUBSET_ROOT
        / str(manifest_row["relative_image_path"])
    )

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for _, box in image_boxes.iterrows():
        class_name = str(
            box["mapped_class_name"]
        )

        color = CLASS_COLORS.get(
            class_name,
            (255, 255, 0),
        )

        xmin = float(box["xmin"])
        ymin = float(box["ymin"])
        xmax = float(box["xmax"])
        ymax = float(box["ymax"])

        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline=color,
            width=3,
        )

        label = class_name

        text_bbox = draw.textbbox(
            (xmin, ymin),
            label,
            font=font,
        )

        text_width = (
            text_bbox[2] - text_bbox[0]
        )

        text_height = (
            text_bbox[3] - text_bbox[1]
        )

        label_y = max(
            0,
            ymin - text_height - 6,
        )

        draw.rectangle(
            [
                xmin,
                label_y,
                xmin + text_width + 6,
                label_y + text_height + 6,
            ],
            fill=color,
        )

        draw.text(
            (xmin + 3, label_y + 3),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    output_file = (
        VISUAL_DIR / output_name
    )

    image.save(
        output_file,
        format="JPEG",
        quality=95,
    )


def main() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing manifest:\n"
            f"{MANIFEST_FILE.resolve()}"
        )

    if not BOXES_FILE.exists():
        raise FileNotFoundError(
            f"Missing boxes file:\n"
            f"{BOXES_FILE.resolve()}"
        )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    boxes = pd.read_csv(
        BOXES_FILE
    )

    VISUAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors: list[str] = []

    # ------------------------------------------------------
    # 1. Manifest-level validation
    # ------------------------------------------------------
    duplicate_image_ids = int(
        manifest["image_id"]
        .duplicated()
        .sum()
    )

    if duplicate_image_ids > 0:
        errors.append(
            f"Duplicate image IDs: "
            f"{duplicate_image_ids}"
        )

    missing_image_files = []
    unreadable_image_files = []

    for _, row in manifest.iterrows():
        image_path = (
            SUBSET_ROOT
            / str(row["relative_image_path"])
        )

        if not image_path.exists():
            missing_image_files.append(
                str(image_path)
            )
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

        except Exception as error:
            unreadable_image_files.append(
                {
                    "path": str(image_path),
                    "error": str(error),
                }
            )

    if missing_image_files:
        errors.append(
            f"Missing image files: "
            f"{len(missing_image_files)}"
        )

    if unreadable_image_files:
        errors.append(
            f"Unreadable image files: "
            f"{len(unreadable_image_files)}"
        )

    # ------------------------------------------------------
    # 2. Annotation-reference validation
    # ------------------------------------------------------
    manifest_image_ids = set(
        manifest["image_id"].astype(str)
    )

    box_image_ids = set(
        boxes["image_id"].astype(str)
    )

    unknown_box_image_ids = sorted(
        box_image_ids - manifest_image_ids
    )

    if unknown_box_image_ids:
        errors.append(
            "Annotations reference unknown images: "
            f"{len(unknown_box_image_ids)}"
        )

    # ------------------------------------------------------
    # 3. Class validation
    # ------------------------------------------------------
    allowed_classes = {
        "Vehicle",
        "Pedestrian",
        "Cyclist",
    }

    found_classes = set(
        boxes["mapped_class_name"]
        .dropna()
        .astype(str)
        .unique()
    )

    unexpected_classes = sorted(
        found_classes - allowed_classes
    )

    if unexpected_classes:
        errors.append(
            "Unexpected classes: "
            + ", ".join(unexpected_classes)
        )

    # ------------------------------------------------------
    # 4. Bounding-box validation
    # ------------------------------------------------------
    invalid_width_height = boxes[
        (boxes["xmax"] <= boxes["xmin"])
        | (boxes["ymax"] <= boxes["ymin"])
    ]

    out_of_bounds_boxes = boxes[
        (boxes["xmin"] < 0)
        | (boxes["ymin"] < 0)
        | (boxes["xmax"] > boxes["image_width"])
        | (boxes["ymax"] > boxes["image_height"])
    ]

    if not invalid_width_height.empty:
        errors.append(
            "Boxes with invalid dimensions: "
            f"{len(invalid_width_height)}"
        )

    if not out_of_bounds_boxes.empty:
        errors.append(
            "Out-of-bounds boxes: "
            f"{len(out_of_bounds_boxes)}"
        )

    # ------------------------------------------------------
    # 5. Verify manifest object counts
    # ------------------------------------------------------
    class_counts = (
        boxes.groupby(
            ["image_id", "mapped_class_name"]
        )
        .size()
        .unstack(fill_value=0)
    )

    for class_name in allowed_classes:
        if class_name not in class_counts.columns:
            class_counts[class_name] = 0

    class_counts = class_counts.rename(
        columns={
            "Vehicle": "calculated_vehicle_count",
            "Pedestrian": "calculated_pedestrian_count",
            "Cyclist": "calculated_cyclist_count",
        }
    )

    calculated_totals = (
        boxes.groupby("image_id")
        .size()
        .rename(
            "calculated_number_of_target_boxes"
        )
    )

    verification = (
        manifest.set_index("image_id")
        .join(class_counts, how="left")
        .join(calculated_totals, how="left")
        .fillna(
            {
                "calculated_vehicle_count": 0,
                "calculated_pedestrian_count": 0,
                "calculated_cyclist_count": 0,
                "calculated_number_of_target_boxes": 0,
            }
        )
    )

    count_mismatches = verification[
        (
            verification["vehicle_count"]
            != verification[
                "calculated_vehicle_count"
            ]
        )
        | (
            verification["pedestrian_count"]
            != verification[
                "calculated_pedestrian_count"
            ]
        )
        | (
            verification["cyclist_count"]
            != verification[
                "calculated_cyclist_count"
            ]
        )
        | (
            verification[
                "number_of_target_boxes"
            ]
            != verification[
                "calculated_number_of_target_boxes"
            ]
        )
    ]

    if not count_mismatches.empty:
        errors.append(
            "Manifest/annotation count mismatches: "
            f"{len(count_mismatches)}"
        )

    # ------------------------------------------------------
    # 6. Select representative visual checks
    # ------------------------------------------------------
    samples = {}

    vehicle_sample = select_first(
        manifest,
        (
            (manifest["vehicle_count"] > 0)
            & (manifest["pedestrian_count"] == 0)
            & (manifest["cyclist_count"] == 0)
        ),
    )

    pedestrian_sample = select_first(
        manifest,
        manifest["pedestrian_count"] > 0,
    )

    cyclist_sample = select_first(
        manifest,
        manifest["cyclist_count"] > 0,
    )

    mixed_sample = select_first(
        manifest,
        (
            (manifest["vehicle_count"] > 0)
            & (manifest["pedestrian_count"] > 0)
            & (manifest["cyclist_count"] > 0)
        ),
    )

    night_sample = select_first(
        manifest,
        manifest["time_of_day"] == "Night",
    )

    rain_sample = select_first(
        manifest,
        manifest["weather"].str.lower() == "rain",
    )

    negative_sample = select_first(
        manifest,
        manifest["number_of_target_boxes"] == 0,
    )

    crowded_sample = (
        manifest.sort_values(
            "number_of_target_boxes",
            ascending=False,
        ).iloc[0]
        if not manifest.empty
        else None
    )

    sample_definitions = [
        (
            "sample_vehicle.jpg",
            vehicle_sample,
        ),
        (
            "sample_pedestrian.jpg",
            pedestrian_sample,
        ),
        (
            "sample_cyclist.jpg",
            cyclist_sample,
        ),
        (
            "sample_mixed_scene.jpg",
            mixed_sample,
        ),
        (
            "sample_night_scene.jpg",
            night_sample,
        ),
        (
            "sample_rain_scene.jpg",
            rain_sample,
        ),
        (
            "sample_negative_scene.jpg",
            negative_sample,
        ),
        (
            "sample_crowded_scene.jpg",
            crowded_sample,
        ),
    ]

    for filename, sample_row in sample_definitions:
        if sample_row is None:
            continue

        image_id = str(
            sample_row["image_id"]
        )

        image_boxes = boxes[
            boxes["image_id"].astype(str)
            == image_id
        ]

        draw_annotations(
            sample_row,
            image_boxes,
            filename,
        )

        samples[filename] = image_id

    # ------------------------------------------------------
    # 7. Final report
    # ------------------------------------------------------
    actual_jpg_files = list(
        (
            SUBSET_ROOT
            / "images"
            / "front"
        ).rglob("*.jpg")
    )

    report = {
        "validation_passed": len(errors) == 0,
        "errors": errors,
        "manifest_rows": int(
            len(manifest)
        ),
        "jpg_files_found": int(
            len(actual_jpg_files)
        ),
        "unique_segments": int(
            manifest["segment_id"].nunique()
        ),
        "duplicate_image_ids": (
            duplicate_image_ids
        ),
        "missing_image_files": int(
            len(missing_image_files)
        ),
        "unreadable_image_files": int(
            len(unreadable_image_files)
        ),
        "annotation_rows": int(
            len(boxes)
        ),
        "annotations_with_unknown_image_ids": int(
            len(unknown_box_image_ids)
        ),
        "invalid_dimension_boxes": int(
            len(invalid_width_height)
        ),
        "out_of_bounds_boxes": int(
            len(out_of_bounds_boxes)
        ),
        "manifest_annotation_count_mismatches": int(
            len(count_mismatches)
        ),
        "classes_found": sorted(
            found_classes
        ),
        "vehicle_boxes": int(
            (
                boxes["mapped_class_name"]
                == "Vehicle"
            ).sum()
        ),
        "pedestrian_boxes": int(
            (
                boxes["mapped_class_name"]
                == "Pedestrian"
            ).sum()
        ),
        "cyclist_boxes": int(
            (
                boxes["mapped_class_name"]
                == "Cyclist"
            ).sum()
        ),
        "negative_images": int(
            (
                manifest[
                    "number_of_target_boxes"
                ]
                == 0
            ).sum()
        ),
        "visual_checks_created": samples,
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("WAYMO REPRESENTATIVE SUBSET VALIDATION")
    print("=" * 72)

    print(
        f"Manifest images: "
        f"{len(manifest)}"
    )

    print(
        f"JPG files found: "
        f"{len(actual_jpg_files)}"
    )

    print(
        f"Segments represented: "
        f"{manifest['segment_id'].nunique()}"
    )

    print(
        f"Annotation rows: "
        f"{len(boxes)}"
    )

    print(
        f"Missing images: "
        f"{len(missing_image_files)}"
    )

    print(
        f"Unreadable images: "
        f"{len(unreadable_image_files)}"
    )

    print(
        f"Unknown annotation image IDs: "
        f"{len(unknown_box_image_ids)}"
    )

    print(
        f"Invalid boxes: "
        f"{len(invalid_width_height)}"
    )

    print(
        f"Out-of-bounds boxes: "
        f"{len(out_of_bounds_boxes)}"
    )

    print(
        f"Manifest count mismatches: "
        f"{len(count_mismatches)}"
    )

    print(
        f"Visual checks created: "
        f"{len(samples)}"
    )

    if errors:
        print("\nValidation status: FAILED")

        for error in errors:
            print(f"  - {error}")
    else:
        print("\nValidation status: PASSED")

    print(
        f"\nValidation report:\n"
        f"{REPORT_FILE.resolve()}"
    )

    print(
        f"\nVisual checks:\n"
        f"{VISUAL_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()