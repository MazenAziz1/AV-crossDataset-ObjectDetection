from pathlib import Path
import json

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


IMAGE_DIR = Path(
    "data/kitti/raw/training/image_2"
)

STATISTICS_DIR = Path(
    "data/kitti/statistics"
)

IMAGE_STATS_FILE = (
    STATISTICS_DIR
    / "image_level_statistics.csv"
)

OBJECT_STATS_FILE = (
    STATISTICS_DIR
    / "object_level_statistics.csv"
)

OUTPUT_DIR = Path(
    "data/kitti/visual_checks"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "visual_checks_manifest.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "visual_checks_summary.json"
)


BOX_COLORS = {
    "Vehicle": (255, 80, 80),
    "Pedestrian": (80, 255, 80),
    "Cyclist": (80, 160, 255),
    "Ignored": (255, 220, 80),
}


def convert_boolean_column(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def first_or_none(
    dataframe: pd.DataFrame,
):
    if dataframe.empty:
        return None

    return dataframe.iloc[0]


def choose_samples(
    images: pd.DataFrame,
    objects: pd.DataFrame,
) -> dict:
    samples = {}

    vehicle_only = images[
        (images["vehicle_count"] > 0)
        & (images["pedestrian_count"] == 0)
        & (images["cyclist_count"] == 0)
    ].sort_values(
        "vehicle_count",
        ascending=False,
    )

    samples["sample_vehicle.jpg"] = (
        first_or_none(vehicle_only)
    )

    pedestrian_images = images[
        images["pedestrian_count"] > 0
    ].sort_values(
        "pedestrian_count",
        ascending=False,
    )

    samples["sample_pedestrian.jpg"] = (
        first_or_none(pedestrian_images)
    )

    cyclist_images = images[
        images["cyclist_count"] > 0
    ].sort_values(
        "cyclist_count",
        ascending=False,
    )

    samples["sample_cyclist.jpg"] = (
        first_or_none(cyclist_images)
    )

    mixed_images = images[
        (images["vehicle_count"] > 0)
        & (images["pedestrian_count"] > 0)
        & (images["cyclist_count"] > 0)
    ].sort_values(
        "target_box_count",
        ascending=False,
    )

    if mixed_images.empty:
        mixed_images = images[
            (
                images[
                    [
                        "contains_vehicle",
                        "contains_pedestrian",
                        "contains_cyclist",
                    ]
                ].sum(axis=1)
                >= 2
            )
        ].sort_values(
            "target_box_count",
            ascending=False,
        )

    samples["sample_mixed_scene.jpg"] = (
        first_or_none(mixed_images)
    )

    crowded = images.sort_values(
        "target_box_count",
        ascending=False,
    )

    samples["sample_crowded_scene.jpg"] = (
        first_or_none(crowded)
    )

    occluded_ids = (
        objects[
            objects["is_target_class"]
            & objects[
                "occlusion_name"
            ].isin(
                [
                    "partly_occluded",
                    "largely_occluded",
                ]
            )
        ]
        .groupby("image_id")
        .size()
        .sort_values(
            ascending=False
        )
    )

    if not occluded_ids.empty:
        occluded_image_id = str(
            occluded_ids.index[0]
        )

        samples[
            "sample_occluded_scene.jpg"
        ] = first_or_none(
            images[
                images["image_id"]
                == occluded_image_id
            ]
        )
    else:
        samples[
            "sample_occluded_scene.jpg"
        ] = None

    truncated_ids = (
        objects[
            objects["is_target_class"]
            & (
                objects["truncation"]
                > 0.30
            )
        ]
        .groupby("image_id")
        .size()
        .sort_values(
            ascending=False
        )
    )

    if not truncated_ids.empty:
        truncated_image_id = str(
            truncated_ids.index[0]
        )

        samples[
            "sample_truncated_scene.jpg"
        ] = first_or_none(
            images[
                images["image_id"]
                == truncated_image_id
            ]
        )
    else:
        samples[
            "sample_truncated_scene.jpg"
        ] = None

    dontcare_ids = (
        objects[
            objects["original_class"]
            == "DontCare"
        ]
        .groupby("image_id")
        .size()
        .sort_values(
            ascending=False
        )
    )

    if not dontcare_ids.empty:
        dontcare_image_id = str(
            dontcare_ids.index[0]
        )

        samples[
            "sample_dontcare_scene.jpg"
        ] = first_or_none(
            images[
                images["image_id"]
                == dontcare_image_id
            ]
        )
    else:
        samples[
            "sample_dontcare_scene.jpg"
        ] = None

    return samples


def draw_sample(
    image_id: str,
    image_objects: pd.DataFrame,
    output_file: Path,
    include_ignored: bool,
) -> int:
    image_file = (
        IMAGE_DIR / f"{image_id}.png"
    )

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_file.resolve()}"
        )

    with Image.open(image_file) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    drawn_count = 0

    for _, row in image_objects.iterrows():
        is_target = bool(
            row["is_target_class"]
        )

        if not is_target and not include_ignored:
            continue

        if is_target:
            class_name = str(
                row["mapped_class_name"]
            )

            label = (
                f"{row['original_class']}"
                f" -> {class_name}"
            )
        else:
            class_name = "Ignored"

            label = (
                f"{row['original_class']}"
                " -> Ignore"
            )

        color = BOX_COLORS[
            class_name
        ]

        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

        draw.rectangle(
            [
                xmin,
                ymin,
                xmax,
                ymax,
            ],
            outline=color,
            width=3,
        )

        text_box = draw.textbbox(
            (xmin, ymin),
            label,
            font=font,
        )

        text_width = (
            text_box[2] - text_box[0]
        )

        text_height = (
            text_box[3] - text_box[1]
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
            (
                xmin + 3,
                label_y + 3,
            ),
            label,
            fill=(0, 0, 0),
            font=font,
        )

        drawn_count += 1

    image.save(
        output_file,
        format="JPEG",
        quality=95,
    )

    return drawn_count


def main() -> None:
    required_files = [
        IMAGE_STATS_FILE,
        OBJECT_STATS_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file missing:\n"
                f"{file.resolve()}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = pd.read_csv(
        IMAGE_STATS_FILE,
        dtype={"image_id": str},
    )

    objects = pd.read_csv(
        OBJECT_STATS_FILE,
        dtype={"image_id": str},
    )

    images["image_id"] = (
        images["image_id"]
        .astype(str)
        .str.zfill(6)
    )

    objects["image_id"] = (
        objects["image_id"]
        .astype(str)
        .str.zfill(6)
    )

    objects["is_target_class"] = (
        convert_boolean_column(
            objects["is_target_class"]
        )
    )

    samples = choose_samples(
        images,
        objects,
    )

    manifest_records = []

    for output_name, sample_row in samples.items():
        if sample_row is None:
            print(
                f"Skipped {output_name}: "
                "no suitable image found"
            )
            continue

        image_id = str(
            sample_row["image_id"]
        ).zfill(6)

        image_objects = objects[
            objects["image_id"]
            == image_id
        ].copy()

        include_ignored = (
            output_name
            == "sample_dontcare_scene.jpg"
        )

        output_file = (
            OUTPUT_DIR / output_name
        )

        drawn_boxes = draw_sample(
            image_id=image_id,
            image_objects=image_objects,
            output_file=output_file,
            include_ignored=include_ignored,
        )

        manifest_records.append(
            {
                "output_file": output_name,
                "image_id": image_id,
                "split": str(
                    sample_row["split"]
                ),
                "target_box_count": int(
                    sample_row[
                        "target_box_count"
                    ]
                ),
                "ignored_box_count": int(
                    sample_row[
                        "ignored_box_count"
                    ]
                ),
                "vehicle_count": int(
                    sample_row[
                        "vehicle_count"
                    ]
                ),
                "pedestrian_count": int(
                    sample_row[
                        "pedestrian_count"
                    ]
                ),
                "cyclist_count": int(
                    sample_row[
                        "cyclist_count"
                    ]
                ),
                "boxes_drawn": (
                    drawn_boxes
                ),
                "ignored_boxes_included": (
                    include_ignored
                ),
            }
        )

        print(
            f"Created {output_name} "
            f"from image {image_id}"
        )

    manifest = pd.DataFrame(
        manifest_records
    )

    manifest.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    summary = {
        "visual_checks_created": int(
            len(manifest)
        ),
        "files": manifest_records,
        "legend": {
            "Vehicle": "red",
            "Pedestrian": "green",
            "Cyclist": "blue",
            "Ignored": "yellow",
        },
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("KITTI VISUAL CHECKS CREATED")
    print("=" * 68)

    print(
        f"Visual checks created: "
        f"{len(manifest)}"
    )

    print(
        f"\nManifest:\n"
        f"{MANIFEST_FILE.resolve()}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_FILE.resolve()}"
    )

    print(
        f"\nImages:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()