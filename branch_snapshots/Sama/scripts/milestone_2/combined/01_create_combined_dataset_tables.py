from pathlib import Path
import csv
import json
import pandas as pd
import yaml


KITTI_MAPPING_FILE = Path(
    "data/kitti/selection/class_mapping.yaml"
)

KITTI_SUMMARY_FILE = Path(
    "data/kitti/statistics/dataset_summary.json"
)

WAYMO_MAPPING_FILE = Path(
    "data/waymo/representative_subset/"
    "annotations/class_mapping.yaml"
)

WAYMO_SUBSET_SUMMARY_FILE = Path(
    "data/waymo/representative_subset/"
    "metadata/subset_summary.json"
)

WAYMO_SELECTION_SUMMARY_FILE = Path(
    "data/waymo/selection/selection_summary.json"
)

WAYMO_MANIFEST_FILE = Path(
    "data/waymo/representative_subset/"
    "metadata/manifest.csv"
)

OUTPUT_DIR = Path("docs/milestone_2")

CLASS_MAPPING_CSV = (
    OUTPUT_DIR / "class_mapping_table.csv"
)

CLASS_MAPPING_MD = (
    OUTPUT_DIR / "class_mapping_table.md"
)

DATASET_COMPARISON_CSV = (
    OUTPUT_DIR / "dataset_comparison_table.csv"
)

DATASET_COMPARISON_MD = (
    OUTPUT_DIR / "dataset_comparison_table.md"
)

COMBINED_SUMMARY_FILE = (
    OUTPUT_DIR / "combined_dataset_summary.json"
)


KITTI_CLASS_NOTES = {
    "Car": "Merged into the common Vehicle superclass.",
    "Van": "Merged into the common Vehicle superclass.",
    "Truck": "Merged into the common Vehicle superclass.",
    "Pedestrian": "Direct mapping to Pedestrian.",
    "Person_sitting": (
        "Merged into Pedestrian as a human road-user class."
    ),
    "Cyclist": "Direct mapping to Cyclist.",
    "Tram": (
        "Ignored because it is outside the common "
        "KITTI-Waymo three-class task."
    ),
    "Misc": (
        "Ignored because it is a heterogeneous and "
        "ambiguous category."
    ),
    "DontCare": (
        "Retained as an ignored evaluation region; "
        "not trained as an object class."
    ),
}


WAYMO_CLASS_NOTES = {
    "Vehicle": "Direct mapping to Vehicle.",
    "Pedestrian": "Direct mapping to Pedestrian.",
    "Cyclist": "Direct mapping to Cyclist.",
    "Sign": (
        "Ignored because traffic signs are outside "
        "the common three-class task."
    ),
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found:\n{path.resolve()}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required YAML file not found:\n{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(
            f"YAML file does not contain a mapping:\n{path}"
        )

    return content


def write_csv(
    path: Path,
    records: list[dict],
    fieldnames: list[str],
) -> None:
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
        writer.writerows(records)


def markdown_escape(value) -> str:
    if value is None:
        return ""

    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def write_markdown_table(
    path: Path,
    title: str,
    records: list[dict],
    columns: list[tuple[str, str]],
    introduction: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        introduction,
        "",
        "| "
        + " | ".join(
            heading
            for _, heading in columns
        )
        + " |",
        "| "
        + " | ".join(
            "---"
            for _ in columns
        )
        + " |",
    ]

    for record in records:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(
                    record.get(key, "")
                )
                for key, _ in columns
            )
            + " |"
        )

    lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_class_mapping_records(
    kitti_mapping_config: dict,
    waymo_mapping_config: dict,
) -> list[dict]:
    records: list[dict] = []

    kitti_mapping = (
        kitti_mapping_config.get(
            "kitti_mapping",
            {},
        )
    )

    for original_class, entry in kitti_mapping.items():
        action = str(entry.get("action", ""))

        records.append(
            {
                "source_dataset": "KITTI",
                "source_class_id": "",
                "source_class_name": original_class,
                "action": action,
                "unified_class_id": (
                    entry.get("mapped_class_id")
                    if action == "map"
                    else ""
                ),
                "unified_class_name": (
                    entry.get("mapped_class_name")
                    if action == "map"
                    else ""
                ),
                "notes": KITTI_CLASS_NOTES.get(
                    original_class,
                    "",
                ),
            }
        )

    waymo_mapping = (
        waymo_mapping_config.get(
            "waymo_mapping",
            {},
        )
    )

    for original_id, entry in waymo_mapping.items():
        mapped_id = entry.get(
            "mapped_class_id"
        )

        mapped_name = entry.get(
            "mapped_class_name"
        )

        original_name = str(
            entry.get(
                "original_name",
                original_id,
            )
        )

        is_mapped = (
            mapped_id is not None
            and mapped_name not in {
                None,
                "ignore",
            }
        )

        records.append(
            {
                "source_dataset": "Waymo",
                "source_class_id": original_id,
                "source_class_name": original_name,
                "action": (
                    "map"
                    if is_mapped
                    else "ignore"
                ),
                "unified_class_id": (
                    mapped_id
                    if is_mapped
                    else ""
                ),
                "unified_class_name": (
                    mapped_name
                    if is_mapped
                    else ""
                ),
                "notes": WAYMO_CLASS_NOTES.get(
                    original_name,
                    "",
                ),
            }
        )

    return records


def value_or_blank(
    dictionary: dict,
    key: str,
):
    value = dictionary.get(key)

    if value is None:
        return ""

    return value


def build_dataset_comparison_records(
    kitti_summary: dict,
    waymo_subset_summary: dict,
    waymo_selection_summary: dict,
    waymo_image_presence: dict,
) -> list[dict]:
    """
    Build the combined KITTI-Waymo comparison table.

    KITTI is used for training and in-domain validation.
    Waymo is used only for external validation without retraining.

    Parameters
    ----------
    kitti_summary:
        Parsed contents of data/kitti/statistics/dataset_summary.json.

    waymo_subset_summary:
        Parsed contents of the Waymo representative-subset summary.

    waymo_selection_summary:
        Parsed contents of the Waymo segment-selection summary.

    waymo_image_presence:
        Dictionary containing the number of Waymo images that contain
        at least one Vehicle, Pedestrian, or Cyclist box.

        Expected keys:
        - vehicle
        - pedestrian
        - cyclist

    Returns
    -------
    list[dict]
        Records used to create the CSV and Markdown comparison tables.
    """
    kitti_all = kitti_summary["all"]
    kitti_train = kitti_summary["train"]
    kitti_val = kitti_summary["validation"]

    records = [
        {
            "metric": "Experimental role",
            "kitti_all": "Source labeled benchmark",
            "kitti_train": "Model training",
            "kitti_validation": "In-domain validation",
            "waymo_external_validation": (
                "External validation only"
            ),
            "notes": (
                "Waymo is not used for training, fine-tuning, "
                "hyperparameter selection, or model selection."
            ),
        },
        {
            "metric": "Source split",
            "kitti_all": (
                "Official labeled training set"
            ),
            "kitti_train": (
                "Project train split"
            ),
            "kitti_validation": (
                "Project validation split"
            ),
            "waymo_external_validation": (
                "Official Waymo validation split"
            ),
            "notes": (
                "KITTI official testing labels are not "
                "publicly available."
            ),
        },
        {
            "metric": "Images",
            "kitti_all": kitti_all["images"],
            "kitti_train": kitti_train["images"],
            "kitti_validation": kitti_val["images"],
            "waymo_external_validation": (
                waymo_subset_summary[
                    "number_of_selected_images"
                ]
            ),
            "notes": "",
        },
        {
            "metric": "Driving segments",
            "kitti_all": "",
            "kitti_train": "",
            "kitti_validation": "",
            "waymo_external_validation": (
                waymo_subset_summary[
                    "number_of_segments"
                ]
            ),
            "notes": (
                "KITTI object-detection images are treated "
                "as independent benchmark samples."
            ),
        },
        {
            "metric": "Target boxes",
            "kitti_all": kitti_all[
                "target_boxes"
            ],
            "kitti_train": kitti_train[
                "target_boxes"
            ],
            "kitti_validation": kitti_val[
                "target_boxes"
            ],
            "waymo_external_validation": (
                waymo_subset_summary[
                    "number_of_target_boxes"
                ]
            ),
            "notes": (
                "Target boxes include only Vehicle, "
                "Pedestrian, and Cyclist."
            ),
        },
        {
            "metric": "Vehicle boxes",
            "kitti_all": kitti_all[
                "vehicle_boxes"
            ],
            "kitti_train": kitti_train[
                "vehicle_boxes"
            ],
            "kitti_validation": kitti_val[
                "vehicle_boxes"
            ],
            "waymo_external_validation": (
                waymo_subset_summary[
                    "vehicle_box_count"
                ]
            ),
            "notes": "",
        },
        {
            "metric": "Pedestrian boxes",
            "kitti_all": kitti_all[
                "pedestrian_boxes"
            ],
            "kitti_train": kitti_train[
                "pedestrian_boxes"
            ],
            "kitti_validation": kitti_val[
                "pedestrian_boxes"
            ],
            "waymo_external_validation": (
                waymo_subset_summary[
                    "pedestrian_box_count"
                ]
            ),
            "notes": "",
        },
        {
            "metric": "Cyclist boxes",
            "kitti_all": kitti_all[
                "cyclist_boxes"
            ],
            "kitti_train": kitti_train[
                "cyclist_boxes"
            ],
            "kitti_validation": kitti_val[
                "cyclist_boxes"
            ],
            "waymo_external_validation": (
                waymo_subset_summary[
                    "cyclist_box_count"
                ]
            ),
            "notes": "",
        },
        {
            "metric": "Ignored boxes",
            "kitti_all": kitti_all[
                "ignored_boxes"
            ],
            "kitti_train": kitti_train[
                "ignored_boxes"
            ],
            "kitti_validation": kitti_val[
                "ignored_boxes"
            ],
            "waymo_external_validation": "",
            "notes": (
                "KITTI ignored boxes include Tram, Misc, "
                "and DontCare under the harmonized task. "
                "Waymo Sign annotations are excluded before "
                "the representative boxes table is created."
            ),
        },
        {
            "metric": "Images with no target boxes",
            "kitti_all": kitti_all[
                "target_empty_images"
            ],
            "kitti_train": kitti_train[
                "target_empty_images"
            ],
            "kitti_validation": kitti_val[
                "target_empty_images"
            ],
            "waymo_external_validation": (
                waymo_subset_summary[
                    "images_without_target_boxes"
                ]
            ),
            "notes": (
                "Negative images remain available for "
                "false-positive analysis."
            ),
        },
        {
            "metric": "Images containing Vehicle",
            "kitti_all": kitti_all[
                "images_containing_vehicle"
            ],
            "kitti_train": kitti_train[
                "images_containing_vehicle"
            ],
            "kitti_validation": kitti_val[
                "images_containing_vehicle"
            ],
            "waymo_external_validation": (
                waymo_image_presence["vehicle"]
            ),
            "notes": (
                "Counts images containing at least one "
                "Vehicle target box."
            ),
        },
        {
            "metric": "Images containing Pedestrian",
            "kitti_all": kitti_all[
                "images_containing_pedestrian"
            ],
            "kitti_train": kitti_train[
                "images_containing_pedestrian"
            ],
            "kitti_validation": kitti_val[
                "images_containing_pedestrian"
            ],
            "waymo_external_validation": (
                waymo_image_presence["pedestrian"]
            ),
            "notes": (
                "Counts images containing at least one "
                "Pedestrian target box."
            ),
        },
        {
            "metric": "Images containing Cyclist",
            "kitti_all": kitti_all[
                "images_containing_cyclist"
            ],
            "kitti_train": kitti_train[
                "images_containing_cyclist"
            ],
            "kitti_validation": kitti_val[
                "images_containing_cyclist"
            ],
            "waymo_external_validation": (
                waymo_image_presence["cyclist"]
            ),
            "notes": (
                "Counts images containing at least one "
                "Cyclist target box."
            ),
        },
        {
            "metric": "Camera",
            "kitti_all": (
                "Left color camera (image_2)"
            ),
            "kitti_train": (
                "Left color camera (image_2)"
            ),
            "kitti_validation": (
                "Left color camera (image_2)"
            ),
            "waymo_external_validation": (
                waymo_subset_summary[
                    "camera_name"
                ]
            ),
            "notes": (
                "Both datasets use a forward-facing "
                "monocular RGB view."
            ),
        },
        {
            "metric": "Sampling rule",
            "kitti_all": (
                "Use all official labeled images"
            ),
            "kitti_train": (
                "Frozen stratified split"
            ),
            "kitti_validation": (
                "Frozen stratified split"
            ),
            "waymo_external_validation": (
                waymo_subset_summary[
                    "frame_sampling_rule"
                ]
            ),
            "notes": (
                "Waymo temporal redundancy is reduced "
                "through uniform frame sampling."
            ),
        },
        {
            "metric": "Random seed",
            "kitti_all": "",
            "kitti_train": 42,
            "kitti_validation": 42,
            "waymo_external_validation": "",
            "notes": (
                "The KITTI split was created once and "
                "frozen before model training."
            ),
        },
        {
            "metric": (
                "Selected segments by time of day"
            ),
            "kitti_all": (
                "Not provided as official image metadata"
            ),
            "kitti_train": "",
            "kitti_validation": "",
            "waymo_external_validation": (
                json.dumps(
                    waymo_selection_summary[
                        "time_of_day_distribution"
                    ],
                    sort_keys=True,
                )
            ),
            "notes": (
                "Waymo values are counts of selected "
                "driving segments, not image counts."
            ),
        },
        {
            "metric": (
                "Selected segments by location"
            ),
            "kitti_all": (
                "KITTI collection area; no comparable "
                "per-image location metadata used"
            ),
            "kitti_train": "",
            "kitti_validation": "",
            "waymo_external_validation": (
                json.dumps(
                    waymo_selection_summary[
                        "location_distribution"
                    ],
                    sort_keys=True,
                )
            ),
            "notes": (
                "Waymo values are counts of selected "
                "driving segments."
            ),
        },
        {
            "metric": (
                "Selected segments by weather"
            ),
            "kitti_all": (
                "Not provided as official image metadata"
            ),
            "kitti_train": "",
            "kitti_validation": "",
            "waymo_external_validation": (
                json.dumps(
                    waymo_selection_summary[
                        "weather_distribution"
                    ],
                    sort_keys=True,
                )
            ),
            "notes": (
                "Waymo values are segment counts. The "
                "available validation candidate pool "
                "contained only one rainy segment."
            ),
        },
    ]

    return records


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    kitti_mapping = load_yaml(
        KITTI_MAPPING_FILE
    )

    kitti_summary = load_json(
        KITTI_SUMMARY_FILE
    )

    waymo_mapping = load_yaml(
        WAYMO_MAPPING_FILE
    )

    waymo_subset_summary = load_json(
        WAYMO_SUBSET_SUMMARY_FILE
    )

    waymo_selection_summary = load_json(
        WAYMO_SELECTION_SUMMARY_FILE
    )

    if not WAYMO_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Required Waymo manifest not found:\n"
            f"{WAYMO_MANIFEST_FILE.resolve()}"
        )

    waymo_manifest = pd.read_csv(
        WAYMO_MANIFEST_FILE
    )

    waymo_image_presence = {
        "vehicle": int(
            (waymo_manifest["vehicle_count"] > 0).sum()
        ),
        "pedestrian": int(
            (waymo_manifest["pedestrian_count"] > 0).sum()
        ),
        "cyclist": int(
            (waymo_manifest["cyclist_count"] > 0).sum()
        ),
    }

    mapping_records = (
        build_class_mapping_records(
            kitti_mapping,
            waymo_mapping,
        )
    )

    mapping_fields = [
        "source_dataset",
        "source_class_id",
        "source_class_name",
        "action",
        "unified_class_id",
        "unified_class_name",
        "notes",
    ]

    write_csv(
        CLASS_MAPPING_CSV,
        mapping_records,
        mapping_fields,
    )

    write_markdown_table(
        path=CLASS_MAPPING_MD,
        title=(
            "Unified KITTI–Waymo Class Mapping"
        ),
        records=mapping_records,
        columns=[
            (
                "source_dataset",
                "Source dataset",
            ),
            (
                "source_class_id",
                "Source ID",
            ),
            (
                "source_class_name",
                "Source class",
            ),
            (
                "action",
                "Action",
            ),
            (
                "unified_class_id",
                "Unified ID",
            ),
            (
                "unified_class_name",
                "Unified class",
            ),
            (
                "notes",
                "Notes",
            ),
        ],
        introduction=(
            "This table defines the frozen class "
            "harmonization policy used by all models."
        ),
    )

    comparison_records = (
        build_dataset_comparison_records(
            kitti_summary,
            waymo_subset_summary,
            waymo_selection_summary,
            waymo_image_presence,
        )
    )

    comparison_fields = [
        "metric",
        "kitti_all",
        "kitti_train",
        "kitti_validation",
        "waymo_external_validation",
        "notes",
    ]

    write_csv(
        DATASET_COMPARISON_CSV,
        comparison_records,
        comparison_fields,
    )

    write_markdown_table(
        path=DATASET_COMPARISON_MD,
        title=(
            "KITTI–Waymo Dataset Comparison"
        ),
        records=comparison_records,
        columns=[
            ("metric", "Metric"),
            ("kitti_all", "KITTI all"),
            ("kitti_train", "KITTI train"),
            (
                "kitti_validation",
                "KITTI validation",
            ),
            (
                "waymo_external_validation",
                "Waymo external validation",
            ),
            ("notes", "Notes"),
        ],
        introduction=(
            "KITTI provides training and in-domain "
            "validation. Waymo provides external "
            "validation without retraining."
        ),
    )

    combined_summary = {
        "milestone": 2,
        "task": (
            "Three-class cross-dataset "
            "2D object detection"
        ),
        "unified_classes": {
            "0": "Vehicle",
            "1": "Pedestrian",
            "2": "Cyclist",
        },
        "kitti": {
            "role": (
                "training_and_in_domain_validation"
            ),
            "total_images": (
                kitti_summary["all"]["images"]
            ),
            "train_images": (
                kitti_summary["train"]["images"]
            ),
            "validation_images": (
                kitti_summary[
                    "validation"
                ]["images"]
            ),
            "random_seed": 42,
            "target_boxes": (
                kitti_summary[
                    "all"
                ]["target_boxes"]
            ),
            "integrity_passed": (
                kitti_summary[
                    "statistics_validation_passed"
                ]
            ),
        },
        "waymo": {
            "role": (
                "external_validation_without_retraining"
            ),
            "segments": (
                waymo_subset_summary[
                    "number_of_segments"
                ]
            ),
            "images": (
                waymo_subset_summary[
                    "number_of_selected_images"
                ]
            ),
            "target_boxes": (
                waymo_subset_summary[
                    "number_of_target_boxes"
                ]
            ),
            "camera": (
                waymo_subset_summary[
                    "camera_name"
                ]
            ),
            "sampling_rule": (
                waymo_subset_summary[
                    "frame_sampling_rule"
                ]
            ),
            "selection_frozen": (
                waymo_subset_summary[
                    "selection_frozen_before_model_evaluation"
                ]
            ),
        },
        "reproducibility_rules": [
            (
                "All detectors use the same KITTI "
                "train and validation split."
            ),
            (
                "All detectors use the same unified "
                "class mapping."
            ),
            (
                "All detectors are evaluated on the "
                "same Waymo representative subset."
            ),
            (
                "Waymo is not used for training, "
                "fine-tuning, or model selection."
            ),
        ],
        "generated_files": [
            CLASS_MAPPING_CSV.as_posix(),
            CLASS_MAPPING_MD.as_posix(),
            DATASET_COMPARISON_CSV.as_posix(),
            DATASET_COMPARISON_MD.as_posix(),
        ],
    }

    COMBINED_SUMMARY_FILE.write_text(
        json.dumps(
            combined_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("COMBINED KITTI-WAYMO TABLES CREATED")
    print("=" * 72)

    print(
        f"Class-mapping rows: "
        f"{len(mapping_records)}"
    )

    print(
        f"Dataset-comparison rows: "
        f"{len(comparison_records)}"
    )

    print("\nCreated files:")

    for path in [
        CLASS_MAPPING_CSV,
        CLASS_MAPPING_MD,
        DATASET_COMPARISON_CSV,
        DATASET_COMPARISON_MD,
        COMBINED_SUMMARY_FILE,
    ]:
        print(f"  {path.resolve()}")

    print("\nStep 8 completed successfully.")


if __name__ == "__main__":
    main()