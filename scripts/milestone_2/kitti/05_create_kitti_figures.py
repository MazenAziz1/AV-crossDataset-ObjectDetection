from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATISTICS_DIR = Path(
    "data/kitti/statistics"
)

FIGURES_DIR = (
    STATISTICS_DIR / "figures"
)

OBJECT_STATS_FILE = (
    STATISTICS_DIR
    / "object_level_statistics.csv"
)

IMAGE_STATS_FILE = (
    STATISTICS_DIR
    / "image_level_statistics.csv"
)

ORIGINAL_DISTRIBUTION_FILE = (
    STATISTICS_DIR
    / "original_class_distribution.csv"
)

MAPPED_DISTRIBUTION_FILE = (
    STATISTICS_DIR
    / "mapped_class_distribution.csv"
)

OCCLUSION_STATISTICS_FILE = (
    STATISTICS_DIR
    / "occlusion_statistics.csv"
)

TRUNCATION_STATISTICS_FILE = (
    STATISTICS_DIR
    / "truncation_statistics.csv"
)

DIFFICULTY_STATISTICS_FILE = (
    STATISTICS_DIR
    / "difficulty_statistics.csv"
)

SUMMARY_FILE = (
    FIGURES_DIR / "figures_summary.json"
)


TARGET_CLASSES = [
    "Vehicle",
    "Pedestrian",
    "Cyclist",
]


def save_current_figure(
    output_file: Path,
) -> None:
    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


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


def create_original_class_distribution(
    distribution: pd.DataFrame,
) -> Path:
    data = (
        distribution[
            distribution["partition"] == "all"
        ]
        .sort_values(
            "count",
            ascending=False,
        )
    )

    output_file = (
        FIGURES_DIR
        / "original_class_distribution.png"
    )

    plt.figure(figsize=(11, 6))
    plt.bar(
        data["class_name"],
        data["count"],
    )
    plt.title(
        "KITTI Original Class Distribution"
    )
    plt.xlabel("Original KITTI class")
    plt.ylabel("Bounding-box count")
    plt.xticks(rotation=35, ha="right")

    save_current_figure(output_file)
    return output_file


def create_mapped_class_distribution(
    distribution: pd.DataFrame,
) -> Path:
    data = (
        distribution[
            distribution["partition"] == "all"
        ]
        .sort_values(
            "count",
            ascending=False,
        )
    )

    output_file = (
        FIGURES_DIR
        / "mapped_class_distribution.png"
    )

    plt.figure(figsize=(9, 6))
    plt.bar(
        data["class_name"],
        data["count"],
    )
    plt.title(
        "KITTI Harmonized Class Distribution"
    )
    plt.xlabel("Harmonized class")
    plt.ylabel("Bounding-box count")
    plt.xticks(rotation=20, ha="right")

    save_current_figure(output_file)
    return output_file


def create_train_val_distribution(
    objects: pd.DataFrame,
) -> Path:
    target_objects = objects[
        objects["is_target_class"]
    ].copy()

    counts = (
        target_objects.groupby(
            [
                "mapped_class_name",
                "split",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    pivot = counts.pivot(
        index="mapped_class_name",
        columns="split",
        values="count",
    ).fillna(0)

    pivot = pivot.reindex(
        index=TARGET_CLASSES,
        columns=["train", "val"],
        fill_value=0,
    )

    output_file = (
        FIGURES_DIR
        / "train_val_class_distribution.png"
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    axis.set_title(
        "KITTI Train and Validation Class Distribution"
    )
    axis.set_xlabel("Harmonized class")
    axis.set_ylabel("Bounding-box count")
    axis.tick_params(
        axis="x",
        rotation=0,
    )
    axis.legend(
        title="Split",
    )

    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return output_file


def create_bbox_area_distribution(
    objects: pd.DataFrame,
) -> Path:
    target_objects = objects[
        objects["is_target_class"]
    ].copy()

    areas = pd.to_numeric(
        target_objects["bbox_area"],
        errors="coerce",
    )

    areas = areas[
        np.isfinite(areas)
        & (areas > 0)
    ]

    log_areas = np.log10(
        areas.to_numpy()
    )

    output_file = (
        FIGURES_DIR
        / "bbox_area_distribution.png"
    )

    plt.figure(figsize=(10, 6))
    plt.hist(
        log_areas,
        bins=50,
    )
    plt.title(
        "KITTI Target Bounding-Box Area Distribution"
    )
    plt.xlabel(
        "log10 bounding-box area in pixels²"
    )
    plt.ylabel("Bounding-box count")

    save_current_figure(output_file)
    return output_file


def create_objects_per_image_distribution(
    images: pd.DataFrame,
) -> Path:
    counts = pd.to_numeric(
        images["target_box_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    maximum = int(counts.max())

    bins = np.arange(
        -0.5,
        maximum + 1.5,
        1,
    )

    output_file = (
        FIGURES_DIR
        / "objects_per_image_distribution.png"
    )

    plt.figure(figsize=(11, 6))
    plt.hist(
        counts,
        bins=bins,
    )
    plt.title(
        "KITTI Target Objects per Image"
    )
    plt.xlabel(
        "Number of Vehicle, Pedestrian, and Cyclist boxes"
    )
    plt.ylabel("Image count")

    save_current_figure(output_file)
    return output_file


def create_grouped_statistics_figure(
    dataframe: pd.DataFrame,
    group_column: str,
    title: str,
    xlabel: str,
    output_name: str,
) -> Path:
    data = dataframe[
        dataframe["partition"] == "all"
    ].copy()

    pivot = data.pivot_table(
        index=group_column,
        columns="mapped_class",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    pivot = pivot.reindex(
        columns=TARGET_CLASSES,
        fill_value=0,
    )

    output_file = (
        FIGURES_DIR / output_name
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(12, 6),
    )

    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(
        "Bounding-box count"
    )
    axis.tick_params(
        axis="x",
        rotation=30,
    )
    axis.legend(
        title="Mapped class"
    )

    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return output_file


def main() -> None:
    required_files = [
        OBJECT_STATS_FILE,
        IMAGE_STATS_FILE,
        ORIGINAL_DISTRIBUTION_FILE,
        MAPPED_DISTRIBUTION_FILE,
        OCCLUSION_STATISTICS_FILE,
        TRUNCATION_STATISTICS_FILE,
        DIFFICULTY_STATISTICS_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required statistics file missing:\n"
                f"{file.resolve()}"
            )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    objects = pd.read_csv(
        OBJECT_STATS_FILE,
        dtype={"image_id": str},
    )

    images = pd.read_csv(
        IMAGE_STATS_FILE,
        dtype={"image_id": str},
    )

    original_distribution = pd.read_csv(
        ORIGINAL_DISTRIBUTION_FILE
    )

    mapped_distribution = pd.read_csv(
        MAPPED_DISTRIBUTION_FILE
    )

    occlusion_statistics = pd.read_csv(
        OCCLUSION_STATISTICS_FILE
    )

    truncation_statistics = pd.read_csv(
        TRUNCATION_STATISTICS_FILE
    )

    difficulty_statistics = pd.read_csv(
        DIFFICULTY_STATISTICS_FILE
    )

    objects["is_target_class"] = (
        convert_boolean_column(
            objects["is_target_class"]
        )
    )

    created_files = []

    created_files.append(
        create_original_class_distribution(
            original_distribution
        )
    )

    created_files.append(
        create_mapped_class_distribution(
            mapped_distribution
        )
    )

    created_files.append(
        create_train_val_distribution(
            objects
        )
    )

    created_files.append(
        create_bbox_area_distribution(
            objects
        )
    )

    created_files.append(
        create_objects_per_image_distribution(
            images
        )
    )

    created_files.append(
        create_grouped_statistics_figure(
            dataframe=occlusion_statistics,
            group_column="occlusion_group",
            title=(
                "KITTI Target-Class Occlusion Distribution"
            ),
            xlabel="Occlusion group",
            output_name=(
                "occlusion_distribution.png"
            ),
        )
    )

    created_files.append(
        create_grouped_statistics_figure(
            dataframe=truncation_statistics,
            group_column="truncation_group",
            title=(
                "KITTI Target-Class Truncation Distribution"
            ),
            xlabel="Truncation group",
            output_name=(
                "truncation_distribution.png"
            ),
        )
    )

    created_files.append(
        create_grouped_statistics_figure(
            dataframe=difficulty_statistics,
            group_column="difficulty_group",
            title=(
                "KITTI Descriptive Difficulty Distribution"
            ),
            xlabel="Difficulty group",
            output_name=(
                "difficulty_distribution.png"
            ),
        )
    )

    summary = {
        "figure_count": len(created_files),
        "figures": [
            file.as_posix()
            for file in created_files
        ],
        "source_object_rows": int(
            len(objects)
        ),
        "source_image_rows": int(
            len(images)
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 68)
    print("KITTI STATISTICAL FIGURES CREATED")
    print("=" * 68)

    for file in created_files:
        print(f"Created: {file}")

    print(
        f"\nFigures created: "
        f"{len(created_files)}"
    )

    print(
        f"\nFigure summary:\n"
        f"{SUMMARY_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()