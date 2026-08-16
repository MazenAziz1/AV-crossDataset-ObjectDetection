from pathlib import Path
from typing import Any

import pandas as pd


STATS_DIR = Path("data/waymo/raw/validation/stats")
OUTPUT_FILE = Path("data/waymo/selection/segment_catalog.csv")

SEGMENT_COLUMN = "key.segment_context_name"
TIMESTAMP_COLUMN = "key.frame_timestamp_micros"

TIME_COLUMN = "[StatsComponent].time_of_day"
LOCATION_COLUMN = "[StatsComponent].location"
WEATHER_COLUMN = "[StatsComponent].weather"

TYPES_COLUMN = "[StatsComponent].camera_object_counts.types"
COUNTS_COLUMN = "[StatsComponent].camera_object_counts.counts"

# Official Waymo object type identifiers.
VEHICLE_TYPE = 1
PEDESTRIAN_TYPE = 2
SIGN_TYPE = 3
CYCLIST_TYPE = 4


def to_list(value: Any) -> list:
    """Convert a Parquet list value into a normal Python list."""
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    return list(value)


def get_object_count(
    object_types: Any,
    object_counts: Any,
    target_type: int,
) -> int:
    """Return the count for one Waymo object type in one frame."""
    types = to_list(object_types)
    counts = to_list(object_counts)

    for object_type, count in zip(types, counts):
        if int(object_type) == target_type:
            return int(count)

    return 0


def get_representative_value(series: pd.Series) -> str:
    """
    Return the most frequent non-null value.

    Time, location and weather should normally remain constant throughout
    one segment.
    """
    clean = series.dropna().astype(str)

    if clean.empty:
        return "unknown"

    modes = clean.mode()

    if not modes.empty:
        return str(modes.iloc[0])

    return str(clean.iloc[0])


def main() -> None:
    parquet_files = sorted(STATS_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in: {STATS_DIR.resolve()}"
        )

    required_columns = [
        SEGMENT_COLUMN,
        TIMESTAMP_COLUMN,
        TIME_COLUMN,
        LOCATION_COLUMN,
        WEATHER_COLUMN,
        TYPES_COLUMN,
        COUNTS_COLUMN,
    ]

    segment_records: list[dict] = []

    for file_number, parquet_file in enumerate(parquet_files, start=1):
        dataframe = pd.read_parquet(
            parquet_file,
            columns=required_columns,
        )

        dataframe["vehicle_count"] = dataframe.apply(
            lambda row: get_object_count(
                row[TYPES_COLUMN],
                row[COUNTS_COLUMN],
                VEHICLE_TYPE,
            ),
            axis=1,
        )

        dataframe["pedestrian_count"] = dataframe.apply(
            lambda row: get_object_count(
                row[TYPES_COLUMN],
                row[COUNTS_COLUMN],
                PEDESTRIAN_TYPE,
            ),
            axis=1,
        )

        dataframe["cyclist_count"] = dataframe.apply(
            lambda row: get_object_count(
                row[TYPES_COLUMN],
                row[COUNTS_COLUMN],
                CYCLIST_TYPE,
            ),
            axis=1,
        )

        dataframe["sign_count"] = dataframe.apply(
            lambda row: get_object_count(
                row[TYPES_COLUMN],
                row[COUNTS_COLUMN],
                SIGN_TYPE,
            ),
            axis=1,
        )

        # Normally one Parquet file contains one segment, but grouping makes
        # the script safe if a file contains more than one.
        for segment_id, segment_data in dataframe.groupby(SEGMENT_COLUMN):
            number_of_frames = len(segment_data)

            average_vehicle_count = segment_data["vehicle_count"].mean()
            average_pedestrian_count = segment_data[
                "pedestrian_count"
            ].mean()
            average_cyclist_count = segment_data["cyclist_count"].mean()

            average_target_objects = (
                average_vehicle_count
                + average_pedestrian_count
                + average_cyclist_count
            )

            segment_records.append(
                {
                    "segment_id": segment_id,
                    "source_file": parquet_file.name,
                    "number_of_frames": number_of_frames,
                    "time_of_day": get_representative_value(
                        segment_data[TIME_COLUMN]
                    ),
                    "location": get_representative_value(
                        segment_data[LOCATION_COLUMN]
                    ),
                    "weather": get_representative_value(
                        segment_data[WEATHER_COLUMN]
                    ),
                    "average_vehicles_per_frame": round(
                        average_vehicle_count,
                        3,
                    ),
                    "maximum_vehicles_in_frame": int(
                        segment_data["vehicle_count"].max()
                    ),
                    "vehicle_frame_coverage_percent": round(
                        100
                        * (segment_data["vehicle_count"] > 0).mean(),
                        2,
                    ),
                    "average_pedestrians_per_frame": round(
                        average_pedestrian_count,
                        3,
                    ),
                    "maximum_pedestrians_in_frame": int(
                        segment_data["pedestrian_count"].max()
                    ),
                    "pedestrian_frame_coverage_percent": round(
                        100
                        * (segment_data["pedestrian_count"] > 0).mean(),
                        2,
                    ),
                    "average_cyclists_per_frame": round(
                        average_cyclist_count,
                        3,
                    ),
                    "maximum_cyclists_in_frame": int(
                        segment_data["cyclist_count"].max()
                    ),
                    "cyclist_frame_coverage_percent": round(
                        100
                        * (segment_data["cyclist_count"] > 0).mean(),
                        2,
                    ),
                    "average_signs_per_frame": round(
                        segment_data["sign_count"].mean(),
                        3,
                    ),
                    "average_target_objects_per_frame": round(
                        average_target_objects,
                        3,
                    ),
                }
            )

        print(
            f"Processed {file_number}/{len(parquet_files)}: "
            f"{parquet_file.name}"
        )

    catalog = pd.DataFrame(segment_records)

    if catalog.empty:
        raise ValueError("No segment records were generated.")

    # Divide scenes into approximate low, medium and high density groups.
    catalog["density_group"] = pd.qcut(
        catalog["average_target_objects_per_frame"].rank(
            method="first"
        ),
        q=3,
        labels=["low", "medium", "high"],
    )

    catalog = catalog.sort_values(
        by=[
            "time_of_day",
            "weather",
            "location",
            "segment_id",
        ]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("WAYMO SEGMENT CATALOG CREATED")
    print("=" * 60)

    print(f"Segments: {len(catalog)}")

    print("\nTime-of-day distribution:")
    print(catalog["time_of_day"].value_counts(dropna=False))

    print("\nWeather distribution:")
    print(catalog["weather"].value_counts(dropna=False))

    print("\nLocation distribution:")
    print(catalog["location"].value_counts(dropna=False))

    print("\nDensity distribution:")
    print(catalog["density_group"].value_counts(dropna=False))

    print(f"\nCatalog saved to:\n{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()