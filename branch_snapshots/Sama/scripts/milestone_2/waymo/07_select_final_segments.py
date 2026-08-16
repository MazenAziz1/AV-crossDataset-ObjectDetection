from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


INPUT_FILE = Path(
    "data/waymo/selection/candidate_front_camera_stats.csv"
)

OUTPUT_FILE = Path(
    "data/waymo/selection/final_segments.csv"
)

SUMMARY_FILE = Path(
    "data/waymo/selection/selection_summary.json"
)

TARGET_SEGMENTS = 25


NUMERIC_COLUMNS = [
    "vehicle_box_count",
    "pedestrian_box_count",
    "cyclist_box_count",
    "small_box_count",
    "medium_box_count",
    "large_box_count",
    "front_target_frame_coverage_percent",
]


# The script first tries the preferred profile.
# If the exact combination is infeasible, it automatically
# tries slightly more relaxed profiles.
SELECTION_PROFILES = [
    {
        "name": "preferred",
        "time_ranges": {
            "Day": (14, 16),
            "Dawn/Dusk": (4, 6),
            "Night": (4, 6),
        },
        "location_ranges": {
            "location_phx": (9, 12),
            "location_sf": (9, 12),
            "location_other": (2, 5),
        },
        "density_ranges": {
            "low": (7, 10),
            "medium": (7, 10),
            "high": (7, 10),
        },
        "minimum_pedestrian_segments": 15,
        "minimum_cyclist_segments": 10,
    },
    {
        "name": "moderately_relaxed",
        "time_ranges": {
            "Day": (13, 18),
            "Dawn/Dusk": (3, 7),
            "Night": (3, 7),
        },
        "location_ranges": {
            "location_phx": (8, 13),
            "location_sf": (8, 13),
            "location_other": (1, 6),
        },
        "density_ranges": {
            "low": (6, 11),
            "medium": (6, 11),
            "high": (6, 11),
        },
        "minimum_pedestrian_segments": 12,
        "minimum_cyclist_segments": 8,
    },
    {
        "name": "minimum_diversity",
        "time_ranges": {
            "Day": (10, 20),
            "Dawn/Dusk": (2, 8),
            "Night": (2, 8),
        },
        "location_ranges": {
            "location_phx": (6, 15),
            "location_sf": (6, 15),
            "location_other": (1, 8),
        },
        "density_ranges": {
            "low": (5, 12),
            "medium": (5, 12),
            "high": (5, 12),
        },
        "minimum_pedestrian_segments": 10,
        "minimum_cyclist_segments": 6,
    },
]


def normalize_score(series: pd.Series) -> np.ndarray:
    """
    Log-transform and normalize a non-negative numeric column
    to the interval [0, 1].
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    transformed = np.log1p(
        np.maximum(numeric.to_numpy(dtype=float), 0.0)
    )

    minimum = transformed.min()
    maximum = transformed.max()

    if maximum == minimum:
        return np.zeros(len(transformed), dtype=float)

    return (transformed - minimum) / (maximum - minimum)


def create_selection_score(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Prefer segments with useful vulnerable-road-user and
    object-size coverage.

    Time, location and density diversity are enforced separately
    as optimization constraints.
    """
    contains_pedestrian = (
        dataframe["pedestrian_box_count"] > 0
    ).astype(float).to_numpy()

    contains_cyclist = (
        dataframe["cyclist_box_count"] > 0
    ).astype(float).to_numpy()

    score = (
        4.0 * contains_cyclist
        + 2.0 * contains_pedestrian
        + 1.25 * normalize_score(
            dataframe["cyclist_box_count"]
        )
        + 0.80 * normalize_score(
            dataframe["pedestrian_box_count"]
        )
        + 0.20 * normalize_score(
            dataframe["vehicle_box_count"]
        )
        + 0.45 * normalize_score(
            dataframe["small_box_count"]
        )
        + 0.20 * normalize_score(
            dataframe["medium_box_count"]
        )
        + 0.45 * normalize_score(
            dataframe["large_box_count"]
        )
        + 0.20 * normalize_score(
            dataframe[
                "front_target_frame_coverage_percent"
            ]
        )
    )

    # Tiny deterministic tie breaker.
    score += np.linspace(
        0.0,
        0.000001,
        len(dataframe),
    )

    return score


def add_constraint(
    rows: list[np.ndarray],
    lower_bounds: list[float],
    upper_bounds: list[float],
    names: list[str],
    mask: np.ndarray,
    lower: float,
    upper: float,
    name: str,
) -> None:
    rows.append(mask.astype(float))
    lower_bounds.append(float(lower))
    upper_bounds.append(float(upper))
    names.append(name)


def solve_profile(
    dataframe: pd.DataFrame,
    score: np.ndarray,
    profile: dict,
):
    number_of_candidates = len(dataframe)

    rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    constraint_names: list[str] = []

    # Exactly 25 final segments.
    add_constraint(
        rows,
        lower_bounds,
        upper_bounds,
        constraint_names,
        np.ones(number_of_candidates),
        TARGET_SEGMENTS,
        TARGET_SEGMENTS,
        "total_segments",
    )

    # Include the only rainy segment.
    rain_mask = (
        dataframe["weather"]
        .astype(str)
        .str.lower()
        .eq("rain")
        .to_numpy()
    )

    if rain_mask.any():
        add_constraint(
            rows,
            lower_bounds,
            upper_bounds,
            constraint_names,
            rain_mask,
            1,
            np.inf,
            "minimum_rain_segments",
        )

    # Time-of-day coverage.
    for category, limits in profile["time_ranges"].items():
        category_mask = (
            dataframe["time_of_day"]
            .astype(str)
            .eq(category)
            .to_numpy()
        )

        add_constraint(
            rows,
            lower_bounds,
            upper_bounds,
            constraint_names,
            category_mask,
            limits[0],
            limits[1],
            f"time_{category}",
        )

    # Location coverage.
    for category, limits in profile[
        "location_ranges"
    ].items():
        category_mask = (
            dataframe["location"]
            .astype(str)
            .eq(category)
            .to_numpy()
        )

        add_constraint(
            rows,
            lower_bounds,
            upper_bounds,
            constraint_names,
            category_mask,
            limits[0],
            limits[1],
            f"location_{category}",
        )

    # Traffic-density coverage.
    for category, limits in profile[
        "density_ranges"
    ].items():
        category_mask = (
            dataframe["density_group"]
            .astype(str)
            .eq(category)
            .to_numpy()
        )

        add_constraint(
            rows,
            lower_bounds,
            upper_bounds,
            constraint_names,
            category_mask,
            limits[0],
            limits[1],
            f"density_{category}",
        )

    # Safety-relevant class coverage.
    pedestrian_mask = (
        dataframe["pedestrian_box_count"] > 0
    ).to_numpy()

    cyclist_mask = (
        dataframe["cyclist_box_count"] > 0
    ).to_numpy()

    add_constraint(
        rows,
        lower_bounds,
        upper_bounds,
        constraint_names,
        pedestrian_mask,
        profile["minimum_pedestrian_segments"],
        np.inf,
        "minimum_pedestrian_segments",
    )

    add_constraint(
        rows,
        lower_bounds,
        upper_bounds,
        constraint_names,
        cyclist_mask,
        profile["minimum_cyclist_segments"],
        np.inf,
        "minimum_cyclist_segments",
    )

    matrix = np.vstack(rows)

    constraints = LinearConstraint(
        matrix,
        np.asarray(lower_bounds),
        np.asarray(upper_bounds),
    )

    # scipy.optimize.milp minimizes, so negate the score.
    result = milp(
        c=-score,
        integrality=np.ones(
            number_of_candidates,
            dtype=int,
        ),
        bounds=Bounds(
            np.zeros(number_of_candidates),
            np.ones(number_of_candidates),
        ),
        constraints=constraints,
        options={
            "time_limit": 120,
        },
    )

    return result, constraint_names


def create_selection_reason(
    row: pd.Series,
) -> str:
    reasons: list[str] = []

    if str(row["weather"]).lower() == "rain":
        reasons.append("rare_rain")

    reasons.append(
        str(row["time_of_day"])
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
    )

    reasons.append(str(row["location"]))
    reasons.append(f"density_{row['density_group']}")

    if int(row["pedestrian_box_count"]) > 0:
        reasons.append("front_pedestrian_coverage")

    if int(row["cyclist_box_count"]) > 0:
        reasons.append("front_cyclist_coverage")

    if int(row["small_box_count"]) > 0:
        reasons.append("small_object_coverage")

    return "|".join(reasons)


def value_counts_dict(
    series: pd.Series,
) -> dict:
    counts = series.value_counts(
        dropna=False
    )

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Candidate statistics file not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    dataframe = pd.read_csv(INPUT_FILE)

    if len(dataframe) < TARGET_SEGMENTS:
        raise ValueError(
            f"Only {len(dataframe)} candidate segments are available, "
            f"but {TARGET_SEGMENTS} are required."
        )

    for column in NUMERIC_COLUMNS:
        if column not in dataframe.columns:
            raise KeyError(
                f"Required column missing: {column}"
            )

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).fillna(0)

    # Stable ordering helps reproducibility.
    dataframe = dataframe.sort_values(
        "segment_id"
    ).reset_index(drop=True)

    score = create_selection_score(dataframe)

    chosen_result = None
    chosen_profile = None
    chosen_constraint_names = None

    print("=" * 70)
    print("SELECTING FINAL WAYMO SEGMENTS")
    print("=" * 70)
    print(f"Candidate segments: {len(dataframe)}")
    print(f"Required final segments: {TARGET_SEGMENTS}")

    for profile in SELECTION_PROFILES:
        print(
            f"\nTrying selection profile: "
            f"{profile['name']}"
        )

        result, constraint_names = solve_profile(
            dataframe,
            score,
            profile,
        )

        print(f"Solver status: {result.message}")

        if result.success and result.x is not None:
            selected_count = int(
                np.sum(result.x > 0.5)
            )

            if selected_count == TARGET_SEGMENTS:
                chosen_result = result
                chosen_profile = profile
                chosen_constraint_names = (
                    constraint_names
                )
                break

    if chosen_result is None:
        raise RuntimeError(
            "No feasible 25-segment selection was found, "
            "even with the relaxed profiles."
        )

    selected_mask = chosen_result.x > 0.5

    final_segments = dataframe.loc[
        selected_mask
    ].copy()

    final_segments["selection_score"] = (
        score[selected_mask]
    )

    final_segments["selection_reason"] = (
        final_segments.apply(
            create_selection_reason,
            axis=1,
        )
    )

    final_segments[
        "selection_profile"
    ] = chosen_profile["name"]

    final_segments[
        "selection_frozen_before_model_evaluation"
    ] = True

    final_segments = final_segments.sort_values(
        by=[
            "time_of_day",
            "weather",
            "location",
            "density_group",
            "segment_id",
        ]
    ).reset_index(drop=True)

    final_segments.insert(
        0,
        "final_segment_number",
        range(1, len(final_segments) + 1),
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_segments.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = {
        "candidate_pool_size": int(len(dataframe)),
        "number_of_final_segments": int(
            len(final_segments)
        ),
        "selection_method": (
            "binary_mixed_integer_linear_optimization"
        ),
        "selection_profile_used": (
            chosen_profile["name"]
        ),
        "solver_message": str(
            chosen_result.message
        ),
        "camera": "FRONT",
        "waymo_split": "validation",
        "selection_frozen_before_model_evaluation": True,
        "time_of_day_distribution": value_counts_dict(
            final_segments["time_of_day"]
        ),
        "weather_distribution": value_counts_dict(
            final_segments["weather"]
        ),
        "location_distribution": value_counts_dict(
            final_segments["location"]
        ),
        "density_distribution": value_counts_dict(
            final_segments["density_group"]
        ),
        "segments_containing_front_vehicles": int(
            (
                final_segments["vehicle_box_count"] > 0
            ).sum()
        ),
        "segments_containing_front_pedestrians": int(
            (
                final_segments["pedestrian_box_count"] > 0
            ).sum()
        ),
        "segments_containing_front_cyclists": int(
            (
                final_segments["cyclist_box_count"] > 0
            ).sum()
        ),
        "total_front_vehicle_boxes": int(
            final_segments["vehicle_box_count"].sum()
        ),
        "total_front_pedestrian_boxes": int(
            final_segments[
                "pedestrian_box_count"
            ].sum()
        ),
        "total_front_cyclist_boxes": int(
            final_segments["cyclist_box_count"].sum()
        ),
        "total_small_boxes": int(
            final_segments["small_box_count"].sum()
        ),
        "total_medium_boxes": int(
            final_segments["medium_box_count"].sum()
        ),
        "total_large_boxes": int(
            final_segments["large_box_count"].sum()
        ),
        "active_constraints": (
            chosen_constraint_names
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL WAYMO SEGMENT SELECTION COMPLETE")
    print("=" * 70)

    print(
        f"Selection profile used: "
        f"{chosen_profile['name']}"
    )

    print(
        f"Final segments: "
        f"{len(final_segments)}"
    )

    print("\nTime-of-day distribution:")
    print(
        final_segments[
            "time_of_day"
        ].value_counts()
    )

    print("\nWeather distribution:")
    print(
        final_segments[
            "weather"
        ].value_counts()
    )

    print("\nLocation distribution:")
    print(
        final_segments[
            "location"
        ].value_counts()
    )

    print("\nDensity distribution:")
    print(
        final_segments[
            "density_group"
        ].value_counts()
    )

    print(
        "\nSegments containing pedestrians: "
        f"{int((final_segments['pedestrian_box_count'] > 0).sum())}"
    )

    print(
        "Segments containing cyclists: "
        f"{int((final_segments['cyclist_box_count'] > 0).sum())}"
    )

    print(
        f"\nFinal segment list saved to:\n"
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        f"\nSelection summary saved to:\n"
        f"{SUMMARY_FILE.resolve()}"
    )

    print(
        "\nThe final segment list is now frozen and must not "
        "be changed after model evaluation begins."
    )


if __name__ == "__main__":
    main()