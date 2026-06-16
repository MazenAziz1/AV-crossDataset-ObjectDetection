from pathlib import Path

import pandas as pd


CATALOG_FILE = Path(
    "data/waymo/selection/segment_catalog.csv"
)

OUTPUT_FILE = Path(
    "data/waymo/selection/candidate_segments.csv"
)

TARGET_CANDIDATES = 70
RANDOM_SEED = 42

TOP_CYCLIST_SEGMENTS = 10
TOP_PEDESTRIAN_SEGMENTS = 10


def add_candidates(
    selected_records: list[pd.DataFrame],
    dataframe: pd.DataFrame,
    reason: str,
) -> None:
    """Add candidate rows while recording why they were selected."""
    if dataframe.empty:
        return

    selected = dataframe.copy()
    selected["selection_reason"] = reason
    selected_records.append(selected)


def combine_selection_reasons(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge duplicate segments while preserving all reasons
    that caused each segment to be selected.
    """
    reasons = (
        candidates.groupby("segment_id")["selection_reason"]
        .apply(lambda values: "|".join(sorted(set(values))))
        .reset_index()
    )

    unique_rows = (
        candidates.drop(columns=["selection_reason"])
        .drop_duplicates(subset=["segment_id"])
    )

    return unique_rows.merge(
        reasons,
        on="segment_id",
        how="left",
    )


def stratified_day_sample(
    available: pd.DataFrame,
    required_count: int,
) -> pd.DataFrame:
    """
    Sample daytime segments across location and density groups.
    """
    if required_count <= 0 or available.empty:
        return available.iloc[0:0].copy()

    groups = list(
        available.groupby(
            ["location", "density_group"],
            dropna=False,
        )
    )

    if not groups:
        return available.sample(
            n=min(required_count, len(available)),
            random_state=RANDOM_SEED,
        )

    selected_parts: list[pd.DataFrame] = []

    # First pass: take approximately the same number from each group.
    base_per_group = max(1, required_count // len(groups))

    for group_number, (_, group) in enumerate(groups):
        sample_size = min(base_per_group, len(group))

        if sample_size > 0:
            selected_parts.append(
                group.sample(
                    n=sample_size,
                    random_state=RANDOM_SEED + group_number,
                )
            )

    if selected_parts:
        sampled = pd.concat(
            selected_parts,
            ignore_index=True,
        ).drop_duplicates(subset=["segment_id"])
    else:
        sampled = available.iloc[0:0].copy()

    # Second pass: fill any remaining positions.
    remaining_needed = required_count - len(sampled)

    if remaining_needed > 0:
        remaining_pool = available[
            ~available["segment_id"].isin(sampled["segment_id"])
        ]

        if not remaining_pool.empty:
            extra = remaining_pool.sample(
                n=min(remaining_needed, len(remaining_pool)),
                random_state=RANDOM_SEED + 100,
            )

            sampled = pd.concat(
                [sampled, extra],
                ignore_index=True,
            )

    # If the first pass selected too many, reduce deterministically.
    if len(sampled) > required_count:
        sampled = sampled.sample(
            n=required_count,
            random_state=RANDOM_SEED + 200,
        )

    return sampled


def main() -> None:
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Catalog not found: {CATALOG_FILE.resolve()}"
        )

    catalog = pd.read_csv(CATALOG_FILE)

    if catalog.empty:
        raise ValueError("The segment catalog is empty.")

    selected_parts: list[pd.DataFrame] = []

    # 1. Preserve the extremely rare rainy condition.
    rainy = catalog[
        catalog["weather"].str.lower() == "rain"
    ]

    add_candidates(
        selected_parts,
        rainy,
        "rare_rain_weather",
    )

    # 2. Preserve all nighttime segments.
    night = catalog[
        catalog["time_of_day"] == "Night"
    ]

    add_candidates(
        selected_parts,
        night,
        "night_condition",
    )

    # 3. Preserve all dawn/dusk segments.
    dawn_dusk = catalog[
        catalog["time_of_day"] == "Dawn/Dusk"
    ]

    add_candidates(
        selected_parts,
        dawn_dusk,
        "dawn_dusk_condition",
    )

    # 4. Include cyclist-rich segments.
    cyclist_rich = catalog.nlargest(
        TOP_CYCLIST_SEGMENTS,
        [
            "cyclist_frame_coverage_percent",
            "average_cyclists_per_frame",
        ],
    )

    add_candidates(
        selected_parts,
        cyclist_rich,
        "cyclist_rich",
    )

    # 5. Include pedestrian-rich segments.
    pedestrian_rich = catalog.nlargest(
        TOP_PEDESTRIAN_SEGMENTS,
        [
            "pedestrian_frame_coverage_percent",
            "average_pedestrians_per_frame",
        ],
    )

    add_candidates(
        selected_parts,
        pedestrian_rich,
        "pedestrian_rich",
    )

    combined = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    combined = combine_selection_reasons(combined)

    # 6. Fill remaining positions using stratified daytime sampling.
    remaining_count = TARGET_CANDIDATES - len(combined)

    already_selected_ids = set(combined["segment_id"])

    day_pool = catalog[
        (catalog["time_of_day"] == "Day")
        & (~catalog["segment_id"].isin(already_selected_ids))
    ].copy()

    daytime_sample = stratified_day_sample(
        day_pool,
        remaining_count,
    )

    if not daytime_sample.empty:
        daytime_sample["selection_reason"] = (
            "day_location_density_balance"
        )

        combined = pd.concat(
            [combined, daytime_sample],
            ignore_index=True,
        )

    # Safety check in case duplicate handling produced fewer than target.
    if len(combined) < TARGET_CANDIDATES:
        remaining_pool = catalog[
            ~catalog["segment_id"].isin(combined["segment_id"])
        ]

        extra_needed = TARGET_CANDIDATES - len(combined)

        extra = remaining_pool.sample(
            n=min(extra_needed, len(remaining_pool)),
            random_state=RANDOM_SEED + 300,
        ).copy()

        extra["selection_reason"] = "final_random_fill"

        combined = pd.concat(
            [combined, extra],
            ignore_index=True,
        )

    combined = combined.drop_duplicates(
        subset=["segment_id"]
    )

    combined["candidate_selection_seed"] = RANDOM_SEED

    combined = combined.sort_values(
        by=[
            "time_of_day",
            "weather",
            "location",
            "density_group",
            "segment_id",
        ]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("WAYMO CANDIDATE SEGMENTS SELECTED")
    print("=" * 60)

    print(f"Target candidates: {TARGET_CANDIDATES}")
    print(f"Selected candidates: {len(combined)}")
    print(f"Random seed: {RANDOM_SEED}")

    print("\nTime-of-day distribution:")
    print(
        combined["time_of_day"].value_counts(
            dropna=False
        )
    )

    print("\nWeather distribution:")
    print(
        combined["weather"].value_counts(
            dropna=False
        )
    )

    print("\nLocation distribution:")
    print(
        combined["location"].value_counts(
            dropna=False
        )
    )

    print("\nDensity distribution:")
    print(
        combined["density_group"].value_counts(
            dropna=False
        )
    )

    print("\nSelection reasons:")
    print(
        combined["selection_reason"]
        .str.get_dummies(sep="|")
        .sum()
        .sort_values(ascending=False)
    )

    print(f"\nCandidate list saved to:\n{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()