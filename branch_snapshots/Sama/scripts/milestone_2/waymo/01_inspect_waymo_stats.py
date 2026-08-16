from pathlib import Path

import pyarrow.parquet as pq


STATS_DIR = Path("data/waymo/raw/validation/stats")
OUTPUT_FILE = Path("data/waymo/selection/stats_schema.txt")


def main() -> None:
    parquet_files = sorted(STATS_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in: {STATS_DIR.resolve()}"
        )

    first_file = parquet_files[0]
    parquet_metadata = pq.ParquetFile(first_file)

    # Read only the first few rows for inspection.
    preview_table = pq.read_table(first_file).slice(0, 3)
    preview_dataframe = preview_table.to_pandas()

    report = [
        "WAYMO VALIDATION STATS INSPECTION",
        "=" * 50,
        f"Stats directory: {STATS_DIR.resolve()}",
        f"Number of Parquet files: {len(parquet_files)}",
        f"Inspected file: {first_file.name}",
        f"Rows in inspected file: {parquet_metadata.metadata.num_rows}",
        "",
        "COLUMN NAMES",
        "-" * 50,
        *preview_table.column_names,
        "",
        "PARQUET SCHEMA",
        "-" * 50,
        str(parquet_metadata.schema_arrow),
        "",
        "FIRST THREE ROWS",
        "-" * 50,
        preview_dataframe.to_string(index=False),
    ]

    report_text = "\n".join(report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nReport saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()