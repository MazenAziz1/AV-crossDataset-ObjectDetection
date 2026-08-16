from pathlib import Path

import pyarrow.parquet as pq


CAMERA_BOX_DIR = Path(
    "data/waymo/raw/validation/camera_box/candidates"
)

OUTPUT_FILE = Path(
    "data/waymo/selection/camera_box_schema.txt"
)


def main() -> None:
    parquet_files = sorted(CAMERA_BOX_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No camera-box Parquet files found in:\n"
            f"{CAMERA_BOX_DIR.resolve()}"
        )

    first_file = parquet_files[0]
    parquet_file = pq.ParquetFile(first_file)

    # Read only a few rows for inspection.
    preview_table = pq.read_table(first_file).slice(0, 5)
    preview_dataframe = preview_table.to_pandas()

    report = [
        "WAYMO CAMERA BOX SCHEMA INSPECTION",
        "=" * 60,
        f"Camera-box directory: {CAMERA_BOX_DIR.resolve()}",
        f"Number of Parquet files: {len(parquet_files)}",
        f"Inspected file: {first_file.name}",
        f"Rows in inspected file: {parquet_file.metadata.num_rows}",
        "",
        "COLUMN NAMES",
        "-" * 60,
        *preview_table.column_names,
        "",
        "PARQUET SCHEMA",
        "-" * 60,
        str(parquet_file.schema_arrow),
        "",
        "FIRST FIVE ROWS",
        "-" * 60,
        preview_dataframe.to_string(index=False),
    ]

    report_text = "\n".join(report)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    print(report_text)

    print(
        f"\nReport saved to:\n"
        f"{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()