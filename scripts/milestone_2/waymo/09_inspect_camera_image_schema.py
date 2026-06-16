from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


CAMERA_IMAGE_DIR = Path(
    "data/waymo/raw/validation/camera_image/final"
)

OUTPUT_FILE = Path(
    "data/waymo/selection/camera_image_schema.txt"
)


def simplify_value(value):
    """
    Make large binary/image values readable in the preview.
    """
    if isinstance(value, (bytes, bytearray)):
        return f"<binary length={len(value)} bytes>"

    return value


def main() -> None:
    parquet_files = sorted(CAMERA_IMAGE_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No camera-image Parquet files found in:\n"
            f"{CAMERA_IMAGE_DIR.resolve()}"
        )

    first_file = parquet_files[0]
    parquet_file = pq.ParquetFile(first_file)

    # Read a few rows for preview.
    preview_table = pq.read_table(first_file).slice(0, 3)
    preview_dataframe = preview_table.to_pandas()

    # Make preview readable if a binary image column exists.
    preview_dataframe = preview_dataframe.map(simplify_value)

    report = [
        "WAYMO CAMERA IMAGE SCHEMA INSPECTION",
        "=" * 60,
        f"Camera-image directory: {CAMERA_IMAGE_DIR.resolve()}",
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
        "FIRST THREE ROWS",
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
    print(f"\nReport saved to:\n{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()