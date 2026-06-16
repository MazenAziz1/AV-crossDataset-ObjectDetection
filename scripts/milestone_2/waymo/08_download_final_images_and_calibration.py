from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


FINAL_SEGMENTS_FILE = Path(
    "data/waymo/selection/final_segments.csv"
)

CAMERA_IMAGE_DIR = Path(
    "data/waymo/raw/validation/camera_image/final"
)

CAMERA_CALIBRATION_DIR = Path(
    "data/waymo/raw/validation/camera_calibration/final"
)

REPORT_FILE = Path(
    "data/waymo/selection/"
    "final_components_download_report.csv"
)

BUCKET_ROOT = (
    "gs://waymo_open_dataset_v_2_0_1/validation"
)

COMPONENTS = {
    "camera_image": CAMERA_IMAGE_DIR,
    "camera_calibration": CAMERA_CALIBRATION_DIR,
}


def find_gcloud() -> str:
    """Find Google Cloud CLI on Windows."""
    executable = (
        shutil.which("gcloud.cmd")
        or shutil.which("gcloud")
    )

    if executable is None:
        raise FileNotFoundError(
            "Google Cloud CLI could not be found in PATH.\n"
            "Run 'where gcloud' in Command Prompt."
        )

    return executable


def save_report(records: list[dict]) -> None:
    """Save download progress after every attempt."""
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(records).to_csv(
        REPORT_FILE,
        index=False,
    )


def download_component(
    gcloud_executable: str,
    segment_id: str,
    component_name: str,
    output_directory: Path,
) -> dict:
    """Download one Waymo component for one segment."""
    filename = f"{segment_id}.parquet"

    local_file = output_directory / filename

    cloud_uri = (
        f"{BUCKET_ROOT}/{component_name}/{filename}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if local_file.exists() and local_file.stat().st_size > 0:
        print("    Status: already downloaded — skipped")

        return {
            "segment_id": segment_id,
            "component": component_name,
            "cloud_uri": cloud_uri,
            "local_path": str(local_file),
            "file_size_bytes": local_file.stat().st_size,
            "status": "already_exists",
            "return_code": 0,
            "error_message": "",
        }

    # Remove an incomplete empty file before retrying.
    if local_file.exists() and local_file.stat().st_size == 0:
        print("    Removing incomplete zero-byte file")
        local_file.unlink()

    command = [
        "cmd.exe",
        "/c",
        gcloud_executable,
        "storage",
        "cp",
        "--no-clobber",
        cloud_uri,
        str(output_directory),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        return {
            "segment_id": segment_id,
            "component": component_name,
            "cloud_uri": cloud_uri,
            "local_path": str(local_file),
            "file_size_bytes": 0,
            "status": "failed",
            "return_code": -1,
            "error_message": str(error),
        }

    if (
        result.returncode == 0
        and local_file.exists()
        and local_file.stat().st_size > 0
    ):
        print("    Status: downloaded")

        return {
            "segment_id": segment_id,
            "component": component_name,
            "cloud_uri": cloud_uri,
            "local_path": str(local_file),
            "file_size_bytes": local_file.stat().st_size,
            "status": "downloaded",
            "return_code": result.returncode,
            "error_message": "",
        }

    error_message = (
        result.stderr.strip()
        or result.stdout.strip()
        or "Unknown download error"
    )

    print(f"    Status: FAILED\n    {error_message}")

    return {
        "segment_id": segment_id,
        "component": component_name,
        "cloud_uri": cloud_uri,
        "local_path": str(local_file),
        "file_size_bytes": (
            local_file.stat().st_size
            if local_file.exists()
            else 0
        ),
        "status": "failed",
        "return_code": result.returncode,
        "error_message": error_message,
    }


def main() -> None:
    if not FINAL_SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Final segment file not found:\n"
            f"{FINAL_SEGMENTS_FILE.resolve()}"
        )

    final_segments = pd.read_csv(
        FINAL_SEGMENTS_FILE
    )

    if "segment_id" not in final_segments.columns:
        raise KeyError(
            "final_segments.csv does not contain "
            "a segment_id column."
        )

    segment_ids = (
        final_segments["segment_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(segment_ids) != 25:
        raise ValueError(
            "Expected exactly 25 frozen final segments, "
            f"but found {len(segment_ids)}."
        )

    gcloud_executable = find_gcloud()

    print(
        f"Using gcloud executable:\n"
        f"{gcloud_executable}"
    )

    records: list[dict] = []

    print("\n" + "=" * 70)
    print("DOWNLOADING FINAL WAYMO COMPONENTS")
    print("=" * 70)

    print(f"Final segments: {len(segment_ids)}")
    print("Components:")
    print("  - camera_image")
    print("  - camera_calibration")

    for segment_number, segment_id in enumerate(
        segment_ids,
        start=1,
    ):
        print(
            f"\n[{segment_number}/{len(segment_ids)}] "
            f"Segment: {segment_id}"
        )

        for component_name, output_directory in COMPONENTS.items():
            print(f"  Component: {component_name}")

            record = download_component(
                gcloud_executable=gcloud_executable,
                segment_id=segment_id,
                component_name=component_name,
                output_directory=output_directory,
            )

            records.append(record)
            save_report(records)

    report = pd.DataFrame(records)

    successful_statuses = {
        "downloaded",
        "already_exists",
    }

    successful_count = int(
        report["status"]
        .isin(successful_statuses)
        .sum()
    )

    failed_count = int(
        (report["status"] == "failed").sum()
    )

    expected_component_files = (
        len(segment_ids) * len(COMPONENTS)
    )

    image_files = list(
        CAMERA_IMAGE_DIR.glob("*.parquet")
    )

    calibration_files = list(
        CAMERA_CALIBRATION_DIR.glob("*.parquet")
    )

    image_size_bytes = sum(
        file.stat().st_size
        for file in image_files
    )

    calibration_size_bytes = sum(
        file.stat().st_size
        for file in calibration_files
    )

    print("\n" + "=" * 70)
    print("FINAL COMPONENT DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        f"Expected component files: "
        f"{expected_component_files}"
    )

    print(
        f"Successful or existing: "
        f"{successful_count}"
    )

    print(f"Failed: {failed_count}")

    print(
        f"Camera-image files: "
        f"{len(image_files)}"
    )

    print(
        f"Camera-calibration files: "
        f"{len(calibration_files)}"
    )

    print(
        "Camera-image size: "
        f"{image_size_bytes / (1024 ** 3):.2f} GB"
    )

    print(
        "Camera-calibration size: "
        f"{calibration_size_bytes / (1024 ** 2):.2f} MB"
    )

    print(
        f"Report saved to:\n"
        f"{REPORT_FILE.resolve()}"
    )

    if failed_count > 0:
        print(
            "\nSome downloads failed. Run this same script "
            "again; completed files will be skipped."
        )
        sys.exit(1)

    if len(image_files) != len(segment_ids):
        print(
            "\nERROR: camera-image file count does not "
            "match final segment count."
        )
        sys.exit(1)

    if len(calibration_files) != len(segment_ids):
        print(
            "\nERROR: camera-calibration file count does not "
            "match final segment count."
        )
        sys.exit(1)

    print("\nStep 10 completed successfully.")


if __name__ == "__main__":
    main()