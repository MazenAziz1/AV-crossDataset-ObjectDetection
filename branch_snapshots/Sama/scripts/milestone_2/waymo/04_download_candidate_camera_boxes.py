from pathlib import Path
import subprocess
import sys
import shutil

import pandas as pd


CANDIDATES_FILE = Path(
    "data/waymo/selection/candidate_segments.csv"
)

OUTPUT_DIR = Path(
    "data/waymo/raw/validation/camera_box/candidates"
)

REPORT_FILE = Path(
    "data/waymo/selection/candidate_camera_box_download_report.csv"
)

BUCKET_FOLDER = (
    "gs://waymo_open_dataset_v_2_0_1/"
    "validation/camera_box"
)


def main() -> None:
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n{CANDIDATES_FILE.resolve()}"
        )

    candidates = pd.read_csv(CANDIDATES_FILE)

    if "segment_id" not in candidates.columns:
        raise KeyError(
            "candidate_segments.csv does not contain a segment_id column."
        )

    segment_ids = (
        candidates["segment_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if not segment_ids:
        raise ValueError("No candidate segment IDs were found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    gcloud_executable = (
        shutil.which("gcloud.cmd")
        or shutil.which("gcloud")
    )

    if gcloud_executable is None:
        raise FileNotFoundError(
            "Google Cloud CLI could not be found in PATH.\n"
            "Run 'where gcloud' in Command Prompt and confirm that "
            "gcloud.cmd is available."
        )

    print(f"Using gcloud executable: {gcloud_executable}")

    records: list[dict] = []

    print("=" * 65)
    print("DOWNLOADING WAYMO CANDIDATE CAMERA BOX FILES")
    print("=" * 65)
    print(f"Candidate segments: {len(segment_ids)}")
    print(f"Destination: {OUTPUT_DIR.resolve()}\n")

    for number, segment_id in enumerate(segment_ids, start=1):
        filename = f"{segment_id}.parquet"
        local_file = OUTPUT_DIR / filename
        cloud_uri = f"{BUCKET_FOLDER}/{filename}"

        print(f"[{number}/{len(segment_ids)}] {filename}")

        # Allows the script to resume without downloading completed files again.
        if local_file.exists() and local_file.stat().st_size > 0:
            print("  Status: already downloaded — skipped")

            records.append(
                {
                    "segment_id": segment_id,
                    "cloud_uri": cloud_uri,
                    "local_path": str(local_file),
                    "status": "already_exists",
                    "return_code": 0,
                    "error_message": "",
                }
            )
            continue

        if local_file.exists() and local_file.stat().st_size == 0:
            print("  Found incomplete zero-byte file — deleting it")
            local_file.unlink()

        command = [
            "cmd.exe",
            "/c",
            gcloud_executable,
            "storage",
            "cp",
            "--no-clobber",
            cloud_uri,
            str(OUTPUT_DIR),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0 and local_file.exists():
                status = "downloaded"
                error_message = ""
                print("  Status: downloaded")
            else:
                status = "failed"
                error_message = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or "Unknown download error"
                )
                print(f"  Status: FAILED\n  {error_message}")

            records.append(
                {
                    "segment_id": segment_id,
                    "cloud_uri": cloud_uri,
                    "local_path": str(local_file),
                    "status": status,
                    "return_code": result.returncode,
                    "error_message": error_message,
                }
            )

        except FileNotFoundError:
            print(
                "\nERROR: The gcloud command was not found.\n"
                "Confirm that Google Cloud CLI is installed and available "
                "in this Command Prompt."
            )
            sys.exit(1)

        # Save after every attempt so progress is not lost.
        pd.DataFrame(records).to_csv(REPORT_FILE, index=False)

    report = pd.DataFrame(records)

    downloaded_count = int(
        report["status"].isin(
            ["downloaded", "already_exists"]
        ).sum()
    )
    failed_count = int((report["status"] == "failed").sum())

    local_files = list(OUTPUT_DIR.glob("*.parquet"))

    print("\n" + "=" * 65)
    print("CAMERA BOX DOWNLOAD SUMMARY")
    print("=" * 65)
    print(f"Requested segments: {len(segment_ids)}")
    print(f"Successful or existing: {downloaded_count}")
    print(f"Failed: {failed_count}")
    print(f"Local Parquet files: {len(local_files)}")
    print(f"Report saved to: {REPORT_FILE.resolve()}")

    if failed_count > 0:
        print(
            "\nSome downloads failed. Run the same script again; "
            "completed files will be skipped."
        )
        sys.exit(1)

    if len(local_files) != len(segment_ids):
        print(
            "\nWarning: the number of local files does not match "
            "the candidate count."
        )
        sys.exit(1)

    print("\nStep 6 completed successfully.")


if __name__ == "__main__":
    main()