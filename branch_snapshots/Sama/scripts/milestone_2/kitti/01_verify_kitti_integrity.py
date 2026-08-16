from collections import Counter
from pathlib import Path
import csv
import json
import math

from PIL import Image
from tqdm import tqdm


KITTI_ROOT = Path("data/kitti/raw/training")

IMAGE_DIR = KITTI_ROOT / "image_2"
LABEL_DIR = KITTI_ROOT / "label_2"
CALIB_DIR = KITTI_ROOT / "calib"
DEVKIT_DIR = Path("data/kitti/raw/devkit_object")

OUTPUT_DIR = Path("data/kitti/statistics")

REPORT_FILE = OUTPUT_DIR / "dataset_integrity_report.json"
FILENAME_ISSUES_FILE = OUTPUT_DIR / "filename_mismatches.csv"
IMAGE_ISSUES_FILE = OUTPUT_DIR / "image_issues.csv"
LABEL_ISSUES_FILE = OUTPUT_DIR / "label_issues.csv"
CALIBRATION_ISSUES_FILE = OUTPUT_DIR / "calibration_issues.csv"
BOX_WARNINGS_FILE = OUTPUT_DIR / "box_warnings.csv"


EXPECTED_IMAGE_COUNT = 7481

KNOWN_KITTI_CLASSES = {
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
    "DontCare",
}


def write_csv(
    output_file: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def is_finite(*values: float) -> bool:
    return all(
        math.isfinite(value)
        for value in values
    )


def inspect_image(
    image_file: Path,
) -> tuple[int, int, str | None]:
    try:
        with Image.open(image_file) as image:
            width, height = image.size
            image.verify()

        return int(width), int(height), None

    except Exception as error:
        return 0, 0, str(error)


def inspect_calibration(
    calibration_file: Path,
) -> list[str]:
    issues: list[str] = []

    try:
        content = calibration_file.read_text(
            encoding="utf-8",
        ).strip()

    except Exception as error:
        return [f"Could not read calibration file: {error}"]

    if not content:
        return ["Calibration file is empty"]

    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    calibration_keys = {
        line.split(":", 1)[0]
        for line in lines
        if ":" in line
    }

    if "P2" not in calibration_keys:
        issues.append(
            "Required P2 camera projection matrix is missing"
        )

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if ":" not in line:
            issues.append(
                f"Line {line_number} has no key separator"
            )
            continue

        key, values_text = line.split(":", 1)
        values = values_text.split()

        if not values:
            issues.append(
                f"Calibration entry {key} has no values"
            )
            continue

        try:
            numeric_values = [
                float(value)
                for value in values
            ]

        except ValueError:
            issues.append(
                f"Calibration entry {key} contains non-numeric values"
            )
            continue

        if not is_finite(*numeric_values):
            issues.append(
                f"Calibration entry {key} contains non-finite values"
            )

    return issues


def main() -> None:
    required_directories = [
        IMAGE_DIR,
        LABEL_DIR,
        CALIB_DIR,
        DEVKIT_DIR,
    ]

    for directory in required_directories:
        if not directory.exists():
            raise FileNotFoundError(
                f"Required directory not found:\n"
                f"{directory.resolve()}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        IMAGE_DIR.glob("*.png")
    )

    label_files = sorted(
        LABEL_DIR.glob("*.txt")
    )

    calibration_files = sorted(
        CALIB_DIR.glob("*.txt")
    )

    image_ids = {
        file.stem
        for file in image_files
    }

    label_ids = {
        file.stem
        for file in label_files
    }

    calibration_ids = {
        file.stem
        for file in calibration_files
    }

    all_ids = sorted(
        image_ids
        | label_ids
        | calibration_ids
    )

    filename_issues: list[dict] = []

    for sample_id in all_ids:
        has_image = sample_id in image_ids
        has_label = sample_id in label_ids
        has_calibration = (
            sample_id in calibration_ids
        )

        if not (
            has_image
            and has_label
            and has_calibration
        ):
            filename_issues.append(
                {
                    "sample_id": sample_id,
                    "has_image": has_image,
                    "has_label": has_label,
                    "has_calibration": (
                        has_calibration
                    ),
                }
            )

    image_issues: list[dict] = []
    label_issues: list[dict] = []
    calibration_issues: list[dict] = []
    box_warnings: list[dict] = []

    image_dimensions: dict[
        str,
        tuple[int, int],
    ] = {}

    print("=" * 72)
    print("KITTI DATASET INTEGRITY VERIFICATION")
    print("=" * 72)

    print("\nChecking images...")

    for image_file in tqdm(
        image_files,
        unit="image",
    ):
        width, height, error = inspect_image(
            image_file
        )

        image_dimensions[image_file.stem] = (
            width,
            height,
        )

        if error is not None:
            image_issues.append(
                {
                    "sample_id": image_file.stem,
                    "image_path": str(image_file),
                    "issue": error,
                }
            )

    class_counts: Counter = Counter()
    total_annotation_rows = 0
    valid_annotation_rows = 0
    empty_label_files = 0
    invalid_box_count = 0
    out_of_image_box_count = 0
    unknown_class_count = 0

    print("\nChecking annotations...")

    for label_file in tqdm(
        label_files,
        unit="label",
    ):
        sample_id = label_file.stem

        try:
            lines = label_file.read_text(
                encoding="utf-8",
            ).splitlines()

        except Exception as error:
            label_issues.append(
                {
                    "sample_id": sample_id,
                    "line_number": "",
                    "issue_type": (
                        "unreadable_label_file"
                    ),
                    "details": str(error),
                    "raw_line": "",
                }
            )
            continue

        non_empty_lines = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        if not non_empty_lines:
            empty_label_files += 1
            continue

        image_width, image_height = (
            image_dimensions.get(
                sample_id,
                (0, 0),
            )
        )

        for line_number, line in enumerate(
            non_empty_lines,
            start=1,
        ):
            total_annotation_rows += 1

            fields = line.split()

            # KITTI ground-truth rows normally contain
            # 15 fields. Detection output may contain
            # an optional 16th score field.
            if len(fields) not in {15, 16}:
                label_issues.append(
                    {
                        "sample_id": sample_id,
                        "line_number": line_number,
                        "issue_type": (
                            "invalid_field_count"
                        ),
                        "details": (
                            f"Expected 15 or 16 "
                            f"fields, found {len(fields)}"
                        ),
                        "raw_line": line,
                    }
                )
                continue

            class_name = fields[0]
            class_counts[class_name] += 1

            if class_name not in KNOWN_KITTI_CLASSES:
                unknown_class_count += 1

                label_issues.append(
                    {
                        "sample_id": sample_id,
                        "line_number": line_number,
                        "issue_type": (
                            "unknown_class"
                        ),
                        "details": class_name,
                        "raw_line": line,
                    }
                )

            try:
                truncation = float(fields[1])
                occlusion = int(fields[2])
                alpha = float(fields[3])

                xmin = float(fields[4])
                ymin = float(fields[5])
                xmax = float(fields[6])
                ymax = float(fields[7])

                dimensions = [
                    float(value)
                    for value in fields[8:11]
                ]

                location = [
                    float(value)
                    for value in fields[11:14]
                ]

                rotation_y = float(fields[14])

                numeric_values = [
                    truncation,
                    float(occlusion),
                    alpha,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    *dimensions,
                    *location,
                    rotation_y,
                ]

                if len(fields) == 16:
                    numeric_values.append(
                        float(fields[15])
                    )

            except ValueError as error:
                label_issues.append(
                    {
                        "sample_id": sample_id,
                        "line_number": line_number,
                        "issue_type": (
                            "non_numeric_field"
                        ),
                        "details": str(error),
                        "raw_line": line,
                    }
                )
                continue

            if not is_finite(*numeric_values):
                label_issues.append(
                    {
                        "sample_id": sample_id,
                        "line_number": line_number,
                        "issue_type": (
                            "non_finite_value"
                        ),
                        "details": (
                            "One or more numeric "
                            "values are NaN or infinite"
                        ),
                        "raw_line": line,
                    }
                )
                continue

            # DontCare annotations use placeholder values such as
            # truncation=-1 and occlusion=-1. These are valid for DontCare.
            if class_name != "DontCare":
                if not 0.0 <= truncation <= 1.0:
                    label_issues.append(
                        {
                            "sample_id": sample_id,
                            "line_number": line_number,
                            "issue_type": "invalid_truncation",
                            "details": f"Truncation={truncation}",
                            "raw_line": line,
                        }
                    )

                if occlusion not in {0, 1, 2, 3}:
                    label_issues.append(
                        {
                            "sample_id": sample_id,
                            "line_number": line_number,
                            "issue_type": "invalid_occlusion",
                            "details": f"Occlusion={occlusion}",
                            "raw_line": line,
                        }
                    )

            if xmax <= xmin or ymax <= ymin:
                invalid_box_count += 1

                label_issues.append(
                    {
                        "sample_id": sample_id,
                        "line_number": line_number,
                        "issue_type": (
                            "invalid_bounding_box"
                        ),
                        "details": (
                            f"xmin={xmin}, ymin={ymin}, "
                            f"xmax={xmax}, ymax={ymax}"
                        ),
                        "raw_line": line,
                    }
                )
                continue

            valid_annotation_rows += 1

            # Out-of-image coordinates are logged as
            # warnings rather than fatal errors because
            # truncated objects may touch image borders.
            if (
                image_width > 0
                and image_height > 0
                and (
                    xmin < 0
                    or ymin < 0
                    or xmax > image_width
                    or ymax > image_height
                )
            ):
                out_of_image_box_count += 1

                box_warnings.append(
                    {
                        "sample_id": sample_id,
                        "line_number": line_number,
                        "class_name": class_name,
                        "image_width": image_width,
                        "image_height": image_height,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "warning": (
                            "Bounding box extends "
                            "outside image dimensions"
                        ),
                    }
                )

    print("\nChecking calibration files...")

    for calibration_file in tqdm(
        calibration_files,
        unit="calibration",
    ):
        issues = inspect_calibration(
            calibration_file
        )

        for issue in issues:
            calibration_issues.append(
                {
                    "sample_id": (
                        calibration_file.stem
                    ),
                    "calibration_path": str(
                        calibration_file
                    ),
                    "issue": issue,
                }
            )

    devkit_files = [
        file
        for file in DEVKIT_DIR.rglob("*")
        if file.is_file()
    ]

    fatal_issue_count = (
        len(filename_issues)
        + len(image_issues)
        + len(label_issues)
        + len(calibration_issues)
    )

    integrity_passed = fatal_issue_count == 0

    report = {
        "dataset": "KITTI Object Detection",
        "source_subset": (
            "official labeled training set"
        ),
        "official_testing_set_used": False,
        "expected_training_images": (
            EXPECTED_IMAGE_COUNT
        ),
        "image_files_found": len(image_files),
        "label_files_found": len(label_files),
        "calibration_files_found": (
            len(calibration_files)
        ),
        "devkit_files_found": len(devkit_files),
        "matching_sample_ids": len(
            image_ids
            & label_ids
            & calibration_ids
        ),
        "filename_mismatch_count": len(
            filename_issues
        ),
        "corrupted_or_unreadable_images": len(
            image_issues
        ),
        "empty_label_files": (
            empty_label_files
        ),
        "total_annotation_rows": (
            total_annotation_rows
        ),
        "valid_annotation_rows": (
            valid_annotation_rows
        ),
        "label_issue_count": len(
            label_issues
        ),
        "invalid_bounding_boxes": (
            invalid_box_count
        ),
        "out_of_image_box_warnings": (
            out_of_image_box_count
        ),
        "unknown_class_count": (
            unknown_class_count
        ),
        "calibration_issue_count": len(
            calibration_issues
        ),
        "original_class_counts": dict(
            sorted(class_counts.items())
        ),
        "integrity_passed": (
            integrity_passed
        ),
        "fatal_issue_count": (
            fatal_issue_count
        ),
        "notes": [
            (
                "Out-of-image bounding boxes are "
                "reported as warnings, not fatal errors."
            ),
            (
                "The official KITTI testing split is "
                "not included because public labels "
                "are unavailable."
            ),
        ],
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(
        FILENAME_ISSUES_FILE,
        filename_issues,
        [
            "sample_id",
            "has_image",
            "has_label",
            "has_calibration",
        ],
    )

    write_csv(
        IMAGE_ISSUES_FILE,
        image_issues,
        [
            "sample_id",
            "image_path",
            "issue",
        ],
    )

    write_csv(
        LABEL_ISSUES_FILE,
        label_issues,
        [
            "sample_id",
            "line_number",
            "issue_type",
            "details",
            "raw_line",
        ],
    )

    write_csv(
        CALIBRATION_ISSUES_FILE,
        calibration_issues,
        [
            "sample_id",
            "calibration_path",
            "issue",
        ],
    )

    write_csv(
        BOX_WARNINGS_FILE,
        box_warnings,
        [
            "sample_id",
            "line_number",
            "class_name",
            "image_width",
            "image_height",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "warning",
        ],
    )

    print("\n" + "=" * 72)
    print("KITTI INTEGRITY VERIFICATION COMPLETE")
    print("=" * 72)

    print(
        f"Images found: "
        f"{len(image_files)}"
    )

    print(
        f"Labels found: "
        f"{len(label_files)}"
    )

    print(
        f"Calibration files found: "
        f"{len(calibration_files)}"
    )

    print(
        f"Matching sample IDs: "
        f"{len(image_ids & label_ids & calibration_ids)}"
    )

    print(
        f"Total annotation rows: "
        f"{total_annotation_rows}"
    )

    print(
        f"Valid annotation rows: "
        f"{valid_annotation_rows}"
    )

    print(
        f"Empty label files: "
        f"{empty_label_files}"
    )

    print(
        f"Filename mismatches: "
        f"{len(filename_issues)}"
    )

    print(
        f"Unreadable images: "
        f"{len(image_issues)}"
    )

    print(
        f"Label issues: "
        f"{len(label_issues)}"
    )

    print(
        f"Invalid bounding boxes: "
        f"{invalid_box_count}"
    )

    print(
        f"Out-of-image box warnings: "
        f"{out_of_image_box_count}"
    )

    print(
        f"Calibration issues: "
        f"{len(calibration_issues)}"
    )

    print("\nOriginal class distribution:")

    for class_name, count in sorted(
        class_counts.items()
    ):
        print(
            f"  {class_name}: {count}"
        )

    print(
        "\nIntegrity status: "
        + (
            "PASSED"
            if integrity_passed
            else "FAILED — inspect issue CSV files"
        )
    )

    print(
        f"\nReport saved to:\n"
        f"{REPORT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()