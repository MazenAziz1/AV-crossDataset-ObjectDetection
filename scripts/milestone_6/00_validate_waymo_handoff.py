from pathlib import Path
import json
from datetime import datetime
from PIL import Image

PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

IMAGE_DIR_CANDIDATES = [
    PROJECT / "data" / "processed" / "milestone_3" / "images" / "waymo" / "external",
    PROJECT / "data" / "processed" / "milestone_3" / "images" / "waymo",
]

LABEL_DIR_CANDIDATES = [
    PROJECT / "data" / "processed" / "milestone_3" / "labels" / "waymo" / "external",
    PROJECT / "data" / "processed" / "milestone_3" / "labels" / "waymo",
]

OUTPUT_DIR = PROJECT / "outputs" / "milestone_6" / "handoff_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_CLASS_IDS = {0, 1, 2}
CLASS_NAMES = {
    0: "Vehicle",
    1: "Pedestrian",
    2: "Cyclist",
}


def find_existing_dir(candidates):
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def collect_images(image_dir):
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_labels(label_dir):
    return sorted(
        p for p in label_dir.rglob("*.txt")
        if p.is_file()
    )


def parse_yolo_label(label_path):
    issues = []
    annotations = []

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            issues.append({
                "line": line_number,
                "issue": "Expected 5 YOLO fields: class_id x_center y_center width height",
                "content": line,
            })
            continue

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            issues.append({
                "line": line_number,
                "issue": "Non-numeric YOLO values",
                "content": line,
            })
            continue

        if class_id not in EXPECTED_CLASS_IDS:
            issues.append({
                "line": line_number,
                "issue": f"Unexpected class id {class_id}",
                "content": line,
            })

        bbox_values = [x_center, y_center, width, height]

        if not all(0.0 <= value <= 1.0 for value in bbox_values):
            issues.append({
                "line": line_number,
                "issue": "YOLO bbox values should be normalized between 0 and 1",
                "content": line,
            })

        if width <= 0 or height <= 0:
            issues.append({
                "line": line_number,
                "issue": "YOLO width and height must be positive",
                "content": line,
            })

        annotations.append({
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        })

    return annotations, issues


def main():
    print("=" * 100)
    print("STEP 1/10 - Validate Waymo external handoff")
    print("=" * 100)

    image_dir = find_existing_dir(IMAGE_DIR_CANDIDATES)
    label_dir = find_existing_dir(LABEL_DIR_CANDIDATES)

    errors = []
    warnings = []

    if image_dir is None:
        errors.append("Waymo image directory was not found.")
        images = []
    else:
        images = collect_images(image_dir)

    if label_dir is None:
        errors.append("Waymo label directory was not found.")
        labels = []
    else:
        labels = collect_labels(label_dir)

    print("Image dir:", image_dir if image_dir else "NOT FOUND")
    print("Label dir:", label_dir if label_dir else "NOT FOUND")
    print("Images found:", len(images))
    print("Labels found:", len(labels))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)

    if images and labels:
        print("Matched pairs:", len(image_stems & label_stems))
        print("Images without labels:", len(images_without_labels))
        print("Labels without images:", len(labels_without_images))

        if images_without_labels:
            warnings.append(f"{len(images_without_labels)} images do not have matching labels.")

        if labels_without_images:
            warnings.append(f"{len(labels_without_images)} labels do not have matching images.")

    class_counts = {str(k): 0 for k in sorted(EXPECTED_CLASS_IDS)}
    total_annotations = 0
    label_files_with_issues = {}
    empty_label_files = []

    for label_path in labels:
        annotations, issues = parse_yolo_label(label_path)

        if not annotations:
            empty_label_files.append(str(label_path.relative_to(PROJECT)))

        for ann in annotations:
            class_counts[str(ann["class_id"])] = class_counts.get(str(ann["class_id"]), 0) + 1

        total_annotations += len(annotations)

        if issues:
            label_files_with_issues[str(label_path.relative_to(PROJECT))] = issues[:10]

    if label_files_with_issues:
        errors.append(f"{len(label_files_with_issues)} label files contain format/class/bbox issues.")

    sample_image_info = []

    for image_path in images[:10]:
        try:
            with Image.open(image_path) as img:
                sample_image_info.append({
                    "file": str(image_path.relative_to(PROJECT)),
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                })
        except Exception as exc:
            warnings.append(f"Could not open sample image {image_path}: {exc}")

    status = "FAILED" if errors else "PASSED"

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "project": str(PROJECT),
        "image_dir": str(image_dir) if image_dir else None,
        "label_dir": str(label_dir) if label_dir else None,
        "num_images": len(images),
        "num_labels": len(labels),
        "num_matched_pairs": len(image_stems & label_stems) if images and labels else 0,
        "num_images_without_labels": len(images_without_labels),
        "num_labels_without_images": len(labels_without_images),
        "total_annotations": total_annotations,
        "class_mapping": CLASS_NAMES,
        "class_counts": class_counts,
        "num_empty_label_files": len(empty_label_files),
        "sample_empty_label_files": empty_label_files[:20],
        "label_files_with_issues": label_files_with_issues,
        "sample_image_info": sample_image_info,
        "warnings": warnings,
        "errors": errors,
    }

    json_path = OUTPUT_DIR / "waymo_handoff_summary.json"
    md_path = OUTPUT_DIR / "waymo_handoff_summary.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 6 - Waymo Handoff Validation")
    md.append("")
    md.append(f"Created at: `{summary['created_at']}`")
    md.append("")
    md.append(f"Status: **{status}**")
    md.append("")
    md.append("## Directories")
    md.append("")
    md.append(f"- Image directory: `{summary['image_dir']}`")
    md.append(f"- Label directory: `{summary['label_dir']}`")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append(f"- Images: `{summary['num_images']}`")
    md.append(f"- Labels: `{summary['num_labels']}`")
    md.append(f"- Matched image-label pairs: `{summary['num_matched_pairs']}`")
    md.append(f"- Total annotations: `{summary['total_annotations']}`")
    md.append("")
    md.append("## Class Mapping")
    md.append("")
    for class_id, name in CLASS_NAMES.items():
        md.append(f"- `{class_id}` = `{name}` | count = `{class_counts[str(class_id)]}`")
    md.append("")
    md.append("## Warnings")
    md.append("")
    if warnings:
        for warning in warnings:
            md.append(f"- {warning}")
    else:
        md.append("- None")
    md.append("")
    md.append("## Errors")
    md.append("")
    if errors:
        for error in errors:
            md.append(f"- {error}")
    else:
        md.append("- None")
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")

    print()
    print("Class counts:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {class_id} {name}: {class_counts[str(class_id)]}")

    print()
    print("Output JSON:", json_path)
    print("Output MD:", md_path)

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print("WARNING:", warning)

    if errors:
        print()
        print("Errors:")
        for error in errors:
            print("ERROR:", error)
        print()
        print("STEP 1/10 FAILED ❌")
        raise SystemExit(1)

    print()
    print("STEP 1/10 COMPLETE ✅")
    print("Waymo handoff is valid for Milestone 6 external validation.")
    print("=" * 100)


if __name__ == "__main__":
    main()