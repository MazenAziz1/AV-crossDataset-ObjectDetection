import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def main():
    print("=" * 79)
    print("Milestone 6 - Phase 1: Waymo external handoff validation")
    print("=" * 79)

    project_root = Path(__file__).resolve().parents[2]
    m3_root = project_root / "data" / "processed" / "milestone_3"

    coco_gt_path = m3_root / "annotations" / "coco" / "waymo_external.json"
    ignore_path = m3_root / "annotations" / "ignore_regions" / "waymo_external_ignore.json"
    images_dir = m3_root / "images" / "waymo" / "external"
    yolo_labels_dir = m3_root / "annotations" / "yolo" / "yolo" / "waymo" / "external"

    out_dir = project_root / "outputs" / "milestone_6" / "handoff_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = {}
    issues = []

    def add_issue(severity, check, message):
        issues.append({"severity": severity, "check": check, "message": message})

    # --- 1. COCO GT present and well-formed ---
    if not coco_gt_path.exists():
        add_issue("ERROR", "coco_gt", f"Missing COCO GT: {coco_gt_path}")
    else:
        with open(coco_gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        images = gt.get("images", [])
        annotations = gt.get("annotations", [])
        categories = gt.get("categories", [])
        checks["coco_num_images"] = len(images)
        checks["coco_num_annotations"] = len(annotations)

        cat_map = {c["id"]: c["name"] for c in categories}
        expected_cats = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}
        if cat_map != expected_cats:
            add_issue("ERROR", "coco_categories", f"Categories mismatch: {cat_map}")
        else:
            checks["coco_categories"] = "PASSED"

    # --- 2. Image files present and paired with COCO records ---
    png_files = sorted(images_dir.glob("*.png")) if images_dir.exists() else []
    checks["png_image_count"] = len(png_files)

    if not images_dir.exists():
        add_issue("ERROR", "images_dir", f"Missing images dir: {images_dir}")

    coco_names = set()
    if coco_gt_path.exists():
        coco_names = {img["file_name"] for img in images}

    disk_names = {p.name for p in png_files}

    missing_on_disk = coco_names - disk_names
    missing_in_coco = disk_names - coco_names
    if missing_on_disk:
        add_issue("ERROR", "image_pairing", f"{len(missing_on_disk)} COCO images missing on disk")
    if missing_in_coco:
        add_issue("ERROR", "image_pairing", f"{len(missing_in_coco)} disk images missing from COCO")
    if not missing_on_disk and not missing_in_coco:
        checks["image_coco_pairing"] = "PASSED"

    # --- 3. Image dimensions (sample) ---
    dims_ok = True
    sample = png_files[:5]
    for p in sample:
        with Image.open(p) as im:
            w, h = im.size
            if (w, h) != (640, 640):
                add_issue("ERROR", "image_dims", f"{p.name} is {w}x{h}, expected 640x640")
                dims_ok = False
    if dims_ok and sample:
        checks["image_dimensions_sample"] = "PASSED (640x640)"

    # --- 4. YOLO label mirror present and valid ---
    yolo_files = sorted(yolo_labels_dir.glob("*.txt")) if yolo_labels_dir.exists() else []
    checks["yolo_label_count"] = len(yolo_files)
    if not yolo_labels_dir.exists():
        add_issue("WARNING", "yolo_labels", f"No YOLO label mirror at {yolo_labels_dir}")

    yolo_names = {p.stem for p in yolo_files}
    disk_stems = {p.stem for p in png_files}
    if yolo_names != disk_stems:
        add_issue("ERROR", "yolo_pairing",
                  f"YOLO labels do not match image stems: "
                  f"labels-only={len(yolo_names - disk_stems)}, images-only={len(disk_stems - yolo_names)}")
    else:
        checks["yolo_pairing"] = "PASSED"

    invalid_bbox_lines = 0
    invalid_class_lines = 0
    valid_class_ids = {0, 1, 2}
    for p in yolo_files[:50]:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                add_issue("ERROR", "yolo_format", f"{p.name}: bad line '{line}'")
                continue
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
            if cls not in valid_class_ids:
                invalid_class_lines += 1
            if any(v < 0.0 or v > 1.0 for v in vals):
                invalid_bbox_lines += 1

    if invalid_bbox_lines == 0 and invalid_class_lines == 0:
        checks["yolo_bbox_and_class_sample"] = "PASSED"
    else:
        add_issue("ERROR", "yolo_values",
                  f"invalid bbox lines={invalid_bbox_lines}, invalid class lines={invalid_class_lines} (sample of 50)")

    # --- 5. Ignore regions sidecar ---
    if not ignore_path.exists():
        add_issue("WARNING", "ignore_regions", f"Missing ignore regions sidecar: {ignore_path}")
    else:
        with open(ignore_path, encoding="utf-8") as f:
            ignore_data = json.load(f)
        ignore_classes = ignore_data.get("info", {}).get("classes", [])
        checks["ignore_classes"] = ignore_classes

    # --- 6. Expected counts vs. frozen subset summary ---
    expected_images = 996
    if checks.get("coco_num_images") != expected_images:
        add_issue("ERROR", "image_count", f"Expected {expected_images} Waymo images, got {checks.get('coco_num_images')}")

    # --- Assemble result ---
    errors = [i for i in issues if i["severity"] == "ERROR"]
    warnings = [i for i in issues if i["severity"] == "WARNING"]
    status = "PASSED" if not errors else "FAILED"

    summary = {
        "milestone": 6,
        "phase": 1,
        "purpose": "Validate Waymo external subset before external evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "errors_count": len(errors),
        "warnings_count": len(warnings),
        "checks": checks,
        "issues": issues,
        "class_mapping": {"coco_category_id": {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"},
                          "yolo_class_id": {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}},
    }

    json_path = out_dir / "waymo_handoff_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Milestone 6 - Waymo External Handoff Validation",
        "",
        f"- Status: **{status}**",
        f"- Timestamp: {summary['timestamp']}",
        f"- Errors: {len(errors)}  Warnings: {len(warnings)}",
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    md_lines.append("## Class mapping (frozen in Milestone 3)")
    md_lines.append("")
    md_lines.append("| COCO category id | YOLO class id | Name |")
    md_lines.append("|---|---|---|")
    md_lines.append("| 1 | 0 | Vehicle |")
    md_lines.append("| 2 | 1 | Pedestrian |")
    md_lines.append("| 3 | 2 | Cyclist |")
    md_lines.append("")
    if issues:
        md_lines.append("## Issues")
        md_lines.append("")
        for i in issues:
            md_lines.append(f"- [{i['severity']}] {i['check']}: {i['message']}")
        md_lines.append("")
    md_path = out_dir / "waymo_handoff_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Status: {status}")
    print(f"  COCO images: {checks.get('coco_num_images')}")
    print(f"  COCO annotations: {checks.get('coco_num_annotations')}")
    print(f"  PNG images: {checks.get('png_image_count')}")
    print(f"  YOLO labels: {checks.get('yolo_label_count')}")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")

    if status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
