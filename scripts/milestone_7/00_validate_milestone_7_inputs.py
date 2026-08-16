from pathlib import Path
from datetime import datetime
import json
import pandas as pd


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

OUTPUT_DIR = PROJECT / "outputs" / "milestone_7" / "data_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_JSON = OUTPUT_DIR / "milestone_7_input_audit.json"
AUDIT_MD = OUTPUT_DIR / "MILESTONE_7_INPUT_AUDIT.md"
PREDICTION_INVENTORY_CSV = OUTPUT_DIR / "prediction_inventory.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DETECTORS = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]

CLASS_NAMES = {
    0: "Vehicle",
    1: "Pedestrian",
    2: "Cyclist",
}

EXPECTED_COUNTS = {
    "kitti_val_images": 1496,
    "kitti_val_labels": 1496,
    "waymo_external_images": 996,
    "waymo_external_labels": 996,
}

PATHS = {
    # KITTI validation data from Milestone 3/5
    "kitti_val_images": PROJECT / "data" / "processed" / "milestone_3" / "images" / "kitti" / "val",
    "kitti_val_labels": PROJECT / "data" / "processed" / "milestone_3" / "labels" / "kitti" / "val",

    # Waymo external data from Milestone 3/6
    "waymo_external_images": PROJECT / "data" / "processed" / "milestone_3" / "images" / "waymo" / "external",
    "waymo_external_labels": PROJECT / "data" / "processed" / "milestone_3" / "labels" / "waymo" / "external",

    # Milestone 5 outputs
    "m5_kitti_summary_csv": PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "tables" / "comparison_summary_full.csv",
    "m5_kitti_metrics_dir": PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "metrics",
    "m5_kitti_predictions_dir": PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "predictions",

    # Milestone 6 outputs
    "m6_waymo_summary_csv": PROJECT / "outputs" / "milestone_6" / "waymo_external_validation" / "tables" / "waymo_external_summary.csv",
    "m6_waymo_metrics_dir": PROJECT / "outputs" / "milestone_6" / "waymo_external_validation" / "metrics",
    "m6_waymo_predictions_dir": PROJECT / "outputs" / "milestone_6" / "waymo_external_validation" / "predictions",
    "m6_generalization_ratio_csv": PROJECT / "outputs" / "milestone_6" / "generalization_analysis" / "tables" / "generalization_ratio_table.csv",
    "m6_generalization_summary_json": PROJECT / "outputs" / "milestone_6" / "generalization_analysis" / "generalization_analysis_summary.json",

    # Checkpoint registry
    "locked_checkpoint_registry": PROJECT / "outputs" / "milestone_4" / "locked_final_checkpoints" / "final_checkpoint_registry.json",
}


def collect_files(directory: Path, extensions=None):
    if not directory.exists():
        return []

    if extensions is None:
        return sorted(p for p in directory.rglob("*") if p.is_file())

    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def collect_images(directory: Path):
    return collect_files(directory, IMAGE_EXTENSIONS)


def collect_labels(directory: Path):
    return collect_files(directory, {".txt"})


def count_label_annotations(label_files):
    class_counts = {str(k): 0 for k in CLASS_NAMES.keys()}
    invalid_lines = []
    total_annotations = 0

    for label_path in label_files:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        for line_number, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                invalid_lines.append({
                    "file": str(label_path.relative_to(PROJECT)),
                    "line": line_number,
                    "content": line,
                    "reason": "Expected 5 YOLO fields",
                })
                continue

            try:
                class_id = int(float(parts[0]))
                values = [float(x) for x in parts[1:]]
            except Exception:
                invalid_lines.append({
                    "file": str(label_path.relative_to(PROJECT)),
                    "line": line_number,
                    "content": line,
                    "reason": "Could not parse numeric YOLO values",
                })
                continue

            if class_id not in CLASS_NAMES:
                invalid_lines.append({
                    "file": str(label_path.relative_to(PROJECT)),
                    "line": line_number,
                    "content": line,
                    "reason": f"Unexpected class id {class_id}",
                })
                continue

            if not all(0.0 <= v <= 1.0 for v in values):
                invalid_lines.append({
                    "file": str(label_path.relative_to(PROJECT)),
                    "line": line_number,
                    "content": line,
                    "reason": "YOLO normalized values outside [0, 1]",
                })
                continue

            class_counts[str(class_id)] += 1
            total_annotations += 1

    return {
        "total_annotations": total_annotations,
        "class_counts": class_counts,
        "invalid_lines": invalid_lines,
    }


def matched_image_label_summary(image_files, label_dir: Path):
    images_without_labels = []
    matched_pairs = 0

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            matched_pairs += 1
        else:
            images_without_labels.append(str(image_path.relative_to(PROJECT)))

    return {
        "matched_pairs": matched_pairs,
        "images_without_labels": images_without_labels,
    }


def inspect_csv_detectors(csv_path: Path):
    if not csv_path.exists():
        return {
            "exists": False,
            "detectors": [],
            "rows": 0,
            "columns": [],
        }

    df = pd.read_csv(csv_path)

    detectors = []
    if "detector" in df.columns:
        detectors = sorted(df["detector"].dropna().astype(str).tolist())

    return {
        "exists": True,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "detectors": detectors,
    }


def count_jsonl_records(path: Path, max_sample_lines=3):
    record_count = 0
    valid_sample_count = 0
    sample_keys = []
    sample_prediction_counts = []
    parse_errors = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            record_count += 1

            if valid_sample_count < max_sample_lines:
                try:
                    obj = json.loads(line)
                    sample_keys.append(sorted(list(obj.keys())))

                    preds = obj.get("predictions", [])
                    if isinstance(preds, list):
                        sample_prediction_counts.append(len(preds))
                    else:
                        sample_prediction_counts.append(None)

                    valid_sample_count += 1

                except Exception as exc:
                    parse_errors.append({
                        "line": line_number,
                        "error": repr(exc),
                    })

    return {
        "records": record_count,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "sample_keys": sample_keys,
        "sample_prediction_counts": sample_prediction_counts,
        "sample_parse_errors": parse_errors,
    }


def find_detector_prediction_files(prediction_dir: Path, dataset_name: str):
    results = []

    jsonl_files = sorted(prediction_dir.glob("*.jsonl")) if prediction_dir.exists() else []

    for detector in DETECTORS:
        detector_files = [
            p for p in jsonl_files
            if detector.lower() in p.name.lower()
        ]

        selected = None

        if detector_files:
            dataset_filtered = [
                p for p in detector_files
                if dataset_name.lower() in p.name.lower()
            ]
            selected = dataset_filtered[0] if dataset_filtered else detector_files[0]

        if selected is None:
            results.append({
                "dataset": dataset_name,
                "detector": detector,
                "exists": False,
                "path": None,
                "records": 0,
                "file_size_mb": 0,
                "sample_keys": [],
                "sample_prediction_counts": [],
                "sample_parse_errors": [],
            })
        else:
            inspection = count_jsonl_records(selected)

            results.append({
                "dataset": dataset_name,
                "detector": detector,
                "exists": True,
                "path": str(selected.relative_to(PROJECT)),
                **inspection,
            })

    return results


def check_detector_set(name, inspection, errors):
    expected = sorted(DETECTORS)
    actual = sorted(inspection.get("detectors", []))

    if actual != expected:
        errors.append(
            f"{name} detector mismatch. Expected {expected}, found {actual}."
        )


def main():
    print("=" * 100)
    print("STEP 1/10 - Validate Milestone 7 inputs")
    print("=" * 100)

    errors = []
    warnings = []

    print("Checking required paths...")

    path_status = {}

    for name, path in PATHS.items():
        exists = path.exists()
        path_status[name] = {
            "path": str(path.relative_to(PROJECT)) if path.exists() or PROJECT in path.parents else str(path),
            "exists": exists,
            "is_dir": path.is_dir() if exists else None,
            "is_file": path.is_file() if exists else None,
        }

        if exists:
            print("OK:", name, "->", path_status[name]["path"])
        else:
            print("MISSING:", name, "->", path)
            errors.append(f"Missing required input path: {name} -> {path}")

    print()
    print("Checking KITTI validation image/label counts...")

    kitti_images = collect_images(PATHS["kitti_val_images"])
    kitti_labels = collect_labels(PATHS["kitti_val_labels"])

    kitti_match = matched_image_label_summary(kitti_images, PATHS["kitti_val_labels"])
    kitti_label_stats = count_label_annotations(kitti_labels)

    print("KITTI images:", len(kitti_images))
    print("KITTI labels:", len(kitti_labels))
    print("KITTI matched pairs:", kitti_match["matched_pairs"])
    print("KITTI annotations:", kitti_label_stats["total_annotations"])

    if len(kitti_images) != EXPECTED_COUNTS["kitti_val_images"]:
        warnings.append(
            f"KITTI val image count expected {EXPECTED_COUNTS['kitti_val_images']}, found {len(kitti_images)}."
        )

    if len(kitti_labels) != EXPECTED_COUNTS["kitti_val_labels"]:
        warnings.append(
            f"KITTI val label count expected {EXPECTED_COUNTS['kitti_val_labels']}, found {len(kitti_labels)}."
        )

    if kitti_match["images_without_labels"]:
        errors.append(f"KITTI has {len(kitti_match['images_without_labels'])} images without labels.")

    if kitti_label_stats["invalid_lines"]:
        errors.append(f"KITTI labels contain {len(kitti_label_stats['invalid_lines'])} invalid lines.")

    print()
    print("Checking Waymo external image/label counts...")

    waymo_images = collect_images(PATHS["waymo_external_images"])
    waymo_labels = collect_labels(PATHS["waymo_external_labels"])

    waymo_match = matched_image_label_summary(waymo_images, PATHS["waymo_external_labels"])
    waymo_label_stats = count_label_annotations(waymo_labels)

    print("Waymo images:", len(waymo_images))
    print("Waymo labels:", len(waymo_labels))
    print("Waymo matched pairs:", waymo_match["matched_pairs"])
    print("Waymo annotations:", waymo_label_stats["total_annotations"])

    if len(waymo_images) != EXPECTED_COUNTS["waymo_external_images"]:
        warnings.append(
            f"Waymo image count expected {EXPECTED_COUNTS['waymo_external_images']}, found {len(waymo_images)}."
        )

    if len(waymo_labels) != EXPECTED_COUNTS["waymo_external_labels"]:
        warnings.append(
            f"Waymo label count expected {EXPECTED_COUNTS['waymo_external_labels']}, found {len(waymo_labels)}."
        )

    if waymo_match["images_without_labels"]:
        errors.append(f"Waymo has {len(waymo_match['images_without_labels'])} images without labels.")

    if waymo_label_stats["invalid_lines"]:
        errors.append(f"Waymo labels contain {len(waymo_label_stats['invalid_lines'])} invalid lines.")

    print()
    print("Checking result summary tables...")

    m5_table = inspect_csv_detectors(PATHS["m5_kitti_summary_csv"])
    m6_table = inspect_csv_detectors(PATHS["m6_waymo_summary_csv"])
    m6_ratio_table = inspect_csv_detectors(PATHS["m6_generalization_ratio_csv"])

    print("M5 KITTI summary detectors:", m5_table["detectors"])
    print("M6 Waymo summary detectors:", m6_table["detectors"])
    print("M6 Generalization ratio detectors:", m6_ratio_table["detectors"])

    if m5_table["exists"]:
        check_detector_set("M5 KITTI summary table", m5_table, errors)

    if m6_table["exists"]:
        check_detector_set("M6 Waymo summary table", m6_table, errors)

    if m6_ratio_table["exists"]:
        check_detector_set("M6 generalization ratio table", m6_ratio_table, errors)

    print()
    print("Checking prediction JSONL files...")

    prediction_inventory = []
    prediction_inventory.extend(
        find_detector_prediction_files(PATHS["m5_kitti_predictions_dir"], "kitti")
    )
    prediction_inventory.extend(
        find_detector_prediction_files(PATHS["m6_waymo_predictions_dir"], "waymo")
    )

    for item in prediction_inventory:
        status = "OK" if item["exists"] else "MISSING"
        print(
            f"{status}: {item['dataset']} / {item['detector']} -> "
            f"{item['path']} | records={item['records']} | size={item['file_size_mb']} MB"
        )

        if not item["exists"]:
            errors.append(
                f"Missing prediction JSONL for {item['dataset']} / {item['detector']}."
            )

        if item["exists"]:
            expected_records = 1496 if item["dataset"] == "kitti" else 996

            if item["records"] != expected_records:
                warnings.append(
                    f"{item['dataset']} / {item['detector']} prediction records expected "
                    f"{expected_records}, found {item['records']}."
                )

            if item["sample_parse_errors"]:
                errors.append(
                    f"{item['dataset']} / {item['detector']} prediction JSONL has parse errors."
                )

    prediction_df = pd.DataFrame(prediction_inventory)
    prediction_df.to_csv(PREDICTION_INVENTORY_CSV, index=False)

    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "FAILED" if errors else "PASSED",
        "errors": errors,
        "warnings": warnings,
        "paths": path_status,
        "datasets": {
            "kitti_val": {
                "num_images": len(kitti_images),
                "num_labels": len(kitti_labels),
                "matched_pairs": kitti_match["matched_pairs"],
                "total_annotations": kitti_label_stats["total_annotations"],
                "class_counts": kitti_label_stats["class_counts"],
                "invalid_label_lines": kitti_label_stats["invalid_lines"][:20],
                "num_invalid_label_lines": len(kitti_label_stats["invalid_lines"]),
            },
            "waymo_external": {
                "num_images": len(waymo_images),
                "num_labels": len(waymo_labels),
                "matched_pairs": waymo_match["matched_pairs"],
                "total_annotations": waymo_label_stats["total_annotations"],
                "class_counts": waymo_label_stats["class_counts"],
                "invalid_label_lines": waymo_label_stats["invalid_lines"][:20],
                "num_invalid_label_lines": len(waymo_label_stats["invalid_lines"]),
            },
        },
        "summary_tables": {
            "m5_kitti_summary": m5_table,
            "m6_waymo_summary": m6_table,
            "m6_generalization_ratio": m6_ratio_table,
        },
        "prediction_inventory": prediction_inventory,
        "outputs": {
            "audit_json": str(AUDIT_JSON.relative_to(PROJECT)),
            "audit_md": str(AUDIT_MD.relative_to(PROJECT)),
            "prediction_inventory_csv": str(PREDICTION_INVENTORY_CSV.relative_to(PROJECT)),
        },
        "recovery_if_predictions_missing": {
            "kitti": "Re-run: python scripts\\milestone_5\\03_run_final_kitti_validation.py",
            "waymo": "Re-run: python scripts\\milestone_6\\03_run_waymo_external_validation.py",
        },
    }

    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 7 Input Audit")
    md.append("")
    md.append(f"Created at: `{audit['created_at']}`")
    md.append("")
    md.append(f"Status: **{audit['status']}**")
    md.append("")
    md.append("## Dataset Summary")
    md.append("")
    md.append("| Dataset | Images | Labels | Matched Pairs | Annotations | Vehicle | Pedestrian | Cyclist |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    md.append(
        f"| KITTI val | {len(kitti_images)} | {len(kitti_labels)} | {kitti_match['matched_pairs']} | "
        f"{kitti_label_stats['total_annotations']} | {kitti_label_stats['class_counts']['0']} | "
        f"{kitti_label_stats['class_counts']['1']} | {kitti_label_stats['class_counts']['2']} |"
    )
    md.append(
        f"| Waymo external | {len(waymo_images)} | {len(waymo_labels)} | {waymo_match['matched_pairs']} | "
        f"{waymo_label_stats['total_annotations']} | {waymo_label_stats['class_counts']['0']} | "
        f"{waymo_label_stats['class_counts']['1']} | {waymo_label_stats['class_counts']['2']} |"
    )
    md.append("")
    md.append("## Prediction Inventory")
    md.append("")
    md.append("| Dataset | Detector | Exists | Records | Size MB | Path |")
    md.append("|---|---|---:|---:|---:|---|")
    for item in prediction_inventory:
        md.append(
            f"| {item['dataset']} | {item['detector']} | {item['exists']} | "
            f"{item['records']} | {item['file_size_mb']} | {item['path']} |"
        )
    md.append("")
    md.append("## Errors")
    md.append("")
    if errors:
        for error in errors:
            md.append(f"- {error}")
    else:
        md.append("- None")
    md.append("")
    md.append("## Warnings")
    md.append("")
    if warnings:
        for warning in warnings:
            md.append(f"- {warning}")
    else:
        md.append("- None")
    md.append("")
    md.append("## Recovery Commands If Prediction JSONL Files Are Missing")
    md.append("")
    md.append("```cmd")
    md.append("python scripts\\milestone_5\\03_run_final_kitti_validation.py")
    md.append("python scripts\\milestone_6\\03_run_waymo_external_validation.py")
    md.append("```")
    md.append("")

    AUDIT_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("Created:", AUDIT_JSON)
    print("Created:", AUDIT_MD)
    print("Created:", PREDICTION_INVENTORY_CSV)

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
        print("Fix the missing inputs above before continuing Milestone 7.")
        print("=" * 100)
        raise SystemExit(1)

    print()
    print("STEP 1/10 COMPLETE ✅")
    print("All Milestone 7 inputs are available for robustness and safety analysis.")
    print("=" * 100)


if __name__ == "__main__":
    main()