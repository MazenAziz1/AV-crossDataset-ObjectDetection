import os
import sys
import json
import csv
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime, timezone


def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def main():
    print("=" * 65)
    print("Creating Kaggle Training Package (KITTI-only, No Waymo)")
    print("=" * 65)

    project_root = Path(__file__).resolve().parents[3]
    package_dir = project_root / "outputs" / "milestone_4" / "kaggle_packages"
    reports_dir = project_root / "outputs" / "milestone_4" / "reports"
    manifests_dir = project_root / "outputs" / "milestone_4" / "manifests"

    os.makedirs(package_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    package_name = "milestone4_kaggle_training_package"
    zip_path = package_dir / f"{package_name}.zip"
    manifest_csv_path = manifests_dir / "kaggle_training_package_manifest.csv"
    report_json_path = reports_dir / "kaggle_training_package_report.json"
    issues_csv_path = reports_dir / "kaggle_training_package_issues.csv"

    issues = []
    manifest_rows = []
    stats = {
        "total_files": 0,
        "total_size_bytes": 0,
        "kitti_images_train": 0,
        "kitti_images_val": 0,
        "kitti_labels_train": 0,
        "kitti_labels_val": 0,
        "coco_annotations": 0,
        "sidecar_files": 0,
        "config_files": 0,
        "script_files": 0,
        "doc_files": 0,
        "other_files": 0,
        "waymo_files_found_locally": 0,
        "waymo_files_blocked": 0,
        "pretrained_weights_blocked": 0,
    }

    def log_issue(check_name, file_path, description, severity="ERROR"):
        issues.append({
            "check_name": check_name,
            "file_path": str(file_path),
            "issue_description": description,
            "severity": severity,
        })

    # --- Inclusion Rules ---

    include_rules = [
        # Configs
        ("configs/models/milestone_4/", "configs/models/milestone_4/", "*.yaml"),
        ("configs/datasets/milestone_3/", "configs/datasets/milestone_3/", "*.yaml"),
        # Scripts
        ("scripts/__init__.py", "scripts/", None),
        ("scripts/milestone_4/", "scripts/milestone_4/", "*.py"),
        ("scripts/milestone_4/kaggle/", "scripts/milestone_4/kaggle/", "*.py"),
        ("scripts/milestone_4/adapters/", "scripts/milestone_4/adapters/", "*.py"),
        ("scripts/milestone_4/trainers/", "scripts/milestone_4/trainers/", "*.py"),
        ("scripts/milestone_4/evaluation/", "scripts/milestone_4/evaluation/", "*.py"),
        ("scripts/milestone_4/utilities/", "scripts/milestone_4/utilities/", "*.py"),
        # KITTI Data only
        ("data/processed/milestone_3/images/kitti/", "data/processed/milestone_3/images/kitti/", "*.*"),
        ("data/processed/milestone_3/labels/kitti/", "data/processed/milestone_3/labels/kitti/", "*.*"),
        ("data/processed/milestone_3/annotations/coco/", "data/processed/milestone_3/annotations/coco/", "kitti_*.json"),
        ("data/processed/milestone_3/annotations/ignore_regions/", "data/processed/milestone_3/annotations/ignore_regions/", "kitti_*.json"),
        ("data/processed/milestone_3/annotations/excluded_objects/", "data/processed/milestone_3/annotations/excluded_objects/", "kitti_*.json"),
        # Docs
        ("docs/milestone_4/", "docs/milestone_4/", "*.md"),
        # Requirements
        ("requirements.txt", "", None),
    ]

    # --- Hard Exclusion Patterns (only for data files, not configs/scripts/docs) ---
    exclude_patterns = [
        "waymo",
        "Waymo",
        ".git",
        ".venv",
        "__pycache__",
        ".pyc",
        "pretrained",
        "checkpoints",
        "kaggle_packages",
        "kaggle_downloads",
        ".zip",
        ".pt",
        ".pth",
    ]

    def is_excluded(rel_path_str):
        lower = rel_path_str.lower()
        if lower.endswith("__init__.py"):
            return True
        if "kaggle/" in lower and "02_prepare_kaggle_training_package" in lower:
            return True
        if "/configs/" in lower or "/scripts/" in lower or "/docs/" in lower:
            if lower.endswith("__init__.py"):
                return True
            return False
        for pat in exclude_patterns:
            if pat.lower() in lower:
                return True
        return False

    def add_to_zip(zf, local_path, arcname):
        arcname_str = str(arcname).replace("\\", "/")
        if is_excluded(arcname_str):
            stats["waymo_files_found_locally"] += 1
            if "waymo" in arcname_str.lower():
                stats["waymo_files_blocked"] += 1
            if arcname_str.endswith((".pt", ".pth")):
                stats["pretrained_weights_blocked"] += 1
            return False

        file_size = local_path.stat().st_size
        stats["total_files"] += 1
        stats["total_size_bytes"] += file_size

        zf.write(local_path, arcname_str)

        if "images/kitti/train" in arcname_str:
            stats["kitti_images_train"] += 1
        elif "images/kitti/val" in arcname_str:
            stats["kitti_images_val"] += 1
        elif "labels/kitti/train" in arcname_str:
            stats["kitti_labels_train"] += 1
        elif "labels/kitti/val" in arcname_str:
            stats["kitti_labels_val"] += 1
        elif "annotations/coco" in arcname_str:
            stats["coco_annotations"] += 1
        elif "ignore_regions" in arcname_str or "excluded_objects" in arcname_str:
            stats["sidecar_files"] += 1
        elif "configs" in arcname_str:
            stats["config_files"] += 1
        elif "scripts" in arcname_str:
            stats["script_files"] += 1
        elif "docs" in arcname_str:
            stats["doc_files"] += 1
        else:
            stats["other_files"] += 1

        sha256_hash = compute_sha256(local_path)
        manifest_rows.append({
            "archive_path": arcname_str,
            "source_path": str(local_path),
            "file_size_bytes": file_size,
            "sha256_hash": sha256_hash,
        })

        print(f"  Added: {arcname_str}")
        return True

    # --- Build the ZIP ---
    print(f"\nCreating package: {zip_path}\n")

    package_prefix = Path(package_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for source_pattern, archive_prefix, file_pattern in include_rules:
            source_path = project_root / source_pattern

            if not source_path.exists():
                log_issue("Missing Source", source_path, "Source path does not exist", "WARNING")
                continue

            if source_path.is_file():
                arcname = package_prefix / archive_prefix / source_path.name
                add_to_zip(zf, source_path, arcname)

            elif source_path.is_dir():
                if file_pattern:
                    matched = list(source_path.rglob(file_pattern))
                else:
                    matched = [source_path]

                for file_path in matched:
                    if file_path.is_file():
                        rel = file_path.relative_to(project_root)
                        if archive_prefix:
                            arcname = package_prefix / archive_prefix / file_path.relative_to(source_path)
                        else:
                            arcname = package_prefix / rel
                        add_to_zip(zf, file_path, arcname)

    # --- Write Manifest ---
    with open(manifest_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["archive_path", "source_path", "file_size_bytes", "sha256_hash"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    print(f"\nManifest saved: {manifest_csv_path}")

    # --- Write Issues ---
    with open(issues_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "file_path", "issue_description", "severity"])
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue)
    if issues:
        print(f"Issues saved: {issues_csv_path}")

    # --- Validate and Write Report ---
    expected_kitti_train = 5985
    expected_kitti_val = 1496
    expected_coco = 2

    # Post-build ZIP audit
    zip_waymo_count = 0
    zip_weight_count = 0
    with zipfile.ZipFile(zip_path, "r") as audit_zf:
        for entry_name in audit_zf.namelist():
            entry_lower = entry_name.lower()
            if "waymo" in entry_lower and "/data/" in entry_lower:
                zip_waymo_count += 1
                log_issue("ZIP Waymo Check", entry_name, "Waymo data file found in package")
            if entry_name.endswith((".pt", ".pth")):
                zip_weight_count += 1
                log_issue("ZIP Weight Check", entry_name, "Pretrained weight file found in package")

    validation_checks = {
        "kitti_train_images_count": {
            "expected": expected_kitti_train,
            "actual": stats["kitti_images_train"],
            "passed": stats["kitti_images_train"] == expected_kitti_train,
        },
        "kitti_val_images_count": {
            "expected": expected_kitti_val,
            "actual": stats["kitti_images_val"],
            "passed": stats["kitti_images_val"] == expected_kitti_val,
        },
        "coco_annotations_count": {
            "expected": expected_coco,
            "actual": stats["coco_annotations"],
            "passed": stats["coco_annotations"] == expected_coco,
        },
        "zip_waymo_data_files": {
            "expected": 0,
            "actual": zip_waymo_count,
            "passed": zip_waymo_count == 0,
        },
        "zip_pretrained_weights": {
            "expected": 0,
            "actual": zip_weight_count,
            "passed": zip_weight_count == 0,
        },
    }

    for check, detail in validation_checks.items():
        print(f"  {check}: expected={detail['expected']} actual={detail['actual']} -> {'PASSED' if detail['passed'] else 'FAILED'}")
        if not detail["passed"]:
            log_issue(check, zip_path, f"Expected {detail['expected']}, got {detail['actual']}")

    zip_size_mb = round(os.path.getsize(zip_path) / (1024 * 1024), 2)
    all_checks_passed = all(v["passed"] for v in validation_checks.values())
    errors_count = len([i for i in issues if i["severity"] == "ERROR"])
    final_status = "PASSED" if (all_checks_passed and errors_count == 0) else "FAILED"

    report = {
        "milestone": 4,
        "step": 6,
        "purpose": "Create self-contained Kaggle training package with KITTI data only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "package": {
            "path": str(zip_path),
            "size_mb": zip_size_mb,
            "total_files": stats["total_files"],
            "total_size_bytes": stats["total_size_bytes"],
        },
        "content_counts": {
            "kitti_images_train": stats["kitti_images_train"],
            "kitti_images_val": stats["kitti_images_val"],
            "kitti_labels_train": stats["kitti_labels_train"],
            "kitti_labels_val": stats["kitti_labels_val"],
            "coco_annotations": stats["coco_annotations"],
            "sidecar_files": stats["sidecar_files"],
            "config_files": stats["config_files"],
            "script_files": stats["script_files"],
            "doc_files": stats["doc_files"],
            "other_files": stats["other_files"],
        },
        "exclusion_audit": {
            "waymo_data_files_in_zip": zip_waymo_count,
            "pretrained_weights_in_zip": zip_weight_count,
            "waymo_files_found_locally": stats["waymo_files_found_locally"],
            "waymo_files_blocked": stats["waymo_files_blocked"],
        },
        "validation_checks": {k: {"expected": v["expected"], "actual": v["actual"], "passed": v["passed"]} for k, v in validation_checks.items()},
        "errors_count": errors_count,
        "warnings_count": len([i for i in issues if i["severity"] == "WARNING"]),
        "final_status": final_status,
    }

    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_json_path}")

    print("\n" + "-" * 65)
    print(f"Package Status: {final_status}")
    print(f"  Package: {zip_path} ({zip_size_mb} MB)")
    print(f"  Files: {stats['total_files']}")
    print(f"  Waymo data files in ZIP: {zip_waymo_count} (must be 0)")
    print(f"  Pretrained weights in ZIP: {zip_weight_count} (must be 0)")
    print("=" * 65)

    if final_status == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
