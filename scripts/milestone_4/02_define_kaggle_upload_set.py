from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

CONFIG_PATH = PROJECT_ROOT / "configs" / "models" / "milestone_4" / "kaggle_upload_set.yaml"
DOC_PATH = PROJECT_ROOT / "docs" / "milestone_4" / "kaggle_upload_set.md"

MANIFEST_JSON = PROJECT_ROOT / "outputs" / "milestone_4" / "manifests" / "kaggle_upload_set_manifest.json"
MANIFEST_CSV = PROJECT_ROOT / "outputs" / "milestone_4" / "manifests" / "kaggle_upload_set_manifest.csv"

M3_ROOT = PROJECT_ROOT / "data" / "processed" / "milestone_3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, group: str, required: bool = True) -> dict[str, Any]:
    exists = path.exists()
    return {
        "group": group,
        "required": required,
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256": sha256_file(path) if exists and path.is_file() else None,
    }


def collect_files(root: Path, group: str, patterns: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        records.append(file_record(root, group, required=True))
        return records

    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                records.append(file_record(path, group, required=True))

    return records


def write_yaml_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    content = """status: frozen
milestone: "4_plus_5"
title: "Kaggle Upload Set Definition"

purpose: "Define the exact local files and directories that are allowed to enter the Kaggle training package."

allowed_upload_groups:
  source_code:
    include:
      - scripts/milestone_4
    exclude:
      - outputs
      - checkpoints
      - pretrained weights
      - kaggle downloaded packages

  configs:
    include:
      - configs/models/milestone_4
      - configs/datasets/milestone_3
    exclude:
      - local secrets
      - credentials

  dataset:
    include:
      - data/processed/milestone_3/images
      - data/processed/milestone_3/annotations/coco
      - data/processed/milestone_3/labels
      - data/processed/milestone_3/annotations/ignore_regions
      - data/processed/milestone_3/annotations/excluded_objects
      - data/processed/milestone_3/manifests
      - data/processed/milestone_3/reports
    exclude:
      - Waymo raw data
      - original KITTI raw ZIP files
      - Milestone 6 external validation packages

  documentation:
    include:
      - docs/milestone_4
    exclude:
      - private notes
      - temporary reports

dataset_policy:
  include_kitti_train: true
  include_kitti_val: true
  include_waymo_external: false
  waymo_deferred_to_milestone: 6

training_policy:
  kaggle_training_only: true
  local_final_evaluation_preferred: true
  local_training_allowed: false

prohibited_uploads:
  - model checkpoints
  - pretrained model weights
  - Kaggle output ZIP files
  - secrets
  - credentials
  - raw Waymo data
  - Waymo external validation subset
  - Milestone 6 evaluation outputs

required_before_packaging:
  - configs/models/milestone_4/kaggle_compute_plan.yaml
  - configs/models/milestone_4/experiment_protocol.yaml
  - configs/models/milestone_4/class_contract.yaml
  - configs/models/milestone_4/evaluation_policy.yaml
  - configs/models/milestone_4/kaggle_training_policy.yaml
  - outputs/milestone_4/reports/milestone_3_handoff_validation.json
"""
    CONFIG_PATH.write_text(content, encoding="utf-8")


def write_documentation() -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    content = """# Kaggle Upload Set — Milestone 4 + 5

## Purpose

This document defines what is allowed to enter the Kaggle training package.

Kaggle is used for GPU training only. The local machine remains the source of truth for code, configuration, validation, documentation, imported outputs, and Git commits.

## Included Groups

| Group | Included |
|---|---|
| Source code | `scripts/milestone_4` |
| Model configs | `configs/models/milestone_4` |
| Dataset configs | `configs/datasets/milestone_3` |
| Processed images | `data/processed/milestone_3/images` |
| COCO annotations | `data/processed/milestone_3/annotations/coco` |
| YOLO labels | `data/processed/milestone_3/labels` |
| Ignore regions | `data/processed/milestone_3/annotations/ignore_regions` |
| Excluded objects | `data/processed/milestone_3/annotations/excluded_objects` |
| Manifests | `data/processed/milestone_3/manifests` |
| Reports | `data/processed/milestone_3/reports` |
| Documentation | `docs/milestone_4` |

## Dataset Boundary

| Dataset | Included? |
|---|---|
| KITTI train | Yes |
| KITTI validation | Yes |
| Waymo external validation | No |

Waymo remains excluded from Milestone 4 + 5 and is deferred to Milestone 6.

## Prohibited Uploads

The Kaggle package must not include:

- model checkpoints;
- pretrained weights;
- Kaggle output ZIP packages;
- credentials or secrets;
- raw Waymo data;
- Waymo external validation data;
- Milestone 6 evaluation outputs.

## Completion Gate

This step is complete when the upload-set YAML and manifest exist and the manifest confirms all required groups are present.
"""
    DOC_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    write_yaml_config()
    write_documentation()

    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)

    required_single_files = [
        CONFIG_PATH,
        DOC_PATH,
        PROJECT_ROOT / "configs" / "models" / "milestone_4" / "kaggle_compute_plan.yaml",
        PROJECT_ROOT / "configs" / "models" / "milestone_4" / "experiment_protocol.yaml",
        PROJECT_ROOT / "configs" / "models" / "milestone_4" / "class_contract.yaml",
        PROJECT_ROOT / "configs" / "models" / "milestone_4" / "evaluation_policy.yaml",
        PROJECT_ROOT / "configs" / "models" / "milestone_4" / "kaggle_training_policy.yaml",
        PROJECT_ROOT / "outputs" / "milestone_4" / "reports" / "milestone_3_handoff_validation.json",
    ]

    records: list[dict[str, Any]] = []
    for path in required_single_files:
        records.append(file_record(path, "required_single_file", required=True))

    collection_specs = [
        (PROJECT_ROOT / "scripts" / "milestone_4", "source_code", ["*.py"]),
        (PROJECT_ROOT / "configs" / "models" / "milestone_4", "milestone_4_configs", ["*.yaml"]),
        (PROJECT_ROOT / "configs" / "datasets" / "milestone_3", "milestone_3_dataset_configs", ["*.yaml"]),
        (M3_ROOT / "images", "processed_images", ["*.png", "*.jpg", "*.jpeg"]),
        (M3_ROOT / "annotations" / "coco", "coco_annotations", ["*.json"]),
        (M3_ROOT / "labels", "yolo_labels", ["*.txt"]),
        (M3_ROOT / "annotations" / "ignore_regions", "ignore_regions", ["*"]),
        (M3_ROOT / "annotations" / "excluded_objects", "excluded_objects", ["*"]),
        (M3_ROOT / "manifests", "m3_manifests", ["*.json", "*.csv", "*.yaml"]),
        (M3_ROOT / "reports", "m3_reports", ["*.json", "*.csv", "*.txt", "*.md"]),
        (PROJECT_ROOT / "docs" / "milestone_4", "documentation", ["*.md"]),
    ]

    for root, group, patterns in collection_specs:
        records.extend(collect_files(root, group, patterns))

    # Remove duplicate files while preserving order.
    seen = set()
    unique_records = []
    for row in records:
        key = row["relative_path"]
        if key not in seen:
            seen.add(key)
            unique_records.append(row)

    records = unique_records

    # Milestone 4 + 5 package must exclude Waymo completely.
    # This removes Waymo images, annotations, labels, manifests, reports,
    # and config files such as kitti_waymo_yolo.yaml.
    excluded_records = []
    filtered_records = []

    for row in records:
    	rel = row["relative_path"].replace("\\", "/").lower()
    	if "waymo" in rel:
        	excluded_records.append(row)
    	else:
        	filtered_records.append(row)

    records = filtered_records

    missing_required = [row for row in records if row["required"] and not row["exists"]]

    group_counts: dict[str, int] = {}
    group_sizes: dict[str, int] = {}
    for row in records:
        group_counts[row["group"]] = group_counts.get(row["group"], 0) + 1
        group_sizes[row["group"]] = group_sizes.get(row["group"], 0) + int(row["size_bytes"] or 0)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED" if not missing_required else "FAILED",
        "project_root": str(PROJECT_ROOT),
        "total_files": len(records),
        "total_size_bytes": sum(int(row["size_bytes"] or 0) for row in records),
        "missing_required_count": len(missing_required),
        "group_counts": group_counts,
        "group_sizes_bytes": group_sizes,
        "waymo_included": False,
	"excluded_waymo_files": len(excluded_records),
	"note": "This is a definition/manifest step only. It does not create the Kaggle ZIP package.",
    }

    MANIFEST_JSON.write_text(
        json.dumps({"summary": summary, "files": records}, indent=2),
        encoding="utf-8",
    )

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "required",
                "relative_path",
                "exists",
                "size_bytes",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print("Kaggle upload set definition complete.")
    print(f"Status: {summary['status']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Total size bytes: {summary['total_size_bytes']}")
    print(f"Missing required: {summary['missing_required_count']}")
    print(f"Manifest JSON: {MANIFEST_JSON}")
    print(f"Manifest CSV: {MANIFEST_CSV}")

    if missing_required:
        print("Missing required files:")
        for row in missing_required:
            print("-", row["relative_path"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()