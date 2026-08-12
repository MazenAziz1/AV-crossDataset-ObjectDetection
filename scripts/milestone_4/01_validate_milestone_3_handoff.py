from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()
M3_ROOT = PROJECT_ROOT / "data" / "processed" / "milestone_3"
M4_REPORTS = PROJECT_ROOT / "outputs" / "milestone_4" / "reports"

EXPECTED_COUNTS = {
    "kitti_train": 5985,
    "kitti_val": 1496,
}

TARGET_CLASSES = {"Vehicle", "Pedestrian", "Cyclist"}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for this validation script.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_result(
    results: list[dict[str, Any]],
    check: str,
    passed: bool,
    details: str,
    severity: str = "error",
) -> None:
    results.append(
        {
            "check": check,
            "passed": bool(passed),
            "severity": severity,
            "details": details,
        }
    )


def find_coco_file(split: str) -> Path | None:
    coco_dir = M3_ROOT / "annotations" / "coco"
    if not coco_dir.exists():
        return None

    candidates = []
    for path in coco_dir.rglob("*.json"):
        name = path.name.lower()
        if split == "kitti_train":
            if "kitti" in name and "train" in name:
                candidates.append(path)
        elif split == "kitti_val":
            if "kitti" in name and ("val" in name or "validation" in name):
                candidates.append(path)

    if not candidates:
        return None

    # Prefer files with the split name closest to the filename.
    candidates = sorted(candidates, key=lambda p: len(str(p)))
    return candidates[0]


def build_image_index(images_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    if not images_root.exists():
        return image_index

    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in images_root.rglob(ext):
            image_index.setdefault(path.name, path)
            rel = path.relative_to(images_root).as_posix()
            image_index.setdefault(rel, path)

    return image_index


def resolve_image_path(file_name: str, images_root: Path, image_index: dict[str, Path]) -> Path | None:
    raw = Path(file_name)

    candidates = [
        images_root / raw,
        images_root / raw.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    normalized = file_name.replace("\\", "/")
    if normalized in image_index:
        return image_index[normalized]

    if raw.name in image_index:
        return image_index[raw.name]

    return None


def get_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        pass

    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return None
        height, width = img.shape[:2]
        return width, height
    except Exception:
        return None


def validate_coco_split(
    results: list[dict[str, Any]],
    split: str,
    expected_images: int,
    image_index: dict[str, Path],
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {
        "split": split,
        "expected_images": expected_images,
        "coco_file": None,
        "actual_images": None,
        "actual_annotations": None,
        "missing_images": 0,
        "wrong_size_images": 0,
        "checked_image_sizes": 0,
    }

    coco_file = find_coco_file(split)
    add_result(
        results,
        f"{split}: COCO file exists",
        coco_file is not None,
        str(coco_file) if coco_file else "No matching COCO file found.",
    )

    if coco_file is None:
        return split_summary

    split_summary["coco_file"] = str(coco_file)

    try:
        coco = load_json(coco_file)
    except Exception as exc:
        add_result(results, f"{split}: COCO file readable", False, repr(exc))
        return split_summary

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    split_summary["actual_images"] = len(images)
    split_summary["actual_annotations"] = len(annotations)

    add_result(
        results,
        f"{split}: image count",
        len(images) == expected_images,
        f"expected={expected_images}, actual={len(images)}",
    )

    category_names = {str(cat.get("name")) for cat in categories}
    has_target_classes = TARGET_CLASSES.issubset(category_names)

    add_result(
        results,
        f"{split}: target categories exist",
        has_target_classes,
        f"categories={sorted(category_names)}",
    )

    images_root = M3_ROOT / "images"
    missing_images = []
    wrong_size_images = []

    for item in images:
        file_name = str(item.get("file_name", ""))
        image_path = resolve_image_path(file_name, images_root, image_index)

        if image_path is None:
            missing_images.append(file_name)
            continue

        size = get_image_size(image_path)
        if size is None:
            wrong_size_images.append(f"{file_name}: unreadable")
            continue

        split_summary["checked_image_sizes"] += 1
        if size != (640, 640):
            wrong_size_images.append(f"{file_name}: {size}")

    split_summary["missing_images"] = len(missing_images)
    split_summary["wrong_size_images"] = len(wrong_size_images)

    add_result(
        results,
        f"{split}: referenced images exist",
        len(missing_images) == 0,
        f"missing={len(missing_images)}; examples={missing_images[:5]}",
    )

    add_result(
        results,
        f"{split}: images are 640x640",
        len(wrong_size_images) == 0,
        f"wrong_size={len(wrong_size_images)}; examples={wrong_size_images[:5]}",
    )

    return split_summary


def find_status_in_json_files(root: Path, keywords: list[str]) -> tuple[bool, list[str]]:
    matches = []
    if not root.exists():
        return False, matches

    for path in root.rglob("*.json"):
        name = path.name.lower()
        if all(keyword in name for keyword in keywords):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            if "passed" in text or '"final_status": true' in text or '"final_dataset_audit": true' in text:
                matches.append(str(path))

    return bool(matches), matches


def main() -> None:
    M4_REPORTS.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "milestone_3_root": str(M3_ROOT),
        "expected_counts": EXPECTED_COUNTS,
        "split_summaries": {},
    }

    add_result(
        results,
        "Milestone 3 root exists",
        M3_ROOT.exists(),
        str(M3_ROOT),
    )

    required_dirs = [
        M3_ROOT / "images",
        M3_ROOT / "annotations" / "coco",
        M3_ROOT / "labels",
        M3_ROOT / "annotations" / "ignore_regions",
        M3_ROOT / "annotations" / "excluded_objects",
        M3_ROOT / "manifests",
        M3_ROOT / "reports",
    ]

    for directory in required_dirs:
        add_result(
            results,
            f"Required directory exists: {directory.relative_to(PROJECT_ROOT)}",
            directory.exists(),
            str(directory),
        )

    image_index = build_image_index(M3_ROOT / "images")
    add_result(
        results,
        "Image index built",
        len(image_index) > 0,
        f"indexed_entries={len(image_index)}",
    )

    for split, expected_count in EXPECTED_COUNTS.items():
        summary["split_summaries"][split] = validate_coco_split(
            results=results,
            split=split,
            expected_images=expected_count,
            image_index=image_index,
        )

    labels_dir = M3_ROOT / "labels"
    label_files = list(labels_dir.rglob("*.txt")) if labels_dir.exists() else []
    add_result(
        results,
        "YOLO label files exist locally",
        len(label_files) >= sum(EXPECTED_COUNTS.values()),
        f"label_files={len(label_files)}, expected_at_least={sum(EXPECTED_COUNTS.values())}",
    )

    ignore_dir = M3_ROOT / "annotations" / "ignore_regions"
    ignore_files = list(ignore_dir.rglob("*")) if ignore_dir.exists() else []
    ignore_files = [p for p in ignore_files if p.is_file()]
    add_result(
        results,
        "Ignore-region sidecars exist",
        len(ignore_files) > 0,
        f"files={len(ignore_files)}",
    )

    excluded_dir = M3_ROOT / "annotations" / "excluded_objects"
    excluded_files = list(excluded_dir.rglob("*")) if excluded_dir.exists() else []
    excluded_files = [p for p in excluded_files if p.is_file()]
    add_result(
        results,
        "Excluded-object sidecars exist",
        len(excluded_files) > 0,
        f"files={len(excluded_files)}",
    )

    class_mapping_path = PROJECT_ROOT / "configs" / "datasets" / "milestone_3" / "class_mapping.yaml"
    add_result(
        results,
        "Milestone 3 class mapping exists",
        class_mapping_path.exists(),
        str(class_mapping_path),
    )

    if class_mapping_path.exists():
        try:
            class_mapping_text = class_mapping_path.read_text(encoding="utf-8", errors="ignore")
            add_result(
                results,
                "Milestone 3 class mapping contains target classes",
                all(cls in class_mapping_text for cls in TARGET_CLASSES),
                "Checked Vehicle, Pedestrian, Cyclist.",
            )
        except Exception as exc:
            add_result(results, "Milestone 3 class mapping readable", False, repr(exc))

    reports_dir = M3_ROOT / "reports"
    audit_found, audit_matches = find_status_in_json_files(reports_dir, ["audit"])
    add_result(
        results,
        "Milestone 3 audit report indicates passed",
        audit_found,
        f"matches={audit_matches[:5]}",
        severity="warning",
    )

    repro_found, repro_matches = find_status_in_json_files(reports_dir, ["repro"])
    add_result(
        results,
        "Milestone 3 reproducibility report indicates passed",
        repro_found,
        f"matches={repro_matches[:5]}",
        severity="warning",
    )

    # Validate Milestone 4 policy values already frozen in Step 2 and Step 3.
    experiment_policy = PROJECT_ROOT / "configs" / "models" / "milestone_4" / "experiment_protocol.yaml"
    kaggle_policy = PROJECT_ROOT / "configs" / "models" / "milestone_4" / "kaggle_training_policy.yaml"
    compute_plan = PROJECT_ROOT / "configs" / "models" / "milestone_4" / "kaggle_compute_plan.yaml"

    for policy_file in [experiment_policy, kaggle_policy, compute_plan]:
        add_result(
            results,
            f"Milestone 4 policy exists: {policy_file.name}",
            policy_file.exists(),
            str(policy_file),
        )

    try:
        exp = load_yaml(experiment_policy)
        add_result(
            results,
            "Waymo disabled in experiment protocol",
            exp.get("scope", {}).get("waymo_allowed_in_milestone_4_plus_5") is False,
            str(exp.get("scope", {}).get("waymo_allowed_in_milestone_4_plus_5")),
        )
    except Exception as exc:
        add_result(results, "Experiment protocol readable", False, repr(exc))

    try:
        kg = load_yaml(kaggle_policy)
        add_result(
            results,
            "Local training disabled in Kaggle training policy",
            kg.get("execution_policy", {}).get("local_training_allowed") is False,
            str(kg.get("execution_policy", {}).get("local_training_allowed")),
        )
        add_result(
            results,
            "Kaggle training only enabled",
            kg.get("execution_policy", {}).get("kaggle_training_only") is True,
            str(kg.get("execution_policy", {}).get("kaggle_training_only")),
        )
    except Exception as exc:
        add_result(results, "Kaggle training policy readable", False, repr(exc))

    try:
        cp = load_yaml(compute_plan)
        add_result(
            results,
            "Compute plan has Slot A and Slot B",
            "slot_a" in cp.get("compute_slots", {}) and "slot_b" in cp.get("compute_slots", {}),
            str(cp.get("compute_slots", {}).keys()),
        )
        add_result(
            results,
            "RT-DETR resume policy exists",
            "rtdetr_resume_policy" in cp,
            "rtdetr_resume_policy present",
        )
    except Exception as exc:
        add_result(results, "Kaggle compute plan readable", False, repr(exc))

    error_issues = [row for row in results if not row["passed"] and row["severity"] == "error"]
    warning_issues = [row for row in results if not row["passed"] and row["severity"] == "warning"]

    final_status = len(error_issues) == 0

    summary["checks_total"] = len(results)
    summary["checks_passed"] = sum(1 for row in results if row["passed"])
    summary["error_issues"] = len(error_issues)
    summary["warning_issues"] = len(warning_issues)
    summary["final_status"] = "PASSED" if final_status else "FAILED"

    validation_json = M4_REPORTS / "milestone_3_handoff_validation.json"
    issues_csv = M4_REPORTS / "milestone_3_handoff_issues.csv"

    validation_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "checks": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with issues_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed", "severity", "details"])
        writer.writeheader()
        for row in results:
            if not row["passed"]:
                writer.writerow(row)

    print("Milestone 3 handoff validation complete.")
    print(f"Final status: {summary['final_status']}")
    print(f"Checks passed: {summary['checks_passed']} / {summary['checks_total']}")
    print(f"Error issues: {summary['error_issues']}")
    print(f"Warning issues: {summary['warning_issues']}")
    print(f"Validation report: {validation_json}")
    print(f"Issues CSV: {issues_csv}")

    if not final_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()