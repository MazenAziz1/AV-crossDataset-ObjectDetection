from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

UPLOAD_SET_MANIFEST = PROJECT_ROOT / "outputs" / "milestone_4" / "manifests" / "kaggle_upload_set_manifest.json"

PACKAGE_DIR = PROJECT_ROOT / "outputs" / "milestone_4" / "kaggle_packages"
PACKAGE_PATH = PACKAGE_DIR / "milestone4_kaggle_training_package.zip"

PACKAGE_MANIFEST_JSON = PROJECT_ROOT / "outputs" / "milestone_4" / "manifests" / "kaggle_package_manifest.json"
PACKAGE_MANIFEST_CSV = PROJECT_ROOT / "outputs" / "milestone_4" / "manifests" / "kaggle_package_manifest.csv"

PACKAGE_ROOT_NAME = "milestone4_kaggle_training_package"

PROHIBITED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "checkpoints",
    "pretrained",
    "kaggle_packages",
    "kaggle_downloads",
    "runtime",
    "cache",
}

PROHIBITED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".engine",
    ".pkl",
    ".pickle",
}

PROHIBITED_NAME_FRAGMENTS = {
    "secret",
    "credential",
    "token",
    "password",
    "apikey",
    "api_key",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_prohibited(relative_path: str) -> tuple[bool, str]:
    normalized = relative_path.replace("\\", "/")
    parts = set(normalized.split("/"))
    lower_name = Path(normalized).name.lower()
    suffix = Path(normalized).suffix.lower()

    blocked_parts = sorted(parts.intersection(PROHIBITED_PARTS))
    if blocked_parts:
        return True, f"prohibited path part: {blocked_parts}"

    if suffix in PROHIBITED_SUFFIXES:
        return True, f"prohibited suffix: {suffix}"

    for fragment in PROHIBITED_NAME_FRAGMENTS:
        if fragment in lower_name:
            return True, f"prohibited name fragment: {fragment}"

    if "waymo" in normalized.lower():
        return True, "Waymo is prohibited in Milestone 4 + 5 package"

    return False, ""


def load_upload_set() -> list[dict[str, Any]]:
    if not UPLOAD_SET_MANIFEST.exists():
        raise FileNotFoundError(f"Upload set manifest not found: {UPLOAD_SET_MANIFEST}")

    data = json.loads(UPLOAD_SET_MANIFEST.read_text(encoding="utf-8"))

    if data.get("summary", {}).get("status") != "PASSED":
        raise RuntimeError("Upload set manifest status is not PASSED.")

    if data.get("summary", {}).get("waymo_included") is not False:
        raise RuntimeError("Upload set manifest does not explicitly exclude Waymo.")

    records = data.get("files", [])
    if not records:
        raise RuntimeError("Upload set manifest has no files.")

    return records


def write_package_readme() -> Path:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    readme_path = PACKAGE_DIR / "PACKAGE_README.md"

    content = (
        "# Milestone 4 + 5 Kaggle Training Package\n\n"
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "Purpose:\n"
        "This ZIP package contains the source code, frozen configs, processed KITTI data, "
        "labels, annotations, manifests, reports, and documentation required for Kaggle-based training.\n\n"
        "Important boundary:\n"
        "Waymo is intentionally excluded from this package. Waymo external validation is deferred to Milestone 6.\n\n"
        "Expected Kaggle extraction path:\n"
        "/kaggle/working/project/milestone4_kaggle_training_package\n\n"
        "Training role:\n"
        "Kaggle is used for GPU training only. The local machine remains the source of truth.\n\n"
        f"Package root:\n{PACKAGE_ROOT_NAME}/\n"
    )

    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)

    upload_records = load_upload_set()
    package_readme = write_package_readme()

    package_records: list[dict[str, Any]] = []
    blocked_records: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []
    files_to_zip: list[tuple[Path, str, str]] = []

    for row in upload_records:
        relative_path = row["relative_path"]
        exists = bool(row.get("exists"))

        blocked, reason = is_prohibited(relative_path)
        if blocked:
            blocked_records.append({"relative_path": relative_path, "reason": reason})
            continue

        source_path = PROJECT_ROOT / relative_path

        if not exists or not source_path.exists():
            missing_records.append({"relative_path": relative_path, "reason": "missing file"})
            continue

        if not source_path.is_file():
            continue

        archive_path = f"{PACKAGE_ROOT_NAME}/{relative_path.replace(chr(92), '/')}"
        files_to_zip.append((source_path, archive_path, row.get("group", "unknown")))

    files_to_zip.append((package_readme, f"{PACKAGE_ROOT_NAME}/PACKAGE_README.md", "package_metadata"))

    if blocked_records:
        print("Blocked records found:")
        for item in blocked_records[:30]:
            print("-", item["relative_path"], "=>", item["reason"])
        raise SystemExit("Packaging stopped because prohibited files were detected.")

    if missing_records:
        print("Missing records found:")
        for item in missing_records[:30]:
            print("-", item["relative_path"])
        raise SystemExit("Packaging stopped because required files are missing.")

    if PACKAGE_PATH.exists():
        PACKAGE_PATH.unlink()

    print(f"Creating package: {PACKAGE_PATH}")
    print(f"Files to zip: {len(files_to_zip)}")

    with zipfile.ZipFile(
        PACKAGE_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as zf:
        for idx, (source_path, archive_path, group) in enumerate(files_to_zip, start=1):
            zf.write(source_path, archive_path)

            package_records.append(
                {
                    "group": group,
                    "source_relative_path": source_path.relative_to(PROJECT_ROOT).as_posix(),
                    "archive_path": archive_path,
                    "size_bytes": source_path.stat().st_size,
                    "sha256": sha256_file(source_path),
                }
            )

            if idx % 500 == 0:
                print(f"Zipped {idx}/{len(files_to_zip)} files...")

    package_sha256 = sha256_file(PACKAGE_PATH)
    package_size = PACKAGE_PATH.stat().st_size

    group_counts: dict[str, int] = {}
    group_sizes: dict[str, int] = {}

    for row in package_records:
        group = row["group"]
        group_counts[group] = group_counts.get(group, 0) + 1
        group_sizes[group] = group_sizes.get(group, 0) + int(row["size_bytes"])

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "package_path": PACKAGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "package_size_bytes": package_size,
        "package_sha256": package_sha256,
        "package_root_name": PACKAGE_ROOT_NAME,
        "files_packaged": len(package_records),
        "blocked_records": len(blocked_records),
        "missing_records": len(missing_records),
        "waymo_included": False,
        "group_counts": group_counts,
        "group_sizes_bytes": group_sizes,
    }

    PACKAGE_MANIFEST_JSON.write_text(
        json.dumps({"summary": summary, "files": package_records}, indent=2),
        encoding="utf-8",
    )

    with PACKAGE_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "source_relative_path", "archive_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(package_records)

    print("Kaggle training package created successfully.")
    print(f"Status: {summary['status']}")
    print(f"Package: {PACKAGE_PATH}")
    print(f"Package size bytes: {package_size}")
    print(f"Package SHA256: {package_sha256}")
    print(f"Files packaged: {len(package_records)}")
    print(f"Manifest JSON: {PACKAGE_MANIFEST_JSON}")
    print(f"Manifest CSV: {PACKAGE_MANIFEST_CSV}")


if __name__ == "__main__":
    main()