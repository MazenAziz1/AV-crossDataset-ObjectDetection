import os
import json
import csv
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def main():
    project_root = Path(__file__).resolve().parents[2]
    checkpoints_root = project_root / "outputs" / "milestone_4" / "checkpoints"
    manifests_dir = project_root / "outputs" / "milestone_4" / "manifests"
    reports_dir = project_root / "outputs" / "milestone_4" / "reports"

    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    detectors = ["yolo", "faster_rcnn", "retinanet", "rtdetr"]
    registry = []
    issues = []

    for det in detectors:
        best_pt = checkpoints_root / det / "final" / "best.pt"
        if not best_pt.exists():
            issues.append({
                "detector": det,
                "issue": f"best.pt not found at {best_pt}",
                "severity": "ERROR",
            })
            continue

        sha256 = compute_sha256(best_pt)
        size_mb = round(os.path.getsize(best_pt) / (1024 * 1024), 2)

        # Load test
        load_ok = False
        try:
            if det in ("yolo", "rtdetr"):
                from ultralytics import YOLO, RTDETR
                cls = YOLO if det == "yolo" else RTDETR
                m = cls(str(best_pt))
                load_ok = True
            else:
                import torch
                from scripts.milestone_4.adapters.torchvision_adapter import _build_model, _load_state
                m = _build_model(det, "cuda" if torch.cuda.is_available() else "cpu")
                _load_state(best_pt, m, "cuda" if torch.cuda.is_available() else "cpu")
                load_ok = True
        except Exception as e:
            load_ok = False
            issues.append({
                "detector": det,
                "issue": f"Failed to load checkpoint: {str(e)}",
                "severity": "ERROR",
            })

        registry.append({
            "detector": det,
            "checkpoint_path": str(best_pt),
            "file_size_mb": size_mb,
            "sha256": sha256,
            "load_test": "PASSED" if load_ok else "FAILED",
            "locked_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"{det}: {size_mb} MB, sha256={sha256[:16]}..., load={'PASSED' if load_ok else 'FAILED'}")

    registry_path = manifests_dir / "final_checkpoint_registry.csv"
    with open(registry_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["detector", "checkpoint_path", "file_size_mb", "sha256", "load_test", "locked_at"])
        writer.writeheader()
        writer.writerows(registry)

    issues_path = reports_dir / "checkpoint_loading_issues.csv"
    with open(issues_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["detector", "issue", "severity"])
        writer.writeheader()
        writer.writerows(issues)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "locked_checkpoints": len(registry),
        "expected_checkpoints": len(detectors),
        "issues": len(issues),
        "registry": registry,
    }
    report_path = reports_dir / "checkpoint_selection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nLocked {len(registry)}/{len(detectors)} checkpoints.")
    print(f"Registry: {registry_path}")
    print(f"Report: {report_path}")

    if len(issues) > 0:
        for i in issues:
            print(f"  [ISSUE] {i['detector']}: {i['issue']}")


if __name__ == "__main__":
    main()
