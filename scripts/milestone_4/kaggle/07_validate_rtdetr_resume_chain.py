import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / "outputs" / "milestone_4" / "reports"
    manifests_dir = project_root / "outputs" / "milestone_4" / "manifests"
    os.makedirs(reports_dir, exist_ok=True)

    session_manifest_path = manifests_dir / "kaggle_session_manifest.csv"
    issues = []

    if not session_manifest_path.exists():
        print(f"Session manifest not found: {session_manifest_path}")
        print("This validation requires the RT-DETR training session manifest imported from Kaggle.")
        print("Reporting as NOT RUNNABLE yet (RT-DETR training in progress).")
        report = {
            "status": "PENDING",
            "reason": "Session manifest not yet imported (RT-DETR training in progress)",
            "checks": {},
        }
    else:
        import csv
        with open(session_manifest_path) as f:
            rows = list(csv.DictReader(f))

        rtdetr_sessions = [r for r in rows if r.get("detector") == "rtdetr"]

        if not rtdetr_sessions:
            report = {
                "status": "PENDING",
                "reason": "No RT-DETR sessions recorded yet",
                "checks": {},
            }
        else:
            checks = {}
            epochs = []
            for s in rtdetr_sessions:
                start = int(s.get("start_epoch", -1))
                end = int(s.get("end_epoch", -1))
                epochs.append((start, end))

            # Check epoch continuity
            gaps = []
            prev_end = -1
            for start, end in sorted(epochs):
                if prev_end != -1 and start != prev_end + 1:
                    gaps.append(f"gap: expected {prev_end + 1}, got {start}")
                prev_end = end

            checks["epoch_continuity"] = "PASSED" if not gaps else "FAILED"
            checks["gaps"] = gaps
            checks["sessions"] = len(rtdetr_sessions)
            checks["final_epoch"] = max(end for _, end in epochs)

            report = {
                "status": "PASSED" if not gaps else "FAILED",
                "reason": "",
                "checks": checks,
            }

    report_path = reports_dir / "rtdetr_resume_chain_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
