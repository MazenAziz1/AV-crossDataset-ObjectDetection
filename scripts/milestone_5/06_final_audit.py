from pathlib import Path
import json
import sys
from datetime import datetime

PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

REQUIRED_FILES = [
    "docs/milestone_5/Milestone_4_5_Final_Report.docx",

    "outputs/milestone_4/locked_final_checkpoints/final_checkpoint_registry.json",
    "outputs/milestone_4/locked_final_checkpoints/LOCKED_FINAL_CHECKPOINTS.md",

    "outputs/milestone_5/final_kitti_validation/tables/comparison_summary_full.csv",
    "outputs/milestone_5/final_kitti_validation/tables/comparison_summary_full.json",

    "outputs/milestone_5/benchmark_comparison/tables/detector_ranking_full.csv",
    "outputs/milestone_5/benchmark_comparison/tables/class_level_ap50_95_full.csv",
    "outputs/milestone_5/benchmark_comparison/figures/map50_95_comparison.png",
    "outputs/milestone_5/benchmark_comparison/figures/map50_comparison.png",
    "outputs/milestone_5/benchmark_comparison/figures/inference_time_comparison.png",
    "outputs/milestone_5/benchmark_comparison/MILESTONE_5_BENCHMARK_COMPARISON.md",

    "scripts/milestone_5/00_check_local_eval_inputs.py",
    "scripts/milestone_5/01_inspect_eval_formats.py",
    "scripts/milestone_5/02_smoke_load_final_models.py",
    "scripts/milestone_5/03_run_final_kitti_validation.py",
    "scripts/milestone_5/04_create_benchmark_outputs.py",
    "scripts/milestone_5/06_final_audit.py",
]

FINAL_CHECKPOINTS = [
    "outputs/milestone_4/checkpoints/yolo/yolo_final_20260813_153831/weights/best.pt",
    "outputs/milestone_4/checkpoints/rtdetr/rtdetr_final_20260813_215051/weights/best.pt",
    "outputs/milestone_4/checkpoints/retinanet/retinanet_final_resume_if_needed_20260814_100422/best.pth",
    "outputs/milestone_4/checkpoints/faster_rcnn/faster_rcnn_final_resume_if_needed_20260814_100004/best.pth",
]

OUTPUT_DIR = PROJECT / "outputs" / "milestone_5" / "final_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 100)
    print("STEP 32 - Final audit")
    print("=" * 100)

    errors = []
    warnings = []

    for rel in REQUIRED_FILES:
        path = PROJECT / rel
        if path.exists():
            print("OK:", rel)
        else:
            print("MISSING:", rel)
            errors.append(rel)

    print()
    print("-" * 100)
    print("Checking final checkpoints exist locally")
    print("-" * 100)

    for rel in FINAL_CHECKPOINTS:
        path = PROJECT / rel
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"OK: {rel} | {size_mb:.2f} MB")
        else:
            print("MISSING:", rel)
            errors.append(rel)

    print()
    print("-" * 100)
    print("Checking final metrics")
    print("-" * 100)

    summary_path = PROJECT / "outputs/milestone_5/final_kitti_validation/tables/comparison_summary_full.json"

    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        detectors = {row["detector"] for row in rows}
        expected = {"yolo", "rtdetr", "retinanet", "faster_rcnn"}

        print("Detectors in summary:", sorted(detectors))

        if detectors != expected:
            errors.append(f"Detector mismatch in summary: {detectors}")

        for row in rows:
            print(
                row["detector"],
                "mAP50=", round(row["mAP50"], 4),
                "mAP50_95=", round(row["mAP50_95"], 4),
                "ms=", round(row["mean_inference_ms"], 2),
            )
    else:
        errors.append("Missing comparison_summary_full.json")

    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "FAILED" if errors else "PASSED",
        "errors": errors,
        "warnings": warnings,
    }

    audit_json = OUTPUT_DIR / "final_audit.json"
    audit_md = OUTPUT_DIR / "FINAL_AUDIT.md"

    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 4 + 5 Final Audit")
    md.append("")
    md.append(f"Created at: `{audit['created_at']}`")
    md.append("")
    md.append(f"Status: **{audit['status']}**")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- All final detector checkpoints should remain local artifacts unless Git LFS is configured.")
    md.append("- The final DOCX report, scripts, summary tables, and benchmark figures are intended for commit.")
    md.append("- Large files such as `.pt`, `.pth`, `.zip`, and `.jsonl` should not be committed to normal Git unless intentionally tracked with Git LFS.")
    audit_md.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Audit JSON:", audit_json)
    print("Audit MD:", audit_md)

    if errors:
        print("STEP 32 FAILED ❌")
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)

    print("STEP 32 COMPLETE ✅")
    print("Final audit passed.")
    print("=" * 100)


if __name__ == "__main__":
    main()