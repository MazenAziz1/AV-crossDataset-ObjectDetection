from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import subprocess
import sys

PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

OUTPUT_DIR = PROJECT / "outputs" / "milestone_6" / "final_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_DETECTORS = {"yolo", "rtdetr", "retinanet", "faster_rcnn"}

REQUIRED_FILES = [
    # Report
    "docs/milestone_6/Milestone_6_Waymo_External_Validation_Report.docx",

    # Scripts
    "scripts/milestone_6/00_validate_waymo_handoff.py",
    "scripts/milestone_6/01_create_milestone_6_configs.py",
    "scripts/milestone_6/02_smoke_test_waymo_models.py",
    "scripts/milestone_6/03_run_waymo_external_validation.py",
    "scripts/milestone_6/04_create_generalization_analysis.py",
    "scripts/milestone_6/05_create_waymo_generalization_figures.py",
    "scripts/milestone_6/07_final_audit.py",

    # Configs
    "configs/datasets/milestone_6/waymo_external_subset.yaml",
    "configs/evaluation/milestone_6/external_validation_policy.yaml",
    "configs/evaluation/milestone_6/generalization_metrics.yaml",
    "configs/evaluation/milestone_6/README.md",

    # Handoff validation
    "outputs/milestone_6/handoff_validation/waymo_handoff_summary.json",
    "outputs/milestone_6/handoff_validation/waymo_handoff_summary.md",

    # Smoke test
    "outputs/milestone_6/waymo_external_validation/smoke_test_summary.json",
    "outputs/milestone_6/waymo_external_validation/smoke_test_summary.md",

    # Waymo metrics
    "outputs/milestone_6/waymo_external_validation/metrics/yolo_waymo_metrics.json",
    "outputs/milestone_6/waymo_external_validation/metrics/rtdetr_waymo_metrics.json",
    "outputs/milestone_6/waymo_external_validation/metrics/retinanet_waymo_metrics.json",
    "outputs/milestone_6/waymo_external_validation/metrics/faster_rcnn_waymo_metrics.json",

    # Waymo summary tables
    "outputs/milestone_6/waymo_external_validation/tables/waymo_external_summary.csv",
    "outputs/milestone_6/waymo_external_validation/tables/waymo_external_summary.json",
    "outputs/milestone_6/waymo_external_validation/waymo_external_validation_run_metadata.json",

    # Generalization analysis
    "outputs/milestone_6/generalization_analysis/tables/kitti_vs_waymo_comparison.csv",
    "outputs/milestone_6/generalization_analysis/tables/kitti_vs_waymo_comparison.json",
    "outputs/milestone_6/generalization_analysis/tables/generalization_ratio_table.csv",
    "outputs/milestone_6/generalization_analysis/tables/generalization_ratio_table.json",
    "outputs/milestone_6/generalization_analysis/tables/class_wise_degradation.csv",
    "outputs/milestone_6/generalization_analysis/tables/class_wise_degradation.json",
    "outputs/milestone_6/generalization_analysis/tables/largest_class_degradation_by_detector.csv",
    "outputs/milestone_6/generalization_analysis/tables/largest_class_degradation_by_detector.json",
    "outputs/milestone_6/generalization_analysis/generalization_analysis_summary.json",
    "outputs/milestone_6/generalization_analysis/MILESTONE_6_GENERALIZATION_ANALYSIS.md",

    # Figures
    "outputs/milestone_6/figures/kitti_vs_waymo_map50_95.png",
    "outputs/milestone_6/figures/kitti_vs_waymo_map50.png",
    "outputs/milestone_6/figures/generalization_ratio_map50_95.png",
    "outputs/milestone_6/figures/generalization_ratio_map50.png",
    "outputs/milestone_6/figures/class_wise_degradation_ap50_95.png",
    "outputs/milestone_6/figures/class_wise_generalization_ratio_ap50_95.png",
    "outputs/milestone_6/figures/waymo_inference_time_comparison.png",
    "outputs/milestone_6/figures/largest_class_degradation_by_detector.png",
    "outputs/milestone_6/figures/milestone_6_figure_manifest.json",
    "outputs/milestone_6/figures/MILESTONE_6_FIGURES.md",

    # Required M5/M4 inputs
    "outputs/milestone_5/final_kitti_validation/tables/comparison_summary_full.csv",
    "outputs/milestone_4/locked_final_checkpoints/final_checkpoint_registry.json",
]


def run_cmd(command):
    result = subprocess.run(
        command,
        shell=True,
        cwd=PROJECT,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def load_json(rel_path):
    path = PROJECT / rel_path
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    print("=" * 100)
    print("STEP 8/10 - Milestone 6 final audit")
    print("=" * 100)

    errors = []
    warnings = []

    print("Checking required files...")
    for rel in REQUIRED_FILES:
        path = PROJECT / rel
        if path.exists():
            print("OK:", rel)
        else:
            print("MISSING:", rel)
            errors.append(f"Missing required file: {rel}")

    report_path = PROJECT / "docs/milestone_6/Milestone_6_Waymo_External_Validation_Injection_Ready_Report.docx"
    if report_path.exists():
        report_size_mb = report_path.stat().st_size / (1024 * 1024)
        print(f"Report size: {report_size_mb:.2f} MB")
        if report_size_mb < 0.2:
            warnings.append("DOCX report exists but looks unusually small.")

    print()
    print("Checking handoff summary...")
    handoff_path = PROJECT / "outputs/milestone_6/handoff_validation/waymo_handoff_summary.json"
    if handoff_path.exists():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        if handoff.get("status") != "PASSED":
            errors.append("Waymo handoff status is not PASSED.")
        if handoff.get("num_images") != 996:
            warnings.append(f"Expected 996 Waymo images, found {handoff.get('num_images')}.")
        if handoff.get("num_labels") != 996:
            warnings.append(f"Expected 996 Waymo labels, found {handoff.get('num_labels')}.")
        if handoff.get("num_matched_pairs") != 996:
            errors.append(f"Expected 996 matched pairs, found {handoff.get('num_matched_pairs')}.")

    print("Checking smoke test...")
    smoke_path = PROJECT / "outputs/milestone_6/waymo_external_validation/smoke_test_summary.json"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASSED":
            errors.append("Smoke test status is not PASSED.")
        detector_keys = set(smoke.get("detectors", {}).keys())
        if detector_keys != EXPECTED_DETECTORS:
            errors.append(f"Smoke detector mismatch: {detector_keys}")

    print("Checking Waymo external validation table...")
    waymo_csv = PROJECT / "outputs/milestone_6/waymo_external_validation/tables/waymo_external_summary.csv"
    if waymo_csv.exists():
        waymo_df = pd.read_csv(waymo_csv)
        detectors = set(waymo_df["detector"].tolist())
        if detectors != EXPECTED_DETECTORS:
            errors.append(f"Waymo detector mismatch: {detectors}")
        if not (waymo_df["num_images"] == 996).all():
            errors.append("Not all Waymo summary rows have 996 images.")
        metric_cols = ["mAP50", "mAP50_95", "mean_inference_ms"]
        if waymo_df[metric_cols].isna().any().any():
            errors.append("Waymo summary contains NaN metric values.")

    print("Checking generalization ratio table...")
    ratio_csv = PROJECT / "outputs/milestone_6/generalization_analysis/tables/generalization_ratio_table.csv"
    if ratio_csv.exists():
        ratio_df = pd.read_csv(ratio_csv)
        detectors = set(ratio_df["detector"].tolist())
        if detectors != EXPECTED_DETECTORS:
            errors.append(f"Generalization detector mismatch: {detectors}")
        if ratio_df["mAP50_95_generalization_ratio"].isna().any():
            errors.append("Generalization ratio table contains NaN values.")
        if (ratio_df["mAP50_95_generalization_ratio"] <= 0).any():
            errors.append("Generalization ratio table contains non-positive values.")

    print("Checking raw prediction files...")
    prediction_dir = PROJECT / "outputs/milestone_6/waymo_external_validation/predictions"
    prediction_files = list(prediction_dir.glob("*.jsonl")) if prediction_dir.exists() else []
    if prediction_files:
        warnings.append(
            f"{len(prediction_files)} raw prediction JSONL files exist locally. "
            "Do not commit them unless intentionally required."
        )

    print("Checking git status...")
    git_status = run_cmd("git status --short")

    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "FAILED" if errors else "PASSED",
        "errors": errors,
        "warnings": warnings,
        "git_status_short": git_status,
        "expected_detectors": sorted(EXPECTED_DETECTORS),
    }

    audit_json = OUTPUT_DIR / "final_audit.json"
    audit_md = OUTPUT_DIR / "FINAL_AUDIT.md"

    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 6 Final Audit")
    md.append("")
    md.append(f"Created at: `{audit['created_at']}`")
    md.append("")
    md.append(f"Status: **{audit['status']}**")
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
    md.append("## Git Status")
    md.append("")
    md.append("```text")
    md.append(git_status["stdout"] or "(clean)")
    md.append("```")
    md.append("")

    audit_md.write_text("\n".join(md), encoding="utf-8")

    print()
    print("Audit JSON:", audit_json)
    print("Audit MD:", audit_md)

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
        print("STEP 8/10 FAILED ❌")
        raise SystemExit(1)

    print()
    print("STEP 8/10 COMPLETE ✅")
    print("Milestone 6 final audit passed.")
    print("=" * 100)


if __name__ == "__main__":
    main()