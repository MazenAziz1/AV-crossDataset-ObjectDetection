import os
import json
import csv
import sys
from pathlib import Path
from datetime import datetime, timezone


DETECTORS = ["yolo", "faster_rcnn", "retinanet", "rtdetr"]


def main():
    print("=" * 70)
    print("Running Final Milestone 4 Audit...")
    print("=" * 70)

    project_root = Path(__file__).resolve().parents[2]
    m4_outputs = project_root / "outputs" / "milestone_4"
    m4_configs = project_root / "configs" / "models" / "milestone_4"
    m4_docs = project_root / "docs" / "milestone_4"

    reports_dir = m4_outputs / "reports"
    manifests_dir = m4_outputs / "manifests"
    metrics_dir = m4_outputs / "metrics" / "kitti_validation"
    benchmarks_dir = m4_outputs / "benchmarks"
    figures_dir = m4_outputs / "figures"
    checkpoints_root = m4_outputs / "checkpoints"
    packages_dir = m4_outputs / "kaggle_packages"

    os.makedirs(reports_dir, exist_ok=True)

    report_json_path = reports_dir / "final_milestone_4_audit.json"
    issues_csv_path = reports_dir / "final_milestone_4_audit_issues.csv"

    issues = []
    checks = {}

    def log_issue(check_name, file_path, description, severity="ERROR"):
        issues.append({
            "check_name": check_name,
            "file_path": str(file_path),
            "issue_description": description,
            "severity": severity,
        })

    # --- 1. Milestone 3 handoff ---
    m3_report = m4_outputs / "reports" / "milestone_3_handoff_validation.json"
    if not m3_report.exists():
        log_issue("Milestone 3 handoff", m3_report, "milestone_3_handoff_validation.json not found.")
    else:
        with open(m3_report) as f:
            m3 = json.load(f)
        if m3.get("final_status") != "PASSED":
            log_issue("Milestone 3 handoff", m3_report, "Milestone 3 handoff status is not PASSED.")
        else:
            checks["milestone_3_handoff"] = "PASSED"

    # --- 2. Kaggle package exists ---
    if packages_dir.exists() and any(packages_dir.glob("milestone4_kaggle_training_package.*")):
        checks["kaggle_package_created"] = "PASSED"
    else:
        log_issue("Kaggle package", packages_dir, "milestone4_kaggle_training_package package not found.")

    # --- 3. Kaggle package report: Waymo exclusion ---
    pkg_report = m4_outputs / "reports" / "kaggle_training_package_report.json"
    if not pkg_report.exists():
        log_issue("Kaggle package report", pkg_report, "kaggle_training_package_report.json not found.")
    else:
        with open(pkg_report) as f:
            pr = json.load(f)
        waymo_count = pr.get("waymo_files_included", pr.get("waymo_in_package", None))
        if waymo_count is None:
            checks["waymo_exclusion"] = "PASSED (no explicit count, report present)"
        elif waymo_count == 0:
            checks["waymo_exclusion"] = "PASSED (0 Waymo files)"
        else:
            log_issue("Waymo exclusion", pkg_report, f"Waymo files included: {waymo_count}.")

    # --- 4. Compute allocation frozen ---
    if (m4_configs / "kaggle_compute_plan.yaml").exists():
        checks["compute_allocation"] = "PASSED"
    else:
        log_issue("Compute allocation", m4_configs / "kaggle_compute_plan.yaml", "kaggle_compute_plan.yaml not found.")

    # --- 5. Four checkpoints present ---
    missing_ckpt = [d for d in DETECTORS if not (checkpoints_root / d / "final" / "best.pt").exists()]
    if missing_ckpt:
        for d in missing_ckpt:
            log_issue("Checkpoints", checkpoints_root / d / "final" / "best.pt", f"best.pt missing for {d}.")
    else:
        checks["four_training_runs"] = "PASSED (4/4 best.pt present)"

    # --- 6. Locked checkpoint registry ---
    registry = manifests_dir / "final_checkpoint_registry.csv"
    if not registry.exists():
        log_issue("Checkpoint registry", registry, "final_checkpoint_registry.csv not found.")
    else:
        with open(registry) as f:
            rows = list(csv.DictReader(f))
        locked = [r for r in rows if r.get("load_test") == "PASSED"]
        if len(rows) != 4:
            log_issue("Checkpoint registry", registry, f"Expected 4 locked checkpoints, found {len(rows)}.")
        elif len(locked) != 4:
            log_issue("Checkpoint registry", registry, "Not all checkpoints passed load test.")
        else:
            checks["locked_checkpoints"] = "PASSED (4/4 locked)"

    # --- 7. RT-DETR resume chain ---
    resume_report = reports_dir / "rtdetr_resume_chain_validation.json"
    if not resume_report.exists():
        log_issue("RT-DETR resume chain", resume_report, "rtdetr_resume_chain_validation.json not found.")
    else:
        with open(resume_report) as f:
            rr = json.load(f)
        if rr.get("status") != "PASSED":
            log_issue("RT-DETR resume chain", resume_report, f"Resume chain status is {rr.get('status')}.")
        else:
            checks["rtdetr_resume_chain"] = "PASSED"

    # --- 8. KITTI evaluations (incl. operating-point metrics) ---
    missing_metrics = [d for d in DETECTORS if not (metrics_dir / f"{d}_metrics.json").exists()]
    if missing_metrics:
        for d in missing_metrics:
            log_issue("KITTI evaluation", metrics_dir / f"{d}_metrics.json", f"Metrics missing for {d}.")
    else:
        missing_op = []
        for d in DETECTORS:
            with open(metrics_dir / f"{d}_metrics.json") as f:
                md = json.load(f)
            if "operating_point" not in md:
                missing_op.append(d)
        if missing_op:
            log_issue("Operating point", metrics_dir, f"Operating-point metrics missing for: {', '.join(missing_op)}.")
        else:
            checks["kitti_evaluation"] = "PASSED (4/4 metrics, incl. operating point)"

    # --- 9. Benchmarks ---
    missing_bench = [d for d in DETECTORS if not (benchmarks_dir / f"{d}_benchmark.json").exists()]
    if missing_bench:
        for d in missing_bench:
            log_issue("Benchmarks", benchmarks_dir / f"{d}_benchmark.json", f"Benchmark missing for {d}.")
    else:
        checks["benchmarks"] = "PASSED (4/4 benchmarks)"

    # --- 10. Comparison outputs ---
    acc_csv = figures_dir / "accuracy_comparison.csv"
    eff_csv = figures_dir / "efficiency_comparison.csv"
    if not (acc_csv.exists() and eff_csv.exists()):
        log_issue("Comparison outputs", figures_dir, "accuracy_comparison.csv or efficiency_comparison.csv missing.")
    else:
        with open(acc_csv) as f:
            acc_dets = {r["detector"] for r in csv.DictReader(f)}
        with open(eff_csv) as f:
            eff_dets = {r["detector"] for r in csv.DictReader(f)}
        if acc_dets != set(DETECTORS) or eff_dets != set(DETECTORS):
            log_issue("Comparison outputs", acc_csv, f"Comparison CSVs do not cover all 4 detectors (acc={sorted(acc_dets)}, eff={sorted(eff_dets)}).")
        else:
            checks["comparison_outputs"] = "PASSED (4/4 detectors)"

    # --- 11. Documentation ---
    doc_items = {
        "outputs README": m4_outputs / "README.md",
        "methodology md": m4_docs / "milestone_4_methodology.md",
        "docs README": m4_docs / "README.md",
        "draft docx": m4_docs / "model_training_and_kitti_evaluation_draft.docx",
    }
    missing_docs = [name for name, p in doc_items.items() if not p.exists()]
    if missing_docs:
        log_issue("Documentation", m4_docs, "Missing documentation: " + ", ".join(missing_docs) + ".")
    else:
        checks["documentation"] = "PASSED"

    # --- 12. Training metrics present (training vs validation comparison) ---
    train_metrics = m4_outputs / "metrics" / "training_metrics.json"
    if train_metrics.exists():
        checks["training_metrics"] = "PASSED"
    else:
        log_issue("Training metrics", train_metrics, "metrics/training_metrics.json not found.")

    # --- 13. No Waymo evaluation outputs ---
    waymo_artifacts = list(m4_outputs.rglob("*waymo*"))
    if waymo_artifacts:
        log_issue("Waymo evaluation", waymo_artifacts[0], f"Waymo-related artifact found in outputs: {len(waymo_artifacts)} item(s).")
    else:
        checks["waymo_evaluation_excluded"] = "PASSED"

    # --- Save issues CSV ---
    with open(issues_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "file_path", "issue_description", "severity"])
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue)

    errors = [i for i in issues if i["severity"] == "ERROR"]
    final_status = "PASSED" if not errors else "FAILED"

    summary = {
        "milestone": "4 + 5",
        "step": 32,
        "purpose": "Final audit of Milestone 4 training, import, evaluation, benchmarking, and documentation.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_performed": checks,
        "issues_found": len(issues),
        "errors_count": len(errors),
        "warnings_count": len([i for i in issues if i["severity"] == "WARNING"]),
        "final_status": final_status,
    }
    with open(report_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Audit report JSON saved to: {report_json_path}")
    print(f"Audit issues CSV saved to: {issues_csv_path}")
    print("-" * 70)
    print(f"Final Milestone 4 Audit Status: {final_status} (checks passed: {len(checks)}, issues: {len(issues)})")
    print("=" * 70)

    sys.exit(1 if final_status == "FAILED" else 0)


if __name__ == "__main__":
    main()
