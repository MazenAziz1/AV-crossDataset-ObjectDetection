import json
import csv
import sys
from pathlib import Path
from datetime import datetime, timezone


DETECTORS = ["yolo", "faster_rcnn", "retinanet", "rtdetr"]


def main():
    print("=" * 70)
    print("Running Final Milestone 5 Audit (KITTI In-Domain Evaluation)...")
    print("=" * 70)

    project_root = Path(__file__).resolve().parents[2]
    m4_outputs = project_root / "outputs" / "milestone_4"
    m5_outputs = project_root / "outputs" / "milestone_5"
    m5_docs = project_root / "docs" / "milestone_5"

    metrics_dir = m5_outputs / "metrics" / "kitti_validation"
    benchmarks_dir = m5_outputs / "benchmarks"
    figures_dir = m5_outputs / "figures"
    registry = m4_outputs / "manifests" / "final_checkpoint_registry.csv"
    audit_dir = m5_outputs / "final_audit"

    audit_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = audit_dir / "final_audit.json"
    issues_csv_path = audit_dir / "final_audit_issues.csv"

    issues = []
    checks = {}

    def log_issue(check_name, file_path, description, severity="ERROR"):
        issues.append({
            "check_name": check_name,
            "file_path": str(file_path),
            "issue_description": description,
            "severity": severity,
        })

    # 1. Locked checkpoint registry exists (input dependency from Milestone 4)
    if not registry.exists():
        log_issue("Checkpoint registry", registry, "final_checkpoint_registry.csv not found.")
    else:
        with open(registry) as f:
            rows = list(csv.DictReader(f))
        locked = [r for r in rows if r.get("load_test") == "PASSED"]
        if len(rows) != 4 or len(locked) != 4:
            log_issue("Checkpoint registry", registry, f"Expected 4 locked checkpoints, found {len(rows)}.")
        else:
            checks["locked_checkpoint_registry"] = "PASSED (4/4 locked)"

    # 2. KITTI evaluations (incl. operating-point metrics)
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

    # 3. Benchmarks
    missing_bench = [d for d in DETECTORS if not (benchmarks_dir / f"{d}_benchmark.json").exists()]
    if missing_bench:
        for d in missing_bench:
            log_issue("Benchmarks", benchmarks_dir / f"{d}_benchmark.json", f"Benchmark missing for {d}.")
    else:
        checks["benchmarks"] = "PASSED (4/4 benchmarks)"

    # 4. Comparison outputs
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
            log_issue("Comparison outputs", acc_csv, "Comparison CSVs do not cover all 4 detectors.")
        else:
            checks["comparison_outputs"] = "PASSED (4/4 detectors)"

    # 5. Documentation
    doc_items = {
        "evaluation protocol": m5_docs / "evaluation_protocol.md",
        "methodology md": m5_docs / "milestone_5_methodology.md",
        "docs README": m5_docs / "README.md",
        "in-domain report docx": m5_docs / "model_training_and_kitti_evaluation_draft.docx",
    }
    missing_docs = [name for name, p in doc_items.items() if not p.exists()]
    if missing_docs:
        log_issue("Documentation", m5_docs, "Missing documentation: " + ", ".join(missing_docs) + ".")
    else:
        checks["documentation"] = "PASSED"

    # 6. No Waymo evaluation artifacts in Milestone 5 outputs
    waymo_artifacts = list(m5_outputs.rglob("*waymo*"))
    if waymo_artifacts:
        log_issue("Waymo boundary", waymo_artifacts[0], f"Waymo-related artifact found in outputs: {len(waymo_artifacts)} item(s).")
    else:
        checks["waymo_evaluation_excluded"] = "PASSED"

    # Save issues CSV
    with open(issues_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "file_path", "issue_description", "severity"])
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue)

    errors = [i for i in issues if i["severity"] == "ERROR"]
    final_status = "PASSED" if not errors else "FAILED"

    summary = {
        "milestone": 5,
        "purpose": "Final audit of Milestone 5 KITTI in-domain evaluation, benchmarking, and documentation.",
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
    print(f"Final Milestone 5 Audit Status: {final_status} (checks passed: {len(checks)}, issues: {len(issues)})")
    print("=" * 70)

    sys.exit(1 if final_status == "FAILED" else 0)


if __name__ == "__main__":
    main()
