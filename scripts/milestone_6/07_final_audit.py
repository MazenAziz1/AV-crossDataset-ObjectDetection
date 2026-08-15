import json
import sys
from datetime import datetime, timezone
from pathlib import Path


DETECTORS = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]
FIGURES = [
    "kitti_vs_waymo_map50_95.png",
    "generalization_ratio_map50_95.png",
    "class_wise_degradation.png",
    "waymo_inference_time_comparison.png",
]
FORBIDDEN_EXTENSIONS = {".pt", ".pth", ".zip", ".jsonl", ".tar", ".gz"}


def main():
    print("=" * 79)
    print("Milestone 6 - Phase 9: Final audit")
    print("=" * 79)

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "outputs" / "milestone_6" / "final_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = []
    failures = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "status": "PASSED" if ok else "FAILED", "detail": detail})
        if not ok:
            failures.append(name)

    # 1. Required configs
    configs = [
        "configs/datasets/milestone_6/waymo_external_subset.yaml",
        "configs/evaluation/milestone_6/external_validation_policy.yaml",
        "configs/evaluation/milestone_6/generalization_metrics.yaml",
    ]
    missing_cfg = [c for c in configs if not (root / c).exists()]
    check("required_configs_exist", not missing_cfg, f"missing={missing_cfg}")

    # 2. Handoff validation + format inspection
    handoff = root / "outputs" / "milestone_6" / "handoff_validation"
    check("waymo_handoff_summary", (handoff / "waymo_handoff_summary.json").exists()
          and (handoff / "waymo_handoff_summary.md").exists())
    check("waymo_format_inspection", (handoff / "waymo_format_inspection.json").exists())

    # 3. Waymo metrics for all four detectors
    metrics_dir = root / "outputs" / "milestone_6" / "waymo_external_validation" / "metrics"
    missing_metrics = [d for d in DETECTORS if not (metrics_dir / f"{d}_waymo_metrics.json").exists()]
    check("waymo_metrics_all_detectors", not missing_metrics, f"missing={missing_metrics}")

    # 4. Comparison tables + summary
    ga_tables = root / "outputs" / "milestone_6" / "generalization_analysis" / "tables"
    for name in ["kitti_vs_waymo_comparison.csv", "generalization_ratio_table.csv", "class_wise_degradation.csv"]:
        check(f"table_{name}", (ga_tables / name).exists())
    check("generalization_summary_json",
          (root / "outputs" / "milestone_6" / "generalization_analysis" / "generalization_analysis_summary.json").exists())

    # 5. Figures
    figs_dir = root / "outputs" / "milestone_6" / "figures"
    missing_figs = [f for f in FIGURES if not (figs_dir / f).exists()]
    check("figures_exist", not missing_figs, f"missing={missing_figs}")

    # 6. Reports
    docs_dir = root / "docs" / "milestone_6"
    check("docx_report", (docs_dir / "Milestone_6_Waymo_External_Validation_Report.docx").exists())
    check("markdown_report", (docs_dir / "Milestone_6_Waymo_External_Validation_Report.md").exists())

    # 7. No forbidden artifacts under milestone_6 outputs (exclude empty predictions dir)
    forbidden = []
    m6_out = root / "outputs" / "milestone_6"
    for p in m6_out.rglob("*"):
        if p.is_file() and p.suffix.lower() in FORBIDDEN_EXTENSIONS:
            forbidden.append(str(p.relative_to(root)))
    check("no_forbidden_artifacts", not forbidden, f"forbidden={forbidden}")

    status = "PASSED" if not failures else "FAILED"

    audit = {
        "milestone": 6,
        "phase": 9,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "failures": failures,
    }
    (out_dir / "final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md_lines = ["# Milestone 6 Final Audit", "",
                f"- Status: **{status}**",
                f"- Timestamp: {audit['timestamp']}", ""]
    md_lines.append("| Check | Status | Detail |")
    md_lines.append("|---|---|---|")
    for c in checks:
        md_lines.append(f"| {c['name']} | {c['status']} | {c['detail']} |")
    md_lines.append("")
    (out_dir / "FINAL_AUDIT.md").write_text("\n".join(md_lines), encoding="utf-8")

    for c in checks:
        print(f"  [{c['status']}] {c['name']} {c['detail']}")
    print(f"\nFinal audit status: {status}")
    print(f"Saved: {out_dir / 'final_audit.json'}")

    if status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
