import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common


def main():
    print("=" * 79)
    print("Milestone 7 - Step 10: Final audit")
    print("=" * 79)

    out_dir = common.M7_OUT / "final_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = []
    failures = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "status": "PASSED" if ok else "FAILED", "detail": detail})
        if not ok:
            failures.append(name)

    # 1. Input audit passed
    ia = common.M7_OUT / "data_audit" / "milestone_7_input_audit.json"
    if ia.exists():
        status = json.load(open(ia, encoding="utf-8"))["status"]
        check("input_audit_passed", status == "PASSED", status)
    else:
        check("input_audit_passed", False, "missing")

    # 2. Configs exist
    cfg = common.M7_CFG
    missing_cfg = [f for f in ["failure_case_policy.yaml", "safety_error_policy.yaml", "object_size_bins.yaml", "README.md"]
                   if not (cfg / f).exists()]
    check("configs_exist", not missing_cfg, f"missing={missing_cfg}")

    # 3. Detection error index
    idx_csv = common.M7_OUT / "safety_error_analysis" / "detection_error_index.csv"
    idx_json = common.M7_OUT / "safety_error_analysis" / "detection_error_index.json"
    idx_detail = ""
    if idx_json.exists():
        idx_meta = json.load(open(idx_json, encoding="utf-8"))
        idx_detail = (f"{idx_meta['total_records']} records, "
                      f"{len(idx_meta['counts_by_dataset_detector_error'])} detector-dataset combos")
    check("detection_error_index", idx_csv.exists() and idx_json.exists(), idx_detail)

    # 4. Object-size analysis
    os_csv = common.M7_OUT / "object_size_analysis" / "object_size_summary.csv"
    os_small = common.M7_OUT / "object_size_analysis" / "small_object_failure_summary.csv"
    os_detail = ""
    if os_csv.exists():
        os_detail = f"{len(pd.read_csv(os_csv))} size-bin rows"
        if os_small.exists():
            os_detail += f", {len(pd.read_csv(os_small))} small-object rows"
    check("object_size_analysis", os_csv.exists() and os_small.exists(), os_detail)

    # 5. Safety FN analysis
    sfn_csv = common.M7_OUT / "safety_error_analysis" / "safety_false_negative_summary.csv"
    sfn_top = common.M7_OUT / "safety_error_analysis" / "top_safety_critical_images.csv"
    sfn_detail = ""
    if sfn_csv.exists():
        sfn_detail = f"{len(pd.read_csv(sfn_csv))} summary rows"
        if sfn_top.exists():
            sfn_detail += f", {len(pd.read_csv(sfn_top))} top safety-critical images"
    check("safety_fn_analysis", sfn_csv.exists() and sfn_top.exists(), sfn_detail)

    # 6. Failure-type analysis
    ft_csv = common.M7_OUT / "safety_error_analysis" / "failure_type_summary.csv"
    cc_csv = common.M7_OUT / "safety_error_analysis" / "class_confusion_summary.csv"
    ft_detail = ""
    if ft_csv.exists():
        ft_detail = f"{len(pd.read_csv(ft_csv))} failure-type rows"
        if cc_csv.exists():
            ft_detail += f", {len(pd.read_csv(cc_csv))} class-confusion rows"
    check("failure_type_analysis", ft_csv.exists() and cc_csv.exists(), ft_detail)

    # 7. Failure gallery
    fg_manifest = common.M7_OUT / "failure_cases" / "failure_case_manifest.json"
    fg_kitti = common.M7_OUT / "failure_cases" / "panels" / "failure_case_panel_kitti.png"
    fg_waymo = common.M7_OUT / "failure_cases" / "panels" / "failure_case_panel_waymo.png"
    fg_detail = ""
    if fg_manifest.exists():
        fg_cases = json.load(open(fg_manifest, encoding="utf-8"))
        fg_detail = f"{len(fg_cases)} failure cases, 2 panels"
    check("failure_gallery", fg_manifest.exists() and fg_kitti.exists() and fg_waymo.exists(), fg_detail)

    # 8. Deployment trade-off
    dt_csv = common.M7_OUT / "deployment_tradeoff" / "deployment_suitability_table.csv"
    dt_md = common.M7_OUT / "deployment_tradeoff" / "deployment_recommendations.md"
    dt_detail = ""
    if dt_csv.exists():
        dt_detail = f"{len(pd.read_csv(dt_csv))} detectors in suitability table"
    check("deployment_tradeoff", dt_csv.exists() and dt_md.exists(), dt_detail)

    # 9. Figures (6)
    figs = ["small_medium_large_recall.png", "small_object_failure_rate.png",
            "pedestrian_cyclist_false_negative_rate.png", "failure_type_breakdown.png",
            "class_confusion_heatmap.png", "deployment_suitability_comparison.png"]
    missing_figs = [f for f in figs if not (common.M7_OUT / "figures" / f).exists()]
    check("figures_exist", not missing_figs, f"missing={missing_figs}")

    # 10. DOCX report
    docx_path = common.PROJECT_ROOT / "docs" / "milestone_7" / "Milestone_7_Robustness_Failure_Case_Safety_Report.docx"
    docx_detail = ""
    if docx_path.exists():
        docx_detail = "Milestone_7_Robustness_Failure_Case_Safety_Report.docx present"
    check("docx_report", docx_path.exists(), docx_detail)

    # 11-14. Representation checks (index covers 4 detectors, both datasets, safety classes)
    rep_ok = False
    rep_detail = ""
    if idx_csv.exists():
        idx = pd.read_csv(idx_csv)
        dets = set(idx["detector"])
        dsets = set(idx["dataset"])
        classes = set(idx["class_name"])
        rep_ok = dets == set(common.DETECTORS) and dsets == {"kitti", "waymo"} and {"Pedestrian", "Cyclist"} <= classes
        rep_detail = f"detectors={sorted(dets)} datasets={sorted(dsets)} classes={sorted(classes)}"
    check("detectors_datasets_classes_represented", rep_ok, rep_detail)

    status = "PASSED" if not failures else "FAILED"
    audit = {
        "milestone": 7,
        "step": 10,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "failures": failures,
    }
    (out_dir / "final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md = ["# Milestone 7 Final Audit", "", f"- Status: **{status}**",
          f"- Timestamp: {audit['timestamp']}", ""]
    md.append("| Check | Status | Detail |")
    md.append("|---|---|---|")
    for c in checks:
        md.append(f"| {c['name']} | {c['status']} | {c['detail']} |")
    md.append("")
    (out_dir / "FINAL_AUDIT.md").write_text("\n".join(md), encoding="utf-8")

    for c in checks:
        print(f"  [{c['status']}] {c['name']} {c['detail']}")
    print(f"\nFinal audit status: {status}")

    if status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
