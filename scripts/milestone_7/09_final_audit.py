from pathlib import Path
from datetime import datetime
import json


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

OUT_DIR = PROJECT / "outputs" / "milestone_7" / "final_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_JSON = OUT_DIR / "milestone_7_final_audit.json"
AUDIT_MD = OUT_DIR / "MILESTONE_7_FINAL_AUDIT.md"


REQUIRED_FILES = [
    PROJECT / "configs" / "analysis" / "milestone_7" / "failure_case_policy.yaml",
    PROJECT / "configs" / "analysis" / "milestone_7" / "safety_error_policy.yaml",
    PROJECT / "configs" / "analysis" / "milestone_7" / "object_size_bins.yaml",
    PROJECT / "configs" / "analysis" / "milestone_7" / "README.md",

    PROJECT / "outputs" / "milestone_7" / "data_audit" / "milestone_7_input_audit.json",
    PROJECT / "outputs" / "milestone_7" / "data_audit" / "MILESTONE_7_INPUT_AUDIT.md",
    PROJECT / "outputs" / "milestone_7" / "data_audit" / "milestone_7_config_manifest.json",

    PROJECT / "outputs" / "milestone_7" / "object_size_analysis" / "object_size_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "object_size_analysis" / "small_object_failure_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "object_size_analysis" / "object_size_dataset_comparison.csv",
    PROJECT / "outputs" / "milestone_7" / "object_size_analysis" / "MILESTONE_7_OBJECT_SIZE_ANALYSIS.md",

    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_error_index.json",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_error_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_core_tp_fp_fn_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "safety_false_negative_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "top_safety_critical_images.csv",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "failure_type_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "false_positive_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "localization_error_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "class_confusion_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "duplicate_detection_summary.csv",

    PROJECT / "outputs" / "milestone_7" / "failure_cases" / "failure_case_manifest.json",
    PROJECT / "outputs" / "milestone_7" / "failure_cases" / "FAILURE_CASE_GALLERY.md",
    PROJECT / "outputs" / "milestone_7" / "failure_cases" / "panels" / "failure_case_panel_waymo.png",
    PROJECT / "outputs" / "milestone_7" / "failure_cases" / "panels" / "failure_case_panel_kitti.png",
    PROJECT / "outputs" / "milestone_7" / "failure_cases" / "panels" / "failure_case_panel_safety_vru.png",

    PROJECT / "outputs" / "milestone_7" / "deployment_tradeoff" / "deployment_suitability_summary.csv",
    PROJECT / "outputs" / "milestone_7" / "deployment_tradeoff" / "deployment_tradeoff_ranking.csv",
    PROJECT / "outputs" / "milestone_7" / "deployment_tradeoff" / "deployment_recommendations.json",
    PROJECT / "outputs" / "milestone_7" / "deployment_tradeoff" / "MILESTONE_7_DEPLOYMENT_TRADEOFF_ANALYSIS.md",

    PROJECT / "outputs" / "milestone_7" / "figures" / "milestone_7_figure_manifest.json",
    PROJECT / "outputs" / "milestone_7" / "figures" / "MILESTONE_7_FIGURES.md",
]


REQUIRED_SCRIPTS = [
    "00_validate_milestone_7_inputs.py",
    "01_create_milestone_7_configs.py",
    "02_build_detection_error_index.py",
    "03_object_size_analysis.py",
    "04_safety_false_negative_analysis.py",
    "05_failure_type_analysis.py",
    "06_create_failure_case_gallery.py",
    "07_deployment_tradeoff_analysis.py",
    "08_create_figures_and_report_bundle.py",
    "09_final_audit.py",
]


def rel(path: Path):
    return str(path.relative_to(PROJECT))


def main():
    print("=" * 100)
    print("STEP 10/10 - Milestone 7 final audit")
    print("=" * 100)

    errors = []
    warnings = []

    for script in REQUIRED_SCRIPTS:
        path = PROJECT / "scripts" / "milestone_7" / script
        if not path.exists():
            errors.append(f"Missing script: {rel(path)}")

    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing required file: {rel(path)}")

    docx_files = sorted((PROJECT / "docs" / "milestone_7").glob("*.docx"))

    if not docx_files:
        errors.append("Missing Milestone 7 DOCX report in docs/milestone_7")
    else:
        print("DOCX report found:")
        for path in docx_files:
            print("  ", rel(path))

    figure_pngs = sorted((PROJECT / "outputs" / "milestone_7" / "figures").glob("*.png"))
    panel_pngs = sorted((PROJECT / "outputs" / "milestone_7" / "failure_cases" / "panels").glob("*.png"))
    gallery_images = sorted((PROJECT / "outputs" / "milestone_7" / "failure_cases" / "images").glob("*.png"))

    if len(figure_pngs) < 10:
        errors.append(f"Expected at least 10 figure PNGs, found {len(figure_pngs)}")

    if len(panel_pngs) < 3:
        errors.append(f"Expected 3 failure-case panels, found {len(panel_pngs)}")

    if len(gallery_images) > 0:
        warnings.append(
            f"Found {len(gallery_images)} individual gallery PNGs. Keep local/regeneratable; do not commit if ignored."
        )

    large_detection_index = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_error_index.csv"
    if large_detection_index.exists():
        warnings.append("Large detection_error_index.csv exists locally. It should stay ignored, not committed.")

    bundle_zip = PROJECT / "outputs" / "milestone_7" / "report_bundle" / "milestone_7_report_source_bundle.zip"
    if bundle_zip.exists():
        warnings.append("Report source ZIP exists locally. It should stay ignored, not committed.")

    manifest_path = PROJECT / "outputs" / "milestone_7" / "failure_cases" / "failure_case_manifest.json"
    gallery_summary = {}

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gallery_summary = {
            "generated_count": manifest.get("generated_count"),
            "type_counts": manifest.get("type_counts"),
            "dataset_counts": manifest.get("dataset_counts"),
            "panels": manifest.get("panels"),
        }

    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED" if not errors else "FAILED",
        "errors": errors,
        "warnings": warnings,
        "docx_reports": [rel(p) for p in docx_files],
        "figure_png_count": len(figure_pngs),
        "failure_panel_count": len(panel_pngs),
        "individual_gallery_image_count_local": len(gallery_images),
        "gallery_summary": gallery_summary,
        "commit_policy": {
            "commit": [
                "scripts/milestone_7",
                "configs/analysis/milestone_7",
                "docs/milestone_7",
                "outputs/milestone_7 summaries, figures, panels, manifests, and audit files",
            ],
            "do_not_commit": [
                "outputs/milestone_7/safety_error_analysis/detection_error_index.csv",
                "outputs/milestone_7/report_bundle/*.zip",
                "outputs/milestone_7/failure_cases/images/*.png",
            ],
        },
    }

    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 7 Final Audit")
    md.append("")
    md.append(f"Created at: `{audit['created_at']}`")
    md.append("")
    md.append(f"Status: **{audit['status']}**")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append(f"- Figure PNGs: `{len(figure_pngs)}`")
    md.append(f"- Failure-case panels: `{len(panel_pngs)}`")
    md.append(f"- Local individual gallery images: `{len(gallery_images)}`")
    md.append("")
    md.append("## DOCX Reports")
    md.append("")
    for path in docx_files:
        md.append(f"- `{rel(path)}`")
    md.append("")
    md.append("## Warnings")
    md.append("")
    for warning in warnings:
        md.append(f"- {warning}")
    md.append("")
    md.append("## Errors")
    md.append("")
    if errors:
        for error in errors:
            md.append(f"- {error}")
    else:
        md.append("- None")
    md.append("")

    AUDIT_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("Created:", AUDIT_JSON)
    print("Created:", AUDIT_MD)

    print()
    print("Figure PNGs:", len(figure_pngs))
    print("Failure-case panels:", len(panel_pngs))
    print("Local individual gallery images:", len(gallery_images))

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print("  WARNING:", warning)

    if errors:
        print()
        print("Errors:")
        for error in errors:
            print("  ERROR:", error)

        print()
        print("STEP 10/10 FAILED ❌")
        raise SystemExit(1)

    print()
    print("STEP 10/10 AUDIT COMPLETE ✅")
    print("Milestone 7 is ready for Git safety check, commit, and push.")
    print("=" * 100)


if __name__ == "__main__":
    main()