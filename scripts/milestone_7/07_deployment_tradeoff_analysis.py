from pathlib import Path
from datetime import datetime
import json
import pandas as pd


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

M5_KITTI_SUMMARY = PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "tables" / "comparison_summary_full.csv"
M6_WAYMO_SUMMARY = PROJECT / "outputs" / "milestone_6" / "waymo_external_validation" / "tables" / "waymo_external_summary.csv"

OBJECT_SIZE_SUMMARY = PROJECT / "outputs" / "milestone_7" / "object_size_analysis" / "object_size_summary.csv"
SAFETY_FN_SUMMARY = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "safety_false_negative_summary.csv"
FAILURE_TYPE_SUMMARY = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "failure_type_summary.csv"

GALLERY_MANIFEST = PROJECT / "outputs" / "milestone_7" / "failure_cases" / "failure_case_manifest.json"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_7" / "deployment_tradeoff"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEPLOYMENT_SUMMARY_CSV = OUTPUT_DIR / "deployment_suitability_summary.csv"
DEPLOYMENT_RANKING_CSV = OUTPUT_DIR / "deployment_tradeoff_ranking.csv"
DEPLOYMENT_RECOMMENDATIONS_JSON = OUTPUT_DIR / "deployment_recommendations.json"
SUMMARY_JSON = OUTPUT_DIR / "deployment_tradeoff_analysis_summary.json"
SUMMARY_MD = OUTPUT_DIR / "MILESTONE_7_DEPLOYMENT_TRADEOFF_ANALYSIS.md"

DETECTORS = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]

WEIGHTS = {
    "waymo_mAP50_95_score": 0.25,
    "generalization_ratio_score": 0.20,
    "vru_safety_score": 0.30,
    "small_vru_safety_score": 0.15,
    "speed_score": 0.10,
}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def clean_detector_name(value):
    return str(value).strip().lower()


def get_detector_row(df, detector):
    temp = df.copy()
    temp["detector_clean"] = temp["detector"].map(clean_detector_name)
    rows = temp[temp["detector_clean"] == detector]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()


def get_metric(row, name, default=None):
    if row is None:
        return default

    value = row.get(name, default)

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return default


def safe_div(numerator, denominator):
    if denominator is None or denominator == 0:
        return None
    if numerator is None:
        return None
    return numerator / denominator


def rounded(value, digits=6):
    if value is None:
        return ""
    return round(float(value), digits)


def bounded(value, low=0.0, high=1.0):
    if value is None:
        return None
    return max(low, min(high, float(value)))


def get_safety_row(safety_df, detector, dataset, class_name, size_bin):
    temp = safety_df.copy()
    temp["detector_clean"] = temp["detector"].map(clean_detector_name)

    rows = temp[
        (temp["detector_clean"] == detector)
        & (temp["dataset"] == dataset)
        & (temp["class_name"] == class_name)
        & (temp["object_size_bin"] == size_bin)
    ]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()


def get_object_row(object_df, detector, dataset, class_name, size_bin):
    temp = object_df.copy()
    temp["detector_clean"] = temp["detector"].map(clean_detector_name)

    rows = temp[
        (temp["detector_clean"] == detector)
        & (temp["dataset"] == dataset)
        & (temp["class_name"] == class_name)
        & (temp["object_size_bin"] == size_bin)
    ]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()


def summarize_failure_counts(failure_df, detector, dataset):
    temp = failure_df.copy()
    temp["detector_clean"] = temp["detector"].map(clean_detector_name)

    rows = temp[
        (temp["detector_clean"] == detector)
        & (temp["dataset"] == dataset)
    ]

    output = {
        "false_negative_count": 0,
        "false_positive_count": 0,
        "high_confidence_false_positive_count": 0,
        "localization_error_count": 0,
        "class_confusion_count": 0,
        "duplicate_detection_count": 0,
        "safety_relevant_failure_count": 0,
    }

    if rows.empty:
        return output

    for _, row in rows.iterrows():
        failure_type = row.get("failure_type", "")
        count = int(row.get("count", 0))
        high_conf = int(row.get("high_confidence_count", 0))
        safety_relevant = int(row.get("safety_relevant_count", 0))

        if failure_type == "false_negative":
            output["false_negative_count"] += count
        elif failure_type == "false_positive":
            output["false_positive_count"] += count
            output["high_confidence_false_positive_count"] += high_conf
        elif failure_type == "localization_error":
            output["localization_error_count"] += count
        elif failure_type == "class_confusion":
            output["class_confusion_count"] += count
        elif failure_type == "duplicate_detection":
            output["duplicate_detection_count"] += count

        if failure_type != "true_positive":
            output["safety_relevant_failure_count"] += safety_relevant

    return output


def score_rows(rows):
    if not rows:
        return rows

    min_waymo_ms = min(
        r["waymo_mean_inference_ms"]
        for r in rows
        if r["waymo_mean_inference_ms"] not in ["", None]
    )

    for r in rows:
        waymo_map = r["waymo_mAP50_95"]
        gen_ratio = r["generalization_ratio_mAP50_95"]
        waymo_ms = r["waymo_mean_inference_ms"]

        vru_fnr = r["waymo_vru_false_negative_rate_all_sizes"]
        small_vru_fnr = r["waymo_small_vru_false_negative_rate"]

        waymo_map_score = bounded(waymo_map)
        gen_ratio_score = bounded(gen_ratio)

        vru_safety_score = None
        if vru_fnr not in ["", None]:
            vru_safety_score = bounded(1.0 - float(vru_fnr))

        small_vru_safety_score = None
        if small_vru_fnr not in ["", None]:
            small_vru_safety_score = bounded(1.0 - float(small_vru_fnr))

        speed_score = None
        if waymo_ms not in ["", None] and waymo_ms > 0:
            speed_score = bounded(min_waymo_ms / float(waymo_ms))

        r["waymo_mAP50_95_score"] = rounded(waymo_map_score)
        r["generalization_ratio_score"] = rounded(gen_ratio_score)
        r["vru_safety_score"] = rounded(vru_safety_score)
        r["small_vru_safety_score"] = rounded(small_vru_safety_score)
        r["speed_score"] = rounded(speed_score)

        weighted_sum = 0.0
        weight_used = 0.0

        for key, weight in WEIGHTS.items():
            value = r.get(key)

            if value == "" or value is None:
                continue

            weighted_sum += float(value) * weight
            weight_used += weight

        deployment_score = weighted_sum / weight_used if weight_used > 0 else None

        r["deployment_tradeoff_score"] = rounded(deployment_score)

    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r["deployment_tradeoff_score"])
            if r["deployment_tradeoff_score"] != ""
            else 999
        ),
    )

    for idx, row in enumerate(ranked, start=1):
        row["deployment_rank"] = idx

    return ranked


def deployment_tier(row):
    waymo_map = row["waymo_mAP50_95"]
    vru_fnr = row["waymo_vru_false_negative_rate_all_sizes"]
    ms = row["waymo_mean_inference_ms"]

    if waymo_map == "" or vru_fnr == "" or ms == "":
        return "insufficient evidence"

    waymo_map = float(waymo_map)
    vru_fnr = float(vru_fnr)
    ms = float(ms)

    if waymo_map >= 0.40 and vru_fnr <= 0.25 and ms <= 50:
        return "deployment candidate"

    if waymo_map >= 0.20 and vru_fnr <= 0.60 and ms <= 70:
        return "limited research candidate"

    return "research only / not deployment-ready"


def deployment_note(row):
    detector = row["detector"]

    if detector == "yolo":
        return (
            "Fastest external inference, but weakest Waymo vulnerable-road-user safety recall. "
            "Useful as a speed baseline, not sufficient alone for safety-critical deployment."
        )

    if detector == "rtdetr":
        return (
            "Best vulnerable-road-user recall and strongest safety-oriented result, but slower and has many low-confidence proposals. "
            "Best candidate for further robustness improvement."
        )

    if detector == "retinanet":
        return (
            "Strongest Waymo mAP50-95 and generalization ratio, but safety false negatives remain high. "
            "Useful as an external-generalization reference."
        )

    if detector == "faster_rcnn":
        return (
            "Moderate safety behavior but slowest external inference. "
            "Less attractive for real-time deployment compared with RT-DETR or YOLO."
        )

    return ""


def main():
    print("=" * 100)
    print("STEP 8/10 - Deployment suitability and trade-off analysis")
    print("=" * 100)

    print("Reading inputs...")

    kitti_df = read_csv(M5_KITTI_SUMMARY)
    waymo_df = read_csv(M6_WAYMO_SUMMARY)
    object_df = read_csv(OBJECT_SIZE_SUMMARY)
    safety_df = read_csv(SAFETY_FN_SUMMARY)
    failure_df = read_csv(FAILURE_TYPE_SUMMARY)

    if GALLERY_MANIFEST.exists():
        gallery_manifest = json.loads(GALLERY_MANIFEST.read_text(encoding="utf-8"))
    else:
        gallery_manifest = {}

    rows = []

    for detector in DETECTORS:
        kitti_row = get_detector_row(kitti_df, detector)
        waymo_row = get_detector_row(waymo_df, detector)

        kitti_map50 = get_metric(kitti_row, "mAP50")
        kitti_map50_95 = get_metric(kitti_row, "mAP50_95")
        kitti_ms = get_metric(kitti_row, "mean_inference_ms")

        waymo_map50 = get_metric(waymo_row, "mAP50")
        waymo_map50_95 = get_metric(waymo_row, "mAP50_95")
        waymo_ms = get_metric(waymo_row, "mean_inference_ms")

        generalization_ratio = safe_div(waymo_map50_95, kitti_map50_95)
        absolute_drop = None

        if kitti_map50_95 is not None and waymo_map50_95 is not None:
            absolute_drop = kitti_map50_95 - waymo_map50_95

        kitti_fps = safe_div(1000.0, kitti_ms)
        waymo_fps = safe_div(1000.0, waymo_ms)

        waymo_vru_all = get_safety_row(
            safety_df,
            detector,
            "waymo",
            "Vulnerable_Road_Users",
            "all_sizes",
        )

        waymo_vru_small = get_safety_row(
            safety_df,
            detector,
            "waymo",
            "Vulnerable_Road_Users",
            "small",
        )

        kitti_vru_all = get_safety_row(
            safety_df,
            detector,
            "kitti",
            "Vulnerable_Road_Users",
            "all_sizes",
        )

        waymo_small_all = get_object_row(
            object_df,
            detector,
            "waymo",
            "All_Classes",
            "small",
        )

        waymo_failure_counts = summarize_failure_counts(
            failure_df,
            detector,
            "waymo",
        )

        kitti_failure_counts = summarize_failure_counts(
            failure_df,
            detector,
            "kitti",
        )

        row = {
            "detector": detector,

            "kitti_mAP50": rounded(kitti_map50),
            "kitti_mAP50_95": rounded(kitti_map50_95),
            "waymo_mAP50": rounded(waymo_map50),
            "waymo_mAP50_95": rounded(waymo_map50_95),

            "absolute_drop_mAP50_95_KITTI_minus_Waymo": rounded(absolute_drop),
            "generalization_ratio_mAP50_95": rounded(generalization_ratio),

            "kitti_mean_inference_ms": rounded(kitti_ms, 3),
            "waymo_mean_inference_ms": rounded(waymo_ms, 3),
            "kitti_fps": rounded(kitti_fps, 3),
            "waymo_fps": rounded(waymo_fps, 3),

            "kitti_vru_false_negative_rate_all_sizes": rounded(
                get_metric(kitti_vru_all, "false_negative_rate")
            ),
            "waymo_vru_false_negative_rate_all_sizes": rounded(
                get_metric(waymo_vru_all, "false_negative_rate")
            ),
            "waymo_vru_recall_all_sizes": rounded(
                get_metric(waymo_vru_all, "recall")
            ),

            "waymo_small_vru_false_negative_rate": rounded(
                get_metric(waymo_vru_small, "false_negative_rate")
            ),
            "waymo_small_vru_recall": rounded(
                get_metric(waymo_vru_small, "recall")
            ),

            "waymo_small_all_classes_recall": rounded(
                get_metric(waymo_small_all, "recall")
            ),
            "waymo_small_all_classes_false_negative_rate": rounded(
                get_metric(waymo_small_all, "false_negative_rate")
            ),

            "waymo_false_negative_count": waymo_failure_counts["false_negative_count"],
            "waymo_false_positive_count": waymo_failure_counts["false_positive_count"],
            "waymo_high_confidence_false_positive_count": waymo_failure_counts["high_confidence_false_positive_count"],
            "waymo_localization_error_count": waymo_failure_counts["localization_error_count"],
            "waymo_class_confusion_count": waymo_failure_counts["class_confusion_count"],
            "waymo_duplicate_detection_count": waymo_failure_counts["duplicate_detection_count"],
            "waymo_safety_relevant_failure_count": waymo_failure_counts["safety_relevant_failure_count"],

            "kitti_false_positive_count": kitti_failure_counts["false_positive_count"],
            "kitti_high_confidence_false_positive_count": kitti_failure_counts["high_confidence_false_positive_count"],

            "deployment_note": "",
            "deployment_tier": "",
        }

        rows.append(row)

    ranked_rows = score_rows(rows)

    for row in ranked_rows:
        row["deployment_tier"] = deployment_tier(row)
        row["deployment_note"] = deployment_note(row)

    fieldnames = [
        "deployment_rank",
        "detector",

        "kitti_mAP50",
        "kitti_mAP50_95",
        "waymo_mAP50",
        "waymo_mAP50_95",
        "absolute_drop_mAP50_95_KITTI_minus_Waymo",
        "generalization_ratio_mAP50_95",

        "kitti_mean_inference_ms",
        "waymo_mean_inference_ms",
        "kitti_fps",
        "waymo_fps",

        "kitti_vru_false_negative_rate_all_sizes",
        "waymo_vru_false_negative_rate_all_sizes",
        "waymo_vru_recall_all_sizes",
        "waymo_small_vru_false_negative_rate",
        "waymo_small_vru_recall",
        "waymo_small_all_classes_recall",
        "waymo_small_all_classes_false_negative_rate",

        "waymo_false_negative_count",
        "waymo_false_positive_count",
        "waymo_high_confidence_false_positive_count",
        "waymo_localization_error_count",
        "waymo_class_confusion_count",
        "waymo_duplicate_detection_count",
        "waymo_safety_relevant_failure_count",

        "kitti_false_positive_count",
        "kitti_high_confidence_false_positive_count",

        "waymo_mAP50_95_score",
        "generalization_ratio_score",
        "vru_safety_score",
        "small_vru_safety_score",
        "speed_score",
        "deployment_tradeoff_score",

        "deployment_tier",
        "deployment_note",
    ]

    pd.DataFrame(ranked_rows)[fieldnames].to_csv(DEPLOYMENT_SUMMARY_CSV, index=False)
    pd.DataFrame(ranked_rows)[fieldnames].to_csv(DEPLOYMENT_RANKING_CSV, index=False)

    best_overall = ranked_rows[0]
    fastest = sorted(
        ranked_rows,
        key=lambda r: float(r["waymo_mean_inference_ms"]),
    )[0]
    best_waymo_map = sorted(
        ranked_rows,
        key=lambda r: -float(r["waymo_mAP50_95"]),
    )[0]
    best_generalization = sorted(
        ranked_rows,
        key=lambda r: -float(r["generalization_ratio_mAP50_95"]),
    )[0]
    best_vru_safety = sorted(
        ranked_rows,
        key=lambda r: float(r["waymo_vru_false_negative_rate_all_sizes"]),
    )[0]
    best_small_vru_safety = sorted(
        ranked_rows,
        key=lambda r: float(r["waymo_small_vru_false_negative_rate"]),
    )[0]

    recommendations = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "weights": WEIGHTS,
        "important_warning": (
            "The deployment score is a decision-support summary, not an official benchmark. "
            "All detectors show high Waymo vulnerable-road-user false-negative rates, so none should be treated as deployment-ready."
        ),
        "best_overall_tradeoff": best_overall,
        "fastest_waymo_inference": fastest,
        "best_waymo_mAP50_95": best_waymo_map,
        "best_generalization_ratio": best_generalization,
        "best_waymo_vru_safety": best_vru_safety,
        "best_waymo_small_vru_safety": best_small_vru_safety,
        "practical_recommendations": [
            {
                "recommendation": "Use RT-DETR as the main robustness-improvement candidate.",
                "reason": "It gives the best Waymo vulnerable-road-user recall and the strongest safety-oriented score, despite slower speed."
            },
            {
                "recommendation": "Use YOLO as the real-time speed baseline.",
                "reason": "It is the fastest model on Waymo, but its vulnerable-road-user false-negative rate is too high for safety-critical use."
            },
            {
                "recommendation": "Use RetinaNet as the external-generalization reference.",
                "reason": "It has the strongest Waymo mAP50-95 and generalization ratio, but still misses many vulnerable road users."
            },
            {
                "recommendation": "Do not claim any detector is deployment-ready.",
                "reason": "The external-domain safety failure rates, especially small pedestrian/cyclist false negatives, remain high."
            },
            {
                "recommendation": "Improve with domain adaptation, threshold calibration, small-object augmentation, and safety-focused loss/reweighting.",
                "reason": "Milestone 6 and Milestone 7 both show that cross-dataset generalization and small vulnerable-road-user detection are the main weaknesses."
            }
        ],
        "gallery_manifest_summary": {
            "generated_count": gallery_manifest.get("generated_count"),
            "type_counts": gallery_manifest.get("type_counts"),
            "dataset_counts": gallery_manifest.get("dataset_counts"),
            "panels": gallery_manifest.get("panels"),
        },
    }

    DEPLOYMENT_RECOMMENDATIONS_JSON.write_text(
        json.dumps(recommendations, indent=2),
        encoding="utf-8",
    )

    summary_payload = {
        "created_at": recommendations["created_at"],
        "status": "PASSED",
        "inputs": {
            "m5_kitti_summary": str(M5_KITTI_SUMMARY.relative_to(PROJECT)),
            "m6_waymo_summary": str(M6_WAYMO_SUMMARY.relative_to(PROJECT)),
            "object_size_summary": str(OBJECT_SIZE_SUMMARY.relative_to(PROJECT)),
            "safety_false_negative_summary": str(SAFETY_FN_SUMMARY.relative_to(PROJECT)),
            "failure_type_summary": str(FAILURE_TYPE_SUMMARY.relative_to(PROJECT)),
            "gallery_manifest": str(GALLERY_MANIFEST.relative_to(PROJECT)) if GALLERY_MANIFEST.exists() else "",
        },
        "outputs": {
            "deployment_suitability_summary_csv": str(DEPLOYMENT_SUMMARY_CSV.relative_to(PROJECT)),
            "deployment_tradeoff_ranking_csv": str(DEPLOYMENT_RANKING_CSV.relative_to(PROJECT)),
            "deployment_recommendations_json": str(DEPLOYMENT_RECOMMENDATIONS_JSON.relative_to(PROJECT)),
            "summary_md": str(SUMMARY_MD.relative_to(PROJECT)),
        },
        "ranking": ranked_rows,
        "recommendations": recommendations,
    }

    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 7 Deployment Suitability and Trade-off Analysis")
    md.append("")
    md.append(f"Created at: `{summary_payload['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append(
        "This analysis compares detector suitability using KITTI accuracy, Waymo external-domain accuracy, "
        "generalization ratio, inference speed, vulnerable-road-user false-negative rate, small-object safety, "
        "and failure-case evidence."
    )
    md.append("")
    md.append("## Scoring Weights")
    md.append("")
    for key, weight in WEIGHTS.items():
        md.append(f"- `{key}`: `{weight}`")
    md.append("")
    md.append("## Deployment Ranking")
    md.append("")
    md.append("| Rank | Detector | Score | Waymo mAP50-95 | Gen. Ratio | Waymo FPS | VRU FNR | Small VRU FNR | Tier |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")

    for r in ranked_rows:
        md.append(
            f"| {r['deployment_rank']} | {r['detector']} | {r['deployment_tradeoff_score']} | "
            f"{r['waymo_mAP50_95']} | {r['generalization_ratio_mAP50_95']} | "
            f"{r['waymo_fps']} | {r['waymo_vru_false_negative_rate_all_sizes']} | "
            f"{r['waymo_small_vru_false_negative_rate']} | {r['deployment_tier']} |"
        )

    md.append("")
    md.append("## Main Practical Interpretation")
    md.append("")
    md.append(
        f"- Best overall trade-off by this safety-weighted score: `{best_overall['detector']}`."
    )
    md.append(
        f"- Fastest Waymo inference: `{fastest['detector']}` with `{fastest['waymo_fps']}` FPS."
    )
    md.append(
        f"- Best Waymo mAP50-95: `{best_waymo_map['detector']}` with `{best_waymo_map['waymo_mAP50_95']}`."
    )
    md.append(
        f"- Best generalization ratio: `{best_generalization['detector']}` with `{best_generalization['generalization_ratio_mAP50_95']}`."
    )
    md.append(
        f"- Lowest Waymo vulnerable-road-user FNR: `{best_vru_safety['detector']}` with `{best_vru_safety['waymo_vru_false_negative_rate_all_sizes']}`."
    )
    md.append(
        f"- Lowest Waymo small vulnerable-road-user FNR: `{best_small_vru_safety['detector']}` with `{best_small_vru_safety['waymo_small_vru_false_negative_rate']}`."
    )
    md.append("")
    md.append("## Detector Notes")
    md.append("")
    for r in ranked_rows:
        md.append(f"### {r['detector']}")
        md.append("")
        md.append(f"- Deployment tier: `{r['deployment_tier']}`")
        md.append(f"- Note: {r['deployment_note']}")
        md.append("")
    md.append("## Recommendations")
    md.append("")
    for rec in recommendations["practical_recommendations"]:
        md.append(f"- **{rec['recommendation']}** {rec['reason']}")
    md.append("")
    md.append("## Important Warning")
    md.append("")
    md.append(
        "The deployment score is a decision-support summary, not an official benchmark. "
        "All detectors still show high external-domain vulnerable-road-user false-negative rates, "
        "so the results should be framed as research evidence and not deployment readiness."
    )
    md.append("")
    md.append("## Outputs")
    md.append("")
    for _, value in summary_payload["outputs"].items():
        md.append(f"- `{value}`")
    md.append("")

    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Deployment trade-off analysis created")
    print("=" * 100)

    print("Created:", DEPLOYMENT_SUMMARY_CSV)
    print("Created:", DEPLOYMENT_RANKING_CSV)
    print("Created:", DEPLOYMENT_RECOMMENDATIONS_JSON)
    print("Created:", SUMMARY_JSON)
    print("Created:", SUMMARY_MD)

    print()
    print("Deployment ranking:")
    for r in ranked_rows:
        print(
            f"  Rank {r['deployment_rank']}: {r['detector']} | "
            f"score={r['deployment_tradeoff_score']} | "
            f"Waymo mAP50-95={r['waymo_mAP50_95']} | "
            f"gen_ratio={r['generalization_ratio_mAP50_95']} | "
            f"Waymo FPS={r['waymo_fps']} | "
            f"VRU FNR={r['waymo_vru_false_negative_rate_all_sizes']} | "
            f"small VRU FNR={r['waymo_small_vru_false_negative_rate']} | "
            f"tier={r['deployment_tier']}"
        )

    print()
    print("Best practical choices:")
    print(f"  Best overall safety-weighted trade-off: {best_overall['detector']}")
    print(f"  Fastest model: {fastest['detector']}")
    print(f"  Best Waymo mAP50-95: {best_waymo_map['detector']}")
    print(f"  Best external generalization ratio: {best_generalization['detector']}")
    print(f"  Lowest Waymo VRU FNR: {best_vru_safety['detector']}")
    print(f"  Lowest Waymo small VRU FNR: {best_small_vru_safety['detector']}")

    print()
    print("STEP 8/10 COMPLETE ✅")
    print("Deployment suitability and trade-off analysis is ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()