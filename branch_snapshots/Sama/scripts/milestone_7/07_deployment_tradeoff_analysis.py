import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common

SAFETY_NAMES = ["Pedestrian", "Cyclist"]


def main():
    print("=" * 79)
    print("Milestone 7 - Step 7: Deployment trade-off analysis")
    print("=" * 79)

    out_dir = common.M7_OUT / "deployment_tradeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    # KITTI mAP50-95 (M5)
    kitti_map = {}
    for d in common.DETECTORS:
        m = json.load(open(common.M5_METRICS_DIR / f"{d}_metrics.json", encoding="utf-8"))
        kitti_map[d] = m["metrics"]["mAP_50_95"]

    # Waymo mAP50-95 + inference time (M6 summary)
    wsum = pd.read_csv(common.M6_SUMMARY_CSV)
    waymo_map = dict(zip(wsum["detector"], wsum["mAP50_95"]))
    waymo_ms = dict(zip(wsum["detector"], wsum["mean_inference_ms"]))

    # Safety FN rate + small-object recall from the error index
    idx = pd.read_csv(common.M7_OUT / "safety_error_analysis" / "detection_error_index.csv")

    rows = []
    for d in common.DETECTORS:
        sub = idx[idx["detector"] == d]
        safety = sub[sub["class_name"].isin(SAFETY_NAMES)]
        k_safe = safety[(safety["dataset"] == "kitti")]
        w_safe = safety[(safety["dataset"] == "waymo")]

        def fn_rate(x):
            tp = (x["error_type"] == "true_positive").sum()
            fn = (x["error_type"] == "false_negative").sum()
            return fn / (tp + fn) if (tp + fn) else None

        def small_recall(x):
            s = x[x["size_category"] == "small"]
            tp = (s["error_type"] == "true_positive").sum()
            fn = (s["error_type"] == "false_negative").sum()
            return tp / (tp + fn) if (tp + fn) else None

        rows.append({
            "detector": d,
            "KITTI_mAP50_95": round(kitti_map[d], 4),
            "Waymo_mAP50_95": round(waymo_map[d], 4),
            "waymo_mean_inference_ms": round(waymo_ms[d], 2),
            "KITTI_safety_fn_rate": round(fn_rate(k_safe), 4),
            "Waymo_safety_fn_rate": round(fn_rate(w_safe), 4),
            "KITTI_small_recall": round(small_recall(sub[sub["dataset"] == "kitti"]), 4),
            "Waymo_small_recall": round(small_recall(w_safe), 4),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "deployment_suitability_table.csv", index=False)
    (out_dir / "deployment_suitability_table.json").write_text(
        table.to_json(orient="records", indent=2), encoding="utf-8")

    print(table.to_string(index=False))

    # Cautious recommendations
    fastest = table.loc[table["waymo_mean_inference_ms"].idxmin(), "detector"]
    best_kitti = table.loc[table["KITTI_mAP50_95"].idxmax(), "detector"]
    best_waymo = table.loc[table["Waymo_mAP50_95"].idxmax(), "detector"]
    best_waymo_safety = table.loc[table["Waymo_safety_fn_rate"].idxmin(), "detector"]

    md = ["# Milestone 7 - Deployment Suitability Recommendations", "",
          "These recommendations are derived from the frozen M5/M6 results and the Milestone 7",
          "error index. They are stated cautiously; no detector is declared deployment-ready on",
          "the basis of this analysis alone.", "",
          "## Observations", "",
          f"- Fastest inference (Waymo): **{common.DETECTOR_DISPLAY[fastest]}**.",
          f"- Highest KITTI accuracy: **{common.DETECTOR_DISPLAY[best_kitti]}**.",
          f"- Highest Waymo generalization: **{common.DETECTOR_DISPLAY[best_waymo]}**.",
          f"- Lowest Waymo safety false-negative rate: **{common.DETECTOR_DISPLAY[best_waymo_safety]}**.", "",
          "## Interpretation", "",
          "- The in-domain accuracy ranking does not transfer to the out-of-domain setting: the",
          "  strongest KITTI detector generalizes worst to Waymo.",
          "- All detectors miss the large majority of small Waymo objects (pedestrians/cyclists in",
          "  particular), which is the dominant safety risk in cross-dataset deployment.",
          "- No single detector simultaneously optimizes accuracy, generalization, latency, and",
          "  safety; the choice of detector depends on the deployment priority.", "",
          "## Deployment-table columns", "",
          "- KITTI_mAP50_95 / Waymo_mAP50_95: COCO mAP@[0.50:0.95].",
          "- waymo_mean_inference_ms: per-image latency measured on real Waymo images.",
          "- safety_fn_rate: pedestrian+cyclist false-negative rate at the operating point.",
          "- small_recall: recall on small objects (<32^2 px).", ""]
    (out_dir / "deployment_recommendations.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nSaved deployment_suitability_table.csv/json + deployment_recommendations.md under {out_dir}")


if __name__ == "__main__":
    main()
