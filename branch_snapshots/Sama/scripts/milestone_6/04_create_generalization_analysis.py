import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DETECTORS = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]
CLASSES = [("1", "Vehicle"), ("2", "Pedestrian"), ("3", "Cyclist")]


def safe_ratio(num, den):
    if num is None or den is None:
        return None
    if float(den) == 0.0:
        return None
    return float(num) / float(den)


def safe_drop(kitti, waymo):
    if kitti is None or waymo is None:
        return None
    return float(kitti) - float(waymo)


def safe_drop_pct(kitti, waymo):
    drop = safe_drop(kitti, waymo)
    if drop is None or float(kitti) == 0.0:
        return None
    return 100.0 * drop / float(kitti)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 79)
    print("Milestone 6 - Phase 6: KITTI vs Waymo generalization analysis")
    print("=" * 79)

    project_root = Path(__file__).resolve().parents[2]
    kitti_dir = project_root / "outputs" / "milestone_5" / "metrics" / "kitti_validation"
    waymo_dir = project_root / "outputs" / "milestone_6" / "waymo_external_validation" / "metrics"
    out_dir = project_root / "outputs" / "milestone_6" / "generalization_analysis"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    kitti_num_images = None
    kitti_val_coco = project_root / "data" / "processed" / "milestone_3" / "annotations" / "coco" / "kitti_val.json"
    if kitti_val_coco.exists():
        with open(kitti_val_coco, encoding="utf-8") as f:
            kitti_num_images = len(json.load(f)["images"])

    kitti = {}
    waymo = {}
    for det in DETECTORS:
        kitti_path = kitti_dir / f"{det}_metrics.json"
        waymo_path = waymo_dir / f"{det}_waymo_metrics.json"
        if not kitti_path.exists():
            print(f"ERROR: missing KITTI baseline {kitti_path}")
            sys.exit(1)
        if not waymo_path.exists():
            print(f"ERROR: missing Waymo results {waymo_path}")
            sys.exit(1)
        kitti[det] = load_json(kitti_path)
        waymo[det] = load_json(waymo_path)

    # --- Detector-level comparison ---
    comparison_rows = []
    for det in DETECTORS:
        km = kitti[det]["metrics"]
        wm = waymo[det]["metrics"]
        row = {
            "detector": det,
            "KITTI_num_images": kitti_num_images,
            "Waymo_num_images": waymo[det].get("num_images", 996),
            "KITTI_mAP50": km["mAP_50"],
            "Waymo_mAP50": wm["mAP_50"],
            "mAP50_absolute_drop": safe_drop(km["mAP_50"], wm["mAP_50"]),
            "mAP50_drop_percent": safe_drop_pct(km["mAP_50"], wm["mAP_50"]),
            "mAP50_generalization_ratio": safe_ratio(wm["mAP_50"], km["mAP_50"]),
            "KITTI_mAP50_95": km["mAP_50_95"],
            "Waymo_mAP50_95": wm["mAP_50_95"],
            "mAP50_95_absolute_drop": safe_drop(km["mAP_50_95"], wm["mAP_50_95"]),
            "mAP50_95_drop_percent": safe_drop_pct(km["mAP_50_95"], wm["mAP_50_95"]),
            "mAP50_95_generalization_ratio": safe_ratio(wm["mAP_50_95"], km["mAP_50_95"]),
            "Waymo_mean_inference_ms": waymo[det].get("mean_inference_ms"),
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values(
        by=["mAP50_95_generalization_ratio", "Waymo_mAP50_95"],
        ascending=[False, False]).reset_index(drop=True)
    comparison_df.insert(1, "rank_by_mAP50_95_generalization", comparison_df.index + 1)

    # --- Class-wise AP50_95 degradation ---
    class_rows = []
    for det in DETECTORS:
        kpc = kitti[det]["per_class"]
        wpc = waymo[det]["per_class"]
        for cat_id, cls_name in CLASSES:
            kpc_c = kpc.get(cat_id, kpc.get(int(cat_id), {}))
            wpc_c = wpc.get(cat_id, wpc.get(int(cat_id), {}))
            k_val = kpc_c.get("AP_50_95")
            w_val = wpc_c.get("AP_50_95")
            class_rows.append({
                "detector": det,
                "class_name": cls_name,
                "KITTI_AP50_95": k_val,
                "Waymo_AP50_95": w_val,
                "absolute_drop": safe_drop(k_val, w_val),
                "drop_percent": safe_drop_pct(k_val, w_val),
                "generalization_ratio": safe_ratio(w_val, k_val),
            })
    class_df = pd.DataFrame(class_rows)

    # --- Generalization ratio table (detector-level + class-level AP50_95) ---
    ratio_rows = []
    for det in DETECTORS:
        row = {"detector": det}
        row["mAP50_generalization_ratio"] = safe_ratio(
            waymo[det]["metrics"]["mAP_50"], kitti[det]["metrics"]["mAP_50"])
        row["mAP50_95_generalization_ratio"] = safe_ratio(
            waymo[det]["metrics"]["mAP_50_95"], kitti[det]["metrics"]["mAP_50_95"])
        kpc = kitti[det]["per_class"]
        wpc = waymo[det]["per_class"]
        for cat_id, cls_name in CLASSES:
            kpc_c = kpc.get(cat_id, kpc.get(int(cat_id), {}))
            wpc_c = wpc.get(cat_id, wpc.get(int(cat_id), {}))
            row[f"{cls_name}_AP50_95_generalization_ratio"] = safe_ratio(
                wpc_c.get("AP_50_95"), kpc_c.get("AP_50_95"))
        ratio_rows.append(row)
    ratio_df = pd.DataFrame(ratio_rows)

    # --- Summary derivations ---
    best = comparison_df.iloc[0]["detector"]
    worst = comparison_df.iloc[-1]["detector"]

    class_ap5095 = class_df.copy()
    class_mean_drop = class_ap5095.groupby("class_name")["absolute_drop"].mean().sort_values(ascending=False)
    largest_degraded_class = class_mean_drop.index[0] if len(class_mean_drop) else None

    summary = {
        "milestone": 6,
        "phase": 6,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best_generalizing_detector": best,
        "worst_generalizing_detector": worst,
        "largest_degraded_class": largest_degraded_class,
        "class_mean_absolute_drop_ap50_95": {k: round(float(v), 6) for k, v in class_mean_drop.items()},
        "detector_ranking_by_mAP50_95_generalization": comparison_df["detector"].tolist(),
    }

    # --- Write outputs ---
    comparison_df.to_csv(tables_dir / "kitti_vs_waymo_comparison.csv", index=False)
    ratio_df.to_csv(tables_dir / "generalization_ratio_table.csv", index=False)
    class_df.to_csv(tables_dir / "class_wise_degradation.csv", index=False)
    (out_dir / "generalization_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\nKITTI vs Waymo comparison:")
    print(comparison_df.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
