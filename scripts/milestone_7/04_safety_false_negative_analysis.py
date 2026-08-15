import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common

SAFETY_NAMES = ["Pedestrian", "Cyclist"]


def main():
    print("=" * 79)
    print("Milestone 7 - Step 4: Safety-oriented false-negative analysis")
    print("=" * 79)

    out_dir = common.M7_OUT / "safety_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out_dir / "detection_error_index.csv")
    safety = df[df["class_name"].isin(SAFETY_NAMES)]

    # FN / TP per (dataset, detector, class, size)
    tp = safety[safety["error_type"] == "true_positive"].groupby(
        ["dataset", "detector", "class_name", "size_category"]).size().rename("tp")
    fn = safety[safety["error_type"] == "false_negative"].groupby(
        ["dataset", "detector", "class_name", "size_category"]).size().rename("fn")

    s = pd.concat([tp, fn], axis=1).fillna(0).astype(int).reset_index()
    s["gt_count"] = s["tp"] + s["fn"]
    s["fn_rate"] = s["fn"] / s["gt_count"]
    s["miss_rate"] = s["fn_rate"]

    # class-agnostic (pedestrian+cyclist combined) per detector/dataset/size
    agg = s.groupby(["dataset", "detector", "size_category"]).agg(
        {"tp": "sum", "fn": "sum"}).reset_index()
    agg["class_name"] = "pedestrian+cyclist"
    agg["gt_count"] = agg["tp"] + agg["fn"]
    agg["fn_rate"] = agg["fn"] / agg["gt_count"]
    agg["miss_rate"] = agg["fn_rate"]

    full = pd.concat([s, agg], ignore_index=True)
    full = full.sort_values(["dataset", "detector", "class_name", "size_category"])
    full.to_csv(out_dir / "safety_false_negative_summary.csv", index=False)

    # Top safety-critical images (most pedestrian+cyclist misses)
    fn_rows = safety[safety["error_type"] == "false_negative"]
    top = fn_rows.groupby(["dataset", "detector", "image_id", "file_name"]).size().rename(
        "safety_misses").reset_index().sort_values(
        ["safety_misses"], ascending=False).head(10)
    top.to_csv(out_dir / "top_safety_critical_images.csv", index=False)

    result = {
        "milestone": 7,
        "step": 4,
        "note": "fn_rate = FN/(TP+FN) for safety-priority classes (Pedestrian, Cyclist) at conf>=0.25.",
        "safety_classes": SAFETY_NAMES,
        "safety_false_negative_summary": full.to_dict(orient="records"),
        "top_safety_critical_images": top.to_dict(orient="records"),
    }
    (out_dir / "safety_false_negative_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Safety FN rate by detector (pedestrian+cyclist, all sizes):")
    print(agg[agg["size_category"] == "small"].to_string(index=False))
    print("\nTop safety-critical images:")
    print(top.to_string(index=False))
    print(f"\nSaved summary CSV/JSON + top_safety_critical_images.csv under {out_dir}")


if __name__ == "__main__":
    main()
