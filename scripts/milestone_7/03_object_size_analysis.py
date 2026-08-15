import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common

SIZE_ORDER = ["small", "medium", "large"]


def main():
    print("=" * 79)
    print("Milestone 7 - Step 3: Object-size robustness analysis")
    print("=" * 79)

    out_dir = common.M7_OUT / "object_size_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(common.M7_OUT / "safety_error_analysis" / "detection_error_index.csv")

    # TP / FN per (dataset, detector, class, size)
    tp = df[df["error_type"] == "true_positive"].groupby(
        ["dataset", "detector", "class_name", "size_category"]).size().rename("tp")
    fn = df[df["error_type"] == "false_negative"].groupby(
        ["dataset", "detector", "class_name", "size_category"]).size().rename("fn")

    summary = pd.concat([tp, fn], axis=1).fillna(0).astype(int).reset_index()
    summary["gt_count"] = summary["tp"] + summary["fn"]
    summary["recall"] = summary["tp"] / summary["gt_count"]
    summary["miss_rate"] = 1.0 - summary["recall"]

    # Add class-agnostic rows (aggregate over class within each detector/dataset/size)
    agg = summary.groupby(["dataset", "detector", "size_category"]).agg(
        {"tp": "sum", "fn": "sum"}).reset_index()
    agg["class_name"] = "all"
    agg["gt_count"] = agg["tp"] + agg["fn"]
    agg["recall"] = agg["tp"] / agg["gt_count"]
    agg["miss_rate"] = 1.0 - agg["recall"]

    full = pd.concat([summary, agg], ignore_index=True)
    full = full.sort_values(["dataset", "detector", "class_name", "size_category"])

    full.to_csv(out_dir / "object_size_summary.csv", index=False)

    # Small-object failure summary (rank detectors by small-object recall/miss rate)
    small = full[(full["size_category"] == "small") & (full["class_name"] == "all")].copy()
    small = small.sort_values(["dataset", "miss_rate"], ascending=[True, False])
    small.to_csv(out_dir / "small_object_failure_summary.csv", index=False)

    # JSON summary
    result = {
        "milestone": 7,
        "step": 3,
        "note": "recall = TP/(TP+FN); miss_rate = FN/(TP+FN) at the frozen operating point (conf>=0.25, IoU>=0.5).",
        "size_bins": common.load_bins()["bins"],
        "object_size_summary": full.to_dict(orient="records"),
        "small_object_ranking": small.to_dict(orient="records"),
    }
    (out_dir / "object_size_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Object-size recall/miss-rate (class='all'):")
    print(full[full["class_name"] == "all"].to_string(index=False))
    print(f"\nSaved: {out_dir / 'object_size_summary.csv'}")
    print(f"Saved: {out_dir / 'object_size_summary.json'}")
    print(f"Saved: {out_dir / 'small_object_failure_summary.csv'}")


if __name__ == "__main__":
    main()
