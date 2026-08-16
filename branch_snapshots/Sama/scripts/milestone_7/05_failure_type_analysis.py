import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common

ERROR_TYPES = ["true_positive", "false_negative", "false_positive",
               "localization_error", "class_confusion", "over_detection"]


def main():
    print("=" * 79)
    print("Milestone 7 - Step 5: Failure-type analysis")
    print("=" * 79)

    out_dir = common.M7_OUT / "safety_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out_dir / "detection_error_index.csv")

    # Failure-type counts per detector/dataset
    counts = df.groupby(["dataset", "detector", "error_type"]).size().rename("count").reset_index()
    piv = counts.pivot_table(index=["dataset", "detector"], columns="error_type",
                             values="count", fill_value=0).reset_index()
    piv = piv[[c for c in ["dataset", "detector"] + ERROR_TYPES if c in piv.columns]]
    piv.to_csv(out_dir / "failure_type_summary.csv", index=False)

    # Class confusion matrix (predicted class vs ground-truth class)
    conf = df[df["error_type"] == "class_confusion"]
    cm = conf.groupby(["dataset", "detector", "class_name", "gt_class_name"]).size().rename(
        "count").reset_index()
    cm.to_csv(out_dir / "class_confusion_summary.csv", index=False)

    result = {
        "milestone": 7,
        "step": 5,
        "failure_type_summary": piv.to_dict(orient="records"),
        "class_confusion_summary": cm.to_dict(orient="records"),
    }
    (out_dir / "failure_type_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Failure-type counts by dataset/detector:")
    print(piv.to_string(index=False))
    print("\nClass confusion (predicted -> ground truth):")
    print(cm.to_string(index=False))
    print(f"\nSaved failure_type_summary.csv/json + class_confusion_summary.csv under {out_dir}")


if __name__ == "__main__":
    main()
