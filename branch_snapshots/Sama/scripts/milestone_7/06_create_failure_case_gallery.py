import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_7 import common


def _gt_boxes_for_image(dataset, image_id):
    gt = common.load_coco(common.gt_path(dataset))
    boxes = []
    for a in gt["annotations"]:
        if a["image_id"] == image_id:
            x, y, w, h = a["bbox"]
            boxes.append({"cat": a["category_id"], "xyxy": [x, y, x + w, y + h]})
    return boxes


def _select_cases(df, dataset):
    """Reproducible case selection (no manual cherry-picking)."""
    cases = []

    def pick(error_type, class_name=None, size=None, order_by="area", ascending=False):
        sub = df[(df["dataset"] == dataset) & (df["error_type"] == error_type)]
        if class_name:
            sub = sub[sub["class_name"] == class_name]
        if size:
            sub = sub[sub["size_category"] == size]
        if sub.empty:
            return None
        return sub.sort_values(order_by, ascending=ascending).iloc[0]

    cases.append(("missed_pedestrian", pick("false_negative", class_name="Pedestrian")))
    cases.append(("missed_cyclist", pick("false_negative", class_name="Cyclist")))
    cases.append(("small_object_failure", pick("false_negative", class_name="Pedestrian", size="small")))
    cases.append(("false_positive", pick("false_positive", order_by="confidence")))
    cases.append(("localization_error", pick("localization_error", order_by="confidence")))
    cases.append(("class_confusion", pick("class_confusion", order_by="confidence")))
    return [c for c in cases if c[1] is not None]


def _draw_case(dataset, detector, err_type, row, img_path, gt_boxes):
    img = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.set_axis_off()

    # All GT boxes (thin green)
    for b in gt_boxes:
        x1, y1, x2, y2 = b["xyxy"]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor="green", linewidth=0.8))

    # The target box (red for miss, blue for detection error)
    x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
    color = "red" if err_type in ("missed_pedestrian", "missed_cyclist", "small_object_failure") else "blue"
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=color, linewidth=2.5))

    label = f"{common.DETECTOR_DISPLAY[detector]} | {err_type}\n{row['class_name']}"
    ax.set_title(label, fontsize=9)

    fig.tight_layout()
    return fig


def main():
    print("=" * 79)
    print("Milestone 7 - Step 6: Failure-case gallery")
    print("=" * 79)

    img_out = common.M7_OUT / "failure_cases" / "images"
    panel_out = common.M7_OUT / "failure_cases" / "panels"
    img_out.mkdir(parents=True, exist_ok=True)
    panel_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(common.M7_OUT / "safety_error_analysis" / "detection_error_index.csv")
    manifest = []

    for dataset in common.DATASETS:
        cases = _select_cases(df, dataset)
        figs = []
        for err_type, row in cases:
            detector = row["detector"]
            image_id = int(row["image_id"])
            fname = row["file_name"]
            img_path = common.img_dir(dataset) / fname
            if not img_path.exists():
                print(f"  [skip] missing image {img_path}")
                continue

            gt_boxes = _gt_boxes_for_image(dataset, image_id)
            fig = _draw_case(dataset, detector, err_type, row, img_path, gt_boxes)

            out_name = f"{dataset}_{detector}_{err_type}.png"
            fig.savefig(img_out / out_name, dpi=120)
            plt.close(fig)
            figs.append((err_type, img_out / out_name))
            manifest.append({
                "dataset": dataset, "detector": detector, "error_type": err_type,
                "image_id": image_id, "file_name": fname, "image": str(img_out / out_name),
                "confidence": ("" if pd.isna(row["confidence"]) else float(row["confidence"])),
                "iou": float(row["iou"]), "class_name": row["class_name"],
                "gt_class_name": ("" if pd.isna(row["gt_class_name"]) else row["gt_class_name"]),
            })
            print(f"  [{dataset}/{detector}] {err_type} -> {out_name}")

        # Build panel (grid of up to 6)
        if figs:
            n = len(figs)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten() if n > 1 else [axes]
            for ax, (err_type, p) in zip(axes, figs):
                im = Image.open(p).convert("RGB")
                ax.imshow(im); ax.set_axis_off(); ax.set_title(err_type, fontsize=8)
            for ax in axes[len(figs):]:
                ax.axis("off")
            fig.suptitle(f"Failure cases - {common.DATASET_DISPLAY[dataset]}", fontsize=13)
            fig.tight_layout()
            fig.savefig(panel_out / f"failure_case_panel_{dataset}.png", dpi=120)
            plt.close(fig)
            print(f"  panel -> failure_case_panel_{dataset}.png")

    (common.M7_OUT / "failure_cases" / "failure_case_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# Milestone 7 Failure-Case Gallery", "",
          "Reproducible annotated examples of detector failures, selected by rule (not manually):",
          "- missed pedestrian / cyclist: largest-area false negative",
          "- small-object failure: largest-area small pedestrian false negative",
          "- false positive / localization / confusion: highest-confidence occurrence", "",
          "Annotation convention: green = all ground-truth boxes; red = missed object; blue = erroneous detection.", "",
          "| Dataset | Detector | Error type | Image |", "|---|---|---|---|"]
    for m in manifest:
        md.append(f"| {m['dataset']} | {m['detector']} | {m['error_type']} | {Path(m['image']).name} |")
    md.append("")
    (common.M7_OUT / "failure_cases" / "FAILURE_CASE_GALLERY.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nManifest entries: {len(manifest)}")
    print("Saved failure_case_manifest.json + FAILURE_CASE_GALLERY.md")


if __name__ == "__main__":
    main()
