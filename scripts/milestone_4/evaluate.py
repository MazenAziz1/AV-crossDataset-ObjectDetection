import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description="KITTI Local Evaluation (YOLO)")
    parser.add_argument("--detector", required=True, choices=["yolo", "rtdetr", "faster_rcnn", "retinanet"])
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--partition", default="kitti_val")
    parser.add_argument("--output-dir", default="outputs/milestone_4")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "processed" / "milestone_3"

    gt_json = data_root / "annotations" / "coco" / f"{args.partition}.json"
    ignore_json = data_root / "annotations" / "ignore_regions" / f"{args.partition}_ignore.json"
    images_dir = data_root / "images" / "kitti" / "val"

    if not gt_json.exists():
        print(f"ERROR: GT annotations not found: {gt_json}")
        sys.exit(1)

    with open(gt_json) as f:
        gt_data = json.load(f)
    image_ids = gt_data["images"]

    # Generate predictions
    from scripts.milestone_4.adapters.yolo_adapter import run_yolo_predictions

    print(f"Running inference with checkpoint: {args.checkpoint}")
    predictions = run_yolo_predictions(
        checkpoint_path=args.checkpoint,
        images_dir=images_dir,
        image_ids=image_ids,
    )
    print(f"Raw predictions: {len(predictions)}")

    # DontCare suppression
    suppress_count = 0
    if ignore_json.exists():
        from scripts.milestone_4.evaluation.ignore_region_suppression import suppress_dontcare_predictions
        with open(ignore_json) as f:
            ignore_data = json.load(f)
        predictions, suppress_count = suppress_dontcare_predictions(
            predictions, ignore_data["regions"], min_iou_overlap=0.5
        )
        print(f"DontCare suppressed: {suppress_count} predictions removed")
        print(f"Predictions after suppression: {len(predictions)}")

    # Evaluate
    from scripts.milestone_4.evaluation.coco_evaluator import evaluate_predictions

    metrics, per_class = evaluate_predictions(gt_json, predictions)

    result = {
        "detector": args.detector,
        "checkpoint": str(args.checkpoint),
        "partition": args.partition,
        "num_predictions": len(predictions),
        "dontcare_suppressed": suppress_count,
        "metrics": metrics,
        "per_class": per_class,
    }

    out_dir = Path(args.output_dir) / "metrics" / "kitti_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.detector}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Results for {args.detector}:")
    print(f"  mAP@0.50:0.95 = {metrics['mAP_50_95']:.4f}")
    print(f"  mAP@0.50     = {metrics['mAP_50']:.4f}")
    print(f"  mAP@0.75     = {metrics['mAP_75']:.4f}")
    print("  Per-class mAP@0.50:0.95:")
    for cat_id, info in sorted(per_class.items()):
        print(f"    {info['name']:12s} = {info['AP_50_95']:.4f}")
    print(f"\nSaved to: {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
