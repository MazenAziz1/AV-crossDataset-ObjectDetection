import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_6 import waymo_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Milestone 6: full Waymo external validation")
    parser.add_argument("--detector", choices=["all", "yolo", "rtdetr", "retinanet", "faster_rcnn"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for debugging).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.001, help="Low confidence threshold for AP curves.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 79)
    print("Milestone 6 - Phase 5: Full Waymo external validation")
    print("=" * 79)

    project_root = Path(__file__).resolve().parents[2]
    m3_root = project_root / "data" / "processed" / "milestone_3"
    coco_gt_path = m3_root / "annotations" / "coco" / "waymo_external.json"
    ignore_path = m3_root / "annotations" / "ignore_regions" / "waymo_external_ignore.json"
    images_dir = m3_root / "images" / "waymo" / "external"

    metrics_dir = project_root / "outputs" / "milestone_6" / "waymo_external_validation" / "metrics"
    tables_dir = project_root / "outputs" / "milestone_6" / "waymo_external_validation" / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_gt_path, encoding="utf-8") as f:
        coco = json.load(f)

    image_id_map = {img["file_name"]: img["id"] for img in coco["images"]}
    all_images = sorted(images_dir.glob("*.png"))
    if args.limit is not None:
        all_images = all_images[:args.limit]

    checkpoints = waymo_inference.load_checkpoint_paths(project_root)
    detectors = ["yolo", "rtdetr", "retinanet", "faster_rcnn"] if args.detector == "all" else [args.detector]

    summary_rows = []

    for detector in detectors:
        print(f"\n[{detector}] running inference on {len(all_images)} images ...")
        ckpt = checkpoints[detector]

        if detector in ("yolo", "rtdetr"):
            predictions, times_ms = waymo_inference.predict_ultralytics(
                detector, ckpt, all_images, image_id_map, device_arg=args.device, conf=args.conf)
        else:
            predictions, times_ms = waymo_inference.predict_torchvision(
                detector, ckpt, all_images, image_id_map, device_arg=args.device, conf=args.conf)

        print(f"[{detector}] raw predictions: {len(predictions)}")

        # DontCare-style ignore suppression (empty for Waymo; Sign is excluded, not suppressed).
        suppress_count = 0
        if ignore_path.exists():
            from scripts.milestone_4.evaluation.ignore_region_suppression import suppress_dontcare_predictions
            with open(ignore_path, encoding="utf-8") as f:
                ignore_data = json.load(f)
            predictions, suppress_count = suppress_dontcare_predictions(
                predictions, ignore_data["regions"], min_iou_overlap=0.5)

        # Reuse the Milestone 4/5 pycocotools evaluator.
        from scripts.milestone_4.evaluation.coco_evaluator import evaluate_predictions
        metrics, per_class = evaluate_predictions(coco_gt_path, predictions)

        from scripts.milestone_4.evaluation.operating_point_metrics import compute_operating_point_metrics
        operating_point = compute_operating_point_metrics(coco_gt_path, predictions)

        mean_inference_ms = float(sum(times_ms) / len(times_ms)) if times_ms else 0.0
        median_inference_ms = float(sorted(times_ms)[len(times_ms) // 2]) if times_ms else 0.0

        result = {
            "detector": detector,
            "checkpoint": str(ckpt),
            "partition": "waymo_external",
            "dataset": "Waymo external representative subset",
            "policy": "no_retraining_locked_kitti_checkpoint",
            "num_images": len(all_images),
            "num_predictions": len(predictions),
            "ignore_suppressed": suppress_count,
            "confidence_threshold_curve": args.conf,
            "input_size": [640, 640],
            "mean_inference_ms": round(mean_inference_ms, 4),
            "median_inference_ms": round(median_inference_ms, 4),
            "metrics": metrics,
            "per_class": per_class,
            "operating_point": operating_point,
        }

        out_path = metrics_dir / f"{detector}_waymo_metrics.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        row = {
            "detector": detector,
            "num_images": len(all_images),
            "mAP50": round(metrics["mAP_50"], 6),
            "mAP50_95": round(metrics["mAP_50_95"], 6),
            "mean_inference_ms": round(mean_inference_ms, 4),
        }
        for cat_id, name in [(1, "Vehicle"), (2, "Pedestrian"), (3, "Cyclist")]:
            pc = per_class.get(cat_id, per_class.get(str(cat_id), {}))
            row[f"{name}_AP50"] = round(pc.get("AP_50", 0.0), 6)
            row[f"{name}_AP50_95"] = round(pc.get("AP_50_95", 0.0), 6)
        summary_rows.append(row)

        print(f"[{detector}] mAP50={metrics['mAP_50']:.6f} mAP50_95={metrics['mAP_50_95']:.6f} "
              f"mean={mean_inference_ms:.2f}ms preds={len(predictions)}")
        print(f"  saved: {out_path}")

    # --- Summary table ---
    summary_path = tables_dir / "waymo_external_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    csv_path = tables_dir / "waymo_external_summary.csv"
    import csv as csv_mod
    fieldnames = ["detector", "num_images", "mAP50", "mAP50_95", "mean_inference_ms",
                  "Vehicle_AP50", "Vehicle_AP50_95", "Pedestrian_AP50", "Pedestrian_AP50_95",
                  "Cyclist_AP50", "Cyclist_AP50_95"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    metadata = {
        "milestone": 6,
        "phase": 5,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Waymo external representative subset",
        "num_images": len(all_images),
        "confidence_threshold": args.conf,
        "image_size": [640, 640],
        "policy": "No retraining. Locked KITTI-trained checkpoints evaluated directly on Waymo.",
        "detectors": {r["detector"]: r for r in summary_rows},
    }
    (tables_dir.parent / "waymo_external_validation_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved summary: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
