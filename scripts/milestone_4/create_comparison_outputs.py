import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description="Generate cross-model comparison outputs")
    parser.add_argument("--metrics-dir", default="outputs/milestone_4/metrics/kitti_validation")
    parser.add_argument("--benchmarks-dir", default="outputs/milestone_4/benchmarks")
    parser.add_argument("--output-dir", default="outputs/milestone_4")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    metrics_dir = project_root / args.metrics_dir
    benchmarks_dir = project_root / args.benchmarks_dir
    output_dir = project_root / args.output_dir

    detectors = ["yolo", "faster_rcnn", "retinanet", "rtdetr"]

    train_metrics_file = metrics_dir.parent / "training_metrics.json"
    train_metrics = {}
    if train_metrics_file.exists():
        with open(train_metrics_file) as f:
            train_metrics = json.load(f).get("detectors", {})

    accuracy_rows = []
    efficiency_rows = []
    operating_point_rows = []

    for det in detectors:
        metric_file = metrics_dir / f"{det}_metrics.json"
        bench_file = benchmarks_dir / f"{det}_benchmark.json"

        if metric_file.exists():
            with open(metric_file) as f:
                m = json.load(f)
            metrics = m["metrics"]
            per_class = m["per_class"]
            tm = train_metrics.get(det, {})
            row = {
                "detector": det,
                "train_mAP_50_95": round(tm["mAP_50_95"], 4) if tm.get("mAP_50_95") is not None else "",
                "train_mAP_50": round(tm["mAP_50"], 4) if tm.get("mAP_50") is not None else "",
                "mAP_50_95": round(metrics["mAP_50_95"], 4),
                "mAP_50": round(metrics["mAP_50"], 4),
                "mAP_75": round(metrics["mAP_75"], 4),
                "AP_small": round(metrics["AP_small"], 4),
                "AP_medium": round(metrics["AP_medium"], 4),
                "AP_large": round(metrics["AP_large"], 4),
                "Vehicle_AP": round(per_class.get("1", {}).get("AP_50_95", 0), 4),
                "Pedestrian_AP": round(per_class.get("2", {}).get("AP_50_95", 0), 4),
                "Cyclist_AP": round(per_class.get("3", {}).get("AP_50_95", 0), 4),
            }
            accuracy_rows.append(row)

            op = m.get("operating_point", {})
            if op:
                operating_point_rows.append({
                    "detector": det,
                    "precision": op.get("precision", ""),
                    "recall": op.get("recall", ""),
                    "f1_score": op.get("f1_score", ""),
                    "detections_per_image": op.get("detections_per_image", ""),
                    "false_positives_per_image": op.get("false_positives_per_image", ""),
                })

        if bench_file.exists():
            with open(bench_file) as f:
                b = json.load(f)
            row = {
                "detector": det,
                "parameters": b["parameter_count"],
                "checkpoint_size_mb": b["checkpoint_size_mb"],
                "gpu_memory_peak_mb": b["gpu_memory_peak_mb"],
                "latency_total_ms": b["latency_total_ms"],
                "inference_ms": b["latency_inference_ms"],
                "frames_per_second": b["frames_per_second"],
            }
            efficiency_rows.append(row)

    # Write accuracy table
    if accuracy_rows:
        acc_path = output_dir / "figures" / "accuracy_comparison.csv"
        acc_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["detector", "train_mAP_50_95", "train_mAP_50", "mAP_50_95", "mAP_50", "mAP_75", "AP_small", "AP_medium", "AP_large", "Vehicle_AP", "Pedestrian_AP", "Cyclist_AP"]
        with open(acc_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(accuracy_rows)
        print(f"Accuracy table: {acc_path}")

    # Write efficiency table
    if efficiency_rows:
        eff_path = output_dir / "figures" / "efficiency_comparison.csv"
        eff_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["detector", "parameters", "checkpoint_size_mb", "gpu_memory_peak_mb", "latency_total_ms", "inference_ms", "frames_per_second"]
        with open(eff_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(efficiency_rows)
        print(f"Efficiency table: {eff_path}")

    # Write training vs local validation comparison
    tv_rows = []
    for a in accuracy_rows:
        tv = a.get("train_mAP_50_95")
        val = a["mAP_50_95"]
        tv_rows.append({
            "detector": a["detector"],
            "train_mAP_50_95": a["train_mAP_50_95"],
            "local_val_mAP_50_95": val,
            "delta_mAP_50_95": round(val - tv, 4) if isinstance(tv, (int, float)) else "",
            "train_mAP_50": a["train_mAP_50"],
            "local_val_mAP_50": a["mAP_50"],
        })
    if tv_rows:
        tv_path = output_dir / "figures" / "training_vs_validation_comparison.csv"
        tv_path.parent.mkdir(parents=True, exist_ok=True)
        tv_fieldnames = ["detector", "train_mAP_50_95", "local_val_mAP_50_95", "delta_mAP_50_95", "train_mAP_50", "local_val_mAP_50"]
        with open(tv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tv_fieldnames)
            writer.writeheader()
            writer.writerows(tv_rows)
        print(f"Training vs validation table: {tv_path}")

    # Write operating point comparison (conf >= 0.25, IoU >= 0.50)
    if operating_point_rows:
        op_path = output_dir / "figures" / "operating_point_comparison.csv"
        op_path.parent.mkdir(parents=True, exist_ok=True)
        op_fieldnames = ["detector", "precision", "recall", "f1_score", "detections_per_image", "false_positives_per_image"]
        with open(op_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=op_fieldnames)
            writer.writeheader()
            writer.writerows(operating_point_rows)
        print(f"Operating point table: {op_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Detector':12s} {'train50-95':>11s} {'mAP50-95':>10s} {'mAP50':>8s} {'FPS':>8s}")
    print("-" * 70)
    for a in accuracy_rows:
        fps = next((e['frames_per_second'] for e in efficiency_rows if e['detector'] == a['detector']), 'N/A')
        tv = a['train_mAP_50_95']
        tv_s = f"{tv:.4f}" if isinstance(tv, (int, float)) else "n/a"
        print(f"{a['detector']:12s} {tv_s:>11} {a['mAP_50_95']:>10.4f} {a['mAP_50']:>8.4f} {fps:>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
