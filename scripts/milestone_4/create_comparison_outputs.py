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

    accuracy_rows = []
    efficiency_rows = []

    for det in detectors:
        metric_file = metrics_dir / f"{det}_metrics.json"
        bench_file = benchmarks_dir / f"{det}_benchmark.json"

        if metric_file.exists():
            with open(metric_file) as f:
                m = json.load(f)
            metrics = m["metrics"]
            per_class = m["per_class"]
            row = {
                "detector": det,
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
        fieldnames = ["detector", "mAP_50_95", "mAP_50", "mAP_75", "AP_small", "AP_medium", "AP_large", "Vehicle_AP", "Pedestrian_AP", "Cyclist_AP"]
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

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Detector':12s} {'mAP50-95':>10s} {'mAP50':>8s} {'FPS':>8s}")
    print("-" * 70)
    for a in accuracy_rows:
        fps = next((e['frames_per_second'] for e in efficiency_rows if e['detector'] == a['detector']), 'N/A')
        print(f"{a['detector']:12s} {a['mAP_50_95']:>10.4f} {a['mAP_50']:>8.4f} {fps:>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
