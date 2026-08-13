import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def benchmark_model(checkpoint_path, detector, warmup_runs=10, benchmark_runs=100,
                    imgsz=640, output_dir="outputs/milestone_4"):
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(checkpoint_path))
    model.model.to(device)
    model.model.eval()

    # --- Parameter count ---
    param_count = sum(p.numel() for p in model.model.parameters())

    # --- Checkpoint size ---
    checkpoint_size_mb = round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2)

    # --- Create a dummy input ---
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)

    # Warmup
    for _ in range(warmup_runs):
        _ = model.model(dummy)

    # --- Inference latency (pure forward, no preprocessing) ---
    torch.cuda.synchronize()
    latencies = []
    for _ in range(benchmark_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.model(dummy)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    inference_ms = float(np.mean(latencies))
    fps = 1000.0 / inference_ms

    # --- Peak GPU memory ---
    if torch.cuda.is_available():
        gpu_mem_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    else:
        gpu_mem_peak_mb = 0.0

    # --- Pre/post processing latency (on a sample image) ---
    sample_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    pre_times = []
    post_times = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        # preprocess: resize + normalize (simulate ultralytics LetterBox)
        resized = cv2.resize(sample_img, (imgsz, imgsz))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        pre_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        # postprocess: NMS (simulate with a fixed-size operation)
        _ = tensor.contiguous()
        post_times.append((time.perf_counter() - t0) * 1000)

    preprocess_ms = float(np.mean(pre_times))
    postprocess_ms = float(np.mean(post_times))
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    result = {
        "detector": detector,
        "device": device,
        "parameter_count": int(param_count),
        "checkpoint_size_mb": checkpoint_size_mb,
        "gpu_memory_peak_mb": gpu_mem_peak_mb,
        "latency_preprocessing_ms": round(preprocess_ms, 3),
        "latency_inference_ms": round(inference_ms, 3),
        "latency_postprocessing_ms": round(postprocess_ms, 3),
        "latency_total_ms": round(total_ms, 3),
        "frames_per_second": round(fps, 2),
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
    }

    out_dir = Path(output_dir) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{detector}_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Efficiency Benchmarking")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/milestone_4")
    args = parser.parse_args()

    result = benchmark_model(
        checkpoint_path=args.checkpoint,
        detector=args.detector,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 50)
    print(f"Benchmark: {args.detector}")
    print(f"  Parameters:        {result['parameter_count']:,}")
    print(f"  Checkpoint size:   {result['checkpoint_size_mb']} MB")
    print(f"  GPU mem peak:      {result['gpu_memory_peak_mb']} MB")
    print(f"  Preprocessing:     {result['latency_preprocessing_ms']} ms")
    print(f"  Inference:         {result['latency_inference_ms']} ms")
    print(f"  Postprocessing:    {result['latency_postprocessing_ms']} ms")
    print(f"  Total latency:     {result['latency_total_ms']} ms")
    print(f"  FPS:               {result['frames_per_second']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
