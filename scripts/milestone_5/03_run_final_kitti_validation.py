from pathlib import Path
import argparse
import json
import time
import gc
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from ultralytics import YOLO


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

REGISTRY_PATH = PROJECT / "outputs" / "milestone_4" / "locked_final_checkpoints" / "final_checkpoint_registry.json"

IMAGE_VAL_DIR = PROJECT / "data" / "processed" / "milestone_3" / "images" / "kitti" / "val"
LABEL_VAL_DIR = PROJECT / "data" / "processed" / "milestone_3" / "labels" / "kitti" / "val"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_5" / "final_kitti_validation"
PRED_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
TABLE_DIR = OUTPUT_DIR / "tables"

for d in [PRED_DIR, METRICS_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = {
    0: "Vehicle",
    1: "Pedestrian",
    2: "Cyclist",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run final local KITTI validation for Milestone 5.")
    parser.add_argument(
        "--detector",
        choices=["all", "yolo", "rtdetr", "retinanet", "faster_rcnn"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of validation images for smoke testing.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or 0")
    parser.add_argument("--conf", type=float, default=0.001, help="Low confidence threshold for AP calculation.")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device_arg in {"0", "cuda", "cuda:0"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


def resolve_ultralytics_device(device_arg: str):
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"

    if device_arg in {"0", "cuda", "cuda:0"}:
        return 0 if torch.cuda.is_available() else "cpu"

    return "cpu"


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing final checkpoint registry: {REGISTRY_PATH}")

    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def get_image_paths(limit: int | None) -> List[Path]:
    images = sorted(IMAGE_VAL_DIR.glob("*.png"))

    if not images:
        raise RuntimeError(f"No validation images found in {IMAGE_VAL_DIR}")

    if limit is not None:
        images = images[:limit]

    return images


def load_ground_truth_for_image(image_path: Path) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    label_path = LABEL_VAL_DIR / f"{image_path.stem}.txt"

    boxes = []
    labels = []

    if label_path.exists():
        text = label_path.read_text(encoding="utf-8").strip()
        lines = text.splitlines() if text else []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")

            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height

            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))

            if x2 > x1 and y2 > y1:
                labels.append(cls)
                boxes.append([x1, y1, x2, y2])

    return {
        "image_id": image_path.stem,
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": labels,
    }


def load_all_ground_truth(image_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    return {p.stem: load_ground_truth_for_image(p) for p in image_paths}


def build_torchvision_model(detector: str, checkpoint_path: Path, device: torch.device):
    if detector == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=4,
        )
        model.roi_heads.score_thresh = 0.001
        model.roi_heads.detections_per_img = 300

    elif detector == "retinanet":
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=4,
        )
        model.score_thresh = 0.001
        model.detections_per_img = 300

    else:
        raise ValueError(f"Unsupported Torchvision detector: {detector}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.to(device)
    model.eval()

    return model


def predict_ultralytics(
    detector: str,
    checkpoint_path: Path,
    image_paths: List[Path],
    device_arg: str,
    conf: float,
    imgsz: int,
) -> Tuple[List[Dict[str, Any]], float]:
    model = YOLO(str(checkpoint_path))
    ultra_device = resolve_ultralytics_device(device_arg)

    rows = []
    inference_times_ms = []

    for idx, image_path in enumerate(image_paths, start=1):
        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=0.7,
            max_det=300,
            device=ultra_device,
            verbose=False,
        )

        if torch.cuda.is_available() and ultra_device != "cpu":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        inference_times_ms.append(elapsed_ms)

        result = results[0]
        predictions = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, label in zip(xyxy, scores, labels):
                predictions.append({
                    "class_id": int(label),
                    "score": float(score),
                    "box": [float(x) for x in box.tolist()],
                })

        rows.append({
            "image_id": image_path.stem,
            "detector": detector,
            "inference_ms": elapsed_ms,
            "predictions": predictions,
        })

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"[{detector}] predicted {idx}/{len(image_paths)} images")

    return rows, float(np.mean(inference_times_ms))


def predict_torchvision(
    detector: str,
    checkpoint_path: Path,
    image_paths: List[Path],
    device: torch.device,
    conf: float,
) -> Tuple[List[Dict[str, Any]], float]:
    model = build_torchvision_model(detector, checkpoint_path, device)

    rows = []
    inference_times_ms = []

    for idx, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        tensor = F.to_tensor(image).to(device)

        start = time.perf_counter()

        with torch.no_grad():
            output = model([tensor])[0]

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        inference_times_ms.append(elapsed_ms)

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy().astype(int)

        predictions = []

        for box, score, label in zip(boxes, scores, labels):
            if float(score) < conf:
                continue

            # Torchvision uses labels 1,2,3 because label 0 is background.
            class_id = int(label) - 1

            if class_id not in CLASS_NAMES:
                continue

            predictions.append({
                "class_id": class_id,
                "score": float(score),
                "box": [float(x) for x in box.tolist()],
            })

        rows.append({
            "image_id": image_path.stem,
            "detector": detector,
            "inference_ms": elapsed_ms,
            "predictions": predictions,
        })

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"[{detector}] predicted {idx}/{len(image_paths)} images")

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows, float(np.mean(inference_times_ms))


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    if len(box_a) == 0 or len(box_b) == 0:
        return np.zeros((len(box_a), len(box_b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3]
    bx1, by1, bx2, by2 = box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / np.maximum(union, 1e-9)


def ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    # COCO-style 101-point interpolation
    points = np.linspace(0.0, 1.0, 101)
    ap = 0.0

    for t in points:
        mask = recalls >= t
        ap += np.max(precisions[mask]) if np.any(mask) else 0.0

    return float(ap / 101.0)


def evaluate_detector(
    detector: str,
    prediction_rows: List[Dict[str, Any]],
    gt_by_image: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    metrics = {
        "detector": detector,
        "num_images": len(gt_by_image),
        "classes": {},
        "overall": {},
    }

    all_ap = []
    ap50_values = []

    for class_id, class_name in CLASS_NAMES.items():
        class_metrics = {}

        gt_for_class = {}
        total_gt = 0

        for image_id, gt in gt_by_image.items():
            labels = np.array(gt["labels"], dtype=np.int64)
            boxes = np.array(gt["boxes"], dtype=np.float32)

            mask = labels == class_id
            class_boxes = boxes[mask] if len(boxes) else np.zeros((0, 4), dtype=np.float32)

            gt_for_class[image_id] = class_boxes
            total_gt += len(class_boxes)

        predictions = []

        for row in prediction_rows:
            image_id = row["image_id"]

            for pred in row["predictions"]:
                if pred["class_id"] == class_id:
                    predictions.append({
                        "image_id": image_id,
                        "score": float(pred["score"]),
                        "box": pred["box"],
                    })

        predictions.sort(key=lambda x: x["score"], reverse=True)

        aps_for_thresholds = []

        for iou_thr in iou_thresholds:
            matched = {
                image_id: np.zeros(len(boxes), dtype=bool)
                for image_id, boxes in gt_for_class.items()
            }

            tp = np.zeros(len(predictions), dtype=np.float32)
            fp = np.zeros(len(predictions), dtype=np.float32)

            for pred_idx, pred in enumerate(predictions):
                image_id = pred["image_id"]
                pred_box = np.array([pred["box"]], dtype=np.float32)

                gt_boxes = gt_for_class.get(image_id, np.zeros((0, 4), dtype=np.float32))

                if len(gt_boxes) == 0:
                    fp[pred_idx] = 1.0
                    continue

                ious = box_iou(pred_box, gt_boxes)[0]
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                if best_iou >= iou_thr and not matched[image_id][best_idx]:
                    tp[pred_idx] = 1.0
                    matched[image_id][best_idx] = True
                else:
                    fp[pred_idx] = 1.0

            if total_gt == 0:
                ap = float("nan")
            elif len(predictions) == 0:
                ap = 0.0
            else:
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)

                recalls = tp_cum / max(float(total_gt), 1e-9)
                precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

                ap = ap_from_pr(recalls, precisions)

            class_metrics[f"AP@{iou_thr:.2f}"] = ap
            aps_for_thresholds.append(ap)

        ap50 = class_metrics["AP@0.50"]
        map_50_95 = float(np.nanmean(aps_for_thresholds))

        class_metrics["AP50"] = ap50
        class_metrics["AP50_95"] = map_50_95
        class_metrics["gt_count"] = int(total_gt)
        class_metrics["prediction_count"] = int(len(predictions))

        metrics["classes"][class_name] = class_metrics

        all_ap.append(map_50_95)
        ap50_values.append(ap50)

    metrics["overall"]["mAP50"] = float(np.nanmean(ap50_values))
    metrics["overall"]["mAP50_95"] = float(np.nanmean(all_ap))
    metrics["overall"]["mean_inference_ms"] = float(np.mean([r["inference_ms"] for r in prediction_rows]))

    return metrics


def save_predictions(detector: str, rows: List[Dict[str, Any]], limit: int | None) -> Path:
    suffix = f"limit_{limit}" if limit is not None else "full"
    out_path = PRED_DIR / f"{detector}_predictions_{suffix}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return out_path


def save_metrics(detector: str, metrics: Dict[str, Any], limit: int | None) -> Path:
    suffix = f"limit_{limit}" if limit is not None else "full"
    out_path = METRICS_DIR / f"{detector}_metrics_{suffix}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out_path


def run_detector(detector: str, registry: Dict[str, Any], image_paths: List[Path], gt_by_image: Dict[str, Any], args):
    print()
    print("=" * 100)
    print("Running detector:", detector)
    print("=" * 100)

    checkpoint_path = PROJECT / registry["models"][detector]["canonical_checkpoint"]
    print("Checkpoint:", checkpoint_path.relative_to(PROJECT))

    torch_device = resolve_torch_device(args.device)
    print("Torch device:", torch_device)
    print("CUDA available:", torch.cuda.is_available())

    if detector in {"yolo", "rtdetr"}:
        prediction_rows, mean_ms = predict_ultralytics(
            detector=detector,
            checkpoint_path=checkpoint_path,
            image_paths=image_paths,
            device_arg=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    else:
        prediction_rows, mean_ms = predict_torchvision(
            detector=detector,
            checkpoint_path=checkpoint_path,
            image_paths=image_paths,
            device=torch_device,
            conf=args.conf,
        )

    print(f"[{detector}] mean inference time: {mean_ms:.2f} ms/image")

    pred_path = save_predictions(detector, prediction_rows, args.limit)
    print("Saved predictions:", pred_path.relative_to(PROJECT))

    metrics = evaluate_detector(detector, prediction_rows, gt_by_image)

    metrics["checkpoint"] = str(checkpoint_path.relative_to(PROJECT))
    metrics["limit"] = args.limit
    metrics["confidence_threshold"] = args.conf
    metrics["image_size"] = args.imgsz

    metrics_path = save_metrics(detector, metrics, args.limit)
    print("Saved metrics:", metrics_path.relative_to(PROJECT))

    print("mAP50:", round(metrics["overall"]["mAP50"], 4))
    print("mAP50-95:", round(metrics["overall"]["mAP50_95"], 4))
    print("mean inference ms:", round(metrics["overall"]["mean_inference_ms"], 2))

    return metrics


def write_summary_table(metrics_list: List[Dict[str, Any]], limit: int | None):
    rows = []

    for metrics in metrics_list:
        row = {
            "detector": metrics["detector"],
            "num_images": metrics["num_images"],
            "mAP50": metrics["overall"]["mAP50"],
            "mAP50_95": metrics["overall"]["mAP50_95"],
            "mean_inference_ms": metrics["overall"]["mean_inference_ms"],
        }

        for class_name in CLASS_NAMES.values():
            row[f"{class_name}_AP50"] = metrics["classes"][class_name]["AP50"]
            row[f"{class_name}_AP50_95"] = metrics["classes"][class_name]["AP50_95"]

        rows.append(row)

    df = pd.DataFrame(rows)
    suffix = f"limit_{limit}" if limit is not None else "full"

    csv_path = TABLE_DIR / f"comparison_summary_{suffix}.csv"
    json_path = TABLE_DIR / f"comparison_summary_{suffix}.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print()
    print("=" * 100)
    print("Comparison summary")
    print("=" * 100)
    print(df.to_string(index=False))
    print()
    print("Saved CSV:", csv_path.relative_to(PROJECT))
    print("Saved JSON:", json_path.relative_to(PROJECT))


def main():
    args = parse_args()

    print("=" * 100)
    print("STEP 29 - Final local KITTI validation")
    print("=" * 100)
    print("Detector:", args.detector)
    print("Limit:", args.limit)
    print("Device:", args.device)
    print("Confidence threshold:", args.conf)

    registry = load_registry()
    image_paths = get_image_paths(args.limit)

    print("Validation images:", len(image_paths))
    print("Image dir:", IMAGE_VAL_DIR)
    print("Label dir:", LABEL_VAL_DIR)

    gt_by_image = load_all_ground_truth(image_paths)

    detectors = ["yolo", "rtdetr", "retinanet", "faster_rcnn"] if args.detector == "all" else [args.detector]

    metrics_list = []

    for detector in detectors:
        metrics = run_detector(detector, registry, image_paths, gt_by_image, args)
        metrics_list.append(metrics)

    write_summary_table(metrics_list, args.limit)

    print()
    print("=" * 100)
    print("STEP 29 COMPLETE ✅")
    print("Local KITTI validation finished.")
    print("Next: run full validation, then Step 30 benchmark/comparison outputs.")
    print("=" * 100)


if __name__ == "__main__":
    main()