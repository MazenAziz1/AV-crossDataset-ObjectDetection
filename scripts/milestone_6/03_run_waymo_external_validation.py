from pathlib import Path
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ultralytics import YOLO
from torchvision.transforms import functional as F
from torchvision.models.detection import retinanet_resnet50_fpn, fasterrcnn_resnet50_fpn


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

REGISTRY_PATH = PROJECT / "outputs" / "milestone_4" / "locked_final_checkpoints" / "final_checkpoint_registry.json"

WAYMO_IMAGE_DIR = PROJECT / "data" / "processed" / "milestone_3" / "images" / "waymo" / "external"
WAYMO_LABEL_DIR = PROJECT / "data" / "processed" / "milestone_3" / "labels" / "waymo" / "external"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_6" / "waymo_external_validation"
METRICS_DIR = OUTPUT_DIR / "metrics"
TABLES_DIR = OUTPUT_DIR / "tables"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

for d in [OUTPUT_DIR, METRICS_DIR, TABLES_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "Vehicle",
    1: "Pedestrian",
    2: "Cyclist",
}

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)

CONF_THRESHOLD = 0.001
ULTRALYTICS_IOU_NMS = 0.70
IMAGE_SIZE = 640

SAVE_PREDICTIONS = True

FALLBACK_CHECKPOINTS = {
    "yolo": PROJECT / "outputs" / "milestone_4" / "checkpoints" / "yolo" / "yolo_final_20260813_153831" / "weights" / "best.pt",
    "rtdetr": PROJECT / "outputs" / "milestone_4" / "checkpoints" / "rtdetr" / "rtdetr_final_20260813_215051" / "weights" / "best.pt",
    "retinanet": PROJECT / "outputs" / "milestone_4" / "checkpoints" / "retinanet" / "retinanet_final_resume_if_needed_20260814_100422" / "best.pth",
    "faster_rcnn": PROJECT / "outputs" / "milestone_4" / "checkpoints" / "faster_rcnn" / "faster_rcnn_final_resume_if_needed_20260814_100004" / "best.pth",
}


def collect_images(image_dir: Path):
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_path_in_registry(obj, detector_name):
    matches = []

    def walk(x):
        if isinstance(x, dict):
            for key, value in x.items():
                key_lower = str(key).lower()

                if isinstance(value, str):
                    value_lower = value.lower().replace("\\", "/")
                    if detector_name in value_lower and (
                        value_lower.endswith(".pt") or value_lower.endswith(".pth")
                    ):
                        matches.append(value)

                if detector_name in key_lower and isinstance(value, dict):
                    for _, v2 in value.items():
                        if isinstance(v2, str):
                            v2_lower = v2.lower().replace("\\", "/")
                            if v2_lower.endswith(".pt") or v2_lower.endswith(".pth"):
                                matches.append(v2)

                walk(value)

        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)

    best_matches = [m for m in matches if "best" in m.lower()]
    selected = best_matches[0] if best_matches else (matches[0] if matches else None)

    if selected:
        p = Path(selected)
        if not p.is_absolute():
            p = PROJECT / selected
        return p

    return FALLBACK_CHECKPOINTS[detector_name]


def load_checkpoint_paths():
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = {}

    return {
        detector: find_path_in_registry(registry, detector)
        for detector in ["yolo", "rtdetr", "retinanet", "faster_rcnn"]
    }


def yolo_label_to_xyxy(values, image_width, image_height):
    class_id, xc, yc, w, h = values

    x1 = (xc - w / 2.0) * image_width
    y1 = (yc - h / 2.0) * image_height
    x2 = (xc + w / 2.0) * image_width
    y2 = (yc + h / 2.0) * image_height

    x1 = max(0.0, min(float(x1), image_width))
    y1 = max(0.0, min(float(y1), image_height))
    x2 = max(0.0, min(float(x2), image_width))
    y2 = max(0.0, min(float(y2), image_height))

    return {
        "class_id": int(class_id),
        "bbox": [x1, y1, x2, y2],
    }


def load_ground_truth(images):
    print("Loading Waymo ground truth labels...")

    ground_truth = {}
    image_sizes = {}
    total_annotations = 0

    for idx, image_path in enumerate(images, start=1):
        image_id = image_path.stem
        label_path = WAYMO_LABEL_DIR / f"{image_path.stem}.txt"

        with Image.open(image_path) as img:
            width, height = img.size

        image_sizes[image_id] = {
            "width": width,
            "height": height,
            "image_path": str(image_path.relative_to(PROJECT)),
        }

        anns = []

        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid label line in {label_path}: {line}")

                class_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])

                if class_id not in CLASS_NAMES:
                    raise ValueError(f"Unexpected class id {class_id} in {label_path}")

                ann = yolo_label_to_xyxy(
                    [class_id, xc, yc, w, h],
                    width,
                    height,
                )
                anns.append(ann)

        ground_truth[image_id] = anns
        total_annotations += len(anns)

        if idx % 200 == 0:
            print(f"  loaded {idx}/{len(images)} images")

    print("Ground truth loaded.")
    print("Total annotations:", total_annotations)

    return ground_truth, image_sizes


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def compute_ap_for_class(predictions, ground_truth, class_id, iou_threshold):
    gt_by_image = {}
    total_gt = 0

    for image_id, anns in ground_truth.items():
        class_boxes = [ann["bbox"] for ann in anns if ann["class_id"] == class_id]
        gt_by_image[image_id] = {
            "boxes": class_boxes,
            "matched": [False] * len(class_boxes),
        }
        total_gt += len(class_boxes)

    if total_gt == 0:
        return None

    class_predictions = [
        pred for pred in predictions
        if pred["class_id"] == class_id
    ]

    class_predictions.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(class_predictions), dtype=np.float32)
    fp = np.zeros(len(class_predictions), dtype=np.float32)

    for idx, pred in enumerate(class_predictions):
        image_id = pred["image_id"]
        pred_box = pred["bbox"]

        image_gt = gt_by_image.get(image_id, {"boxes": [], "matched": []})
        gt_boxes = image_gt["boxes"]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if image_gt["matched"][gt_idx]:
                continue

            iou = box_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[idx] = 1.0
            image_gt["matched"][best_gt_idx] = True
        else:
            fp[idx] = 1.0

    if len(class_predictions) == 0:
        return 0.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recall = cumulative_tp / max(total_gt, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)

    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = []

    for r in recall_points:
        valid_precisions = precision[recall >= r]
        interpolated_precisions.append(
            np.max(valid_precisions) if valid_precisions.size else 0.0
        )

    return float(np.mean(interpolated_precisions))


def compute_metrics(predictions, ground_truth):
    class_metrics = {}
    ap_matrix = {}

    for class_id, class_name in CLASS_NAMES.items():
        ap_values = {}

        for iou_threshold in IOU_THRESHOLDS:
            ap = compute_ap_for_class(
                predictions,
                ground_truth,
                class_id,
                float(iou_threshold),
            )
            ap_values[str(float(iou_threshold))] = ap

        ap50 = ap_values["0.5"]
        valid_aps = [v for v in ap_values.values() if v is not None]
        ap50_95 = float(np.mean(valid_aps)) if valid_aps else None

        class_metrics[class_name] = {
            "AP50": ap50,
            "AP50_95": ap50_95,
            "AP_by_IoU": ap_values,
        }

        ap_matrix[class_id] = ap_values

    map50_values = [
        class_metrics[name]["AP50"]
        for name in CLASS_NAMES.values()
        if class_metrics[name]["AP50"] is not None
    ]

    map50_95_values = [
        class_metrics[name]["AP50_95"]
        for name in CLASS_NAMES.values()
        if class_metrics[name]["AP50_95"] is not None
    ]

    return {
        "mAP50": float(np.mean(map50_values)) if map50_values else None,
        "mAP50_95": float(np.mean(map50_95_values)) if map50_95_values else None,
        "class_metrics": class_metrics,
    }


def load_torchvision_model(detector_name, checkpoint_path, device):
    if detector_name == "retinanet":
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=4,
        )
    elif detector_name == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=4,
        )
    else:
        raise ValueError(f"Unsupported torchvision detector: {detector_name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def run_ultralytics_detector(detector_name, checkpoint_path, images, device_name):
    model = YOLO(str(checkpoint_path))

    predictions = []
    inference_times = []

    prediction_file = PREDICTIONS_DIR / f"{detector_name}_waymo_predictions.jsonl"

    if prediction_file.exists():
        prediction_file.unlink()

    print(f"Running {detector_name} on {len(images)} Waymo images...")

    with prediction_file.open("w", encoding="utf-8") as f:
        for idx, image_path in enumerate(images, start=1):
            image_id = image_path.stem

            start = time.perf_counter()

            results = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                iou=ULTRALYTICS_IOU_NMS,
                device=device_name,
                verbose=False,
            )

            elapsed = time.perf_counter() - start
            inference_times.append(elapsed * 1000.0)

            boxes = results[0].boxes

            image_predictions = []

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)

                for box, score, class_id in zip(xyxy, confs, classes):
                    if int(class_id) not in CLASS_NAMES:
                        continue

                    pred = {
                        "image_id": image_id,
                        "image_path": str(image_path.relative_to(PROJECT)),
                        "class_id": int(class_id),
                        "class_name": CLASS_NAMES[int(class_id)],
                        "score": float(score),
                        "bbox": [float(x) for x in box.tolist()],
                    }

                    predictions.append(pred)
                    image_predictions.append(pred)

            if SAVE_PREDICTIONS:
                f.write(json.dumps({
                    "image_id": image_id,
                    "image_path": str(image_path.relative_to(PROJECT)),
                    "predictions": image_predictions,
                }) + "\n")

            if idx % 100 == 0 or idx == len(images):
                mean_ms = float(np.mean(inference_times))
                print(f"  {detector_name}: {idx}/{len(images)} images | mean {mean_ms:.2f} ms")

    return predictions, inference_times


def run_torchvision_detector(detector_name, checkpoint_path, images, device):
    model = load_torchvision_model(detector_name, checkpoint_path, device)

    predictions = []
    inference_times = []

    prediction_file = PREDICTIONS_DIR / f"{detector_name}_waymo_predictions.jsonl"

    if prediction_file.exists():
        prediction_file.unlink()

    print(f"Running {detector_name} on {len(images)} Waymo images...")

    with torch.no_grad(), prediction_file.open("w", encoding="utf-8") as f:
        for idx, image_path in enumerate(images, start=1):
            image_id = image_path.stem

            image = Image.open(image_path).convert("RGB")
            tensor = F.to_tensor(image).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            outputs = model([tensor])

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            inference_times.append(elapsed * 1000.0)

            output = outputs[0]

            boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
            scores = output.get("scores", torch.empty((0,))).detach().cpu().numpy()
            labels = output.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().numpy()

            image_predictions = []

            for box, score, raw_label in zip(boxes, scores, labels):
                # Torchvision detector format:
                # 0 = background, 1 = Vehicle, 2 = Pedestrian, 3 = Cyclist
                class_id = int(raw_label) - 1

                if class_id not in CLASS_NAMES:
                    continue

                pred = {
                    "image_id": image_id,
                    "image_path": str(image_path.relative_to(PROJECT)),
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "score": float(score),
                    "bbox": [float(x) for x in box.tolist()],
                }

                predictions.append(pred)
                image_predictions.append(pred)

            if SAVE_PREDICTIONS:
                f.write(json.dumps({
                    "image_id": image_id,
                    "image_path": str(image_path.relative_to(PROJECT)),
                    "predictions": image_predictions,
                }) + "\n")

            if idx % 100 == 0 or idx == len(images):
                mean_ms = float(np.mean(inference_times))
                print(f"  {detector_name}: {idx}/{len(images)} images | mean {mean_ms:.2f} ms")

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return predictions, inference_times


def make_summary_row(detector_name, metrics, inference_times, num_images):
    class_metrics = metrics["class_metrics"]

    row = {
        "detector": detector_name,
        "num_images": num_images,
        "mAP50": metrics["mAP50"],
        "mAP50_95": metrics["mAP50_95"],
        "mean_inference_ms": float(np.mean(inference_times)) if inference_times else None,
    }

    for class_name in CLASS_NAMES.values():
        row[f"{class_name}_AP50"] = class_metrics[class_name]["AP50"]
        row[f"{class_name}_AP50_95"] = class_metrics[class_name]["AP50_95"]

    return row


def main():
    print("=" * 100)
    print("STEP 4/10 - Full Waymo external validation")
    print("=" * 100)

    errors = []

    if not WAYMO_IMAGE_DIR.exists():
        errors.append(f"Missing Waymo image directory: {WAYMO_IMAGE_DIR}")

    if not WAYMO_LABEL_DIR.exists():
        errors.append(f"Missing Waymo label directory: {WAYMO_LABEL_DIR}")

    if not REGISTRY_PATH.exists():
        errors.append(f"Missing checkpoint registry: {REGISTRY_PATH}")

    images = collect_images(WAYMO_IMAGE_DIR) if WAYMO_IMAGE_DIR.exists() else []

    if not images:
        errors.append("No Waymo images found.")

    checkpoint_paths = load_checkpoint_paths()

    for detector, path in checkpoint_paths.items():
        if not path.exists():
            errors.append(f"Missing checkpoint for {detector}: {path}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        print("STEP 4/10 FAILED ❌")
        raise SystemExit(1)

    print("Images:", len(images))
    print("Labels dir:", WAYMO_LABEL_DIR)
    print("Output dir:", OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ultralytics_device = 0 if device.type == "cuda" else "cpu"

    print("Device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0))

    print()
    print("Checkpoints:")
    for detector, path in checkpoint_paths.items():
        print(f" - {detector}: {path.relative_to(PROJECT)}")

    ground_truth, image_sizes = load_ground_truth(images)

    all_rows = []

    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "Waymo external subset",
        "num_images": len(images),
        "class_mapping": CLASS_NAMES,
        "iou_thresholds": [float(x) for x in IOU_THRESHOLDS],
        "confidence_threshold": CONF_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "policy": "No retraining. Locked Milestone 4/5 KITTI-trained checkpoints only.",
        "detectors": {},
    }

    for detector_name in ["yolo", "rtdetr", "retinanet", "faster_rcnn"]:
        print()
        print("=" * 100)
        print("Evaluating detector:", detector_name)
        print("=" * 100)

        checkpoint_path = checkpoint_paths[detector_name]
        detector_start = time.perf_counter()

        if detector_name in ["yolo", "rtdetr"]:
            predictions, inference_times = run_ultralytics_detector(
                detector_name,
                checkpoint_path,
                images,
                ultralytics_device,
            )
        else:
            predictions, inference_times = run_torchvision_detector(
                detector_name,
                checkpoint_path,
                images,
                device,
            )

        print("Computing AP metrics for:", detector_name)
        metrics = compute_metrics(predictions, ground_truth)

        elapsed_total = time.perf_counter() - detector_start

        metrics_payload = {
            "detector": detector_name,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT)),
            "dataset": "Waymo external subset",
            "num_images": len(images),
            "num_predictions": len(predictions),
            "mean_inference_ms": float(np.mean(inference_times)) if inference_times else None,
            "median_inference_ms": float(np.median(inference_times)) if inference_times else None,
            "total_detector_runtime_seconds": elapsed_total,
            "metrics": metrics,
        }

        metrics_path = METRICS_DIR / f"{detector_name}_waymo_metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        row = make_summary_row(detector_name, metrics, inference_times, len(images))
        all_rows.append(row)

        run_metadata["detectors"][detector_name] = {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT)),
            "metrics_file": str(metrics_path.relative_to(PROJECT)),
            "num_predictions": len(predictions),
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "mean_inference_ms": float(np.mean(inference_times)) if inference_times else None,
        }

        print()
        print("Result:", detector_name)
        print("  mAP50:", round(metrics["mAP50"], 6))
        print("  mAP50_95:", round(metrics["mAP50_95"], 6))
        print("  mean inference ms:", round(row["mean_inference_ms"], 3))
        print("  predictions:", len(predictions))
        print("  metrics saved:", metrics_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)

    summary_csv = TABLES_DIR / "waymo_external_summary.csv"
    summary_json = TABLES_DIR / "waymo_external_summary.json"
    metadata_json = OUTPUT_DIR / "waymo_external_validation_run_metadata.json"

    df.to_csv(summary_csv, index=False)
    summary_json.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    metadata_json.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print()
    print("=" * 100)
    print("Waymo external validation summary")
    print("=" * 100)
    print(df.to_string(index=False))

    print()
    print("Created:", summary_csv)
    print("Created:", summary_json)
    print("Created:", metadata_json)

    print()
    print("STEP 4/10 COMPLETE ✅")
    print("Full Waymo external validation finished for all four locked models.")
    print("=" * 100)


if __name__ == "__main__":
    main()