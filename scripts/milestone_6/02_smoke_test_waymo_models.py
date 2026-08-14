from pathlib import Path
import json
import time
from datetime import datetime

import torch
from PIL import Image

from ultralytics import YOLO
from torchvision.transforms import functional as F
from torchvision.models.detection import retinanet_resnet50_fpn, fasterrcnn_resnet50_fpn


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

REGISTRY_PATH = PROJECT / "outputs" / "milestone_4" / "locked_final_checkpoints" / "final_checkpoint_registry.json"

WAYMO_IMAGE_DIR = PROJECT / "data" / "processed" / "milestone_3" / "images" / "waymo" / "external"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_6" / "waymo_external_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_OUTPUT_JSON = OUTPUT_DIR / "smoke_test_summary.json"
SMOKE_OUTPUT_MD = OUTPUT_DIR / "smoke_test_summary.md"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SAMPLE_SIZE = 5

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
    """
    Robust registry parser.
    It searches nested dict/list structures for checkpoint paths related to a detector.
    Falls back to known locked paths if needed.
    """
    matches = []

    def walk(x, parent_key=""):
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
                    for k2, v2 in value.items():
                        if isinstance(v2, str):
                            v2_lower = v2.lower().replace("\\", "/")
                            if v2_lower.endswith(".pt") or v2_lower.endswith(".pth"):
                                matches.append(v2)

                walk(value, key_lower)

        elif isinstance(x, list):
            for item in x:
                walk(item, parent_key)

    walk(obj)

    # Prefer best checkpoint paths.
    best_matches = [m for m in matches if "best" in m.lower()]
    selected = best_matches[0] if best_matches else (matches[0] if matches else None)

    if selected:
        p = Path(selected)
        if not p.is_absolute():
            p = PROJECT / selected
        return p

    return FALLBACK_CHECKPOINTS[detector_name]


def load_registry_paths():
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = {}

    paths = {}
    for detector in ["yolo", "rtdetr", "retinanet", "faster_rcnn"]:
        paths[detector] = find_path_in_registry(registry, detector)

    return paths


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


def smoke_ultralytics(detector_name, checkpoint_path, images, device_name):
    model = YOLO(str(checkpoint_path))

    records = []
    total_time = 0.0

    for image_path in images:
        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.001,
            iou=0.70,
            device=device_name,
            verbose=False,
        )

        elapsed = time.perf_counter() - start
        total_time += elapsed

        boxes = results[0].boxes
        num_predictions = 0 if boxes is None else len(boxes)

        sample_classes = []
        if boxes is not None and boxes.cls is not None:
            sample_classes = [int(x) for x in boxes.cls[:10].detach().cpu().tolist()]

        records.append({
            "image": str(image_path.relative_to(PROJECT)),
            "num_predictions": int(num_predictions),
            "sample_predicted_classes": sample_classes,
            "elapsed_ms": elapsed * 1000.0,
        })

    return {
        "status": "PASSED",
        "checkpoint": str(checkpoint_path.relative_to(PROJECT)),
        "sample_images": len(images),
        "mean_elapsed_ms": total_time * 1000.0 / max(len(images), 1),
        "records": records,
    }


def smoke_torchvision(detector_name, checkpoint_path, images, device):
    model = load_torchvision_model(detector_name, checkpoint_path, device)

    records = []
    total_time = 0.0

    with torch.no_grad():
        for image_path in images:
            img = Image.open(image_path).convert("RGB")
            tensor = F.to_tensor(img).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            outputs = model([tensor])

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            total_time += elapsed

            output = outputs[0]
            labels = output.get("labels", torch.empty(0)).detach().cpu()
            scores = output.get("scores", torch.empty(0)).detach().cpu()

            # Torchvision labels are expected to be:
            # 1 = Vehicle, 2 = Pedestrian, 3 = Cyclist, 0 = background.
            sample_classes_raw = [int(x) for x in labels[:10].tolist()]
            sample_scores = [float(x) for x in scores[:10].tolist()]

            records.append({
                "image": str(image_path.relative_to(PROJECT)),
                "num_predictions": int(len(labels)),
                "sample_predicted_classes_raw_torchvision": sample_classes_raw,
                "sample_scores": sample_scores,
                "elapsed_ms": elapsed * 1000.0,
            })

    return {
        "status": "PASSED",
        "checkpoint": str(checkpoint_path.relative_to(PROJECT)),
        "sample_images": len(images),
        "mean_elapsed_ms": total_time * 1000.0 / max(len(images), 1),
        "records": records,
    }


def main():
    print("=" * 100)
    print("STEP 3/10 - Smoke test locked models on Waymo sample images")
    print("=" * 100)

    errors = []

    if not WAYMO_IMAGE_DIR.exists():
        errors.append(f"Missing Waymo image directory: {WAYMO_IMAGE_DIR}")

    if not REGISTRY_PATH.exists():
        errors.append(f"Missing checkpoint registry: {REGISTRY_PATH}")

    images = collect_images(WAYMO_IMAGE_DIR) if WAYMO_IMAGE_DIR.exists() else []

    if not images:
        errors.append("No Waymo images found.")

    sample_images = images[:SAMPLE_SIZE]

    checkpoint_paths = load_registry_paths()

    for detector, path in checkpoint_paths.items():
        if not path.exists():
            errors.append(f"Missing checkpoint for {detector}: {path}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        print("STEP 3/10 FAILED ❌")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ultralytics_device = 0 if device.type == "cuda" else "cpu"

    print("Device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0))

    print("Sample images:", len(sample_images))
    for image_path in sample_images:
        print(" -", image_path.relative_to(PROJECT))

    print()
    print("Checkpoints:")
    for detector, path in checkpoint_paths.items():
        print(f" - {detector}: {path.relative_to(PROJECT)}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "sample_size": len(sample_images),
        "checkpoint_registry": str(REGISTRY_PATH.relative_to(PROJECT)),
        "detectors": {},
        "errors": [],
    }

    for detector in ["yolo", "rtdetr", "retinanet", "faster_rcnn"]:
        print()
        print("-" * 100)
        print("Testing:", detector)
        print("-" * 100)

        try:
            checkpoint_path = checkpoint_paths[detector]

            if detector in ["yolo", "rtdetr"]:
                result = smoke_ultralytics(
                    detector,
                    checkpoint_path,
                    sample_images,
                    ultralytics_device,
                )
            else:
                result = smoke_torchvision(
                    detector,
                    checkpoint_path,
                    sample_images,
                    device,
                )

            summary["detectors"][detector] = result

            print("PASSED:", detector)
            print("Mean elapsed ms:", round(result["mean_elapsed_ms"], 2))

        except Exception as exc:
            summary["status"] = "FAILED"
            error_message = f"{detector} failed: {repr(exc)}"
            summary["errors"].append(error_message)
            print("FAILED:", error_message)

    SMOKE_OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 6 - Waymo Smoke Test")
    md.append("")
    md.append(f"Created at: `{summary['created_at']}`")
    md.append("")
    md.append(f"Status: **{summary['status']}**")
    md.append("")
    md.append(f"Device: `{summary['device']}`")
    md.append("")
    md.append(f"Sample images: `{summary['sample_size']}`")
    md.append("")
    md.append("## Detector Smoke Results")
    md.append("")
    md.append("| Detector | Status | Mean elapsed ms | Checkpoint |")
    md.append("|---|---:|---:|---|")

    for detector, result in summary["detectors"].items():
        md.append(
            f"| {detector} | {result['status']} | "
            f"{result['mean_elapsed_ms']:.2f} | `{result['checkpoint']}` |"
        )

    md.append("")
    md.append("## Errors")
    md.append("")
    if summary["errors"]:
        for error in summary["errors"]:
            md.append(f"- {error}")
    else:
        md.append("- None")

    SMOKE_OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Output JSON:", SMOKE_OUTPUT_JSON)
    print("Output MD:", SMOKE_OUTPUT_MD)

    if summary["status"] != "PASSED":
        print("STEP 3/10 FAILED ❌")
        raise SystemExit(1)

    print("STEP 3/10 COMPLETE ✅")
    print("All four locked models loaded and produced predictions on Waymo sample images.")
    print("=" * 100)


if __name__ == "__main__":
    main()