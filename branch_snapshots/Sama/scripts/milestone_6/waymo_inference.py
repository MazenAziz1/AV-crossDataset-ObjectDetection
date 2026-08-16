"""Shared Waymo inference helpers for Milestone 6.

These helpers mirror the Milestone 4 adapter output contract so that the
Milestone 4/5 evaluator (pycocotools COCOeval) can be reused unchanged:

    - predictions are lists of dicts:
        {"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}
      where category_id is the canonical COCO id (1=Vehicle, 2=Pedestrian, 3=Cyclist)
      and bbox is absolute xywh in pixels (640x640).
    - inference time is measured per real image and returned alongside predictions.

Ultralytics (YOLO / RT-DETR) and Torchvision (RetinaNet / Faster R-CNN) are
loaded exactly as in Milestone 4 evaluation, and the Torchvision ImageNet
normalization that matched training is preserved.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_checkpoint_paths(project_root):
    """Read the Milestone 4 final checkpoint registry (CSV) -> {detector: Path}."""
    import csv

    registry_path = project_root / "outputs" / "milestone_4" / "manifests" / "final_checkpoint_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing checkpoint registry: {registry_path}")

    paths = {}
    with open(registry_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths[row["detector"]] = Path(row["checkpoint_path"])

    for det in ["yolo", "rtdetr", "retinanet", "faster_rcnn"]:
        if det not in paths or not paths[det].exists():
            raise FileNotFoundError(f"Checkpoint for {det} missing: {paths.get(det)}")
    return paths


def _resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg in {"0", "cuda", "cuda:0"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_ultralytics(detector, checkpoint_path, image_paths, image_id_map, device_arg="auto",
                        conf=0.001, iou=0.70, max_det=300, imgsz=640):
    """Run YOLO or RT-DETR per-image, returning COCO-format predictions + per-image ms.

    image_id_map: dict mapping file_name (with extension) -> COCO image id.
    """
    if detector == "yolo":
        from ultralytics import YOLO
        model = YOLO(str(checkpoint_path))
    elif detector == "rtdetr":
        from ultralytics import RTDETR
        model = RTDETR(str(checkpoint_path))
    else:
        raise ValueError(f"Not an ultralytics detector: {detector}")

    torch_device = _resolve_device(device_arg)
    ultra_device = 0 if torch_device.type == "cuda" else "cpu"

    predictions = []
    times_ms = []

    for image_path in image_paths:
        image_id = image_id_map[image_path.name]

        start = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=ultra_device,
            verbose=False,
        )
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)

        for box, score, cls in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            if w < 1 or h < 1:
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": int(cls) + 1,  # YOLO/RT-DETR 0,1,2 -> COCO 1,2,3
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
            })

    return predictions, times_ms


def predict_torchvision(detector, checkpoint_path, image_paths, image_id_map, device_arg="auto", conf=0.001):
    """Run RetinaNet / Faster R-CNN per-image, returning COCO-format predictions + per-image ms.

    image_id_map: dict mapping file_name (with extension) -> COCO image id.
    """
    from scripts.milestone_5.adapters.torchvision_adapter import _build_model, _load_state
    from torchvision.transforms import functional as F
    from PIL import Image

    torch_device = _resolve_device(device_arg)
    model = _build_model(detector, torch_device)
    model = _load_state(checkpoint_path, model, torch_device)

    predictions = []
    times_ms = []

    for image_path in image_paths:
        image_id = image_id_map[image_path.name]
        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image).to(torch_device)
        # Match training-time preprocessing (see torchvision_adapter.run_torchvision_predictions).
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        start = time.perf_counter()
        with torch.no_grad():
            output = model([image])[0]
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy().astype(int)

        for box, score, label in zip(boxes, scores, labels):
            if float(score) < conf:
                continue
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            if w < 1 or h < 1:
                continue
            # Torchvision label 1,2,3 equals COCO id; label 0 (background) ignored.
            if int(label) not in (1, 2, 3):
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
            })

    return predictions, times_ms
