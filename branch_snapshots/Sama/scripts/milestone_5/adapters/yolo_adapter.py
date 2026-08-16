import os
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def run_yolo_predictions(checkpoint_path, images_dir, image_ids, conf_threshold=0.001,
                         imgsz=640, batch=16, device="cuda"):
    model = YOLO(str(checkpoint_path))
    device = device if torch.cuda.is_available() else "cpu"

    images_dir = Path(images_dir)
    predictions = []

    image_id_map = {img["file_name"]: img["id"] for img in image_ids}

    results = model.predict(
        source=str(images_dir),
        conf=conf_threshold,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
    )

    for r in results:
        fname = Path(r.path).name
        if fname not in image_id_map:
            continue
        image_id = image_id_map[fname]

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            if w < 1 or h < 1:
                continue
            cls = int(box.cls[0].item())
            score = float(box.conf[0].item())
            category_id = cls + 1  # YOLO 0,1,2 -> COCO 1,2,3

            predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "score": score,
            })

    return predictions
