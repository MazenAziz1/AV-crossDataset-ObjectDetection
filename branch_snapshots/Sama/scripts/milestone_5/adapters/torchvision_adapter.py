import json
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from PIL import Image


def _build_model(detector, device):
    if detector == "faster_rcnn":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    elif detector == "retinanet":
        weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights)
        in_channels = model.backbone.out_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head = RetinaNetHead(in_channels, num_anchors, 4)
    else:
        raise ValueError(f"Unsupported detector: {detector}")

    model.to(device)
    model.eval()
    return model


def _load_state(checkpoint_path, model, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    return model


def run_torchvision_predictions(checkpoint_path, detector, images_dir, image_ids,
                                conf_threshold=0.001, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    model = _build_model(detector, device)
    model = _load_state(checkpoint_path, model, device)

    images_dir = Path(images_dir)
    image_id_map = {img["file_name"]: img["id"] for img in image_ids}

    predictions = []
    from torchvision.transforms import functional as F

    for fname, image_id in image_id_map.items():
        img_path = images_dir / fname
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image).to(device)
        # Match training-time preprocessing (Albumentations A.Normalize). The torchvision
        # model additionally normalizes internally; this duplication existed during training
        # and is replicated here so evaluation uses the model's trained input distribution.
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with torch.no_grad():
            output = model([image])[0]

        boxes = output["boxes"].cpu()
        scores = output["scores"].cpu()
        labels = output["labels"].cpu()

        for box, score, label in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            if w < 1 or h < 1:
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [x1, y1, w, h],
                "score": float(score.item()),
            })

    return predictions
