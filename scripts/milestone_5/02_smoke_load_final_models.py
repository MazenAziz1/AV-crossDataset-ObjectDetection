from pathlib import Path
import json
import sys

import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)

PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

REGISTRY_PATH = PROJECT / "outputs" / "milestone_4" / "locked_final_checkpoints" / "final_checkpoint_registry.json"
IMAGE_VAL_DIR = PROJECT / "data" / "processed" / "milestone_3" / "images" / "kitti" / "val"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_5" / "smoke_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def pick_sample_image():
    images = sorted(IMAGE_VAL_DIR.glob("*.png"))
    if not images:
        raise RuntimeError(f"No validation images found in {IMAGE_VAL_DIR}")
    return images[0]


def summarize_torchvision_state_dict(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    print("Checkpoint epoch:", ckpt.get("epoch"))
    print("Checkpoint best_loss:", ckpt.get("best_loss"))
    print("State dict keys:", len(state))

    interesting = []
    for key, tensor in state.items():
        low = key.lower()
        if any(token in low for token in ["box_predictor", "classification_head", "regression_head", "cls_score", "bbox_pred"]):
            if hasattr(tensor, "shape"):
                interesting.append((key, tuple(tensor.shape)))

    for key, shape in interesting[:40]:
        print(" ", key, shape)

    return state


def try_load_model(detector: str, checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    candidates = []

    if detector == "faster_rcnn":
        for num_classes in [4, 3]:
            candidates.append((
                f"fasterrcnn_resnet50_fpn(num_classes={num_classes})",
                lambda n=num_classes: fasterrcnn_resnet50_fpn(
                    weights=None,
                    weights_backbone=None,
                    num_classes=n,
                ),
            ))

    elif detector == "retinanet":
        for num_classes in [3, 4]:
            candidates.append((
                f"retinanet_resnet50_fpn(num_classes={num_classes})",
                lambda n=num_classes: retinanet_resnet50_fpn(
                    weights=None,
                    weights_backbone=None,
                    num_classes=n,
                ),
            ))

    else:
        raise ValueError(f"Unsupported Torchvision detector: {detector}")

    errors = []

    for name, builder in candidates:
        print()
        print("Trying:", name)

        try:
            model = builder()
            missing, unexpected = model.load_state_dict(state, strict=False)

            if missing or unexpected:
                print("strict=False load had issues")
                print("Missing keys:", len(missing))
                print("Unexpected keys:", len(unexpected))

                # Now test strict=True so we do not accidentally accept wrong architecture.
                try:
                    model.load_state_dict(state, strict=True)
                except Exception as strict_exc:
                    print("strict=True failed:", strict_exc)
                    errors.append((name, str(strict_exc)))
                    continue

            else:
                model.load_state_dict(state, strict=True)

            print("LOAD OK:", name)
            return model, name, ckpt

        except Exception as exc:
            print("FAILED:", name)
            print(exc)
            errors.append((name, str(exc)))

    raise RuntimeError(f"Could not load {detector}. Tried: {errors}")


def smoke_predict_torchvision(detector: str, model, image_path: Path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image)

    with torch.no_grad():
        outputs = model([tensor])

    out = outputs[0]
    boxes = out.get("boxes", torch.empty((0, 4)))
    scores = out.get("scores", torch.empty((0,)))
    labels = out.get("labels", torch.empty((0,), dtype=torch.long))

    print("Prediction boxes:", len(boxes))

    if len(boxes) > 0:
        top_k = min(5, len(boxes))
        print("Top predictions:")
        for i in range(top_k):
            print(
                " ",
                "label=", int(labels[i].item()),
                "score=", round(float(scores[i].item()), 4),
                "box=", [round(float(x), 2) for x in boxes[i].tolist()],
            )

    return {
        "num_predictions": int(len(boxes)),
        "top_scores": [float(x) for x in scores[:5].tolist()],
        "top_labels": [int(x) for x in labels[:5].tolist()],
    }


def smoke_predict_ultralytics(detector: str, checkpoint_path: Path, image_path: Path):
    from ultralytics import YOLO

    print("Loading with ultralytics.YOLO:", checkpoint_path)
    model = YOLO(str(checkpoint_path))

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    num_predictions = 0 if boxes is None else len(boxes)

    print("Prediction boxes:", num_predictions)

    top_scores = []
    top_labels = []

    if boxes is not None and len(boxes) > 0:
        conf = boxes.conf.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()

        top_k = min(5, len(conf))
        print("Top predictions:")
        for i in range(top_k):
            print(
                " ",
                "label=", int(cls[i]),
                "score=", round(float(conf[i]), 4),
                "box=", [round(float(x), 2) for x in xyxy[i]],
            )

        top_scores = [float(x) for x in conf[:5]]
        top_labels = [int(x) for x in cls[:5]]

    return {
        "num_predictions": int(num_predictions),
        "top_scores": top_scores,
        "top_labels": top_labels,
    }


def main():
    print("=" * 100)
    print("STEP 28C - Smoke-load final models")
    print("=" * 100)

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")

    sample_image = pick_sample_image()
    print("Sample image:", sample_image.relative_to(PROJECT))

    registry = load_registry()
    models = registry["models"]

    smoke_summary = {
        "sample_image": str(sample_image.relative_to(PROJECT)),
        "models": {},
    }

    for detector in ["yolo", "rtdetr", "retinanet", "faster_rcnn"]:
        print()
        print("=" * 100)
        print("Detector:", detector)
        print("=" * 100)

        checkpoint_path = PROJECT / models[detector]["canonical_checkpoint"]
        print("Checkpoint:", checkpoint_path.relative_to(PROJECT))

        if detector in {"yolo", "rtdetr"}:
            prediction_summary = smoke_predict_ultralytics(detector, checkpoint_path, sample_image)
            smoke_summary["models"][detector] = {
                "checkpoint": str(checkpoint_path.relative_to(PROJECT)),
                "loader": "ultralytics.YOLO",
                "prediction_summary": prediction_summary,
            }

        else:
            print("Checkpoint structure:")
            summarize_torchvision_state_dict(checkpoint_path)

            model, constructor_name, ckpt = try_load_model(detector, checkpoint_path)
            prediction_summary = smoke_predict_torchvision(detector, model, sample_image)

            smoke_summary["models"][detector] = {
                "checkpoint": str(checkpoint_path.relative_to(PROJECT)),
                "loader": constructor_name,
                "checkpoint_epoch": ckpt.get("epoch"),
                "best_loss": ckpt.get("best_loss"),
                "prediction_summary": prediction_summary,
            }

    out_path = OUTPUT_DIR / "final_model_smoke_load_summary.json"
    out_path.write_text(json.dumps(smoke_summary, indent=2), encoding="utf-8")

    print()
    print("=" * 100)
    print("STEP 28C COMPLETE ✅")
    print("Smoke-load summary written to:", out_path)
    print("Next: build the final KITTI evaluator.")
    print("=" * 100)


if __name__ == "__main__":
    main()