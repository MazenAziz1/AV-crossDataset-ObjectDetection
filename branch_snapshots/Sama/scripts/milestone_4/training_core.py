import os
import json
import time
import copy
import csv
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image

from scripts.milestone_4.utilities.seeding import set_all_seeds, seed_worker, get_seed_generator
from scripts.milestone_4.utilities.checkpointing import (
    get_checkpoint_dir, get_best_path, get_last_path,
    save_resume_state, save_training_report, cleanup_old_checkpoints
)
from scripts.milestone_4.utilities.runtime_guard import RuntimeGuard


# ---------------------------------------------------------------------------
# COCO Dataset for Torchvision
# ---------------------------------------------------------------------------

class KITTICOCODataset(Dataset):
    def __init__(self, coco_json_path, images_dir, transforms=None, input_size=640):
        with open(coco_json_path) as f:
            coco = json.load(f)
        self.images_dir = Path(images_dir)
        self.input_size = input_size

        img_map = {img["id"]: img for img in coco["images"]}
        self.image_list = []
        for img in coco["images"]:
            anns = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
            self.image_list.append({
                "file_name": img["file_name"],
                "image_id": img["id"],
                "width": img["width"],
                "height": img["height"],
            })

        anns_by_image = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            anns_by_image.setdefault(iid, []).append(ann)
        self.anns_by_image = anns_by_image

        self.transforms = transforms
        if self.transforms is None:
            self.transforms = A.Compose([
                A.Resize(input_size, input_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels"], min_visibility=0.0
            ))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        info = self.image_list[idx]
        img_path = self.images_dir / info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        iid = info["image_id"]
        anns = self.anns_by_image.get(iid, [])

        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x2 = x + w
            y2 = y + h
            area = w * h
            if area < 1:
                continue
            boxes.append([x, y, x2, y2])
            labels.append(ann["category_id"])
            areas.append(area)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        areas = np.array(areas, dtype=np.float32) if areas else np.zeros((0,), dtype=np.float32)

        if len(boxes) > 0:
            transformed = self.transforms(image=image, bboxes=boxes.tolist(), labels=labels.tolist())
            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            transformed = self.transforms(image=image, bboxes=[], labels=[])
            image = transformed["image"]
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            areas = torch.zeros((0,), dtype=torch.float32)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([iid]),
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def _enable_backbone_checkpointing(model):
    import torch.utils.checkpoint as checkpoint

    backbone_body = model.backbone.body
    return_layers = getattr(backbone_body, "return_layers", {})

    for name in return_layers.keys():
        layer = backbone_body._modules.get(name)
        if layer is None:
            continue
        orig_forward = layer.forward

        def _make_ckpt(orig_fn):
            def _ckpt_forward(x):
                return checkpoint.checkpoint(orig_fn, x, use_reentrant=False)
            return _ckpt_forward

        layer.forward = _make_ckpt(orig_forward)


# ---------------------------------------------------------------------------
# Torchvision Trainer (Faster R-CNN / RetinaNet)
# ---------------------------------------------------------------------------

def train_torchvision(config, detector, run_type, ckpt_dir, runtime_guard, resume=False):
    set_all_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Device: {device}")
    if resume:
        print("[Trainer] Resume mode enabled.")

    project_root = Path(config["environment"]["project_root"])
    data_config = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Build dataset
    trainable_layers = train_cfg.get("trainable_backbone_layers", 5)
    if detector == "faster_rcnn":
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights, trainable_backbone_layers=trainable_layers)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    else:
        import torchvision
        from torchvision.models.detection import retinanet_resnet50_fpn_v2
        from torchvision.models.detection.retinanet import RetinaNetHead
        weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=trainable_layers)
        in_channels = model.backbone.out_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head = RetinaNetHead(in_channels, num_anchors, 4)

    model.to(device)

    if train_cfg.get("gradient_checkpointing", False):
        _enable_backbone_checkpointing(model)
        print("[Trainer] Gradient checkpointing enabled on backbone.")

    train_json = project_root / data_config["kitti_train_coco"]
    val_json = project_root / data_config["kitti_val_coco"]
    train_img_dir = project_root / data_config["kitti_train_images"]
    val_img_dir = project_root / data_config["kitti_val_images"]

    # Subset for tiny_overfit
    subset_size = None
    if run_type == "tiny_overfit":
        subset_size = 10
    elif run_type == "pilot":
        subset_size = 200

    train_aug = A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0))

    val_aug = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0))

    full_train_ds = KITTICOCODataset(str(train_json), str(train_img_dir), transforms=train_aug)
    val_ds = KITTICOCODataset(str(val_json), str(val_img_dir), transforms=val_aug)

    if subset_size:
        indices = list(range(min(subset_size, len(full_train_ds))))
        train_ds = torch.utils.data.Subset(full_train_ds, indices)
    else:
        train_ds = full_train_ds

    batch_size = train_cfg["batch_size"]
    accum_steps = train_cfg.get("gradient_accumulation_factor", 1)
    workers = train_cfg["dataloader_workers"]

    if run_type in ("tiny_overfit", "pilot"):
        accum_steps = 1

    g = get_seed_generator()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=workers, worker_init_fn=seed_worker, generator=g, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=min(workers, 4), pin_memory=True
    )

    # Optimizer and scheduler
    opt_cfg = train_cfg["optimizer"]
    opt_name = opt_cfg.get("name", "SGD")
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg["learning_rate"],
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=opt_cfg["learning_rate"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            nesterov=False,
        )

    t_max = opt_cfg.get("lr_scheduler_t_max", train_cfg["target_epochs"])
    warmup_epochs = opt_cfg.get("warmup_epochs", 0)

    if warmup_epochs > 0 and warmup_epochs < t_max:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max
        )
    scaler = torch.amp.GradScaler("cuda") if train_cfg.get("amp", True) else None

    target_epochs = train_cfg["target_epochs"]
    if run_type == "tiny_overfit":
        target_epochs = 5
    elif run_type == "pilot":
        target_epochs = 10

    # Defaults (may be overwritten by resume)
    best_map = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 20)
    history = {"epoch": [], "train_loss": [], "val_map": [], "val_map50": []}

    start_epoch = 1
    if resume:
        last_path = get_last_path(ckpt_dir)
        if not last_path.exists():
            print(f"[Trainer] No checkpoint found at {last_path}, starting fresh.")
        else:
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_map = ckpt.get("best_map", 0.0)
            best_epoch = ckpt.get("best_epoch", 0)
            patience_counter = ckpt.get("patience_counter", 0)
            history = ckpt.get("history", {"epoch": [], "train_loss": [], "val_map": [], "val_map50": []})
            print(f"[Trainer] Resumed from epoch {ckpt['epoch']} (best mAP: {best_map:.4f} at epoch {best_epoch})")
            print(f"[Trainer] Continuing from epoch {start_epoch} to {target_epochs}")

    for epoch in range(start_epoch, target_epochs + 1):
        if runtime_guard.should_stop():
            print("[Trainer] Runtime guard triggered stop.")
            break

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if scaler:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values()) / accum_steps
                scaler.scale(losses).backward()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) / accum_steps
                losses.backward()

            epoch_loss += losses.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        # Catch any leftover gradients
        if len(train_loader) % accum_steps != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        coco_predictions = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                if scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                else:
                    outputs = model(images)

                for img_idx, output in enumerate(outputs):
                    boxes = output["boxes"].cpu()
                    scores = output["scores"].cpu()
                    labels = output["labels"].cpu()
                    img_id = targets[img_idx]["image_id"].item()

                    for box, score, label in zip(boxes, scores, labels):
                        if score < 0.001:
                            continue
                        x1, y1, x2, y2 = box.tolist()
                        coco_predictions.append({
                            "image_id": img_id,
                            "category_id": int(label.item()),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score.item()),
                        })

        # Compute mAP
        map_val = 0.0
        map50_val = 0.0
        if coco_predictions:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            coco_gt = COCO(str(val_json))
            pred_file = ckpt_dir / f"pred_epoch_{epoch}.json"
            with open(pred_file, "w") as f:
                json.dump(coco_predictions, f)
            coco_dt = coco_gt.loadRes(str(pred_file))
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            map_val = coco_eval.stats[0]
            map50_val = coco_eval.stats[1]
            pred_file.unlink()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(epoch_loss))
        history["val_map"].append(float(map_val))
        history["val_map50"].append(float(map50_val))

        print(f"[Epoch {epoch:3d}/{target_epochs}] Loss: {epoch_loss:.4f} | "
              f"mAP: {map_val:.4f} | mAP50: {map50_val:.4f}")

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map": best_map,
            "history": history,
        }
        epoch_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, epoch_path)
        torch.save(ckpt, get_last_path(ckpt_dir))

        if map_val > best_map:
            best_map = map_val
            best_epoch = epoch
            patience_counter = 0
            torch.save(ckpt, get_best_path(ckpt_dir))
        else:
            patience_counter += 1

        cleanup_old_checkpoints(ckpt_dir, keep_last_n=3)

        if patience_counter >= patience:
            print(f"[Trainer] Early stopping at epoch {epoch} (patience={patience})")
            break

        runtime_guard.wait_for_interval(epoch, target_epochs)

    # Save final state
    exit_reason = "completed" if epoch >= target_epochs else "runtime_guard_stop" if runtime_guard.should_stop() else "early_stop"
    save_resume_state(ckpt_dir, epoch, best_map, detector, run_type, "001", exit_reason)

    report = {
        "detector": detector,
        "run_type": run_type,
        "device": str(device),
        "final_epoch": epoch,
        "best_epoch": best_epoch,
        "best_map": best_map,
        "exit_reason": exit_reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_training_report(ckpt_dir, report)

    return report


# ---------------------------------------------------------------------------
# Ultralytics Trainer (YOLO / RT-DETR)
# ---------------------------------------------------------------------------

def train_ultralytics(config, detector, run_type, ckpt_dir, runtime_guard):
    from ultralytics import YOLO, RTDETR

    project_root = Path(config["environment"]["project_root"])
    data_config = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    if detector == "yolo":
        model_class = YOLO
    else:
        model_class = RTDETR

    # Build data.yaml for ultralytics
    data_yaml = {
        "path": str(project_root),
        "train": str(data_config["kitti_train_images"]),
        "val": str(data_config["kitti_val_images"]),
        "names": {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"},
        "nc": 3,
    }
    data_yaml_path = ckpt_dir / "dataset.yaml"
    with open(data_yaml_path, "w") as f:
        json.dump(data_yaml, f, indent=2)

    # Load pretrained
    model = model_class(f"{detector}-l.pt" if detector == "rtdetr" else "yolov8s.pt")

    target_epochs = train_cfg["target_epochs"]
    if run_type == "tiny_overfit":
        target_epochs = 5
    elif run_type == "pilot":
        target_epochs = 10

    results = model.train(
        data=str(data_yaml_path),
        epochs=target_epochs,
        imgsz=640,
        batch=train_cfg["batch_size"],
        workers=train_cfg["dataloader_workers"],
        device=0,
        project=str(ckpt_dir.parent),
        name=run_type,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=train_cfg["optimizer"]["lr0"],
        lrf=train_cfg["optimizer"]["lrf"],
        momentum=train_cfg["optimizer"]["momentum"],
        weight_decay=train_cfg["optimizer"]["weight_decay"],
        warmup_epochs=train_cfg["optimizer"]["warmup_epochs"],
        warmup_momentum=train_cfg["optimizer"]["warmup_momentum"],
        warmup_bias_lr=train_cfg["optimizer"]["warmup_bias_lr"],
        patience=train_cfg["early_stopping_patience"],
        save=True,
        save_period=1,
        val=True,
        plots=False,
        amp=train_cfg.get("amp", True),
        seed=42,
        deterministic=True,
        verbose=True,
    )

    report = {
        "detector": detector,
        "run_type": run_type,
        "final_epoch": target_epochs,
        "results": str(results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_training_report(ckpt_dir, report)

    return report


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_training(config_path, detector, run_type, slot, resume=False,
                 max_runtime_hours=10.5, save_every_epochs=1, package_on_exit=False):
    import yaml

    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)

    shared_cfg_path = config_path.parent / "shared_training_policy.yaml"
    if shared_cfg_path.exists():
        with open(shared_cfg_path) as f:
            shared_cfg = yaml.safe_load(f)
    else:
        shared_cfg = {}

    config = {**shared_cfg, **model_cfg}

    framework = model_cfg["model"]["framework"]
    project_root = Path(config.get("environment", {}).get("project_root", "."))
    output_root = Path(config.get("environment", {}).get("output_root", project_root / "outputs/milestone_4"))
    os.makedirs(str(output_root), exist_ok=True)

    ckpt_dir = Path(str(get_checkpoint_dir(output_root, detector, run_type)))
    os.makedirs(str(ckpt_dir), exist_ok=True)

    guard = RuntimeGuard(max_hours=max_runtime_hours)

    print("=" * 60)
    print(f"Training: {detector} | Run: {run_type} | Slot: {slot}")
    print(f"Framework: {framework} | Output: {ckpt_dir}")
    print("=" * 60)

    if framework == "ultralytics":
        report = train_ultralytics(config, detector, run_type, ckpt_dir, guard)
    else:
        report = train_torchvision(config, detector, run_type, ckpt_dir, guard, resume=resume)

    print("=" * 60)
    print(f"Training complete. Report: {ckpt_dir / 'training_report.json'}")
    print("=" * 60)

    return report
