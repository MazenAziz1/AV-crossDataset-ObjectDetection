"""Shared constants and helpers for Milestone 7 robustness/failure analysis."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Frozen three-class contract (COCO category ids).
CLASS_NAMES = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}
CLASS_IDS = [1, 2, 3]
# Safety-priority vulnerable road users.
SAFETY_CLASSES = [2, 3]

DETECTORS = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]
DETECTOR_DISPLAY = {
    "yolo": "YOLOv8s",
    "rtdetr": "RT-DETR-L",
    "retinanet": "RetinaNet",
    "faster_rcnn": "Faster R-CNN",
}
DATASETS = ["kitti", "waymo"]
DATASET_DISPLAY = {"kitti": "KITTI", "waymo": "Waymo"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

M5_PRED_DIR = PROJECT_ROOT / "outputs" / "milestone_5" / "predictions" / "kitti_validation"
M6_PRED_DIR = PROJECT_ROOT / "outputs" / "milestone_6" / "waymo_external_validation" / "predictions"
KITTI_GT = PROJECT_ROOT / "data" / "processed" / "milestone_3" / "annotations" / "coco" / "kitti_val.json"
WAYMO_GT = PROJECT_ROOT / "data" / "processed" / "milestone_3" / "annotations" / "coco" / "waymo_external.json"
KITTI_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "milestone_3" / "images" / "kitti" / "val"
WAYMO_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "milestone_3" / "images" / "waymo" / "external"

M5_METRICS_DIR = PROJECT_ROOT / "outputs" / "milestone_5" / "metrics" / "kitti_validation"
M6_METRICS_DIR = PROJECT_ROOT / "outputs" / "milestone_6" / "waymo_external_validation" / "metrics"
M6_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "milestone_6" / "waymo_external_validation" / "tables" / "waymo_external_summary.csv"
REGISTRY = PROJECT_ROOT / "outputs" / "milestone_4" / "manifests" / "final_checkpoint_registry.csv"

M7_OUT = PROJECT_ROOT / "outputs" / "milestone_7"
M7_CFG = PROJECT_ROOT / "configs" / "analysis" / "milestone_7"


def load_coco(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path, conf):
    """Load prediction JSONL, keeping only rows with confidence >= conf."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["confidence"] >= conf:
                rows.append(r)
    return rows


def load_bins():
    """Read the frozen object-size bins (reused from Milestone 5 policy)."""
    import yaml
    with open(M7_CFG / "object_size_bins.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def size_category(area, bins):
    """Map a pixel area to small/medium/large using the frozen COCO bins."""
    b = bins["bins"]
    if area < b["small"]["max_area"]:
        return "small"
    if area < b["medium"]["max_area"]:
        return "medium"
    return "large"


def pred_path(dataset, detector):
    if dataset == "kitti":
        return M5_PRED_DIR / f"{detector}_predictions.jsonl"
    return M6_PRED_DIR / f"{detector}_waymo_predictions.jsonl"


def gt_path(dataset):
    return KITTI_GT if dataset == "kitti" else WAYMO_GT


def img_dir(dataset):
    return KITTI_IMG_DIR if dataset == "kitti" else WAYMO_IMG_DIR
