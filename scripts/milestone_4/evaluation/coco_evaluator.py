import json
import tempfile
import os
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_predictions(gt_json_path, predictions):
    gt_json_path = str(gt_json_path)
    coco_gt = COCO(gt_json_path)

    tmp_dir = tempfile.mkdtemp()
    pred_path = os.path.join(tmp_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    metrics = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_max_1": float(stats[6]),
        "AR_max_10": float(stats[7]),
        "AR_max_100": float(stats[8]),
    }

    per_class = {}
    cat_ids = coco_gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    for cat_id in cat_ids:
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        precisions = coco_eval.eval["precision"]
        if precisions.size == 0:
            ap50_95 = 0.0
            ap50 = 0.0
        else:
            ap50_95 = float(precisions[:, :, 0, 0, 2].mean())
            ap50 = float(precisions[0, :, 0, 0, 2].mean())
        per_class[int(cat_id)] = {
            "name": cat_names[cat_id],
            "AP_50_95": ap50_95,
            "AP_50": ap50,
        }

    os.remove(pred_path)
    os.rmdir(tmp_dir)

    return metrics, per_class
