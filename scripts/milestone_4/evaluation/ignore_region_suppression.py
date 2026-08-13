import json


def compute_iou(box_a, box_b):
    xa1, ya1, wa, ha = box_a
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb1, yb1, wb, hb = box_b
    xb2, yb2 = xb1 + wb, yb1 + hb

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = wa * ha
    area_b = wb * hb
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def suppress_dontcare_predictions(predictions, ignore_regions, min_iou_overlap=0.5):
    regions_by_image = {}
    for region in ignore_regions:
        regions_by_image.setdefault(region["image_id"], []).append(region["bbox"])

    suppressed_count = 0
    kept = []

    for pred in predictions:
        img_id = pred["image_id"]
        pred_box = pred["bbox"]
        regions = regions_by_image.get(img_id, [])

        is_ignored = False
        for region_box in regions:
            if compute_iou(pred_box, region_box) >= min_iou_overlap:
                is_ignored = True
                break

        if is_ignored:
            suppressed_count += 1
        else:
            kept.append(pred)

    return kept, suppressed_count
