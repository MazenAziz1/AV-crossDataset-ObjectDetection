from pathlib import Path
from datetime import datetime
import csv
import json
import heapq
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

DETECTION_INDEX_CSV = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "detection_error_index.csv"
TOP_SAFETY_IMAGES_CSV = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "top_safety_critical_images.csv"
CANDIDATES_CSV = PROJECT / "outputs" / "milestone_7" / "safety_error_analysis" / "failure_case_candidate_rows.csv"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_7" / "failure_cases"
IMAGE_DIR = OUTPUT_DIR / "images"
PANEL_DIR = OUTPUT_DIR / "panels"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_JSON = OUTPUT_DIR / "failure_case_manifest.json"
MANIFEST_MD = OUTPUT_DIR / "FAILURE_CASE_GALLERY.md"

PANEL_WAYMO = PANEL_DIR / "failure_case_panel_waymo.png"
PANEL_KITTI = PANEL_DIR / "failure_case_panel_kitti.png"
PANEL_SAFETY = PANEL_DIR / "failure_case_panel_safety_vru.png"

MAX_SINGLE_EXAMPLES_PER_TYPE = 16
MAX_TOP_SAFETY_IMAGES = 16
MAX_FALSE_NEGATIVE_HEAP = 100
MAX_PANEL_IMAGES = 12

COLORS = {
    "ground_truth": (0, 180, 0),
    "true_positive": (0, 120, 255),
    "false_negative": (255, 0, 0),
    "false_positive": (255, 150, 0),
    "localization_error": (255, 220, 0),
    "class_confusion": (180, 0, 255),
    "duplicate_detection": (0, 180, 255),
    "text_bg": (0, 0, 0),
    "text_fg": (255, 255, 255),
}

SAFETY_CLASSES = {"Pedestrian", "Cyclist"}

TYPE_DISPLAY = {
    "false_negative": "False Negative",
    "false_positive": "False Positive",
    "localization_error": "Localization Error",
    "class_confusion": "Class Confusion",
    "duplicate_detection": "Duplicate Detection",
    "safety_composite": "Safety-Critical Misses",
}


def safe_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        if value == "" or value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def project_path(rel_path):
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return PROJECT / rel_path


def load_font(size=16):
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue

    return ImageFont.load_default()


FONT = load_font(16)
SMALL_FONT = load_font(13)


def get_box(row, prefix):
    values = [
        safe_float(row.get(f"{prefix}_x1")),
        safe_float(row.get(f"{prefix}_y1")),
        safe_float(row.get(f"{prefix}_x2")),
        safe_float(row.get(f"{prefix}_y2")),
    ]

    if any(v is None for v in values):
        return None

    x1, y1, x2, y2 = values

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def scale_box(box, sx, sy):
    return [
        box[0] * sx,
        box[1] * sy,
        box[2] * sx,
        box[3] * sy,
    ]


def draw_label(draw, xy, text, color):
    x, y = xy
    y = max(0, y - 18)

    try:
        bbox = draw.textbbox((x, y), text, font=SMALL_FONT)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w = len(text) * 8
        text_h = 14

    draw.rectangle([x, y, x + text_w + 6, y + text_h + 4], fill=color)
    draw.text((x + 3, y + 2), text, fill=COLORS["text_fg"], font=SMALL_FONT)


def draw_box(draw, box, color, label, width=4):
    x1, y1, x2, y2 = box

    for offset in range(width):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color,
        )

    draw_label(draw, (x1, y1), label, color)


def add_header(image, title, subtitle):
    width, height = image.size
    header_h = 70

    canvas = Image.new("RGB", (width, height + header_h), (245, 245, 245))
    canvas.paste(image, (0, header_h))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, header_h], fill=(20, 20, 20))
    draw.text((12, 10), title, fill=(255, 255, 255), font=FONT)
    draw.text((12, 38), subtitle, fill=(230, 230, 230), font=SMALL_FONT)

    return canvas


def load_and_resize_image(image_path, max_width=1100):
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size

    if original_w > max_width:
        new_w = max_width
        new_h = int(original_h * (new_w / original_w))
        image = image.resize((new_w, new_h), Image.LANCZOS)

    new_w, new_h = image.size
    sx = new_w / original_w
    sy = new_h / original_h

    return image, sx, sy, original_w, original_h


def sanitize_name(text):
    allowed = []
    for ch in str(text):
        if ch.isalnum() or ch in ["-", "_"]:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed)[:140]


def draw_single_candidate(row, output_path):
    image_path = project_path(row["image_path"])

    if not image_path.exists():
        return None

    image, sx, sy, _, _ = load_and_resize_image(image_path)
    draw = ImageDraw.Draw(image)

    candidate_type = row.get("candidate_type", "")
    dataset = row.get("dataset", "")
    detector = row.get("detector", "")
    image_id = row.get("image_id", "")
    risk = row.get("risk_level", "")

    gt_box = get_box(row, "gt")
    pred_box = get_box(row, "pred")

    if gt_box is not None:
        gt_box = scale_box(gt_box, sx, sy)

    if pred_box is not None:
        pred_box = scale_box(pred_box, sx, sy)

    gt_class = row.get("gt_class_name", "")
    pred_class = row.get("pred_class_name", "")
    score = row.get("score", "")
    iou = row.get("iou", "")

    if candidate_type == "false_positive":
        if pred_box is not None:
            draw_box(
                draw,
                pred_box,
                COLORS["false_positive"],
                f"FP {pred_class} {score}",
            )

    elif candidate_type == "localization_error":
        if gt_box is not None:
            draw_box(
                draw,
                gt_box,
                COLORS["ground_truth"],
                f"GT {gt_class}",
                width=3,
            )
        if pred_box is not None:
            draw_box(
                draw,
                pred_box,
                COLORS["localization_error"],
                f"LOC {pred_class} IoU={iou}",
                width=4,
            )

    elif candidate_type == "class_confusion":
        if gt_box is not None:
            draw_box(
                draw,
                gt_box,
                COLORS["ground_truth"],
                f"GT {gt_class}",
                width=3,
            )
        if pred_box is not None:
            draw_box(
                draw,
                pred_box,
                COLORS["class_confusion"],
                f"CONF {pred_class} IoU={iou}",
                width=4,
            )

    elif candidate_type == "duplicate_detection":
        if gt_box is not None:
            draw_box(
                draw,
                gt_box,
                COLORS["ground_truth"],
                f"GT {gt_class}",
                width=3,
            )
        if pred_box is not None:
            draw_box(
                draw,
                pred_box,
                COLORS["duplicate_detection"],
                f"DUP {pred_class} {score}",
                width=4,
            )

    else:
        return None

    title = f"{TYPE_DISPLAY.get(candidate_type, candidate_type)} | {dataset} | {detector}"
    subtitle = f"image={image_id} | risk={risk} | gt={gt_class} | pred={pred_class} | score={score} | IoU={iou}"

    image = add_header(image, title, subtitle)
    image.save(output_path)

    return output_path


def draw_safety_composite(group, output_path):
    if not group:
        return None

    first = group[0]
    image_path = project_path(first["image_path"])

    if not image_path.exists():
        return None

    image, sx, sy, _, _ = load_and_resize_image(image_path)
    draw = ImageDraw.Draw(image)

    dataset = first.get("dataset", "")
    detector = first.get("detector", "")
    image_id = first.get("image_id", "")

    missed_ped = 0
    missed_cyc = 0
    small_missed = 0

    for row in group:
        gt_box = get_box(row, "gt")
        if gt_box is None:
            continue

        gt_box = scale_box(gt_box, sx, sy)

        cls = row.get("gt_class_name", row.get("analysis_class_name", ""))
        size_bin = row.get("object_size_bin", "")

        if cls == "Pedestrian":
            missed_ped += 1
        elif cls == "Cyclist":
            missed_cyc += 1

        if size_bin == "small":
            small_missed += 1

        draw_box(
            draw,
            gt_box,
            COLORS["false_negative"],
            f"MISS {cls} {size_bin}",
            width=4,
        )

    title = f"Safety-Critical False Negatives | {dataset} | {detector}"
    subtitle = (
        f"image={image_id} | missed_ped={missed_ped} | "
        f"missed_cyclist={missed_cyc} | small_missed={small_missed}"
    )

    image = add_header(image, title, subtitle)
    image.save(output_path)

    return output_path


def load_top_safety_keys():
    keys = []

    if not TOP_SAFETY_IMAGES_CSV.exists():
        return keys

    with TOP_SAFETY_IMAGES_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (
                row.get("dataset", ""),
                row.get("detector", ""),
                row.get("image_id", ""),
            )
            keys.append(key)

            if len(keys) >= MAX_TOP_SAFETY_IMAGES:
                break

    return keys


def load_candidate_rows():
    """
    Balanced candidate selection for gallery creation.

    The earlier candidate CSV can be dominated by one failure type.
    This function instead scans the full detection error index and keeps
    balanced examples per dataset, detector, and failure type.
    """
    target_types = {
        "false_positive",
        "localization_error",
        "class_confusion",
        "duplicate_detection",
    }

    per_bucket_limit = 4
    buckets = defaultdict(list)
    serial = 0

    def candidate_priority(row):
        failure_type = row.get("failure_type", "")
        risk = row.get("risk_level", "unknown")
        score = safe_float(row.get("score")) or 0.0
        iou = safe_float(row.get("iou")) or 0.0
        size_bin = row.get("object_size_bin", "")

        priority = 0.0

        if risk == "critical":
            priority += 4.0
        elif risk == "high":
            priority += 3.0
        elif risk == "medium":
            priority += 2.0
        elif risk == "low":
            priority += 1.0
        else:
            priority += 0.5

        if row.get("analysis_class_name") in SAFETY_CLASSES:
            priority += 1.0
        if row.get("gt_class_name") in SAFETY_CLASSES:
            priority += 1.0
        if row.get("pred_class_name") in SAFETY_CLASSES:
            priority += 1.0

        if size_bin == "small":
            priority += 0.5

        if failure_type == "false_positive":
            priority += score
        elif failure_type == "localization_error":
            priority += score
            priority += max(0.0, 0.50 - iou)
        elif failure_type == "class_confusion":
            priority += score
            priority += iou
        elif failure_type == "duplicate_detection":
            priority += score
            priority += iou

        return priority

    def make_candidate_from_index_row(row):
        failure_type = row.get("failure_type", "")
        priority = candidate_priority(row)

        return {
            "candidate_type": failure_type,
            "dataset": row.get("dataset", ""),
            "detector": row.get("detector", ""),
            "image_id": row.get("image_id", ""),
            "image_path": row.get("image_path", ""),
            "risk_level": row.get("risk_level", ""),

            "analysis_class_name": row.get("analysis_class_name", ""),
            "gt_class_name": row.get("gt_class_name", ""),
            "pred_class_name": row.get("pred_class_name", ""),
            "object_size_bin": row.get("object_size_bin", ""),

            "score": row.get("score", ""),
            "iou": row.get("iou", ""),
            "best_same_class_iou": row.get("best_same_class_iou", ""),
            "best_wrong_class_iou": row.get("best_wrong_class_iou", ""),
            "priority": round(priority, 6),

            "gt_x1": row.get("gt_x1", ""),
            "gt_y1": row.get("gt_y1", ""),
            "gt_x2": row.get("gt_x2", ""),
            "gt_y2": row.get("gt_y2", ""),

            "pred_x1": row.get("pred_x1", ""),
            "pred_y1": row.get("pred_y1", ""),
            "pred_x2": row.get("pred_x2", ""),
            "pred_y2": row.get("pred_y2", ""),

            "image_width": row.get("image_width", ""),
            "image_height": row.get("image_height", ""),
            "total_gt_in_image": row.get("total_gt_in_image", ""),
            "total_predictions_in_image": row.get("total_predictions_in_image", ""),
        }

    print("Scanning detection error index for balanced failure-case candidates...")

    with DETECTION_INDEX_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            failure_type = row.get("failure_type", "")

            if failure_type not in target_types:
                continue

            dataset = row.get("dataset", "")
            detector = row.get("detector", "")

            if not dataset or not detector:
                continue

            # For false positives, prefer visible/high-confidence examples.
            if failure_type == "false_positive":
                score = safe_float(row.get("score")) or 0.0
                if score < 0.25:
                    continue

            # Gallery drawing needs at least one relevant box.
            if failure_type == "false_positive":
                if get_box(row, "pred") is None:
                    continue
            else:
                if get_box(row, "gt") is None or get_box(row, "pred") is None:
                    continue

            candidate = make_candidate_from_index_row(row)
            priority = safe_float(candidate["priority"]) or 0.0

            bucket_key = (dataset, failure_type, detector)
            serial += 1

            heapq.heappush(buckets[bucket_key], (priority, serial, candidate))

            if len(buckets[bucket_key]) > per_bucket_limit:
                heapq.heappop(buckets[bucket_key])

    selected = []

    for bucket_key, heap_items in buckets.items():
        selected.extend(
            item[2]
            for item in sorted(heap_items, key=lambda x: x[0], reverse=True)
        )

    selected = sorted(
        selected,
        key=lambda r: (
            r.get("dataset", ""),
            r.get("candidate_type", ""),
            r.get("detector", ""),
            -(safe_float(r.get("priority")) or 0.0),
        ),
    )

    return selected

def false_negative_priority(row):
    priority = 0.0

    risk = row.get("risk_level", "")
    cls = row.get("analysis_class_name", "")
    size_bin = row.get("object_size_bin", "")
    dataset = row.get("dataset", "")

    if risk == "critical":
        priority += 4.0
    elif risk == "high":
        priority += 3.0
    elif risk == "medium":
        priority += 2.0
    else:
        priority += 1.0

    if cls in SAFETY_CLASSES:
        priority += 2.0

    if cls == "Cyclist":
        priority += 0.5

    if size_bin == "small":
        priority += 0.75

    if dataset == "waymo":
        priority += 0.5

    best_same = safe_float(row.get("best_same_class_iou")) or 0.0
    best_wrong = safe_float(row.get("best_wrong_class_iou")) or 0.0

    priority += best_same * 0.25
    priority += best_wrong * 0.25

    return priority


def collect_false_negative_rows(top_keys):
    top_key_set = set(top_keys)
    composite_groups = defaultdict(list)

    heap = []
    serial = 0

    with DETECTION_INDEX_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("failure_type") != "false_negative":
                continue

            cls = row.get("analysis_class_name", "")
            if cls not in SAFETY_CLASSES:
                continue

            key = (
                row.get("dataset", ""),
                row.get("detector", ""),
                row.get("image_id", ""),
            )

            if key in top_key_set:
                composite_groups[key].append(row)

            priority = false_negative_priority(row)
            serial += 1

            heapq.heappush(heap, (priority, serial, row))

            if len(heap) > MAX_FALSE_NEGATIVE_HEAP:
                heapq.heappop(heap)

    top_false_negatives = [
        item[2]
        for item in sorted(heap, key=lambda x: x[0], reverse=True)
    ]

    return composite_groups, top_false_negatives


def draw_false_negative_single(row, output_path):
    image_path = project_path(row["image_path"])

    if not image_path.exists():
        return None

    image, sx, sy, _, _ = load_and_resize_image(image_path)
    draw = ImageDraw.Draw(image)

    gt_box = get_box(row, "gt")
    if gt_box is None:
        return None

    gt_box = scale_box(gt_box, sx, sy)

    dataset = row.get("dataset", "")
    detector = row.get("detector", "")
    image_id = row.get("image_id", "")
    cls = row.get("analysis_class_name", "")
    size_bin = row.get("object_size_bin", "")
    best_same = row.get("best_same_class_iou", "")
    best_wrong = row.get("best_wrong_class_iou", "")

    draw_box(
        draw,
        gt_box,
        COLORS["false_negative"],
        f"MISS {cls} {size_bin}",
        width=4,
    )

    title = f"False Negative | {dataset} | {detector}"
    subtitle = f"image={image_id} | class={cls} | size={size_bin} | best_same_iou={best_same} | best_wrong_iou={best_wrong}"

    image = add_header(image, title, subtitle)
    image.save(output_path)

    return output_path


def create_panel(image_paths, output_path, title):
    if not image_paths:
        return None

    thumbs = []

    for path in image_paths[:MAX_PANEL_IMAGES]:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((420, 300), Image.LANCZOS)
            thumbs.append((path, img.copy()))
        except Exception:
            continue

    if not thumbs:
        return None

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols

    cell_w = 460
    cell_h = 350
    header_h = 70

    panel = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), (245, 245, 245))
    draw = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, panel.size[0], header_h], fill=(20, 20, 20))
    draw.text((15, 12), title, fill=(255, 255, 255), font=FONT)
    draw.text((15, 42), f"Generated examples: {len(thumbs)}", fill=(230, 230, 230), font=SMALL_FONT)

    for idx, (path, img) in enumerate(thumbs):
        r = idx // cols
        c = idx % cols

        x = c * cell_w + 20
        y = header_h + r * cell_h + 15

        panel.paste(img, (x, y))

        label = path.stem[:52]
        draw.text((x, y + img.size[1] + 8), label, fill=(0, 0, 0), font=SMALL_FONT)

    panel.save(output_path)
    return output_path


def main():
    print("=" * 100)
    print("STEP 7/10 - Create failure-case image gallery")
    print("=" * 100)

    missing = []

    for path in [DETECTION_INDEX_CSV, TOP_SAFETY_IMAGES_CSV, CANDIDATES_CSV]:
        if not path.exists():
            missing.append(path)

    if missing:
        for path in missing:
            print("ERROR: Missing input:", path)
        print("STEP 7/10 FAILED ❌")
        raise SystemExit(1)

    print("Loading top safety-critical image keys...")
    top_safety_keys = load_top_safety_keys()
    print("Top safety image keys:", len(top_safety_keys))

    print("Collecting false-negative rows...")
    composite_groups, top_false_negatives = collect_false_negative_rows(top_safety_keys)
    print("Safety composite groups:", len(composite_groups))
    print("Top false-negative candidates:", len(top_false_negatives))

    print("Loading failure-case candidates...")
    candidate_rows = load_candidate_rows()
    print("Candidate rows selected:", len(candidate_rows))

    generated = []
    safety_generated = []
    waymo_generated = []
    kitti_generated = []

    # Safety composite images.
    for idx, key in enumerate(top_safety_keys, start=1):
        group = composite_groups.get(key, [])
        if not group:
            continue

        dataset, detector, image_id = key

        out = IMAGE_DIR / f"{idx:03d}_safety_composite_{dataset}_{detector}_{sanitize_name(image_id)}.png"
        result = draw_safety_composite(group, out)

        if result:
            rel_path = str(result.relative_to(PROJECT))
            generated.append({
                "type": "safety_composite",
                "dataset": dataset,
                "detector": detector,
                "image_id": image_id,
                "path": rel_path,
                "num_rows_drawn": len(group),
            })
            safety_generated.append(result)

            if dataset == "waymo":
                waymo_generated.append(result)
            elif dataset == "kitti":
                kitti_generated.append(result)

            print("Created:", result)

    # Individual high-priority false negatives.
    fn_written = 0
    for row in top_false_negatives:
        if fn_written >= MAX_SINGLE_EXAMPLES_PER_TYPE:
            break

        dataset = row.get("dataset", "")
        detector = row.get("detector", "")
        image_id = row.get("image_id", "")
        cls = row.get("analysis_class_name", "")
        size_bin = row.get("object_size_bin", "")

        out = IMAGE_DIR / (
            f"{len(generated)+1:03d}_false_negative_{dataset}_{detector}_"
            f"{sanitize_name(cls)}_{sanitize_name(size_bin)}_{sanitize_name(image_id)}.png"
        )

        result = draw_false_negative_single(row, out)

        if result:
            generated.append({
                "type": "false_negative",
                "dataset": dataset,
                "detector": detector,
                "image_id": image_id,
                "class_name": cls,
                "object_size_bin": size_bin,
                "path": str(result.relative_to(PROJECT)),
            })
            fn_written += 1

            if dataset == "waymo":
                waymo_generated.append(result)
            elif dataset == "kitti":
                kitti_generated.append(result)

            print("Created:", result)

    # Other candidate types: FP, localization, confusion, duplicate.
    for row in candidate_rows:
        candidate_type = row.get("candidate_type", "")

        dataset = row.get("dataset", "")
        detector = row.get("detector", "")
        image_id = row.get("image_id", "")
        risk = row.get("risk_level", "")

        out = IMAGE_DIR / (
            f"{len(generated)+1:03d}_{sanitize_name(candidate_type)}_{dataset}_{detector}_"
            f"{sanitize_name(risk)}_{sanitize_name(image_id)}.png"
        )

        result = draw_single_candidate(row, out)

        if result:
            generated.append({
                "type": candidate_type,
                "dataset": dataset,
                "detector": detector,
                "image_id": image_id,
                "risk_level": risk,
                "path": str(result.relative_to(PROJECT)),
            })

            if dataset == "waymo":
                waymo_generated.append(result)
            elif dataset == "kitti":
                kitti_generated.append(result)

            print("Created:", result)

    print()
    print("Creating figure panels...")

    created_panels = {}

    panel = create_panel(
        waymo_generated,
        PANEL_WAYMO,
        "Milestone 7 Failure-Case Panel - Waymo External Validation",
    )
    if panel:
        created_panels["waymo_panel"] = str(panel.relative_to(PROJECT))
        print("Created:", panel)

    panel = create_panel(
        kitti_generated,
        PANEL_KITTI,
        "Milestone 7 Failure-Case Panel - KITTI Validation",
    )
    if panel:
        created_panels["kitti_panel"] = str(panel.relative_to(PROJECT))
        print("Created:", panel)

    panel = create_panel(
        safety_generated,
        PANEL_SAFETY,
        "Milestone 7 Safety-Critical Vulnerable-Road-User Misses",
    )
    if panel:
        created_panels["safety_vru_panel"] = str(panel.relative_to(PROJECT))
        print("Created:", panel)

    type_counts = defaultdict(int)
    dataset_counts = defaultdict(int)

    for item in generated:
        type_counts[item["type"]] += 1
        dataset_counts[item["dataset"]] += 1

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASSED",
        "inputs": {
            "detection_error_index_csv": str(DETECTION_INDEX_CSV.relative_to(PROJECT)),
            "top_safety_images_csv": str(TOP_SAFETY_IMAGES_CSV.relative_to(PROJECT)),
            "failure_case_candidates_csv": str(CANDIDATES_CSV.relative_to(PROJECT)),
        },
        "annotation_colors": {
            "ground_truth": "green",
            "false_negative": "red",
            "false_positive": "orange",
            "localization_error": "yellow",
            "class_confusion": "purple",
            "duplicate_detection": "cyan",
        },
        "generated_count": len(generated),
        "type_counts": dict(type_counts),
        "dataset_counts": dict(dataset_counts),
        "panels": created_panels,
        "examples": generated,
    }

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = []
    md.append("# Milestone 7 Failure-Case Gallery")
    md.append("")
    md.append(f"Created at: `{manifest['created_at']}`")
    md.append("")
    md.append("Status: **PASSED**")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append(
        "This gallery visualizes safety-critical false negatives, false positives, "
        "localization errors, class confusion, and duplicate detections."
    )
    md.append("")
    md.append("## Annotation Colors")
    md.append("")
    for key, value in manifest["annotation_colors"].items():
        md.append(f"- `{key}`: {value}")
    md.append("")
    md.append("## Generated Panels")
    md.append("")
    for key, value in created_panels.items():
        md.append(f"- `{key}`: `{value}`")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append(f"- Total generated images: `{len(generated)}`")
    for key, value in sorted(type_counts.items()):
        md.append(f"- `{key}`: `{value}`")
    md.append("")
    md.append("## Example Files")
    md.append("")
    for item in generated[:50]:
        md.append(f"- `{item['type']}` / `{item['dataset']}` / `{item['detector']}`: `{item['path']}`")
    md.append("")

    MANIFEST_MD.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 100)
    print("Failure-case gallery created")
    print("=" * 100)
    print("Generated images:", len(generated))
    print("Type counts:", dict(type_counts))
    print("Dataset counts:", dict(dataset_counts))
    print("Created:", MANIFEST_JSON)
    print("Created:", MANIFEST_MD)
    for key, value in created_panels.items():
        print("Panel:", key, "->", value)

    print()
    print("STEP 7/10 COMPLETE ✅")
    print("Failure-case gallery and panels are ready.")
    print("=" * 100)


if __name__ == "__main__":
    main()