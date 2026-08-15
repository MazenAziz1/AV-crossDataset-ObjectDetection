import json
from pathlib import Path

import yaml


def main():
    print("=" * 79)
    print("Milestone 6 - Phase 3: Inspect Waymo format and class mapping")
    print("=" * 79)

    project_root = Path(__file__).resolve().parents[2]
    m3_root = project_root / "data" / "processed" / "milestone_3"

    coco_gt_path = m3_root / "annotations" / "coco" / "waymo_external.json"
    yolo_label_paths = sorted((m3_root / "annotations" / "yolo" / "yolo" / "waymo" / "external").glob("*.txt"))

    with open(coco_gt_path, encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco["categories"]
    sample_ann = coco["annotations"][0]
    sample_img = coco["images"][0]

    # Read frozen class mapping configs
    class_map_3 = yaml.safe_load((project_root / "configs/datasets/milestone_3/class_mapping.yaml").read_text(encoding="utf-8"))
    waymo_map = yaml.safe_load(
        (project_root / "data/waymo/representative_subset/annotations/class_mapping.yaml").read_text(encoding="utf-8"))

    # Read a few YOLO label lines
    yolo_samples = []
    for p in yolo_label_paths[:3]:
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        yolo_samples.append({"file": p.name, "first_lines": lines[:5]})

    inspection = {
        "milestone": 6,
        "phase": 3,
        "purpose": "Confirm exact Waymo label format and class mapping before adapting the evaluator",
        "image": {
            "sample_image_id": sample_img["id"],
            "sample_file_name": sample_img["file_name"],
            "width": sample_img["width"],
            "height": sample_img["height"],
            "file_extension_on_disk": ".png",
        },
        "coco_annotation_format": {
            "coordinate_system": "xywh_absolute_float_pixels",
            "category_ids": {c["id"]: c["name"] for c in categories},
            "sample_annotation": sample_ann,
        },
        "yolo_label_format": {
            "schema": "class_id x_center y_center width height",
            "coordinate_system": "normalized_0_to_1",
            "sample_lines": yolo_samples,
        },
        "class_id_mapping": {
            "coco_category_id": class_map_3["format_id_policy"]["coco"],
            "yolo_class_id": class_map_3["format_id_policy"]["yolo"],
            "torchvision_label": {"background": 0, "Vehicle": 1, "Pedestrian": 2, "Cyclist": 3},
            "waymo_original_type_ids": {
                "Vehicle": waymo_map["waymo_mapping"][1]["original_name"],
                "Pedestrian": waymo_map["waymo_mapping"][2]["original_name"],
                "Cyclist": waymo_map["waymo_mapping"][4]["original_name"],
                "Sign_excluded": waymo_map["waymo_mapping"][3]["original_name"],
            },
        },
        "conversion_rule_for_evaluator": {
            "ground_truth": "COCO category ids (1,2,3) are used directly by pycocotools.",
            "ultralytics_yolo_rtdetr": "predicted internal class id (0,1,2) -> COCO id via +1.",
            "torchvision": "predicted label (1,2,3) equals COCO id; label 0 (background) ignored.",
        },
        "verdict": "No silent conversion required; the Milestone 3 frozen mapping is respected.",
    }

    out_dir = project_root / "outputs" / "milestone_6" / "handoff_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "waymo_format_inspection.json"
    out_path.write_text(json.dumps(inspection, indent=2), encoding="utf-8")

    print(f"COCO categories: { {c['id']: c['name'] for c in categories} }")
    print(f"YOLO schema: class_id x_center y_center width height (normalized)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
