import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.milestone_6 import waymo_inference


SMOKE_N = 20


def main():
    print("=" * 79)
    print("Milestone 6 - Phase 4: Smoke test locked models on Waymo")
    print("=" * 79)

    project_root = Path(__file__).resolve().parents[2]
    m3_root = project_root / "data" / "processed" / "milestone_3"
    coco_gt_path = m3_root / "annotations" / "coco" / "waymo_external.json"
    images_dir = m3_root / "images" / "waymo" / "external"

    out_dir = project_root / "outputs" / "milestone_6" / "waymo_external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_gt_path, encoding="utf-8") as f:
        coco = json.load(f)

    image_id_map = {img["file_name"]: img["id"] for img in coco["images"]}
    all_images = sorted(images_dir.glob("*.png"))
    smoke_images = all_images[:SMOKE_N]

    checkpoints = waymo_inference.load_checkpoint_paths(project_root)

    results = {}
    errors = []

    for detector, ckpt in checkpoints.items():
        print(f"\n[{detector}] checkpoint: {ckpt}")
        try:
            if detector in ("yolo", "rtdetr"):
                preds, times = waymo_inference.predict_ultralytics(
                    detector, ckpt, smoke_images, image_id_map, device_arg="auto")
            else:
                preds, times = waymo_inference.predict_torchvision(
                    detector, ckpt, smoke_images, image_id_map, device_arg="auto")

            category_ids = {p["category_id"] for p in preds}
            box_ok = all(
                p["bbox"][2] >= 0 and p["bbox"][3] >= 0 and p["score"] > 0 for p in preds
            )
            mean_ms = sum(times) / len(times) if times else 0.0

            results[detector] = {
                "checkpoint": str(ckpt),
                "num_images": len(smoke_images),
                "num_predictions": len(preds),
                "category_ids_seen": sorted(category_ids),
                "bbox_and_score_valid": bool(box_ok),
                "mean_inference_ms": round(mean_ms, 3),
            }
            print(f"  predictions={len(preds)} category_ids={sorted(category_ids)} "
                  f"valid={box_ok} mean={mean_ms:.2f}ms")
        except Exception as exc:  # noqa: BLE001
            errors.append({"detector": detector, "error": str(exc)})
            results[detector] = {"checkpoint": str(ckpt), "error": str(exc)}
            print(f"  ERROR: {exc}")

    status = "PASSED" if not errors else "FAILED"
    summary = {
        "milestone": 6,
        "phase": 4,
        "purpose": "Smoke test: load all four locked checkpoints and run a tiny Waymo sample",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke_image_count": len(smoke_images),
        "status": status,
        "detectors": results,
    }

    json_path = out_dir / "smoke_test_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Milestone 6 - Waymo Model Smoke Test",
        "",
        f"- Status: **{status}**",
        f"- Images: {len(smoke_images)}",
        "",
        "| Detector | Predictions | Category ids | Bbox/score valid | Mean inference (ms) |",
        "|---|---|---|---|---|",
    ]
    for det, r in results.items():
        if "error" in r:
            md_lines.append(f"| {det} | ERROR | - | - | - |")
        else:
            md_lines.append(
                f"| {det} | {r['num_predictions']} | {r['category_ids_seen']} | "
                f"{r['bbox_and_score_valid']} | {r['mean_inference_ms']} |"
            )
    md_lines.append("")
    (out_dir / "smoke_test_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nSmoke test status: {status}")
    print(f"Saved: {json_path}")

    if status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
