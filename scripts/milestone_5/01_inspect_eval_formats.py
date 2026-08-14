from pathlib import Path
import json
import torch

PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

REGISTRY_PATH = PROJECT / "outputs" / "milestone_4" / "locked_final_checkpoints" / "final_checkpoint_registry.json"

IMAGE_VAL_DIR = PROJECT / "data" / "processed" / "milestone_3" / "images" / "kitti" / "val"
LABEL_VAL_DIR = PROJECT / "data" / "processed" / "milestone_3" / "labels" / "kitti" / "val"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_5" / "inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_label_files():
    print("=" * 100)
    print("Inspecting KITTI validation labels")
    print("=" * 100)

    label_files = sorted(LABEL_VAL_DIR.glob("*.txt"))
    image_files = sorted(IMAGE_VAL_DIR.glob("*.png"))

    print("Label dir:", LABEL_VAL_DIR)
    print("Image dir:", IMAGE_VAL_DIR)
    print("Label count:", len(label_files))
    print("Image count:", len(image_files))

    if not label_files:
        raise RuntimeError("No validation label files found.")

    samples = []

    for label_path in label_files[:5]:
        text = label_path.read_text(encoding="utf-8").strip()
        lines = text.splitlines() if text else []

        print()
        print("LABEL:", label_path.name)
        print("Line count:", len(lines))

        for line in lines[:5]:
            print(" ", line)

        image_path = IMAGE_VAL_DIR / f"{label_path.stem}.png"
        print("Matching image exists:", image_path.exists(), image_path.name)

        samples.append({
            "label_file": str(label_path.relative_to(PROJECT)),
            "matching_image": str(image_path.relative_to(PROJECT)),
            "matching_image_exists": image_path.exists(),
            "line_count": len(lines),
            "first_lines": lines[:5],
        })

    return {
        "label_count": len(label_files),
        "image_count": len(image_files),
        "samples": samples,
    }


def inspect_registry_and_checkpoints():
    print()
    print("=" * 100)
    print("Inspecting final checkpoint registry and checkpoint formats")
    print("=" * 100)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    models = registry.get("models", {})

    checkpoint_summary = {}

    for detector, info in models.items():
        print()
        print("-" * 100)
        print("Detector:", detector)
        print("Run ID:", info.get("run_id"))

        ckpt_path = PROJECT / info["canonical_checkpoint"]
        print("Checkpoint:", ckpt_path.relative_to(PROJECT))
        print("Size MB:", round(ckpt_path.stat().st_size / (1024 * 1024), 3))

        summary = {
            "run_id": info.get("run_id"),
            "checkpoint": str(ckpt_path.relative_to(PROJECT)),
            "size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 3),
        }

        if detector in {"faster_rcnn", "retinanet"}:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
            print("Checkpoint type:", type(ckpt))
            print("Top-level keys:", keys)

            summary["checkpoint_type"] = str(type(ckpt))
            summary["top_level_keys"] = keys

            for possible_key in ["model_state_dict", "model", "state_dict", "optimizer_state_dict", "epoch", "best_loss"]:
                if isinstance(ckpt, dict) and possible_key in ckpt:
                    value = ckpt[possible_key]
                    if isinstance(value, dict):
                        print(f"{possible_key}: dict with {len(value)} keys")
                        summary[possible_key] = f"dict with {len(value)} keys"
                    else:
                        print(f"{possible_key}:", value)
                        summary[possible_key] = str(value)

        else:
            print("Ultralytics checkpoint. Will be loaded with ultralytics.YOLO during evaluator smoke test.")
            summary["checkpoint_type"] = "ultralytics_pt"

        checkpoint_summary[detector] = summary

    return checkpoint_summary


def main():
    print("=" * 100)
    print("STEP 28B - Inspect evaluation formats")
    print("=" * 100)

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")

    if not IMAGE_VAL_DIR.exists():
        raise FileNotFoundError(f"Missing image val dir: {IMAGE_VAL_DIR}")

    if not LABEL_VAL_DIR.exists():
        raise FileNotFoundError(f"Missing label val dir: {LABEL_VAL_DIR}")

    label_summary = inspect_label_files()
    checkpoint_summary = inspect_registry_and_checkpoints()

    output = {
        "project": str(PROJECT),
        "image_val_dir": str(IMAGE_VAL_DIR.relative_to(PROJECT)),
        "label_val_dir": str(LABEL_VAL_DIR.relative_to(PROJECT)),
        "label_summary": label_summary,
        "checkpoint_summary": checkpoint_summary,
    }

    out_path = OUTPUT_DIR / "eval_format_inspection.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print("=" * 100)
    print("STEP 28B COMPLETE ✅")
    print("Inspection written to:", out_path)
    print("Next: build evaluator based on these confirmed formats.")
    print("=" * 100)


if __name__ == "__main__":
    main()