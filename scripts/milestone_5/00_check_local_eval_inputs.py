from pathlib import Path
import json
import sys

PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

REGISTRY = PROJECT / "outputs" / "milestone_4" / "locked_final_checkpoints" / "final_checkpoint_registry.json"

EXPECTED_DATA_PATHS = [
    PROJECT / "data" / "processed" / "milestone_3",
    PROJECT / "data" / "processed" / "milestone_3" / "images",
    PROJECT / "data" / "processed" / "milestone_3" / "labels",
]

EXPECTED_LABEL_PATHS = [
    PROJECT / "data" / "processed" / "milestone_3" / "labels" / "kitti" / "val",
    PROJECT / "data" / "processed" / "milestone_3" / "images" / "kitti" / "val",
]


def main():
    print("=" * 100)
    print("STEP 28A - Check local Milestone 5 evaluation inputs")
    print("=" * 100)

    errors = []

    print("Project:", PROJECT)
    print()

    if not REGISTRY.exists():
        errors.append(f"Missing final checkpoint registry: {REGISTRY}")
    else:
        print("OK registry:", REGISTRY.relative_to(PROJECT))

    print()
    print("-" * 100)
    print("Checking final checkpoint registry")
    print("-" * 100)

    if REGISTRY.exists():
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

        models = registry.get("models", {})
        expected_models = ["yolo", "rtdetr", "retinanet", "faster_rcnn"]

        for model_name in expected_models:
            if model_name not in models:
                errors.append(f"Missing model in registry: {model_name}")
                print("MISSING model:", model_name)
                continue

            model = models[model_name]
            ckpt = PROJECT / model["canonical_checkpoint"]
            report = PROJECT / model["report"]
            manifest = PROJECT / model["manifest"]

            print()
            print("Model:", model_name)
            print("Run ID:", model.get("run_id"))
            print("Status:", model.get("final_status"))
            print("Canonical checkpoint:", model.get("canonical_checkpoint"))

            for label, path in {
                "checkpoint": ckpt,
                "report": report,
                "manifest": manifest,
            }.items():
                if path.exists():
                    print(f"OK {label}: {path.relative_to(PROJECT)}")
                else:
                    errors.append(f"{model_name}: missing {label}: {path}")
                    print(f"MISSING {label}: {path}")

    print()
    print("-" * 100)
    print("Checking Milestone 3 validation data")
    print("-" * 100)

    for path in EXPECTED_DATA_PATHS + EXPECTED_LABEL_PATHS:
        if path.exists():
            if path.is_dir():
                count = sum(1 for _ in path.iterdir())
                print(f"OK dir: {path.relative_to(PROJECT)} | items: {count}")
            else:
                print(f"OK file: {path.relative_to(PROJECT)}")
        else:
            errors.append(f"Missing expected data path: {path}")
            print("MISSING:", path)

    print()
    print("-" * 100)
    print("Checking local Python packages")
    print("-" * 100)

    package_checks = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("yaml", "pyyaml"),
    ]

    for import_name, package_name in package_checks:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "version unavailable")
            print(f"OK {package_name}: {version}")
        except Exception as exc:
            errors.append(f"Missing or broken package: {package_name} ({exc})")
            print(f"MISSING/BROKEN {package_name}: {exc}")

    print()
    print("=" * 100)

    if errors:
        print("STEP 28A FAILED ❌")
        print("Problems found:")
        for error in errors:
            print("-", error)
        print()
        print("Fix these before building the evaluator.")
        sys.exit(1)

    print("STEP 28A COMPLETE ✅")
    print("Local checkpoints, reports, manifests, validation data, and packages are ready.")
    print("Next: build the permanent prediction adapters and evaluator.")
    print("=" * 100)


if __name__ == "__main__":
    main()