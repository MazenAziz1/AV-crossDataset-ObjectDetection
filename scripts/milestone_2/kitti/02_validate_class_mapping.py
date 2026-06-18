from collections import Counter
from pathlib import Path
import json

import yaml


LABEL_DIR = Path(
    "data/kitti/raw/training/label_2"
)

MAPPING_FILE = Path(
    "data/kitti/selection/class_mapping.yaml"
)

OUTPUT_FILE = Path(
    "data/kitti/statistics/class_mapping_validation.json"
)


def main() -> None:
    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found:\n"
            f"{LABEL_DIR.resolve()}"
        )

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Mapping file not found:\n"
            f"{MAPPING_FILE.resolve()}"
        )

    with MAPPING_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        mapping_config = yaml.safe_load(file)

    final_classes = mapping_config.get(
        "final_classes",
        {}
    )

    kitti_mapping = mapping_config.get(
        "kitti_mapping",
        {}
    )

    if final_classes != {
        0: "Vehicle",
        1: "Pedestrian",
        2: "Cyclist",
    }:
        raise ValueError(
            "final_classes must be exactly:\n"
            "0: Vehicle\n"
            "1: Pedestrian\n"
            "2: Cyclist"
        )

    original_counts: Counter = Counter()
    mapped_counts: Counter = Counter()

    mapped_source_classes: set[str] = set()
    ignored_source_classes: set[str] = set()
    unmapped_source_classes: set[str] = set()

    label_files = sorted(
        LABEL_DIR.glob("*.txt")
    )

    for label_file in label_files:
        lines = label_file.read_text(
            encoding="utf-8",
        ).splitlines()

        for line in lines:
            if not line.strip():
                continue

            original_class = line.split()[0]
            original_counts[original_class] += 1

            mapping_entry = kitti_mapping.get(
                original_class
            )

            if mapping_entry is None:
                unmapped_source_classes.add(
                    original_class
                )
                continue

            action = mapping_entry.get("action")

            if action == "map":
                mapped_name = mapping_entry.get(
                    "mapped_class_name"
                )

                mapped_id = mapping_entry.get(
                    "mapped_class_id"
                )

                if mapped_name not in {
                    "Vehicle",
                    "Pedestrian",
                    "Cyclist",
                }:
                    raise ValueError(
                        f"Invalid mapped class name for "
                        f"{original_class}: {mapped_name}"
                    )

                if mapped_id not in {0, 1, 2}:
                    raise ValueError(
                        f"Invalid mapped class ID for "
                        f"{original_class}: {mapped_id}"
                    )

                mapped_counts[mapped_name] += 1
                mapped_source_classes.add(
                    original_class
                )

            elif action == "ignore":
                mapped_counts["Ignored"] += 1
                ignored_source_classes.add(
                    original_class
                )

            else:
                raise ValueError(
                    f"Unknown action for {original_class}: "
                    f"{action}"
                )

    passed = len(unmapped_source_classes) == 0

    report = {
        "dataset": "KITTI Object Detection",
        "label_files_scanned": len(label_files),
        "final_classes": {
            str(key): value
            for key, value in final_classes.items()
        },
        "original_class_counts": dict(
            sorted(original_counts.items())
        ),
        "mapped_class_counts": dict(
            sorted(mapped_counts.items())
        ),
        "mapped_source_classes": sorted(
            mapped_source_classes
        ),
        "ignored_source_classes": sorted(
            ignored_source_classes
        ),
        "unmapped_source_classes": sorted(
            unmapped_source_classes
        ),
        "mapping_validation_passed": passed,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 68)
    print("KITTI CLASS MAPPING VALIDATION")
    print("=" * 68)

    print(
        f"Label files scanned: "
        f"{len(label_files)}"
    )

    print("\nOriginal class counts:")

    for class_name, count in sorted(
        original_counts.items()
    ):
        print(f"  {class_name}: {count}")

    print("\nMapped class counts:")

    for class_name, count in sorted(
        mapped_counts.items()
    ):
        print(f"  {class_name}: {count}")

    print(
        "\nUnmapped source classes: "
        + (
            ", ".join(
                sorted(unmapped_source_classes)
            )
            if unmapped_source_classes
            else "None"
        )
    )

    print(
        "\nMapping status: "
        + (
            "PASSED"
            if passed
            else "FAILED"
        )
    )

    print(
        f"\nReport saved to:\n"
        f"{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()