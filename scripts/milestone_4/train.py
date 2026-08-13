import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description="Kaggle Training Runner (Resume-Safe)")
    parser.add_argument("--detector", required=True, choices=["yolo", "faster_rcnn", "retinanet", "rtdetr"])
    parser.add_argument("--run-type", required=True, choices=["tiny_overfit", "pilot", "final"])
    parser.add_argument("--slot", required=True, choices=["slot_a", "slot_b"])
    parser.add_argument("--resume", default=None, choices=["latest", "best"])
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--target-epochs", type=int, default=None)
    parser.add_argument("--max-runtime-hours", type=float, default=10.5)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--package-on-exit", action="store_true", default=False)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs" / "models" / "milestone_4"
    config_path = config_dir / f"{args.detector}.yaml"

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    from scripts.milestone_4.training_core import run_training

    report = run_training(
        config_path=config_path,
        detector=args.detector,
        run_type=args.run_type,
        slot=args.slot,
        resume=args.resume if args.resume else False,
        max_runtime_hours=args.max_runtime_hours,
        save_every_epochs=args.save_every_epochs,
        package_on_exit=args.package_on_exit,
    )

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
