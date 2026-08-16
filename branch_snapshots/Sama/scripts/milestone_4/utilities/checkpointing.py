import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone


def get_checkpoint_dir(output_root, detector, run_type):
    return Path(output_root) / "checkpoints" / detector / run_type


def get_best_path(ckpt_dir):
    return ckpt_dir / "best.pt"


def get_last_path(ckpt_dir):
    return ckpt_dir / "last.pt"


def get_resume_state_path(ckpt_dir):
    return ckpt_dir / "resume_state.json"


def save_resume_state(ckpt_dir, epoch, best_metric, detector, run_type, session_id, exit_reason):
    state = {
        "detector": detector,
        "run_type": run_type,
        "session_id": session_id,
        "last_epoch": epoch,
        "best_metric": best_metric,
        "best_metric_name": "mAP_50_95",
        "exit_reason": exit_reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = get_resume_state_path(ckpt_dir)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    return state


def load_resume_state(ckpt_dir):
    path = get_resume_state_path(ckpt_dir)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def cleanup_old_checkpoints(ckpt_dir, keep_last_n=3):
    epoch_files = sorted(
        ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for f in epoch_files[keep_last_n:]:
        f.unlink()


def save_training_report(ckpt_dir, report):
    path = ckpt_dir / "training_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path
