from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


# ============================================================
# PATHS
# ============================================================

SCRIPT_FILE = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT_FILE.parents[2]

MILESTONE_SCRIPT_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "milestone_3"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "milestone_3"
)

REPORT_DIR = (
    PROCESSED_ROOT
    / "reports"
)

MANIFEST_DIR = (
    PROCESSED_ROOT
    / "manifests"
)

REPRODUCIBILITY_REPORT_FILE = (
    REPORT_DIR
    / "reproducibility_report.json"
)

REPRODUCIBILITY_MANIFEST_FILE = (
    MANIFEST_DIR
    / "reproducibility_run_manifest.csv"
)

FINAL_AUDIT_REPORT_FILE = (
    REPORT_DIR
    / "final_dataset_audit.json"
)


# ============================================================
# SAFETY TOKENS
# ============================================================

FULL_RUN_CONFIRMATION = (
    "RUN_MILESTONE_3_FULL"
)

CLEAN_RUN_CONFIRMATION = (
    "REGENERATE_MILESTONE_3"
)


# ============================================================
# PIPELINE STAGES
# ============================================================

@dataclass(frozen=True)
class PipelineStage:
    number: int
    name: str
    script_name: str

    # True means that the stage does not regenerate the main
    # processed image set or canonical annotation datasets.
    validation_only: bool

    # Arguments used only during an explicitly confirmed clean run.
    clean_arguments: tuple[str, ...] = ()

    description: str = ""

    @property
    def script_path(self) -> Path:
        return (
            MILESTONE_SCRIPT_DIR
            / self.script_name
        )


PIPELINE_STAGES = [
    PipelineStage(
        number=1,
        name="Validate Milestone 2 inputs",
        script_name=(
            "01_validate_milestone_2_inputs.py"
        ),
        validation_only=True,
        description=(
            "Validate frozen KITTI and Waymo "
            "source inputs."
        ),
    ),

    PipelineStage(
        number=2,
        name="Build source manifest",
        script_name=(
            "02_build_source_manifest.py"
        ),
        validation_only=False,
        description=(
            "Create the authoritative source "
            "image manifest."
        ),
    ),

    PipelineStage(
        number=3,
        name="Preprocessing dry run",
        script_name=(
            "03_preprocessing_dry_run.py"
        ),
        validation_only=True,
        description=(
            "Validate letterboxing on eight "
            "representative samples."
        ),
    ),

    PipelineStage(
        number=4,
        name="Preprocess all images",
        script_name=(
            "04_preprocess_all_images.py"
        ),
        validation_only=False,
        clean_arguments=(
            "--clean",
        ),
        description=(
            "Generate all 8,477 processed "
            "640x640 images."
        ),
    ),

    PipelineStage(
        number=5,
        name="Create canonical COCO annotations",
        script_name=(
            "05_create_coco_annotations.py"
        ),
        validation_only=False,
        description=(
            "Generate canonical COCO files "
            "for all partitions."
        ),
    ),

    PipelineStage(
        number=6,
        name="Create region-policy sidecars",
        script_name=(
            "06_create_region_policy_sidecars.py"
        ),
        validation_only=False,
        description=(
            "Generate evaluation-ignore and "
            "excluded-object sidecars."
        ),
    ),

    PipelineStage(
        number=7,
        name="Convert COCO to YOLO",
        script_name=(
            "07_convert_coco_to_yolo.py"
        ),
        validation_only=False,
        clean_arguments=(
            "--clean",
        ),
        description=(
            "Generate derived YOLO labels "
            "from canonical COCO."
        ),
    ),

    PipelineStage(
        number=8,
        name="Create dataset configurations",
        script_name=(
            "08_create_dataset_configs.py"
        ),
        validation_only=False,
        clean_arguments=(
            "--clean",
        ),
        description=(
            "Generate YOLO and COCO framework "
            "configurations and label adapters."
        ),
    ),

    PipelineStage(
        number=9,
        name="Validate COCO annotations",
        script_name=(
            "09_validate_coco_annotations.py"
        ),
        validation_only=True,
        description=(
            "Independently validate every "
            "canonical COCO annotation."
        ),
    ),

    PipelineStage(
        number=10,
        name="Validate YOLO annotations",
        script_name=(
            "10_validate_yolo_annotations.py"
        ),
        validation_only=True,
        description=(
            "Independently validate every "
            "canonical and framework YOLO label."
        ),
    ),

    PipelineStage(
        number=11,
        name="Validate COCO-YOLO equivalence",
        script_name=(
            "11_validate_coco_yolo_equivalence.py"
        ),
        validation_only=True,
        description=(
            "Prove numerical equivalence for "
            "all 63,905 target boxes."
        ),
    ),

    PipelineStage(
        number=12,
        name="Create visual annotation checks",
        script_name=(
            "12_create_visual_annotation_checks.py"
        ),
        validation_only=True,
        description=(
            "Refresh deterministic COCO-YOLO "
            "visual comparison artifacts."
        ),
    ),

    PipelineStage(
        number=13,
        name="Validate augmentation policy",
        script_name=(
            "13_validate_augmentation_policy.py"
        ),
        validation_only=True,
        description=(
            "Validate the frozen online "
            "augmentation policy."
        ),
    ),

    PipelineStage(
        number=14,
        name="Validate dataset loaders",
        script_name=(
            "14_validate_dataset_loaders.py"
        ),
        validation_only=True,
        description=(
            "Validate model-ready PyTorch "
            "datasets and DataLoaders."
        ),
    ),

    PipelineStage(
        number=15,
        name="Run final dataset audit",
        script_name=(
            "15_final_dataset_audit.py"
        ),
        validation_only=True,
        description=(
            "Perform the final read-only "
            "end-to-end dataset audit."
        ),
    ),
]


VALIDATION_STAGE_NUMBERS = {
    stage.number
    for stage in PIPELINE_STAGES
    if stage.validation_only
}


MANIFEST_COLUMNS = [
    "stage_number",
    "stage_name",
    "script",
    "command",
    "mode",
    "started_at_utc",
    "finished_at_utc",
    "elapsed_seconds",
    "return_code",
    "status",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def atomic_write_json(
    path: Path,
    content: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                content,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def load_json(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        content = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(
        content,
        dict,
    ):
        return None

    return content


def command_text(
    command: list[str],
) -> str:
    return subprocess.list2cmdline(
        command
    )


def run_git_command(
    arguments: list[str],
) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    except OSError:
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def git_snapshot() -> dict[str, Any]:
    commit = run_git_command(
        [
            "rev-parse",
            "HEAD",
        ]
    )

    branch = run_git_command(
        [
            "branch",
            "--show-current",
        ]
    )

    status_output = run_git_command(
        [
            "status",
            "--short",
        ]
    )

    status_lines = (
        status_output.splitlines()
        if status_output
        else []
    )

    return {
        "commit": commit,
        "branch": branch,
        "working_tree_clean": (
            len(status_lines) == 0
        ),
        "status_entry_count": (
            len(status_lines)
        ),
    }


def validate_project_layout() -> None:
    required_paths = [
        (
            PROJECT_ROOT
            / "configs"
            / "datasets"
            / "milestone_3"
            / "preprocessing.yaml"
        ),

        (
            PROJECT_ROOT
            / "configs"
            / "datasets"
            / "milestone_3"
            / "class_mapping.yaml"
        ),

        MILESTONE_SCRIPT_DIR,

        (
            PROJECT_ROOT
            / ".gitignore"
        ),
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        formatted = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "The repository layout is incomplete. "
            "Missing paths:\n"
            f"{formatted}"
        )


def validate_stage_scripts(
    stages: list[PipelineStage],
) -> None:
    missing_scripts = [
        stage.script_path
        for stage in stages
        if not stage.script_path.exists()
    ]

    if missing_scripts:
        formatted = "\n".join(
            str(path)
            for path in missing_scripts
        )

        raise FileNotFoundError(
            "Required Milestone 3 scripts "
            "were not found:\n"
            f"{formatted}"
        )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely validate or regenerate "
            "the Milestone 3 dataset pipeline."
        )
    )

    mode_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode_group.add_argument(
        "--list",
        action="store_true",
        help=(
            "List all pipeline stages without "
            "executing anything."
        ),
    )

    mode_group.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Run only non-destructive validation, "
            "dry-run, visualization, augmentation, "
            "DataLoader, and final-audit stages."
        ),
    )

    mode_group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run all Milestone 3 pipeline stages "
            "in order. Requires a confirmation token."
        ),
    )

    parser.add_argument(
        "--from-stage",
        type=int,
        choices=range(
            1,
            len(
                PIPELINE_STAGES
            ) + 1,
        ),
        help=(
            "For --full mode, begin from this "
            "pipeline stage number."
        ),
    )

    parser.add_argument(
        "--to-stage",
        type=int,
        choices=range(
            1,
            len(
                PIPELINE_STAGES
            ) + 1,
        ),
        help=(
            "For --full mode, stop after this "
            "pipeline stage number."
        ),
    )

    parser.add_argument(
        "--clean-generated",
        action="store_true",
        help=(
            "During --full mode, pass explicit "
            "clean flags to stages that support them. "
            "Requires the stronger regeneration token."
        ),
    )

    parser.add_argument(
        "--confirm",
        default="",
        help=(
            "Safety confirmation token required "
            "for full pipeline execution."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the commands that would run "
            "without executing them."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue to later stages after a "
            "failure. The default is fail-fast."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.list:
        incompatible = any(
            [
                arguments.from_stage
                is not None,

                arguments.to_stage
                is not None,

                arguments.clean_generated,

                arguments.dry_run,

                arguments.continue_on_error,

                bool(
                    arguments.confirm
                ),
            ]
        )

        if incompatible:
            raise ValueError(
                "--list cannot be combined with "
                "execution-related arguments."
            )

        return

    if arguments.validate_only:
        if (
            arguments.from_stage
            is not None
            or arguments.to_stage
            is not None
        ):
            raise ValueError(
                "--from-stage and --to-stage "
                "are supported only with --full."
            )

        if arguments.clean_generated:
            raise ValueError(
                "--clean-generated cannot be used "
                "with --validate-only."
            )

        if arguments.confirm:
            raise ValueError(
                "--validate-only does not require "
                "a confirmation token."
            )

        return

    if arguments.full:
        start_stage = (
            arguments.from_stage
            if arguments.from_stage
            is not None
            else 1
        )

        end_stage = (
            arguments.to_stage
            if arguments.to_stage
            is not None
            else len(
                PIPELINE_STAGES
            )
        )

        if start_stage > end_stage:
            raise ValueError(
                "--from-stage cannot be greater "
                "than --to-stage."
            )

        required_token = (
            CLEAN_RUN_CONFIRMATION
            if arguments.clean_generated
            else FULL_RUN_CONFIRMATION
        )

        if (
            arguments.confirm
            != required_token
        ):
            raise ValueError(
                "Full execution was blocked by "
                "the safety guard.\n\n"
                f"Required token:\n"
                f"  {required_token}"
            )


# ============================================================
# STAGE SELECTION
# ============================================================

def selected_stages(
    arguments: argparse.Namespace,
) -> list[PipelineStage]:
    if arguments.validate_only:
        return [
            stage
            for stage in PIPELINE_STAGES
            if (
                stage.number
                in VALIDATION_STAGE_NUMBERS
            )
        ]

    start_stage = (
        arguments.from_stage
        if arguments.from_stage
        is not None
        else 1
    )

    end_stage = (
        arguments.to_stage
        if arguments.to_stage
        is not None
        else len(
            PIPELINE_STAGES
        )
    )

    return [
        stage
        for stage in PIPELINE_STAGES
        if (
            start_stage
            <= stage.number
            <= end_stage
        )
    ]


def execution_mode(
    arguments: argparse.Namespace,
) -> str:
    if arguments.validate_only:
        return "validate_only"

    if arguments.clean_generated:
        return "full_clean_regeneration"

    return "full_incremental"


def build_stage_command(
    stage: PipelineStage,
    arguments: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(
            stage.script_path
        ),
    ]

    if (
        arguments.full
        and arguments.clean_generated
        and stage.clean_arguments
    ):
        command.extend(
            stage.clean_arguments
        )

    return command


# ============================================================
# PIPELINE LISTING
# ============================================================

def print_pipeline_listing() -> None:
    print("=" * 92)
    print("MILESTONE 3 REPRODUCIBILITY PIPELINE")
    print("=" * 92)

    for stage in PIPELINE_STAGES:
        mode = (
            "validation"
            if stage.validation_only
            else "generation"
        )

        clean_arguments = (
            " ".join(
                stage.clean_arguments
            )
            if stage.clean_arguments
            else "-"
        )

        print(
            f"\nStage {stage.number:02d}: "
            f"{stage.name}"
        )

        print(
            f"  Script: "
            f"{stage.script_name}"
        )

        print(
            f"  Type: "
            f"{mode}"
        )

        print(
            f"  Clean arguments: "
            f"{clean_arguments}"
        )

        print(
            f"  Purpose: "
            f"{stage.description}"
        )

    print(
        "\nValidation-only stages: "
        + ", ".join(
            str(number)
            for number
            in sorted(
                VALIDATION_STAGE_NUMBERS
            )
        )
    )

    print(
        "\nFull-run confirmation token:"
    )

    print(
        f"  {FULL_RUN_CONFIRMATION}"
    )

    print(
        "\nClean-regeneration confirmation token:"
    )

    print(
        f"  {CLEAN_RUN_CONFIRMATION}"
    )


# ============================================================
# STAGE EXECUTION
# ============================================================

def execute_stage(
    stage: PipelineStage,
    command: list[str],
    mode: str,
) -> dict[str, Any]:
    started_at = utc_now()
    started_time = time.perf_counter()

    print("\n" + "=" * 92)

    print(
        f"STAGE {stage.number:02d}: "
        f"{stage.name}"
    )

    print("=" * 92)

    print(
        f"Command: "
        f"{command_text(command)}"
    )

    environment = os.environ.copy()

    environment[
        "PYTHONUTF8"
    ] = "1"

    output_tail: deque[str] = deque(
        maxlen=60
    )

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )

    try:
        assert (
            process.stdout
            is not None
        )

        for line in process.stdout:
            print(
                line,
                end="",
            )

            output_tail.append(
                line.rstrip("\r\n")
            )

        return_code = process.wait()

    except KeyboardInterrupt:
        process.terminate()

        try:
            process.wait(
                timeout=5
            )

        except subprocess.TimeoutExpired:
            process.kill()

        raise

    elapsed_seconds = (
        time.perf_counter()
        - started_time
    )

    finished_at = utc_now()

    status = (
        "PASSED"
        if return_code == 0
        else "FAILED"
    )

    print(
        f"\nStage {stage.number:02d} "
        f"finished with status: "
        f"{status}"
    )

    print(
        f"Elapsed: "
        f"{elapsed_seconds:.2f} seconds"
    )

    return {
        "stage_number": (
            stage.number
        ),

        "stage_name": (
            stage.name
        ),

        "script": (
            stage.script_name
        ),

        "description": (
            stage.description
        ),

        "command": (
            command_text(command)
        ),

        "mode": mode,

        "started_at_utc": (
            started_at
        ),

        "finished_at_utc": (
            finished_at
        ),

        "elapsed_seconds": round(
            elapsed_seconds,
            4,
        ),

        "return_code": (
            return_code
        ),

        "status": status,

        "output_tail": list(
            output_tail
        ),
    }


def planned_stage_result(
    stage: PipelineStage,
    command: list[str],
    mode: str,
) -> dict[str, Any]:
    return {
        "stage_number": (
            stage.number
        ),

        "stage_name": (
            stage.name
        ),

        "script": (
            stage.script_name
        ),

        "description": (
            stage.description
        ),

        "command": (
            command_text(command)
        ),

        "mode": mode,

        "started_at_utc": None,

        "finished_at_utc": None,

        "elapsed_seconds": 0.0,

        "return_code": None,

        "status": "PLANNED",

        "output_tail": [],
    }


# ============================================================
# REPORTING
# ============================================================

def create_base_report(
    arguments: argparse.Namespace,
    stages: list[PipelineStage],
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,

        "milestone": 3,

        "step": 17,

        "purpose": (
            "Provide a safe, reproducible, "
            "ordered entry point for validating "
            "or regenerating the complete "
            "Milestone 3 dataset pipeline."
        ),

        "mode": mode,

        "dry_run": bool(
            arguments.dry_run
        ),

        "continue_on_error": bool(
            arguments.continue_on_error
        ),

        "clean_generated": bool(
            arguments.clean_generated
        ),

        "project_root": str(
            PROJECT_ROOT
        ),

        "python_executable": (
            sys.executable
        ),

        "environment": {
            "python_version": (
                platform.python_version()
            ),

            "platform": (
                platform.platform()
            ),
        },

        "git_before_run": (
            git_snapshot()
        ),

        "started_at_utc": (
            utc_now()
        ),

        "finished_at_utc": None,

        "selected_stage_numbers": [
            stage.number
            for stage in stages
        ],

        "selected_stage_names": [
            stage.name
            for stage in stages
        ],

        "stages": [],

        "successful_stage_count": 0,

        "failed_stage_count": 0,

        "planned_stage_count": 0,

        "final_audit": None,

        "overall_status": (
            "RUNNING"
        ),

        "reproducibility_run_passed": (
            False
        ),
    }


def manifest_rows(
    stage_results: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    rows = []

    for result in stage_results:
        rows.append(
            {
                "stage_number": (
                    result[
                        "stage_number"
                    ]
                ),

                "stage_name": (
                    result[
                        "stage_name"
                    ]
                ),

                "script": (
                    result["script"]
                ),

                "command": (
                    result["command"]
                ),

                "mode": (
                    result["mode"]
                ),

                "started_at_utc": (
                    result[
                        "started_at_utc"
                    ]
                    or ""
                ),

                "finished_at_utc": (
                    result[
                        "finished_at_utc"
                    ]
                    or ""
                ),

                "elapsed_seconds": (
                    result[
                        "elapsed_seconds"
                    ]
                ),

                "return_code": (
                    ""
                    if result[
                        "return_code"
                    ]
                    is None
                    else result[
                        "return_code"
                    ]
                ),

                "status": (
                    result["status"]
                ),
            }
        )

    return rows


def final_audit_summary() -> dict[str, Any]:
    report = load_json(
        FINAL_AUDIT_REPORT_FILE
    )

    if report is None:
        return {
            "report_exists": False,
            "passed": False,
        }

    return {
        "report_exists": True,

        "passed": bool(
            report.get(
                "final_dataset_audit_passed",
                False,
            )
        ),

        "issue_count": report.get(
            "issue_count"
        ),

        "checks": report.get(
            "checks",
            {},
        ),

        "path": (
            FINAL_AUDIT_REPORT_FILE
            .relative_to(
                PROJECT_ROOT
            )
            .as_posix()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    arguments = parse_arguments()

    try:
        validate_arguments(
            arguments
        )

        validate_project_layout()

    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )

        raise SystemExit(2)

    if arguments.list:
        print_pipeline_listing()
        return

    stages = selected_stages(
        arguments
    )

    try:
        validate_stage_scripts(
            stages
        )

    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )

        raise SystemExit(2)

    mode = execution_mode(
        arguments
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = create_base_report(
        arguments=arguments,
        stages=stages,
        mode=mode,
    )

    atomic_write_json(
        REPRODUCIBILITY_REPORT_FILE,
        report,
    )

    print("=" * 92)
    print("MILESTONE 3 SAFE REPRODUCIBILITY RUNNER")
    print("=" * 92)

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Mode: "
        f"{mode}"
    )

    print(
        f"Dry run: "
        f"{arguments.dry_run}"
    )

    print(
        f"Selected stages: "
        + ", ".join(
            f"{stage.number:02d}"
            for stage in stages
        )
    )

    if arguments.validate_only:
        print(
            "\nValidation-only mode will not "
            "regenerate the complete processed "
            "image set or canonical annotations."
        )

        print(
            "Small reports, manifests, dry-run "
            "previews, and visual validation "
            "artifacts may be refreshed."
        )

    if arguments.full:
        print(
            "\nWARNING: Full mode can overwrite "
            "generated Milestone 3 outputs."
        )

        if arguments.clean_generated:
            print(
                "Clean regeneration was explicitly "
                "authorized."
            )

    stage_results: list[
        dict[str, Any]
    ] = []

    interrupted = False

    try:
        for stage in stages:
            command = build_stage_command(
                stage,
                arguments,
            )

            if arguments.dry_run:
                result = (
                    planned_stage_result(
                        stage=stage,
                        command=command,
                        mode=mode,
                    )
                )

                stage_results.append(
                    result
                )

                print(
                    f"\n[PLANNED] Stage "
                    f"{stage.number:02d}: "
                    f"{stage.name}"
                )

                print(
                    f"  {result['command']}"
                )

                continue

            result = execute_stage(
                stage=stage,
                command=command,
                mode=mode,
            )

            stage_results.append(
                result
            )

            report["stages"] = (
                stage_results
            )

            report[
                "successful_stage_count"
            ] = sum(
                result["status"]
                == "PASSED"
                for result
                in stage_results
            )

            report[
                "failed_stage_count"
            ] = sum(
                result["status"]
                == "FAILED"
                for result
                in stage_results
            )

            atomic_write_json(
                REPRODUCIBILITY_REPORT_FILE,
                report,
            )

            write_csv(
                REPRODUCIBILITY_MANIFEST_FILE,
                manifest_rows(
                    stage_results
                ),
                MANIFEST_COLUMNS,
            )

            if (
                result["status"]
                == "FAILED"
                and not arguments
                .continue_on_error
            ):
                print(
                    "\nExecution stopped because "
                    "fail-fast mode is active."
                )

                break

    except KeyboardInterrupt:
        interrupted = True

        print(
            "\nExecution interrupted by user.",
            file=sys.stderr,
        )

    report["stages"] = (
        stage_results
    )

    report[
        "successful_stage_count"
    ] = sum(
        result["status"]
        == "PASSED"
        for result
        in stage_results
    )

    report[
        "failed_stage_count"
    ] = sum(
        result["status"]
        == "FAILED"
        for result
        in stage_results
    )

    report[
        "planned_stage_count"
    ] = sum(
        result["status"]
        == "PLANNED"
        for result
        in stage_results
    )

    if arguments.dry_run:
        report["final_audit"] = None

        overall_status = (
            "DRY_RUN_COMPLETED"
        )

        overall_passed = True

    elif interrupted:
        report["final_audit"] = (
            final_audit_summary()
        )

        overall_status = (
            "INTERRUPTED"
        )

        overall_passed = False

    else:
        audit_summary = (
            final_audit_summary()
        )

        report["final_audit"] = (
            audit_summary
        )

        every_selected_stage_ran = (
            len(stage_results)
            == len(stages)
        )

        all_selected_stages_passed = (
            every_selected_stage_ran
            and all(
                result["status"]
                == "PASSED"
                for result
                in stage_results
            )
        )

        final_audit_required = any(
            stage.number == 15
            for stage in stages
        )

        final_audit_passed = (
            bool(
                audit_summary.get(
                    "passed",
                    False,
                )
            )
            if final_audit_required
            else True
        )

        overall_passed = (
            all_selected_stages_passed
            and final_audit_passed
        )

        overall_status = (
            "PASSED"
            if overall_passed
            else "FAILED"
        )

    report[
        "overall_status"
    ] = overall_status

    report[
        "reproducibility_run_passed"
    ] = bool(
        overall_passed
    )

    report[
        "finished_at_utc"
    ] = utc_now()

    report[
        "git_after_run"
    ] = git_snapshot()

    atomic_write_json(
        REPRODUCIBILITY_REPORT_FILE,
        report,
    )

    write_csv(
        REPRODUCIBILITY_MANIFEST_FILE,
        manifest_rows(
            stage_results
        ),
        MANIFEST_COLUMNS,
    )

    print("\n" + "=" * 92)
    print("REPRODUCIBILITY RUN SUMMARY")
    print("=" * 92)

    print(
        f"\nMode: "
        f"{mode}"
    )

    print(
        f"Selected stages: "
        f"{len(stages)}"
    )

    print(
        f"Passed stages: "
        f"{report['successful_stage_count']}"
    )

    print(
        f"Failed stages: "
        f"{report['failed_stage_count']}"
    )

    print(
        f"Planned stages: "
        f"{report['planned_stage_count']}"
    )

    if report["final_audit"]:
        print(
            f"Final dataset audit: "
            f"{report['final_audit']['passed']}"
        )

    print(
        f"\nFinal status: "
        f"{overall_status}"
    )

    print(
        f"\nReproducibility report:\n"
        f"{REPRODUCIBILITY_REPORT_FILE.resolve()}"
    )

    print(
        f"\nExecution manifest:\n"
        f"{REPRODUCIBILITY_MANIFEST_FILE.resolve()}"
    )

    if not overall_passed:
        raise SystemExit(1)

    if arguments.dry_run:
        print(
            "\nDry run completed successfully. "
            "No pipeline stages were executed."
        )

    else:
        print(
            "\nStep 17 completed successfully. "
            "The Milestone 3 pipeline can be "
            "revalidated through one controlled "
            "entry point."
        )


if __name__ == "__main__":
    main()