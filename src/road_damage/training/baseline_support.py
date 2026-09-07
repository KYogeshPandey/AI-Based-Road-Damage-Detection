"""Validation and provenance support for the frozen public YOLOv8s baseline."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from road_damage.dataset.rdd2022_common import PROJECT_ROOT, sha256_file, write_json_atomic


CLASS_NAMES = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}
EXPECTED_MODEL_SHA256 = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
EXPECTED_AMP_ARTIFACT_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
EXPECTED_ENVIRONMENT = {
    "python": "3.12.10",
    "ultralytics": "8.4.130",
    "torch": "2.13.0+cu126",
    "torchvision": "0.28.0+cu126",
    "cuda": "12.6",
}
FROZEN_TRAINING_PARAMETERS: dict[str, Any] = {
    "task": "detect",
    "imgsz": 640,
    "epochs": 100,
    "patience": 20,
    "batch": 4,
    "nbs": 64,
    "device": 0,
    "workers": 2,
    "cache": False,
    "amp": True,
    "compile": False,
    "pretrained": True,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": True,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "seed": 42,
    "deterministic": True,
    "fraction": 1.0,
    "rect": False,
    "multi_scale": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.40,
    "hsv_v": 0.30,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.25,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.25,
    "close_mosaic": 15,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "bgr": 0.0,
    "val": True,
    "iou": 0.70,
    "max_det": 300,
    "plots": True,
    "save": True,
    "save_period": 5,
}
EXPECTED_METADATA: dict[str, Any] = {
    "schema_version": "baseline_public_v1.yolov8s.v1",
    "experiment_id": "baseline_public_v1_yolov8s",
    "model": "models/pretrained/yolov8s.pt",
    "model_sha256": EXPECTED_MODEL_SHA256.upper(),
    "source_dataset": "data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1",
    "data": "configs/training/baseline_public_v1_train_val.yaml",
    "project": "outputs/training/baseline_public_v1",
    "run_name": "20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42",
    "source_state_manifest": "reproducibility/baseline_public_v1_source_state_manifest.json",
    "expected_environment": EXPECTED_ENVIRONMENT,
    "expected_gpu": "NVIDIA GeForce RTX 2050",
    "amp_check_artifact": "weights/yolo26n.pt",
    "amp_check_artifact_sha256": EXPECTED_AMP_ARTIFACT_SHA256.upper(),
}


class BaselineTrainingError(RuntimeError):
    """Raised when a baseline safety or reproducibility gate fails."""


@dataclass(frozen=True)
class BaselinePaths:
    """Resolved paths used by one frozen baseline experiment."""

    config: Path
    model: Path
    source_dataset: Path
    data_yaml: Path
    output_base: Path
    run_dir: Path
    amp_check_artifact: Path
    source_state_manifest: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineTrainingError(f"Could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineTrainingError(f"{label.capitalize()} must be a JSON object: {path}")
    return value


def load_baseline_config(path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load the JSON-compatible YAML and enforce the exact approved baseline."""
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    config = _read_json(resolved, "baseline config")
    expected_keys = set(EXPECTED_METADATA) | set(FROZEN_TRAINING_PARAMETERS)
    if set(config) != expected_keys:
        missing = sorted(expected_keys - set(config))
        extra = sorted(set(config) - expected_keys)
        raise BaselineTrainingError(f"Baseline config keys changed: missing={missing}, extra={extra}")
    for key, expected in EXPECTED_METADATA.items():
        if config.get(key) != expected:
            raise BaselineTrainingError(
                f"Frozen baseline metadata {key!r} must be {expected!r}, got {config.get(key)!r}."
            )
    for key, expected in FROZEN_TRAINING_PARAMETERS.items():
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            raise BaselineTrainingError(
                f"Frozen baseline parameter {key!r} must be {expected!r}, got {config.get(key)!r}."
            )
    config["_config_path"] = str(resolved)
    return config


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def resolve_baseline_paths(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> BaselinePaths:
    """Resolve every path and prove the run is directly under the intended output base."""
    output_base = _resolve_project_path(str(config["project"]), project_root)
    run_name = str(config["run_name"])
    if not run_name or Path(run_name).name != run_name or any(char in run_name for char in ("/", "\\")):
        raise BaselineTrainingError(f"Unsafe baseline run name: {run_name!r}")
    run_dir = (output_base / run_name).resolve()
    if run_dir.parent != output_base:
        raise BaselineTrainingError("Resolved run directory escaped the baseline output directory.")
    try:
        output_base.relative_to(project_root.resolve())
    except ValueError as exc:
        raise BaselineTrainingError("Baseline output must remain inside the project workspace.") from exc
    return BaselinePaths(
        config=_resolve_project_path(str(config.get("_config_path", "configs/training/baseline_public_v1_yolov8s.yaml")), project_root),
        model=_resolve_project_path(str(config["model"]), project_root),
        source_dataset=_resolve_project_path(str(config["source_dataset"]), project_root),
        data_yaml=_resolve_project_path(str(config["data"]), project_root),
        output_base=output_base,
        run_dir=run_dir,
        amp_check_artifact=_resolve_project_path(str(config["amp_check_artifact"]), project_root),
        source_state_manifest=_resolve_project_path(str(config["source_state_manifest"]), project_root),
    )


def verify_file_sha256(path: Path, expected_sha256: str, label: str) -> str:
    """Verify one immutable input file against its frozen SHA-256."""
    if not path.is_file():
        raise BaselineTrainingError(f"Required {label} is missing: {path}")
    observed = sha256_file(path)
    if observed.casefold() != expected_sha256.casefold():
        raise BaselineTrainingError(
            f"{label.capitalize()} SHA-256 mismatch: expected {expected_sha256}, observed {observed}."
        )
    return observed


def validate_train_val_yaml(path: Path, source_dataset: Path) -> dict[str, Any]:
    """Require exactly train, val, and the frozen four-class names; never a test key."""
    try:
        import yaml
    except ImportError as exc:
        raise BaselineTrainingError("PyYAML is unavailable in the baseline environment.") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BaselineTrainingError(f"Could not read train/val dataset YAML: {path}") from exc
    if not isinstance(value, dict) or set(value) != {"train", "val", "names"}:
        keys = sorted(value) if isinstance(value, dict) else []
        raise BaselineTrainingError(
            f"Dataset YAML must contain exactly train, val, names and no test key; found {keys}."
        )
    if value["names"] != CLASS_NAMES:
        raise BaselineTrainingError("Dataset YAML four-class mapping or numeric order changed.")
    expected = {
        "train": (source_dataset / "images" / "train").resolve(),
        "val": (source_dataset / "images" / "val").resolve(),
    }
    observed: dict[str, Path] = {}
    for split in ("train", "val"):
        raw = Path(str(value[split]))
        observed[split] = raw.resolve() if raw.is_absolute() else (path.parent / raw).resolve()
        if observed[split] != expected[split]:
            raise BaselineTrainingError(
                f"Dataset YAML {split} path is {observed[split]}, expected {expected[split]}."
            )
        if not observed[split].is_dir() or not (source_dataset / "labels" / split).is_dir():
            raise BaselineTrainingError(f"Approved source {split} image/label directories are missing.")
    return value


def validate_dataset_inputs(paths: BaselinePaths) -> None:
    """Validate only approved train/val configuration and export provenance metadata."""
    validate_train_val_yaml(paths.data_yaml, paths.source_dataset)
    validation = _read_json(
        paths.source_dataset / "validation_report.json", "approved export validation report"
    )
    exclusions = validation.get("exclusions", {})
    if (
        validation.get("phase2c1_complete") is not True
        or validation.get("status") != "passed"
        or not isinstance(exclusions, dict)
        or exclusions.get("passed") is not True
    ):
        raise BaselineTrainingError("Approved YOLO export validation status is not passed.")
    for filename in ("export_manifest.json", "export_report.json", "validation_report.json"):
        if not (paths.source_dataset / filename).is_file():
            raise BaselineTrainingError(f"Approved export provenance file is missing: {filename}")


def ensure_fresh_run_available(run_dir: Path) -> None:
    """Refuse reuse of any pre-existing experiment path."""
    if run_dir.exists():
        raise BaselineTrainingError(f"Refusing to overwrite existing baseline run: {run_dir}")


def reserve_fresh_run_directory(run_dir: Path) -> None:
    """Atomically reserve the fresh run path before recording provenance."""
    ensure_fresh_run_available(run_dir)
    try:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise BaselineTrainingError(f"Refusing to overwrite existing baseline run: {run_dir}") from exc
    except OSError as exc:
        raise BaselineTrainingError(f"Could not reserve baseline run directory: {run_dir}") from exc


def build_fresh_ultralytics_args(
    config: Mapping[str, Any], paths: BaselinePaths
) -> dict[str, Any]:
    """Build the exact absolute-path argument set passed to Ultralytics."""
    arguments = dict(FROZEN_TRAINING_PARAMETERS)
    arguments.update(
        {
            "model": str(paths.model),
            "data": str(paths.data_yaml),
            "project": str(paths.output_base),
            "name": str(config["run_name"]),
            "save_dir": str(paths.run_dir),
            "exist_ok": False,
        }
    )
    return arguments


def collect_dataset_fingerprints(paths: BaselinePaths) -> dict[str, Any]:
    """Fingerprint frozen export metadata and the train/val-only view without scanning test."""
    artifacts: dict[str, Any] = {}
    candidates = {
        "export_manifest": paths.source_dataset / "export_manifest.json",
        "export_report": paths.source_dataset / "export_report.json",
        "validation_report": paths.source_dataset / "validation_report.json",
        "train_val_dataset_yaml": paths.data_yaml,
    }
    for name, path in candidates.items():
        artifacts[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    artifacts["test_split_scanned"] = False
    return artifacts


def verify_recorded_fingerprints(
    paths: BaselinePaths, observed: Mapping[str, Any]
) -> None:
    """Require resume inputs to match the fingerprints frozen at fresh start."""
    recorded = _read_json(
        paths.run_dir / "reproducibility" / "dataset_export_fingerprints.json",
        "recorded dataset/export fingerprints",
    )
    if recorded != dict(observed):
        raise BaselineTrainingError(
            "Current dataset/export fingerprints differ from the fresh-start record."
        )


def collect_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify CUDA/framework identity and return a reproducible environment snapshot."""
    try:
        import torch
        import torchvision
        import ultralytics
        from ultralytics.utils import WEIGHTS_DIR
    except ImportError as exc:
        raise BaselineTrainingError("Required baseline framework packages are unavailable.") from exc
    observed_versions = {
        "python": platform.python_version(),
        "ultralytics": str(ultralytics.__version__),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda": str(torch.version.cuda),
    }
    if observed_versions != config["expected_environment"]:
        raise BaselineTrainingError(
            f"Baseline environment version drift: expected {config['expected_environment']}, "
            f"observed {observed_versions}."
        )
    configured_amp_artifact = _resolve_project_path(
        str(config["amp_check_artifact"]), PROJECT_ROOT
    )
    ultralytics_amp_artifact = (Path(WEIGHTS_DIR) / "yolo26n.pt").resolve()
    if configured_amp_artifact != ultralytics_amp_artifact:
        raise BaselineTrainingError(
            "Configured AMP-check artifact does not match Ultralytics WEIGHTS_DIR; "
            "refusing a run that could trigger another download."
        )
    if not torch.cuda.is_available():
        raise BaselineTrainingError("CUDA is not available; refusing to start the GPU baseline.")
    device_index = int(config["device"])
    if device_index >= torch.cuda.device_count():
        raise BaselineTrainingError(f"Configured CUDA device {device_index} is unavailable.")
    properties = torch.cuda.get_device_properties(device_index)
    gpu_name = torch.cuda.get_device_name(device_index)
    driver: dict[str, Any]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        driver = {"query": result.stdout.strip(), "error": None}
    except (OSError, subprocess.SubprocessError) as exc:
        driver = {"query": None, "error": str(exc)}
    package_versions = sorted(
        {
            f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
        },
        key=str.casefold,
    )
    return {
        "recorded_utc": utc_now(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "versions": observed_versions,
        "cuda_available": True,
        "cuda_device_index": device_index,
        "gpu": {
            "name": gpu_name,
            "expected_name": config["expected_gpu"],
            "matches_expected": gpu_name == config["expected_gpu"],
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        },
        "nvidia_smi": driver,
        "ultralytics_weights_dir": str(Path(WEIGHTS_DIR).resolve()),
        "ultralytics_amp_check_artifact": str(ultralytics_amp_artifact),
        "packages": package_versions,
    }


def collect_git_state(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Require a clean local Git commit and return its exact identity."""
    project_root = project_root.resolve()

    def run_git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(project_root), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BaselineTrainingError(
                "A local Git repository with an initial commit is required before baseline training."
            ) from exc
        return result.stdout.strip()

    top_level = Path(run_git("rev-parse", "--show-toplevel")).resolve()
    if top_level != project_root:
        raise BaselineTrainingError(
            f"Git top-level directory is {top_level}, expected {project_root}."
        )
    commit = run_git("rev-parse", "HEAD")
    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = run_git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise BaselineTrainingError(
            "Git worktree is not clean; commit the frozen baseline source before training."
        )
    return {
        "repository_root": str(project_root),
        "commit": commit,
        "branch": branch,
        "clean": True,
    }


def verify_recorded_git_state(paths: BaselinePaths, observed: Mapping[str, Any]) -> None:
    """Require resume to use the same clean commit recorded at fresh start."""
    manifest = _read_json(
        paths.run_dir / "reproducibility" / "run_manifest.json", "baseline run manifest"
    )
    if manifest.get("git") != dict(observed):
        raise BaselineTrainingError(
            "Current Git source identity differs from the fresh-start baseline commit."
        )


def _write_text_atomic(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False, newline="\n"
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    except OSError as exc:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise BaselineTrainingError(f"Could not write reproducibility artifact: {path}") from exc


def record_fresh_provenance(
    config: Mapping[str, Any],
    paths: BaselinePaths,
    resolved_args: Mapping[str, Any],
    environment: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    source_state: Mapping[str, Any],
    git_state: Mapping[str, Any],
) -> None:
    """Write pre-training provenance into an already reserved fresh run directory."""
    directory = paths.run_dir / "reproducibility"
    try:
        directory.mkdir(exist_ok=False)
        shutil.copyfile(paths.config, directory / "requested_config.yaml")
        shutil.copyfile(paths.data_yaml, directory / "train_val_dataset.yaml")
    except OSError as exc:
        raise BaselineTrainingError("Could not preserve frozen baseline configuration.") from exc
    write_json_atomic(directory / "resolved_config.json", dict(resolved_args))
    write_json_atomic(directory / "environment.json", dict(environment))
    write_json_atomic(directory / "dataset_export_fingerprints.json", dict(fingerprints))
    write_json_atomic(directory / "source_state.json", dict(source_state))
    packages = "\n".join(environment["packages"]) + "\n"
    _write_text_atomic(directory / "package_freeze.txt", packages)
    amp_artifact = {
        "path": str(paths.amp_check_artifact),
        "sha256": sha256_file(paths.amp_check_artifact),
        "classification": "automatic_ultralytics_amp_check_artifact",
        "is_training_input": False,
        "is_proposed_yolo26_experiment": False,
        "is_evaluated_model": False,
    }
    started = utc_now()
    write_json_atomic(
        directory / "run_manifest.json",
        {
            "schema_version": "baseline_public_v1.run_manifest.v1",
            "experiment_id": config["experiment_id"],
            "run_start_utc": started,
            "status": "prepared_for_fresh_training",
            "logical_experiment": {"target_epochs": 100, "single_resumable_run": True},
            "run_directory": str(paths.run_dir),
            "pretrained_checkpoint_sha256": sha256_file(paths.model),
            "git": dict(git_state),
            "source_tree_sha256": source_state["source_tree_sha256"],
            "amp_check_artifact": amp_artifact,
            "internal_test_exposed_to_training": False,
            "io_warning": "Smoke training reported slow image access; cache remains disabled and no dataset copy is authorized.",
        },
    )
    write_json_atomic(
        directory / "resume_history.json",
        {
            "schema_version": "baseline_public_v1.resume_history.v1",
            "sessions": [
                {
                    "kind": "fresh_start",
                    "recorded_utc": started,
                    "target_total_epochs": 100,
                    "run_directory": str(paths.run_dir),
                }
            ],
        },
    )


def _comparable_value(key: str, value: Any) -> Any:
    if key == "device":
        return str(value)
    if key in {"model", "data", "project", "save_dir"}:
        return str(Path(str(value)).resolve())
    return value


def validate_resume_checkpoint(
    inspection: Mapping[str, Any],
    config: Mapping[str, Any],
    paths: BaselinePaths,
) -> None:
    """Block unsafe resume before Ultralytics can fall back to a new run."""
    expected_checkpoint = (paths.run_dir / "weights" / "last.pt").resolve()
    observed_checkpoint = Path(str(inspection.get("checkpoint_path", ""))).resolve()
    if observed_checkpoint != expected_checkpoint:
        raise BaselineTrainingError(
            f"Resume must use this run's exact last.pt: {expected_checkpoint}, got {observed_checkpoint}."
        )
    if inspection.get("read_only_unchanged") is not True:
        raise BaselineTrainingError("Checkpoint inspection did not remain read-only.")
    if config.get("amp") is not True:
        raise BaselineTrainingError("The frozen public baseline must retain amp=true.")
    if inspection.get("scaler_state_present") is not True:
        raise BaselineTrainingError(
            "AMP scaler state is missing; complete training-state continuity cannot be guaranteed."
        )
    if inspection.get("appears_resumable") is not True:
        raise BaselineTrainingError("Checkpoint lacks resumable epoch/optimizer/train-argument state.")
    if inspection.get("target_total_epochs") != 100:
        raise BaselineTrainingError("Resume checkpoint target epoch count is not 100.")
    early_stopping = inspection.get("early_stopping")
    if not isinstance(early_stopping, dict) or early_stopping.get("consistent") is not True:
        raise BaselineTrainingError("Persistent early-stopping state is missing or inconsistent.")
    if early_stopping.get("patience") != 20:
        raise BaselineTrainingError("Persistent early-stopping patience is not 20.")
    if early_stopping.get("patience_exhausted") is not False:
        raise BaselineTrainingError(
            "Persistent early-stopping patience is exhausted; this experiment must not resume."
        )
    state = early_stopping.get("state")
    if not isinstance(state, dict):
        raise BaselineTrainingError("Persistent early-stopping state payload is unavailable.")
    if (
        Path(str(state.get("config_path", ""))).resolve() != paths.config.resolve()
        or str(state.get("config_sha256", "")).casefold()
        != sha256_file(paths.config).casefold()
        or state.get("run_directory") != str(paths.run_dir.resolve())
        or state.get("experiment_id") != config["experiment_id"]
    ):
        raise BaselineTrainingError("Early-stopping state config/run identity mismatch.")
    identity = inspection.get("run_output_identity", {})
    if not isinstance(identity, dict) or identity.get("matches_checkpoint_location") is not True:
        raise BaselineTrainingError("Checkpoint stored run/output identity does not match its location.")
    if Path(str(identity.get("actual_run_directory", ""))).resolve() != paths.run_dir:
        raise BaselineTrainingError("Resume checkpoint belongs to a different run directory.")
    train_args = inspection.get("train_args")
    if not isinstance(train_args, dict):
        raise BaselineTrainingError("Resume checkpoint has no train_args dictionary.")
    expected_args = build_fresh_ultralytics_args(config, paths)
    keys = set(FROZEN_TRAINING_PARAMETERS) | {
        "model", "data", "project", "name", "save_dir", "exist_ok"
    }
    mismatches = {
        key: {"expected": expected_args[key], "observed": train_args.get(key)}
        for key in sorted(keys)
        if _comparable_value(key, train_args.get(key))
        != _comparable_value(key, expected_args[key])
    }
    if mismatches:
        raise BaselineTrainingError(f"Resume scientific/run arguments changed: {mismatches}")


def build_resume_ultralytics_args(checkpoint: Path) -> dict[str, Any]:
    """Return the sole argument used for genuine Ultralytics resume."""
    return {"resume": str(checkpoint.resolve())}


def record_resume_attempt(
    paths: BaselinePaths,
    inspection: Mapping[str, Any],
    environment: Mapping[str, Any],
    git_state: Mapping[str, Any],
) -> None:
    """Atomically append one validated manual-session resume attempt."""
    directory = paths.run_dir / "reproducibility"
    history_path = directory / "resume_history.json"
    history = _read_json(history_path, "resume history")
    sessions = history.get("sessions")
    if not isinstance(sessions, list):
        raise BaselineTrainingError("Resume history sessions are missing or invalid.")
    timestamp = utc_now()
    session_id = timestamp.replace(":", "").replace("+", "_")
    session_path = directory / "sessions" / f"resume_{session_id}.json"
    session = {
        "kind": "resume",
        "recorded_utc": timestamp,
        "checkpoint": {
            key: inspection.get(key)
            for key in (
                "checkpoint_path", "size_bytes", "sha256", "stored_epoch",
                "completed_epoch_number", "next_epoch_number", "target_total_epochs",
                "optimizer_state_present", "scaler_state_present", "appears_resumable",
            )
        },
        "environment": dict(environment),
        "git": dict(git_state),
    }
    write_json_atomic(session_path, session)
    sessions.append(
        {
            "kind": "resume",
            "recorded_utc": timestamp,
            "completed_epoch_number": inspection.get("completed_epoch_number"),
            "checkpoint_sha256": inspection.get("sha256"),
            "session_record": str(session_path),
        }
    )
    write_json_atomic(history_path, history)
