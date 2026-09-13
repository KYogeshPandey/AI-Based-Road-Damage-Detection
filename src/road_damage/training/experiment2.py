"""Prepare and explicitly launch the matched YOLO26s experiment; no implicit acquisition."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence
from urllib.request import Request, urlopen
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_common import PROJECT_ROOT, sha256_file, write_json_atomic
from road_damage.training import baseline_support as shared
from road_damage.training.create_smoke_subset import validate_smoke_dataset
from road_damage.training.source_state import (
    SOURCE_STATE_MODE_COMMITTED_GIT_TREE,
    build_source_state_manifest,
)

LOGGER = logging.getLogger(__name__)
Error = shared.BaselineTrainingError
CONFIG = Path("configs/training/experiment2_yolo26s_matched.yaml")
CONFIG_SHA256 = "c299e47b9c04a95d3c5ddcd7215046d3a0d0118bbe05d382ac8e9211c3aadb01"
TOOLING = (str(CONFIG.as_posix()), "src/road_damage/training/experiment2.py",
           "tests/test_experiment2.py", "road-damage-project-docs/EXPERIMENT2_YOLO26S_RUNBOOK.md")
ADDITIONAL_MATCHED = {
    "mode": "train", "time": None, "resume": False, "split": "val",
    "warmup_momentum": .8, "warmup_bias_lr": .1, "box": 7.5, "cls": .5, "dfl": 1.5,
    "cls_pw": 0., "cls_remap": True, "single_cls": False, "freeze": None,
    "profile": False, "channels_last": False, "conf": None, "agnostic_nms": False,
    "classes": None, "distill_model": None,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Error(f"Expected a JSON object: {path}")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    """Reject redirects before opening a configured file or following a manifest path."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise Error(f"Expected a workspace-relative path: {relative}")
    candidate = root.absolute() / rel
    if candidate.resolve() != candidate or not candidate.is_relative_to(root.absolute()):
        raise Error(f"Symlink/junction or escaped path is not accepted: {candidate}")
    return candidate


def load_config(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Require the reviewed config, baseline identity, and every matched parameter."""
    path = _path(root, CONFIG)
    shared.verify_file_sha256(path, CONFIG_SHA256, "Experiment 2 configuration")
    config = _read(path)
    training = config.get("training")
    fraction = training.get("fraction") if isinstance(training, Mapping) else None
    if type(fraction) is not float or fraction != 1.0:
        raise Error("Experiment 2 training fraction must be exactly the float 1.0.")
    baseline_path = _path(root, config["baseline_config"])
    shared.verify_file_sha256(baseline_path, config["baseline_config_sha256"], "baseline config")
    baseline = shared.load_baseline_config(baseline_path, root)
    expected = {**shared.FROZEN_TRAINING_PARAMETERS, **ADDITIONAL_MATCHED, "end2end": True}
    if config["training"] != expected:
        raise Error("Experiment 2 training parameters differ from the reviewed matched protocol.")
    for key in shared.FROZEN_TRAINING_PARAMETERS:
        if config["training"][key] != baseline[key]:
            raise Error(f"Baseline parameter parity failed: {key}")
    return config


def verify_baseline_args(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    import yaml
    path = _path(root, config["baseline_args"])
    shared.verify_file_sha256(path, config["baseline_args_sha256"], "completed baseline args")
    args = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, expected in config["training"].items():
        if key in {"resume", "end2end"}:
            continue  # Baseline's saved resume path and architecture-native setting.
        observed = args.get(key)
        if key == "device":
            observed = int(observed)
        if observed != expected:
            raise Error(f"Recorded baseline parameter differs: {key}={observed!r}")
    if args.get("end2end") is not None:
        raise Error("Recorded YOLOv8s architecture mode changed.")
    return {"sha256": config["baseline_args_sha256"], "matched": True}


def architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the installed small detection YAML without constructing a model."""
    import ultralytics
    from ultralytics.nn.tasks import guess_model_task, yaml_model_load
    if ultralytics.__version__ != "8.4.130":
        raise Error("Architecture resolution requires installed Ultralytics 8.4.130.")
    directory = Path(ultralytics.__file__).parent / "cfg/models/26"
    shared.verify_file_sha256(directory / "yolo26.yaml", config["model"]["architecture_sha256"],
                              "installed YOLO26 architecture")
    resolved = yaml_model_load(directory / "yolo26s.yaml")
    if (resolved.get("scale") != "s" or resolved["scales"].get("s") != [.5, .5, 1024]
            or guess_model_task(resolved) != "detect" or resolved.get("end2end") is not True
            or resolved.get("reg_max") != 1):
        raise Error("Installed architecture is not the exact native YOLO26s detection model.")
    return resolved


def verify_environment(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    expected_executable = _path(root, ".venv/Scripts/python.exe")
    if Path(sys.executable).resolve() != expected_executable:
        raise Error(f"Use only the intended interpreter: {expected_executable}")
    expected = dict(config["expected_environment"])
    numpy_version = expected.pop("numpy")
    if importlib.metadata.version("numpy") != numpy_version:
        raise Error("NumPy version differs from 2.5.2.")
    env_config = {**config, **config["training"], "expected_environment": expected}
    environment = shared.collect_environment(env_config)
    if not environment["gpu"]["matches_expected"]:
        raise Error("CUDA device 0 must be NVIDIA GeForce RTX 2050.")
    if not 3.5 * 1024**3 <= environment["gpu"]["total_memory_bytes"] <= 4.5 * 1024**3:
        raise Error("GPU memory does not match the reviewed 4 GB device.")
    shared.verify_file_sha256(_path(root, config["amp_check_artifact"]),
                              config["amp_check_artifact_sha256"], "existing AMP-check weight")
    environment["versions"]["numpy"] = numpy_version
    return environment


def verify_dataset(root: Path, config: Mapping[str, Any], *, hashes: bool = False) -> dict[str, Any]:
    """Preflight reads metadata/names only; launch may hash exactly train/val pairs."""
    source = _path(root, config["source_dataset"])
    data = _path(root, config["data"])
    shared.verify_file_sha256(data, config["train_val_yaml_sha256"], "train/val-only YAML")
    shared.validate_train_val_yaml(data, source)
    for filename, digest in config["dataset_metadata_sha256"].items():
        shared.verify_file_sha256(_path(root, Path(config["source_dataset"]) / filename), digest,
                                  f"approved export metadata {filename}")
    report = _read(source / "validation_report.json")
    if (report.get("status") != "passed" or report.get("phase2c1_complete") is not True
            or report.get("exclusions", {}).get("passed") is not True):
        raise Error("Approved export validation/exclusion evidence failed.")
    observed = {}
    for split in ("train", "val"):
        if report["split_image_counts"][split] != config["split_counts"][split]:
            raise Error(f"Approved {split} count changed.")
        sets = []
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            directory = _path(root, Path(config["source_dataset"]) / kind / split)
            names = []
            for file in directory.iterdir():
                _path(root, file.relative_to(root))
                if (not re.fullmatch(r"(?:India|Japan)_\d+" + re.escape(suffix), file.name)
                        or not file.is_file()):
                    raise Error(f"Unexpected source {split} entry: {file.name}")
                names.append(file.stem)
            if len(names) != len(set(names)) or len(names) != config["split_counts"][split]:
                raise Error(f"Source {split} {kind} count mismatch.")
            sets.append(set(names))
        if sets[0] != sets[1]:
            raise Error(f"Source {split} image/label pairs differ.")
        observed[split] = len(sets[0])
    if hashes:
        counts = {"train": 0, "val": 0}
        seen = set()
        for record in _read(source / "export_manifest.json")["images"]:
            split = record["derived_split"]
            if split not in ("train", "val"):
                continue  # Never resolve or open paths in other split records.
            filename = record["original_filename"]
            if not re.fullmatch(r"(?:India|Japan)_\d+\.jpg", filename) or (split, filename) in seen:
                raise Error("Invalid/duplicate training manifest identity.")
            seen.add((split, filename))
            for kind, field, digest_key in (("images", "exported_image_path", "exported_image_sha256"),
                                           ("labels", "exported_label_path", "label_sha256")):
                name = filename if kind == "images" else str(Path(filename).with_suffix(".txt"))
                expected = f"{kind}/{split}/{name}"
                if record[field] != expected:
                    raise Error("Manifest path is outside the exact train/val pair.")
                shared.verify_file_sha256(_path(root, Path(config["source_dataset"]) / expected),
                                          record[digest_key], f"{split} source {kind}")
            counts[split] += 1
        if counts != dict(config["split_counts"]):
            raise Error("Manifest train/val coverage is incomplete.")
    return {"counts": observed, "metadata_sha256": config["dataset_metadata_sha256"],
            "train_val_bytes_verified": hashes, "internal_test_files_accessed": False}


def _preserve_dataset_cache_metadata_without_write(
    prefix: str,
    path: Path,
    x: dict[str, Any],
    version: str,
) -> None:
    """Preserve Ultralytics' in-memory cache schema without writing cache files."""
    del prefix, path
    x["version"] = version


@contextmanager
def offline_framework(root: Path, config: Mapping[str, Any], mode: str | None = None) -> Iterator[None]:
    """Disable framework downloads/installations/integrations and source-cache writes."""
    from ultralytics.utils import checks, downloads, callbacks
    from ultralytics.data import dataset
    allowed = {_path(root, config["model"]["path"]), _path(root, config["amp_check_artifact"])}
    if mode is not None:
        allowed.update(run_directory(root, config, mode) / "weights" / name for name in ("best.pt", "last.pt"))

    def local_asset(file: Any, *args: Any, **kwargs: Any) -> str:
        path = Path(file).resolve()
        if path not in allowed or not path.is_file():
            raise Error(f"Automatic weight acquisition/fallback is prohibited: {file}")
        return str(path)

    def no_download(*args: Any, **kwargs: Any) -> None:
        raise Error("Automatic download is prohibited; perform acquisition separately.")

    with patch.object(downloads, "attempt_download_asset", side_effect=local_asset), \
         patch.object(downloads, "safe_download", side_effect=no_download), \
         patch.object(checks, "AUTOINSTALL", False), \
         patch.object(checks, "check_pip_update_available", return_value=None), \
         patch.object(callbacks, "add_integration_callbacks", return_value=None), \
         patch.object(
             dataset,
             "save_dataset_cache_file",
             side_effect=_preserve_dataset_cache_metadata_without_write,
         ):
        yield


def verify_checkpoint_architecture(model: Any, expected: Mapping[str, Any], *, classes: int = 80) -> None:
    """Check loaded checkpoint structure, not its filename alone; no forward pass."""
    actual = getattr(model, "yaml", {})
    for key in ("backbone", "head", "scales", "scale", "end2end", "reg_max"):
        if actual.get(key) != expected.get(key):
            raise Error(f"Pretrained checkpoint is not YOLO26s: architecture {key} differs.")
    head = model.model[-1]
    if (type(head).__name__ != "Detect" or head.nc != classes or head.reg_max != 1
            or getattr(head, "end2end", False) is not True):
        raise Error(f"Expected the {classes}-class native YOLO26s Detect head.")


def _verify_completed_checkpoint(path: Path, config: Mapping[str, Any]) -> None:
    from ultralytics.nn.tasks import torch_safe_load
    checkpoint, _ = torch_safe_load(str(path), safe_only=True)
    model = checkpoint.get("ema") if checkpoint.get("ema") is not None else checkpoint.get("model")
    verify_checkpoint_architecture(model, architecture(config), classes=4)
    if model.names != shared.CLASS_NAMES:
        raise Error("Completed checkpoint road-damage class names/order changed.")


def _inspect_weight(root: Path, config: Mapping[str, Any]) -> None:
    from ultralytics.nn.tasks import torch_safe_load
    with offline_framework(root, config):
        checkpoint, _ = torch_safe_load(str(_path(root, config["model"]["path"])), safe_only=True)
    model = checkpoint.get("ema") if checkpoint.get("ema") is not None else checkpoint.get("model")
    verify_checkpoint_architecture(model, architecture(config))


def verify_identity(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["model"]
    identity = _read(_path(root, spec["identity"]))
    expected = {"schema_version": "experiment2.pretrained_identity.v1", "family": "YOLO26",
                "variant": "s", "task": "detect", "path": spec["path"], "source": spec["source"],
                "ultralytics": "8.4.130", "architecture_sha256": spec["architecture_sha256"]}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise Error("Pretrained identity metadata changed or does not identify YOLO26s.")
    digest = identity.get("sha256", "")
    if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(identity.get("size_bytes")) is not int or identity["size_bytes"] <= 100000
            or identity.get("publisher_digest") != f"sha256:{digest}"):
        raise Error("Pretrained SHA/size/publisher digest has not been frozen.")
    model_path = _path(root, spec["path"])
    shared.verify_file_sha256(model_path, digest, "frozen YOLO26s pretrained weight")
    if model_path.stat().st_size != identity["size_bytes"]:
        raise Error("YOLO26s weight size differs from the frozen identity.")
    _inspect_weight(root, config)
    return identity


def acquire_pretrained(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Explicit USER operation: pinned publisher release, hash first, then inspect/freeze."""
    config = load_config(root)
    verify_environment(root, config)
    architecture(config)
    spec = config["model"]
    identity_path = _path(root, spec["identity"])
    if identity_path.exists():
        raise Error("Pretrained identity already exists; refusing to replace it.")
    request = Request(spec["release_api"], headers={"User-Agent": "road-damage-experiment2",
                                                  "Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=60) as response:
        release = json.load(response)
    assets = [a for a in release.get("assets", []) if a.get("name") == "yolo26s.pt"]
    if release.get("tag_name") != "v8.4.0" or len(assets) != 1:
        raise Error("Pinned publisher release does not contain exactly one yolo26s.pt.")
    asset = assets[0]
    publisher_digest = asset.get("digest", "")
    if (asset.get("browser_download_url") != spec["source"] or
            re.fullmatch(r"sha256:[0-9a-f]{64}", publisher_digest) is None
            or type(asset.get("size")) is not int or not 100000 < asset["size"] < 200_000_000):
        raise Error("Publisher asset URL/SHA/size unavailable; stop for manual review, no fallback.")
    model_path = _path(root, spec["path"])
    digest = publisher_digest.split(":", 1)[1]
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        partial = model_path.with_suffix(".pt.partial")
        with partial.open("xb") as target, urlopen(spec["source"], timeout=60) as source:
            size = 0
            while block := source.read(1024 * 1024):
                size += len(block)
                if size > asset["size"]:
                    raise Error("Download exceeded publisher size; partial retained for diagnosis.")
                target.write(block)
        shared.verify_file_sha256(partial, digest, "downloaded publisher asset")
        if partial.stat().st_size != asset["size"]:
            raise Error("Downloaded size differs; partial retained for diagnosis.")
        # Windows rename refuses an existing target; never replace an acquired model.
        if model_path.exists():
            raise Error("Model appeared during acquisition; refusing overwrite.")
        partial.rename(model_path)
    shared.verify_file_sha256(model_path, digest, "publisher YOLO26s weight")
    if model_path.stat().st_size != asset["size"]:
        raise Error("Local YOLO26s file size differs from publisher metadata.")
    _inspect_weight(root, config)
    identity = {"schema_version": "experiment2.pretrained_identity.v1", "family": "YOLO26",
        "variant": "s", "task": "detect", "path": spec["path"], "source": spec["source"],
        "size_bytes": asset["size"], "sha256": digest, "publisher_digest": publisher_digest,
        "publisher_asset_id": asset["id"], "publisher_release": "v8.4.0",
        "ultralytics": "8.4.130", "architecture_sha256": spec["architecture_sha256"],
        "recorded_utc": shared.utc_now(), "training_started": False}
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    with identity_path.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return identity


def verify_git(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    state = shared.collect_git_state(root)
    if state["commit"] == config["prerequisite_commit"]:
        raise Error("Commit the reviewed Experiment 2 tooling and acquired identity before execution.")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["prerequisite_commit"], "HEAD"],
                   cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *TOOLING, config["model"]["identity"]],
                   cwd=root, check=True, capture_output=True)
    source_state = build_source_state_manifest(
        root, mode=SOURCE_STATE_MODE_COMMITTED_GIT_TREE
    )
    if source_state.get("git_commit") != state["commit"]:
        raise Error("Git HEAD changed while Experiment 2 source state was being verified.")
    state["source_state"] = source_state
    return state


def run_directory(root: Path, config: Mapping[str, Any], mode: str) -> Path:
    if mode not in {"smoke", "train"}:
        raise Error("Only explicit smoke or train modes can select a run directory.")
    name = config["smoke"]["run_name"] if mode == "smoke" else config["run_name"]
    if Path(name).name != name or any(c in name for c in ("/", "\\")):
        raise Error("Invalid experiment run name.")
    if config["project"] != "outputs/training/experiment2_yolo26s_matched":
        raise Error("Experiment 2 output namespace cannot change.")
    return _path(root, Path(config["project"]) / name)


def preflight(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Read-only inspection. Missing acquisition/commit is a blocker, never an action."""
    config = load_config(root)
    checks: dict[str, Any] = {}
    blockers = {}
    for name, operation in (
        ("baseline_parity", lambda: verify_baseline_args(root, config)),
        ("environment", lambda: verify_environment(root, config)),
        ("architecture", lambda: architecture(config)),
        ("dataset", lambda: verify_dataset(root, config)),
        ("pretrained_identity", lambda: verify_identity(root, config)),
        ("git", lambda: verify_git(root, config)),
        ("fresh_output", lambda: verify_output(root, config)),
    ):
        try:
            checks[name] = operation()
        except (Error, OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as exc:
            blockers[name] = str(exc)
    return {"phase": "preflight", "ready_for_smoke": not blockers, "blockers": blockers,
            "checks": checks, "training_performed": False, "inference_performed": False,
            "internal_test_files_accessed": False}


def verify_output(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    directory = run_directory(root, config, "train")
    shared.ensure_fresh_run_available(directory)
    parent = directory.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise Error(f"Experiment output ancestor is not writable: {parent}")
    return {"directory": str(directory), "exists": False, "writable_ancestor": str(parent)}


def training_arguments(root: Path, config: Mapping[str, Any], mode: str, data: Path) -> dict[str, Any]:
    run_dir = run_directory(root, config, mode)
    # Reuse the original argument builder without changing its YOLOv8s callers/policy.
    paths = shared.BaselinePaths(_path(root, CONFIG), _path(root, config["model"]["path"]),
        _path(root, config["source_dataset"]), data, run_dir.parent, run_dir,
        _path(root, config["amp_check_artifact"]), run_dir / "source_state.json")
    args = shared.build_fresh_ultralytics_args({"run_name": run_dir.name}, paths)
    args.update(config["training"])
    if mode == "smoke":
        args.update(config["smoke"]["overrides"])
    return args


def _smoke_data(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    smoke = _path(root, config["smoke"]["dataset"])
    shared.verify_file_sha256(smoke / "smoke_manifest.json", config["smoke"]["manifest_sha256"],
                              "existing smoke manifest")
    shared.verify_file_sha256(smoke / "data.yaml", config["smoke"]["yaml_sha256"], "smoke YAML")
    # Validate redirect-free paths before the trusted independent byte/lineage validator.
    for record in _read(smoke / "smoke_manifest.json")["samples"]:
        if record.get("source_split") not in ("train", "val") or record.get("smoke_split") != record["source_split"]:
            raise Error("Smoke lineage must preserve source train/val.")
        filename = record.get("filename", "")
        if re.fullmatch(r"(?:India|Japan)_\d+\.jpg", filename) is None:
            raise Error("Invalid smoke filename.")
        for prefix, base in (("source", config["source_dataset"]), ("copied", config["smoke"]["dataset"])):
            for kind, suffix in (("image", ".jpg"), ("label", ".txt")):
                expected = f"{base}/{kind}s/{record['source_split']}/{Path(filename).stem}{suffix}"
                if record.get(f"{prefix}_{kind}_path") != expected:
                    raise Error("Smoke path must remain in its exact approved source/copy split.")
                _path(root, expected)
    result = validate_smoke_dataset(smoke, _path(root, config["source_dataset"]), project_root=root)
    if not result["passed"]:
        raise Error(f"Existing smoke subset failed independent validation: {result['errors']}")
    return result


def _smoke_receipt(
    root: Path,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    source_sha: str,
    git_commit: str,
) -> dict[str, Any]:
    directory = run_directory(root, config, "smoke")
    manifest = _read(directory / "run_manifest.json")
    receipt = _read(directory / "completion.json")
    receipt_commit = receipt.get("git_commit")
    if (not isinstance(git_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", git_commit) is None
            or not isinstance(receipt_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", receipt_commit) is None
            or receipt_commit != git_commit
            or manifest.get("mode") != "smoke"
            or manifest.get("git_commit") != receipt_commit
            or manifest.get("source_tree_sha256") != receipt.get("source_tree_sha256")
            or receipt.get("mode") != "smoke" or receipt.get("status") != "COMPLETED"
            or receipt.get("epochs_completed") != 1 or receipt.get("model_sha256") != identity["sha256"]
            or receipt.get("config_sha256") != CONFIG_SHA256 or receipt.get("source_tree_sha256") != source_sha
            or receipt.get("smoke_manifest_sha256") != config["smoke"]["manifest_sha256"]
            or receipt.get("batches_completed", 0) != 32 or receipt.get("optimizer_updates", 0) < 1
            or receipt.get("amp_enabled") is not True or receipt.get("cuda_device") != "cuda:0"
            or receipt.get("optimizer_state_present") is not True or receipt.get("scaler_state_present") is not True
            or receipt.get("dataset_metadata_sha256") != config["dataset_metadata_sha256"]):
        raise Error("A completed, identical one-epoch smoke run is required before full training.")
    if set(receipt.get("artifacts", {})) != {"weights/last.pt", "weights/best.pt", "results.csv"}:
        raise Error("Smoke completion checkpoint evidence is incomplete.")
    for relative, digest in receipt["artifacts"].items():
        shared.verify_file_sha256(_path(root, directory.relative_to(root) / relative), digest, "smoke artifact")
    return receipt


def launch(mode: str, root: Path = PROJECT_ROOT) -> None:
    """Explicit fresh execution only; all checks repeat before any model.train call."""
    if root.absolute() != PROJECT_ROOT.absolute():
        raise Error("Project root override is not supported for real execution.")
    config = load_config(root)
    directory = run_directory(root, config, mode)
    shared.ensure_fresh_run_available(directory)
    checks = preflight(root)
    if checks["blockers"]:
        raise Error(f"Preflight blocked execution: {checks['blockers']}")
    identity = checks["checks"]["pretrained_identity"]
    git_state = checks["checks"]["git"]
    git_commit = git_state["commit"]
    source_state = git_state["source_state"]
    source_sha = source_state["source_tree_sha256"]
    verify_dataset(root, config, hashes=True)
    smoke = _smoke_data(root, config)
    if mode == "train":
        _smoke_receipt(root, config, identity, source_sha, git_commit)
    data_path = _path(root, Path(config["smoke"]["dataset"]) / "data.yaml") if mode == "smoke" else _path(root, config["data"])
    data = shared.validate_train_val_yaml(data_path, _path(root,
        config["smoke"]["dataset"] if mode == "smoke" else config["source_dataset"]))
    args = training_arguments(root, config, mode, directory / "train_val.yaml")
    # The snapshot uses absolute train/val paths; canonical YAML and source bytes stay unchanged.
    for split in ("train", "val"):
        value = Path(data[split])
        data[split] = str(value if value.is_absolute() else (data_path.parent / value).resolve())
    shared.reserve_fresh_run_directory(directory)
    import yaml
    (directory / "train_val.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    provenance = {"mode": mode, "status": "STARTED", "started_utc": shared.utc_now(),
        "config": config, "resolved_args": args, "preflight": checks, "smoke_validation": smoke,
        "model_sha256": identity["sha256"], "config_sha256": CONFIG_SHA256,
        "git_commit": git_commit, "source_tree_sha256": source_sha,
        "internal_test_files_accessed": False}
    write_json_atomic(directory / "run_manifest.json", provenance)
    progress: dict[str, Any] = {"batches_completed": 0, "epochs_completed": 0}
    try:
        import torch
        from ultralytics import YOLO
        torch.cuda.reset_peak_memory_stats(0)
        with offline_framework(root, config, mode):
            model = YOLO(str(_path(root, config["model"]["path"])), task="detect")
            verify_checkpoint_architecture(model.model, architecture(config))

            def before_training(trainer: Any) -> None:
                if Path(trainer.save_dir).resolve() != directory or trainer.start_epoch != 0:
                    raise Error("Trainer run identity/start epoch changed.")
                for key, value in args.items():
                    if key in {"model", "save_dir"}:
                        continue
                    observed = getattr(trainer.args, key, None)
                    if key == "device":
                        observed = int(observed)
                    if observed != value:
                        raise Error(f"Trainer changed an explicit argument: {key}")
                if (trainer.data["names"] != shared.CLASS_NAMES or trainer.model.model[-1].nc != 4
                        or not trainer.model.end2end or trainer.model.model[-1].reg_max != 1
                        or str(trainer.device) != "cuda:0" or not bool(trainer.amp)
                        or type(trainer.optimizer) is not torch.optim.SGD):
                    raise Error("Runtime class mapping/YOLO26/CUDA/AMP/SGD policy failed.")
                for split in ("train", "val"):
                    if Path(trainer.data[split]).resolve() != Path(data[split]):
                        raise Error("Runtime dataset path changed.")

            def after_batch(trainer: Any) -> None:
                if not bool(torch.isfinite(trainer.loss).all()):
                    raise Error("Non-finite training loss; do not continue as a successful smoke run.")
                progress["batches_completed"] += 1

            def after_save(trainer: Any) -> None:
                progress.update(epochs_completed=int(trainer.epoch) + 1,
                    optimizer_updates=int(trainer.ema.updates), amp_enabled=bool(trainer.amp),
                    cuda_device=str(trainer.device),
                    optimizer_state_present=bool(trainer.optimizer.state_dict()["state"]),
                    scaler_state_present=bool(trainer.scaler.state_dict()))
                state = {"experiment_id": config["experiment_id"], "mode": mode,
                    "completed_epoch": int(trainer.epoch) + 1, "best_epoch": trainer.stopper.best_epoch,
                    "best_fitness": trainer.stopper.best_fitness, "patience": trainer.stopper.patience,
                    "last_sha256": sha256_file(Path(trainer.last)), "config_sha256": CONFIG_SHA256,
                    "optimizer_state_present": bool(trainer.optimizer.state_dict()["state"]),
                    "scaler_state_present": bool(trainer.scaler.state_dict())}
                write_json_atomic(directory / "epoch_state.json", state)

            model.add_callback("on_pretrain_routine_end", before_training)
            model.add_callback("on_train_batch_end", after_batch)
            model.add_callback("on_model_save", after_save)
            call_args = dict(args)
            call_args.pop("model")
            model.train(**call_args)
        with (directory / "results.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if (not rows or len(rows) != progress["epochs_completed"]
                or [int(r["epoch"]) for r in rows] != list(range(1, len(rows) + 1))
                or any(not math.isfinite(float(v)) for row in rows for v in row.values())
                or progress.get("optimizer_updates", 0) < 1
                or progress.get("optimizer_state_present") is not True
                or progress.get("scaler_state_present") is not True):
            raise Error("Training returned without complete finite epoch/optimizer evidence.")
        if mode == "smoke" and (len(rows) != 1 or progress["batches_completed"] != 32):
            raise Error("Smoke must complete exactly one epoch / 32 training batches.")
        artifacts = {relative: sha256_file(directory / relative)
                     for relative in ("weights/last.pt", "weights/best.pt", "results.csv")
                     if (directory / relative).stat().st_size > 0}
        if len(artifacts) != 3:
            raise Error("Completed checkpoint/results evidence is missing.")
        with offline_framework(root, config, mode):
            for name in ("best.pt", "last.pt"):
                _verify_completed_checkpoint(directory / "weights" / name, config)
        receipt = {"status": "COMPLETED", "mode": mode, **progress,
            "model_sha256": identity["sha256"], "config_sha256": CONFIG_SHA256,
            "git_commit": git_commit, "source_tree_sha256": source_sha,
            "smoke_manifest_sha256": config["smoke"]["manifest_sha256"],
            "dataset_metadata_sha256": config["dataset_metadata_sha256"], "artifacts": artifacts,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0), "completed_utc": shared.utc_now()}
        write_json_atomic(directory / "completion.json", receipt)
    except BaseException as exc:
        write_json_atomic(directory / "failure.json", {"status": "FAILED_TECHNICAL", "mode": mode,
            "reason": str(exc), **progress, "recorded_utc": shared.utc_now()})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("acquire-pretrained", "preflight", "smoke", "train"):
        actions.add_argument(f"--{action}", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.acquire_pretrained:
            print(json.dumps(acquire_pretrained(), indent=2))
        elif args.preflight:
            report = preflight()
            print(json.dumps(report, indent=2))
            return 0 if report["ready_for_smoke"] else 2
        else:
            launch("smoke" if args.smoke else "train")
    except (Error, OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as exc:
        LOGGER.error("Experiment 2 stopped: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted. Keep partial outputs; resume is not authorized by this fresh-run launcher.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
