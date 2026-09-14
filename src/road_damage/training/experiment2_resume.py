"""Controlled repeated canonical resumes; no checkpoint or scientific overrides.

Implements the Experiment 2 / RQ2 continuation boundary. Constants below are
approval identities, not an alternative training configuration. Scientific
settings still come only from the byte-pinned original YAML and checkpoint.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass, field, replace
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterator, Mapping
from unittest.mock import patch

from road_damage.training import experiment2 as tool

Error = tool.Error
ORIGINAL_COMMIT = "39070294088a725edf9092d861e5da4d750acd25"
ORIGINAL_SOURCE_SHA256 = "a33facf065a4004fd32f673df716c05b1f5783dbd5899481ef99c957ac1b2b57"
FROZEN_CONFIG_SHA256 = "0ae18ab9fa2d1098eefed24b2b75f5ed6f94b9e2bce773396d6402339cdaafe9"
CHECKPOINT_SHA256 = "12eb8c95a0bc9a34943836d8362d1058e86c8ff56db19e0ff906267451bb5809"
CHECKPOINT_BYTES = 40_374_049
MIGRATION_COMMIT = "0312871fb275725699913e08baaacab570c7296f"
MIGRATION_SOURCE_SHA256 = "7740bc7d2c1516969f081885ea4fe317ef3c9237cb5ce5612b01e4474e44d28a"
MIGRATION_CHECKPOINT_SHA256 = "335a4b825d1dd30f80caf68d844e12f023ad3a5e873bb9dd1d851ec9a946e5e4"
MIGRATION_CHECKPOINT_BYTES = 40_377_697
MIGRATION_ARTIFACTS = {
    "results.csv": "1f9839b3510e0e81e80c127008469be056d7fab71079759f2ee37eb4c29446e0",
    "resume_attempts/0001/STARTED.json": "6938980d0348f42fb8e0361516201ad455dcc9053a6053b6af662f1ccdb9d51a",
    "resume_attempts/0001/INTERRUPTED.json": "785467c4df17bfe1f322d6ecea09098d01c0fe15a727967d2341ce11df349897",
    "resume_attempts/0001/epochs/epoch_060.json": "d6ab80d1043106e58155247e9df803aaabfd0dd39ee43d1c4994ee5d78c72196",
}
PRETRAINED_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
RUN_RELATIVE = Path("outputs/training/experiment2_yolo26s_matched/") / "yolo26s_rdd2022-india-japan-v1.1.0_640_seed42"
ORIGINAL_ARTIFACTS = {
    "run_manifest.json": "b8e7920fd291fe904af1f46f3f5682ad5c4995aadfccc8d8757fc7cd848d3a6c",
    "epoch_state.json": "1571e22ab4a544130a681b7dc39333a0d16d10df529310da1af57d5aeb6a2f43",
    "failure.json": "3a5599304d64a6340adf6416432b3d51e7cad353efef8f670443cec73122b5f9",
    "results.csv": "664f7f04532cc60fcdd5776adb9b2302127091a16d44b5f5c4b5408ff2ad03e7",
    "args.yaml": "9c8c44b63a58036d4c9eb4c86c2ca25213d59de027848b0145fbc2ad4d37c6b5",
    "train_val.yaml": "aa6bbaf2f5b2081bfa702824b2a2fab6fe210cffba6389b4c707c035f0fa1903",
}
RESUME_FILES = (
    "src/road_damage/training/experiment2.py",
    "src/road_damage/training/experiment2_resume.py",
    "tests/test_experiment2.py",
    "tests/test_experiment2_resume.py",
    "road-damage-project-docs/EXPERIMENT2_YOLO26S_RUNBOOK.md",
)
FRAMEWORK_FILES = {
    "engine/trainer.py": "9ad7d4f050cbb632e81c0548bdfeca9d5d9b2792118c899ff5036c89b15959ba",
    "engine/model.py": "98D34492F56BBFF737BF140845ACDCBD3A1984000ABB33446F88A286FDA74778",
    "utils/torch_utils.py": "977cddb4bd1550f059d241227f6e00c0476663a43e11ad4b246e18ab3bd5d975",
    "utils/metrics.py": "974116d855398c48cd4cb30919b5ee424f8eaf555198086d7d5dc3068bc71ce2",
    "engine/validator.py": "61b75e0ad585d64c170a2ce225dff5cf05f45bd50a66202bc07e1e93ae725e0d",
}


@dataclass(frozen=True)
class ResumeState:
    root: Path
    run: Path
    checkpoint_path: Path
    config: dict[str, Any]
    checkpoint: dict[str, Any]
    manifest: dict[str, Any]
    epoch_state: dict[str, Any]
    saved_args: dict[str, Any]
    rows: list[dict[str, str]]
    dataset: dict[str, Any]
    identity: dict[str, Any]
    attempt_number: int = 1
    migration: bool = False
    segments: list[dict[str, Any]] = field(default_factory=list)
    history_sha256: dict[str, str] = field(default_factory=dict)
    input_artifacts: dict[str, str] = field(default_factory=dict)
    observed_progress: dict[str, Any] = field(default_factory=dict)

    @property
    def attempt(self) -> Path:
        return self.run / "resume_attempts" / f"{self.attempt_number:04d}"

    @property
    def snapshot(self) -> Path:
        return self.attempt / ("original_segment" if self.attempt_number == 1 else "input_snapshot")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Error(message)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"epoch", "train/box_loss", "train/cls_loss", "train/l1_loss",
                "val/box_loss", "val/cls_loss", "val/l1_loss", "metrics/mAP50-95(B)"}
    try:
        _require(bool(rows) and all(required <= row.keys() for row in rows), "Required epoch/loss results are missing.")
        _require(all(math.isfinite(float(v)) for row in rows for v in row.values()), "Non-finite epoch results.")
        _require([float(row["epoch"]) for row in rows] == list(range(1, len(rows) + 1)), "Results epochs are not contiguous.")
    except (TypeError, ValueError) as exc:
        raise Error("Invalid numerical epoch results.") from exc
    return rows


def _recover_stopper(rows: list[dict[str, str]]) -> dict[str, Any]:
    """8.4.130 detection fitness is mAP50-95; strict improvement, except initial zero.

    The training validator rounds fitness and mAP50-95 to five decimal places
    before BaseTrainer/EarlyStopping and CSV recording. No unrounded metric is
    inferred. Equal nonzero fitness does not restart patience.
    """
    best, best_epoch = 0.0, 0
    for epoch, row in enumerate(rows, 1):
        _require(float(row["epoch"]) == epoch, "Cannot recover patience from gapped results.")
        fitness = float(row["metrics/mAP50-95(B)"])
        _require(math.isfinite(fitness) and 0 <= fitness <= 1, "Invalid detection fitness.")
        if fitness > best or best == 0:
            best, best_epoch = fitness, epoch
    elapsed = len(rows) - best_epoch
    return {"best_epoch": best_epoch, "best_fitness": best, "patience": 20,
            "epochs_without_improvement": elapsed, "patience_exhausted": elapsed >= 20}


def _same_value(key: str, observed: Any, expected: Any) -> bool:
    if key == "device":
        return str(observed) == str(expected) == "0"
    if isinstance(expected, bool):
        return observed is expected
    if key == "fraction":
        return type(observed) is float and observed == 1.0
    return observed == expected


def _check_args(args: Mapping[str, Any], state: ResumeState, *, resumed: bool) -> None:
    expected = dict(state.saved_args)
    if resumed:
        expected.update(model=str(state.checkpoint_path), resume=str(state.checkpoint_path))
    _require(set(args) == set(expected), "Resume argument names differ from original args.yaml.")
    for key, value in expected.items():
        _require(_same_value(key, args.get(key), value), f"Frozen resume argument changed: {key}.")
    for key, value in state.config["training"].items():
        if key != "resume":
            _require(_same_value(key, args.get(key), value), f"Scientific argument changed: {key}.")
    for key, value in {
        "data": state.run / "train_val.yaml", "save_dir": state.run,
        "project": state.run.parent,
        "model": state.checkpoint_path if resumed else state.root / state.config["model"]["path"],
    }.items():
        _require(Path(str(args[key])).resolve() == value, f"Resume {key} path changed.")
    _require(args["name"] == state.run.name, "Resume run name changed.")


def _verify_framework() -> None:
    import ultralytics
    _require(ultralytics.__version__ == "8.4.130", "Resume requires Ultralytics 8.4.130.")
    for relative, digest in FRAMEWORK_FILES.items():
        tool.shared.verify_file_sha256(Path(ultralytics.__file__).parent / relative, digest, "reviewed resume framework")


def _verify_resume_git(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    git = tool.verify_git(root, config)
    _require(git["commit"] not in {ORIGINAL_COMMIT, MIGRATION_COMMIT}, "Commit the reviewed repeated-resume fix before using --resume.")
    subprocess.run(["git", "merge-base", "--is-ancestor", MIGRATION_COMMIT, "HEAD"], cwd=root,
                   check=True, capture_output=True)
    changed = subprocess.run(["git", "diff", "--no-renames", "--name-status", MIGRATION_COMMIT, "HEAD", "--"],
                             cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    _require(bool(changed), "Resume support has not been committed.")
    for line in changed:
        status, filename = line.split("\t", 1)
        _require(status in {"A", "M"} and filename in RESUME_FILES,
                 f"Unreviewed change outside the resume transition: {line}.")
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *RESUME_FILES], cwd=root,
                   check=True, capture_output=True)
    return git


def _load_checkpoint(path: Path) -> dict[str, Any]:
    # Deserialization is allowed only after the exact approved byte identity gate.
    import torch
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Error("Could not read the approved full checkpoint on CPU.") from exc
    _require(isinstance(checkpoint, dict), "Checkpoint is not a training-state mapping.")
    return checkpoint


def _validate_checkpoint(state: ResumeState) -> None:
    import torch
    c = state.checkpoint
    completed = state.epoch_state["completed_epoch"]
    _require(type(c.get("epoch")) is int and c["epoch"] == completed - 1 and 1 <= completed <= 100,
             "Unstripped checkpoint epoch disagrees with completed state.")
    _require(c.get("version") == "8.4.130", "Checkpoint framework version changed.")
    _require(type(c.get("updates")) is int and c["updates"] == state.epoch_state["optimizer_step_attempts"] > 0,
             "Checkpoint EMA attempt counter changed.")
    _require(c.get("best_fitness") == state.epoch_state["best_fitness"], "Checkpoint best fitness changed.")
    ema = c.get("ema")
    _require(isinstance(ema, torch.nn.Module), "Checkpoint EMA network is missing.")
    tool.verify_checkpoint_architecture(ema, tool.architecture(state.config), classes=4)
    _require(ema.names == tool.shared.CLASS_NAMES, "Checkpoint class order changed.")
    _require(all(torch.isfinite(v).all().item() for v in ema.state_dict().values()), "Non-finite EMA state.")
    optimizer = c.get("optimizer")
    _require(isinstance(optimizer, dict) and len(optimizer.get("state", {})) == 366,
             "Full SGD optimizer state (366 entries) is required.")
    groups = optimizer.get("param_groups", [])
    _require(len(groups) == 3 and [len(g.get("params", [])) for g in groups] == [126, 114, 126],
             "SGD parameter groups changed.")
    ids = [p for g in groups for p in g["params"]]
    _require(len(set(ids)) == 366 and set(ids) == set(optimizer["state"]), "Optimizer parameter identity mismatch.")
    _require([g.get("param_group") for g in groups] == ["weight", "bn", "bias"], "SGD group order changed.")
    for g in groups:
        expected_lr = .01 * (((1 - math.cos((completed - 1) * math.pi / 100)) / 2) * (.01 - 1) + 1)
        _require(math.isclose(g.get("lr", -1), expected_lr, abs_tol=1e-12) and g.get("initial_lr") == .01,
                 "Stored optimizer LR/schedule identity changed.")
    for values in optimizer["state"].values():
        buffer = values.get("momentum_buffer")
        _require(isinstance(buffer, torch.Tensor) and buffer.numel() > 0 and torch.isfinite(buffer).all().item(),
                 "A finite momentum buffer is required for every SGD parameter.")
    scaler = c.get("scaler")
    _require(isinstance(scaler, dict) and set(scaler) == {
        "scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}, "AMP scaler state missing.")
    _require(math.isfinite(scaler["scale"]) and scaler["scale"] > 0 and scaler["growth_factor"] == 2.
             and scaler["backoff_factor"] == .5 and scaler["growth_interval"] == 2000
             and type(scaler["_growth_tracker"]) is int and 0 <= scaler["_growth_tracker"] < 2000,
             "AMP scaler state invalid.")
    if completed == 33:
        _require(scaler["scale"] == 512. and scaler["_growth_tracker"] == 1645, "Original AMP state changed.")
    if state.migration:
        _require(scaler["scale"] == 512. and scaler["_growth_tracker"] == 864, "Migration AMP state changed.")
    _require(isinstance(c.get("train_args"), dict), "Checkpoint training args are missing.")
    _check_args(c["train_args"], state, resumed=completed > 33)
    embedded = c.get("train_results")
    _require(isinstance(embedded, dict) and set(embedded) == set(state.rows[0]), "Checkpoint results are missing.")
    for key, values in embedded.items():
        _require(values == [float(row[key]) for row in state.rows], f"Checkpoint/results.csv mismatch: {key}.")


def _original_segment() -> dict[str, Any]:
    return {"first_epoch": 1, "last_completed_epoch": 33, "git_commit": ORIGINAL_COMMIT,
            "source_tree_sha256": ORIGINAL_SOURCE_SHA256, "config_sha256": FROZEN_CONFIG_SHA256,
            "status": "INTERRUPTED", "internal_test_files_accessed": False, "artifacts": dict(ORIGINAL_ARTIFACTS)}


def _read_history(root: Path, run: Path, config: Mapping[str, Any], identity: Mapping[str, Any],
                  rows: list[dict[str, str]]) -> dict[str, Any]:
    """Reconcile append-only attempts; never silently skip an unknown or partial journal."""
    parent = tool._path(root, RUN_RELATIVE / "resume_attempts")
    attempts = sorted(parent.iterdir()) if parent.exists() and parent.is_dir() else []
    _require(not parent.exists() or parent.is_dir(), "Invalid resume history directory.")
    _require([p.name for p in attempts] == [f"{i:04d}" for i in range(1, len(attempts) + 1)],
             "Resume attempt numbering is not contiguous.")
    segments, hashes, latest, observed = [_original_segment()], {}, None, {}
    completed = 33
    for attempt in attempts:
        tool._path(root, attempt.relative_to(root))
        _require(attempt.is_dir(), "Invalid resume attempt.")
        started = tool._read(tool._path(root, (attempt / "STARTED.json").relative_to(root)))
        _require(started.get("status") == "STARTED"
                 and started.get("checkpoint_completed_epoch") == completed
                 and started.get("resumed_start_epoch") == completed + 1
                 and started.get("target_epoch") == 100
                 and started.get("config_sha256") == FROZEN_CONFIG_SHA256
                 and started.get("checkpoint_path") == str(run / "weights/last.pt")
                 and started.get("pretrained_lineage") == identity
                 and started.get("dataset_identities") == {"train_val_yaml": config["train_val_yaml_sha256"],
                                                          **config["dataset_metadata_sha256"]}
                 and started.get("internal_test_files_accessed") is False,
                 "Historical resume identity changed.")
        if attempt.name == "0001":
            _require(started.get("git_commit") == MIGRATION_COMMIT
                     and started.get("source_tree_sha256") == MIGRATION_SOURCE_SHA256,
                     "First-resume provenance changed.")
            for relative, digest in MIGRATION_ARTIFACTS.items():
                if relative.startswith("resume_attempts/"):
                    tool.shared.verify_file_sha256(tool._path(root, RUN_RELATIVE / relative), digest, "reviewed first resume")
        else:
            _require(started.get("schema_version") == "experiment2.resume.v2"
                     and started.get("history_sha256") == hashes
                     and started.get("prior_training_segments") == segments,
                     "Historical resume chain changed.")
        _require(re.fullmatch(r"[0-9a-f]{40}", str(started.get("git_commit"))) is not None
                 and re.fullmatch(r"[0-9a-f]{64}", str(started.get("source_tree_sha256"))) is not None,
                 "Historical source identity missing.")
        previous = started["durable_progress_at_start"]
        expected_sha = latest["record"]["last_sha256"] if latest else CHECKPOINT_SHA256
        _require(started.get("checkpoint_sha256") == expected_sha, "Historical starting checkpoint changed.")
        if latest:
            _require(all(previous.get(k) == latest["record"].get(k) for k in _progress_keys()),
                     "Historical resume did not start from the last durable counters.")
        epoch_paths = sorted(tool._path(root, (attempt / "epochs").relative_to(root)).iterdir())
        base_epoch = completed
        _require([p.name for p in epoch_paths] == [f"epoch_{n:03d}.json" for n in range(completed + 1, completed + len(epoch_paths) + 1)],
                 "Historical completed epoch journal has gaps.")
        for path in epoch_paths:
            completed += 1
            evidence = tool._read(tool._path(root, path.relative_to(root)))
            _require(evidence.get("epochs_completed") == completed <= len(rows)
                     and evidence.get("batches_completed") == completed * 3155
                     and evidence.get("config_sha256") == FROZEN_CONFIG_SHA256
                     and evidence.get("internal_test_files_accessed") is False,
                     "Historical epoch identity disagrees with complete results.")
            if started.get("schema_version") == "experiment2.resume.v2":
                _require(evidence.get("status") == "COMPLETE_EPOCH" and evidence.get("completed_epoch") == completed
                         and evidence.get("git_commit") == started["git_commit"]
                         and evidence.get("source_tree_sha256") == started["source_tree_sha256"],
                         "Completed epoch/source attribution changed.")
            tool._validate_training_evidence(rows[:completed], evidence, "train", config)
            for key, value in _recover_stopper(rows[:completed]).items():
                _require(_same_value(key, evidence.get(key), value), "Historical patience evidence changed.")
            for counter in ("optimizer_step_attempts", "optimizer_updates"):
                _require(evidence[counter] >= previous[counter], "Historical optimizer counters went backwards.")
            previous, latest = evidence, {"path": path, "record": evidence}
        terminals = [p for p in attempt.iterdir() if p.name in {"INTERRUPTED.json", "FAILED.json", "COMPLETED.json"}]
        _require(len(terminals) <= 1 and not (attempt / "COMPLETED.json").exists(), "Completed or ambiguous attempt locks the run.")
        status = "STARTED_UNFINISHED"
        observed = {}
        if terminals:
            terminal = tool._read(tool._path(root, terminals[0].relative_to(root)))
            status = terminal.get("status")
            _require(status == terminals[0].stem and terminal.get("internal_test_files_accessed") is False,
                     "Historical terminal status changed.")
            for key, value in started.items():
                if key != "status":
                    _require(terminal.get(key) == value, "Historical terminal provenance changed.")
            for key in _progress_keys():
                _require(terminal.get("durable_progress", {}).get(key) == previous.get(key), "Terminal/durable progress mismatch.")
            expected_sha = latest["record"]["last_sha256"] if latest else started["checkpoint_sha256"]
            _require(terminal.get("resulting_last_checkpoint_sha256") == expected_sha, "Interrupted checkpoint identity changed.")
            observed = terminal.get("observed_progress", {})
        if completed > base_epoch:
            segments.append({"first_epoch": base_epoch + 1, "last_completed_epoch": completed,
                "git_commit": started["git_commit"], "source_tree_sha256": started["source_tree_sha256"],
                "config_sha256": FROZEN_CONFIG_SHA256, "status": status, "internal_test_files_accessed": False})
        for path in sorted(attempt.rglob("*")):
            tool._path(root, path.relative_to(root))
            if path.is_file():
                hashes[path.relative_to(run).as_posix()] = tool.sha256_file(path)
    return {"attempt_number": len(attempts) + 1, "segments": segments, "hashes": hashes,
            "latest": latest, "completed": completed, "observed": observed}


def _progress_keys() -> tuple[str, ...]:
    return ("epochs_completed", "batches_completed", "optimizer_step_attempts", "optimizer_updates",
            "optimizer_state_present", "scaler_state_present", "amp_enabled", "cuda_device")


def _inspect_run(root: Path, config: dict[str, Any], identity: dict[str, Any], dataset: dict[str, Any]) -> ResumeState:
    import yaml
    tool.shared.verify_file_sha256(tool._path(root, tool.CONFIG), FROZEN_CONFIG_SHA256, "frozen Experiment 2 config")
    run = tool._path(root, RUN_RELATIVE)
    _require(run == tool.run_directory(root, config, "train") and run.is_dir(), "Canonical Experiment 2 run is missing.")
    # Check metadata/history redirects before opening or enumerating anything.
    for relative in (*ORIGINAL_ARTIFACTS, "completion.json", "resume_attempts"):
        tool._path(root, RUN_RELATIVE / relative)
    _require(not (run / "completion.json").exists(), "A full-run completion record already exists; resume refused.")
    checkpoint_path = tool._path(root, RUN_RELATIVE / "weights/last.pt")
    for relative, digest in ORIGINAL_ARTIFACTS.items():
        if relative not in {"results.csv", "epoch_state.json"}:
            tool.shared.verify_file_sha256(tool._path(root, RUN_RELATIVE / relative), digest, f"original {relative}")
    manifest = tool._read(run / "run_manifest.json")
    epoch_state = tool._read(run / "epoch_state.json")
    failure = tool._read(run / "failure.json")
    rows = _read_rows(run / "results.csv")
    history = _read_history(root, run, config, identity, rows)
    migration = epoch_state.get("completed_epoch") == 33 and len(rows) == 60
    if migration:
        _require(history["attempt_number"] >= 2 and history["completed"] == 60
                 and history["latest"]["path"] == run / "resume_attempts/0001/epochs/epoch_060.json",
                 "Only the reviewed epoch-60 migration is allowed.")
        tool.shared.verify_file_sha256(run / "epoch_state.json", ORIGINAL_ARTIFACTS["epoch_state.json"], "reviewed stale state")
        for relative, digest in MIGRATION_ARTIFACTS.items():
            tool.shared.verify_file_sha256(tool._path(root, RUN_RELATIVE / relative), digest, "reviewed epoch-60 recovery")
        epoch_state = {**history["latest"]["record"], "completed_epoch": 60,
                       "last_bytes": MIGRATION_CHECKPOINT_BYTES}
        _require(epoch_state["last_sha256"] == MIGRATION_CHECKPOINT_SHA256
                 and epoch_state["optimizer_step_attempts"] == 13038
                 and epoch_state["optimizer_updates"] == 13026, "Migration checkpoint/counters disagree.")
    elif history["attempt_number"] == 1:
        for relative in ("results.csv", "epoch_state.json"):
            tool.shared.verify_file_sha256(run / relative, ORIGINAL_ARTIFACTS[relative], "original durable state")
        _require(len(rows) == epoch_state.get("completed_epoch") == 33, "Original durable epoch mismatch.")
        epoch_state = {**_recover_stopper(rows), **epoch_state, "last_bytes": CHECKPOINT_BYTES}
        _require(epoch_state["last_sha256"] == CHECKPOINT_SHA256, "Original checkpoint identity changed.")
        _require(epoch_state.get("optimizer_step_attempts") == 7713 and epoch_state.get("optimizer_updates") == 7703,
                 "Original durable optimizer counters changed.")
    else:
        latest = history["latest"]
        _require(latest is not None and epoch_state.get("status") == "COMPLETE_EPOCH"
                 and epoch_state.get("epoch_record") == latest["path"].relative_to(run).as_posix()
                 and epoch_state.get("epoch_record_sha256") == tool.sha256_file(latest["path"]),
                 "Latest complete project state is missing, stale or unsealed.")
        _require({k: v for k, v in epoch_state.items() if k not in {"epoch_record", "epoch_record_sha256"}} == latest["record"],
                 "Project state differs from the latest complete epoch journal.")
        _require(epoch_state.get("results_sha256") == tool.sha256_file(run / "results.csv"), "Complete results SHA changed.")
    completed = epoch_state["completed_epoch"]
    _require(len(rows) == history["completed"] == completed and 33 <= completed < 100, "Results/durable epoch mismatch or target completed.")
    recovered = _recover_stopper(rows)
    for key, value in recovered.items():
        _require(_same_value(key, epoch_state.get(key), value), f"Recovered EarlyStopping disagrees: {key}.")
    _require(not recovered["patience_exhausted"], "EarlyStopping patience already exhausted.")
    tool.shared.verify_file_sha256(checkpoint_path, epoch_state["last_sha256"], "latest complete canonical last.pt")
    _require(checkpoint_path.stat().st_size == epoch_state["last_bytes"], "Complete checkpoint size changed.")
    _require(manifest.get("mode") == "train" and manifest.get("status") == "STARTED"
             and manifest.get("git_commit") == ORIGINAL_COMMIT
             and manifest.get("source_tree_sha256") == ORIGINAL_SOURCE_SHA256
             and manifest.get("config_sha256") == FROZEN_CONFIG_SHA256
             and manifest.get("config") == config
             and manifest.get("internal_test_files_accessed") is False, "Original run/config/safety provenance mismatch.")
    _require(manifest.get("model_sha256") == identity.get("sha256") == PRETRAINED_SHA256,
             "Original pretrained lineage changed.")
    preflight = manifest.get("preflight", {})
    _require(preflight.get("internal_test_files_accessed") is False
             and preflight.get("checks", {}).get("pretrained_identity") == identity,
             "Original pretrained/safety evidence mismatch.")
    _require(dataset.get("internal_test_files_accessed") is False
             and dataset.get("train_val_bytes_verified") is True
             and dataset.get("metadata_sha256") == config["dataset_metadata_sha256"]
             and dataset.get("counts") == {"train": 12620, "val": 2602}, "Train/val dataset verification failed.")
    expected_state = {"config_sha256": FROZEN_CONFIG_SHA256, "optimizer_state_present": True, "scaler_state_present": True}
    _require(all(_same_value(k, epoch_state.get(k), v) for k, v in expected_state.items()), "Durable state mismatch.")
    _require(failure.get("status") == "FAILED_TECHNICAL" and failure.get("mode") == "train"
             and failure.get("epochs_completed") == 33 and failure.get("batches_completed") == 104150
             and failure.get("optimizer_step_attempts") == 7715 and failure.get("optimizer_updates") == 7705
             and isinstance(failure.get("recorded_utc"), str), "Recorded original interruption is missing or changed.")
    data = tool.shared.validate_train_val_yaml(run / "train_val.yaml", tool._path(root, config["source_dataset"]))
    _require(set(data) == {"train", "val", "names"}, "Run YAML must be train/val only.")
    expected_args = tool.training_arguments(root, config, "train", run / "train_val.yaml")
    _require(manifest.get("resolved_args") == expected_args, "Original resolved training args changed.")
    saved_args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    artifacts = {relative: tool.sha256_file(run / relative) for relative in ORIGINAL_ARTIFACTS}
    state = ResumeState(root, run, checkpoint_path, config, _load_checkpoint(checkpoint_path),
                        manifest, epoch_state, saved_args, rows, data, identity,
                        history["attempt_number"], migration, history["segments"], history["hashes"], artifacts,
                        history["observed"] if history["attempt_number"] > 1 else failure)
    _validate_checkpoint(state)
    return state


def _initial_progress(state: ResumeState) -> dict[str, Any]:
    # Only durable epoch/checkpoint evidence is used; failure.json is never a counter source.
    completed = state.epoch_state["completed_epoch"]
    batches = completed * math.ceil(state.config["split_counts"]["train"] / state.config["training"]["batch"])
    _require(33 <= completed <= 100 and batches == completed * 3155, "Durable epoch/batch boundary changed.")
    return {"epochs_completed": completed, "batches_completed": batches,
            "optimizer_step_attempts": state.checkpoint["updates"],
            "optimizer_updates": state.epoch_state["optimizer_updates"],
            "optimizer_state_present": True, "scaler_state_present": True,
            "amp_enabled": True, "cuda_device": "cuda:0"}


def _assert_state_equal(actual: Any, expected: Any, label: str) -> None:
    import torch
    if isinstance(expected, torch.Tensor):
        _require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape
                 and torch.equal(actual.detach().cpu(), expected.detach().cpu().to(actual.dtype)), f"Restored {label} tensor differs.")
    elif isinstance(expected, Mapping):
        _require(isinstance(actual, Mapping) and set(actual) == set(expected), f"Restored {label} keys differ.")
        for key in expected:
            _assert_state_equal(actual[key], expected[key], label)
    elif isinstance(expected, (list, tuple)):
        _require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"Restored {label} length differs.")
        for a, e in zip(actual, expected):
            _assert_state_equal(a, e, label)
    else:
        _require(actual == expected, f"Restored {label} value differs.")


def _write_new_json(path: Path, record: Mapping[str, Any]) -> None:
    """Exclusive, durable publication: historical records are never replaced."""
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_epoch_state(path: Path, record: Mapping[str, Any]) -> None:
    """Commit pointer: fsync a new same-directory file, then atomically replace.

    Immutable epoch evidence is published first. A crash before pointer commit
    leaves a detectable disagreement and cannot authorize a later resume.
    """
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", delete=False,
                                     dir=path.parent, prefix=".epoch_state_", suffix=".tmp") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


class _ResumeCallbacks:
    def __init__(self, state: ResumeState) -> None:
        self.state = state
        self.progress = _initial_progress(state)
        self.durable = dict(self.progress)
        self.handle: Any = None
        self.stopper_state = _recover_stopper(state.rows)
        self.restored = False

    def before_dataset(self, trainer: Any) -> None:
        _check_args(vars(trainer.args), self.state, resumed=True)
        _require(Path(trainer.save_dir).resolve() == self.state.run and trainer.resume is True,
                 "Native trainer did not select the canonical genuine resume.")

    def restore(self, trainer: Any) -> None:
        import torch
        self.before_dataset(trainer)
        completed = self.progress["epochs_completed"]
        _require(not self.restored and trainer.start_epoch == completed and trainer.epochs == 100,
                 "Resume must start after the last completed epoch within the original 100 epochs.")
        _require(trainer.scheduler.last_epoch == completed - 1, "Scheduler position was not restored.")
        _require(type(trainer.optimizer) is torch.optim.SGD and bool(trainer.amp)
                 and str(trainer.device) == "cuda:0", "Resume optimizer/AMP/device changed.")
        _require(len(trainer.train_loader) == 3155 and len(trainer.train_loader.dataset) == 12620
                 and len(trainer.test_loader.dataset) == 2602, "Runtime train/validation counts changed.")
        _require(set(trainer.data) >= {"train", "val", "names"} and "test" not in trainer.data,
                 "Runtime data includes internal test.")
        for split in ("train", "val"):
            _require(Path(trainer.data[split]).resolve() == Path(self.state.dataset[split]).resolve(), "Runtime dataset path changed.")
        _require(trainer.data["names"] == tool.shared.CLASS_NAMES, "Runtime class mapping changed.")
        tool.verify_checkpoint_architecture(trainer.model, tool.architecture(self.state.config), classes=4)
        _assert_state_equal(trainer.optimizer.state_dict(), self.state.checkpoint["optimizer"], "optimizer/momentum")
        _assert_state_equal(trainer.scaler.state_dict(), self.state.checkpoint["scaler"], "GradScaler")
        _assert_state_equal(trainer.ema.ema.state_dict(), self.state.checkpoint["ema"].state_dict(), "EMA")
        _assert_state_equal(trainer.model.state_dict(), self.state.checkpoint["ema"].state_dict(), "network")
        _require(trainer.ema.updates == self.progress["optimizer_step_attempts"]
                 and trainer.best_fitness == self.stopper_state["best_fitness"], "EMA/fitness continuity changed.")
        for epoch in (0, 32, 33, 99, 100):
            expected = ((1 - math.cos(epoch * math.pi / 100)) / 2) * (.01 - 1) + 1
            _require(math.isclose(trainer.lf(epoch), expected, abs_tol=1e-12), "Cosine schedule no longer uses 100 epochs.")
        # Native EarlyStopping uses human epoch numbers, passed as epoch + 1.
        trainer.stopper.best_epoch = self.stopper_state["best_epoch"]
        trainer.stopper.best_fitness = self.stopper_state["best_fitness"]
        trainer.stopper.patience = 20
        trainer.stopper.possible_stop = self.stopper_state["epochs_without_improvement"] >= 19
        self.handle = tool._register_successful_optimizer_step_counter(trainer.optimizer, self.progress)
        self.restored = True

    def before_epoch(self, trainer: Any) -> None:
        _require(self.restored and trainer.epoch == self.progress["epochs_completed"]
                 and trainer.scheduler.last_epoch == trainer.epoch - 1, "Epoch/scheduler continuity failed.")
        _require(self.progress["batches_completed"] == trainer.epoch * 3155, "Partial epoch cannot be carried forward.")

    def before_batch(self, trainer: Any) -> None:
        _require(trainer.scheduler.last_epoch == trainer.epoch, "Scheduler failed to advance to the resumed epoch.")
        expected_lr = .01 * trainer.lf(trainer.epoch)
        _require(all(math.isclose(g["lr"], expected_lr, abs_tol=1e-12) for g in trainer.optimizer.param_groups),
                 "Runtime cosine LR restarted or changed.")

    def after_batch(self, trainer: Any) -> None:
        import torch
        _require(self.restored and torch.isfinite(trainer.loss).all().item(), "Non-finite resumed training loss.")
        self.progress["batches_completed"] += 1
        self.progress["optimizer_step_attempts"] = int(trainer.ema.updates)
        _require(self.progress["batches_completed"] <= (trainer.epoch + 1) * 3155, "Repeated/extra resumed training batches.")

    def after_epoch(self, trainer: Any) -> None:
        _require(self.progress["batches_completed"] == (trainer.epoch + 1) * 3155, "Incomplete resumed epoch cannot be saved.")

    def after_save(self, trainer: Any) -> None:
        self.after_epoch(trainer)
        _require(Path(trainer.last).resolve() == self.state.checkpoint_path, "Saved checkpoint escaped the canonical run.")
        completed = int(trainer.epoch) + 1
        _require(self.restored and completed == self.durable["epochs_completed"] + 1,
                 "Checkpoint save skipped or repeated a complete epoch.")
        self.progress.update(epochs_completed=completed, optimizer_step_attempts=int(trainer.ema.updates),
                             optimizer_state_present=bool(trainer.optimizer.state_dict()["state"]),
                             scaler_state_present=bool(trainer.scaler.state_dict()))
        rows = _read_rows(self.state.run / "results.csv")
        tool._validate_training_evidence(rows, self.progress, "train", self.state.config)
        _require(rows[:len(self.state.rows)] == self.state.rows, "Historical epoch results were rewritten.")
        self.stopper_state = _recover_stopper(rows)
        _require((trainer.stopper.best_epoch, trainer.stopper.best_fitness, trainer.stopper.patience) ==
                 (self.stopper_state["best_epoch"], self.stopper_state["best_fitness"], 20), "Runtime/recovered EarlyStopping differs.")
        started = tool._read(self.state.attempt / "STARTED.json")
        record = {"schema_version": "experiment2.epoch_state.v2", "status": "COMPLETE_EPOCH",
                  "experiment_id": self.state.config["experiment_id"], "mode": "train", "completed_epoch": completed,
                  **self.progress, **self.stopper_state, "last_sha256": tool.sha256_file(self.state.checkpoint_path),
                  "last_bytes": self.state.checkpoint_path.stat().st_size,
                  "results_sha256": tool.sha256_file(self.state.run / "results.csv"),
                  "git_commit": started["git_commit"], "source_tree_sha256": started["source_tree_sha256"],
                  "config_sha256": FROZEN_CONFIG_SHA256, "internal_test_files_accessed": False,
                  "recorded_utc": tool.shared.utc_now()}
        checkpoint = _load_checkpoint(self.state.checkpoint_path)
        _validate_checkpoint(replace(self.state, checkpoint=checkpoint, epoch_state=record, rows=rows, migration=False))
        tool.shared.verify_file_sha256(self.state.checkpoint_path, record["last_sha256"], "just-saved complete checkpoint")
        path = self.state.attempt / "epochs" / f"epoch_{completed:03d}.json"
        _write_new_json(path, record)
        _publish_epoch_state(self.state.run / "epoch_state.json", {
            **record, "epoch_record": path.relative_to(self.state.run).as_posix(),
            "epoch_record_sha256": tool.sha256_file(path)})
        self.durable = dict(self.progress)

    def attach(self, model: Any) -> None:
        for event, callback in (("on_pretrain_routine_start", self.before_dataset),
                ("on_pretrain_routine_end", self.restore), ("on_train_epoch_start", self.before_epoch),
                ("on_train_batch_start", self.before_batch), ("on_train_batch_end", self.after_batch),
                ("on_train_epoch_end", self.after_epoch), ("on_model_save", self.after_save)):
            model.add_callback(event, callback)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@contextmanager
def _resume_framework(state: ResumeState) -> Iterator[None]:
    """Retain original args.yaml while native resume records its args in the new attempt."""
    from ultralytics.utils import YAML
    save = YAML.save

    def save_args(file: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(file).resolve() == state.run / "args.yaml":
            file = state.attempt / "args.yaml"
        return save(file, data, *args, **kwargs)

    with tool.offline_framework(state.root, state.config, "train"), patch.object(YAML, "save", side_effect=save_args):
        yield


def _start_attempt(state: ResumeState, git: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    # Caller holds the OS lock; exclusive numbering still prevents any attempt overwrite.
    state.attempt.parent.mkdir(exist_ok=True)
    state.attempt.mkdir()
    (state.attempt / "epochs").mkdir()
    progress = _initial_progress(state)
    discarded = {"epoch": progress["epochs_completed"] + 1}
    for output, key in (("batches", "batches_completed"), ("optimizer_step_attempts", "optimizer_step_attempts"),
                        ("optimizer_updates", "optimizer_updates")):
        observed = state.observed_progress.get(key)
        _require(observed is None or type(observed) is int and observed >= progress[key], "Partial progress went backwards.")
        discarded[output] = None if observed is None else observed - progress[key]
    record = {"schema_version": "experiment2.resume.v2", "status": "STARTED",
        "resume_utc": tool.shared.utc_now(), "git_commit": git["commit"],
        "source_tree_sha256": git["source_state"]["source_tree_sha256"], "config_sha256": FROZEN_CONFIG_SHA256,
        "checkpoint_path": str(state.checkpoint_path), "checkpoint_sha256": state.epoch_state["last_sha256"],
        "checkpoint_completed_epoch": progress["epochs_completed"], "resumed_start_epoch": progress["epochs_completed"] + 1,
        "target_epoch": 100, "epoch60_migration": state.migration,
        "dataset_identities": {"train_val_yaml": state.config["train_val_yaml_sha256"],
                               **state.config["dataset_metadata_sha256"]},
        "pretrained_lineage": state.identity, "original_segment": _original_segment(),
        "prior_training_segments": state.segments, "history_sha256": state.history_sha256,
        "input_artifacts": state.input_artifacts,
        "durable_progress_at_start": progress, "environment": environment,
        "discarded_partial_epoch": discarded,
        "internal_test_files_accessed": False}
    _write_new_json(state.attempt / "STARTED.json", record)
    return record


def _snapshot_original(state: ResumeState) -> None:
    snapshot = state.snapshot
    snapshot.mkdir()
    for relative, digest in {**state.input_artifacts, "weights/last.pt": state.epoch_state["last_sha256"]}.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(state.run / relative, target)
        tool.shared.verify_file_sha256(target, digest, "original segment snapshot")


def _finish(state: ResumeState, callbacks: _ResumeCallbacks, started: dict[str, Any]) -> None:
    _require(not any((state.attempt / f"{status}.json").exists() for status in ("INTERRUPTED", "FAILED", "COMPLETED")),
             "A terminal attempt cannot be completed again.")
    rows = _read_rows(state.run / "results.csv")
    tool._validate_training_evidence(rows, callbacks.progress, "train", state.config)
    completed = callbacks.progress["epochs_completed"]
    first_epoch = len(state.rows) + 1
    _require(callbacks.restored and first_epoch <= completed <= 100
             and callbacks.progress == callbacks.durable
             and callbacks.progress["batches_completed"] == completed * 3155
             and (completed == 100 or callbacks.stopper_state["patience_exhausted"] is True),
             "Partial or unexplained early return cannot complete the full run.")
    _require(tool._read(state.attempt / "STARTED.json") == started, "Resume start provenance was changed.")
    epoch_paths = sorted((state.attempt / "epochs").iterdir())
    _require([p.name for p in epoch_paths] == [f"epoch_{epoch:03d}.json" for epoch in range(first_epoch, completed + 1)],
             "Persisted resume epoch coverage is incomplete or unexpected.")
    previous = _initial_progress(state)
    epoch_artifacts = {}
    for epoch, path in enumerate(epoch_paths, start=first_epoch):
        evidence = tool._read(path)
        _require(evidence.get("epochs_completed") == epoch
                 and evidence.get("batches_completed") == epoch * 3155
                 and evidence.get("config_sha256") == FROZEN_CONFIG_SHA256
                 and evidence.get("internal_test_files_accessed") is False,
                 "Persisted resume epoch identity changed.")
        tool._validate_training_evidence(rows[:epoch], evidence, "train", state.config)
        for counter in ("optimizer_updates", "optimizer_step_attempts"):
            _require(evidence[counter] >= previous[counter], "Persisted optimizer counters went backwards.")
        previous = evidence
        epoch_artifacts[path.name] = tool.sha256_file(path)
    for key, value in {**callbacks.durable, **callbacks.stopper_state}.items():
        _require(_same_value(key, previous.get(key), value), f"Final persisted epoch state differs: {key}.")
    pointer = tool._read(state.run / "epoch_state.json")
    _require(pointer == {**previous, "epoch_record": epoch_paths[-1].relative_to(state.run).as_posix(),
                         "epoch_record_sha256": epoch_artifacts[epoch_paths[-1].name]}, "Final epoch commit pointer differs.")
    _require(rows[:len(state.rows)] == state.rows, "Historical results were rewritten.")
    _require((state.run / "results.csv").read_bytes().startswith(
        (state.snapshot / "results.csv").read_bytes()), "Historical results bytes were rewritten.")
    for relative, digest in ORIGINAL_ARTIFACTS.items():
        if relative not in {"results.csv", "epoch_state.json"}:
            tool.shared.verify_file_sha256(state.run / relative, digest, "preserved original run evidence")
    for relative, digest in state.history_sha256.items():
        tool.shared.verify_file_sha256(tool._path(state.root, RUN_RELATIVE / relative), digest, "preserved resume history")
    for name in ("best.pt", "last.pt"):
        tool._verify_completed_checkpoint(state.run / "weights" / name, state.config)
    artifacts = {rel: tool.sha256_file(state.run / rel) for rel in ("weights/last.pt", "weights/best.pt", "results.csv")}
    record = {**started, "status": "COMPLETED", "mode": "train", **callbacks.durable,
              "early_stopping": callbacks.stopper_state, "completed_utc": tool.shared.utc_now(),
              "epoch_record_sha256": epoch_artifacts,
              "artifacts": artifacts, "resulting_last_checkpoint_sha256": artifacts["weights/last.pt"],
              "training_segments": [*state.segments, {
                  "first_epoch": first_epoch, "last_completed_epoch": completed, "git_commit": started["git_commit"],
                  "source_tree_sha256": started["source_tree_sha256"], "config_sha256": FROZEN_CONFIG_SHA256,
                  "status": "COMPLETED", "internal_test_files_accessed": False}]}
    _write_new_json(state.attempt / "COMPLETED.json", record)
    _write_new_json(state.run / "completion.json", record)


@contextmanager
def _process_lock(root: Path) -> Iterator[None]:
    """OS-held lock survives Python errors and is released by process death; never unlink it."""
    run = tool._path(root, RUN_RELATIVE)
    _require(run.is_dir(), "Canonical interrupted run is missing.")
    with tool._path(root, RUN_RELATIVE / ".resume.lock").open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise Error("Another controlled resume process holds the run lock.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def resume() -> None:
    """USER operation: canonical checkpoint and latest fully committed epoch only."""
    root = tool.PROJECT_ROOT
    config = tool.load_config(root)
    environment = tool.verify_environment(root, config)
    _verify_framework()
    tool.verify_baseline_args(root, config)
    git = _verify_resume_git(root, config)
    identity = tool.verify_identity(root, config)
    dataset = tool.verify_dataset(root, config, hashes=True)
    with _process_lock(root):
        state = _inspect_run(root, config, identity, dataset)
        started = _start_attempt(state, git, environment)
        callbacks = _ResumeCallbacks(state)
        try:
            _snapshot_original(state)
            from ultralytics import YOLO
            with _resume_framework(state):
                # Explicit existing path prevents native latest-run fallback.
                tool.shared.verify_file_sha256(state.checkpoint_path, state.epoch_state["last_sha256"], "approved resume checkpoint")
                model = YOLO(str(state.checkpoint_path), task="detect")
                callbacks.attach(model)
                model.train(resume=str(state.checkpoint_path))
                _finish(state, callbacks, started)
        except BaseException as exc:
            if not (state.attempt / "COMPLETED.json").exists():
                status = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
                record = {**started, "status": status, "reason": str(exc), "recorded_utc": tool.shared.utc_now(),
                          "observed_progress": callbacks.progress, "durable_progress": callbacks.durable,
                          "resulting_last_checkpoint_sha256": (tool.sha256_file(state.checkpoint_path)
                                                                if state.checkpoint_path.is_file() else None)}
                _write_new_json(state.attempt / f"{status}.json", record)
            raise
        finally:
            callbacks.close()
