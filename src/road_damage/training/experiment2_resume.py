"""One explicitly approved epoch-33 resume; no checkpoint or scientific overrides.

Implements the Experiment 2 / RQ2 continuation boundary. Constants below are
approval identities, not an alternative training configuration. Scientific
settings still come only from the byte-pinned original YAML and checkpoint.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator, Mapping
from unittest.mock import patch

from road_damage.training import experiment2 as tool

Error = tool.Error
ORIGINAL_COMMIT = "39070294088a725edf9092d861e5da4d750acd25"
ORIGINAL_SOURCE_SHA256 = "a33facf065a4004fd32f673df716c05b1f5783dbd5899481ef99c957ac1b2b57"
FROZEN_CONFIG_SHA256 = "0ae18ab9fa2d1098eefed24b2b75f5ed6f94b9e2bce773396d6402339cdaafe9"
CHECKPOINT_SHA256 = "12eb8c95a0bc9a34943836d8362d1058e86c8ff56db19e0ff906267451bb5809"
CHECKPOINT_BYTES = 40_374_049
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

    @property
    def attempt(self) -> Path:
        return self.run / "resume_attempts" / "0001"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Error(message)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"epoch", "train/box_loss", "train/cls_loss", "train/l1_loss",
                "val/box_loss", "val/cls_loss", "val/l1_loss"}
    try:
        _require(bool(rows) and all(required <= row.keys() for row in rows), "Required epoch/loss results are missing.")
        _require(all(math.isfinite(float(v)) for row in rows for v in row.values()), "Non-finite epoch results.")
        _require([float(row["epoch"]) for row in rows] == list(range(1, len(rows) + 1)), "Results epochs are not contiguous.")
    except (TypeError, ValueError) as exc:
        raise Error("Invalid numerical epoch results.") from exc
    return rows


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
    _require(git["commit"] != ORIGINAL_COMMIT, "Commit the reviewed resume support before using --resume.")
    subprocess.run(["git", "merge-base", "--is-ancestor", ORIGINAL_COMMIT, "HEAD"], cwd=root,
                   check=True, capture_output=True)
    changed = subprocess.run(["git", "diff", "--no-renames", "--name-status", ORIGINAL_COMMIT, "HEAD", "--"],
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
    _require(type(c.get("epoch")) is int and c["epoch"] == 32, "Resume requires unstripped checkpoint epoch 32.")
    _require(c.get("version") == "8.4.130", "Checkpoint framework version changed.")
    _require(type(c.get("updates")) is int and c["updates"] == 7713, "Checkpoint EMA attempt counter changed.")
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
        _require(g.get("lr") == 0.007702342635146034 and g.get("initial_lr") == .01,
                 "Stored optimizer LR/schedule identity changed.")
    for values in optimizer["state"].values():
        buffer = values.get("momentum_buffer")
        _require(isinstance(buffer, torch.Tensor) and buffer.numel() > 0 and torch.isfinite(buffer).all().item(),
                 "A finite momentum buffer is required for every SGD parameter.")
    _require(c.get("scaler") == {"scale": 512.0, "growth_factor": 2.0, "backoff_factor": .5,
                                 "growth_interval": 2000, "_growth_tracker": 1645}, "AMP scaler state changed or missing.")
    _require(isinstance(c.get("train_args"), dict), "Checkpoint training args are missing.")
    _check_args(c["train_args"], state, resumed=False)
    embedded = c.get("train_results")
    _require(isinstance(embedded, dict) and set(embedded) == set(state.rows[0]), "Checkpoint results are missing.")
    for key, values in embedded.items():
        _require(values == [float(row[key]) for row in state.rows], f"Checkpoint/results.csv mismatch: {key}.")


def _inspect_run(root: Path, config: dict[str, Any], identity: dict[str, Any], dataset: dict[str, Any]) -> ResumeState:
    import yaml
    tool.shared.verify_file_sha256(tool._path(root, tool.CONFIG), FROZEN_CONFIG_SHA256, "frozen Experiment 2 config")
    run = tool._path(root, RUN_RELATIVE)
    _require(run == tool.run_directory(root, config, "train") and run.is_dir(), "Canonical Experiment 2 run is missing.")
    _require(not (run / "completion.json").exists(), "A full-run completion record already exists; resume refused.")
    _require(not (run / "resume_attempts").exists(), "This first-resume attempt was already reserved; stop for review.")
    checkpoint_path = tool._path(root, RUN_RELATIVE / "weights/last.pt")
    tool.shared.verify_file_sha256(checkpoint_path, CHECKPOINT_SHA256, "approved epoch-33 last.pt")
    _require(checkpoint_path.stat().st_size == CHECKPOINT_BYTES, "Approved checkpoint size changed.")
    for relative, digest in ORIGINAL_ARTIFACTS.items():
        tool.shared.verify_file_sha256(tool._path(root, RUN_RELATIVE / relative), digest, f"original {relative}")
    manifest = tool._read(run / "run_manifest.json")
    epoch_state = tool._read(run / "epoch_state.json")
    failure = tool._read(run / "failure.json")
    rows = _read_rows(run / "results.csv")
    _require(len(rows) == 33, "Exactly 33 completed results rows are required; epoch 34 must not exist.")
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
    expected_state = {"experiment_id": config["experiment_id"], "mode": "train", "completed_epoch": 33,
        "best_epoch": 33, "best_fitness": .18374, "patience": 20, "last_sha256": CHECKPOINT_SHA256,
        "config_sha256": FROZEN_CONFIG_SHA256, "optimizer_step_attempts": 7713, "optimizer_updates": 7703,
        "optimizer_state_present": True, "scaler_state_present": True}
    _require(all(_same_value(k, epoch_state.get(k), v) for k, v in expected_state.items()), "Epoch-33 durable state mismatch.")
    _require(failure.get("status") == "FAILED_TECHNICAL" and failure.get("mode") == "train"
             and failure.get("epochs_completed") == 33 and failure.get("batches_completed") == 104150
             and failure.get("optimizer_step_attempts") == 7715 and failure.get("optimizer_updates") == 7705
             and isinstance(failure.get("recorded_utc"), str), "Recorded original interruption is missing or changed.")
    data = tool.shared.validate_train_val_yaml(run / "train_val.yaml", tool._path(root, config["source_dataset"]))
    _require(set(data) == {"train", "val", "names"}, "Run YAML must be train/val only.")
    expected_args = tool.training_arguments(root, config, "train", run / "train_val.yaml")
    _require(manifest.get("resolved_args") == expected_args, "Original resolved training args changed.")
    saved_args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    state = ResumeState(root, run, checkpoint_path, config, _load_checkpoint(checkpoint_path),
                        manifest, epoch_state, saved_args, rows, data, identity)
    _validate_checkpoint(state)
    return state


def _initial_progress(state: ResumeState) -> dict[str, Any]:
    # Only durable epoch/checkpoint evidence is used; failure.json is never a counter source.
    completed = state.epoch_state["completed_epoch"]
    batches = completed * math.ceil(state.config["split_counts"]["train"] / state.config["training"]["batch"])
    _require((completed, batches) == (33, 104115), "Durable epoch/batch boundary changed.")
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


class _ResumeCallbacks:
    def __init__(self, state: ResumeState) -> None:
        self.state = state
        self.progress = _initial_progress(state)
        self.durable = dict(self.progress)
        self.handle: Any = None
        self.stopper_state = {"best_epoch": 33, "best_fitness": .18374, "patience": 20,
                              "epochs_without_improvement": 0, "patience_exhausted": False}
        self.restored = False

    def before_dataset(self, trainer: Any) -> None:
        _check_args(vars(trainer.args), self.state, resumed=True)
        _require(Path(trainer.save_dir).resolve() == self.state.run and trainer.resume is True,
                 "Native trainer did not select the canonical genuine resume.")

    def restore(self, trainer: Any) -> None:
        import torch
        self.before_dataset(trainer)
        _require(not self.restored and trainer.start_epoch == 33 and trainer.epochs == 100,
                 "Resume must start at zero-based epoch 33 within the original 100 epochs.")
        _require(trainer.scheduler.last_epoch == 32, "Scheduler position was not restored to epoch 32.")
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
        _require(trainer.ema.updates == 7713 and trainer.best_fitness == .18374, "EMA/fitness continuity changed.")
        for epoch in (0, 32, 33, 99, 100):
            expected = ((1 - math.cos(epoch * math.pi / 100)) / 2) * (.01 - 1) + 1
            _require(math.isclose(trainer.lf(epoch), expected, abs_tol=1e-12), "Cosine schedule no longer uses 100 epochs.")
        # Native EarlyStopping uses human epoch numbers, passed as epoch + 1.
        trainer.stopper.best_epoch = 33
        trainer.stopper.best_fitness = .18374
        trainer.stopper.patience = 20
        trainer.stopper.possible_stop = False
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
        self.progress.update(epochs_completed=completed, optimizer_step_attempts=int(trainer.ema.updates),
                             optimizer_state_present=bool(trainer.optimizer.state_dict()["state"]),
                             scaler_state_present=bool(trainer.scaler.state_dict()))
        rows = _read_rows(self.state.run / "results.csv")
        tool._validate_training_evidence(rows, self.progress, "train", self.state.config)
        _require(rows[:33] == self.state.rows, "Original epoch results were rewritten.")
        self.stopper_state = {"best_epoch": trainer.stopper.best_epoch, "best_fitness": trainer.stopper.best_fitness,
            "patience": trainer.stopper.patience, "epochs_without_improvement": completed - trainer.stopper.best_epoch,
            "patience_exhausted": completed - trainer.stopper.best_epoch >= trainer.stopper.patience}
        _require(self.stopper_state["patience"] == 20, "EarlyStopping patience changed.")
        record = {**self.progress, **self.stopper_state, "last_sha256": tool.sha256_file(self.state.checkpoint_path),
                  "config_sha256": FROZEN_CONFIG_SHA256, "internal_test_files_accessed": False,
                  "recorded_utc": tool.shared.utc_now()}
        _write_new_json(self.state.attempt / "epochs" / f"epoch_{completed:03d}.json", record)
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
    # Exclusive mkdir is also the cross-process gate. Even a crashed attempt is never silently reused.
    state.attempt.parent.mkdir()
    state.attempt.mkdir()
    (state.attempt / "epochs").mkdir()
    original = {"first_epoch": 1, "last_completed_epoch": 33, "git_commit": ORIGINAL_COMMIT,
                "source_tree_sha256": ORIGINAL_SOURCE_SHA256, "config_sha256": FROZEN_CONFIG_SHA256,
                "status": "INTERRUPTED", "internal_test_files_accessed": False,
                "artifacts": dict(ORIGINAL_ARTIFACTS)}
    record = {"schema_version": "experiment2.resume_epoch33.v1", "status": "STARTED",
        "resume_utc": tool.shared.utc_now(), "git_commit": git["commit"],
        "source_tree_sha256": git["source_state"]["source_tree_sha256"], "config_sha256": FROZEN_CONFIG_SHA256,
        "checkpoint_path": str(state.checkpoint_path), "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_completed_epoch": 33, "resumed_start_epoch": 34, "target_epoch": 100,
        "dataset_identities": {"train_val_yaml": state.config["train_val_yaml_sha256"],
                               **state.config["dataset_metadata_sha256"]},
        "pretrained_lineage": state.identity, "original_segment": original,
        "durable_progress_at_start": _initial_progress(state), "environment": environment,
        "discarded_partial_epoch": {"epoch": 34, "batches": 35, "optimizer_step_attempts": 2, "optimizer_updates": 2},
        "internal_test_files_accessed": False}
    _write_new_json(state.attempt / "STARTED.json", record)
    return record


def _snapshot_original(state: ResumeState) -> None:
    snapshot = state.attempt / "original_segment"
    snapshot.mkdir()
    for relative, digest in {**ORIGINAL_ARTIFACTS, "weights/last.pt": CHECKPOINT_SHA256}.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(state.run / relative, target)
        tool.shared.verify_file_sha256(target, digest, "original segment snapshot")


def _finish(state: ResumeState, callbacks: _ResumeCallbacks, started: dict[str, Any]) -> None:
    rows = _read_rows(state.run / "results.csv")
    tool._validate_training_evidence(rows, callbacks.progress, "train", state.config)
    completed = callbacks.progress["epochs_completed"]
    _require(callbacks.restored and 34 <= completed <= 100
             and callbacks.progress == callbacks.durable
             and callbacks.progress["batches_completed"] == completed * 3155
             and (completed == 100 or callbacks.stopper_state["patience_exhausted"] is True),
             "Partial or unexplained early return cannot complete the full run.")
    _require(tool._read(state.attempt / "STARTED.json") == started, "Resume start provenance was changed.")
    epoch_paths = sorted((state.attempt / "epochs").iterdir())
    _require([p.name for p in epoch_paths] == [f"epoch_{epoch:03d}.json" for epoch in range(34, completed + 1)],
             "Persisted resume epoch coverage is incomplete or unexpected.")
    previous = _initial_progress(state)
    epoch_artifacts = {}
    for epoch, path in enumerate(epoch_paths, start=34):
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
    _require(rows[:33] == state.rows, "Original results were rewritten.")
    _require((state.run / "results.csv").read_bytes().startswith(
        (state.attempt / "original_segment/results.csv").read_bytes()), "Original results bytes were rewritten.")
    for relative, digest in ORIGINAL_ARTIFACTS.items():
        if relative != "results.csv":
            tool.shared.verify_file_sha256(state.run / relative, digest, "preserved original run evidence")
    for name in ("best.pt", "last.pt"):
        tool._verify_completed_checkpoint(state.run / "weights" / name, state.config)
    artifacts = {rel: tool.sha256_file(state.run / rel) for rel in ("weights/last.pt", "weights/best.pt", "results.csv")}
    record = {**started, "status": "COMPLETED", "mode": "train", **callbacks.durable,
              "early_stopping": callbacks.stopper_state, "completed_utc": tool.shared.utc_now(),
              "epoch_record_sha256": epoch_artifacts,
              "artifacts": artifacts, "resulting_last_checkpoint_sha256": artifacts["weights/last.pt"],
              "training_segments": [started["original_segment"], {
                  "first_epoch": 34, "last_completed_epoch": completed, "git_commit": started["git_commit"],
                  "source_tree_sha256": started["source_tree_sha256"], "config_sha256": FROZEN_CONFIG_SHA256,
                  "status": "COMPLETED", "internal_test_files_accessed": False}]}
    _write_new_json(state.attempt / "COMPLETED.json", record)
    _write_new_json(state.run / "completion.json", record)


def resume() -> None:
    """USER operation: resume only the byte-pinned canonical epoch-33 interruption."""
    root = tool.PROJECT_ROOT
    config = tool.load_config(root)
    environment = tool.verify_environment(root, config)
    _verify_framework()
    tool.verify_baseline_args(root, config)
    git = _verify_resume_git(root, config)
    identity = tool.verify_identity(root, config)
    dataset = tool.verify_dataset(root, config, hashes=True)
    state = _inspect_run(root, config, identity, dataset)
    started = _start_attempt(state, git, environment)
    callbacks = _ResumeCallbacks(state)
    try:
        _snapshot_original(state)
        from ultralytics import YOLO
        with _resume_framework(state):
            # Recheck immediately before native model construction; a missing path cannot trigger latest-run fallback.
            tool.shared.verify_file_sha256(state.checkpoint_path, CHECKPOINT_SHA256, "approved resume checkpoint")
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
