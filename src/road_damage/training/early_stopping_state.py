"""Persistent EarlyStopping state for one cross-process Ultralytics run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from road_damage.dataset.rdd2022_common import sha256_file, write_json_atomic
from road_damage.training.baseline_support import BaselineTrainingError, BaselinePaths, utc_now


STATE_SCHEMA = "baseline_public_v1.early_stopping_state.v1"
STATE_RELATIVE_PATH = Path("state") / "early_stopping_state.json"


def state_path_for_run(run_dir: Path) -> Path:
    """Return the single authoritative early-stopping state path."""
    return run_dir.resolve() / STATE_RELATIVE_PATH


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineTrainingError(f"Could not read early-stopping state: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineTrainingError(f"Early-stopping state must be a JSON object: {path}")
    return value


def inspect_early_stopping_state(
    run_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_stored_epoch: int | None,
    checkpoint_completed_epoch: int | None,
) -> dict[str, Any]:
    """Read state without mutation and report checkpoint/config/run consistency."""
    run_dir = run_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    path = state_path_for_run(run_dir)
    if not path.is_file():
        return {
            "state_file": str(path),
            "present": False,
            "patience": None,
            "best_epoch": None,
            "best_fitness": None,
            "last_completed_epoch": None,
            "epochs_without_improvement": None,
            "patience_exhausted": None,
            "state_checkpoint_epoch_consistent": False,
            "state_checkpoint_sha256_consistent": False,
            "state_run_identity_consistent": False,
            "state_config_fingerprint_consistent": False,
            "read_only_unchanged": True,
            "consistent": False,
            "errors": ["early-stopping state file is missing"],
            "state": None,
        }
    before = path.stat()
    before_hash = sha256_file(path)
    state = _read_state(path)
    errors: list[str] = []
    if state.get("schema_version") != STATE_SCHEMA:
        errors.append("state schema mismatch")
    patience = state.get("patience")
    best_epoch = state.get("best_epoch")
    best_fitness = state.get("best_fitness")
    last_completed = state.get("last_completed_epoch")
    stored_epoch = state.get("checkpoint_stored_epoch")
    if type(patience) is not int or patience != 20:
        errors.append("patience is not exactly 20")
    if type(best_epoch) is not int or best_epoch < 0:
        errors.append("best_epoch is invalid")
    if not isinstance(best_fitness, (int, float)) or not math.isfinite(float(best_fitness)):
        errors.append("best_fitness is invalid")
    if type(last_completed) is not int or last_completed < 1:
        errors.append("last_completed_epoch is invalid")
    if type(stored_epoch) is not int or stored_epoch < 0:
        errors.append("checkpoint_stored_epoch is invalid")
    if isinstance(best_epoch, int) and isinstance(last_completed, int) and best_epoch > last_completed:
        errors.append("best_epoch exceeds last_completed_epoch")
    epochs_without_improvement = (
        last_completed - best_epoch
        if isinstance(best_epoch, int) and isinstance(last_completed, int)
        else None
    )
    patience_exhausted = bool(
        isinstance(epochs_without_improvement, int)
        and isinstance(patience, int)
        and epochs_without_improvement >= patience
    )
    epoch_consistent = (
        stored_epoch == checkpoint_stored_epoch
        and last_completed == checkpoint_completed_epoch
        and isinstance(stored_epoch, int)
        and last_completed == stored_epoch + 1
    )
    if not epoch_consistent:
        errors.append("state/checkpoint epoch mismatch")
    sha_consistent = (
        state.get("checkpoint_path") == str(checkpoint_path)
        and str(state.get("checkpoint_sha256", "")).casefold()
        == checkpoint_sha256.casefold()
    )
    if not sha_consistent:
        errors.append("state/checkpoint identity or SHA-256 mismatch")
    run_consistent = (
        state.get("experiment_id") == "baseline_public_v1_yolov8s"
        and state.get("run_directory") == str(run_dir)
        and checkpoint_path == run_dir / "weights" / "last.pt"
    )
    if not run_consistent:
        errors.append("state/run identity mismatch")
    config_path_value = state.get("config_path")
    config_path = Path(str(config_path_value)).resolve() if config_path_value else None
    config_consistent = bool(
        config_path
        and config_path.is_file()
        and str(state.get("config_sha256", "")).casefold()
        == sha256_file(config_path).casefold()
    )
    if not config_consistent:
        errors.append("state/config fingerprint mismatch")
    after = path.stat()
    after_hash = sha256_file(path)
    unchanged = (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before_hash == after_hash
    )
    if not unchanged:
        errors.append("state file changed during read-only inspection")
    return {
        "state_file": str(path),
        "present": True,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_fitness": best_fitness,
        "last_completed_epoch": last_completed,
        "epochs_without_improvement": epochs_without_improvement,
        "patience_exhausted": patience_exhausted,
        "state_checkpoint_epoch_consistent": epoch_consistent,
        "state_checkpoint_sha256_consistent": sha_consistent,
        "state_run_identity_consistent": run_consistent,
        "state_config_fingerprint_consistent": config_consistent,
        "read_only_unchanged": unchanged,
        "consistent": not errors,
        "errors": errors,
        "state": state,
    }


class PersistentEarlyStopping:
    """Attach project-owned persistence callbacks without modifying Ultralytics."""

    def __init__(
        self,
        paths: BaselinePaths,
        *,
        resume_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.paths = paths
        self.resume_state = dict(resume_state) if resume_state is not None else None
        self.config_sha256 = sha256_file(paths.config)

    def attach(self, model: Any) -> None:
        """Register restore and post-checkpoint-save callbacks on a YOLO model."""
        model.add_callback("on_pretrain_routine_end", self.restore_before_resumed_epoch)
        model.add_callback("on_model_save", self.save_after_completed_epoch)

    def _verify_trainer_identity(self, trainer: Any) -> None:
        save_dir = Path(str(trainer.save_dir)).resolve()
        if save_dir != self.paths.run_dir.resolve():
            raise BaselineTrainingError(
                f"Ultralytics trainer output changed to {save_dir}; expected {self.paths.run_dir}."
            )
        if int(trainer.args.patience) != 20:
            raise BaselineTrainingError("Ultralytics trainer patience changed from 20.")

    def restore_before_resumed_epoch(self, trainer: Any) -> None:
        """Restore best epoch/fitness after trainer setup and before on_train_start."""
        self._verify_trainer_identity(trainer)
        stopper = getattr(trainer, "stopper", None)
        if stopper is None:
            raise BaselineTrainingError("Ultralytics trainer did not create EarlyStopping.")
        if self.resume_state is None:
            if int(stopper.patience) != 20:
                raise BaselineTrainingError("Fresh EarlyStopping patience is not 20.")
            return
        if not getattr(trainer, "resume", False):
            raise BaselineTrainingError("Early-stopping restore was requested on a non-resume trainer.")
        last_completed = int(self.resume_state["last_completed_epoch"])
        if int(trainer.start_epoch) != last_completed:
            raise BaselineTrainingError(
                "Trainer start_epoch does not follow the persisted last completed epoch."
            )
        if sha256_file(self.paths.config) != self.config_sha256:
            raise BaselineTrainingError("Baseline config changed during resume setup.")
        stopper.patience = 20
        stopper.best_epoch = int(self.resume_state["best_epoch"])
        stopper.best_fitness = float(self.resume_state["best_fitness"])
        stopper.possible_stop = (
            last_completed - stopper.best_epoch >= stopper.patience - 1
        )

    def save_after_completed_epoch(self, trainer: Any) -> None:
        """Atomically persist stopper state only after last.pt was fully written."""
        self._verify_trainer_identity(trainer)
        checkpoint_path = Path(str(trainer.last)).resolve()
        expected_checkpoint = self.paths.run_dir.resolve() / "weights" / "last.pt"
        if checkpoint_path != expected_checkpoint or not checkpoint_path.is_file():
            raise BaselineTrainingError(
                f"Authoritative last.pt is missing or belongs to another run: {checkpoint_path}"
            )
        if sha256_file(self.paths.config) != self.config_sha256:
            raise BaselineTrainingError("Baseline config changed during training.")
        stopper = getattr(trainer, "stopper", None)
        if stopper is None:
            raise BaselineTrainingError("Ultralytics trainer has no EarlyStopping state to save.")
        stored_epoch = int(trainer.epoch)
        state = {
            "schema_version": STATE_SCHEMA,
            "experiment_id": "baseline_public_v1_yolov8s",
            "run_directory": str(self.paths.run_dir.resolve()),
            "config_path": str(self.paths.config.resolve()),
            "config_sha256": self.config_sha256,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_stored_epoch": stored_epoch,
            "patience": int(stopper.patience),
            "best_epoch": int(stopper.best_epoch),
            "best_fitness": float(stopper.best_fitness),
            "last_completed_epoch": stored_epoch + 1,
            "saved_utc": utc_now(),
        }
        if state["patience"] != 20:
            raise BaselineTrainingError("Refusing to persist a patience value other than 20.")
        write_json_atomic(state_path_for_run(self.paths.run_dir), state)
