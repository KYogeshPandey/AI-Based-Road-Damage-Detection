"""Read-only inspection of an Ultralytics training checkpoint."""

from __future__ import annotations

import argparse
import json
import numbers
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_common import sha256_file  # noqa: E402
from road_damage.training.baseline_support import BaselineTrainingError  # noqa: E402
from road_damage.training.early_stopping_state import (  # noqa: E402
    inspect_early_stopping_state,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _train_args(checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    value = checkpoint.get("train_args")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if value is not None and hasattr(value, "__dict__"):
        return {str(key): _json_safe(item) for key, item in vars(value).items()}
    return None


def _stored_run_directory(
    train_args: Mapping[str, Any] | None,
) -> Path | None:
    if not train_args:
        return None
    save_dir = train_args.get("save_dir")
    if save_dir:
        return Path(str(save_dir)).resolve()
    project = train_args.get("project")
    name = train_args.get("name")
    if project and name:
        return (Path(str(project)) / str(name)).resolve()
    return None


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Load trusted checkpoint metadata on CPU and prove the file was not modified."""
    checkpoint_path = path.resolve()
    if not checkpoint_path.is_file():
        raise BaselineTrainingError(f"Checkpoint does not exist: {checkpoint_path}")
    before = checkpoint_path.stat()
    before_hash = sha256_file(checkpoint_path)
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise BaselineTrainingError(f"Could not inspect checkpoint: {checkpoint_path}") from exc
    if not isinstance(checkpoint, Mapping):
        raise BaselineTrainingError("Checkpoint root is not a metadata mapping.")
    train_args = _train_args(checkpoint)
    raw_epoch = checkpoint.get("epoch")
    stored_epoch = int(raw_epoch) if isinstance(raw_epoch, numbers.Integral) else None
    completed_epoch = stored_epoch + 1 if stored_epoch is not None and stored_epoch >= 0 else None
    raw_target = train_args.get("epochs") if train_args else None
    target_epochs = int(raw_target) if isinstance(raw_target, numbers.Integral) else None
    next_epoch = completed_epoch + 1 if completed_epoch is not None else None
    optimizer_present = checkpoint.get("optimizer") is not None
    scaler_present = checkpoint.get("scaler") is not None
    amp_scaler_state_required = True
    amp_scaler_state_continuity_satisfied = scaler_present
    model_state_present = checkpoint.get("ema") is not None or checkpoint.get("model") is not None
    actual_run_dir = checkpoint_path.parent.parent.resolve()
    stored_run_dir = _stored_run_directory(train_args)
    identity_matches = stored_run_dir is not None and stored_run_dir == actual_run_dir
    after = checkpoint_path.stat()
    after_hash = sha256_file(checkpoint_path)
    unchanged = (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before_hash == after_hash
    )
    checkpoint_resumable = (
        checkpoint_path.name.casefold() == "last.pt"
        and stored_epoch is not None
        and stored_epoch >= 0
        and completed_epoch is not None
        and target_epochs is not None
        and completed_epoch < target_epochs
        and optimizer_present
        and amp_scaler_state_continuity_satisfied
        and train_args is not None
        and model_state_present
        and identity_matches
        and unchanged
    )
    early_stopping = inspect_early_stopping_state(
        actual_run_dir,
        checkpoint_path,
        after_hash,
        stored_epoch,
        completed_epoch,
    )
    resumable = (
        checkpoint_resumable
        and early_stopping["consistent"]
        and early_stopping["patience_exhausted"] is False
    )
    return {
        "schema_version": "baseline_public_v1.checkpoint_inspection.v1",
        "checkpoint_path": str(checkpoint_path),
        "size_bytes": int(after.st_size),
        "sha256": after_hash,
        "stored_epoch": stored_epoch,
        "stored_epoch_is_zero_based": True,
        "completed_epoch_number": completed_epoch,
        "next_epoch_number": next_epoch,
        "target_total_epochs": target_epochs,
        "optimizer_state_present": optimizer_present,
        "scaler_state_present": scaler_present,
        "amp_scaler_state_required": amp_scaler_state_required,
        "amp_scaler_state_continuity_satisfied": amp_scaler_state_continuity_satisfied,
        "train_args_present": train_args is not None,
        "model_or_ema_state_present": model_state_present,
        "checkpoint_state_resumable": checkpoint_resumable,
        "appears_resumable": resumable,
        "read_only_unchanged": unchanged,
        "run_output_identity": {
            "actual_run_directory": str(actual_run_dir),
            "stored_run_directory": str(stored_run_dir) if stored_run_dir else None,
            "matches_checkpoint_location": identity_matches,
            "run_name": actual_run_dir.name,
        },
        "early_stopping_state_file": early_stopping["state_file"],
        "early_stopping_best_epoch": early_stopping["best_epoch"],
        "early_stopping_best_fitness": early_stopping["best_fitness"],
        "early_stopping_patience": early_stopping["patience"],
        "state_checkpoint_epoch_consistent": early_stopping[
            "state_checkpoint_epoch_consistent"
        ],
        "state_config_identity_consistent": early_stopping[
            "state_config_fingerprint_consistent"
        ],
        "state_run_identity_consistent": early_stopping[
            "state_run_identity_consistent"
        ],
        "early_stopping": early_stopping,
        "train_args": train_args,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect one trusted Ultralytics last.pt without modifying it."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = inspect_checkpoint(args.checkpoint)
    except BaselineTrainingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["appears_resumable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
