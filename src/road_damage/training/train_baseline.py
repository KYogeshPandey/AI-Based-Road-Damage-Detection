"""Launch the single frozen YOLOv8s public baseline, fresh or genuine resume."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.training.baseline_support import (  # noqa: E402
    BaselineTrainingError,
    build_fresh_ultralytics_args,
    build_resume_ultralytics_args,
    collect_dataset_fingerprints,
    collect_environment,
    collect_git_state,
    ensure_fresh_run_available,
    load_baseline_config,
    record_fresh_provenance,
    record_resume_attempt,
    reserve_fresh_run_directory,
    resolve_baseline_paths,
    validate_dataset_inputs,
    validate_resume_checkpoint,
    verify_file_sha256,
    verify_recorded_git_state,
    verify_recorded_fingerprints,
)
from road_damage.training.inspect_checkpoint import inspect_checkpoint  # noqa: E402
from road_damage.training.early_stopping_state import PersistentEarlyStopping  # noqa: E402
from road_damage.training.source_state import verify_source_state_manifest  # noqa: E402


LOGGER = logging.getLogger(__name__)


def launch_baseline(
    config_path: Path,
    resume: Path | None = None,
    *,
    yolo_factory: Callable[..., Any] | None = None,
) -> None:
    """Run a fresh baseline or resume only the configured run's exact last.pt."""
    config = load_baseline_config(config_path)
    paths = resolve_baseline_paths(config)
    if resume is None:
        ensure_fresh_run_available(paths.run_dir)
    elif not paths.run_dir.is_dir():
        raise BaselineTrainingError(f"Configured baseline run does not exist: {paths.run_dir}")

    validate_dataset_inputs(paths)
    verify_file_sha256(
        paths.amp_check_artifact,
        str(config["amp_check_artifact_sha256"]),
        "automatic Ultralytics AMP-check artifact",
    )
    environment = collect_environment(config)
    fingerprints = collect_dataset_fingerprints(paths)
    source_state = verify_source_state_manifest(paths.source_state_manifest)
    git_state = collect_git_state()

    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO

    if resume is not None:
        inspection = inspect_checkpoint(resume)
        validate_resume_checkpoint(inspection, config, paths)
        verify_recorded_fingerprints(paths, fingerprints)
        verify_recorded_git_state(paths, git_state)
        record_resume_attempt(paths, inspection, environment, git_state)
        LOGGER.info(
            "Resuming the same run from completed epoch %s: %s",
            inspection["completed_epoch_number"],
            resume.resolve(),
        )
        model = yolo_factory(str(resume.resolve()), task="detect")
        PersistentEarlyStopping(
            paths, resume_state=inspection["early_stopping"]["state"]
        ).attach(model)
        model.train(**build_resume_ultralytics_args(resume))
        return

    verify_file_sha256(paths.model, str(config["model_sha256"]), "pretrained YOLOv8s checkpoint")
    resolved_args = build_fresh_ultralytics_args(config, paths)
    reserve_fresh_run_directory(paths.run_dir)
    record_fresh_provenance(
        config, paths, resolved_args, environment, fingerprints, source_state, git_state
    )
    LOGGER.info("Starting one logical 100-epoch baseline run at %s", paths.run_dir)
    model = yolo_factory(str(paths.model), task="detect")
    PersistentEarlyStopping(paths).attach(model)
    train_args = dict(resolved_args)
    train_args.pop("model")
    model.train(**train_args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start or genuinely resume the frozen public YOLOv8s baseline."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Exact existing <configured-run>/weights/last.pt; omitted for fresh start.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        launch_baseline(args.config, args.resume)
    except BaselineTrainingError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning(
            "Training interruption received. Do not resume until weights/last.pt passes inspect_checkpoint.py."
        )
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
