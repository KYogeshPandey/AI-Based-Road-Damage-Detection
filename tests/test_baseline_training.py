"""Focused tests for frozen YOLOv8s baseline launch and resume continuity."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.training import train_baseline as baseline_launcher  # noqa: E402
from road_damage.training.baseline_support import (  # noqa: E402
    BaselinePaths,
    BaselineTrainingError,
    build_fresh_ultralytics_args,
    build_resume_ultralytics_args,
    ensure_fresh_run_available,
    load_baseline_config,
    resolve_baseline_paths,
    validate_resume_checkpoint,
    verify_file_sha256,
)
from road_damage.training.early_stopping_state import (  # noqa: E402
    PersistentEarlyStopping,
    STATE_SCHEMA,
    state_path_for_run,
)
from road_damage.training.inspect_checkpoint import inspect_checkpoint  # noqa: E402
from road_damage.training.source_state import (  # noqa: E402
    build_source_state_manifest,
    verify_source_state_manifest,
    write_source_state_manifest,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "baseline_public_v1_yolov8s.yaml"
_DEFAULT = object()


LITERAL_FROZEN_PARAMETERS = {
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
LITERAL_CLASS_NAMES = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}


def _temporary_paths(root: Path, config: dict) -> BaselinePaths:
    output_base = root / "outputs" / "training" / "baseline_public_v1"
    return BaselinePaths(
        config=CONFIG_PATH,
        model=root / "models" / "yolov8s.pt",
        source_dataset=root / "source",
        data_yaml=root / "train_val.yaml",
        output_base=output_base,
        run_dir=output_base / config["run_name"],
        amp_check_artifact=root / "weights" / "yolo26n.pt",
        source_state_manifest=root / "source_state_manifest.json",
    )


def _write_checkpoint(
    path: Path,
    config: dict,
    paths: BaselinePaths,
    *,
    epoch: int = 19,
    optimizer: object = _DEFAULT,
    scaler: object = _DEFAULT,
    override_args: dict | None = None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    train_args = build_fresh_ultralytics_args(config, paths)
    if override_args:
        train_args.update(override_args)
    torch.save(
        {
            "epoch": epoch,
            "optimizer": {"state": {}} if optimizer is _DEFAULT else optimizer,
            "scaler": {"scale": 65536.0} if scaler is _DEFAULT else scaler,
            "ema": {"synthetic": True},
            "model": None,
            "train_args": train_args,
        },
        path,
    )


def _write_state(
    checkpoint: Path,
    paths: BaselinePaths,
    *,
    checkpoint_epoch: int = 19,
    best_epoch: int = 5,
    best_fitness: float = 0.8,
    last_completed_epoch: int | None = None,
) -> dict:
    completed = checkpoint_epoch + 1 if last_completed_epoch is None else last_completed_epoch
    state = {
        "schema_version": STATE_SCHEMA,
        "experiment_id": "baseline_public_v1_yolov8s",
        "run_directory": str(paths.run_dir.resolve()),
        "config_path": str(paths.config.resolve()),
        "config_sha256": hashlib.sha256(paths.config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_stored_epoch": checkpoint_epoch,
        "patience": 20,
        "best_epoch": best_epoch,
        "best_fitness": best_fitness,
        "last_completed_epoch": completed,
        "saved_utc": "2026-09-07T00:00:00+00:00",
    }
    path = state_path_for_run(paths.run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    return state


def _write_resumable_fixture(
    config: dict,
    paths: BaselinePaths,
    *,
    epoch: int = 19,
    best_epoch: int = 5,
    optimizer: object = _DEFAULT,
    scaler: object = _DEFAULT,
    override_args: dict | None = None,
) -> tuple[Path, dict]:
    checkpoint = paths.run_dir / "weights" / "last.pt"
    _write_checkpoint(
        checkpoint,
        config,
        paths,
        epoch=epoch,
        optimizer=optimizer,
        scaler=scaler,
        override_args=override_args,
    )
    state = _write_state(
        checkpoint, paths, checkpoint_epoch=epoch, best_epoch=best_epoch
    )
    return checkpoint, state


class FakeStopper:
    def __init__(self, patience: int = 20) -> None:
        self.patience = patience
        self.best_epoch = 0
        self.best_fitness = 0.0
        self.possible_stop = False

    def __call__(self, epoch: int, fitness: float) -> bool:
        if fitness > self.best_fitness or self.best_fitness == 0:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= self.patience - 1
        return delta >= self.patience


class FakeYOLO:
    def __init__(self, model: str, task: str, paths: BaselinePaths, resume: bool) -> None:
        self.model_argument = model
        self.task = task
        self.paths = paths
        self.resume = resume
        self.callbacks: dict[str, list] = {}
        self.train_arguments: dict | None = None
        self.trainer: SimpleNamespace | None = None

    def add_callback(self, event: str, callback) -> None:
        self.callbacks.setdefault(event, []).append(callback)

    def train(self, **arguments) -> None:
        self.train_arguments = arguments
        trainer = SimpleNamespace(
            save_dir=self.paths.run_dir,
            args=SimpleNamespace(patience=20),
            stopper=FakeStopper(20),
            resume=self.resume,
            start_epoch=20 if self.resume else 0,
        )
        self.trainer = trainer
        for callback in self.callbacks.get("on_pretrain_routine_end", []):
            callback(trainer)


def _launcher_patches(paths: BaselinePaths):
    return (
        patch.object(baseline_launcher, "resolve_baseline_paths", return_value=paths),
        patch.object(baseline_launcher, "validate_dataset_inputs"),
        patch.object(baseline_launcher, "verify_file_sha256", return_value="hash"),
        patch.object(baseline_launcher, "collect_environment", return_value={"packages": []}),
        patch.object(baseline_launcher, "collect_dataset_fingerprints", return_value={}),
        patch.object(baseline_launcher, "verify_source_state_manifest", return_value={}),
        patch.object(
            baseline_launcher,
            "collect_git_state",
            return_value={"repository_root": str(PROJECT_ROOT), "commit": "a" * 40, "branch": "main", "clean": True},
        ),
    )


class BaselineConfigTests(unittest.TestCase):
    def test_all_scientific_parameters_match_independent_literal_oracle(self) -> None:
        config = load_baseline_config(CONFIG_PATH)
        observed = {key: config[key] for key in LITERAL_FROZEN_PARAMETERS}
        self.assertEqual(observed, LITERAL_FROZEN_PARAMETERS)
        self.assertEqual(config["hsv_s"], 0.40)
        self.assertEqual(config["hsv_v"], 0.30)
        self.assertNotIn("class_weights", config)
        self.assertNotIn("oversample", config)

    def test_approved_environment_matches_independent_literal_oracle(self) -> None:
        config = load_baseline_config(CONFIG_PATH)
        self.assertEqual(
            config["expected_environment"],
            {
                "python": "3.12.10",
                "ultralytics": "8.4.130",
                "torch": "2.13.0+cu126",
                "torchvision": "0.28.0+cu126",
                "cuda": "12.6",
            },
        )

    def test_train_val_yaml_has_literal_classes_and_no_test_key(self) -> None:
        import yaml

        config = load_baseline_config(CONFIG_PATH)
        paths = resolve_baseline_paths(config)
        value = yaml.safe_load(paths.data_yaml.read_text(encoding="utf-8"))
        self.assertEqual(set(value), {"train", "val", "names"})
        self.assertNotIn("test", value)
        self.assertEqual(value["names"], LITERAL_CLASS_NAMES)

    def test_output_resolves_directly_under_project_training_directory(self) -> None:
        config = load_baseline_config(CONFIG_PATH)
        paths = resolve_baseline_paths(config)
        expected = (
            PROJECT_ROOT / "outputs" / "training" / "baseline_public_v1" / config["run_name"]
        ).resolve()
        arguments = build_fresh_ultralytics_args(config, paths)
        self.assertEqual(paths.run_dir, expected)
        self.assertEqual(Path(arguments["save_dir"]), expected)
        self.assertEqual(Path(arguments["project"]), expected.parent)
        self.assertNotIn("runs\\detect", str(expected).casefold())


class FreshRunSafetyTests(unittest.TestCase):
    def test_existing_run_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "existing"
            run_dir.mkdir()
            marker = run_dir / "keep.txt"
            marker.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(BaselineTrainingError):
                ensure_fresh_run_available(run_dir)
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_pretrained_sha256_verification_accepts_exact_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"frozen checkpoint bytes")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(
                verify_file_sha256(checkpoint, expected.upper(), "pretrained checkpoint"), expected
            )
            with self.assertRaises(BaselineTrainingError):
                verify_file_sha256(checkpoint, "0" * 64, "pretrained checkpoint")

    def test_fresh_launcher_invokes_intended_training_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            models: list[FakeYOLO] = []

            def factory(model: str, task: str) -> FakeYOLO:
                instance = FakeYOLO(model, task, paths, resume=False)
                models.append(instance)
                return instance

            contexts = _launcher_patches(paths) + (
                patch.object(baseline_launcher, "record_fresh_provenance"),
            )
            with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6], contexts[7]:
                baseline_launcher.launch_baseline(CONFIG_PATH, yolo_factory=factory)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].model_argument, str(paths.model))
            self.assertEqual(models[0].task, "detect")
            self.assertEqual(models[0].train_arguments["epochs"], 100)
            self.assertEqual(models[0].train_arguments["patience"], 20)
            self.assertEqual(models[0].train_arguments["save_dir"], str(paths.run_dir))
            self.assertNotIn("resume", models[0].train_arguments)
            self.assertIn("on_model_save", models[0].callbacks)


class EarlyStoppingContinuityTests(unittest.TestCase):
    def test_state_is_saved_only_for_completed_checkpoint_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint = paths.run_dir / "weights" / "last.pt"
            _write_checkpoint(checkpoint, config, paths, epoch=19)
            trainer = SimpleNamespace(
                save_dir=paths.run_dir,
                args=SimpleNamespace(patience=20),
                last=checkpoint,
                epoch=19,
                stopper=SimpleNamespace(patience=20, best_epoch=5, best_fitness=0.8),
            )
            PersistentEarlyStopping(paths).save_after_completed_epoch(trainer)
            state = json.loads(state_path_for_run(paths.run_dir).read_text())
            self.assertEqual(state["checkpoint_stored_epoch"], 19)
            self.assertEqual(state["last_completed_epoch"], 20)
            self.assertEqual(state["best_epoch"], 5)
            self.assertEqual(state["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_state_survives_process_boundary_and_restores_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint, state = _write_resumable_fixture(config, paths, best_epoch=7)
            inspection = inspect_checkpoint(checkpoint)
            new_process_stopper = FakeStopper(20)
            trainer = SimpleNamespace(
                save_dir=paths.run_dir,
                args=SimpleNamespace(patience=20),
                stopper=new_process_stopper,
                resume=True,
                start_epoch=20,
            )
            PersistentEarlyStopping(paths, resume_state=state).restore_before_resumed_epoch(trainer)
            self.assertEqual(new_process_stopper.best_epoch, 7)
            self.assertEqual(new_process_stopper.best_fitness, 0.8)
            self.assertTrue(inspection["appears_resumable"])

    def test_session_boundary_does_not_restart_patience_clock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            _, state = _write_resumable_fixture(config, paths, best_epoch=5)
            restored = FakeStopper(20)
            trainer = SimpleNamespace(
                save_dir=paths.run_dir,
                args=SimpleNamespace(patience=20),
                stopper=restored,
                resume=True,
                start_epoch=20,
            )
            PersistentEarlyStopping(paths, resume_state=state).restore_before_resumed_epoch(trainer)
            self.assertTrue(restored(25, 0.7))
            incorrectly_reset = FakeStopper(20)
            for epoch in range(21, 26):
                reset_would_stop = incorrectly_reset(epoch, 0.7)
            self.assertFalse(reset_would_stop)


class CheckpointAndLauncherResumeTests(unittest.TestCase):
    def test_amp_scaler_present_allows_resume_when_other_state_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.assertIs(raw_config["amp"], True)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint, _ = _write_resumable_fixture(config, paths)
            report = inspect_checkpoint(checkpoint)
            self.assertTrue(report["amp_scaler_state_required"])
            self.assertTrue(report["scaler_state_present"])
            self.assertTrue(report["amp_scaler_state_continuity_satisfied"])
            self.assertTrue(report["appears_resumable"])
            validate_resume_checkpoint(report, config, paths)

    def test_amp_scaler_missing_is_not_resumable_and_validation_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.assertIs(raw_config["amp"], True)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint, _ = _write_resumable_fixture(config, paths, scaler=None)
            report = inspect_checkpoint(checkpoint)
            self.assertTrue(report["amp_scaler_state_required"])
            self.assertFalse(report["scaler_state_present"])
            self.assertFalse(report["amp_scaler_state_continuity_satisfied"])
            self.assertFalse(report["checkpoint_state_resumable"])
            self.assertFalse(report["appears_resumable"])
            with self.assertRaisesRegex(
                BaselineTrainingError,
                "AMP scaler state is missing; complete training-state continuity cannot be guaranteed",
            ):
                validate_resume_checkpoint(report, config, paths)

    def test_inspector_reports_state_and_remains_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint, _ = _write_resumable_fixture(config, paths)
            state_path = state_path_for_run(paths.run_dir)
            checkpoint_before = (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns)
            state_before = (state_path.read_bytes(), state_path.stat().st_mtime_ns)
            report = inspect_checkpoint(checkpoint)
            self.assertEqual(report["stored_epoch"], 19)
            self.assertEqual(report["completed_epoch_number"], 20)
            self.assertEqual(report["early_stopping_best_epoch"], 5)
            self.assertEqual(report["early_stopping_best_fitness"], 0.8)
            self.assertEqual(report["early_stopping_patience"], 20)
            self.assertTrue(report["state_checkpoint_epoch_consistent"])
            self.assertTrue(report["state_config_identity_consistent"])
            self.assertTrue(report["state_run_identity_consistent"])
            self.assertTrue(report["appears_resumable"])
            self.assertEqual(checkpoint_before, (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns))
            self.assertEqual(state_before, (state_path.read_bytes(), state_path.stat().st_mtime_ns))

    def test_missing_optimizer_or_stale_state_is_not_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint, _ = _write_resumable_fixture(config, paths, optimizer=None)
            report = inspect_checkpoint(checkpoint)
            self.assertFalse(report["optimizer_state_present"])
            self.assertFalse(report["appears_resumable"])
            _write_checkpoint(checkpoint, config, paths, epoch=20)
            stale = inspect_checkpoint(checkpoint)
            self.assertFalse(stale["state_checkpoint_epoch_consistent"])
            self.assertFalse(stale["appears_resumable"])

    def test_patience_exhausted_state_is_not_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint, _ = _write_resumable_fixture(
                config, paths, epoch=24, best_epoch=5
            )
            report = inspect_checkpoint(checkpoint)
            self.assertTrue(report["early_stopping"]["patience_exhausted"])
            self.assertFalse(report["appears_resumable"])
            with self.assertRaises(BaselineTrainingError):
                validate_resume_checkpoint(report, config, paths)

    def test_checkpoint_from_another_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            expected_paths = _temporary_paths(root, config)
            other_paths = BaselinePaths(
                **{
                    **expected_paths.__dict__,
                    "run_dir": expected_paths.output_base / "another_run",
                }
            )
            checkpoint, _ = _write_resumable_fixture(config, other_paths)
            report = inspect_checkpoint(checkpoint)
            self.assertTrue(report["appears_resumable"])
            with self.assertRaises(BaselineTrainingError):
                validate_resume_checkpoint(report, config, expected_paths)

    def test_resume_launcher_uses_genuine_resume_and_restores_stopper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint, _ = _write_resumable_fixture(config, paths, best_epoch=6)
            models: list[FakeYOLO] = []

            def factory(model: str, task: str) -> FakeYOLO:
                instance = FakeYOLO(model, task, paths, resume=True)
                models.append(instance)
                return instance

            contexts = _launcher_patches(paths) + (
                patch.object(baseline_launcher, "verify_recorded_fingerprints"),
                patch.object(baseline_launcher, "verify_recorded_git_state"),
                patch.object(baseline_launcher, "record_resume_attempt"),
            )
            with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6], contexts[7], contexts[8], contexts[9]:
                baseline_launcher.launch_baseline(CONFIG_PATH, resume=checkpoint, yolo_factory=factory)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].model_argument, str(checkpoint.resolve()))
            self.assertEqual(
                models[0].train_arguments,
                build_resume_ultralytics_args(checkpoint),
            )
            self.assertEqual(models[0].trainer.stopper.best_epoch, 6)
            self.assertEqual(models[0].trainer.stopper.best_fitness, 0.8)

    def test_resume_validation_rejects_scientific_parameter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_baseline_config(CONFIG_PATH)
            paths = _temporary_paths(root, config)
            checkpoint, _ = _write_resumable_fixture(
                config, paths, override_args={"hsv_s": 0.7}
            )
            report = inspect_checkpoint(checkpoint)
            self.assertTrue(report["appears_resumable"])
            with self.assertRaises(BaselineTrainingError):
                validate_resume_checkpoint(report, config, paths)


class SourceStateManifestTests(unittest.TestCase):
    def test_source_manifest_is_deterministic_and_excludes_generated_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("src", "configs", "tests", "road-damage-project-docs"):
                (root / directory).mkdir()
                (root / directory / f"{directory.replace('-', '_')}.txt").write_text(
                    directory, encoding="utf-8"
                )
            for filename in ("AGENTS.md", "requirements.txt", ".gitignore"):
                (root / filename).write_text(filename, encoding="utf-8")
            cache = root / "src" / "__pycache__"
            cache.mkdir()
            (cache / "ignored.pyc").write_bytes(b"generated")
            first = build_source_state_manifest(root)
            second = build_source_state_manifest(root)
            self.assertEqual(first, second)
            self.assertEqual(
                [item["path"] for item in first["files"]],
                sorted(item["path"] for item in first["files"]),
            )
            self.assertNotIn("src/__pycache__/ignored.pyc", {item["path"] for item in first["files"]})

    def test_source_manifest_verification_detects_stale_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("src", "configs", "tests", "road-damage-project-docs"):
                (root / directory).mkdir()
                (root / directory / "content.txt").write_text(directory, encoding="utf-8")
            for filename in ("AGENTS.md", "requirements.txt", ".gitignore"):
                (root / filename).write_text(filename, encoding="utf-8")
            manifest_path = root / "manifest.json"
            written = write_source_state_manifest(manifest_path, root)
            self.assertEqual(verify_source_state_manifest(manifest_path, root), written)
            (root / "src" / "content.txt").write_text("changed", encoding="utf-8")
            with self.assertRaises(BaselineTrainingError):
                verify_source_state_manifest(manifest_path, root)


class GitIgnorePolicyTests(unittest.TestCase):
    def test_required_gitignore_rules_match_independent_literal_oracle(self) -> None:
        required_rules = {
            "/data/",
            "/outputs/",
            "/runs/",
            "/models/",
            "/weights/",
            "/.venv*/",
            "*.pt",
            "*.zip",
            "*.7z",
            "*.rar",
            "*.tar",
            "*.tar.gz",
            "*.tgz",
            "*.gz",
            "*.bz2",
            "*.xz",
            "*.avi",
            "*.mp4",
            "*.mkv",
            "*.mov",
            "*.mpeg",
            "*.mpg",
            "*.wmv",
            "*.webm",
            "*.m4v",
        }
        observed_rules = {
            line.strip()
            for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue(
            required_rules.issubset(observed_rules),
            f"Missing required Git ignore rules: {sorted(required_rules - observed_rules)}",
        )

    def test_effective_gitignore_policy_ignores_assets_but_not_source(self) -> None:
        ignored_probes = (
            "data/probe.bin",
            "outputs/probe.json",
            "runs/probe.csv",
            "models/probe.bin",
            "weights/probe.bin",
            ".venv/Scripts/python.exe",
            ".venv_broken_backup/probe.cfg",
            "probe.pt",
            "probe.zip",
            "probe.7z",
            "probe.rar",
            "probe.tar",
            "probe.tar.gz",
            "probe.tgz",
            "probe.gz",
            "probe.bz2",
            "probe.xz",
            "probe.avi",
            "probe.mp4",
            "probe.mkv",
            "probe.mov",
            "probe.mpeg",
            "probe.mpg",
            "probe.wmv",
            "probe.webm",
            "probe.m4v",
        )
        trackable_probes = (
            "src/probe.py",
            "configs/probe.yaml",
            "tests/probe.py",
            "road-damage-project-docs/probe.md",
            "reproducibility/probe.json",
            "AGENTS.md",
            "requirements.txt",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                ["git", "init", "--quiet", str(root)],
                check=True,
                capture_output=True,
                text=True,
            )
            (root / ".gitignore").write_bytes((PROJECT_ROOT / ".gitignore").read_bytes())

            def is_ignored(probe: str) -> bool:
                result = subprocess.run(
                    ["git", "-C", str(root), "check-ignore", "--no-index", "--quiet", "--", probe],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertIn(result.returncode, (0, 1), result.stderr)
                return result.returncode == 0

            for probe in ignored_probes:
                self.assertTrue(is_ignored(probe), probe)
            for probe in trackable_probes:
                self.assertFalse(is_ignored(probe), probe)


if __name__ == "__main__":
    unittest.main()
