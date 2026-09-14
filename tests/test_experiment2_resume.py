"""Synthetic epoch-33 resume tests; never run a real model or open real test data."""

from contextlib import ExitStack, nullcontext
import copy
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import yaml
from road_damage.training import experiment2 as tool
from road_damage.training import experiment2_resume as resume_tool


ARCHITECTURE = {"backbone": [], "head": [], "scales": {"s": [.5, .5, 1024]},
                "scale": "s", "end2end": True, "reg_max": 1}


class Detect(torch.nn.Module):
    nc = 4
    reg_max = 1
    end2end = True


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameters_for_fixture = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor([1.])) for _ in range(366)])
        self.model = torch.nn.ModuleList([Detect()])
        self.yaml = copy.deepcopy(ARCHITECTURE)
        self.names = {0: "D00_longitudinal_crack", 1: "D10_transverse_crack",
                      2: "D20_alligator_crack", 3: "D40_pothole"}
        self.end2end = True

    def init_criterion(self):
        return SimpleNamespace(updates=0, update=lambda: None)


def optimizer_for(model):
    params = list(model.parameters_for_fixture)
    groups = [{"params": params[:126], "param_group": "weight", "weight_decay": .0005},
              {"params": params[126:240], "param_group": "bn", "weight_decay": 0.},
              {"params": params[240:], "param_group": "bias", "weight_decay": 0.}]
    return torch.optim.SGD(groups, lr=.01, momentum=.9369855256207079, nesterov=True)


class PinAndInterfaceTests(unittest.TestCase):
    def test_approval_identity_and_config_are_independent_literals(self):
        self.assertEqual(resume_tool.ORIGINAL_COMMIT, "39070294088a725edf9092d861e5da4d750acd25")
        self.assertEqual(resume_tool.ORIGINAL_SOURCE_SHA256,
                         "a33facf065a4004fd32f673df716c05b1f5783dbd5899481ef99c957ac1b2b57")
        self.assertEqual(resume_tool.CHECKPOINT_SHA256,
                         "12eb8c95a0bc9a34943836d8362d1058e86c8ff56db19e0ff906267451bb5809")
        self.assertEqual(resume_tool.CHECKPOINT_BYTES, 40374049)
        self.assertEqual(resume_tool.MIGRATION_CHECKPOINT_SHA256,
                         "335a4b825d1dd30f80caf68d844e12f023ad3a5e873bb9dd1d851ec9a946e5e4")
        self.assertEqual(resume_tool.MIGRATION_CHECKPOINT_BYTES, 40377697)
        self.assertEqual(resume_tool.MIGRATION_COMMIT, "0312871fb275725699913e08baaacab570c7296f")
        self.assertEqual(resume_tool.MIGRATION_SOURCE_SHA256,
                         "7740bc7d2c1516969f081885ea4fe317ef3c9237cb5ce5612b01e4474e44d28a")
        self.assertEqual(tool.sha256_file(ROOT / tool.CONFIG),
                         "0ae18ab9fa2d1098eefed24b2b75f5ed6f94b9e2bce773396d6402339cdaafe9")
        self.assertEqual(set(resume_tool.RESUME_FILES), {
            "src/road_damage/training/experiment2.py", "src/road_damage/training/experiment2_resume.py",
            "tests/test_experiment2.py", "tests/test_experiment2_resume.py",
            "road-damage-project-docs/EXPERIMENT2_YOLO26S_RUNBOOK.md"})

    def test_installed_resume_sources_match_reviewed_behavior(self):
        resume_tool._verify_framework()

    def test_resume_cli_dispatch_and_all_overrides_refused(self):
        with patch.object(resume_tool, "resume") as operation, patch.object(tool, "launch") as fresh:
            self.assertEqual(tool.main(["--resume"]), 0)
            operation.assert_called_once_with()
            fresh.assert_not_called()
        for argv in (["--resume", "other.pt"], ["--resume", "--train"], ["--resume", "--smoke"],
                     *(["--resume", "--" + option, "x"] for option in
                       ("checkpoint", "model", "output", "epochs", "batch", "imgsz", "optimizer",
                        "lr0", "lrf", "data", "config", "run-dir", "augmentations", "force"))):
            with self.subTest(argv=argv), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                tool.main(argv)
        with self.assertRaises(TypeError):
            resume_tool.resume(checkpoint="other.pt")


class ResumeFixtureTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = json.loads((ROOT / tool.CONFIG).read_bytes())
        config_path = self.root / tool.CONFIG
        config_path.parent.mkdir(parents=True)
        config_path.write_bytes((ROOT / tool.CONFIG).read_bytes())
        self.run = self.root / resume_tool.RUN_RELATIVE
        (self.run / "weights").mkdir(parents=True)
        self.path = self.run / "weights/last.pt"
        args = tool.training_arguments(self.root, self.config, "train", self.run / "train_val.yaml")
        self.saved_args = {**args, "device": "0"}
        (self.run / "args.yaml").write_text(yaml.safe_dump(self.saved_args), encoding="utf-8")
        self.data = {"train": str(self.root / self.config["source_dataset"] / "images/train"),
                     "val": str(self.root / self.config["source_dataset"] / "images/val"),
                     "names": dict(tool.shared.CLASS_NAMES)}
        for kind in ("images", "labels"):
            for split in ("train", "val"):
                (self.root / self.config["source_dataset"] / kind / split).mkdir(parents=True)
        (self.run / "train_val.yaml").write_text(yaml.safe_dump(self.data), encoding="utf-8")
        fields = ["epoch", "time", "train/box_loss", "train/cls_loss", "train/l1_loss",
                  "val/box_loss", "val/cls_loss", "val/l1_loss", "metrics/mAP50-95(B)"]
        self.rows = [{k: str(epoch) if k == "epoch" else "0.1" for k in fields} for epoch in range(1, 34)]
        self.rows[-1]["metrics/mAP50-95(B)"] = "0.18374"
        self.write_rows(self.rows)
        model = TinyDetector()
        optimizer = optimizer_for(model)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()  # isolated tiny-tensor fixture, not a training model
        optimizer_state = optimizer.state_dict()
        for group in optimizer_state["param_groups"]:
            group.update(lr=.007702342635146034, initial_lr=.01)
        for values in optimizer_state["state"].values():
            values["momentum_buffer"] = values["momentum_buffer"].half()
        self.checkpoint = {"epoch": 32, "version": "8.4.130", "model": None, "ema": model.half(),
            "updates": 7713, "best_fitness": .18374, "optimizer": optimizer_state,
            "scaler": {"scale": 512., "growth_factor": 2., "backoff_factor": .5,
                       "growth_interval": 2000, "_growth_tracker": 1645},
            "train_args": self.saved_args,
            "train_results": {key: [float(row[key]) for row in self.rows] for key in fields}}
        torch.save(self.checkpoint, self.path)
        self.checkpoint_sha = tool.sha256_file(self.path)
        self.identity = {"sha256": resume_tool.PRETRAINED_SHA256, "path": self.config["model"]["path"]}
        self.manifest = {"mode": "train", "status": "STARTED", "config": self.config,
            "git_commit": resume_tool.ORIGINAL_COMMIT, "source_tree_sha256": resume_tool.ORIGINAL_SOURCE_SHA256,
            "config_sha256": resume_tool.FROZEN_CONFIG_SHA256, "model_sha256": self.identity["sha256"],
            "resolved_args": args, "internal_test_files_accessed": False,
            "preflight": {"internal_test_files_accessed": False, "checks": {"pretrained_identity": self.identity}}}
        self.epoch_state = {"experiment_id": self.config["experiment_id"], "mode": "train", "completed_epoch": 33,
            "best_epoch": 33, "best_fitness": .18374, "patience": 20, "last_sha256": self.checkpoint_sha,
            "config_sha256": resume_tool.FROZEN_CONFIG_SHA256, "optimizer_step_attempts": 7713,
            "optimizer_updates": 7703, "optimizer_state_present": True, "scaler_state_present": True}
        self.failure = {"status": "FAILED_TECHNICAL", "mode": "train", "epochs_completed": 33,
            "batches_completed": 104150, "optimizer_step_attempts": 7715, "optimizer_updates": 7705,
            "recorded_utc": "2026-09-13T20:14:50.367608+00:00"}
        for name, value in (("run_manifest.json", self.manifest), ("epoch_state.json", self.epoch_state),
                            ("failure.json", self.failure)):
            self.write_json(name, value)
        self.hashes = {name: tool.sha256_file(self.run / name) for name in resume_tool.ORIGINAL_ARTIFACTS}
        self.dataset_report = {"counts": {"train": 12620, "val": 2602}, "train_val_bytes_verified": True,
            "metadata_sha256": self.config["dataset_metadata_sha256"], "internal_test_files_accessed": False}
        for name, value in (("CHECKPOINT_SHA256", self.checkpoint_sha), ("CHECKPOINT_BYTES", self.path.stat().st_size),
                            ("ORIGINAL_ARTIFACTS", self.hashes)):
            self.stack.enter_context(patch.object(resume_tool, name, value))
        self.stack.enter_context(patch.object(tool, "architecture", return_value=copy.deepcopy(ARCHITECTURE)))

    def write_json(self, name, record):
        path = self.run / name
        path.write_text(json.dumps(record), encoding="utf-8")
        if hasattr(self, "hashes"):
            self.hashes[name] = tool.sha256_file(path)

    def write_rows(self, rows):
        with (self.run / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]) if hasattr(self, "rows") else list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def inspect(self):
        return resume_tool._inspect_run(self.root, self.config, self.identity, self.dataset_report)

    def save_synthetic_checkpoint(self, trainer, rows):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint.update(epoch=trainer.epoch, updates=trainer.ema.updates,
                          best_fitness=trainer.stopper.best_fitness, scaler=trainer.scaler.state_dict(),
                          ema=copy.deepcopy(trainer.ema.ema).half(), optimizer=trainer.optimizer.state_dict(),
                          train_args={**self.saved_args, "model": str(self.path), "resume": str(self.path)},
                          train_results={key: [float(row[key]) for row in rows] for key in rows[0]})
        for group in checkpoint["optimizer"]["param_groups"]:
            group["lr"] = .01 * (((1 - math.cos(trainer.epoch * math.pi / 100)) / 2) * (.01 - 1) + 1)
        torch.save(checkpoint, self.path)

    def trainer(self, state):
        from ultralytics.engine.trainer import BaseTrainer
        from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA
        model = copy.deepcopy(state.checkpoint["ema"]).float()
        trainer = SimpleNamespace(model=model, args=SimpleNamespace(**{
            **self.saved_args, "model": str(self.path), "resume": str(self.path)}),
            save_dir=self.run, last=self.path, resume=True, start_epoch=0, epochs=100,
            optimizer=optimizer_for(model), scaler=torch.amp.GradScaler("cpu"), ema=ModelEMA(model),
            best_fitness=None, stopper=EarlyStopping(patience=20), device="cuda:0", amp=True,
            data=self.data, train_loader=MagicMock(), test_loader=SimpleNamespace(dataset=range(2602)))
        trainer.train_loader.__len__.return_value = 3155
        trainer.train_loader.dataset = range(12620)
        trainer._load_checkpoint_state = lambda checkpoint: BaseTrainer._load_checkpoint_state(trainer, checkpoint)
        BaseTrainer._setup_scheduler(trainer)
        BaseTrainer.resume_training(trainer, copy.deepcopy(state.checkpoint))
        # Exact assignment performed by installed _setup_train immediately before the callback.
        trainer.scheduler.last_epoch = trainer.start_epoch - 1
        return trainer

    def start(self, state):
        git = {"commit": "a" * 40, "source_state": {"git_commit": "a" * 40, "source_tree_sha256": "b" * 64}}
        record = resume_tool._start_attempt(state, git, {"fixture": True})
        resume_tool._snapshot_original(state)
        return record

    def test_canonical_checkpoint_is_accepted_read_only(self):
        before = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        state = self.inspect()
        self.assertEqual(state.checkpoint_path, self.path)
        self.assertEqual(state.checkpoint["epoch"], 32)
        self.assertEqual({str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}, before)
        self.assertFalse(state.attempt.exists())

    def test_missing_last_rejected_before_deserialization(self):
        self.path.unlink()
        with patch.object(resume_tool, "_load_checkpoint") as loader, self.assertRaises(tool.Error):
            self.inspect()
        loader.assert_not_called()

    def test_checkpoint_sha_and_size_mismatch_rejected(self):
        for name, value in (("CHECKPOINT_SHA256", "0" * 64), ("CHECKPOINT_BYTES", 1)):
            with self.subTest(field=name), patch.object(resume_tool, name, value), \
                 patch.object(resume_tool, "_load_checkpoint") as loader, self.assertRaises(tool.Error):
                self.inspect()
            loader.assert_not_called()

    def test_completed_run_and_previous_attempt_are_refused(self):
        for name in ("completion.json", "resume_attempts"):
            path = self.run / name
            path.write_text("{}")
            with self.subTest(name=name), self.assertRaises(tool.Error):
                self.inspect()
            path.unlink()

    def test_stripped_wrong_epoch_missing_optimizer_scaler_ema_rejected(self):
        for key, value in (("epoch", -1), ("epoch", 33), ("epoch", 31), ("optimizer", None),
                           ("optimizer", {"state": {}, "param_groups": []}), ("scaler", None),
                           ("ema", None), ("updates", None), ("updates", 7715), ("best_fitness", .1)):
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint[key] = value
            with self.subTest(key=key, value=str(value)[:40]), \
                 patch.object(resume_tool, "_load_checkpoint", return_value=checkpoint), self.assertRaises(tool.Error):
                self.inspect()

    def test_missing_momentum_wrong_groups_scaler_and_embedded_results_rejected(self):
        for mutation in ("momentum", "groups", "lr", "scaler", "results"):
            checkpoint = copy.deepcopy(self.checkpoint)
            if mutation == "momentum":
                checkpoint["optimizer"]["state"][0].pop("momentum_buffer")
            elif mutation == "groups":
                checkpoint["optimizer"]["param_groups"].pop()
            elif mutation == "lr":
                checkpoint["optimizer"]["param_groups"][0]["lr"] = .01
            elif mutation == "scaler":
                checkpoint["scaler"]["scale"] = 65536.
            else:
                checkpoint["train_results"]["epoch"] = [1.]
            with self.subTest(mutation=mutation), patch.object(resume_tool, "_load_checkpoint", return_value=checkpoint), \
                 self.assertRaises(tool.Error):
                self.inspect()

    def test_wrong_architecture_class_count_names_and_head_rejected(self):
        for mutation in ("scale", "nc", "names", "end2end", "reg_max"):
            checkpoint = copy.deepcopy(self.checkpoint)
            if mutation == "scale":
                checkpoint["ema"].yaml["scale"] = "n"
            elif mutation == "names":
                checkpoint["ema"].names[0] = "wrong"
            else:
                setattr(checkpoint["ema"].model[-1], mutation, 80 if mutation == "nc" else False if mutation == "end2end" else 16)
            with self.subTest(mutation=mutation), patch.object(resume_tool, "_load_checkpoint", return_value=checkpoint), \
                 self.assertRaises(tool.Error):
                self.inspect()

    def test_scientific_and_path_argument_changes_rejected(self):
        for key, value in (("epochs", 67), ("epochs", 99), ("batch", 8), ("imgsz", 320), ("nbs", 32),
                           ("optimizer", "AdamW"), ("lr0", .1), ("lrf", .1), ("amp", False), ("fraction", 1),
                           ("seed", 43), ("mosaic", 0), ("data", "test.yaml"), ("model", "smoke/last.pt"),
                           ("save_dir", "elsewhere"), ("resume", True)):
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["train_args"][key] = value
            with self.subTest(key=key), patch.object(resume_tool, "_load_checkpoint", return_value=checkpoint), \
                 self.assertRaises(tool.Error):
                self.inspect()

    def test_results_row_count_epoch34_and_nonfinite_rejected(self):
        for rows in (self.rows[:32], self.rows + [{**self.rows[-1], "epoch": "34"}],
                     self.rows[:32] + [{**self.rows[-1], "val/l1_loss": "nan"}]):
            self.write_rows(rows)
            with self.subTest(rows=len(rows)), self.assertRaises(tool.Error):
                self.inspect()

    def test_epoch_state_checkpoint_and_early_stopping_mismatch_rejected(self):
        for key, value in (("completed_epoch", 34), ("best_epoch", 0), ("best_fitness", .1), ("patience", 50),
                           ("last_sha256", "f" * 64), ("optimizer_updates", 7705), ("optimizer_step_attempts", 7715),
                           ("scaler_state_present", False), ("optimizer_state_present", False)):
            self.write_json("epoch_state.json", {**self.epoch_state, key: value})
            with self.subTest(key=key), self.assertRaises(tool.Error):
                self.inspect()

    def test_config_hash_dataset_identity_and_test_safety_rejected(self):
        path = self.root / tool.CONFIG
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(tool.Error):
            self.inspect()
        path.write_bytes(original)
        for key, value in (("metadata_sha256", {}), ("counts", {"train": 1, "val": 1}),
                           ("train_val_bytes_verified", False), ("internal_test_files_accessed", True),
                           ("internal_test_files_accessed", None), ("internal_test_files_accessed", 0)):
            dataset = {**self.dataset_report, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(tool.Error):
                resume_tool._inspect_run(self.root, self.config, self.identity, dataset)

    def test_original_provenance_and_explicit_nonaccess_required(self):
        for key, value in (("git_commit", "e" * 40), ("source_tree_sha256", "e" * 64), ("config_sha256", "e" * 64),
                           ("model_sha256", "e" * 64), ("internal_test_files_accessed", None),
                           ("internal_test_files_accessed", True), ("internal_test_files_accessed", 0)):
            self.write_json("run_manifest.json", {**self.manifest, key: value})
            with self.subTest(key=key, value=value), self.assertRaises(tool.Error):
                self.inspect()
        missing = dict(self.manifest)
        missing.pop("internal_test_files_accessed")
        self.write_json("run_manifest.json", missing)
        with self.assertRaises(tool.Error):
            self.inspect()

    def test_missing_interruption_or_unsealed_artifact_rejected(self):
        (self.run / "failure.json").unlink()
        with self.assertRaises(tool.Error):
            self.inspect()
        self.write_json("failure.json", {**self.failure, "status": "COMPLETED"})
        with self.assertRaises(tool.Error):
            self.inspect()

    def test_progress_uses_only_durable_epoch33_and_discards_partial_updates(self):
        progress = resume_tool._initial_progress(self.inspect())
        self.assertEqual((progress["epochs_completed"], progress["batches_completed"],
                          progress["optimizer_step_attempts"], progress["optimizer_updates"]), (33, 104115, 7713, 7703))
        self.assertNotEqual(progress["batches_completed"], self.failure["batches_completed"])

    def test_native_resume_restores_optimizer_scaler_ema_and_epoch34(self):
        state = self.inspect()
        trainer = self.trainer(state)
        self.assertEqual(trainer.start_epoch, 33)
        self.assertEqual(trainer.start_epoch + 1, 34)
        resume_tool._assert_state_equal(trainer.optimizer.state_dict(), state.checkpoint["optimizer"], "optimizer")
        self.assertEqual(trainer.scaler.state_dict(), state.checkpoint["scaler"])
        self.assertEqual(trainer.scaler.get_scale(), 512.)
        self.assertEqual(trainer.ema.updates, 7713)
        resume_tool._assert_state_equal(trainer.ema.ema.state_dict(), state.checkpoint["ema"].state_dict(), "EMA")
        self.assertEqual(len(trainer.optimizer.state), 366)

    def test_native_check_resume_selects_explicit_last_without_latest_run_fallback(self):
        from ultralytics.engine.trainer import BaseTrainer
        trainer = SimpleNamespace(args=SimpleNamespace(resume=str(self.path), data="NOT_USED"))
        with patch("ultralytics.engine.trainer.load_checkpoint", return_value=(
                SimpleNamespace(args=copy.deepcopy(self.saved_args)), {})) as loader, \
             patch("ultralytics.engine.trainer.get_latest_run") as latest:
            BaseTrainer.check_resume(trainer, {"resume": str(self.path)})
        loader.assert_called_once_with(self.path)
        latest.assert_not_called()
        self.assertIs(trainer.resume, True)
        self.assertEqual(trainer.args.model, str(self.path))
        self.assertEqual(trainer.args.resume, str(self.path))
        for key, value in self.config["training"].items():
            if key == "device":
                self.assertEqual(str(trainer.args.device), "0")
                self.assertEqual(str(value), "0")
            elif key != "resume":
                self.assertEqual(getattr(trainer.args, key), value, key)

    def test_callback_restores_early_stopping_before_first_resumed_epoch(self):
        state = self.inspect()
        trainer = self.trainer(state)
        self.assertEqual(trainer.stopper.best_epoch, 0)
        callbacks = resume_tool._ResumeCallbacks(state)
        self.addCleanup(callbacks.close)
        callbacks.restore(trainer)
        self.assertEqual((trainer.stopper.best_epoch, trainer.stopper.best_fitness, trainer.stopper.patience), (33, .18374, 20))
        self.assertFalse(trainer.stopper.possible_stop)
        self.assertFalse(trainer.stopper(52, .1))
        self.assertTrue(trainer.stopper(53, .1))

    def test_scheduler_continues_original_cosine_schedule_and_rejects_reset(self):
        state = self.inspect()
        trainer = self.trainer(state)
        callbacks = resume_tool._ResumeCallbacks(state)
        self.addCleanup(callbacks.close)
        callbacks.restore(trainer)
        trainer.epoch = 33
        callbacks.before_epoch(trainer)
        trainer.scheduler.step()
        callbacks.before_batch(trainer)
        expected = .01 * ((1 + math.cos(33 * math.pi / 100)) * .99 / 2 + .01)
        self.assertTrue(all(math.isclose(g["lr"], expected) for g in trainer.optimizer.param_groups))
        self.assertNotEqual(trainer.optimizer.param_groups[0]["lr"], .01)
        trainer.scheduler.last_epoch = 0
        with self.assertRaises(tool.Error):
            callbacks.before_batch(trainer)

    def test_reset_state_or_runtime_dataset_drift_rejected_before_batches(self):
        state = self.inspect()
        for change in ("optimizer", "scaler", "ema", "network", "updates", "start", "scheduler", "test", "arguments"):
            trainer = self.trainer(state)
            if change == "optimizer":
                trainer.optimizer.state.clear()
            elif change == "scaler":
                trainer.scaler.load_state_dict({**state.checkpoint["scaler"], "scale": 65536.})
            elif change in {"ema", "network"}:
                model = trainer.ema.ema if change == "ema" else trainer.model
                next(model.parameters()).data.fill_(9.)
            elif change == "updates":
                trainer.ema.updates = 0
            elif change == "start":
                trainer.start_epoch = 0
            elif change == "scheduler":
                trainer.scheduler.last_epoch = 0
            elif change == "test":
                trainer.data = {**trainer.data, "test": "DO_NOT_OPEN"}
            else:
                trainer.args.batch = 8
            with self.subTest(change=change), self.assertRaises(tool.Error):
                resume_tool._ResumeCallbacks(state).restore(trainer)

    def test_post_hook_continues_successful_update_count_and_is_removed(self):
        state = self.inspect()
        trainer = self.trainer(state)
        callbacks = resume_tool._ResumeCallbacks(state)
        self.addCleanup(callbacks.close)
        callbacks.restore(trainer)
        for p in trainer.model.parameters():
            p.grad = torch.ones_like(p)
        trainer.optimizer.step()
        self.assertEqual(callbacks.progress["optimizer_updates"], 7704)
        callbacks.close()
        trainer.optimizer.step()
        self.assertEqual(callbacks.progress["optimizer_updates"], 7704)

    def test_append_only_start_preserves_original_evidence_and_refuses_reuse(self):
        state = self.inspect()
        before = {name: (self.run / name).read_bytes() for name in self.hashes}
        started = self.start(state)
        self.assertEqual(started["original_segment"]["git_commit"], "39070294088a725edf9092d861e5da4d750acd25")
        self.assertEqual(started["git_commit"], "a" * 40)
        self.assertEqual(started["resumed_start_epoch"], 34)
        self.assertEqual(started["discarded_partial_epoch"]["batches"], 35)
        self.assertEqual(started["durable_progress_at_start"]["optimizer_updates"], 7703)
        for name, original in before.items():
            self.assertEqual((self.run / name).read_bytes(), original)
            self.assertEqual((state.attempt / "original_segment" / name).read_bytes(), original)
        self.assertEqual((state.attempt / "original_segment/weights/last.pt").read_bytes(), self.path.read_bytes())
        with self.assertRaises(FileExistsError):
            resume_tool._start_attempt(state, {}, {})
        with self.assertRaises(tool.Error):
            self.inspect()

    def test_native_args_record_redirect_preserves_original_args(self):
        from ultralytics.utils import YAML
        state = self.inspect()
        self.start(state)
        original = (self.run / "args.yaml").read_bytes()
        with patch.object(tool, "offline_framework", return_value=nullcontext()), resume_tool._resume_framework(state):
            YAML.save(self.run / "args.yaml", {"resume": str(self.path)})
        self.assertEqual((self.run / "args.yaml").read_bytes(), original)
        self.assertEqual(yaml.safe_load((state.attempt / "args.yaml").read_text()), {"resume": str(self.path)})

    def test_incomplete_epoch_nonfinite_loss_and_early_return_cannot_complete(self):
        state = self.inspect()
        started = self.start(state)
        callbacks = resume_tool._ResumeCallbacks(state)
        with self.assertRaises(tool.Error):
            callbacks.after_epoch(SimpleNamespace(epoch=33))
        with self.assertRaises(tool.Error):
            callbacks.after_batch(SimpleNamespace(loss=torch.tensor(float("nan")), epoch=33))
        with self.assertRaises(tool.Error):
            resume_tool._finish(state, callbacks, started)
        self.assertFalse((self.run / "completion.json").exists())
        self.assertFalse((state.attempt / "COMPLETED.json").exists())

    def simulate_completed_epoch_records(self):
        """Exercise save callbacks with synthetic counters/rows, without batch execution."""
        state = self.inspect()
        started = self.start(state)
        callbacks = resume_tool._ResumeCallbacks(state)
        self.addCleanup(callbacks.close)
        trainer = self.trainer(state)
        callbacks.restore(trainer)
        rows = list(self.rows)
        for completed in range(34, 54):
            trainer.epoch = completed - 1
            trainer.ema.updates += 200
            callbacks.progress["optimizer_updates"] += 200
            callbacks.progress["batches_completed"] = completed * 3155
            trainer.stopper(completed, .1)
            rows.append({**rows[-1], "epoch": str(completed)})
            self.write_rows(rows)
            self.save_synthetic_checkpoint(trainer, rows)
            callbacks.after_save(trainer)
        (self.run / "weights/best.pt").write_bytes(self.path.read_bytes())
        return state, callbacks, started

    def test_completion_appends_two_segments_and_hashes_all_epoch_records(self):
        original = {name: (self.run / name).read_bytes() for name in self.hashes if name not in {"results.csv", "epoch_state.json"}}
        state, callbacks, started = self.simulate_completed_epoch_records()
        with patch.object(tool, "_verify_completed_checkpoint") as verify:
            resume_tool._finish(state, callbacks, started)
        self.assertEqual(verify.call_count, 2)
        receipt = json.loads((self.run / "completion.json").read_text())
        self.assertEqual(receipt["status"], "COMPLETED")
        self.assertEqual(receipt["epochs_completed"], 53)
        self.assertEqual(receipt["batches_completed"], 53 * 3155)
        self.assertTrue(receipt["early_stopping"]["patience_exhausted"])
        segments = receipt["training_segments"]
        self.assertEqual([(s["first_epoch"], s["last_completed_epoch"]) for s in segments], [(1, 33), (34, 53)])
        self.assertEqual(segments[0]["git_commit"], "39070294088a725edf9092d861e5da4d750acd25")
        self.assertEqual(segments[0]["source_tree_sha256"],
                         "a33facf065a4004fd32f673df716c05b1f5783dbd5899481ef99c957ac1b2b57")
        self.assertEqual(segments[1]["git_commit"], "a" * 40)
        self.assertEqual(segments[1]["source_tree_sha256"], "b" * 64)
        self.assertEqual(len(receipt["epoch_record_sha256"]), 20)
        for name, digest in receipt["epoch_record_sha256"].items():
            self.assertEqual(tool.sha256_file(state.attempt / "epochs" / name), digest)
        self.assertIs(receipt["internal_test_files_accessed"], False)
        self.assertEqual((self.run / "completion.json").read_bytes(), (state.attempt / "COMPLETED.json").read_bytes())
        self.assertEqual({name: (self.run / name).read_bytes() for name in original}, original)
        self.assertEqual(json.loads((state.attempt / "STARTED.json").read_text()), started)

    def test_missing_or_altered_persisted_epoch_cannot_complete(self):
        state, callbacks, started = self.simulate_completed_epoch_records()
        path = state.attempt / "epochs/epoch_053.json"
        original = path.read_bytes()
        for mutation in ("missing", "counter", "safety", "patience"):
            path.write_bytes(original)
            evidence = json.loads(original)
            if mutation == "missing":
                path.unlink()
            else:
                key, value = {"counter": ("optimizer_updates", 1), "safety": ("internal_test_files_accessed", True),
                              "patience": ("best_epoch", 0)}[mutation]
                evidence[key] = value
                path.write_text(json.dumps(evidence))
            with self.subTest(mutation=mutation), patch.object(tool, "_verify_completed_checkpoint") as verify, \
                 self.assertRaises(tool.Error):
                resume_tool._finish(state, callbacks, started)
            verify.assert_not_called()
            self.assertFalse((self.run / "completion.json").exists())
            self.assertFalse((state.attempt / "COMPLETED.json").exists())

    def test_git_transition_refuses_original_commit_and_unrelated_changes(self):
        for commit, changed in ((resume_tool.ORIGINAL_COMMIT, ""), (resume_tool.MIGRATION_COMMIT, ""),
                                ("a" * 40, "M\tconfigs/training/experiment2_yolo26s_matched.yaml\n"),
                                ("a" * 40, "D\tsrc/road_damage/training/experiment2.py\n")):
            with self.subTest(commit=commit, changed=changed), \
                 patch.object(tool, "verify_git", return_value={"commit": commit}), \
                 patch.object(resume_tool.subprocess, "run", return_value=SimpleNamespace(stdout=changed)), self.assertRaises(tool.Error):
                resume_tool._verify_resume_git(self.root, self.config)
        with patch.object(tool, "verify_git", return_value={"commit": "a" * 40}), \
             patch.object(resume_tool.subprocess, "run", return_value=SimpleNamespace(stdout="A\tsrc/road_damage/training/experiment2_resume.py\n")):
            self.assertEqual(resume_tool._verify_resume_git(self.root, self.config)["commit"], "a" * 40)

    def test_public_resume_uses_only_native_checkpoint_argument_and_preserves_interruption(self):
        original = {name: (self.run / name).read_bytes() for name in self.hashes}
        model = MagicMock()
        def stop(**kwargs):
            self.assertEqual(kwargs, {"resume": str(self.path)})
            self.assertTrue((self.run / "resume_attempts/0001/STARTED.json").exists())
            raise KeyboardInterrupt()
        model.train.side_effect = stop
        git = {"commit": "a" * 40, "source_state": {"source_tree_sha256": "b" * 64}}
        with patch.object(tool, "PROJECT_ROOT", self.root), patch.object(tool, "load_config", return_value=self.config), \
             patch.object(tool, "verify_environment", return_value={}), patch.object(resume_tool, "_verify_framework"), \
             patch.object(tool, "verify_baseline_args"), patch.object(resume_tool, "_verify_resume_git", return_value=git), \
             patch.object(tool, "verify_identity", return_value=self.identity), \
             patch.object(tool, "verify_dataset", return_value=self.dataset_report) as verify_data, \
             patch.object(resume_tool, "_resume_framework", return_value=nullcontext()), \
             patch("ultralytics.YOLO", return_value=model) as yolo, self.assertRaises(KeyboardInterrupt):
            resume_tool.resume()
        yolo.assert_called_once_with(str(self.path), task="detect")
        verify_data.assert_called_once_with(self.root, self.config, hashes=True)
        self.assertEqual({name: (self.run / name).read_bytes() for name in self.hashes}, original)
        record = json.loads((self.run / "resume_attempts/0001/INTERRUPTED.json").read_text())
        self.assertIs(record["internal_test_files_accessed"], False)
        self.assertEqual(record["durable_progress"]["batches_completed"], 104115)
        self.assertFalse((self.run / "completion.json").exists())

    def test_technical_failure_appends_failed_without_rewriting_original_history(self):
        original = {name: (self.run / name).read_bytes() for name in self.hashes}
        git = {"commit": "a" * 40, "source_state": {"source_tree_sha256": "b" * 64}}
        with patch.object(tool, "PROJECT_ROOT", self.root), patch.object(tool, "load_config", return_value=self.config), \
             patch.object(tool, "verify_environment", return_value={}), patch.object(resume_tool, "_verify_framework"), \
             patch.object(tool, "verify_baseline_args"), patch.object(resume_tool, "_verify_resume_git", return_value=git), \
             patch.object(tool, "verify_identity", return_value=self.identity), \
             patch.object(tool, "verify_dataset", return_value=self.dataset_report), \
             patch.object(resume_tool, "_snapshot_original", side_effect=OSError("synthetic storage failure")), \
             patch("ultralytics.YOLO") as yolo, self.assertRaises(OSError):
            resume_tool.resume()
        yolo.assert_not_called()
        self.assertEqual({name: (self.run / name).read_bytes() for name in self.hashes}, original)
        record = json.loads((self.run / "resume_attempts/0001/FAILED.json").read_text())
        self.assertEqual(record["status"], "FAILED")
        self.assertIs(record["internal_test_files_accessed"], False)
        self.assertEqual(record["durable_progress"]["batches_completed"], 104115)
        self.assertEqual(record["resulting_last_checkpoint_sha256"], self.checkpoint_sha)
        self.assertFalse((self.run / "completion.json").exists())

    def test_fresh_train_refuses_existing_run_and_smoke_is_unchanged(self):
        with patch.object(tool, "PROJECT_ROOT", self.root), patch.object(tool, "load_config", return_value=self.config), \
             patch.object(tool, "preflight") as preflight, self.assertRaises(tool.Error):
            tool.launch("train", self.root)
        preflight.assert_not_called()
        smoke = tool.training_arguments(self.root, self.config, "smoke", self.root / "snapshot.yaml")
        self.assertEqual((smoke["epochs"], smoke["batch"], smoke["imgsz"], smoke["resume"]), (3, 4, 640, False))
        self.assertEqual(tool._smoke_workload(self.config), (3, 32, 96))

    def migration_fixture(self):
        """Create the reviewed legacy shape entirely in the temporary directory."""
        state = self.inspect()
        git = {"commit": "0312871fb275725699913e08baaacab570c7296f", "source_state": {
            "source_tree_sha256": "7740bc7d2c1516969f081885ea4fe317ef3c9237cb5ce5612b01e4474e44d28a"}}
        started = resume_tool._start_attempt(state, git, {})
        started["schema_version"] = "experiment2.resume_epoch33.v1"
        (state.attempt / "STARTED.json").write_text(json.dumps(started))
        resume_tool._snapshot_original(state)
        trainer = self.trainer(state)
        rows = list(self.rows)
        for epoch in range(34, 61):
            fitness = .21557 if epoch == 59 else .21535 if epoch == 60 else round(.19 + (epoch - 34) * .0009, 5)
            rows.append({**rows[-1], "epoch": str(epoch), "metrics/mAP50-95(B)": str(fitness)})
            attempts = 7713 + (epoch - 33) * 5325 // 27
            evidence = {**resume_tool._initial_progress(state), "epochs_completed": epoch,
                "batches_completed": epoch * 3155, "optimizer_step_attempts": attempts, "optimizer_updates": attempts - 12,
                **resume_tool._recover_stopper(rows), "config_sha256": resume_tool.FROZEN_CONFIG_SHA256,
                "last_sha256": "e" * 64, "internal_test_files_accessed": False}
            (state.attempt / "epochs" / f"epoch_{epoch:03d}.json").write_text(json.dumps(evidence))
        self.write_rows(rows)
        trainer.epoch = 59
        trainer.ema.updates = 13038
        trainer.stopper.best_epoch, trainer.stopper.best_fitness = 59, .21557
        trainer.scaler.load_state_dict({**trainer.scaler.state_dict(), "_growth_tracker": 864})
        self.save_synthetic_checkpoint(trainer, rows)
        digest = tool.sha256_file(self.path)
        evidence["last_sha256"] = digest
        (state.attempt / "epochs/epoch_060.json").write_text(json.dumps(evidence))
        progress = {key: evidence[key] for key in resume_tool._progress_keys()}
        interrupted = {**started, "status": "INTERRUPTED", "durable_progress": progress,
            "observed_progress": {**progress, "batches_completed": 189556, "optimizer_step_attempts": 13054,
                                  "optimizer_updates": 13042}, "resulting_last_checkpoint_sha256": digest}
        (state.attempt / "INTERRUPTED.json").write_text(json.dumps(interrupted))
        pins = {relative: tool.sha256_file(self.run / relative) for relative in resume_tool.MIGRATION_ARTIFACTS}
        self.stack.enter_context(patch.object(resume_tool, "MIGRATION_ARTIFACTS", pins))
        self.stack.enter_context(patch.object(resume_tool, "MIGRATION_CHECKPOINT_SHA256", digest))
        self.stack.enter_context(patch.object(resume_tool, "MIGRATION_CHECKPOINT_BYTES", self.path.stat().st_size))
        return self.inspect()

    def advance_synthetic_resume(self, state, *, final_epoch=62, fail_pointer=False, interrupted=True):
        started = self.start(state)
        trainer = self.trainer(state)
        callbacks = resume_tool._ResumeCallbacks(state)
        self.addCleanup(callbacks.close)
        callbacks.restore(trainer)
        rows = list(state.rows)
        for completed in range(len(rows) + 1, final_epoch + 1):
            trainer.epoch = completed - 1
            trainer.ema.updates += 200
            callbacks.progress["batches_completed"] = completed * 3155
            callbacks.progress["optimizer_updates"] += 200
            fitness = round(.22 + (completed - 61) * .001, 5)
            trainer.stopper(completed, fitness)
            rows.append({**rows[-1], "epoch": str(completed), "metrics/mAP50-95(B)": str(fitness)})
            self.write_rows(rows)
            self.save_synthetic_checkpoint(trainer, rows)
            if fail_pointer:
                with patch.object(resume_tool.os, "replace", side_effect=OSError("synthetic pointer failure")), self.assertRaises(OSError):
                    callbacks.after_save(trainer)
                return callbacks
            callbacks.after_save(trainer)
            pointer = json.loads((self.run / "epoch_state.json").read_text())
            self.assertEqual(pointer["completed_epoch"], completed)
            self.assertEqual(pointer["last_sha256"], tool.sha256_file(self.path))
            self.assertEqual(pointer["results_sha256"], tool.sha256_file(self.run / "results.csv"))
        interruption = {**started, "status": "INTERRUPTED", "durable_progress": callbacks.durable,
            "observed_progress": {**callbacks.progress, "batches_completed": callbacks.progress["batches_completed"] + 3},
            "resulting_last_checkpoint_sha256": tool.sha256_file(self.path)}
        if interrupted:
            resume_tool._write_new_json(state.attempt / "INTERRUPTED.json", interruption)
        return callbacks

    def test_epoch60_migration_is_read_only_and_recovers_epoch61_durable_progress(self):
        state = self.migration_fixture()
        self.assertTrue(state.migration)
        self.assertEqual(state.epoch_state["completed_epoch"], 60)
        self.assertEqual(state.checkpoint["epoch"], 59)
        self.assertEqual(state.attempt.name, "0002")
        self.assertEqual(json.loads((self.run / "epoch_state.json").read_text())["completed_epoch"], 33)
        progress = resume_tool._initial_progress(state)
        self.assertEqual([progress[key] for key in ("epochs_completed", "batches_completed", "optimizer_step_attempts", "optimizer_updates")],
                         [60, 189300, 13038, 13026])
        started = self.start(state)
        self.assertEqual(started["resumed_start_epoch"], 61)
        self.assertEqual(started["discarded_partial_epoch"], {
            "epoch": 61, "batches": 256, "optimizer_step_attempts": 16, "optimizer_updates": 16})

    def test_epoch61_native_scheduler_state_and_recovered_stopper(self):
        state = self.migration_fixture()
        trainer = self.trainer(state)
        callbacks = resume_tool._ResumeCallbacks(state)
        self.addCleanup(callbacks.close)
        callbacks.restore(trainer)
        self.assertEqual((trainer.start_epoch, trainer.epochs, trainer.scheduler.last_epoch), (60, 100, 59))
        self.assertEqual((trainer.stopper.best_epoch, trainer.stopper.best_fitness, trainer.stopper.patience), (59, .21557, 20))
        self.assertFalse(trainer.stopper.possible_stop)
        self.assertEqual(trainer.scaler.state_dict()["_growth_tracker"], 864)
        self.assertEqual(trainer.ema.updates, 13038)
        trainer.epoch = 60
        callbacks.before_epoch(trainer)
        trainer.scheduler.step()
        callbacks.before_batch(trainer)
        expected = .01 * ((1 + math.cos(60 * math.pi / 100)) * .99 / 2 + .01)
        self.assertTrue(all(math.isclose(g["lr"], expected) for g in trainer.optimizer.param_groups))

    def test_results_fitness_recovery_matches_native_metric_and_early_stopping(self):
        from ultralytics.utils.metrics import Metric
        from ultralytics.utils.torch_utils import EarlyStopping
        for scores in ([0., 0., .2, .2, .1], [.2, .3, .29], [0.] * 4):
            rows = [{"epoch": str(i), "metrics/mAP50-95(B)": str(score)} for i, score in enumerate(scores, 1)]
            native = EarlyStopping(20)
            for i, score in enumerate(scores, 1):
                with patch.object(Metric, "mean_results", return_value=[.9, .8, .7, score]):
                    self.assertEqual(Metric().fitness(), score)
                    native(i, Metric().fitness())
            recovered = resume_tool._recover_stopper(rows)
            self.assertEqual((recovered["best_epoch"], recovered["best_fitness"]), (native.best_epoch, native.best_fitness))
            self.assertEqual(recovered["epochs_without_improvement"], len(rows) - native.best_epoch)
        state = self.migration_fixture()
        self.assertEqual(resume_tool._recover_stopper(state.rows), {
            "best_epoch": 59, "best_fitness": .21557, "patience": 20, "epochs_without_improvement": 1, "patience_exhausted": False})

    def test_completed_epochs_publish_state_and_third_resume_needs_no_new_pin(self):
        state = self.migration_fixture()
        history_before = {str(p): p.read_bytes() for p in (self.run / "resume_attempts/0001").rglob("*") if p.is_file()}
        original_failure = (self.run / "failure.json").read_bytes()
        self.advance_synthetic_resume(state, final_epoch=62)
        third = self.inspect()
        self.assertFalse(third.migration)
        self.assertEqual(third.attempt.name, "0003")
        self.assertEqual(third.epoch_state["completed_epoch"], 62)
        self.assertEqual(self.trainer(third).start_epoch + 1, 63)
        self.assertEqual([(s["first_epoch"], s["last_completed_epoch"]) for s in third.segments], [(1, 33), (34, 60), (61, 62)])
        self.assertEqual(third.segments[1]["git_commit"], "0312871fb275725699913e08baaacab570c7296f")
        self.assertEqual(third.segments[2]["git_commit"], "a" * 40)
        self.assertEqual({str(p): p.read_bytes() for p in (self.run / "resume_attempts/0001").rglob("*") if p.is_file()}, history_before)
        self.assertEqual((self.run / "failure.json").read_bytes(), original_failure)
        # Continue yet again with the same code, still only synthetic checkpoint writes.
        self.advance_synthetic_resume(third, final_epoch=63)
        self.assertEqual(self.inspect().epoch_state["completed_epoch"], 63)

    def test_partial_epoch61_never_advances_state_or_accepts_a_checkpoint_save(self):
        state = self.migration_fixture()
        original = (self.run / "epoch_state.json").read_bytes()
        self.start(state)
        trainer = self.trainer(state)
        callbacks = resume_tool._ResumeCallbacks(state)
        self.addCleanup(callbacks.close)
        callbacks.restore(trainer)
        trainer.epoch, trainer.loss = 60, torch.tensor(.1)
        callbacks.after_batch(trainer)
        with self.assertRaises(tool.Error):
            callbacks.after_save(trainer)
        self.assertEqual((self.run / "epoch_state.json").read_bytes(), original)
        self.assertEqual(callbacks.durable["batches_completed"], 189300)
        self.assertFalse((state.attempt / "epochs/epoch_061.json").exists())

    def test_epoch_pointer_write_failure_does_not_advance_durable_state_or_authorize_resume(self):
        state = self.migration_fixture()
        before = (self.run / "epoch_state.json").read_bytes()
        callbacks = self.advance_synthetic_resume(state, final_epoch=61, fail_pointer=True)
        self.assertEqual((self.run / "epoch_state.json").read_bytes(), before)
        self.assertEqual(callbacks.durable["epochs_completed"], 60)
        self.assertTrue((state.attempt / "epochs/epoch_061.json").exists())
        with self.assertRaises(tool.Error):
            self.inspect()
        self.assertFalse((self.run / "completion.json").exists())

    def test_migration_rejects_unreviewed_sha_stale_state_and_results_disagreement(self):
        state = self.migration_fixture()
        for name, value in (("MIGRATION_CHECKPOINT_SHA256", "0" * 64), ("MIGRATION_CHECKPOINT_BYTES", 1)):
            with self.subTest(name=name), patch.object(resume_tool, name, value), self.assertRaises(tool.Error):
                self.inspect()
        original = (self.run / "results.csv").read_bytes()
        for rows in (state.rows[:-1], state.rows[:40] + state.rows[41:],
                     state.rows + [{**state.rows[-1], "epoch": "61"}]):
            self.write_rows(rows)
            with self.subTest(rows=len(rows)), self.assertRaises(tool.Error):
                self.inspect()
        (self.run / "results.csv").write_bytes(original)
        for key, value in (("epoch", 60), ("epoch", -1), ("optimizer", None), ("scaler", None), ("ema", None)):
            checkpoint = copy.deepcopy(state.checkpoint)
            checkpoint[key] = value
            with self.subTest(key=key), patch.object(resume_tool, "_load_checkpoint", return_value=checkpoint), self.assertRaises(tool.Error):
                self.inspect()

    def test_future_resume_rejects_pointer_sha_and_checkpoint_disagreement(self):
        state = self.migration_fixture()
        self.advance_synthetic_resume(state)
        pointer_path = self.run / "epoch_state.json"
        original = pointer_path.read_bytes()
        for key, value in (("completed_epoch", 61), ("last_sha256", "0" * 64), ("epoch_record_sha256", "0" * 64),
                           ("best_epoch", 33), ("optimizer_updates", 1), ("internal_test_files_accessed", True)):
            pointer = json.loads(original)
            pointer[key] = value
            pointer_path.write_text(json.dumps(pointer))
            with self.subTest(key=key), self.assertRaises(tool.Error):
                self.inspect()
        pointer_path.write_bytes(original)
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        with self.assertRaises(tool.Error):
            self.inspect()

    def test_run_lock_is_exclusive_and_released_after_exception(self):
        with resume_tool._process_lock(self.root):
            with self.assertRaises(tool.Error), resume_tool._process_lock(self.root):
                self.fail("Second lock must not be acquired")
        with resume_tool._process_lock(self.root):
            pass

    def test_metadata_and_history_redirects_fail_before_read_or_enumeration(self):
        original = tool._path
        def guard(root, relative):
            if Path(relative).name in {"epoch_state.json", "resume_attempts"}:
                raise tool.Error("synthetic path redirect")
            return original(root, relative)
        with patch.object(tool, "_path", side_effect=guard), patch.object(tool, "_read") as read, self.assertRaises(tool.Error):
            self.inspect()
        read.assert_not_called()
        with patch.object(tool, "_path", side_effect=guard), patch.object(Path, "iterdir") as enumerate_paths, self.assertRaises(tool.Error):
            resume_tool._read_history(self.root, self.run, self.config, self.identity, self.rows)
        enumerate_paths.assert_not_called()

    def test_completion_after_second_resume_preserves_three_training_segments(self):
        state = self.migration_fixture()
        callbacks = self.advance_synthetic_resume(state, final_epoch=100, interrupted=False)
        (self.run / "weights/best.pt").write_bytes(self.path.read_bytes())
        started = json.loads((state.attempt / "STARTED.json").read_text())
        with patch.object(tool, "_verify_completed_checkpoint"):
            resume_tool._finish(state, callbacks, started)
        receipt = json.loads((self.run / "completion.json").read_text())
        self.assertEqual([(s["first_epoch"], s["last_completed_epoch"]) for s in receipt["training_segments"]],
                         [(1, 33), (34, 60), (61, 100)])
        self.assertEqual(receipt["epochs_completed"], 100)
        self.assertEqual(len(receipt["epoch_record_sha256"]), 40)
        with self.assertRaises(tool.Error):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
