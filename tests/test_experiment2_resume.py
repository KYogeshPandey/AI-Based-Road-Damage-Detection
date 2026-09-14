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
                  "val/box_loss", "val/cls_loss", "val/l1_loss"]
        self.rows = [{k: str(epoch) if k == "epoch" else "0.1" for k in fields} for epoch in range(1, 34)]
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
        if hasattr(self, "hashes"):
            self.hashes["results.csv"] = tool.sha256_file(self.run / "results.csv")

    def inspect(self):
        return resume_tool._inspect_run(self.root, self.config, self.identity, self.dataset_report)

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
            callbacks.after_save(trainer)
        (self.run / "weights/best.pt").write_bytes(self.path.read_bytes())
        return state, callbacks, started

    def test_completion_appends_two_segments_and_hashes_all_epoch_records(self):
        original = {name: (self.run / name).read_bytes() for name in self.hashes if name != "results.csv"}
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
        for commit, changed in ((resume_tool.ORIGINAL_COMMIT, ""), ("a" * 40, "M\tconfigs/training/experiment2_yolo26s_matched.yaml\n"),
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


if __name__ == "__main__":
    unittest.main()
