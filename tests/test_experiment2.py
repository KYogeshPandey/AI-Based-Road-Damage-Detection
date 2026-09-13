"""Synthetic checks for the separate matched YOLO26s experiment; no real model execution."""

from contextlib import nullcontext
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_damage.training import experiment2 as tool


def config():
    return json.loads((ROOT / "configs/training/experiment2_yolo26s_matched.yaml").read_bytes())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def dataset_fixture(root):
    c = config()
    c["split_counts"] = {"train": 1, "val": 1}
    source = root / c["source_dataset"]
    records = []
    for split, country in (("train", "India"), ("val", "Japan")):
        filename = f"{country}_000001.jpg"
        row = {"derived_split": split, "original_filename": filename}
        for kind, suffix, field, digest_key in (("images", ".jpg", "exported_image_path", "exported_image_sha256"),
                                               ("labels", ".txt", "exported_label_path", "label_sha256")):
            relative = f"{kind}/{split}/{Path(filename).stem}{suffix}"
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture" if kind == "images" else b"")
            row[field] = relative
            row[digest_key] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(row)
    # Poison paths must never be opened or resolved by train/val byte verification.
    records.append({"derived_split": "test", "exported_image_path": "DO_NOT_OPEN",
                    "exported_label_path": "DO_NOT_OPEN"})
    write_json(source / "export_manifest.json", {"images": records})
    write_json(source / "validation_report.json", {"status": "passed", "phase2c1_complete": True,
               "split_image_counts": {"train": 1, "val": 1}, "exclusions": {"passed": True}})
    write_json(source / "export_report.json", {})
    yaml_path = root / c["data"]
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    yaml_path.write_text(yaml.safe_dump({"train": str(source / "images/train"),
        "val": str(source / "images/val"), "names": {int(k): v for k, v in c["class_names"].items()}}))
    c["train_val_yaml_sha256"] = tool.sha256_file(yaml_path)
    c["dataset_metadata_sha256"] = {name: tool.sha256_file(source / name)
                                   for name in c["dataset_metadata_sha256"]}
    return c


class Experiment2PolicyTests(unittest.TestCase):
    def test_explicit_parameters_match_baseline_and_independent_oracle(self):
        c = tool.load_config(ROOT)
        baseline = json.loads((ROOT / "configs/training/baseline_public_v1_yolov8s.yaml").read_bytes())
        expected = {"imgsz": 640, "batch": 4, "nbs": 64, "device": 0, "workers": 2,
            "cache": False, "amp": True, "compile": False, "epochs": 100, "patience": 20,
            "optimizer": "SGD", "lr0": .01, "lrf": .01, "cos_lr": True, "momentum": .937,
            "weight_decay": .0005, "warmup_epochs": 3., "seed": 42, "deterministic": True,
            "hsv_h": .015, "hsv_s": .4, "hsv_v": .3, "translate": .05, "scale": .25,
            "fliplr": .5, "mosaic": .25, "close_mosaic": 15, "degrees": 0., "flipud": 0.,
            "shear": 0., "perspective": 0., "mixup": 0., "cutmix": 0., "copy_paste": 0.,
            "bgr": 0., "fraction": 1., "rect": False, "multi_scale": 0., "iou": .7,
            "max_det": 300, "plots": True, "save": True, "save_period": 5,
            "pretrained": True, "task": "detect", "val": True}
        for key, value in expected.items():
            self.assertEqual(c["training"][key], value, key)
            self.assertEqual(c["training"][key], baseline[key], key)
        self.assertEqual(
            {key: c["training"][key] for key in ("box", "cls", "dfl")},
            {"box": 7.5, "cls": 0.5, "dfl": 1.5},
        )
        self.assertEqual({k: c["training"][k] for k in ("warmup_momentum", "warmup_bias_lr", "cls_pw", "end2end")},
                         {"warmup_momentum": .8, "warmup_bias_lr": .1, "cls_pw": 0., "end2end": True})
        self.assertEqual(c["split_counts"], {"train": 12620, "val": 2602})
        self.assertEqual(c["class_names"], {"0": "D00_longitudinal_crack", "1": "D10_transverse_crack",
            "2": "D20_alligator_crack", "3": "D40_pothole"})

    def test_fraction_is_exact_float_and_type_drift_is_rejected(self):
        expected_sha256 = "c299e47b9c04a95d3c5ddcd7215046d3a0d0118bbe05d382ac8e9211c3aadb01"
        c = config()
        self.assertEqual(tool.sha256_file(ROOT / tool.CONFIG), expected_sha256)
        self.assertEqual(tool.CONFIG_SHA256, expected_sha256)
        self.assertIs(type(c["training"]["fraction"]), float)
        self.assertEqual(c["training"]["fraction"], 1.0)

        for invalid in (1, True, "1.0"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                altered = copy.deepcopy(c)
                altered["training"]["fraction"] = invalid
                config_path = root / tool.CONFIG
                write_json(config_path, altered)
                with patch.object(tool, "CONFIG_SHA256", tool.sha256_file(config_path)):
                    with self.assertRaisesRegex(tool.Error, "exactly the float 1.0"):
                        tool.load_config(root)

    def test_recorded_baseline_loss_gains_are_literal_and_drift_is_rejected(self):
        import yaml

        c = config()
        recorded_path = ROOT / c["baseline_args"]
        self.assertEqual(tool.sha256_file(recorded_path), c["baseline_args_sha256"])
        recorded = yaml.safe_load(recorded_path.read_text(encoding="utf-8"))
        expected = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        self.assertEqual({key: recorded[key] for key in expected}, expected)
        self.assertEqual({key: c["training"][key] for key in expected}, expected)
        self.assertTrue(tool.verify_baseline_args(ROOT, c)["matched"])

        for key, changed_value in (("box", 7.6), ("cls", 0.6), ("dfl", 1.6)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                changed = dict(recorded)
                changed[key] = changed_value
                changed_path = root / "baseline_args.yaml"
                changed_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
                changed_config = config()
                changed_config["baseline_args"] = "baseline_args.yaml"
                changed_config["baseline_args_sha256"] = tool.sha256_file(changed_path)
                with self.assertRaises(tool.Error):
                    tool.verify_baseline_args(root, changed_config)

    def test_config_rejects_architecture_dataset_threshold_and_output_mutation(self):
        for field, value in (("model", {"path": "models/pretrained/yolov8s.pt"}),
                             ("source_dataset", "internal_test"), ("project", "outputs/training/baseline_public_v1"),
                             ("training", {"batch": 8})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                altered = config()
                altered[field] = value
                write_json(root / tool.CONFIG, altered)
                with self.assertRaises(tool.Error):
                    tool.load_config(root)

    def test_installed_architecture_is_exact_small_detect_without_model_construction(self):
        with patch("ultralytics.YOLO", side_effect=AssertionError("no model construction")):
            result = tool.architecture(config())
        self.assertEqual(result["scale"], "s")
        self.assertEqual(result["scales"]["s"], [.5, .5, 1024])
        self.assertTrue(result["end2end"])
        self.assertEqual(result["reg_max"], 1)

    def test_checkpoint_rejects_other_models_and_wrong_heads(self):
        expected = tool.architecture(config())
        class Detect:
            nc, reg_max, end2end = 80, 1, True
        model = SimpleNamespace(yaml=copy.deepcopy(expected), model=[Detect()])
        tool.verify_checkpoint_architecture(model, expected)
        for key, value in (("scale", "n"), ("scale", "m"), ("reg_max", 16),
                           ("end2end", False), ("head", [[1, 1, "Segment", []]]), ("backbone", [])):
            with self.subTest(key=key, value=value):
                changed = SimpleNamespace(yaml={**expected, key: value}, model=[Detect()])
                with self.assertRaises(tool.Error):
                    tool.verify_checkpoint_architecture(changed, expected)
        model.model[-1].nc = 4
        with self.assertRaises(tool.Error):
            tool.verify_checkpoint_architecture(model, expected)

    def test_missing_identity_prevents_loading_and_download(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, "_inspect_weight") as inspect, \
             patch.object(tool, "urlopen", side_effect=AssertionError("no acquisition")):
            with self.assertRaises(FileNotFoundError):
                tool.verify_identity(Path(temp), config())
            inspect.assert_not_called()

    def test_identity_hash_size_and_metadata_are_all_required(self):
        c = config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weight = root / c["model"]["path"]
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"fixture" * 20000)
            digest = tool.sha256_file(weight)
            valid = {"schema_version": "experiment2.pretrained_identity.v1", "family": "YOLO26",
                "variant": "s", "task": "detect", "path": c["model"]["path"], "source": c["model"]["source"],
                "ultralytics": "8.4.130", "architecture_sha256": c["model"]["architecture_sha256"],
                "size_bytes": weight.stat().st_size, "sha256": digest, "publisher_digest": f"sha256:{digest}"}
            identity = root / c["model"]["identity"]
            with patch.object(tool, "_inspect_weight") as inspect:
                write_json(identity, valid)
                self.assertEqual(tool.verify_identity(root, c)["sha256"], digest)
                inspect.assert_called_once()
                for key, value in (("variant", "n"), ("path", "weights/yolo26n.pt"), ("sha256", "0" * 64),
                                   ("size_bytes", 999999), ("source", "https://example.com/arbitrary.pt")):
                    with self.subTest(key=key):
                        write_json(identity, {**valid, key: value})
                        with self.assertRaises(tool.Error):
                            tool.verify_identity(root, c)

    def test_offline_framework_has_no_model_fallback_or_download_and_restores_state(self):
        import ultralytics.utils.downloads as downloads
        import ultralytics.utils.checks as checks
        original = downloads.safe_download
        with tempfile.TemporaryDirectory() as temp:
            root, c = Path(temp), config()
            for rel in (c["model"]["path"], c["amp_check_artifact"]):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"local fixture")
            with tool.offline_framework(root, c):
                self.assertFalse(checks.AUTOINSTALL)
                self.assertEqual(downloads.attempt_download_asset(root / c["model"]["path"]),
                                 str(root / c["model"]["path"]))
                for rel in ("yolo26n.pt", "yolov8s.pt", "arbitrary.pt", "outputs/training/baseline_public_v1/weights/last.pt"):
                    with self.assertRaises(tool.Error):
                        downloads.attempt_download_asset(root / rel)
                with self.assertRaises(tool.Error):
                    downloads.safe_download("https://example.com/weights.pt")
        self.assertIs(downloads.safe_download, original)

    def test_offline_framework_preserves_rescanned_cache_metadata_without_disk_write(self):
        import cv2
        import numpy as np
        from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset

        c = config()
        original_callback = tool._preserve_dataset_cache_metadata_without_write
        observed: list[dict[str, object]] = []

        def capture_callback(prefix, path, x, version):
            before_keys = set(x)
            before_values = dict(x)
            original_callback(prefix, path, x, version)
            self.assertEqual(set(x), before_keys | {"version"})
            self.assertEqual(x["version"], DATASET_CACHE_VERSION)
            for key, value in before_values.items():
                self.assertIs(x[key], value)
            observed.append({"path": Path(path), "keys": set(x), "version": x["version"]})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case, existing_cache in (("missing", False), ("stale", True)):
                dataset_root = root / case / "dataset"
                image_dir = dataset_root / "images" / "train"
                label_dir = dataset_root / "labels" / "train"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                image_path = image_dir / "sample.jpg"
                self.assertTrue(cv2.imwrite(str(image_path), np.full((32, 32, 3), 127, dtype=np.uint8)))
                (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
                cache_path = label_dir.with_suffix(".cache")
                original_cache_bytes = None
                if existing_cache:
                    with cache_path.open("wb") as stream:
                        np.save(stream, {
                            "labels": [],
                            "hash": "stale-cache-hash",
                            "results": (0, 0, 0, 0, 0),
                            "msgs": [],
                            "version": DATASET_CACHE_VERSION,
                        })
                    original_cache_bytes = cache_path.read_bytes()

                with patch.object(
                    tool,
                    "_preserve_dataset_cache_metadata_without_write",
                    side_effect=capture_callback,
                ) as callback, tool.offline_framework(root, c):
                    labels = YOLODataset(
                        img_path=str(image_dir),
                        imgsz=32,
                        batch_size=1,
                        augment=False,
                        cache=False,
                        data={"names": dict(tool.shared.CLASS_NAMES)},
                        task="detect",
                    ).labels

                callback.assert_called_once()
                self.assertEqual(len(labels), 1)
                if existing_cache:
                    self.assertEqual(cache_path.read_bytes(), original_cache_bytes)
                else:
                    self.assertFalse(cache_path.exists())

        self.assertEqual(len(observed), 2)
        for record in observed:
            self.assertEqual(record["version"], DATASET_CACHE_VERSION)
            self.assertEqual(record["keys"], {"labels", "hash", "results", "msgs", "version"})

    def test_ultralytics_fraction_type_controls_train_count_but_not_validation(self):
        import cv2
        import numpy as np
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset

        c = config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split_counts = {"train": 5, "val": 3}
            for split, count in split_counts.items():
                image_dir = root / "dataset" / "images" / split
                label_dir = root / "dataset" / "labels" / split
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                for index in range(count):
                    image_path = image_dir / f"sample_{index:02d}.jpg"
                    self.assertTrue(cv2.imwrite(
                        str(image_path), np.full((32, 32, 3), 96 + index, dtype=np.uint8)
                    ))
                    (label_dir / f"sample_{index:02d}.txt").write_text(
                        "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
                    )

            data = {"names": dict(tool.shared.CLASS_NAMES)}

            def build(fraction, split):
                cfg = get_cfg()
                cfg.task = "detect"
                cfg.imgsz = 32
                cfg.cache = False
                cfg.rect = False
                cfg.single_cls = False
                cfg.classes = None
                cfg.fraction = fraction
                return build_yolo_dataset(
                    cfg,
                    str(root / "dataset" / "images" / split),
                    batch=4,
                    data=data,
                    mode=split,
                    stride=32,
                )

            original_callback = tool._preserve_dataset_cache_metadata_without_write
            with patch.object(
                tool,
                "_preserve_dataset_cache_metadata_without_write",
                wraps=original_callback,
            ) as cache_callback, tool.offline_framework(root, c):
                float_train = build(1.0, "train")
                float_val = build(1.0, "val")
                integer_train = build(1, "train")
                integer_val = build(1, "val")

            self.assertEqual(len(float_train), split_counts["train"])
            self.assertEqual(len(float_val), split_counts["val"])
            self.assertEqual(len(integer_train), 1)
            self.assertEqual(len(integer_val), split_counts["val"])
            self.assertEqual(cache_callback.call_count, 4)
            for split in split_counts:
                self.assertFalse((root / "dataset" / "labels" / f"{split}.cache").exists())

    def test_acquisition_requires_publisher_sha_before_download_or_checkpoint_loading(self):
        c = config()
        release = {"tag_name": "v8.4.0", "assets": [{"name": "yolo26s.pt", "size": 200000,
                    "browser_download_url": c["model"]["source"]}]}
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, "load_config", return_value=c), \
             patch.object(tool, "verify_environment"), patch.object(tool, "architecture"), \
             patch.object(tool, "_inspect_weight") as inspect, \
             patch.object(tool, "urlopen", return_value=io.BytesIO(json.dumps(release).encode())) as network:
            with self.assertRaises(tool.Error):
                tool.acquire_pretrained(Path(temp))
            self.assertEqual(network.call_count, 1)
            inspect.assert_not_called()

    def test_controlled_acquisition_freezes_verified_identity_then_stops(self):
        c = config()
        content = b"official synthetic asset" * 10000
        digest = hashlib.sha256(content).hexdigest()
        asset = {"name": "yolo26s.pt", "id": 42, "size": len(content),
                 "browser_download_url": c["model"]["source"], "digest": f"sha256:{digest}"}
        release = {"tag_name": "v8.4.0", "assets": [asset]}
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, "load_config", return_value=c), \
             patch.object(tool, "verify_environment"), patch.object(tool, "architecture"), \
             patch.object(tool, "_inspect_weight") as inspect, \
             patch.object(tool, "launch", side_effect=AssertionError("acquisition must stop")), \
             patch.object(tool, "urlopen", side_effect=[io.BytesIO(json.dumps(release).encode()), io.BytesIO(content)]):
            root = Path(temp)
            identity = tool.acquire_pretrained(root)
            self.assertEqual(identity["sha256"], digest)
            self.assertEqual(identity["size_bytes"], len(content))
            self.assertFalse(identity["training_started"])
            self.assertEqual((root / c["model"]["path"]).read_bytes(), content)
            self.assertEqual(json.loads((root / c["model"]["identity"]).read_bytes()), identity)
            inspect.assert_called_once()
            with self.assertRaises(tool.Error):
                tool.acquire_pretrained(root)

    def test_git_requires_reviewed_descendant_and_tracked_identity(self):
        c = config()
        with patch.object(tool.shared, "collect_git_state", return_value={"commit": c["prerequisite_commit"]}), \
             patch.object(tool.subprocess, "run") as command:
            with self.assertRaises(tool.Error):
                tool.verify_git(ROOT, c)
            command.assert_not_called()
        with patch.object(tool.shared, "collect_git_state", return_value={"commit": "a" * 40}), \
             patch.object(tool.subprocess, "run") as command, \
             patch.object(tool, "build_source_state_manifest", return_value={
                 "git_commit": "a" * 40, "source_tree_sha256": "b" * 64
             }) as source_state:
            self.assertEqual(tool.verify_git(ROOT, c)["commit"], "a" * 40)
            self.assertIn("--is-ancestor", command.call_args_list[0].args[0])
            tracked = command.call_args_list[1].args[0]
            self.assertIn("--error-unmatch", tracked)
            self.assertIn("reproducibility/experiment2_yolo26s_pretrained_identity.json", tracked)
            source_state.assert_called_once_with(
                ROOT, mode=tool.SOURCE_STATE_MODE_COMMITTED_GIT_TREE
            )
        with patch.object(tool.shared, "collect_git_state", return_value={"commit": "a" * 40}), \
             patch.object(tool.subprocess, "run"), \
             patch.object(tool, "build_source_state_manifest", return_value={
                 "git_commit": "c" * 40, "source_tree_sha256": "b" * 64
             }), self.assertRaises(tool.Error):
            tool.verify_git(ROOT, c)

    def test_committed_tree_source_state_excludes_untracked_and_follows_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init", "--quiet")
            git("config", "user.name", "Experiment 2 Test")
            git("config", "user.email", "experiment2-test@example.invalid")
            git("config", "commit.gpgsign", "false")
            tracked_content = b"tracked-v1\n"
            for directory, filename in (
                ("src", "tracked.py"),
                ("configs", "tracked.json"),
                ("tests", "tracked.txt"),
                ("road-damage-project-docs", "tracked.md"),
            ):
                path = root / directory / filename
                path.parent.mkdir(parents=True)
                path.write_bytes(tracked_content if directory == "src" else directory.encode())
            for filename in ("AGENTS.md", "requirements.txt"):
                (root / filename).write_text(filename, encoding="utf-8")
            (root / ".gitignore").write_text(
                "/road-damage-project-docs/literature/\n", encoding="utf-8"
            )
            git("add", "--all")
            git("commit", "--quiet", "-m", "initial source")

            first = tool.build_source_state_manifest(
                root, mode=tool.SOURCE_STATE_MODE_COMMITTED_GIT_TREE
            )
            self.assertEqual(first["mode"], "committed_git_tree")
            self.assertEqual(first["git_commit"], git("rev-parse", "HEAD"))
            first_files = {item["path"]: item for item in first["files"]}
            self.assertEqual(
                first_files["src/tracked.py"]["sha256"], hashlib.sha256(tracked_content).hexdigest()
            )

            info_exclude = root / ".git" / "info" / "exclude"
            with info_exclude.open("a", encoding="utf-8") as stream:
                stream.write("/road-damage-project-docs/research-paper/\n")
            for relative in (
                "road-damage-project-docs/research-paper/untracked.txt",
                "road-damage-project-docs/literature/untracked.txt",
            ):
                path = root / relative
                path.parent.mkdir(parents=True)
                path.write_text("untracked", encoding="utf-8")

            second = tool.build_source_state_manifest(
                root, mode=tool.SOURCE_STATE_MODE_COMMITTED_GIT_TREE
            )
            self.assertEqual(second, first)
            self.assertFalse(
                any("research-paper" in item["path"] or "literature" in item["path"] for item in second["files"])
            )

            changed_content = b"tracked-v2\n"
            (root / "src" / "tracked.py").write_bytes(changed_content)
            self.assertEqual(
                tool.build_source_state_manifest(
                    root, mode=tool.SOURCE_STATE_MODE_COMMITTED_GIT_TREE
                ),
                first,
            )
            filesystem = tool.build_source_state_manifest(root)
            filesystem_paths = {item["path"] for item in filesystem["files"]}
            self.assertIn("road-damage-project-docs/research-paper/untracked.txt", filesystem_paths)
            self.assertIn("road-damage-project-docs/literature/untracked.txt", filesystem_paths)
            self.assertNotIn("mode", filesystem)
            self.assertNotIn("git_commit", filesystem)

            git("add", "src/tracked.py")
            git("commit", "--quiet", "-m", "change tracked source")
            third = tool.build_source_state_manifest(
                root, mode=tool.SOURCE_STATE_MODE_COMMITTED_GIT_TREE
            )
            self.assertEqual(third["git_commit"], git("rev-parse", "HEAD"))
            self.assertNotEqual(third["git_commit"], first["git_commit"])
            self.assertNotEqual(third["source_tree_sha256"], first["source_tree_sha256"])
            third_files = {item["path"]: item for item in third["files"]}
            self.assertEqual(
                third_files["src/tracked.py"]["sha256"], hashlib.sha256(changed_content).hexdigest()
            )

    def test_blocked_launch_never_constructs_model_or_reserves_output(self):
        c = config()
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, "load_config", return_value=c), \
             patch.object(tool, "preflight", return_value={"blockers": {"identity": "missing"}}), \
             patch("ultralytics.YOLO", side_effect=AssertionError("blocked model load")), \
             patch.object(tool.shared, "reserve_fresh_run_directory") as reserve:
            root = Path(temp)
            with patch.object(tool, "PROJECT_ROOT", root):
                for mode in ("smoke", "train"):
                    with self.assertRaises(tool.Error):
                        tool.launch(mode, root)
            reserve.assert_not_called()

    def test_preflight_does_no_training_inference_acquisition_or_output_write(self):
        c = config()
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, "load_config", return_value=c), \
             patch.object(tool, "verify_baseline_args", return_value={}), \
             patch.object(tool, "verify_environment", return_value={}), \
             patch.object(tool, "architecture", return_value={}), \
             patch.object(tool, "verify_dataset", return_value={}) as dataset, \
             patch.object(tool, "verify_identity", side_effect=FileNotFoundError("acquire first")), \
             patch.object(tool, "verify_git", return_value={}), \
             patch.object(tool, "launch", side_effect=AssertionError("no train")), \
             patch.object(tool, "acquire_pretrained", side_effect=AssertionError("no acquire")), \
             patch.object(tool, "urlopen", side_effect=AssertionError("no network")):
            report = tool.preflight(Path(temp))
            self.assertFalse(report["ready_for_smoke"])
            self.assertEqual(set(report["blockers"]), {"pretrained_identity"})
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["inference_performed"])
            dataset.assert_called_once_with(Path(temp), c)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_dataset_identity_pair_counts_and_no_internal_test_file_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = dataset_fixture(root)
            original = Path.open
            opened = []
            def guarded(path, *args, **kwargs):
                self.assertNotIn("test", path.parts)
                self.assertNotIn("DO_NOT_OPEN", path.parts)
                opened.append(path)
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guarded):
                report = tool.verify_dataset(root, c)
                self.assertFalse(any(p.suffix in (".jpg", ".txt") for p in opened))
                self.assertFalse(report["internal_test_files_accessed"])
                self.assertEqual(report["counts"], {"train": 1, "val": 1})
                tool.verify_dataset(root, c, hashes=True)
            self.assertEqual(len([p for p in opened if p.suffix == ".jpg"]), 2)
            label = root / c["source_dataset"] / "labels/train/India_000001.txt"
            label.write_bytes(b"changed")
            with self.assertRaises(tool.Error):
                tool.verify_dataset(root, c, hashes=True)

    def test_dataset_count_pair_and_yaml_test_key_drift_fail(self):
        for mutation in ("count", "pair", "test_key"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                c = dataset_fixture(root)
                if mutation == "count":
                    c["split_counts"]["train"] = 12620
                elif mutation == "pair":
                    source = root / c["source_dataset"] / "labels/train"
                    (source / "India_000001.txt").rename(source / "India_999999.txt")
                else:
                    path = root / c["data"]
                    path.write_text(path.read_text() + "\ntest: forbidden\n")
                    c["train_val_yaml_sha256"] = tool.sha256_file(path)
                with self.assertRaises(tool.Error):
                    tool.verify_dataset(root, c)

    def test_smoke_cannot_become_full_training_and_output_isolation(self):
        c = config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            smoke = tool.training_arguments(root, c, "smoke", root / "snapshot.yaml")
            train = tool.training_arguments(root, c, "train", root / "snapshot.yaml")
            self.assertEqual(smoke["epochs"], 1)
            self.assertEqual(train["epochs"], 100)
            for args in (smoke, train):
                self.assertEqual((args["batch"], args["imgsz"], args["seed"], args["optimizer"]), (4, 640, 42, "SGD"))
                self.assertFalse(args["resume"])
                self.assertEqual(Path(args["project"]), root / "outputs/training/experiment2_yolo26s_matched")
            self.assertNotEqual(smoke["save_dir"], train["save_dir"])
            for mode in ("resume", "test", "full", ""):
                with self.assertRaises(tool.Error):
                    tool.run_directory(root, c, mode)
            (root / Path(train["save_dir"])).mkdir(parents=True)
            with self.assertRaises(tool.Error):
                tool.verify_output(root, c)

    def test_full_training_requires_complete_identical_smoke_receipt(self):
        c = config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = tool.run_directory(root, c, "smoke")
            identity = {"sha256": "a" * 64}
            git_commit = "c" * 40
            valid = {"mode": "smoke", "status": "COMPLETED", "epochs_completed": 1,
                "model_sha256": identity["sha256"], "config_sha256": tool.CONFIG_SHA256,
                "git_commit": git_commit, "source_tree_sha256": "b" * 64,
                "smoke_manifest_sha256": c["smoke"]["manifest_sha256"],
                "batches_completed": 32, "optimizer_updates": 2, "amp_enabled": True, "cuda_device": "cuda:0",
                "optimizer_state_present": True, "scaler_state_present": True,
                "dataset_metadata_sha256": c["dataset_metadata_sha256"], "artifacts": {}}
            for rel in ("weights/last.pt", "weights/best.pt", "results.csv"):
                path = directory / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
                valid["artifacts"][rel] = tool.sha256_file(path)
            with self.assertRaises(FileNotFoundError):
                tool._smoke_receipt(root, c, identity, "b" * 64, git_commit)
            manifest = {"mode": "smoke", "git_commit": git_commit,
                        "source_tree_sha256": "b" * 64}
            write_json(directory / "run_manifest.json", manifest)
            write_json(directory / "completion.json", valid)
            tool._smoke_receipt(root, c, identity, "b" * 64, git_commit)
            missing_commit = dict(valid)
            missing_commit.pop("git_commit")
            write_json(directory / "completion.json", missing_commit)
            with self.assertRaises(tool.Error):
                tool._smoke_receipt(root, c, identity, "b" * 64, git_commit)
            for key, value in (("status", "STARTED"), ("status", "FAILED_TECHNICAL"), ("epochs_completed", 2),
                               ("batches_completed", 31), ("optimizer_updates", 0), ("model_sha256", "f" * 64),
                               ("git_commit", "not-a-git-commit"), ("git_commit", "d" * 40),
                               ("source_tree_sha256", "f" * 64), ("artifacts", {}), ("amp_enabled", False),
                               ("optimizer_state_present", False), ("scaler_state_present", False)):
                with self.subTest(key=key, value=value):
                    write_json(directory / "completion.json", {**valid, key: value})
                    with self.assertRaises(tool.Error):
                        tool._smoke_receipt(root, c, identity, "b" * 64, git_commit)
            write_json(directory / "completion.json", valid)
            for key, value in (("git_commit", "d" * 40), ("source_tree_sha256", "f" * 64)):
                with self.subTest(manifest_key=key):
                    write_json(directory / "run_manifest.json", {**manifest, key: value})
                    with self.assertRaises(tool.Error):
                        tool._smoke_receipt(root, c, identity, "b" * 64, git_commit)
            write_json(directory / "run_manifest.json", manifest)
            (directory / "weights/last.pt").write_bytes(b"corrupt")
            with self.assertRaises(tool.Error):
                tool._smoke_receipt(root, c, identity, "b" * 64, git_commit)

    def test_smoke_receipt_rejects_new_commit_with_identical_scoped_source_hash(self):
        c = config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init", "--quiet")
            git("config", "user.name", "Experiment 2 Test")
            git("config", "user.email", "experiment2-test@example.invalid")
            git("config", "commit.gpgsign", "false")
            for directory, filename in (
                ("src", "tracked.py"),
                ("configs", "tracked.yaml"),
                ("tests", "tracked.py"),
                ("road-damage-project-docs", "tracked.md"),
            ):
                path = root / directory / filename
                path.parent.mkdir(parents=True)
                path.write_text(directory, encoding="utf-8")
            for filename in ("AGENTS.md", "requirements.txt"):
                (root / filename).write_text(filename, encoding="utf-8")
            (root / ".gitignore").write_text("# fixture\n", encoding="utf-8")
            outside = root / "out-of-scope" / "tracked.txt"
            outside.parent.mkdir(parents=True)
            outside.write_text("commit A", encoding="utf-8")
            git("add", "--all")
            git("commit", "--quiet", "-m", "commit A")
            state_a = tool.build_source_state_manifest(
                root, mode=tool.SOURCE_STATE_MODE_COMMITTED_GIT_TREE
            )

            identity = {"sha256": "a" * 64}
            directory = tool.run_directory(root, c, "smoke")
            artifacts = {}
            for relative in ("weights/last.pt", "weights/best.pt", "results.csv"):
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
                artifacts[relative] = tool.sha256_file(path)
            write_json(directory / "completion.json", {
                "mode": "smoke", "status": "COMPLETED", "epochs_completed": 1,
                "model_sha256": identity["sha256"], "config_sha256": tool.CONFIG_SHA256,
                "git_commit": state_a["git_commit"],
                "source_tree_sha256": state_a["source_tree_sha256"],
                "smoke_manifest_sha256": c["smoke"]["manifest_sha256"],
                "batches_completed": 32, "optimizer_updates": 2, "amp_enabled": True,
                "cuda_device": "cuda:0", "optimizer_state_present": True,
                "scaler_state_present": True,
                "dataset_metadata_sha256": c["dataset_metadata_sha256"],
                "artifacts": artifacts,
            })
            write_json(directory / "run_manifest.json", {
                "mode": "smoke", "git_commit": state_a["git_commit"],
                "source_tree_sha256": state_a["source_tree_sha256"],
            })

            outside.write_text("commit B", encoding="utf-8")
            git("add", "out-of-scope/tracked.txt")
            git("commit", "--quiet", "-m", "commit B")
            state_b = tool.build_source_state_manifest(
                root, mode=tool.SOURCE_STATE_MODE_COMMITTED_GIT_TREE
            )
            self.assertNotEqual(state_b["git_commit"], state_a["git_commit"])
            self.assertEqual(state_b["source_tree_sha256"], state_a["source_tree_sha256"])
            with self.assertRaises(tool.Error):
                tool._smoke_receipt(
                    root,
                    c,
                    identity,
                    state_b["source_tree_sha256"],
                    state_b["git_commit"],
                )

    def test_environment_uses_only_expected_interpreter(self):
        with patch.object(tool.sys, "executable", "C:/other/python.exe"), \
             patch.object(tool.shared, "collect_environment") as env:
            with self.assertRaises(tool.Error):
                tool.verify_environment(ROOT, config())
            env.assert_not_called()

    def test_cli_modes_are_exclusive_and_no_model_epoch_or_resume_override(self):
        for argv in ([], ["--train", "--smoke"], ["--smoke", "--epochs", "100"],
                     ["--train", "--model", "yolo26n.pt"], ["--resume", "last.pt"], ["--force"]):
            with self.subTest(argv=argv), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    tool.main(argv)
        with patch.object(tool, "preflight", return_value={"ready_for_smoke": True}), \
             patch.object(tool, "launch") as launch, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(tool.main(["--preflight"]), 0)
            launch.assert_not_called()


class Experiment2LaunchTests(unittest.TestCase):
    def test_mocked_smoke_and_incomplete_smoke_never_release_full_training(self):
        import torch
        for completed, readable in ((True, True), (False, True), (True, False)):
            with self.subTest(completed=completed, readable=readable), tempfile.TemporaryDirectory() as temp:
                root, c = Path(temp), config()
                data = {"train": str(root / "smoke/images/train"), "val": str(root / "smoke/images/val"),
                        "names": dict(tool.shared.CLASS_NAMES)}
                report = {"blockers": {}, "checks": {"pretrained_identity": {"sha256": "a" * 64},
                    "git": {"commit": "c" * 40, "source_state": {
                        "git_commit": "c" * 40, "source_tree_sha256": "b" * 64}}}}
                callbacks = {}
                model = MagicMock()
                model.add_callback.side_effect = lambda key, callback: callbacks.update({key: callback})
                def fake_train(**args):
                    run = Path(args["save_dir"])
                    (run / "weights").mkdir()
                    (run / "weights/last.pt").write_bytes(b"mock checkpoint")
                    (run / "weights/best.pt").write_bytes(b"mock best")
                    (run / "results.csv").write_text("epoch,time,train/box_loss\n1,1.0,0.1\n")
                    tensor = torch.tensor(1., requires_grad=True)
                    trainer = SimpleNamespace(save_dir=run, start_epoch=0, args=SimpleNamespace(**args),
                        data=data, model=SimpleNamespace(model=[SimpleNamespace(nc=4, reg_max=1)], end2end=True),
                        device="cuda:0", amp=True, optimizer=torch.optim.SGD([tensor], lr=.01),
                        epoch=0, loss=torch.tensor(.1), ema=SimpleNamespace(updates=2), last=run / "weights/last.pt",
                        scaler=SimpleNamespace(state_dict=lambda: {"scale": 1.}),
                        stopper=SimpleNamespace(best_epoch=1, best_fitness=.1, patience=20))
                    trainer.optimizer.state[tensor]["momentum_buffer"] = torch.tensor(.1)
                    callbacks["on_pretrain_routine_end"](trainer)
                    for _ in range(32 if completed else 31):
                        callbacks["on_train_batch_end"](trainer)
                    callbacks["on_model_save"](trainer)
                model.train.side_effect = fake_train
                with patch.object(tool, "PROJECT_ROOT", root), patch.object(tool, "load_config", return_value=c), \
                     patch.object(tool, "preflight", return_value=report), patch.object(tool, "verify_dataset"), \
                     patch.object(tool, "_smoke_data", return_value={"passed": True}), \
                     patch.object(tool.shared, "validate_train_val_yaml", return_value=data), \
                     patch.object(tool, "offline_framework", return_value=nullcontext()), \
                     patch.object(tool, "architecture", return_value={}), patch.object(tool, "verify_checkpoint_architecture"), \
                     patch.object(tool, "_verify_completed_checkpoint", side_effect=None if readable else tool.Error("corrupt")), \
                     patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=1), \
                     patch("torch.cuda.max_memory_reserved", return_value=1), patch("ultralytics.YOLO", return_value=model):
                    if completed and readable:
                        tool.launch("smoke", root)
                    else:
                        with self.assertRaises(tool.Error):
                            tool.launch("smoke", root)
                    directory = tool.run_directory(root, c, "smoke")
                    self.assertEqual((directory / "completion.json").exists(), completed and readable)
                    self.assertEqual((directory / "failure.json").exists(), not (completed and readable))
                    self.assertEqual(model.train.call_args.kwargs["epochs"], 1)
                    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["git_commit"], "c" * 40)
                    self.assertEqual(manifest["source_tree_sha256"], "b" * 64)
                    if completed and readable:
                        receipt = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
                        self.assertEqual(receipt["git_commit"], manifest["git_commit"])
                        self.assertEqual(receipt["source_tree_sha256"], manifest["source_tree_sha256"])


if __name__ == "__main__":
    unittest.main()
