"""Synthetic final-test tooling tests; never open project test pairs or model weights."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_damage.evaluation import internal_test as tool  # noqa: E402
from road_damage.evaluation import internal_test_metrics as reports  # noqa: E402
from road_damage.evaluation.threshold_metrics import Box, ThresholdError, sweep_cache  # noqa: E402


def fixture(root: Path, size: tuple[int, int] = (20, 20)) -> tuple[dict, dict]:
    """Six generated images/labels and independent synthetic approved export metadata."""
    from PIL import Image
    config = copy.deepcopy(tool.EXPECTED_CONFIG)
    config["expected_counts"] = {"images": 6, "positive_images": 4, "negative_images": 2,
        "targets": 4, "countries": {"India": 3, "Japan": 3},
        "class_boxes": {"D00": 1, "D10": 1, "D20": 1, "D40": 1}}
    images = root / tool.shared.EXPORT / "images/test"
    labels = root / tool.shared.EXPORT / "labels/test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rows = []
    for index, cls in enumerate((1, 0, None, 2, 3, None)):
        country = "India" if index < 3 else "Japan"
        name = f"{country}_{index:06d}.jpg"
        image_path, label_path = images / name, labels / (Path(name).stem + ".txt")
        Image.new("RGB", size, color=(index * 30, 0, 0)).save(image_path)
        label_path.write_bytes(b"" if cls is None else f"{cls} 0.5 0.5 0.5 0.5\n".encode())
        rows.append({"original_filename": name, "country": country, "derived_split": "test",
            "exported_image_path": "images/test/" + name, "exported_label_path": "labels/test/" + label_path.name,
            "raw_relative_image_path": f"data/external/rdd2022/raw/{country}/train/images/{name}",
            "source_image_sha256": tool.shared.digest(image_path), "exported_image_sha256": tool.shared.digest(image_path),
            "label_sha256": tool.shared.digest(label_path), "is_positive": cls is not None,
            "target_object_count": int(cls is not None), "target_class_counts": {} if cls is None else {tool.CODES[cls]: 1}})
    manifest = {"taxonomy": [{"id": c, "raw_code": code, "name": tool.shared.CLASS_NAMES[c]} for c, code in enumerate(tool.CODES)], "images": rows}
    report = {"status": "export_validated", "counts": {"split_images": {"test": 6}}}
    validation = {"status": "passed", "split_image_counts": {"test": 6},
                  "target_object_counts": {"test": config["expected_counts"]["class_boxes"]},
                  "exclusions": {"passed": True, "intersections": {"official_unlabelled_test": []}, "teacher_reference_count": 0}}
    for name, value in (("export_manifest.json", manifest), ("export_report.json", report), ("validation_report.json", validation)):
        path = root / tool.shared.EXPORT / name
        path.write_bytes(tool.shared.json_bytes(value))
        config["approved_metadata_sha256"][name] = tool.shared.digest(path)
    path = root / tool.CONFIG
    path.parent.mkdir(parents=True)
    path.write_bytes(tool.shared.json_bytes(config))
    gt_rows = []
    width, height = size
    for row in rows:
        cls = next((c for c, code in enumerate(tool.CODES)
                    if row["target_class_counts"].get(code)), None)
        ground_truth = ([] if cls is None else [tool.shared.box_record(Box(
            cls, (.25 * width, .25 * height, .75 * width, .75 * height)))])
        gt_rows.append({"image_id": row["original_filename"], "country": row["country"],
            "image_path": (tool.shared.EXPORT / row["exported_image_path"]).as_posix(),
            "label_path": (tool.shared.EXPORT / row["exported_label_path"]).as_posix(),
            "width": width, "height": height,
            "image_sha256": row["exported_image_sha256"], "label_sha256": row["label_sha256"],
            "ground_truth": ground_truth})
    gt = {"schema_version": "internal_test_ground_truth.v1", "split": "canonical_internal_test",
          "counts": config["expected_counts"], "images": gt_rows}
    return config, gt


def predicted(gt: dict) -> dict[str, list[Box]]:
    return {r["image_id"]: [Box(b["class_id"], tuple(b["xyxy"]), .8) for b in r["ground_truth"]]
            for r in gt["images"]}



def identity(root: Path, config: dict) -> dict:
    records = tool.approved_metadata(root, config)
    provenance = {"evaluation_commit": "synthetic-reviewed-commit", "worktree_clean": True,
                  "post_training_source_fingerprint": "synthetic-source",
                  "source_files": [{"path": "src/synthetic.py", "sha256": "synthetic-code"}],
                  "training_history": {"original_training_commit": "synthetic-training"},
                  "threshold_selection_commit": tool.SELECTION_COMMIT}
    framework = {"python": "3.12.10", "packages": {"ultralytics": "8.4.130", "torch": "synthetic"},
                 "cuda": "synthetic-cuda", "device": 0, "gpu": "synthetic-gpu",
                 "ultralytics_source_sha256": {"ultralytics/models/yolo/detect/val.py": "synthetic-framework"}}
    selection = {"validation_only": True, "internal_test_untouched": True,
                 "artifacts_verified": {"selected_operating_point.json": "synthetic-selection"}}
    return tool._recovery_identity(root, config, records, provenance, framework, selection)


def identity_leaf_paths(value, prefix=()):
    """Derive every scalar/empty-container leaf; no duplicated production field list."""
    if isinstance(value, dict) and value:
        for key in sorted(value):
            yield from identity_leaf_paths(value[key], (*prefix, key))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from identity_leaf_paths(item, (*prefix, index))
    else:
        yield prefix


def mutate_identity_leaf(value: dict, path: tuple) -> dict:
    changed = copy.deepcopy(value)
    parent = changed
    for part in path[:-1]:
        parent = parent[part]
    key = path[-1]
    original = parent[key]
    if type(original) is bool:
        replacement = not original
    elif type(original) is int:
        replacement = original + 1
    elif type(original) is float:
        replacement = original + .001
    elif isinstance(original, str):
        replacement = original + "-changed"
    elif isinstance(original, list):
        replacement = ["changed"]
    elif isinstance(original, dict):
        replacement = {"changed": True}
    else:
        replacement = "changed"
    parent[key] = replacement
    return changed


@contextmanager
def bound(root: Path, config: dict, gt: dict):
    """All authority checks refer to synthetic temp fixtures, never project weights/data."""
    frozen = identity(root, config)
    records = tool.approved_metadata(root, config)
    with patch.object(tool, "ROOT", root), \
         patch.object(tool, "_prepare_metadata", return_value=(config, records, frozen)), \
         patch.object(tool.shared, "verify_model_identity"):
        yield frozen


def native_validator(output: Path):
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    v = DetectionValidator(save_dir=output / "native", args={"conf": .001, "iou": .5, "plots": False,
                           "imgsz": 640, "rect": False, "device": "cpu", "workers": 0, "split": "test"})
    v.device, v.training, v.data = torch.device("cpu"), False, {"test": "synthetic", "names": tool.shared.CLASS_NAMES}
    v.stride = 32
    v.init_metrics(SimpleNamespace(names=tool.shared.CLASS_NAMES, end2end=False))
    return v


def standard_row(r: dict, v=None) -> dict:
    import torch
    xyxy = torch.tensor([b["xyxy"] for b in r["ground_truth"]], dtype=torch.float32).reshape(-1, 4)
    cls = torch.tensor([b["class_id"] for b in r["ground_truth"]], dtype=torch.float32)
    pred = {"bboxes": xyxy, "cls": cls, "conf": torch.full((len(cls),), .8)}
    ground = {"bboxes": xyxy, "cls": cls}
    if v is None:
        tp = [[True] * 10 for _ in cls]
    else:
        tp = v._process_batch(pred, ground)["tp"].tolist()
    return {"image_id": r["image_id"], "country": r["country"], "nms_iou": .5,
            "confidence_floor": .001, "multi_label": True, "input_shape": [r["height"], r["width"]],
            "original_shape": [r["height"], r["width"]], "ratio_pad": [[1., 1.], [0, 0]],
            "ground_truth": reports.tensor_boxes(ground, confidence=False),
            "predictions": reports.tensor_boxes(pred, confidence=True), "native_tp": tp}


def frozen_row(r: dict) -> dict:
    return {"image_id": r["image_id"], "country": r["country"], "nms_iou": .5,
            "confidence_floor": .19, "multi_label": False,
            "predictions": [{**b, "confidence": .8} for b in r["ground_truth"]]}


def sealed(path: Path, branch: str, gt: dict, frozen_sha: str, rows=None) -> None:
    if rows is None:
        rows = [standard_row(r) if branch == "standard" else frozen_row(r) for r in gt["images"]]
    with tool._CacheWriter(path, branch, frozen_sha) as writer:
        for row in rows:
            writer.write(row)


class GuardedTest(unittest.TestCase):
    def setUp(self):
        # A test regression cannot accidentally touch the actual project's held-out pairs.
        self.stack = ExitStack()
        original_open, original_iterdir = Path.open, Path.iterdir
        forbidden = [str((ROOT / tool.shared.EXPORT / kind / "test").absolute()).casefold()
                     for kind in ("images", "labels")]
        def check(path):
            name = str(path.absolute()).casefold()
            if any(name == base or name.startswith(base + "\\") or name.startswith(base + "/") for base in forbidden):
                raise AssertionError("REAL INTERNAL TEST ACCESS IS FORBIDDEN")
        def safe_open(path, *a, **k):
            check(path)
            return original_open(path, *a, **k)
        def safe_iterdir(path):
            check(path)
            return original_iterdir(path)
        self.stack.enter_context(patch.object(Path, "open", safe_open))
        self.stack.enter_context(patch.object(Path, "iterdir", safe_iterdir))
        self.addCleanup(self.stack.close)
        self.addCleanup(setattr, tool, "_ACTIVE_CONTEXT", None)
        self.addCleanup(setattr, tool, "_ACTIVE_INVENTORY_CAPABILITY", None)


class FrozenPolicyTests(GuardedTest):
    def test_scientific_identity_literal_oracle(self):
        c = tool.load_config()
        self.assertEqual(c["model"], "outputs/training/baseline_public_v1/20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42/weights/best.pt")
        self.assertEqual(c["model_sha256"], "BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7")
        self.assertEqual(c["model_size_bytes"], 22524074)
        self.assertEqual((c["confidence"], c["nms_iou"], c["matching_iou"]), (.19, .5, .5))
        self.assertEqual(c["selection"]["commit"], "027f5f41c386f558319799fa8784b32cf3dcefdb")
        self.assertEqual(c["classes"], {"0": "D00_longitudinal_crack", "1": "D10_transverse_crack", "2": "D20_alligator_crack", "3": "D40_pothole"})
        self.assertEqual(c["expected_counts"], {"images": 2775, "positive_images": 1712, "negative_images": 1063,
            "targets": 3535, "countries": {"India": 1116, "Japan": 1659},
            "class_boxes": {"D00": 841, "D10": 557, "D20": 1297, "D40": 840}})
        self.assertEqual(c["evaluation_branches"], ["framework_validator", "frozen_prediction_mode"])

    def test_config_conf_nms_matching_model_overrides_rejected(self):
        for key, value in (("confidence", .2), ("nms_iou", .6), ("matching_iou", .75), ("model", "another.pt"),
                           ("model_sha256", "0" * 64), ("device", "cpu"), ("force", True), ("output", "other")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / tool.CONFIG
                path.parent.mkdir(parents=True)
                path.write_bytes(tool.shared.json_bytes({**tool.EXPECTED_CONFIG, key: value}))
                with self.assertRaises(ThresholdError):
                    tool.load_config(root)

    def test_cli_rejects_conf_nms_matching_and_force(self):
        for flag in ("--conf", "--confidence", "--iou", "--nms", "--matching-iou", "--force", "--model", "--output"):
            with self.subTest(flag=flag), patch.object(tool, "_prepare_metadata") as prepare, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    tool.main(["--evaluate-once", flag, "0.2"])
                prepare.assert_not_called()

    def test_public_api_no_model_config_or_threshold_override(self):
        for key in ("model", "config", "confidence", "nms_iou", "matching_iou", "output"):
            with self.subTest(key=key), patch.object(tool, "_prepare_metadata") as prepare:
                with self.assertRaises(TypeError):
                    tool.run_evaluation(**{key: "override"})
                prepare.assert_not_called()
        self.assertFalse(hasattr(tool, "collect_predictions"))
        with patch.object(tool, "_prepare_metadata") as prepare:
            with self.assertRaises(ThresholdError):
                tool.run_evaluation()
            prepare.assert_not_called()
        with patch.object(tool, "_prepare_metadata") as prepare, tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ThresholdError, "root override"):
                tool.run_evaluation(Path(temporary), acknowledged=True)
            prepare.assert_not_called()

    def test_private_inference_rejects_forged_context_before_import_or_access(self):
        for function in (tool._run_standard, tool._run_frozen):
            with self.subTest(function=function.__name__), patch.object(tool, "_prepare_metadata") as prepare:
                with self.assertRaises(ThresholdError):
                    function(SimpleNamespace())
                prepare.assert_not_called()

    def test_best_sha_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            path.write_bytes(b"synthetic")
            sha = tool.shared.digest(path)
            tool.shared.verify_model_identity(path, expected_sha256=sha, expected_size_bytes=9)
            for bad_sha, size in (("0" * 64, 9), (sha, 10)):
                with self.assertRaises(tool.shared.MentorDemoError):
                    tool.shared.verify_model_identity(path, expected_sha256=bad_sha, expected_size_bytes=size)

    def test_validator_settings_are_fixed_not_predict_defaults(self):
        c = tool.load_config()
        ctx = SimpleNamespace(root=ROOT, output=Path("synthetic"))
        s = tool._validator_settings(ctx, c)
        self.assertEqual((s["conf"], s["iou"], s["imgsz"], s["split"], s["device"], s["batch"]), (.001, .5, 640, "test", 0, 1))
        self.assertFalse(s["single_cls"])
        self.assertFalse(s["agnostic_nms"])
        self.assertFalse(s["rect"])
        self.assertIsNone(s["quantize"])
        self.assertFalse(s["augment"])
        self.assertEqual(s["max_det"], 300)

    def test_inference_context_rechecks_identity_config_model_and_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            with bound(root, c, gt) as frozen:
                out, state = tool._start_attempt(root, frozen, False)
                tool._write_once(out / "ground_truth_cache.json", gt)
                capability = object()
                identity_json, config_json = tool.shared.json_bytes(frozen), tool.shared.json_bytes(c)
                tool._ACTIVE_INVENTORY_CAPABILITY = (
                    capability, root.absolute(), out, identity_json, config_json)
                ctx = tool._EvaluationContext(root, out, tool.shared.json_bytes(frozen),
                    tool.shared.json_bytes(c), tool.shared.digest(out / "ground_truth_cache.json"), capability)
                tool._ACTIVE_CONTEXT = ctx
                for mode in ("identity", "config", "model", "attempt"):
                    with self.subTest(mode=mode), ExitStack() as stack:
                        if mode == "identity":
                            stack.enter_context(patch.object(tool, "_prepare_metadata", return_value=(c, [], {**frozen, "git_commit": "changed"})))
                        elif mode == "config":
                            stack.enter_context(patch.object(tool, "_prepare_metadata", return_value=({**c, "confidence": .2}, [], frozen)))
                        elif mode == "model":
                            stack.enter_context(patch.object(tool.shared, "verify_model_identity", side_effect=tool.shared.MentorDemoError("changed model")))
                        else:
                            stack.enter_context(patch.object(tool, "_receipt", return_value={**state, "status": "FAILED_TECHNICAL"}))
                        with self.assertRaises((ThresholdError, tool.shared.MentorDemoError)):
                            tool._run_standard(ctx)
                        with self.assertRaises((ThresholdError, tool.shared.MentorDemoError)):
                            tool._run_frozen(ctx)

    def test_framework_identity_hashes_model_defaults_and_validator_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ("ultralytics/models/yolo/detect/val.py", "ultralytics/nn/tasks.py",
                     "ultralytics/cfg/default.yaml", "ultralytics/data/augment.py")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            distribution = SimpleNamespace(files=[Path(n) for n in (*names, "ultralytics-8.4.130.dist-info/RECORD")],
                                           locate_file=lambda name: root / name)
            with patch.object(tool.shared, "environment", side_effect=lambda: {}), \
                 patch.object(tool.importlib.metadata, "distribution", return_value=distribution):
                first = tool._framework_identity()
                self.assertEqual(set(first["ultralytics_source_sha256"]), set(names))
                (root / "ultralytics/cfg/default.yaml").write_bytes(b"changed same-version defaults")
                self.assertNotEqual(first, tool._framework_identity())
                distribution.files = []
                with self.assertRaises(ThresholdError):
                    tool._framework_identity()

    def test_git_gate_requires_clean_reviewed_descendant_and_all_tooling_tracked(self):
        with patch.object(tool.shared, "git_provenance", side_effect=ThresholdError("dirty")):
            with self.assertRaises(ThresholdError):
                tool.git_provenance(ROOT)
        for mode in ("selection_commit", "not_descendant", "untracked", "approved"):
            with self.subTest(mode=mode), patch.object(tool.shared, "git_provenance", return_value={
                    "evaluation_commit": tool.SELECTION_COMMIT if mode == "selection_commit" else "reviewed-new-commit"}):
                calls = []
                def git(args, **kwargs):
                    calls.append(args)
                    fails = (mode == "not_descendant" and "merge-base" in args
                             or mode == "untracked" and "ls-files" in args)
                    return SimpleNamespace(returncode=int(fails), stdout="", stderr="synthetic Git failure" if fails else "")
                with patch.object(tool.subprocess, "run", git):
                    if mode == "approved":
                        self.assertEqual(tool.git_provenance(ROOT)["evaluation_commit"], "reviewed-new-commit")
                        self.assertEqual(len([a for a in calls if "ls-files" in a]), 5)
                    else:
                        with self.assertRaises(ThresholdError):
                            tool.git_provenance(ROOT)


class MetadataTests(GuardedTest):
    def test_unauthorized_inventory_rejects_before_any_test_filesystem_access(self):
        forbidden = AssertionError("test filesystem access occurred before authorization")
        with ExitStack() as stack:
            operations = [
                stack.enter_context(patch.object(Path, name, side_effect=forbidden))
                for name in ("open", "read_bytes", "read_text", "iterdir", "rglob", "glob", "stat")
            ]
            operations.append(stack.enter_context(patch.object(tool.os, "walk", side_effect=forbidden)))
            operations.append(stack.enter_context(patch.object(tool.shared, "digest", side_effect=forbidden)))
            operations.append(stack.enter_context(patch("PIL.Image.open", side_effect=forbidden)))
            with self.assertRaisesRegex(ThresholdError, "opaque evaluation capability"):
                tool.inventory_internal_test(object(), ROOT, copy.deepcopy(tool.EXPECTED_CONFIG))
            for operation in operations:
                operation.assert_not_called()

    def test_preflight_forbidden_image_label_header_hash_and_enumeration_access(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            c, gt = fixture(root)
            stack.enter_context(patch.object(tool, "load_config", return_value=c))
            stack.enter_context(patch.object(tool, "git_provenance", return_value={"evaluation_commit": "synthetic"}))
            stack.enter_context(patch.object(tool, "selection_provenance", return_value={}))
            stack.enter_context(patch.object(tool, "_framework_identity", return_value={}))
            stack.enter_context(patch.object(tool.shared, "verify_model_identity"))
            stack.enter_context(patch.object(tool, "inventory_internal_test", side_effect=AssertionError("inventory forbidden")))
            stack.enter_context(patch("PIL.Image.open", side_effect=AssertionError("headers forbidden")))
            old_open, old_iter = Path.open, Path.iterdir
            bases = (root / tool.shared.EXPORT / "images", root / tool.shared.EXPORT / "labels")
            def deny(path, *args, **kwargs):
                if any(base == path or base in path.parents for base in bases):
                    raise AssertionError("test image/label bytes forbidden")
                return old_open(path, *args, **kwargs)
            def deny_iter(path):
                if any(base == path or base in path.parents for base in bases):
                    raise AssertionError("test enumeration forbidden")
                return old_iter(path)
            stack.enter_context(patch.object(Path, "open", deny))
            stack.enter_context(patch.object(Path, "iterdir", deny_iter))
            report = tool.preflight(root)
            self.assertTrue(report["metadata_only"])
            self.assertFalse(report["test_images_or_labels_accessed"])
            self.assertFalse(report["inference_executed"])
            self.assertEqual(report["counts"], gt["counts"])
            self.assertIsNone(tool._ACTIVE_INVENTORY_CAPABILITY)
            self.assertFalse((root / tool.OUTPUT).exists())
            self.assertFalse((root / tool.RECEIPT).exists())

    def test_evaluate_mints_capability_only_after_durable_started_for_fresh_and_recovery(self):
        for recovering in (False, True):
            with self.subTest(recovering=recovering), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                c, gt = fixture(root)
                with bound(root, c, gt) as frozen:
                    if recovering:
                        failed, failed_state = tool._start_attempt(root, frozen, False)
                        tool._failure(root, failed, failed_state, {}, OSError("synthetic technical failure"))
                    def inspect_boundary(capability, actual_root, actual_config):
                        receipt = json.loads((root / tool.RECEIPT).read_bytes())
                        current = receipt["attempts"][-1]
                        self.assertEqual(receipt["status"], "STARTED")
                        self.assertTrue((root / current["path"] / "STARTED.json").is_file())
                        self.assertIs(tool._ACTIVE_INVENTORY_CAPABILITY[0], capability)
                        self.assertEqual(tool._ACTIVE_INVENTORY_CAPABILITY[2], root / current["path"])
                        self.assertEqual(actual_root, root)
                        self.assertEqual(actual_config, c)
                        if recovering:
                            self.assertEqual(receipt["attempts"][0]["status"], "FAILED_TECHNICAL")
                        raise ThresholdError("synthetic stop at authorization boundary")
                    with patch.object(tool, "inventory_internal_test", side_effect=inspect_boundary) as inventory, \
                         patch.object(tool, "_run_standard") as standard, patch.object(tool, "_run_frozen") as frozen_branch:
                        with self.assertRaisesRegex(ThresholdError, "authorization boundary"):
                            tool.run_evaluation(root, acknowledged=True, recover=recovering)
                        inventory.assert_called_once()
                        standard.assert_not_called()
                        frozen_branch.assert_not_called()
                    self.assertIsNone(tool._ACTIVE_INVENTORY_CAPABILITY)

    def test_expected_counts_all_independently_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            c, gt = fixture(Path(temporary))
            self.assertEqual(tool.verify_counts(gt["images"], c["expected_counts"]), c["expected_counts"])
            for key in ("images", "positive_images", "negative_images", "targets", "countries", "class_boxes"):
                bad = copy.deepcopy(c["expected_counts"])
                if isinstance(bad[key], dict):
                    bad[key][next(iter(bad[key]))] += 1
                else:
                    bad[key] += 1
                with self.subTest(key=key), self.assertRaises(ThresholdError):
                    tool.verify_counts(gt["images"], bad)

    def test_metadata_hash_and_contamination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, _ = fixture(root)
            path = root / tool.shared.EXPORT / "export_manifest.json"
            original = json.loads(path.read_bytes())
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(ThresholdError):
                tool.approved_metadata(root, c)
            for key, value in (("derived_split", "val"), ("exported_image_path", "images/val/India_000000.jpg"),
                               ("raw_relative_image_path", "data/external/rdd2022/raw/India/test/images/India_000000.jpg"),
                               ("raw_relative_image_path", "teacher/video.avi")):
                changed = copy.deepcopy(original)
                changed["images"][0][key] = value
                path.write_bytes(tool.shared.json_bytes(changed))
                c["approved_metadata_sha256"][path.name] = tool.shared.digest(path)
                with self.subTest(key=key), self.assertRaises(ThresholdError):
                    tool.approved_metadata(root, c)

    def test_integrity_checks_pairing_and_bytes_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            label = root / gt["images"][0]["label_path"]
            label.write_bytes(b"")
            with bound(root, c, gt), patch.object(tool, "_run_standard") as standard, \
                 patch.object(tool, "_run_frozen") as frozen:
                with self.assertRaises(ThresholdError):
                    tool.run_evaluation(root, acknowledged=True)
                standard.assert_not_called()
                frozen.assert_not_called()


class AttemptTests(GuardedTest):
    def test_started_identity_is_durable_before_test_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, _ = fixture(root)
            frozen = identity(root, c)
            out, state = tool._start_attempt(root, frozen, False)
            self.assertEqual(state["status"], "STARTED")
            self.assertEqual(json.loads((root / tool.RECEIPT).read_bytes())["identity"], frozen)
            self.assertEqual(json.loads((root / tool.OUTPUT / "identity.json").read_bytes()), frozen)
            self.assertTrue((out / "STARTED.json").exists())

    def test_technical_failure_and_identical_recovery_preserve_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, _ = fixture(root)
            frozen = identity(root, c)
            out, state = tool._start_attempt(root, frozen, False)
            tool._failure(root, out, state, {}, KeyboardInterrupt())
            self.assertEqual(json.loads((root / tool.RECEIPT).read_bytes())["status"], "FAILED_TECHNICAL")
            new, recovered = tool._start_attempt(root, frozen, True)
            self.assertNotEqual(new, out)
            self.assertTrue((out / "FAILED_TECHNICAL.json").exists())
            self.assertEqual(recovered["attempts"][0]["status"], "FAILED_TECHNICAL")
            self.assertEqual(recovered["attempts"][1]["status"], "STARTED")
            self.assertEqual(recovered["identity"], frozen)

    def test_recovery_rejects_every_scientific_identity_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            frozen = identity(root, c)
            out, state = tool._start_attempt(root, frozen, False)
            tool._failure(root, out, state, {}, OSError("technical"))
            paths = list(identity_leaf_paths(frozen))
            self.assertEqual({path[0] for path in paths}, set(frozen))
            self.assertEqual(set(frozen), tool.RECOVERY_IDENTITY_FIELDS)
            self.assertGreater(len(paths), len(frozen))  # nested values are covered individually
            records = tool.approved_metadata(root, c)
            for path in paths:
                bad = mutate_identity_leaf(frozen, path)
                receipt_before = (root / tool.RECEIPT).read_bytes()
                history_before = {p.relative_to(out).as_posix(): p.read_bytes()
                                  for p in out.rglob("*") if p.is_file()}
                with self.subTest(identity_leaf="/".join(map(str, path))), \
                     patch.object(tool, "ROOT", root), \
                     patch.object(tool, "_prepare_metadata", return_value=(c, records, bad)), \
                     patch.object(tool, "inventory_internal_test") as inventory, \
                     patch.object(tool, "_run_standard") as standard, \
                     patch.object(tool, "_run_frozen") as frozen_branch:
                    with self.assertRaisesRegex(ThresholdError, "identity changed"):
                        tool.run_evaluation(root, acknowledged=True, recover=True)
                    inventory.assert_not_called()
                    standard.assert_not_called()
                    frozen_branch.assert_not_called()
                self.assertEqual((root / tool.RECEIPT).read_bytes(), receipt_before)
                self.assertEqual({p.relative_to(out).as_posix(): p.read_bytes()
                                  for p in out.rglob("*") if p.is_file()}, history_before)

    def test_recovery_requires_explicit_flag_and_existing_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, _ = fixture(root)
            frozen = identity(root, c)
            with self.assertRaises(ThresholdError):
                tool._start_attempt(root, frozen, True)
            tool._start_attempt(root, frozen, False)
            with self.assertRaises(ThresholdError):
                tool._start_attempt(root, frozen, False)

    def test_started_abrupt_exit_recovery_records_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, _ = fixture(root)
            frozen = identity(root, c)
            first, _ = tool._start_attempt(root, frozen, False)
            _, state = tool._start_attempt(root, frozen, True)
            self.assertEqual(state["attempts"][0]["status"], "FAILED_TECHNICAL")
            self.assertTrue((first / "FAILED_TECHNICAL.json").exists())

    def test_integrity_failure_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, _ = fixture(root)
            frozen = identity(root, c)
            out, state = tool._start_attempt(root, frozen, False)
            tool._failure(root, out, state, {}, ThresholdError("changed data"))
            self.assertTrue((out / "INCOMPLETE.json").exists())
            with self.assertRaises(ThresholdError):
                tool._start_attempt(root, frozen, True)

    def test_completed_receipt_and_independent_manifest_each_lock(self):
        for completed_marker in ("receipt", "ledger_attempt", "attempt_manifest", "attempt_marker", "root_manifest"):
            with self.subTest(marker=completed_marker), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                c, _ = fixture(root)
                frozen = identity(root, c)
                out, state = tool._start_attempt(root, frozen, False)
                if completed_marker == "receipt":
                    state["status"] = "COMPLETED"
                    tool._atomic_json(root / tool.RECEIPT, state)
                elif completed_marker == "ledger_attempt":
                    state["attempts"][-1]["status"] = "COMPLETED"
                    tool._atomic_json(root / tool.RECEIPT, state)
                else:
                    path = {"attempt_manifest": out / "evaluation_manifest.json",
                            "attempt_marker": out / "COMPLETED.json",
                            "root_manifest": root / tool.OUTPUT / "evaluation_manifest.json"}[completed_marker]
                    tool._write_once(path, {"status": "COMPLETED"})
                with self.assertRaises(ThresholdError):
                    tool._start_attempt(root, frozen, True)

    def test_bootstrap_interruption_keeps_identity_and_can_recover(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, _ = fixture(root)
            frozen = identity(root, c)
            with patch.object(tool, "_write_once", side_effect=OSError("interrupted before identity copy")):
                with self.assertRaises(OSError):
                    tool._start_attempt(root, frozen, False)
            self.assertEqual(json.loads((root / tool.RECEIPT).read_bytes())["identity"], frozen)
            self.assertFalse((root / tool.OUTPUT / "identity.json").exists())
            out, state = tool._start_attempt(root, frozen, True)
            self.assertEqual(state["attempts"][0]["status"], "FAILED_TECHNICAL")
            self.assertEqual(out.name, "0002")
            self.assertEqual(json.loads((root / tool.OUTPUT / "identity.json").read_bytes()), frozen)

    def test_process_lock_prevents_concurrent_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with tool._process_lock(root):
                with self.assertRaises(ThresholdError):
                    with tool._process_lock(root):
                        pass
            with tool._process_lock(root):
                pass

    def test_existing_unrecognized_output_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, _ = fixture(root)
            (root / tool.OUTPUT).mkdir(parents=True)
            with self.assertRaises(ThresholdError):
                tool._start_attempt(root, identity(root, c), False)


class CacheTests(GuardedTest):
    def _case(self, branch="frozen"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        c, gt = fixture(root)
        path = root / "predictions.jsonl"
        sealed(path, branch, gt, "identity")
        return path, c, gt

    def test_cache_roundtrip_empty_images_and_hash(self):
        for branch in ("standard", "frozen"):
            path, _, gt = self._case(branch)
            rows, sha = reports.verify_cache(path, branch, gt, "identity")
            self.assertEqual(len(rows), 6)
            self.assertEqual(sha, tool.shared.digest(path))
            self.assertEqual(sum(not r["predictions"] for r in rows), 2)

    def test_corrupted_cache_rejected(self):
        path, _, gt = self._case()
        path.write_bytes(path.read_bytes().replace(b"0.8", b"0.7", 1))
        with self.assertRaises(ThresholdError):
            reports.verify_cache(path, "frozen", gt, "identity")

    def test_truncated_cache_rejected(self):
        path, _, gt = self._case()
        path.write_bytes(path.read_bytes()[:-15])
        with self.assertRaises(ThresholdError):
            reports.verify_cache(path, "frozen", gt, "identity")

    def test_duplicate_record_rejected_even_with_resealed_hash(self):
        path, _, gt = self._case()
        rows = [frozen_row(r) for r in gt["images"]]
        rows[-1] = rows[0]
        other = path.parent / "duplicate.jsonl"
        sealed(other, "frozen", gt, "identity", rows)
        with self.assertRaises(ThresholdError):
            reports.verify_cache(other, "frozen", gt, "identity")

    def test_missing_record_rejected(self):
        path, _, gt = self._case()
        other = path.parent / "missing.jsonl"
        sealed(other, "frozen", gt, "identity", [frozen_row(r) for r in gt["images"][:-1]])
        with self.assertRaises(ThresholdError):
            reports.verify_cache(other, "frozen", gt, "identity")

    def test_invalid_schema_class_confidence_geometry_and_identity(self):
        path, _, gt = self._case()
        changes = ("schema", "class", "confidence", "coordinates", "identity", "nms", "country")
        for index, change in enumerate(changes):
            rows = [frozen_row(r) for r in gt["images"]]
            if change == "schema":
                rows[0]["extra"] = 1
            elif change == "class":
                rows[0]["predictions"][0]["class_id"] = 4
            elif change == "confidence":
                rows[0]["predictions"][0]["confidence"] = .18
            elif change == "coordinates":
                rows[0]["predictions"][0]["xyxy"] = [-1., 0., 10., 10.]
            elif change == "identity":
                rows[0]["image_id"] = "unapproved.jpg"
            elif change == "nms":
                rows[0]["nms_iou"] = .6
            else:
                rows[0]["country"] = "Japan"
            other = path.parent / f"invalid{index}.jsonl"
            sealed(other, "frozen", gt, "identity", rows)
            with self.subTest(change=change), self.assertRaises(ThresholdError):
                reports.verify_cache(other, "frozen", gt, "identity")

    def test_native_unclipped_prediction_is_preserved(self):
        path, _, gt = self._case("standard")
        rows = [standard_row(r) for r in gt["images"]]
        rows[0]["predictions"][0]["xyxy"] = [-10., -10., 30., 30.]
        other = path.parent / "unclipped.jsonl"
        sealed(other, "standard", gt, "identity", rows)
        loaded, _ = reports.verify_cache(other, "standard", gt, "identity")
        self.assertEqual(loaded[0]["predictions"][0]["xyxy"], [-10., -10., 30., 30.])

    def test_invalid_native_tp_and_shape_rejected(self):
        path, _, gt = self._case("standard")
        for index, key in enumerate(("native_tp", "original_shape", "ratio_pad")):
            rows = [standard_row(r) for r in gt["images"]]
            rows[0][key] = []
            other = path.parent / f"badnative{index}.jsonl"
            sealed(other, "standard", gt, "identity", rows)
            with self.subTest(key=key), self.assertRaises(ThresholdError):
                reports.verify_cache(other, "standard", gt, "identity")


class NativeMetricTests(GuardedTest):
    def test_multilabel_native_validator_not_prediction_approximation(self):
        import torch
        from ultralytics.utils import nms
        with tempfile.TemporaryDirectory() as temporary:
            v = native_validator(Path(temporary))
            raw = torch.zeros((1, 8, 7))
            raw[0, :4, 0] = torch.tensor([320., 320., 100., 100.])
            raw[0, 4:6, 0] = torch.tensor([.8, .7])
            native = v.postprocess(raw.clone())[0]
            ordinary = nms.non_max_suppression(raw.clone(), .001, .5, multi_label=False)[0]
            self.assertEqual(native["cls"].tolist(), [0., 1.])
            self.assertEqual(len(ordinary), 1)
            self.assertEqual(native["conf"].tolist(), [float(torch.tensor(.8)), float(torch.tensor(.7))])

    def test_native_duplicates_and_confidence_ties(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            v = native_validator(Path(temporary))
            raw = torch.zeros((1, 8, 7))
            raw[0, :4, :2] = torch.tensor([[320., 320.], [320., 320.], [100., 100.], [100., 100.]])
            raw[0, 4, :2] = .8
            a, b = v.postprocess(raw.clone())[0], v.postprocess(raw.clone())[0]
            self.assertEqual(len(a["cls"]), 1)
            self.assertTrue(torch.equal(a["bboxes"], b["bboxes"]))
            duplicate = {"bboxes": torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]),
                         "cls": torch.tensor([0., 0.]), "conf": torch.tensor([.8, .8])}
            gt = {"bboxes": torch.tensor([[0., 0., 10., 10.]]), "cls": torch.tensor([0.])}
            self.assertEqual(v._process_batch(duplicate, gt)["tp"][:, 0].sum(), 1)

    def test_actual_loader_letterbox_and_gt_coordinate_conversion(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _ = fixture(root, size=(40, 20))
            v = native_validator(root)
            dataset = v.build_dataset(str(root / tool.shared.EXPORT / "images/test"), mode="test", batch=1)
            batch = v.preprocess(dataset.collate_fn([dataset[0]]))
            prepared = v._prepare_batch(0, batch)
            self.assertEqual(tuple(batch["img"].shape), (1, 3, 640, 640))
            self.assertTrue(torch.allclose(prepared["bboxes"], torch.tensor([[160., 240., 480., 400.]])))
            self.assertEqual(prepared["ori_shape"], (20, 40))
            self.assertEqual(prepared["ratio_pad"], ((16., 16.), (0, 160)))

    def test_prediction_clipping_does_not_enter_standard_metrics(self):
        import numpy as np
        import torch
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
        raw = torch.tensor([[-10., 0., 10., 10., .8, 0.]])
        result = DetectionPredictor.construct_result(SimpleNamespace(model=SimpleNamespace(names=tool.shared.CLASS_NAMES)), raw.clone(), torch.zeros((1, 3, 20, 20)),
                                                       np.zeros((20, 20, 3), dtype=np.uint8), "synthetic.jpg")
        self.assertEqual(raw[0, 0].item(), -10.)
        self.assertEqual(result.boxes.xyxy[0, 0].item(), 0.)

    def test_native_metric_replay_exact_with_empty_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            v = native_validator(root)
            rows = [standard_row(r, v) for r in gt["images"]]
            a, b = reports.replay_standard(rows, c), reports.replay_standard(rows, c)
            self.assertEqual(tool.shared.json_bytes(a), tool.shared.json_bytes(b))
            self.assertGreater(a["mAP50"], .99)
            self.assertEqual(a["target_boxes"], {"D00": 1, "D10": 1, "D20": 1, "D40": 1})
            self.assertFalse(a["deployment_threshold_selected"])
            for row in rows:
                row["predictions"], row["native_tp"] = [], []
            self.assertEqual(reports.replay_standard(rows, c)["mAP50"], 0.)

    def test_nonperfect_native_metrics_reconcile_from_disk_with_ties_and_false_positives(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            v = native_validator(root)
            rows = []
            for index, r in enumerate(gt["images"]):
                cls = [b["class_id"] for b in r["ground_truth"]]
                boxes = [b["xyxy"] for b in r["ground_truth"]]
                scores = [.8] * len(cls)
                pred_cls = list(cls)
                if index == 0:  # same-score duplicates and a higher-confidence spatial FP
                    boxes += [list(boxes[0]), [0., 0., 3., 3.]]
                    pred_cls += [cls[0], cls[0]]
                    scores += [.8, .95]
                elif index == 1:  # empty predictions on a positive image
                    boxes, pred_cls, scores = [], [], []
                elif index == 5:  # false positive on a negative image
                    boxes, pred_cls, scores = [[0., 0., 3., 3.]], [0], [.95]
                pred = {"bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                        "cls": torch.tensor(pred_cls, dtype=torch.float32),
                        "conf": torch.tensor(scores, dtype=torch.float32)}
                batch = {"im_file": [r["image_id"]], "img": torch.zeros((1, 3, 20, 20)),
                         "batch_idx": torch.zeros(len(cls)), "cls": torch.tensor(cls).reshape(-1, 1),
                         "bboxes": torch.tensor([[.5, .5, .5, .5] for _ in cls]).reshape(-1, 4),
                         "ori_shape": [(20, 20)], "ratio_pad": [((1., 1.), (0, 0))]}
                prepared = v._prepare_batch(0, batch)
                row = standard_row(r)
                row.update({"ground_truth": reports.tensor_boxes(prepared, confidence=False),
                            "predictions": reports.tensor_boxes(pred, confidence=True),
                            "native_tp": v._process_batch(pred, prepared)["tp"].tolist()})
                rows.append(row)
                v.update_metrics([pred], batch)
            v.get_stats()
            live = reports.metric_values(v.metrics)
            path = root / "standard.jsonl"
            sealed(path, "standard", gt, "identity", rows)
            verified, _ = reports.verify_cache(path, "standard", gt, "identity")
            first = reports.replay_standard(verified, c)
            self.assertEqual(tool.shared.json_bytes(first["native_metric_values"]), tool.shared.json_bytes(live))
            self.assertEqual(tool.shared.json_bytes(first), tool.shared.json_bytes(reports.replay_standard(verified, c)))
            self.assertLess(live["precision"], 1.)
            self.assertLess(live["recall"], 1.)

    def test_frozen_negative_metrics_one_to_one_and_inclusive_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, gt = fixture(Path(temporary))
            predictions = predicted(gt)
            predictions[gt["images"][0]["image_id"]].append(Box(1, (5., 5., 15., 15.), .19))
            predictions[gt["images"][2]["image_id"]] = [Box(0, (1., 1., 8., 8.), .19), Box(1, (1., 1., 8., 8.), .18)]
            with patch.object(tool, "sweep_cache", wraps=sweep_cache) as matching:
                p = tool.operating_point(gt, predictions)
                self.assertEqual(matching.call_args.args[2:], (.5, [.19], .5))
            self.assertEqual((p["total_tp"], p["total_fp"], p["total_fn"]), (4, 2, 0))
            self.assertEqual((p["negative_images"], p["negative_fp"], p["negative_images_with_fp"]), (2, 1, 1))
            self.assertEqual((p["negative_fp_per_image"], p["negative_images_with_fp_fraction"]), (.5, .5))

    def test_country_support_and_low_support_warning(self):
        rows = [{"image_id": f"India_{i:06d}.jpg", "country": "India",
                 "ground_truth": [tool.shared.box_record(Box(1, (0., 0., 10., 10.)))]} for i in range(9)]
        rows[0]["ground_truth"].append(tool.shared.box_record(Box(1, (10., 0., 20., 10.))))
        rows.append({"image_id": "Japan_000009.jpg", "country": "Japan", "ground_truth": []})
        countries = tool.country_metrics({"images": rows}, {r["image_id"]: [] for r in rows})
        self.assertIn("LOW SUPPORT: 10 boxes / 9 images", countries["India"]["warnings"][0])
        self.assertEqual(countries["India"]["support"]["class_boxes"]["D10"], 10)
        self.assertEqual(countries["Japan"]["support"]["negative_images"], 1)

    def test_reporting_separates_curve_derived_standard_from_primary_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            standard = reports.replay_standard([standard_row(r) for r in gt["images"]], c)
            predictions = predicted(gt)
            out = root / "reports"
            out.mkdir()
            reports.write_results(out, gt, tool.operating_point(gt, predictions), tool.country_metrics(gt, predictions), standard, c)
            summary = (out / "internal_test_summary.md").read_text()
            self.assertIn("test-curve-derived descriptive statistics", summary)
            self.assertIn("primary frozen deployment operating point", summary)
            self.assertIn("NOT a deployment threshold", summary)
            self.assertIn("standard_precision", (out / "per_class_metrics.csv").read_text())
            self.assertIn("frozen_precision", (out / "per_class_metrics.csv").read_text())


class EndToEndTests(GuardedTest):
    @contextmanager
    def execution(self, root, c, gt):
        import torch
        with bound(root, c, gt) as frozen, ExitStack() as stack:
            native_calls = []
            def no_forward(validator, trainer=None, model=None):
                native_calls.append(model)
                validator.training, validator.device = False, torch.device("cpu")
                validator.data = {"test": str(root / "synthetic")}
                validator.init_metrics(SimpleNamespace(names=tool.shared.CLASS_NAMES, end2end=False))
                for r in gt["images"]:
                    cls = [b["class_id"] for b in r["ground_truth"]]
                    raw = torch.zeros((1, 8, 7))
                    if cls:
                        raw[0, :4, 0] = torch.tensor([320., 320., 320., 320.])
                        raw[0, 4 + cls[0], 0] = .8
                    batch = {"im_file": [str(root / r["image_id"])], "img": torch.zeros((1, 3, 640, 640)),
                             "batch_idx": torch.zeros(len(cls)), "cls": torch.tensor(cls).reshape(-1, 1),
                             "bboxes": torch.tensor([[.5, .5, .5, .5] for _ in cls]).reshape(-1, 4),
                             "ori_shape": [(r["height"], r["width"])], "ratio_pad": [((32., 32.), (0, 0))]}
                    validator.update_metrics(validator.postprocess(raw), batch)
                return validator.get_stats()
            stack.enter_context(patch("ultralytics.engine.validator.BaseValidator.__call__", no_forward))
            fake_model = MagicMock(names=tool.shared.CLASS_NAMES)
            results = []
            for r in gt["images"]:
                cls = [b["class_id"] for b in r["ground_truth"]]
                results.append([SimpleNamespace(boxes=SimpleNamespace(
                    xyxy=torch.tensor([b["xyxy"] for b in r["ground_truth"]]).reshape(-1, 4),
                    conf=torch.tensor([.8] * len(cls)), cls=torch.tensor(cls)))])
            fake_model.predict.side_effect = results
            stack.enter_context(patch("ultralytics.YOLO", return_value=fake_model))
            yield frozen, native_calls, fake_model

    def test_two_branches_actual_validator_metrics_cache_reconciliation_and_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            before = {str(p): tool.shared.digest(p) for p in (root / tool.shared.EXPORT).rglob("*") if p.is_file()}
            with self.execution(root, c, gt) as (_, native_calls, model), \
                 patch.object(tool.shared, "select_operating_point", side_effect=AssertionError("test tuning forbidden")):
                tool.run_evaluation(root, acknowledged=True)
                self.assertEqual(native_calls, [str(root / c["model"])])
                self.assertEqual(model.predict.call_count, 6)
                for call in model.predict.call_args_list:
                    self.assertEqual((call.kwargs["conf"], call.kwargs["iou"]), (.19, .5))
                with self.assertRaises(ThresholdError):
                    tool.run_evaluation(root, acknowledged=True, recover=True)
                self.assertEqual(model.predict.call_count, 6)
            manifest = json.loads((root / tool.OUTPUT / "evaluation_manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "COMPLETED")
            self.assertEqual(manifest["branch_images"], {"standard": 6, "frozen": 6})
            self.assertTrue(manifest["standard_metrics_reproduced_from_persisted_cache"])
            attempt = root / manifest["completed_attempt"]
            for name, sha in manifest["artifacts"].items():
                self.assertEqual(tool.shared.digest(attempt / name), sha)
            self.assertEqual(before, {str(p): tool.shared.digest(p) for p in (root / tool.shared.EXPORT).rglob("*") if p.is_file()})
            self.assertEqual(json.loads((root / tool.RECEIPT).read_bytes())["status"], "COMPLETED")

    def test_report_write_failure_cannot_complete_and_identity_recovery_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            with self.execution(root, c, gt), patch.object(reports, "write_results", side_effect=OSError("synthetic disk failure")):
                with self.assertRaises(OSError):
                    tool.run_evaluation(root, acknowledged=True)
            self.assertFalse((root / tool.OUTPUT / "evaluation_manifest.json").exists())
            self.assertEqual(json.loads((root / tool.RECEIPT).read_bytes())["status"], "FAILED_TECHNICAL")
            with self.execution(root, c, gt):
                tool.run_evaluation(root, acknowledged=True, recover=True)
            receipt = json.loads((root / tool.RECEIPT).read_bytes())
            self.assertEqual(receipt["status"], "COMPLETED")
            self.assertEqual(receipt["attempts"][0]["status"], "FAILED_TECHNICAL")
            self.assertEqual(len(receipt["attempts"]), 2)

    def test_report_flush_failure_cannot_publish_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            with self.execution(root, c, gt), patch.object(tool, "_durable_artifact_hash", side_effect=OSError("fsync failure")):
                with self.assertRaises(OSError):
                    tool.run_evaluation(root, acknowledged=True)
            self.assertFalse((root / tool.OUTPUT / "evaluation_manifest.json").exists())
            receipt = json.loads((root / tool.RECEIPT).read_bytes())
            self.assertEqual(receipt["status"], "FAILED_TECHNICAL")
            self.assertFalse((root / receipt["attempts"][0]["path"] / "COMPLETED.json").exists())

    def test_publication_failure_after_completed_attempt_never_unlocks_or_downgrades(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            atomic = tool._atomic_json
            def fail_final_publication(path, value):
                if path == root / tool.OUTPUT / "evaluation_manifest.json":
                    raise OSError("final publication disk failure")
                return atomic(path, value)
            with self.execution(root, c, gt) as (_, calls, model), patch.object(tool, "_atomic_json", fail_final_publication):
                with self.assertRaises(OSError):
                    tool.run_evaluation(root, acknowledged=True)
                receipt = json.loads((root / tool.RECEIPT).read_bytes())
                attempt = root / receipt["attempts"][0]["path"]
                self.assertEqual(json.loads((attempt / "evaluation_manifest.json").read_bytes())["status"], "COMPLETED")
                self.assertTrue((attempt / "COMPLETED.json").exists())
                self.assertFalse((attempt / "FAILED_TECHNICAL.json").exists())
                with self.assertRaises(ThresholdError):
                    tool.run_evaluation(root, acknowledged=True, recover=True)
                self.assertEqual(len(calls), 1)
                self.assertEqual(model.predict.call_count, 6)

    def test_cache_corruption_blocks_finalization_before_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            real = tool._run_frozen
            def corrupt(context):
                real(context)
                path = context.output / "frozen_predictions.jsonl"
                path.write_bytes(path.read_bytes()[:-10])
            with self.execution(root, c, gt), patch.object(tool, "_run_frozen", corrupt), patch.object(reports, "write_results") as write:
                with self.assertRaises(ThresholdError):
                    tool.run_evaluation(root, acknowledged=True)
                write.assert_not_called()
            self.assertFalse((root / tool.OUTPUT / "evaluation_manifest.json").exists())

    def test_missing_record_and_live_metric_mismatch_block_completion(self):
        for mode in ("missing", "live_mismatch"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                c, gt = fixture(root)
                real = tool._run_standard
                def change(context):
                    real(context)
                    if mode == "missing":
                        path = context.output / "standard_predictions.jsonl"
                        path.write_bytes(b"\n".join(path.read_bytes().splitlines()[:-1]) + b"\n")
                    else:
                        tool._atomic_json(context.output / "standard_live_metrics.json", {"fake": True})
                with self.execution(root, c, gt), patch.object(tool, "_run_standard", change):
                    with self.assertRaises(ThresholdError):
                        tool.run_evaluation(root, acknowledged=True)
                self.assertFalse((root / tool.OUTPUT / "evaluation_manifest.json").exists())

    def test_integrity_failure_prevents_any_branch_and_started_already_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            def failed_inventory(*args):
                self.assertEqual(json.loads((root / tool.RECEIPT).read_bytes())["status"], "STARTED")
                raise ThresholdError("synthetic integrity mismatch")
            with bound(root, c, gt), patch.object(tool, "inventory_internal_test", failed_inventory), \
                 patch.object(tool, "_run_standard") as a, patch.object(tool, "_run_frozen") as b:
                with self.assertRaises(ThresholdError):
                    tool.run_evaluation(root, acknowledged=True)
                a.assert_not_called()
                b.assert_not_called()

    def test_technical_interrupt_during_inference_preserves_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, gt = fixture(root)
            with bound(root, c, gt), patch.object(tool, "_run_standard", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    tool.run_evaluation(root, acknowledged=True)
            receipt = json.loads((root / tool.RECEIPT).read_bytes())
            self.assertEqual(receipt["status"], "FAILED_TECHNICAL")
            self.assertTrue((root / receipt["attempts"][0]["path"] / "FAILED_TECHNICAL.json").exists())


if __name__ == "__main__":
    unittest.main()
