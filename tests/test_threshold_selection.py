"""Synthetic evaluation tests: never run a real model or open project data."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_damage.evaluation import select_threshold as tool  # noqa: E402
from road_damage.evaluation.threshold_metrics import (  # noqa: E402
    Box, ThresholdError, iou, match_image, metrics_row, select_operating_point, sweep_cache,
)
from road_damage.demo.mentor_demo import MentorDemoError  # noqa: E402


def box(c: int = 0, confidence: float = 1.0, xyxy: tuple = (0., 0., 10., 10.)) -> Box:
    return Box(c, xyxy, confidence)


def fixture(root: Path) -> tuple[dict, dict]:
    """Five tiny generated JPEGs: one GT per class and one negative."""
    from PIL import Image
    config = copy.deepcopy(tool.EXPECTED_CONFIG)
    config["expected_counts"] = {"images": 5, "targets": 4, "negative_images": 1}
    images = root / tool.EXPORT / "images/val"
    labels = root / tool.EXPORT / "labels/val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    yaml_path = root / tool.DATA_YAML
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(
        f'train: "{(root / tool.EXPORT / "images/train").as_posix()}"\n'
        f'val: "{images.as_posix()}"\nnames:\n' +
        "".join(f"  {c}: {name}\n" for c, name in tool.CLASS_NAMES.items()), encoding="utf-8")
    config_path = root / tool.CONFIG
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(tool.json_bytes(config))
    for index in range(5):
        name = f"India_{index:06d}"
        Image.new("RGB", (20, 20), color=(index, 0, 0)).save(images / f"{name}.jpg")
        (labels / f"{name}.txt").write_bytes(
            f"{index} 0.25 0.25 0.5 0.5\n".encode() if index < 4 else b"")
    return config, tool.inventory_validation(root, config)


class MetricTests(unittest.TestCase):
    def test_iou_continuous_overlap_disjoint_and_boundary(self) -> None:
        self.assertEqual(iou(box(), box()), 1)
        self.assertEqual(iou(box(), box(xyxy=(10., 0., 20., 10.))), 0)
        self.assertAlmostEqual(iou(box(), box(xyxy=(5., 0., 15., 10.))), 1 / 3)
        self.assertEqual(iou(box(), box(xyxy=(0., 0., 5., 10.))), .5)

    def test_invalid_boxes_and_class_ids_rejected(self) -> None:
        for coords in ((0, 0, 0, 10), (0, 0, float("nan"), 10), (-1, 0, 10, 10)):
            with self.assertRaises(ThresholdError):
                box(xyxy=coords)
        with self.assertRaises(ThresholdError):
            box(4)

    def test_one_to_one_class_aware_duplicates_and_iou_inclusive(self) -> None:
        predictions = [box(0, .9), box(0, .8), box(1, .7), box(2, .6, (0., 0., 5., 10.))]
        matches = match_image([box(0), box(2)], predictions)
        self.assertEqual([tp for _, tp in matches], [True, False, False, True])
        rows = sweep_cache({"i": [box(0), box(2)]}, {"i": predictions}, .5, [.01])
        r = rows[0]
        self.assertEqual((r["total_tp"], r["total_fp"], r["total_fn"]), (2, 2, 0))
        self.assertEqual((r["D00_precision"], r["D00_recall"]), (.5, 1))
        self.assertAlmostEqual(r["D00_f1"], 2 / 3)
        self.assertAlmostEqual(r["macro_f1"], (2 / 3 + 1) / 4)

    def test_fn_and_zero_denominators_include_absent_classes(self) -> None:
        r = sweep_cache({"i": [box()]}, {"i": []}, .5, [.01])[0]
        self.assertEqual(r["total_fn"], 1)
        self.assertEqual(r["macro_f1"], 0)
        self.assertEqual(r["D10_precision"], 0)
        self.assertEqual(r["D00_recall"], 0)

    def test_negative_image_metrics_use_all_negative_images_as_denominator(self) -> None:
        r = sweep_cache({"a": [], "b": []}, {"a": [box(0, .5), box(1, .6)], "b": []}, .5, [.5])[0]
        self.assertEqual(r["negative_images"], 2)
        self.assertEqual(r["negative_predictions"], 2)
        self.assertEqual(r["negative_fp"], 2)
        self.assertEqual(r["negative_fp_per_image"], 1)
        self.assertEqual(r["negative_images_with_fp_fraction"], .5)

    def test_matching_ties_use_coordinates_and_highest_iou(self) -> None:
        truth = [box(xyxy=(0., 0., 10., 10.)), box(xyxy=(2., 0., 12., 10.))]
        predictions = [box(0, .8, (1., 0., 11., 10.)), box(0, .8, (2., 0., 12., 10.))]
        first = match_image(truth, predictions)
        second = match_image(list(reversed(truth)), list(reversed(predictions)))
        self.assertEqual(first, second)
        self.assertEqual([matched for _, matched in first], [True, True])

    def test_cached_prefix_matching_equals_per_threshold_matching(self) -> None:
        gt = [box(), box(1)]
        pred = [box(0, .01), box(0, .5), box(0, .8), box(1, .6), box(2, .7)]
        for r in sweep_cache({"i": gt}, {"i": pred}, .5, [.01, .5, .6, .8]):
            fresh = match_image(gt, [b for b in pred if b.confidence >= r["confidence"]])
            self.assertEqual(r["total_tp"], sum(tp for _, tp in fresh))
            self.assertEqual(r["total_fp"], sum(not tp for _, tp in fresh))

    def test_tie_break_fp_before_confidence_then_nms_order(self) -> None:
        make = lambda nms, conf, fp: metrics_row([[1, 0, 0]] * 4, nms, conf, 2, fp, int(fp > 0))
        rows = [make(.5, .8, 2), make(.6, .5, 0), make(.7, .6, 0), make(.5, .6, 0)]
        selected = select_operating_point(rows, [.5, .6, .7])
        self.assertEqual(selected["metrics"]["nms_iou"], .5)
        self.assertEqual(selected["metrics"]["confidence"], .6)
        self.assertTrue(selected["tie_break_reasoning"]["fallback_used"])
        self.assertEqual(selected, select_operating_point(list(reversed(rows)), [.5, .6, .7]))

    def test_macro_f1_precedes_negative_fp_tie_break(self) -> None:
        worse = metrics_row([[0, 0, 1]] * 4, .5, .8, 1, 0, 0)
        better = metrics_row([[1, 0, 0]] * 4, .6, .1, 1, 9, 1)
        self.assertEqual(select_operating_point([worse, better], [.5, .6, .7])["metrics"], better)


class SafetyAndCacheTests(unittest.TestCase):
    def test_config_literal_policy_and_per_class_override_rejection(self) -> None:
        config = tool.load_config()
        self.assertEqual(config["nms_iou_candidates"], [.5, .6, .7])
        self.assertEqual(config["confidence_grid"], {"start_percent": 1, "stop_percent": 80, "step_percent": 1})
        self.assertEqual(config["expected_counts"], {"images": 2602, "targets": 3377, "negative_images": 983})
        self.assertEqual(config["model_sha256"], "BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7")
        for changes in ({"per_class_thresholds": {"0": .3}}, {"model": "weights/last.pt"},
                        {"inference_confidence_floor": .25}, {"augment": True}):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / tool.CONFIG
                path.parent.mkdir(parents=True)
                path.write_bytes(tool.json_bytes({**config, **changes}))
                with self.assertRaises(ThresholdError):
                    tool.load_config(root)

    def test_prediction_floor_and_explicit_inference_policy(self) -> None:
        for nms in (.5, .6, .7):
            args = tool.prediction_kwargs(tool.load_config(), nms)
            self.assertEqual((args["conf"], args["iou"], args["imgsz"], args["device"]), (.01, nms, 640, 0))
            for flag in ("augment", "half", "save", "save_txt", "save_crop", "show", "agnostic_nms"):
                self.assertFalse(args[flag])
            self.assertNotIn("data", args)

    def test_best_only_allowlist_rejects_before_filesystem_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path in ("weights/last.pt", "weights/yolo26s.pt", "teacher/frames", "data/images/test", "data/official_test"):
                with patch.object(Path, "is_symlink", side_effect=AssertionError("must reject first")):
                    with self.assertRaises(ThresholdError):
                        tool.checked_path(Path(path), tool.FROZEN_MODEL_RELATIVE_PATH, root)

    def test_model_sha_and_size_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            path.write_bytes(b"trusted synthetic checkpoint")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            tool.verify_model_identity(path, expected_sha256=sha, expected_size_bytes=path.stat().st_size)
            with self.assertRaises(MentorDemoError):
                tool.verify_model_identity(path, expected_sha256="0" * 64, expected_size_bytes=path.stat().st_size)
            with self.assertRaises(MentorDemoError):
                tool.verify_model_identity(path, expected_sha256=sha, expected_size_bytes=1)

    def test_inventory_tallies_only_validation_and_exact_empty_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, gt = fixture(root)
            self.assertEqual(gt["counts"], {"images": 5, "targets": 4, "negative_images": 1})
            self.assertFalse((root / tool.EXPORT / "images/train").exists())
            self.assertFalse((root / tool.EXPORT / "images/test").exists())
            (root / tool.EXPORT / "labels/val/India_000004.txt").write_bytes(b"\n")
            with self.assertRaises(ThresholdError):
                tool.inventory_validation(root, config)

    def test_yaml_rejects_test_teacher_mapping_and_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = fixture(root)
            path = root / tool.DATA_YAML
            original = path.read_text()
            cases = [original + "test: forbidden\n", original.replace("images/val", "images/test"),
                     original.replace("images/val", "teacher_video"), original.replace("D40_pothole", "other"),
                     original + "val: forbidden\n"]
            for text in cases:
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ThresholdError):
                    tool.validation_paths(root, config)

    def test_output_refuses_overwrite_and_preflight_probe_leaves_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool.ensure_output_available(root / "new/results")
            self.assertEqual(list(root.iterdir()), [])
            with self.assertRaises(ThresholdError):
                tool.ensure_output_available(root)

    def test_cuda_unavailable_fails_without_loading_yolo(self) -> None:
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": torch}), patch.object(tool.platform, "python_version", return_value="3.12.10"), \
                patch.object(tool.importlib.metadata, "version", return_value="8.4.130"):
            with self.assertRaisesRegex(ThresholdError, "CUDA"):
                tool.environment()

    def test_preflight_never_constructs_model_or_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, gt = fixture(root)
            with patch.object(tool, "load_config", return_value=config), \
                    patch.object(tool, "verify_model_identity") as verify, \
                    patch.object(tool, "environment", return_value={"device": 0}), \
                    patch.object(tool, "git_provenance", return_value={"evaluation_commit": "new"}), \
                    patch.dict(sys.modules, {"ultralytics": None}):
                _, actual_gt, report = tool.preflight(root)
            self.assertEqual(actual_gt, gt)
            self.assertTrue(report["passed"])
            self.assertFalse(report["inference_executed"])
            verify.assert_called_once()
            self.assertFalse((root / tool.OUTPUT).exists())

    def test_dirty_or_pre_tooling_git_commit_fails(self) -> None:
        def fake_run(args, **kwargs):
            if "rev-parse" in args:
                return SimpleNamespace(returncode=0, stdout=tool.MENTOR_COMMIT, stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(tool.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(ThresholdError, "commit"):
                tool.git_provenance(ROOT)

    def test_cache_coverage_floor_and_forbidden_gt_source_rejected(self) -> None:
        with self.assertRaises(ThresholdError):
            sweep_cache({"one": []}, {}, .5, [.01])
        bad = {"validation_only": True, "nms_iou": .5, "confidence_floor": .25, "images": []}
        with self.assertRaises(ThresholdError):
            tool.decoded_predictions(bad, .5)
        with tempfile.TemporaryDirectory() as temporary:
            _, gt = fixture(Path(temporary))
            gt["images"][0]["image_path"] = "teacher_video/India_000000.jpg"
            with self.assertRaises(ThresholdError):
                tool.decoded_ground_truth(gt)

    def test_mocked_collection_full_sweep_and_byte_identical_replay(self) -> None:
        # Load compiled dependencies outside patch.dict's module-cache restoration.
        importlib.import_module("cv2")
        importlib.import_module("matplotlib.pyplot")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, gt = fixture(root)
            model = MagicMock()
            model.names = tool.CLASS_NAMES
            def tensor(values):
                return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))

            def synthetic_predict(**kwargs):
                index = (model.predict.call_count - 1) % 5
                prediction = SimpleNamespace(
                    xyxy=tensor([[0., 0., 10., 10.]]),
                    conf=tensor([.4 if index < 4 else .2]), cls=tensor([index if index < 4 else 0]))
                return [SimpleNamespace(boxes=prediction)]

            model.predict.side_effect = synthetic_predict
            factory = MagicMock(return_value=model)
            provenance = {"evaluation_commit": "synthetic-evaluation-commit"}
            report = {"provenance": provenance, "environment": {"synthetic": True},
                      "validation_yaml_sha256": "synthetic", "evaluation_config_sha256": "synthetic",
                      "ground_truth_snapshot_sha256": hashlib.sha256(tool.json_bytes(gt)).hexdigest()}
            with patch.object(tool, "preflight", return_value=(config, gt, report)), \
                    patch.object(tool, "git_provenance", return_value=provenance), \
                    patch.object(tool, "verify_model_identity"), \
                    patch.dict(sys.modules, {"torch": MagicMock(), "ultralytics": SimpleNamespace(YOLO=factory)}):
                tool.run_sweep(root)
            self.assertEqual(model.predict.call_count, 15)
            self.assertTrue(all(call.kwargs["conf"] == .01 for call in model.predict.call_args_list))
            output = root / tool.OUTPUT
            self.assertEqual(len((output / "threshold_sweep.csv").read_text().splitlines()), 241)
            original = (output / "selected_operating_point.json").read_bytes()
            selected = json.loads(original)
            self.assertEqual(selected["selected_global_confidence"], .4)
            self.assertEqual(selected["selected_nms_iou"], .5)
            self.assertEqual(selected["macro_f1"], 1)
            self.assertEqual(selected["negative_image_metrics"]["negative_fp"], 0)
            with patch.object(tool, "load_config", return_value=config), \
                    patch.object(tool, "inventory_validation", side_effect=AssertionError("no dataset access")), \
                    patch.dict(sys.modules, {"ultralytics": None, "torch": None}):
                tool.recompute(root)
            replay = root / config["recompute_output"]
            self.assertEqual(original, (replay / "selected_operating_point.json").read_bytes())
            self.assertEqual((output / "threshold_sweep.csv").read_bytes(), (replay / "threshold_sweep.csv").read_bytes())
            for name in ("threshold_selection_summary.md", "macro_f1_vs_confidence.png"):
                self.assertTrue((output / name).is_file())
            # Corruption must fail before any replay output is created or rewritten.
            (output / "predictions_nms_50.json").write_bytes(b"{}")
            with patch.object(tool, "load_config", return_value=config):
                with self.assertRaisesRegex(ThresholdError, "SHA-256"):
                    tool.recompute(root)


if __name__ == "__main__":
    unittest.main()
