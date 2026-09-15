"""Synthetic tests for Experiment 2 validation-only threshold selection."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_damage.evaluation import select_experiment2_threshold as tool  # noqa: E402
from road_damage.evaluation.threshold_metrics import ThresholdError  # noqa: E402


def _gt_and_cache(config: dict) -> tuple[dict, dict]:
    records = []
    predictions = []
    for class_id in range(4):
        name = f"India_{class_id:06d}.jpg"
        box = {"class_id": class_id, "confidence": 1.0, "xyxy": [0.0, 0.0, 10.0, 10.0]}
        records.append({
            "image_id": name, "width": 20, "height": 20,
            "image_path": (tool.EXPORT / "images/val" / name).as_posix(),
            "label_path": (tool.EXPORT / "labels/val" / f"India_{class_id:06d}.txt").as_posix(),
            "image_sha256": str(class_id), "label_sha256": str(class_id), "ground_truth": [box],
        })
        predictions.append({
            "image_id": name,
            "predictions": [{"class_id": class_id, "confidence": 0.4, "xyxy": [0.0, 0.0, 10.0, 10.0]}],
        })
    negative = "Japan_000004.jpg"
    records.append({
        "image_id": negative, "width": 20, "height": 20,
        "image_path": (tool.EXPORT / "images/val" / negative).as_posix(),
        "label_path": (tool.EXPORT / "labels/val/Japan_000004.txt").as_posix(),
        "image_sha256": "4", "label_sha256": "4", "ground_truth": [],
    })
    predictions.append({
        "image_id": negative,
        "predictions": [{"class_id": 0, "confidence": 0.2, "xyxy": [1.0, 1.0, 5.0, 5.0]}],
    })
    counts = {"images": 5, "positive_images": 4, "negative_images": 1, "targets": 4,
              "targets_by_class": {str(index): 1 for index in range(4)}}
    config["expected_validation"] = counts
    config["confidence_grid_milli"] = {"start": 100, "stop": 400, "step": 100}
    ground_truth = {"schema_version": "experiment2_threshold_ground_truth.v1", "validation_only": True,
                    "count_source": "synthetic", "counts": counts, "images": records}
    cache = {
        "schema_version": "experiment2_yolo26s.native_predictions.v1", "validation_only": True,
        "internal_test_files_accessed": False, "model_sha256": config["model_sha256"],
        "confidence_floor": 0.01,
        "postprocessing": {"mode": config["native_postprocessing"]["mode"], "prediction_branch": "one2one",
                           "nms_applicable": False, "nms_iou_swept": False,
                           "iou_argument_supplied": False, "max_det": 300},
        "images": predictions,
    }
    return ground_truth, cache


class FrozenPolicyTests(unittest.TestCase):
    def test_literal_config_checkpoint_grid_and_global_policy(self) -> None:
        config = tool.load_config()
        self.assertEqual(config["model_sha256"],
                         "99c03d56f4b27d6a9cc774dd3880f11807642058934199ed675dc3bf4b9f37a1")
        self.assertEqual(config["model_size_bytes"], 20_320_709)
        self.assertEqual(config["best_epoch"], 63)
        self.assertEqual(config["confidence_grid_milli"], {"start": 10, "stop": 800, "step": 1})
        self.assertTrue(config["global_threshold_only"])
        self.assertFalse(config["native_postprocessing"]["nms_applicable"])
        self.assertFalse(config["native_postprocessing"]["nms_iou_swept"])
        self.assertNotIn("per_class_thresholds", config)
        self.assertNotIn("nms_iou_candidates", config)

    def test_config_rejects_overrides_and_per_class_thresholds(self) -> None:
        original = tool.load_config()
        mutations = (
            {"per_class_thresholds": {"0": 0.2}}, {"model": "weights/last.pt"},
            {"matching_iou": 0.6}, {"nms_iou_candidates": [0.5]},
            {"inference_confidence_floor_milli": 250}, {"global_threshold_only": False},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / tool.CONFIG
                path.parent.mkdir(parents=True)
                path.write_bytes(tool.shared.json_bytes({**original, **mutation}))
                with self.assertRaises(ThresholdError):
                    tool.load_config(root)

    def test_predictor_kwargs_preserve_native_end2end_without_iou(self) -> None:
        kwargs = tool.prediction_kwargs(tool.load_config())
        self.assertEqual((kwargs["conf"], kwargs["imgsz"], kwargs["batch"]), (0.01, 640, 4))
        self.assertNotIn("iou", kwargs)
        self.assertTrue(kwargs["stream"])
        for name in ("augment", "half", "agnostic_nms", "save", "save_txt", "show", "compile"):
            self.assertFalse(kwargs[name])

    def test_installed_end2end_source_semantics_are_pinned(self) -> None:
        report = tool.verify_ultralytics_semantics()
        self.assertEqual(report["mode"], "ultralytics_8.4.130_native_end2end_nms_free")
        self.assertFalse(report["nms_applicable"])
        self.assertFalse(report["nms_iou_swept"])
        self.assertFalse(report["iou_argument_supplied"])
        self.assertTrue(all(report["source_evidence"].values()))
        self.assertEqual({key: value["sha256"] for key, value in report["source_files"].items()},
                         tool.ULTRALYTICS_SOURCE_HASHES)

    def test_exact_architecture_and_class_order_enforced(self) -> None:
        config = tool.load_config()
        head = type("Detect", (), {"nc": 4, "end2end": True, "reg_max": 1, "max_det": 300})()
        model = SimpleNamespace(model=SimpleNamespace(model=[head]), names=tool.CLASS_NAMES)
        self.assertTrue(tool.verify_runtime_architecture(model, config)["end2end"])
        for mutation in ({"end2end": False}, {"reg_max": 16}, {"nc": 80}, {"max_det": 100}):
            bad_head = type("Detect", (), {**{"nc": 4, "end2end": True, "reg_max": 1, "max_det": 300},
                                            **mutation})()
            with self.assertRaises(ThresholdError):
                tool.verify_runtime_architecture(
                    SimpleNamespace(model=SimpleNamespace(model=[bad_head]), names=tool.CLASS_NAMES), config
                )
        with self.assertRaises(ThresholdError):
            tool.verify_runtime_architecture(
                SimpleNamespace(model=SimpleNamespace(model=[head]), names={0: "wrong"}), config
            )

    def test_checkpoint_sha_size_are_independently_pinned(self) -> None:
        config = tool.load_config()
        self.assertEqual(config["model"], tool.MODEL.as_posix())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            path.write_bytes(b"synthetic checkpoint")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            tool.shared.verify_model_identity(
                path, expected_sha256=sha, expected_size_bytes=path.stat().st_size
            )
            with self.assertRaises(tool.shared.MentorDemoError):
                tool.shared.verify_model_identity(
                    path, expected_sha256=tool.EXPECTED_MODEL_SHA256,
                    expected_size_bytes=path.stat().st_size,
                )


class ValidationSafetyTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        from PIL import Image

        config = copy.deepcopy(tool.EXPECTED_CONFIG)
        images = root / tool.EXPORT / "images/val"
        labels = root / tool.EXPORT / "labels/val"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        yaml_path = root / tool.DATA_YAML
        yaml_path.parent.mkdir(parents=True)
        yaml_path.write_text(
            f'train: "{(root / tool.EXPORT / "images/train").as_posix()}"\n'
            f'val: "{images.as_posix()}"\nnames:\n'
            + "".join(f"  {key}: {value}\n" for key, value in tool.CLASS_NAMES.items()), encoding="utf-8"
        )
        for index in range(5):
            stem = ("India" if index < 4 else "Japan") + f"_{index:06d}"
            Image.new("RGB", (20, 20), color=(index, 0, 0)).save(images / f"{stem}.jpg")
            (labels / f"{stem}.txt").write_bytes(
                f"{index} 0.25 0.25 0.5 0.5\n".encode() if index < 4 else b""
            )
        config["expected_validation"] = {
            "images": 5, "positive_images": 4, "negative_images": 1, "targets": 4,
            "targets_by_class": {str(index): 1 for index in range(4)},
        }
        return config

    def test_validation_support_positive_negative_and_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            inventory = tool.inventory_validation(root, config)
            self.assertEqual(inventory["counts"], config["expected_validation"])
            self.assertFalse((root / tool.EXPORT / "images/train").exists())
            self.assertFalse((root / tool.EXPORT / "images/test").exists())

    def test_validation_yaml_rejects_internal_test_teacher_and_wrong_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            path = root / tool.DATA_YAML
            original = path.read_text(encoding="utf-8")
            for text in (
                original + "test: forbidden\n", original.replace("images/val", "images/test"),
                original.replace("images/val", "teacher_video"),
                original.replace("D00_longitudinal_crack", "D10_transverse_crack", 1),
            ):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ThresholdError):
                    tool.shared.validation_paths(root, config)

    def test_prediction_cache_rejects_nonvalidation_or_nms_policy(self) -> None:
        config = copy.deepcopy(tool.EXPECTED_CONFIG)
        _, cache = _gt_and_cache(config)
        for mutation in (
            {"validation_only": False}, {"internal_test_files_accessed": True},
            {"postprocessing": {**cache["postprocessing"], "nms_applicable": True}},
            {"postprocessing": {**cache["postprocessing"], "iou_argument_supplied": True}},
        ):
            with self.assertRaises(ThresholdError):
                tool.decoded_predictions({**cache, **mutation}, config)

    def test_collect_predictions_batches_once_per_image_and_never_supplies_iou(self) -> None:
        config = copy.deepcopy(tool.EXPECTED_CONFIG)
        ground_truth, _ = _gt_and_cache(config)

        def tensor(values):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))

        model = MagicMock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The explicit val paths need not be opened by this function, but their ancestors are checked.
            (root / tool.EXPORT / "images/val").mkdir(parents=True)
            results = []
            for row in ground_truth["images"]:
                boxes = SimpleNamespace(xyxy=tensor([[0.0, 0.0, 10.0, 10.0]]),
                                        conf=tensor([0.4]), cls=tensor([0.0]))
                results.append(SimpleNamespace(path=str(root / row["image_path"]), orig_shape=(20, 20), boxes=boxes))
            by_path = {item.path: item for item in results}
            model.predict.side_effect = lambda source, **kwargs: iter(by_path[path] for path in source)
            cache, _ = tool.collect_predictions(model, ground_truth, config, root)
        self.assertEqual(model.predict.call_count, 2)
        self.assertTrue(all("iou" not in call.kwargs for call in model.predict.call_args_list))
        self.assertEqual(len(cache["images"]), 5)
        self.assertFalse(cache["postprocessing"]["nms_applicable"])


class MetricAndReceiptTests(unittest.TestCase):
    def test_deterministic_metrics_global_threshold_and_negative_behavior(self) -> None:
        config = copy.deepcopy(tool.EXPECTED_CONFIG)
        ground_truth, cache = _gt_and_cache(config)
        rows, selected = tool.calculate_artifacts(ground_truth, cache, config, {"id": "synthetic"})
        reversed_cache = {**cache, "images": list(reversed(cache["images"]))}
        rows_again, selected_again = tool.calculate_artifacts(
            ground_truth, reversed_cache, config, {"id": "synthetic"}
        )
        self.assertEqual(rows, rows_again)
        self.assertEqual(selected, selected_again)
        self.assertEqual(selected["selected_global_confidence"], 0.4)
        self.assertEqual(selected["macro_metrics"], {"precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(selected["micro_metrics"], {"precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(selected["counts"], {"total_tp": 4, "total_fp": 0, "total_fn": 0})
        self.assertEqual(selected["negative_image_metrics"]["negative_fp"], 0)
        self.assertIsNone(selected["per_class_thresholds"])
        self.assertIsNone(selected["nms"]["iou"])
        self.assertTrue(all(row["nms_applicable"] is False and "nms_iou" not in row for row in rows))

    def test_tie_break_macro_then_negative_fp_then_higher_confidence(self) -> None:
        def row(confidence: float, exact: str, negative_fp: int) -> dict:
            return {"confidence": confidence, "macro_f1_exact": exact, "negative_fp": negative_fp,
                    "negative_fp_per_image": float(negative_fp), "nms_applicable": False}

        chosen = tool.select_global_threshold([
            row(0.5, "1/2", 0), row(0.7, "2/3", 2), row(0.6, "2/3", 0), row(0.8, "2/3", 0)
        ])
        self.assertEqual(chosen["metrics"]["confidence"], 0.8)
        self.assertEqual(chosen["tie_break_reasoning"]["points_after_macro_f1"], 3)
        self.assertEqual(chosen["tie_break_reasoning"]["points_after_negative_fp"], 2)
        with self.assertRaises(ThresholdError):
            tool.select_global_threshold([{**row(0.5, "1", 0), "nms_applicable": True}])

    def test_persisted_cache_and_outputs_must_reproduce_before_completion(self) -> None:
        config = copy.deepcopy(tool.EXPECTED_CONFIG)
        ground_truth, cache = _gt_and_cache(config)
        identity = {"id": "synthetic"}
        rows, selected = tool.calculate_artifacts(ground_truth, cache, config, identity)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "ground_truth_cache.json").write_bytes(tool.shared.json_bytes(ground_truth))
            (output / "prediction_cache.json").write_bytes(tool.shared.json_bytes(cache))
            tool.write_scientific_results(output, rows, selected)
            names = ("ground_truth_cache.json", "prediction_cache.json", "threshold_sweep.json",
                     "selected_operating_point.json")
            manifest = {"evaluation_identity": identity,
                        "artifacts": {name: tool.shared.digest(output / name) for name in names}}
            tool._verify_persisted_results(output, config, manifest)
            (output / "prediction_cache.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ThresholdError, "SHA-256"):
                tool._verify_persisted_results(output, config, manifest)

    def test_completion_receipt_cache_corruption_blocks_offline_replay(self) -> None:
        config = copy.deepcopy(tool.EXPECTED_CONFIG)
        ground_truth, cache = _gt_and_cache(config)
        rows, selected = tool.calculate_artifacts(ground_truth, cache, config, {"id": "synthetic"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / tool.OUTPUT
            output.mkdir(parents=True)
            for name, value in (("ground_truth_cache.json", ground_truth), ("prediction_cache.json", cache),
                                ("selected_operating_point.json", selected)):
                (output / name).write_bytes(tool.shared.json_bytes(value))
            manifest = {"status": "COMPLETED", "config": config, "evaluation_identity": {"id": "synthetic"},
                        "artifacts": {name: tool.shared.digest(output / name) for name in
                                      ("ground_truth_cache.json", "prediction_cache.json", "selected_operating_point.json")}}
            (output / "evaluation_manifest.json").write_bytes(tool.shared.json_bytes(manifest))
            receipt = {"status": "COMPLETED",
                       "evaluation_manifest_sha256": tool.shared.digest(output / "evaluation_manifest.json")}
            (output / "completion.json").write_bytes(tool.shared.json_bytes(receipt))
            (output / "prediction_cache.json").write_bytes(b"{}")
            with patch.object(tool, "load_config", return_value=config):
                with self.assertRaisesRegex(ThresholdError, "SHA-256"):
                    tool.recompute(root)
            self.assertFalse((root / tool.RECOMPUTE_OUTPUT).exists())


if __name__ == "__main__":
    unittest.main()
