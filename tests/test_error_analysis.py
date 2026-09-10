"""Synthetic tests for validation-only cache error analysis; no held-out data access."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_damage.evaluation import error_analysis as tool  # noqa: E402
from road_damage.evaluation.threshold_metrics import Box, ThresholdError, match_image, sweep_cache  # noqa: E402


def record(image_id: str, boxes: list[Box]) -> dict:
    return {"image_id": image_id, "country": image_id.split("_")[0], "width": 100, "height": 100,
            "image_path": f"data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/images/val/{image_id}",
            "label_path": f"data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/labels/val/{Path(image_id).stem}.txt",
            "_boxes": boxes}


def synthetic() -> tuple[dict, dict, dict]:
    config = copy.deepcopy(tool.EXPECTED_CONFIG)
    config["expected_counts"] = {"images": 4, "targets": 4, "negative_images": 1}
    gt_records = [
        record("India_000001.jpg", [Box(0, (0., 0., 10., 10.)), Box(1, (20., 0., 30., 10.))]),
        record("India_000002.jpg", []),
        record("Japan_000003.jpg", [Box(2, (0., 0., 10., 10.))]),
        record("Japan_000004.jpg", [Box(3, (20., 20., 40., 40.))]),
    ]
    prediction_boxes = [
        [Box(0, (0., 0., 10., 10.), .9), Box(2, (20., 0., 30., 10.), .8),
         Box(0, (70., 70., 80., 80.), .7)],
        [Box(3, (50., 50., 60., 60.), .6)],
        [Box(2, (0., 0., 4., 10.), .7)],
        [Box(3, (20., 20., 40., 40.), .15)],
    ]
    cache_records = [{"image_id": r["image_id"], "nms_iou": .5, "_boxes": boxes}
                     for r, boxes in zip(gt_records, prediction_boxes, strict=True)]
    gt = {"validation_only": True, "counts": config["expected_counts"], "images": gt_records}
    cache = {"validation_only": True, "confidence_floor": .01, "nms_iou": .5, "images": cache_records}
    return config, gt, cache


def serialized_sources() -> tuple[dict, dict, dict, dict, dict]:
    config, gt, cache = synthetic()
    def serialize_record(value: dict) -> dict:
        row = {k: v for k, v in value.items() if k != "_boxes"}
        row["ground_truth"] = [{"class_id": b.class_id, "confidence": b.confidence, "xyxy": list(b.xyxy)}
                               for b in value["_boxes"]]
        row["image_sha256"], row["label_sha256"] = "a" * 64, "b" * 64
        return row
    gt_json = {"validation_only": True, "counts": config["expected_counts"],
               "images": [serialize_record(r) for r in gt["images"]]}
    cache_json = {"validation_only": True, "confidence_floor": .01, "nms_iou": .5,
        "images": [{"image_id": r["image_id"], "nms_iou": .5,
                    "predictions": [{"class_id": b.class_id, "confidence": b.confidence, "xyxy": list(b.xyxy)}
                                    for b in r["_boxes"]]} for r in cache["images"]]}
    truth = {r["image_id"]: r["_boxes"] for r in gt["images"]}
    predictions = {r["image_id"]: r["_boxes"] for r in cache["images"]}
    metrics = sweep_cache(truth, predictions, .5, [.19], .5)[0]
    selected = {"validation_only": True, "model_sha256": config["model_sha256"],
                "selected_global_confidence": .19, "selected_nms_iou": .5, "matching_iou": .5,
                "metrics": metrics, "git_commit": "synthetic-selection",
                "per_class_metrics": {code: {k: metrics[f"{code}_{k}"] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}
                                      for code in tool.CODES}}
    manifest = {"status": "complete", "identity": {"validation_only": True,
                "model_sha256": config["model_sha256"]},
                "artifacts": {config["ground_truth_cache"]: config["ground_truth_sha256"]},
                "prediction_cache_sha256": {config["prediction_cache"]: config["prediction_sha256"]}}
    return config, gt_json, cache_json, selected, manifest


class ErrorAnalysisTests(unittest.TestCase):
    def test_primary_indexed_matching_is_approved_matcher_equivalent_with_ties_and_duplicates(self):
        gt = [Box(0, (0., 0., 10., 10.)), Box(0, (20., 0., 30., 10.))]
        predictions = [Box(0, (0., 0., 10., 10.), .8), Box(0, (0., 0., 10., 10.), .8),
                       Box(0, (20., 0., 30., 10.), .7)]
        detailed = tool.primary_matches(gt, predictions, .5)
        approved = match_image(gt, predictions, .5)
        self.assertEqual(sum(match for _, match in approved), len(detailed))
        self.assertEqual([(m.prediction_index, m.ground_truth_index) for m in detailed], [(0, 0), (2, 1)])

    def test_fixed_point_confusion_localization_negative_and_below_threshold_diagnoses(self):
        config, gt, cache = synthetic()
        analysis = tool.analyze(config, gt, cache)
        classes = {r["class_code"]: r for r in analysis["per_class_errors"]}
        self.assertEqual((classes["D00"]["tp"], classes["D00"]["fp"], classes["D00"]["fn"]), (1, 1, 0))
        self.assertEqual((classes["D10"]["tp"], classes["D10"]["fp"], classes["D10"]["fn"]), (0, 0, 1))
        self.assertEqual(analysis["summary"]["class_confusion_pairs"], 1)
        self.assertEqual(analysis["summary"]["localization_errors"], 1)
        self.assertEqual((analysis["summary"]["negative_fp"], analysis["summary"]["negative_images_with_fp"]), (1, 1))
        diagnoses = {(r["class_code"], r["diagnosis"]) for r in analysis["false_negatives"]}
        self.assertIn(("D10", "class_confusion"), diagnoses)
        self.assertIn(("D20", "poor_localization"), diagnoses)
        self.assertIn(("D40", "below_frozen_confidence"), diagnoses)
        matrix = {(r["true_class"], r["predicted_class"]): r["count"] for r in analysis["class_confusion_matrix"]}
        self.assertEqual(matrix[("D10", "D20")], 1)
        self.assertEqual(matrix[("D20", tool.BACKGROUND)], 1)

    def test_normalized_size_bucket_boundaries_and_support_recall(self):
        config, gt, cache = synthetic()
        self.assertEqual(tool.size_bucket(.009999, config), "small")
        self.assertEqual(tool.size_bucket(.01, config), "medium")
        self.assertEqual(tool.size_bucket(.05, config), "large")
        self.assertEqual(tool.size_bucket(1., config), "large")
        analysis = tool.analyze(config, gt, cache)
        rows = {(r["class_code"], r["size_bucket"]): r for r in analysis["object_size_metrics"]}
        self.assertEqual(rows[("D00", "medium")]["gt_objects"], 1)
        self.assertEqual(rows[("D00", "medium")]["recall"], 1.)
        self.assertEqual(rows[("D20", "medium")]["recall"], 0.)

    def test_rankings_are_deterministic_and_cover_required_validation_queues(self):
        config, gt, cache = synthetic()
        first = tool.analyze(config, gt, cache)["ranked_examples"]
        second = tool.analyze(config, gt, cache)["ranked_examples"]
        self.assertEqual(first, second)
        categories = {row["category"] for row in first}
        self.assertTrue({"D10_FALSE_NEGATIVE", "D20_FALSE_NEGATIVE", "D40_FALSE_NEGATIVE",
                         "HIGH_CONFIDENCE_FALSE_POSITIVE", "INDIA_ERROR", "JAPAN_ERROR",
                         "NEGATIVE_ROAD_FALSE_POSITIVE", "POSSIBLE_CLASS_CONFUSION",
                         "POOR_LOCALIZATION"}.issubset(categories))
        self.assertFalse(any("test" in row["validation_image_path"].lower().split("/") for row in first))

    def test_source_loader_uses_only_pinned_validation_artifacts_not_dataset_or_model(self):
        config, gt, cache, selected, manifest = serialized_sources()
        forbidden = AssertionError("dataset/model filesystem access is forbidden during cache replay")
        with patch.object(tool, "load_config", return_value=config), \
             patch.object(tool, "_artifact", side_effect=[manifest, gt, cache, selected]) as artifact, \
             patch.object(Path, "iterdir", side_effect=forbidden), \
             patch.object(Path, "glob", side_effect=forbidden), \
             patch.object(Path, "rglob", side_effect=forbidden):
            _, loaded_gt, loaded_cache, _ = tool.load_sources(ROOT)
        self.assertEqual(artifact.call_count, 4)
        self.assertEqual(len(loaded_gt["images"]), 4)
        self.assertEqual(len(loaded_cache["images"]), 4)

    def test_country_support_precedes_metrics_and_india_d10_warning_is_explicit(self):
        config, gt, cache = synthetic()
        rows = tool.analyze(config, gt, cache)["country_errors"]
        india_d10 = next(r for r in rows if r["country"] == "India" and r["class_code"] == "D10")
        india_all = next(r for r in rows if r["country"] == "India" and r["class_code"] == "ALL")
        self.assertEqual((india_d10["support"], india_d10["positive_images"]), (1, 1))
        self.assertIn("LOW SUPPORT", india_d10["warning"])
        self.assertEqual((india_all["images"], india_all["support"]), (2, 2))

    def test_output_manifest_is_validation_only_complete_and_refuses_overwrite(self):
        config, gt_json, cache_json, selected, _ = serialized_sources()
        # load_sources normally converts serialized boxes to private Box caches.
        for r in gt_json["images"]:
            r["country"] = r["image_id"].split("_")[0]
            r["_boxes"] = [Box(b["class_id"], tuple(b["xyxy"]), b["confidence"]) for b in r["ground_truth"]]
        for r in cache_json["images"]:
            r["_boxes"] = [Box(b["class_id"], tuple(b["xyxy"]), b["confidence"]) for b in r["predictions"]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg_path = root / tool.CONFIG
            cfg_path.parent.mkdir(parents=True)
            cfg_path.write_bytes(tool.shared.json_bytes(config))
            completed = SimpleNamespace(stdout="2109c55\n")
            status = SimpleNamespace(stdout="?? validation-only-tooling\n")
            with patch.object(tool, "ROOT", root), \
                 patch.object(tool, "load_sources", return_value=(config, gt_json, cache_json, selected)), \
                 patch.object(tool.subprocess, "run", side_effect=[status, completed]):
                output = tool.run(root)
            manifest = json.loads((output / "error_analysis_manifest.json").read_bytes())
            self.assertEqual((manifest["status"], manifest["validation_only"]), ("complete", True))
            self.assertFalse(manifest["inference_executed"])
            self.assertFalse(manifest["internal_test_images_or_labels_accessed"])
            self.assertIn("Proceed with the planned pretrained YOLO26s", (output / "error_analysis_summary.md").read_text())
            self.assertEqual(len(list(output.glob("*"))), 13)
            with patch.object(tool, "ROOT", root), patch.object(tool, "load_sources", return_value=(config, gt_json, cache_json, selected)):
                with self.assertRaises(ThresholdError):
                    tool.run(root)


if __name__ == "__main__":
    unittest.main()
