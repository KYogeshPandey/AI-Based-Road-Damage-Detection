"""Focused cache-only tests for Experiment 2 validation error analysis."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_damage.evaluation import experiment2_error_analysis as tool  # noqa: E402
from road_damage.evaluation.threshold_metrics import Box, ThresholdError  # noqa: E402


def _record(image_id: str, boxes: list[Box]) -> dict:
    return {"image_id": image_id, "country": image_id.split("_")[0], "width": 100, "height": 100,
            "image_path": f"data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/images/val/{image_id}",
            "label_path": f"data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/labels/val/{Path(image_id).stem}.txt",
            "_boxes": boxes}


def _synthetic() -> tuple[dict, dict, dict, dict]:
    config = copy.deepcopy(tool.EXPECTED_CONFIG)
    gt_rows = [
        _record("India_000001.jpg", [Box(0, (0., 0., 10., 10.))]),
        _record("India_000002.jpg", [Box(1, (20., 0., 30., 10.))]),
        _record("Japan_000003.jpg", [Box(2, (0., 20., 10., 30.))]),
        _record("Japan_000004.jpg", [Box(3, (20., 20., 40., 40.))]),
        _record("Japan_000005.jpg", []),
    ]
    prediction_boxes = [
        [Box(0, (0., 0., 10., 10.), .9)],
        [Box(0, (20., 0., 30., 10.), .8)],
        [Box(2, (0., 20., 4., 30.), .7)],
        [Box(3, (20., 20., 40., 40.), .15)],
        [Box(3, (50., 50., 60., 60.), .6)],
    ]
    cache_rows = [{"image_id": row["image_id"], "_boxes": boxes}
                  for row, boxes in zip(gt_rows, prediction_boxes, strict=True)]
    ground_truth = {"images": gt_rows}
    cache = {"images": cache_rows}
    config["expected_operating_counts"] = {
        "D00": {"tp": 1, "fp": 1, "fn": 0},
        "D10": {"tp": 0, "fp": 0, "fn": 1},
        "D20": {"tp": 0, "fp": 1, "fn": 1},
        "D40": {"tp": 0, "fp": 1, "fn": 1},
    }
    metrics = {}
    for code, values in config["expected_operating_counts"].items():
        metrics[code] = tool._metric(values["tp"], values["fp"], values["fn"])
    selected = {"per_class_metrics": metrics,
                "macro_metrics": {"precision": .25, "recall": .25, "f1": .25},
                "micro_metrics": {"f1": .25},
                "negative_image_metrics": {"negative_fp": 1, "negative_images_with_fp": 1,
                                           "negative_fp_per_image": 1.0}}
    config["expected_validation"] = {"images": 5, "positive_images": 4, "negative_images": 1,
                                     "targets": 4, "targets_by_class": {str(i): 1 for i in range(4)}}
    return config, ground_truth, cache, selected


class PolicyAndSourceTests(unittest.TestCase):
    def test_literal_cache_threshold_class_and_nms_policy(self) -> None:
        config = tool.load_config()
        self.assertEqual(config["prediction_sha256"],
                         "b48aebb298bd4f1881cddd10602de332f8cc0e560f05fd9b2f9dbb212b8b98ca")
        self.assertEqual((config["confidence"], config["matching_iou"], config["cache_floor"]),
                         (.193, .5, .01))
        self.assertFalse(config["nms_applicable"])
        self.assertFalse(config["model_inference_permitted"])
        self.assertEqual(config["classes"], {
            "0": "D00_longitudinal_crack", "1": "D10_transverse_crack",
            "2": "D20_alligator_crack", "3": "D40_pothole"})

    def test_config_rejects_cache_threshold_class_source_and_inference_overrides(self) -> None:
        original = tool.load_config()
        changes = (
            {"prediction_sha256": "0" * 64}, {"confidence": .19}, {"matching_iou": .6},
            {"nms_applicable": True}, {"model_inference_permitted": True},
            {"source": "outputs/evaluation/internal_test"},
            {"classes": {**original["classes"], "0": "wrong"}},
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / tool.CONFIG
                path.parent.mkdir(parents=True)
                path.write_bytes(tool.shared.json_bytes({**original, **change}))
                with self.assertRaises(ThresholdError):
                    tool.load_config(root)

    def test_real_pinned_cache_sha_and_baseline_artifacts_verify_without_inference(self) -> None:
        config = tool.load_config()
        cache_path = ROOT / tool.SOURCE / config["prediction_cache"]
        self.assertEqual(hashlib.sha256(cache_path.read_bytes()).hexdigest(), config["prediction_sha256"])
        baseline = tool._load_baseline_artifacts(ROOT, config)
        self.assertEqual(baseline["manifest"]["status"], "complete")
        self.assertEqual(len(baseline["per_class"]), 4)

    def test_complete_source_load_uses_cache_only_and_exact_validation_identity(self) -> None:
        config, gt, cache, selected, baseline = tool.load_sources(ROOT)
        self.assertEqual(gt["counts"], config["expected_validation"])
        self.assertEqual(len(cache["images"]), 2602)
        self.assertEqual(selected["selected_global_confidence"], .193)
        self.assertEqual(baseline["manifest"]["counts"]["validation_images"], 2602)
        source = inspect.getsource(tool)
        self.assertNotIn("from ultralytics", source)
        self.assertNotIn("model.predict", source)
        self.assertNotIn("images/test", source)
        self.assertNotIn("teacher_video/", source)


class TaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = copy.deepcopy(tool.EXPECTED_CONFIG)
        self.gt = Box(0, (0., 0., 10., 10.))

    def test_fn_below_confidence_and_declared_priority(self) -> None:
        retained = [Box(1, (0., 0., 10., 10.), .9)]
        raw = [*retained, Box(0, (0., 0., 10., 10.), .15)]
        result = tool._diagnose_fn(self.gt, retained, raw, set(), self.config)
        self.assertEqual(result["category"], "below_confidence_same_class_iou_ge_0.50")
        self.assertEqual(result["diagnostic_confidence"], .15)
        self.assertTrue(result["evidence"]["retained_wrong_class_overlap"])

    def test_fn_localization_bands_wrong_class_competition_and_no_candidate(self) -> None:
        cases = (
            ([Box(0, (0., 0., 2., 10.), .7)], set(), "localization_iou_0.10_to_0.30"),
            ([Box(0, (0., 0., 4., 10.), .7)], set(), "localization_iou_0.30_to_0.50"),
            ([Box(1, (0., 0., 10., 10.), .7)], set(), "wrong_class_overlap_iou_ge_0.50"),
            ([Box(0, (0., 0., 10., 10.), .7)], {0}, "same_class_matching_competition_iou_ge_0.50"),
            ([], set(), tool.NO_CANDIDATE),
        )
        for retained, matched, expected in cases:
            with self.subTest(expected=expected):
                result = tool._diagnose_fn(self.gt, retained, retained, matched, self.config)
                self.assertEqual(result["category"], expected)
        self.assertEqual(tool.NO_CANDIDATE,
                         "no_stronger_cached_candidate_above_0.010_inference_floor")

    def test_fp_taxonomy_negative_duplicate_wrong_class_localization_and_unmatched(self) -> None:
        other = Box(1, (0., 0., 10., 10.))
        cases = (
            (Box(0, (0., 0., 10., 10.), .8), [], "negative_only_image"),
            (Box(0, (0., 0., 10., 10.), .8), [self.gt],
             "native_end2end_same_class_overlap_unmatched_iou_ge_0.50"),
            (Box(0, (0., 0., 10., 10.), .8), [other], "wrong_class_overlap_iou_ge_0.50"),
            (Box(0, (0., 0., 4., 10.), .8), [self.gt], "localization_same_class_iou_0.30_to_0.50"),
            (Box(0, (0., 0., 2., 10.), .8), [self.gt], "localization_same_class_iou_0.10_to_0.30"),
            (Box(0, (50., 50., 60., 60.), .8), [self.gt], "positive_image_unmatched_iou_below_0.10"),
        )
        for prediction, truth, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(tool._diagnose_fp(prediction, truth)["category"], expected)

    def test_per_class_negative_object_country_and_deterministic_aggregation(self) -> None:
        config, gt, cache, selected = _synthetic()
        first = tool.analyze(config, gt, cache, selected)
        second = tool.analyze(config, gt, cache, selected)
        self.assertEqual(tool.shared.json_bytes(first), tool.shared.json_bytes(second))
        classes = {row["class_code"]: row for row in first["per_class"]}
        self.assertEqual((classes["D00"]["tp"], classes["D00"]["fp"], classes["D00"]["fn"]), (1, 1, 0))
        self.assertEqual(classes["D10"]["fn_taxonomy"][0]["category"],
                         "wrong_class_overlap_iou_ge_0.50")
        self.assertEqual(classes["D20"]["fn_taxonomy"][0]["category"],
                         "localization_iou_0.30_to_0.50")
        self.assertEqual(classes["D40"]["fn_taxonomy"][0]["category"],
                         "below_confidence_same_class_iou_ge_0.50")
        self.assertEqual(first["summary"]["negative_only"]["false_positives"], 1)
        self.assertEqual(first["summary"]["negative_only"]["images_with_at_least_one_fp"], 1)
        self.assertEqual(len(first["object_size"]["rows"]), 15)
        self.assertEqual(len(first["aspect_ratio"]["rows"]), 15)
        self.assertEqual(len(first["target_density"]["rows"]), 15)
        india_d10 = next(row for row in first["country"]["rows"]
                         if row["country"] == "India" and row["class_code"] == "D10")
        self.assertIn("LOW SUPPORT", india_d10["warning"])

    def test_prediction_profiles_are_deterministic_and_class_specific(self) -> None:
        config, gt, cache, _ = _synthetic()
        profile = tool._prediction_profile(gt, cache, [.01, .193, .5])
        negative = profile["negative_only_images"]
        self.assertEqual([row["predictions"] for row in negative], [1, 1, 1])
        self.assertEqual(negative[-1]["by_class"], {"D00": 0, "D10": 0, "D20": 0, "D40": 1})

    def test_real_cross_model_comparison_includes_country_size_and_negative_qualification(self) -> None:
        config, ground_truth, cache, selected, baseline_artifacts = tool.load_sources(ROOT)
        analysis = tool.analyze(config, ground_truth, cache, selected)
        comparison, negative = tool.build_comparison(
            config, analysis, baseline_artifacts, ground_truth, cache,
        )
        self.assertEqual(len(comparison["object_size_recall"]), 12)
        self.assertEqual(len(comparison["country"]), 10)
        india_d10 = next(row for row in comparison["country"]
                         if row["country"] == "India" and row["class_code"] == "D10")
        self.assertIn("LOW SUPPORT", india_d10["warning"])
        self.assertFalse(negative["measured_interpretation"]
                         ["uniformly_lower_common_confidence_profile"])


if __name__ == "__main__":
    unittest.main()
