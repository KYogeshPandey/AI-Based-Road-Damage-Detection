"""Deterministic YOLO26s validation error analysis from approved caches only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.evaluation import error_analysis as baseline  # noqa: E402
from road_damage.evaluation import select_experiment2_threshold as selection  # noqa: E402
from road_damage.evaluation import select_threshold as shared  # noqa: E402
from road_damage.evaluation.threshold_metrics import Box, ThresholdError, iou  # noqa: E402


ROOT = shared.ROOT
CONFIG = Path("configs/evaluation/experiment2_yolo26s_error_analysis.yaml")
SOURCE = Path("outputs/evaluation/experiment2_yolo26s_threshold_selection")
BASELINE_SOURCE = Path("outputs/evaluation/baseline_public_v1_error_analysis")
OUTPUT = Path("outputs/evaluation/experiment2_yolo26s_error_analysis")
CODES = ("D00", "D10", "D20", "D40")
NO_CANDIDATE = "no_stronger_cached_candidate_above_0.010_inference_floor"
EXPECTED_CONFIG: dict[str, Any] = {
    "schema_version": "experiment2_yolo26s.validation_error_analysis.v1",
    "source": SOURCE.as_posix(),
    "threshold_config": selection.CONFIG.as_posix(),
    "threshold_config_sha256": "995f51eb7c67796de71aea82cacf9bc4f6973d996bd45eba1bfd8499104671c0",
    "ground_truth_cache": "ground_truth_cache.json",
    "ground_truth_sha256": "61289d25f6951929f2876f9c0686d56398dd9cb8773fc247434debb0e993caf4",
    "prediction_cache": "prediction_cache.json",
    "prediction_sha256": "b48aebb298bd4f1881cddd10602de332f8cc0e560f05fd9b2f9dbb212b8b98ca",
    "selected_operating_point": "selected_operating_point.json",
    "selected_operating_point_sha256": "d5bd61eb21df8fd8a5e3c2c1e369253b4e8141528040609b9952f1834c7225c4",
    "source_manifest": "evaluation_manifest.json",
    "source_manifest_sha256": "70cd99b46c2d3a319ed61de0882fcc72b3aa275f52cd384035c9f6443d4e3c49",
    "source_completion": "completion.json",
    "source_completion_sha256": "a2dc73be837270e4244a7c8680f7f47c59cf336df9464faa924e471f535e3fa8",
    "runtime_architecture": "runtime_architecture.json",
    "runtime_architecture_sha256": "f3f21f278d583ff39b1bd95898ce947b5fb26fa545b0fe41afaabfc5f1c3d86f",
    "baseline_error_analysis": BASELINE_SOURCE.as_posix(),
    "baseline_error_manifest": "error_analysis_manifest.json",
    "baseline_error_manifest_sha256": "5e7b5ef3d4584cbd8f8fa85ab3c0886eeb8e0e2bbaa9ea6cf69750b70048ad02",
    "model_sha256": selection.EXPECTED_MODEL_SHA256,
    "confidence": .193, "matching_iou": .5, "cache_floor": .01, "nms_applicable": False,
    "localization_bands": [
        {"name": "localization_iou_0.10_to_0.30", "minimum_inclusive": .1, "maximum_exclusive": .3},
        {"name": "localization_iou_0.30_to_0.50", "minimum_inclusive": .3, "maximum_exclusive": .5},
    ],
    "fn_category_priority": [
        "below_confidence_same_class_iou_ge_0.50",
        "same_class_matching_competition_iou_ge_0.50",
        "localization_iou_0.30_to_0.50", "localization_iou_0.10_to_0.30",
        "wrong_class_overlap_iou_ge_0.50", NO_CANDIDATE,
    ],
    "size_buckets": [
        {"name": "small", "minimum_inclusive": 0., "maximum_exclusive": .01},
        {"name": "medium", "minimum_inclusive": .01, "maximum_exclusive": .05},
        {"name": "large", "minimum_inclusive": .05, "maximum_exclusive": 1.0000001},
    ],
    "aspect_ratio_buckets": [
        {"name": "narrow", "minimum_inclusive": 0., "maximum_exclusive": .5},
        {"name": "balanced", "minimum_inclusive": .5, "maximum_exclusive": 2.},
        {"name": "wide", "minimum_inclusive": 2., "maximum_exclusive": 1_000_000.},
    ],
    "target_density_buckets": [
        {"name": "one_target", "minimum_inclusive": 1, "maximum_exclusive": 2},
        {"name": "two_to_three_targets", "minimum_inclusive": 2, "maximum_exclusive": 4},
        {"name": "four_or_more_targets", "minimum_inclusive": 4, "maximum_exclusive": 1_000_000},
    ],
    "common_confidence_profile": [.01, .05, .10, .15, .19, .193, .25, .50],
    "expected_validation": {
        "images": 2602, "positive_images": 1619, "negative_images": 983, "targets": 3377,
        "targets_by_class": {"0": 822, "1": 576, "2": 1188, "3": 791},
    },
    "expected_operating_counts": {
        "D00": {"tp": 303, "fp": 333, "fn": 519},
        "D10": {"tp": 248, "fp": 287, "fn": 328},
        "D20": {"tp": 769, "fp": 489, "fn": 419},
        "D40": {"tp": 356, "fp": 303, "fn": 435},
    },
    "classes": {str(key): value for key, value in shared.CLASS_NAMES.items()},
    "output": OUTPUT.as_posix(), "validation_only": True,
    "internal_test_access_permitted": False, "official_unlabelled_test_used": False,
    "teacher_video_used": False, "model_inference_permitted": False,
}
BASELINE_ARTIFACT_NAMES = {
    "class_confusion_matrix.csv", "confidence_distributions.csv", "country_errors.csv",
    "error_analysis_summary.md", "false_negatives.csv", "false_positives.csv",
    "iou_distribution.csv", "localization_errors.csv", "negative_image_false_positives.csv",
    "object_size_metrics.csv", "per_class_errors.csv", "ranked_examples.csv",
}


def load_config(root: Path = ROOT) -> dict[str, Any]:
    path = shared.checked_path(CONFIG, CONFIG, root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if shared.json_bytes(value) != shared.json_bytes(EXPECTED_CONFIG):
        raise ThresholdError("Experiment 2 error-analysis config changed or contains an override.")
    return value


def _json_artifact(root: Path, base: Path, name: str, expected_sha256: str) -> dict[str, Any]:
    if Path(name).name != name:
        raise ThresholdError("Pinned cache input must be a filename, not a path override.")
    relative = base / name
    path = shared.checked_path(relative, relative, root)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ThresholdError(f"Pinned validation artifact changed: {relative}")
    return json.loads(payload)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_baseline_artifacts(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _json_artifact(
        root, BASELINE_SOURCE, config["baseline_error_manifest"], config["baseline_error_manifest_sha256"]
    )
    if (manifest.get("status") != "complete" or manifest.get("validation_only") is not True
            or manifest.get("inference_executed") is not False
            or manifest.get("internal_test_images_or_labels_accessed") is not False
            or set(manifest.get("artifacts", {})) != BASELINE_ARTIFACT_NAMES):
        raise ThresholdError("Approved YOLOv8 error-analysis manifest identity mismatch.")
    for name in sorted(BASELINE_ARTIFACT_NAMES):
        path = shared.checked_path(BASELINE_SOURCE / name, BASELINE_SOURCE / name, root)
        if shared.digest(path) != manifest["artifacts"][name]:
            raise ThresholdError(f"Approved YOLOv8 error-analysis artifact changed: {name}")
    baseline_config, baseline_gt, baseline_cache, baseline_selected = baseline.load_sources(root)
    return {
        "manifest": manifest, "config": baseline_config, "ground_truth": baseline_gt,
        "prediction_cache": baseline_cache, "selected": baseline_selected,
        "per_class": _csv_rows(root / BASELINE_SOURCE / "per_class_errors.csv"),
        "false_negatives": _csv_rows(root / BASELINE_SOURCE / "false_negatives.csv"),
        "false_positives": _csv_rows(root / BASELINE_SOURCE / "false_positives.csv"),
        "object_size": _csv_rows(root / BASELINE_SOURCE / "object_size_metrics.csv"),
        "country": _csv_rows(root / BASELINE_SOURCE / "country_errors.csv"),
    }


def _same_ground_truth(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("image_id", "width", "height", "image_path", "label_path", "image_sha256", "label_sha256", "ground_truth")
    left_rows = [{key: row.get(key) for key in keys} for row in left["images"]]
    right_rows = [{key: row.get(key) for key in keys} for row in right["images"]]
    return shared.json_bytes(left_rows) == shared.json_bytes(right_rows)


def load_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load only pinned validation caches/analysis artifacts; never load a model or dataset."""
    config = load_config(root)
    threshold_config_path = shared.checked_path(selection.CONFIG, selection.CONFIG, root)
    if shared.digest(threshold_config_path) != config["threshold_config_sha256"]:
        raise ThresholdError("Experiment 2 threshold-selection config changed.")
    threshold_config = selection.load_config(root)
    ground_truth = _json_artifact(root, SOURCE, config["ground_truth_cache"], config["ground_truth_sha256"])
    cache = _json_artifact(root, SOURCE, config["prediction_cache"], config["prediction_sha256"])
    selected = _json_artifact(
        root, SOURCE, config["selected_operating_point"], config["selected_operating_point_sha256"]
    )
    manifest = _json_artifact(root, SOURCE, config["source_manifest"], config["source_manifest_sha256"])
    completion = _json_artifact(root, SOURCE, config["source_completion"], config["source_completion_sha256"])
    architecture = _json_artifact(
        root, SOURCE, config["runtime_architecture"], config["runtime_architecture_sha256"]
    )
    if (manifest.get("status") != "COMPLETED" or manifest.get("validation_only") is not True
            or manifest.get("internal_test_files_accessed") is not False
            or manifest.get("training_executed") is not False
            or manifest.get("artifacts", {}).get(config["prediction_cache"]) != config["prediction_sha256"]
            or completion.get("status") != "COMPLETED" or completion.get("validation_only") is not True
            or completion.get("internal_test_files_accessed") is not False
            or completion.get("prediction_cache_sha256") != config["prediction_sha256"]
            or completion.get("selected_global_confidence") != config["confidence"]
            or selected.get("selected_global_confidence") != config["confidence"]
            or selected.get("matching_iou") != config["matching_iou"]
            or selected.get("nms") != {
                "applicable": False, "iou": None,
                "reason": "native Ultralytics 8.4.130 YOLO26 end-to-end NMS-free output preserved", "swept": False,
            }
            or architecture.get("nc") != 4 or architecture.get("end2end") is not True
            or architecture.get("reg_max") != 1 or architecture.get("names") != config["classes"]
            or ground_truth.get("counts") != config["expected_validation"]):
        raise ThresholdError("Approved YOLO26 validation cache/operating-point identity mismatch.")
    predictions = selection.decoded_predictions(cache, threshold_config)
    truth = shared.decoded_ground_truth(ground_truth)
    if set(predictions) != set(truth):
        raise ThresholdError("YOLO26 validation cache coverage differs from ground truth.")
    _, replayed = selection.calculate_artifacts(
        ground_truth, cache, threshold_config, selected["evaluation_identity"]
    )
    if shared.json_bytes(replayed) != shared.json_bytes(selected):
        raise ThresholdError("YOLO26 selected operating point does not replay from the approved cache.")
    gt_by_id = {row["image_id"]: row for row in ground_truth["images"]}
    cache_by_id = {row["image_id"]: row for row in cache["images"]}
    for image_id in sorted(gt_by_id):
        record = gt_by_id[image_id]
        country = baseline._country(image_id)
        record["country"] = country
        record["_boxes"] = truth[image_id]
        cache_by_id[image_id]["_boxes"] = predictions[image_id]
    baseline_artifacts = _load_baseline_artifacts(root, config)
    if not _same_ground_truth(ground_truth, baseline_artifacts["ground_truth"]):
        raise ThresholdError("YOLOv8 and YOLO26 comparison caches do not share identical validation ground truth.")
    return config, ground_truth, cache, selected, baseline_artifacts


def _bucket(value: float, definitions: Sequence[Mapping[str, Any]]) -> str:
    for definition in definitions:
        if definition["minimum_inclusive"] <= value < definition["maximum_exclusive"]:
            return str(definition["name"])
    raise ThresholdError(f"Value outside configured descriptive buckets: {value}")


def _aspect_ratio(box: Box) -> float:
    x1, y1, x2, y2 = box.xyxy
    return (x2 - x1) / (y2 - y1)


def _best_candidate(candidates: Sequence[tuple[float, Box]]) -> tuple[float, Box] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1].confidence, -item[1].class_id,
                                             tuple(-coordinate for coordinate in item[1].xyxy)))


def _diagnose_fn(ground: Box, retained: Sequence[Box], raw: Sequence[Box],
                 matched_prediction_indices: set[int], config: Mapping[str, Any]) -> dict[str, Any]:
    """Assign one deterministic, cache-observable FN category in declared priority order."""
    below = _best_candidate([
        (iou(prediction, ground), prediction) for prediction in raw
        if prediction.class_id == ground.class_id and prediction.confidence < config["confidence"]
        and iou(prediction, ground) >= config["matching_iou"]
    ])
    competition = _best_candidate([
        (iou(prediction, ground), prediction) for index, prediction in enumerate(retained)
        if index in matched_prediction_indices and prediction.class_id == ground.class_id
        and iou(prediction, ground) >= config["matching_iou"]
    ])
    localization = _best_candidate([
        (iou(prediction, ground), prediction) for prediction in retained
        if prediction.class_id == ground.class_id
        and config["localization_bands"][0]["minimum_inclusive"] <= iou(prediction, ground) < config["matching_iou"]
    ])
    wrong_class = _best_candidate([
        (iou(prediction, ground), prediction) for prediction in retained
        if prediction.class_id != ground.class_id and iou(prediction, ground) >= config["matching_iou"]
    ])
    if below is not None:
        category, diagnostic = "below_confidence_same_class_iou_ge_0.50", below
    elif competition is not None:
        category, diagnostic = "same_class_matching_competition_iou_ge_0.50", competition
    elif localization is not None:
        overlap = localization[0]
        category = _bucket(overlap, config["localization_bands"])
        diagnostic = localization
    elif wrong_class is not None:
        category, diagnostic = "wrong_class_overlap_iou_ge_0.50", wrong_class
    else:
        category = NO_CANDIDATE
        diagnostic = _best_candidate([(iou(prediction, ground), prediction) for prediction in raw])
    if category not in config["fn_category_priority"]:
        raise ThresholdError(f"Undeclared FN category: {category}")
    overlap, prediction = diagnostic if diagnostic is not None else (0., None)
    return {
        "category": category, "diagnostic_iou": overlap,
        "diagnostic_confidence": None if prediction is None else prediction.confidence,
        "diagnostic_predicted_class_id": None if prediction is None else prediction.class_id,
        "diagnostic_predicted_class": None if prediction is None else CODES[prediction.class_id],
        "evidence": {
            "below_confidence_same_class": below is not None,
            "same_class_matching_competition": competition is not None,
            "retained_same_class_localization": localization is not None,
            "retained_wrong_class_overlap": wrong_class is not None,
        },
    }


def _diagnose_fp(prediction: Box, ground_truth: Sequence[Box]) -> dict[str, Any]:
    """Assign one cache-geometric FP category without claiming legacy-NMS behavior."""
    if not ground_truth:
        return {"category": "negative_only_image", "best_gt_iou": 0.,
                "related_gt_class_id": None, "related_gt_class": None}
    same = _best_candidate([(iou(prediction, ground), ground) for ground in ground_truth
                            if prediction.class_id == ground.class_id])
    different = _best_candidate([(iou(prediction, ground), ground) for ground in ground_truth
                                 if prediction.class_id != ground.class_id])
    same_iou = 0. if same is None else same[0]
    different_iou = 0. if different is None else different[0]
    if same_iou >= .5:
        category, related = "native_end2end_same_class_overlap_unmatched_iou_ge_0.50", same
    elif different_iou >= .5:
        category, related = "wrong_class_overlap_iou_ge_0.50", different
    elif .3 <= same_iou < .5:
        category, related = "localization_same_class_iou_0.30_to_0.50", same
    elif .1 <= same_iou < .3:
        category, related = "localization_same_class_iou_0.10_to_0.30", same
    elif .1 <= different_iou < .5:
        category, related = "wrong_class_weak_overlap_iou_0.10_to_0.50", different
    else:
        category = "positive_image_unmatched_iou_below_0.10"
        related = same if same_iou >= different_iou else different
    overlap, ground = related if related is not None else (0., None)
    return {"category": category, "best_gt_iou": overlap,
            "related_gt_class_id": None if ground is None else ground.class_id,
            "related_gt_class": None if ground is None else CODES[ground.class_id]}


def _metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return baseline._metric(tp, fp, fn)


def _category_counts(records: Sequence[Mapping[str, Any]], total: int) -> list[dict[str, Any]]:
    counts = Counter(str(record["category"]) for record in records)
    return [{"category": category, "count": count, "percent": 100 * count / total if total else 0.}
            for category, count in sorted(counts.items())]


def _dimension_rows(records: Sequence[Mapping[str, Any]], key: str,
                    definitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for class_id, class_code in [*(enumerate(CODES)), (None, "ALL")]:
        for definition in definitions:
            selected_records = [record for record in records
                                if record[key] == definition["name"]
                                and (class_id is None or record["class_id"] == class_id)]
            tp = sum(record["outcome"] == "TP" for record in selected_records)
            fn = len(selected_records) - tp
            fn_records = [record for record in selected_records if record["outcome"] == "FN"]
            rows.append({
                "class_id": class_id, "class_code": class_code, key: definition["name"],
                "support": len(selected_records), "tp": tp, "fn": fn,
                "recall": tp / len(selected_records) if selected_records else 0.,
                "fn_rate": fn / len(selected_records) if selected_records else 0.,
                "weak_support": len(selected_records) < 30,
                "fn_categories": _category_counts(fn_records, fn),
            })
    return rows


def _country_rows(gt_records: Sequence[Mapping[str, Any]], fp_records: Sequence[Mapping[str, Any]],
                  images: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for country in ("India", "Japan"):
        country_images = [row for row in images.values() if row["country"] == country]
        for class_id, class_code in [*(enumerate(CODES)), (None, "ALL")]:
            ground = [record for record in gt_records if record["country"] == country
                      and (class_id is None or record["class_id"] == class_id)]
            fps = [record for record in fp_records if record["country"] == country
                   and (class_id is None or record["class_id"] == class_id)]
            tp = sum(record["outcome"] == "TP" for record in ground)
            fn = len(ground) - tp
            positive_images = sum(
                any(box.class_id == class_id for box in row["_boxes"]) if class_id is not None else bool(row["_boxes"])
                for row in country_images
            )
            warning = (
                "LOW SUPPORT — India D10 has 10 targets in 10 validation images; no strong country-specific conclusion"
                if country == "India" and class_code == "D10" else ""
            )
            rows.append({
                "country": country, "class_id": class_id, "class_code": class_code,
                "images": len(country_images), "positive_images": positive_images,
                "support": len(ground), "predictions": tp + len(fps),
                **_metric(tp, len(fps), fn), "warning": warning,
                "fn_categories": _category_counts([record for record in ground if record["outcome"] == "FN"], fn),
                "fp_categories": _category_counts(fps, len(fps)),
            })
    return rows


def analyze(config: Mapping[str, Any], ground_truth: Mapping[str, Any],
            cache: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    """Classify every fixed-point FN and FP using cache-observable evidence."""
    gt_by_id = {row["image_id"]: row for row in ground_truth["images"]}
    cache_by_id = {row["image_id"]: row for row in cache["images"]}
    gt_records: list[dict[str, Any]] = []
    fp_records: list[dict[str, Any]] = []
    tp_records: list[dict[str, Any]] = []

    for image_id in sorted(gt_by_id):
        image = gt_by_id[image_id]
        raw: list[Box] = cache_by_id[image_id]["_boxes"]
        truth: list[Box] = image["_boxes"]
        retained = [prediction for prediction in raw if prediction.confidence >= config["confidence"]]
        matches = baseline.primary_matches(truth, retained, config["matching_iou"])
        matched_gt = {match.ground_truth_index for match in matches}
        matched_predictions = {match.prediction_index for match in matches}
        match_by_gt = {match.ground_truth_index: match for match in matches}
        density = len(truth)
        density_bucket = _bucket(density, config["target_density_buckets"]) if density else None

        for gt_index, ground in enumerate(truth):
            area = baseline.normalized_area(ground, image["width"], image["height"])
            aspect_ratio = _aspect_ratio(ground)
            common = {
                "image_id": image_id, "validation_image_path": image["image_path"],
                "country": image["country"], "gt_index": gt_index,
                "class_id": ground.class_id, "class_code": CODES[ground.class_id],
                "gt_xyxy": list(ground.xyxy), "normalized_bbox_area": area,
                "size_bucket": _bucket(area, config["size_buckets"]),
                "aspect_ratio_width_over_height": aspect_ratio,
                "aspect_ratio_bucket": _bucket(aspect_ratio, config["aspect_ratio_buckets"]),
                "positive_image_target_count": density, "target_density_bucket": density_bucket,
            }
            if gt_index in matched_gt:
                match = match_by_gt[gt_index]
                prediction = retained[match.prediction_index]
                row = {**common, "outcome": "TP", "category": "true_positive",
                       "matched_prediction_confidence": prediction.confidence,
                       "matched_iou": match.overlap_iou}
                gt_records.append(row)
                tp_records.append(row)
            else:
                diagnosis = _diagnose_fn(ground, retained, raw, matched_predictions, config)
                gt_records.append({**common, "outcome": "FN", **diagnosis})

        for prediction_index, prediction in enumerate(retained):
            if prediction_index in matched_predictions:
                continue
            diagnosis = _diagnose_fp(prediction, truth)
            area = baseline.normalized_area(prediction, image["width"], image["height"])
            fp_records.append({
                "image_id": image_id, "validation_image_path": image["image_path"],
                "country": image["country"], "prediction_index": prediction_index,
                "class_id": prediction.class_id, "class_code": CODES[prediction.class_id],
                "confidence": prediction.confidence, "prediction_xyxy": list(prediction.xyxy),
                "normalized_bbox_area": area, "size_bucket": _bucket(area, config["size_buckets"]),
                "positive_image": bool(truth), "negative_only_image": not truth,
                **diagnosis,
            })

    false_negatives = [record for record in gt_records if record["outcome"] == "FN"]
    per_class = []
    for class_id, code in enumerate(CODES):
        class_gt = [record for record in gt_records if record["class_id"] == class_id]
        class_fn = [record for record in class_gt if record["outcome"] == "FN"]
        class_tp = len(class_gt) - len(class_fn)
        class_fp = [record for record in fp_records if record["class_id"] == class_id]
        positive_images = sum(any(box.class_id == class_id for box in image["_boxes"])
                              for image in gt_by_id.values())
        row = {
            "class_id": class_id, "class_code": code, "class_name": config["classes"][str(class_id)],
            "support": len(class_gt), "positive_images": positive_images,
            "predictions": class_tp + len(class_fp), **_metric(class_tp, len(class_fp), len(class_fn)),
            "fn_taxonomy": _category_counts(class_fn, len(class_fn)),
            "fp_taxonomy": _category_counts(class_fp, len(class_fp)),
        }
        if {key: row[key] for key in ("tp", "fp", "fn")} != config["expected_operating_counts"][code]:
            raise ThresholdError(f"Detailed {code} counts do not reconcile with the frozen operating point.")
        approved = selected["per_class_metrics"][code]
        if any(row[key] != approved[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")):
            raise ThresholdError(f"Detailed {code} metrics do not reproduce the approved selection.")
        per_class.append(row)

    expected_tp = sum(values["tp"] for values in config["expected_operating_counts"].values())
    expected_fp = sum(values["fp"] for values in config["expected_operating_counts"].values())
    expected_fn = sum(values["fn"] for values in config["expected_operating_counts"].values())
    if (len(false_negatives), len(fp_records), len(tp_records)) != (expected_fn, expected_fp, expected_tp):
        raise ThresholdError("Detailed FN/FP/TP records do not reconcile with approved totals.")
    if any(record["category"] == NO_CANDIDATE and any(record["evidence"].values()) for record in false_negatives):
        raise ThresholdError("No-candidate FN has contradictory qualifying cached evidence.")

    size_rows = _dimension_rows(gt_records, "size_bucket", config["size_buckets"])
    aspect_rows = _dimension_rows(gt_records, "aspect_ratio_bucket", config["aspect_ratio_buckets"])
    density_rows = _dimension_rows(gt_records, "target_density_bucket", config["target_density_buckets"])
    country_rows = _country_rows(gt_records, fp_records, gt_by_id)
    negative_fp = [record for record in fp_records if record["negative_only_image"]]
    negative_images_with_fp = len({record["image_id"] for record in negative_fp})
    negatives = config["expected_validation"]["negative_images"]
    negative_summary = {
        "negative_images": negatives, "false_positives": len(negative_fp),
        "fp_per_negative_image": len(negative_fp) / negatives,
        "images_with_at_least_one_fp": negative_images_with_fp,
        "images_with_at_least_one_fp_fraction": negative_images_with_fp / negatives,
        "images_with_at_least_one_fp_percent": 100 * negative_images_with_fp / negatives,
        "fp_by_predicted_class": {code: sum(record["class_code"] == code for record in negative_fp) for code in CODES},
        "retained_fp_confidence": _quantiles([record["confidence"] for record in negative_fp]),
    }
    approved_negative = selected["negative_image_metrics"]
    if (negative_summary["false_positives"] != approved_negative["negative_fp"]
            or negative_summary["images_with_at_least_one_fp"] != approved_negative["negative_images_with_fp"]
            or negative_summary["fp_per_negative_image"] != approved_negative["negative_fp_per_image"]):
        raise ThresholdError("Detailed negative-only FP analysis does not reproduce the approved selection.")

    summary = {
        "validation_identity": config["expected_validation"],
        "operating_point": {"global_confidence": config["confidence"],
                            "matching_iou": config["matching_iou"], "nms_applicable": False},
        "tp": len(tp_records), "fn": len(false_negatives), "fp": len(fp_records),
        "macro_precision": selected["macro_metrics"]["precision"],
        "macro_recall": selected["macro_metrics"]["recall"],
        "macro_f1": selected["macro_metrics"]["f1"],
        "micro_f1": selected["micro_metrics"]["f1"],
        "fn_taxonomy": _category_counts(false_negatives, len(false_negatives)),
        "fp_taxonomy": _category_counts(fp_records, len(fp_records)),
        "negative_only": negative_summary,
    }
    return {
        "summary": summary, "per_class": per_class,
        "fn_taxonomy": {
            "exclusive_category_priority": config["fn_category_priority"],
            "no_candidate_wording": "no stronger cached candidate above the 0.010 inference floor",
            "aggregate": summary["fn_taxonomy"], "records": false_negatives,
        },
        "fp_taxonomy": {
            "native_end2end_note": (
                "Same-class overlapping unmatched predictions are observable; the cache does not establish "
                "legacy-NMS duplicate semantics or why both outputs survived."
            ),
            "aggregate": summary["fp_taxonomy"], "records": fp_records,
        },
        "object_size": {
            "definition": "normalized detector-box footprint = GT box area / image area; not physical damage area",
            "buckets": config["size_buckets"], "rows": size_rows,
        },
        "aspect_ratio": {
            "definition": "GT box width divided by GT box height; descriptive association only",
            "buckets": config["aspect_ratio_buckets"], "rows": aspect_rows,
        },
        "target_density": {
            "definition": "number of retained target GT boxes in the positive validation image",
            "buckets": config["target_density_buckets"], "rows": density_rows,
        },
        "country": {
            "scope_note": "leakage-reduced grouped validation split; not route-independent",
            "india_d10_warning": "India D10 has only 10 targets in 10 validation images.",
            "rows": country_rows,
        },
        "true_positive_records": tp_records,
    }


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {"count": len(values), "minimum": min(values), "median": median(values),
            "mean": mean(values), "maximum": max(values)}


def _prediction_profile(ground_truth: Mapping[str, Any], cache: Mapping[str, Any],
                        thresholds: Sequence[float]) -> dict[str, Any]:
    gt_by_id = {row["image_id"]: row for row in ground_truth["images"]}
    cache_by_id = {row["image_id"]: row for row in cache["images"]}
    negative_ids = {image_id for image_id, row in gt_by_id.items() if not row["_boxes"]}

    def profile(ids: set[str]) -> list[dict[str, Any]]:
        rows = []
        for threshold in thresholds:
            selected_by_image = {
                image_id: [box for box in cache_by_id[image_id]["_boxes"] if box.confidence >= threshold]
                for image_id in ids
            }
            boxes = [box for values in selected_by_image.values() for box in values]
            rows.append({
                "confidence_at_least": threshold, "images": len(ids), "predictions": len(boxes),
                "predictions_per_image": len(boxes) / len(ids) if ids else 0.,
                "images_with_at_least_one_prediction": sum(bool(values) for values in selected_by_image.values()),
                "by_class": {code: sum(box.class_id == class_id for box in boxes)
                             for class_id, code in enumerate(CODES)},
            })
        return rows

    all_ids = set(gt_by_id)
    return {
        "all_validation_images": profile(all_ids),
        "negative_only_images": profile(negative_ids),
        "positive_images": profile(all_ids - negative_ids),
    }


def _baseline_fn_category(row: Mapping[str, str]) -> str:
    diagnosis = row["diagnosis"]
    if diagnosis == "below_frozen_confidence":
        return "below_confidence_same_class_iou_ge_0.50"
    if diagnosis == "class_confusion":
        return "wrong_class_overlap_iou_ge_0.50"
    if diagnosis == "poor_localization":
        overlap = float(row["diagnostic_iou"])
        return "localization_iou_0.30_to_0.50" if overlap >= .3 else "localization_iou_0.10_to_0.30"
    if diagnosis == "no_matching_candidate":
        return NO_CANDIDATE
    raise ThresholdError(f"Unknown approved YOLOv8 FN category: {diagnosis}")


def _broad_fp_category(category: str) -> str:
    if category in ("negative_image", "negative_only_image"):
        return "negative_only_image"
    if "class_confusion" in category or "wrong_class" in category:
        return "wrong_class_overlap"
    if "localization" in category or category == "nearby_unmatched_pattern":
        return "weak_localization_or_nearby"
    if "duplicate_or_competing" in category or "same_class_overlap_unmatched" in category:
        return "same_class_overlap_unmatched"
    if category in ("background_or_unlabelled_pattern", "positive_image_unmatched_iou_below_0.10"):
        return "positive_image_unmatched_low_overlap"
    raise ThresholdError(f"Unknown FP category for comparison: {category}")


def build_comparison(config: Mapping[str, Any], analysis: Mapping[str, Any],
                     baseline_artifacts: Mapping[str, Any], ground_truth: Mapping[str, Any],
                     cache: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare complete approved validation artifacts at each model's own operating point."""
    baseline_per_class = {row["class_code"]: row for row in baseline_artifacts["per_class"]}
    experiment_per_class = {row["class_code"]: row for row in analysis["per_class"]}
    per_class = []
    for code in CODES:
        old, new = baseline_per_class[code], experiment_per_class[code]
        old_values = {key: int(old[key]) for key in ("tp", "fp", "fn")}
        old_values.update({key: float(old[key]) for key in ("precision", "recall", "f1")})
        per_class.append({
            "class_code": code, "yolov8s": old_values,
            "yolo26s": {key: new[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")},
            "yolo26s_minus_yolov8s": {key: new[key] - old_values[key]
                                      for key in ("tp", "fp", "fn", "precision", "recall", "f1")},
        })

    baseline_fn_by_class: dict[str, Counter[str]] = {code: Counter() for code in CODES}
    for row in baseline_artifacts["false_negatives"]:
        baseline_fn_by_class[row["class_code"]][_baseline_fn_category(row)] += 1
    fn_comparison = []
    for code in CODES:
        new_counts = {row["category"]: row["count"] for row in experiment_per_class[code]["fn_taxonomy"]}
        old_counts = dict(sorted(baseline_fn_by_class[code].items()))
        fn_comparison.append({"class_code": code, "yolov8s_approved": old_counts,
                              "yolo26s": new_counts})

    experiment_fp_broad = Counter(_broad_fp_category(row["category"])
                                  for row in analysis["fp_taxonomy"]["records"])
    baseline_fp_broad = Counter(_broad_fp_category(row["diagnosis"])
                                for row in baseline_artifacts["false_positives"])

    baseline_profile = _prediction_profile(
        baseline_artifacts["ground_truth"], baseline_artifacts["prediction_cache"],
        config["common_confidence_profile"],
    )
    experiment_profile = _prediction_profile(ground_truth, cache, config["common_confidence_profile"])

    baseline_negative_rows = [row for row in baseline_artifacts["false_positives"]
                              if row["diagnosis"] == "negative_image"]
    experiment_negative_rows = [row for row in analysis["fp_taxonomy"]["records"]
                                if row["category"] == "negative_only_image"]
    baseline_negative_by_class = Counter(row["class_code"] for row in baseline_negative_rows)
    experiment_negative_by_class = Counter(row["class_code"] for row in experiment_negative_rows)
    negative_tail_uniformly_lower = all(
        experiment_row["predictions"] < baseline_row["predictions"]
        for baseline_row, experiment_row in zip(
            baseline_profile["negative_only_images"],
            experiment_profile["negative_only_images"],
            strict=True,
        )
    )
    negative_comparison = {
        "scope": "same 983 negative-only validation images; each model retains predictions at its own selected confidence",
        "native_semantics_caveat": (
            "YOLOv8 counts come from its approved NMS-IoU-0.50 cache; YOLO26 counts come from native NMS-free "
            "end-to-end output. Candidate-count comparisons are descriptive, not an isolated NMS ablation."
        ),
        "yolov8s": {
            "selected_confidence": .19, "false_positives": len(baseline_negative_rows),
            "images_with_at_least_one_fp": 96, "fp_per_image": 121 / 983,
            "fp_by_class": {code: baseline_negative_by_class[code] for code in CODES},
            "fp_confidence": _quantiles([float(row["confidence"]) for row in baseline_negative_rows]),
        },
        "yolo26s": {
            "selected_confidence": config["confidence"], "false_positives": len(experiment_negative_rows),
            "images_with_at_least_one_fp": analysis["summary"]["negative_only"]["images_with_at_least_one_fp"],
            "fp_per_image": analysis["summary"]["negative_only"]["fp_per_negative_image"],
            "fp_by_class": {code: experiment_negative_by_class[code] for code in CODES},
            "fp_confidence": _quantiles([row["confidence"] for row in experiment_negative_rows]),
        },
        "yolo26s_minus_yolov8s": {
            "false_positives": len(experiment_negative_rows) - len(baseline_negative_rows),
            "images_with_at_least_one_fp": (
                analysis["summary"]["negative_only"]["images_with_at_least_one_fp"] - 96
            ),
            "fp_by_class": {code: experiment_negative_by_class[code] - baseline_negative_by_class[code]
                            for code in CODES},
        },
        "common_confidence_profiles": {"yolov8s": baseline_profile, "yolo26s": experiment_profile},
        "measured_interpretation": {
            "selected_operating_point_advantage": (
                "YOLO26s produces fewer false positives and affects fewer negative-only images at the two "
                "validation-selected operating points."
            ),
            "uniformly_lower_common_confidence_profile": negative_tail_uniformly_lower,
            "qualification": (
                "The advantage is concentrated in low-to-mid-confidence detections and is not a uniformly "
                "lower high-confidence tail; this is descriptive evidence, not a causal architecture claim."
            ),
        },
    }

    baseline_object_size = {
        (row["class_code"], row["size_bucket"]): row
        for row in baseline_artifacts["object_size"]
    }
    object_size_comparison = []
    for new in analysis["object_size"]["rows"]:
        if new["class_code"] == "ALL":
            continue
        key = (new["class_code"], new["size_bucket"])
        old = baseline_object_size[key]
        old_support, old_recall = int(old["gt_objects"]), float(old["recall"])
        if old_support != new["support"]:
            raise ThresholdError(f"Cross-model object-size support mismatch: {key}")
        object_size_comparison.append({
            "class_code": new["class_code"], "size_bucket": new["size_bucket"],
            "support": new["support"],
            "yolov8s_recall": old_recall, "yolo26s_recall": new["recall"],
            "yolo26s_minus_yolov8s_recall": new["recall"] - old_recall,
            "weak_support": bool(new["weak_support"]) or old["weak_support"].lower() == "true",
        })

    baseline_country = {
        (row["country"], row["class_code"]): row
        for row in baseline_artifacts["country"]
    }
    country_comparison = []
    for new in analysis["country"]["rows"]:
        key = (new["country"], new["class_code"])
        old = baseline_country[key]
        old_values = {
            name: int(old[name]) for name in ("support", "tp", "fp", "fn")
        }
        old_values.update({name: float(old[name]) for name in ("precision", "recall", "f1")})
        if old_values["support"] != new["support"]:
            raise ThresholdError(f"Cross-model country support mismatch: {key}")
        country_comparison.append({
            "country": new["country"], "class_code": new["class_code"],
            "support": new["support"],
            "yolov8s": {name: old_values[name] for name in ("tp", "fp", "fn", "precision", "recall", "f1")},
            "yolo26s": {name: new[name] for name in ("tp", "fp", "fn", "precision", "recall", "f1")},
            "yolo26s_minus_yolov8s": {
                name: new[name] - old_values[name]
                for name in ("tp", "fp", "fn", "precision", "recall", "f1")
            },
            "warning": new["warning"] or old["warning"],
        })

    baseline_total_tp = sum(int(row["tp"]) for row in baseline_artifacts["per_class"])
    baseline_total_fp = sum(int(row["fp"]) for row in baseline_artifacts["per_class"])
    experiment_total_fp = analysis["summary"]["fp"]
    comparison = {
        "schema_version": "experiment2_yolo26s.validation_error_comparison.v1",
        "scope": "validation only; each model at its own validation-selected operating point",
        "internal_test_used": False,
        "aggregate": {
            "yolov8s": {"macro_f1": .5055312316193121, "tp": baseline_total_tp,
                        "fp": baseline_total_fp, "negative_fp": 121,
                        "positive_image_fp": baseline_total_fp - 121},
            "yolo26s": {"macro_f1": analysis["summary"]["macro_f1"], "tp": analysis["summary"]["tp"],
                        "fp": experiment_total_fp, "negative_fp": len(experiment_negative_rows),
                        "positive_image_fp": experiment_total_fp - len(experiment_negative_rows)},
            "yolo26s_minus_yolov8s": {
                "macro_f1": analysis["summary"]["macro_f1"] - .5055312316193121,
                "tp": analysis["summary"]["tp"] - baseline_total_tp,
                "fp": experiment_total_fp - baseline_total_fp,
                "negative_fp": len(experiment_negative_rows) - 121,
                "positive_image_fp": (
                    experiment_total_fp - len(experiment_negative_rows) - (baseline_total_fp - 121)
                ),
            },
        },
        "per_class": per_class, "fn_taxonomy": fn_comparison,
        "fp_taxonomy_broad": {"yolov8s": dict(sorted(baseline_fp_broad.items())),
                              "yolo26s": dict(sorted(experiment_fp_broad.items()))},
        "object_size_recall": object_size_comparison,
        "country": country_comparison,
        "baseline_artifact_manifest_sha256": config["baseline_error_manifest_sha256"],
    }
    return comparison, negative_comparison


def _row(rows: Sequence[Mapping[str, Any]], class_code: str, dimension: str, value: str) -> Mapping[str, Any]:
    return next(row for row in rows if row["class_code"] == class_code and row[dimension] == value)


def decision_support(analysis: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = comparison["aggregate"]
    return {
        "decision": "C",
        "recommendation": "Effectively tied for current model selection; choose by deployment trade-off.",
        "qualification": (
            "This is an operational decision from one validation split, not a statistical equivalence claim. "
            "YOLOv8s leads the frozen macro-F1 objective; YOLO26s has cleaner negative-only behavior and higher "
            "D40 recall/F1."
        ),
        "evidence": {
            "macro_f1_yolov8s": aggregate["yolov8s"]["macro_f1"],
            "macro_f1_yolo26s": aggregate["yolo26s"]["macro_f1"],
            "macro_f1_delta": aggregate["yolo26s_minus_yolov8s"]["macro_f1"],
            "yolo26s_extra_true_positives": aggregate["yolo26s_minus_yolov8s"]["tp"],
            "yolo26s_extra_false_positives": aggregate["yolo26s_minus_yolov8s"]["fp"],
            "yolo26s_negative_fp_reduction": -aggregate["yolo26s_minus_yolov8s"]["negative_fp"],
            "yolo26s_positive_image_fp_increase": aggregate["yolo26s_minus_yolov8s"]["positive_image_fp"],
            "yolo26s_d40_f1_gain": next(
                row["yolo26s_minus_yolov8s"]["f1"] for row in comparison["per_class"]
                if row["class_code"] == "D40"
            ),
        },
        "experiment3_justified": False,
        "experiment3_hypothesis": None,
        "reason_no_experiment3": (
            "The measured differences are a broad operating trade-off across class-specific FP and TP counts, "
            "not one isolated cache-observable failure mode that supports changing a single experimental factor."
        ),
    }


def _summary_markdown(summary: Mapping[str, Any], analysis: Mapping[str, Any],
                      comparison: Mapping[str, Any], negative: Mapping[str, Any]) -> str:
    lines = [
        "# YOLO26s validation-only error analysis", "",
        "Source: the SHA-pinned YOLO26 native end-to-end prediction cache and identical validation GT cache.",
        "No model inference, internal-test data, official unlabelled test data, teacher video, or dataset files were used.",
        "NMS is not applicable to this YOLO26 path; no legacy NMS was introduced.", "",
        "## Fixed operating point", "",
        "Global confidence 0.193; class-aware matching IoU 0.50; cache inference floor 0.010.",
        f"TP {analysis['summary']['tp']}; FP {analysis['summary']['fp']}; FN {analysis['summary']['fn']}.",
        f"Macro precision/recall/F1: {analysis['summary']['macro_precision']:.6f} / "
        f"{analysis['summary']['macro_recall']:.6f} / {analysis['summary']['macro_f1']:.6f}.", "",
        "## Per-class operating errors", "",
        "| Class | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["per_class"]:
        lines.append(f"| {row['class_code']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                     f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")
    lines.extend(["", "## Exclusive FN taxonomy", "",
                  "Priority is applied in the recorded order when more than one cached diagnostic exists.", "",
                  "| Class | Below conf | Matching competition | Loc 0.10–0.30 | Loc 0.30–0.50 | Wrong class | No stronger cached candidate |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    fn_keys = (
        "below_confidence_same_class_iou_ge_0.50", "same_class_matching_competition_iou_ge_0.50",
        "localization_iou_0.10_to_0.30", "localization_iou_0.30_to_0.50",
        "wrong_class_overlap_iou_ge_0.50", NO_CANDIDATE,
    )
    for row in analysis["per_class"]:
        counts = {value["category"]: value["count"] for value in row["fn_taxonomy"]}
        lines.append(f"| {row['class_code']} | " + " | ".join(str(counts.get(key, 0)) for key in fn_keys) + " |")
    lines.extend(["", "The final category means exactly: no stronger cached candidate above the 0.010 inference floor. "
                  "It does not mean that the model saw nothing.", "", "## Negative-only behavior", ""])
    new_negative, old_negative = negative["yolo26s"], negative["yolov8s"]
    lines.extend([
        f"YOLO26s: {new_negative['false_positives']} FP on 983 negatives, "
        f"{new_negative['images_with_at_least_one_fp']} images affected, {new_negative['fp_per_image']:.6f} FP/image.",
        f"YOLOv8s: {old_negative['false_positives']} FP, "
        f"{old_negative['images_with_at_least_one_fp']} images affected, {old_negative['fp_per_image']:.6f} FP/image.",
        "These are descriptive comparisons across different native postprocessing paths, not an NMS ablation.", "",
        "## Why YOLO26s macro-F1 is lower", "",
    ])
    delta = comparison["aggregate"]["yolo26s_minus_yolov8s"]
    lines.append(
        f"YOLO26s gains {delta['tp']} total TP and removes {-delta['negative_fp']} negative-image FP, but adds "
        f"{delta['positive_image_fp']} FP on positive images ({delta['fp']} net FP). Fixed-point macro-F1 is therefore "
        "reduced by class-level precision losses even though ranking-based validation mAP is slightly higher. "
        "D00, D10 and D20 F1 are lower; D40 is higher because its recall gain outweighs its FP increase."
    )
    lines.extend(["", "mAP integrates ranked behavior across confidence/IoU settings, whereas this macro-F1 gives each "
                  "class equal weight at one frozen confidence. The two measurements need not rank models identically.",
                  "", "## Object characteristics", ""])
    for code in CODES:
        small = _row(analysis["object_size"]["rows"], code, "size_bucket", "small")
        medium = _row(analysis["object_size"]["rows"], code, "size_bucket", "medium")
        large = _row(analysis["object_size"]["rows"], code, "size_bucket", "large")
        lines.append(f"- {code} recall by small/medium/large footprint: {small['recall']:.3f} / "
                     f"{medium['recall']:.3f} / {large['recall']:.3f}.")
    lines.extend(["", "These are descriptive associations with detector-box geometry, not physical damage area or causes.",
                  "India D10 remains only 10 targets in 10 images; no strong country-specific conclusion is supported.",
                  "", "## Decision support", "",
                  f"**{summary['model_selection']['decision']}. {summary['model_selection']['recommendation']}**",
                  summary["model_selection"]["qualification"],
                  "A new Experiment 3 is not justified by this cache analysis; no experiment was implemented."])
    return "\n".join(lines) + "\n"


def _git_provenance(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", "--no-optional-locks", *args], cwd=root,
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise ThresholdError(f"Git provenance failed: {' '.join(args)}: {result.stderr.strip()}")
        return result.stdout.strip()
    paths = [CONFIG, Path(__file__).relative_to(ROOT), Path("tests/test_experiment2_error_analysis.py"),
             Path("road-damage-project-docs/EXPERIMENT2_YOLO26S_RUNBOOK.md")]
    sources = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise ThresholdError(f"Required error-analysis source is missing: {relative}")
        sources.append({"path": relative.as_posix(), "sha256": shared.digest(path), "size_bytes": path.stat().st_size})
    status = git("status", "--porcelain", "--untracked-files=all").splitlines()
    return {"head_commit": git("rev-parse", "HEAD"), "worktree_clean": not status,
            "git_status_porcelain": status, "source_files": sources,
            "source_fingerprint_sha256": hashlib.sha256(shared.json_bytes(sources)).hexdigest()}


def run(root: Path = ROOT) -> Path:
    """Write aggregate and record-level validation analysis; never run model inference."""
    if root.absolute() != ROOT.absolute():
        raise ThresholdError("Canonical error analysis does not accept a project-root override.")
    config, ground_truth, cache, selected, baseline_artifacts = load_sources(root)
    output = shared.checked_path(Path(config["output"]), OUTPUT, root)
    if output.exists():
        raise ThresholdError(f"Refusing to overwrite existing error analysis: {output}")
    analysis = analyze(config, ground_truth, cache, selected)
    comparison, negative = build_comparison(config, analysis, baseline_artifacts, ground_truth, cache)
    model_selection = decision_support(analysis, comparison)
    summary = {
        "schema_version": config["schema_version"], "validation_only": True,
        "model_inference_rerun": False, "internal_test_accessed": False,
        "source_cache_sha256": config["prediction_sha256"],
        "ground_truth_cache_sha256": config["ground_truth_sha256"],
        "selected_operating_point_sha256": config["selected_operating_point_sha256"],
        "metrics": analysis["summary"], "model_selection": model_selection,
    }
    payloads = {
        "summary.json": summary,
        "per_class_error_taxonomy.json": analysis["per_class"],
        "fn_taxonomy.json": analysis["fn_taxonomy"],
        "fp_taxonomy.json": analysis["fp_taxonomy"],
        "object_size_analysis.json": analysis["object_size"],
        "aspect_ratio_analysis.json": analysis["aspect_ratio"],
        "target_density_analysis.json": analysis["target_density"],
        "country_breakdown.json": analysis["country"],
        "yolov8_vs_yolo26_comparison.json": comparison,
        "negative_image_comparison.json": negative,
    }
    provenance = _git_provenance(root)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, payload in payloads.items():
            (output / name).write_bytes(shared.json_bytes(payload))
        (output / "error_analysis_summary.md").write_text(
            _summary_markdown(summary, analysis, comparison, negative), encoding="utf-8"
        )
        for name, payload in payloads.items():
            if json.loads((output / name).read_text(encoding="utf-8")) != payload:
                raise ThresholdError(f"Persisted analysis JSON failed deterministic replay: {name}")
        artifact_names = [*payloads, "error_analysis_summary.md"]
        artifact_hashes = {name: shared.digest(output / name) for name in sorted(artifact_names)}
        manifest = {
            "schema_version": "experiment2_yolo26s.error_analysis_manifest.v1", "status": "COMPLETED",
            "validation_only": True, "model_inference_executed": False,
            "internal_test_data_accessed": False, "official_unlabelled_test_used": False,
            "teacher_video_used": False, "threshold_tuning_performed": False,
            "model_or_weights_accessed": False, "dataset_files_accessed": False,
            "source_artifacts_sha256": {
                config[key]: config[sha_key] for key, sha_key in (
                    ("ground_truth_cache", "ground_truth_sha256"),
                    ("prediction_cache", "prediction_sha256"),
                    ("selected_operating_point", "selected_operating_point_sha256"),
                    ("source_manifest", "source_manifest_sha256"),
                    ("source_completion", "source_completion_sha256"),
                    ("runtime_architecture", "runtime_architecture_sha256"),
                )
            },
            "baseline_error_manifest_sha256": config["baseline_error_manifest_sha256"],
            "frozen_operating_point": {"confidence": .193, "matching_iou": .5,
                                       "nms_applicable": False, "cache_floor": .01},
            "counts": {"tp": analysis["summary"]["tp"], "fp": analysis["summary"]["fp"],
                       "fn": analysis["summary"]["fn"]},
            "git_provenance": provenance, "analysis_config_sha256": shared.digest(root / CONFIG),
            "analysis_code_sha256": shared.digest(Path(__file__)), "artifacts": artifact_hashes,
        }
        manifest_path = output / "analysis_manifest.json"
        manifest_path.write_bytes(shared.json_bytes(manifest))
        # Verify every persisted payload again before publishing completion.
        for name, expected in artifact_hashes.items():
            if shared.digest(output / name) != expected:
                raise ThresholdError(f"Persisted artifact changed before completion: {name}")
        receipt = {
            "schema_version": "experiment2_yolo26s.error_analysis_receipt.v1", "status": "COMPLETED",
            "validation_only": True, "model_inference_executed": False,
            "internal_test_data_accessed": False, "prediction_cache_sha256": config["prediction_sha256"],
            "analysis_manifest_sha256": shared.digest(manifest_path), "artifacts": artifact_hashes,
            "model_selection_decision": model_selection["decision"],
            "experiment3_justified": model_selection["experiment3_justified"],
        }
        (output / "completion.json").write_bytes(shared.json_bytes(receipt))
    except BaseException:
        # A partial directory contains no completion receipt and cannot be mistaken for a completed analysis.
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args(argv)
    try:
        output = run()
    except (ThresholdError, OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"Experiment 2 validation error analysis stopped: {exc}", file=sys.stderr)
        return 2
    print(f"Experiment 2 validation-only error analysis complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
