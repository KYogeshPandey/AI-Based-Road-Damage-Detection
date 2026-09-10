"""Deterministic validation-only error analysis from frozen prediction caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.evaluation import select_threshold as shared  # noqa: E402
from road_damage.evaluation.threshold_metrics import Box, ThresholdError, iou, sweep_cache  # noqa: E402

ROOT = shared.ROOT
CONFIG = Path("configs/evaluation/baseline_public_v1_error_analysis.yaml")
SOURCE = Path("outputs/evaluation/baseline_public_v1_threshold_selection")
OUTPUT = Path("outputs/evaluation/baseline_public_v1_error_analysis")
CODES = ("D00", "D10", "D20", "D40")
BACKGROUND = "BACKGROUND"
EXPECTED_CONFIG: dict[str, Any] = {
    "schema_version": "baseline_public_v1.validation_error_analysis.v1",
    "source": SOURCE.as_posix(),
    "ground_truth_cache": "ground_truth_cache.json",
    "ground_truth_sha256": "7fed3874d55ccd619c5e85427376ad9cbdd9d6f9706d1856cd77a56cb287b42e",
    "prediction_cache": "predictions_nms_50.json",
    "prediction_sha256": "137a1b1ef8af88fe2f45cb59c771995682c041c39422d1d757d5d6283ea6d9b3",
    "selected_operating_point": "selected_operating_point.json",
    "selected_operating_point_sha256": "5220c1ebcc17b99b5e1d46076024a4c592a5f636fe072a42759431a29f0eea45",
    "source_manifest": "evaluation_manifest.json",
    "source_manifest_sha256": "8950cd975469d89344a6d26903dce3d48314eaded4ee93bdf10649ddfffc5bdb",
    "model_sha256": shared.FROZEN_MODEL_SHA256,
    "confidence": .19, "nms_iou": .5, "matching_iou": .5,
    "localization_iou_floor": .1, "high_confidence": .5,
    "size_buckets": [
        {"name": "small", "minimum_inclusive": 0., "maximum_exclusive": .01},
        {"name": "medium", "minimum_inclusive": .01, "maximum_exclusive": .05},
        {"name": "large", "minimum_inclusive": .05, "maximum_exclusive": 1.0000001},
    ],
    "ranked_examples_per_category": 25,
    "expected_counts": {"images": 2602, "targets": 3377, "negative_images": 983},
    "classes": {str(k): v for k, v in shared.CLASS_NAMES.items()},
    "output": OUTPUT.as_posix(), "validation_only": True,
    "internal_test_access_permitted": False, "teacher_video_used": False,
}


@dataclass(frozen=True)
class DetailedMatch:
    prediction_index: int
    ground_truth_index: int
    overlap_iou: float


def load_config(root: Path = ROOT) -> dict[str, Any]:
    path = shared.checked_path(CONFIG, CONFIG, root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if shared.json_bytes(value) != shared.json_bytes(EXPECTED_CONFIG):
        raise ThresholdError("Validation error-analysis configuration changed or contains unknown keys.")
    return value


def _artifact(root: Path, config: Mapping[str, Any], key: str, sha_key: str) -> dict[str, Any]:
    name = config[key]
    if not isinstance(name, str) or Path(name).name != name:
        raise ThresholdError("Error-analysis input must be a pinned local artifact filename.")
    relative = SOURCE / name
    path = shared.checked_path(relative, relative, root)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config[sha_key]:
        raise ThresholdError(f"Pinned validation artifact changed: {name}")
    return json.loads(payload)


def _country(image_id: str) -> str:
    match = re.fullmatch(r"(India|Japan)_\d+\.jpg", image_id)
    if not match:
        raise ThresholdError(f"Unexpected validation image identifier: {image_id}")
    return match.group(1)


def _boxes(values: Sequence[Mapping[str, Any]], width: int, height: int, *, prediction: bool) -> list[Box]:
    result = []
    for value in values:
        if set(value) != {"class_id", "confidence", "xyxy"}:
            raise ThresholdError("Unexpected validation cache box schema.")
        box = Box(value["class_id"], tuple(value["xyxy"]), value["confidence"])
        if box.xyxy[2] > width + 1e-4 or box.xyxy[3] > height + 1e-4:
            raise ThresholdError("Validation cache box exceeds image bounds.")
        if prediction and box.confidence < .01:
            raise ThresholdError("Prediction cache violates its declared confidence floor.")
        if not prediction and box.confidence != 1.:
            raise ThresholdError("Ground-truth cache contains a non-unit confidence.")
        result.append(box)
    return result


def load_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read only four completed validation artifacts; never discover dataset files or load a model."""
    config = load_config(root)
    manifest = _artifact(root, config, "source_manifest", "source_manifest_sha256")
    gt = _artifact(root, config, "ground_truth_cache", "ground_truth_sha256")
    cache = _artifact(root, config, "prediction_cache", "prediction_sha256")
    selected = _artifact(root, config, "selected_operating_point", "selected_operating_point_sha256")
    if (manifest.get("status") != "complete" or manifest.get("identity", {}).get("validation_only") is not True
            or manifest.get("identity", {}).get("model_sha256") != config["model_sha256"]
            or manifest.get("artifacts", {}).get(config["ground_truth_cache"]) != config["ground_truth_sha256"]
            or manifest.get("prediction_cache_sha256", {}).get(config["prediction_cache"]) != config["prediction_sha256"]):
        raise ThresholdError("Threshold-selection manifest is incomplete or has the wrong validation/model identity.")
    if (gt.get("validation_only") is not True or gt.get("counts") != config["expected_counts"]
            or cache.get("validation_only") is not True or cache.get("confidence_floor") != .01
            or cache.get("nms_iou") != config["nms_iou"]
            or selected.get("validation_only") is not True
            or selected.get("model_sha256") != config["model_sha256"]
            or selected.get("selected_global_confidence") != config["confidence"]
            or selected.get("selected_nms_iou") != config["nms_iou"]
            or selected.get("matching_iou") != config["matching_iou"]):
        raise ThresholdError("Frozen validation cache or selected operating point identity mismatch.")
    gt_by_id = {r.get("image_id"): r for r in gt.get("images", [])}
    cache_by_id = {r.get("image_id"): r for r in cache.get("images", [])}
    if (len(gt_by_id) != config["expected_counts"]["images"] or len(cache_by_id) != len(gt_by_id)
            or set(gt_by_id) != set(cache_by_id)):
        raise ThresholdError("Validation caches have duplicate, missing or extra image identities.")
    target_count = negative_count = 0
    for image_id in sorted(gt_by_id):
        record, prediction = gt_by_id[image_id], cache_by_id[image_id]
        country = _country(image_id)
        expected_image = f"data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/images/val/{image_id}"
        expected_label = f"data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/labels/val/{Path(image_id).stem}.txt"
        if (record.get("image_path") != expected_image or record.get("label_path") != expected_label
                or type(record.get("width")) is not int or record["width"] <= 0
                or type(record.get("height")) is not int or record["height"] <= 0
                or prediction.get("nms_iou") != .5):
            raise ThresholdError("Cache is not the canonical India/Japan validation split.")
        ground_truth = _boxes(record.get("ground_truth", []), record["width"], record["height"], prediction=False)
        predictions = _boxes(prediction.get("predictions", []), record["width"], record["height"], prediction=True)
        record["country"], record["_boxes"] = country, ground_truth
        prediction["_boxes"] = predictions
        target_count += len(ground_truth)
        negative_count += not ground_truth
    if {"images": len(gt_by_id), "targets": target_count, "negative_images": negative_count} != config["expected_counts"]:
        raise ThresholdError("Validation cache counts do not reconcile with the pinned record.")
    truth = {name: gt_by_id[name]["_boxes"] for name in gt_by_id}
    predictions = {name: cache_by_id[name]["_boxes"] for name in cache_by_id}
    replay = sweep_cache(truth, predictions, .5, [.19], .5)[0]
    if shared.json_bytes(replay) != shared.json_bytes(selected["metrics"]):
        raise ThresholdError("Frozen validation metrics do not reproduce from the pinned caches.")
    return config, gt, cache, selected


def primary_matches(ground_truth: Sequence[Box], predictions: Sequence[Box], matching_iou: float) -> list[DetailedMatch]:
    """Indexed equivalent of the approved class-aware matcher for audit records."""
    unmatched = set(range(len(ground_truth)))
    matches = []
    order = sorted(range(len(predictions)), key=lambda i: (
        -predictions[i].confidence, predictions[i].class_id, predictions[i].xyxy, i))
    for prediction_index in order:
        prediction = predictions[prediction_index]
        candidates = [(iou(prediction, ground_truth[i]), i) for i in unmatched
                      if ground_truth[i].class_id == prediction.class_id]
        candidates = [pair for pair in candidates if pair[0] >= matching_iou]
        if candidates:
            overlap, gt_index = min(candidates, key=lambda pair: (
                -pair[0], ground_truth[pair[1]].xyxy, pair[1]))
            unmatched.remove(gt_index)
            matches.append(DetailedMatch(prediction_index, gt_index, overlap))
    return matches


def _residual_pairs(ground_truth: Sequence[Box], predictions: Sequence[Box], gt_indices: set[int],
                    prediction_indices: set[int], *, lower: float, upper: float,
                    same_class: bool) -> list[DetailedMatch]:
    candidates = []
    for pi in prediction_indices:
        for gi in gt_indices:
            same = predictions[pi].class_id == ground_truth[gi].class_id
            overlap = iou(predictions[pi], ground_truth[gi])
            if same is same_class and lower <= overlap < upper:
                candidates.append((overlap, pi, gi))
    candidates.sort(key=lambda item: (-item[0], -predictions[item[1]].confidence,
                                      predictions[item[1]].class_id, predictions[item[1]].xyxy,
                                      ground_truth[item[2]].class_id, ground_truth[item[2]].xyxy,
                                      item[1], item[2]))
    result, used_predictions, used_gt = [], set(), set()
    for overlap, pi, gi in candidates:
        if pi not in used_predictions and gi not in used_gt:
            used_predictions.add(pi)
            used_gt.add(gi)
            result.append(DetailedMatch(pi, gi, overlap))
    return result


def normalized_area(box: Box, width: int, height: int) -> float:
    x1, y1, x2, y2 = box.xyxy
    return (x2 - x1) * (y2 - y1) / (width * height)


def size_bucket(area: float, config: Mapping[str, Any]) -> str:
    for bucket in config["size_buckets"]:
        if bucket["minimum_inclusive"] <= area < bucket["maximum_exclusive"]:
            return bucket["name"]
    raise ThresholdError(f"Normalized box area outside configured buckets: {area}")


def _metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": "", "p10": "", "p25": "", "median": "",
                "p75": "", "p90": "", "maximum": "", "mean": ""}
    ordered = sorted(values)
    def q(fraction: float) -> float:
        return ordered[math.floor(fraction * (len(ordered) - 1))]
    return {"count": len(values), "minimum": ordered[0], "p10": q(.10), "p25": q(.25),
            "median": q(.50), "p75": q(.75), "p90": q(.90), "maximum": ordered[-1],
            "mean": mean(values)}


def _format_optional_float(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def analyze(config: Mapping[str, Any], gt: Mapping[str, Any], cache: Mapping[str, Any]) -> dict[str, Any]:
    """Compute fixed-point failure structure without model, image or label access."""
    gt_by_id = {r["image_id"]: r for r in gt["images"]}
    cache_by_id = {r["image_id"]: r for r in cache["images"]}
    class_counts = {c: {"support": 0, "positive_images": 0, "predictions": 0, "tp": 0, "fp": 0,
                        "fn": 0, "confusions": 0, "localization_errors": 0, "high_confidence_fp": 0}
                    for c in range(4)}
    country_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"images": 0, "positive_images": 0, "support": 0, "predictions": 0,
                 "tp": 0, "fp": 0, "fn": 0})
    size_counts: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: {"gt_objects": 0, "detected_gt_objects": 0, "retained_predictions": 0,
                 "true_positive_prediction_boxes": 0})
    confidence_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    localization_errors: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    confusion_counts: dict[tuple[str, str], int] = defaultdict(int)
    image_errors: dict[str, dict[str, Any]] = {}
    high_confidence_fn_diagnostics = 0

    for image_id in sorted(gt_by_id):
        record, prediction_record = gt_by_id[image_id], cache_by_id[image_id]
        country, width, height = record["country"], record["width"], record["height"]
        ground_truth: list[Box] = record["_boxes"]
        raw_predictions: list[Box] = prediction_record["_boxes"]
        retained = [p for p in raw_predictions if p.confidence >= config["confidence"]]
        matches = primary_matches(ground_truth, retained, config["matching_iou"])
        matched_predictions = {m.prediction_index for m in matches}
        matched_gt = {m.ground_truth_index for m in matches}
        residual_predictions = set(range(len(retained))) - matched_predictions
        residual_gt = set(range(len(ground_truth))) - matched_gt
        confusions = _residual_pairs(ground_truth, retained, residual_gt, residual_predictions,
                                     lower=config["matching_iou"], upper=math.inf, same_class=False)
        confusion_predictions = {m.prediction_index for m in confusions}
        confusion_gt = {m.ground_truth_index for m in confusions}
        localization = _residual_pairs(
            ground_truth, retained, residual_gt - confusion_gt, residual_predictions - confusion_predictions,
            lower=config["localization_iou_floor"], upper=config["matching_iou"], same_class=True)
        localization_predictions = {m.prediction_index for m in localization}
        localization_gt = {m.ground_truth_index for m in localization}
        match_by_prediction = {m.prediction_index: m for m in matches}
        match_by_gt = {m.ground_truth_index: m for m in matches}
        confusion_by_prediction = {m.prediction_index: m for m in confusions}
        confusion_by_gt = {m.ground_truth_index: m for m in confusions}
        localization_by_prediction = {m.prediction_index: m for m in localization}
        localization_by_gt = {m.ground_truth_index: m for m in localization}

        for c in range(4):
            present = any(b.class_id == c for b in ground_truth)
            country_counts[(country, c)]["images"] += 1
            country_counts[(country, c)]["positive_images"] += present
            class_counts[c]["positive_images"] += present
        for gi, ground in enumerate(ground_truth):
            class_counts[ground.class_id]["support"] += 1
            country_counts[(country, ground.class_id)]["support"] += 1
            area = normalized_area(ground, width, height)
            bucket = size_bucket(area, config)
            size_counts[(ground.class_id, bucket)]["gt_objects"] += 1
            if gi in matched_gt:
                size_counts[(ground.class_id, bucket)]["detected_gt_objects"] += 1
        for pi, predicted_box in enumerate(retained):
            c = predicted_box.class_id
            class_counts[c]["predictions"] += 1
            country_counts[(country, c)]["predictions"] += 1
            bucket = size_bucket(normalized_area(predicted_box, width, height), config)
            size_counts[(c, bucket)]["retained_predictions"] += 1
            outcome = "TP" if pi in matched_predictions else "FP"
            confidence_values[(outcome, c)].append(predicted_box.confidence)
            if pi in matched_predictions:
                size_counts[(c, bucket)]["true_positive_prediction_boxes"] += 1

        for match in matches:
            predicted_box, ground = retained[match.prediction_index], ground_truth[match.ground_truth_index]
            c = ground.class_id
            class_counts[c]["tp"] += 1
            country_counts[(country, c)]["tp"] += 1
            confusion_counts[(CODES[c], CODES[c])] += 1
            iou_rows.append({"image_id": image_id, "validation_image_path": record["image_path"],
                "country": country, "class_id": c, "class_code": CODES[c],
                "prediction_confidence": predicted_box.confidence, "iou": match.overlap_iou,
                "gt_normalized_bbox_area": normalized_area(ground, width, height),
                "gt_size_bucket": size_bucket(normalized_area(ground, width, height), config)})

        for match in confusions:
            predicted_box, ground = retained[match.prediction_index], ground_truth[match.ground_truth_index]
            class_counts[predicted_box.class_id]["confusions"] += 1
            confusion_counts[(CODES[ground.class_id], CODES[predicted_box.class_id])] += 1

        for match in localization:
            predicted_box, ground = retained[match.prediction_index], ground_truth[match.ground_truth_index]
            class_counts[ground.class_id]["localization_errors"] += 1
            localization_errors.append({"image_id": image_id, "validation_image_path": record["image_path"],
                "country": country, "class_id": ground.class_id, "class_code": CODES[ground.class_id],
                "prediction_confidence": predicted_box.confidence, "iou": match.overlap_iou,
                "gt_xyxy": json.dumps(ground.xyxy), "prediction_xyxy": json.dumps(predicted_box.xyxy),
                "gt_normalized_bbox_area": normalized_area(ground, width, height),
                "gt_size_bucket": size_bucket(normalized_area(ground, width, height), config)})

        for gi in residual_gt:
            ground = ground_truth[gi]
            c = ground.class_id
            class_counts[c]["fn"] += 1
            country_counts[(country, c)]["fn"] += 1
            diagnostic_prediction: Box | None = None
            diagnostic_iou = 0.
            if gi in confusion_by_gt:
                pair = confusion_by_gt[gi]
                diagnostic_prediction, diagnostic_iou, diagnosis = retained[pair.prediction_index], pair.overlap_iou, "class_confusion"
            elif gi in localization_by_gt:
                pair = localization_by_gt[gi]
                diagnostic_prediction, diagnostic_iou, diagnosis = retained[pair.prediction_index], pair.overlap_iou, "poor_localization"
            else:
                below = [(p.confidence, iou(p, ground), p) for p in raw_predictions
                         if p.class_id == c and p.confidence < config["confidence"] and iou(p, ground) >= config["matching_iou"]]
                if below:
                    confidence, diagnostic_iou, diagnostic_prediction = max(
                        below, key=lambda item: (item[0], item[1], tuple(-v for v in item[2].xyxy)))
                    diagnosis = "below_frozen_confidence"
                else:
                    candidates = [(iou(p, ground), p.confidence, p) for p in retained]
                    if candidates:
                        diagnostic_iou, _, diagnostic_prediction = max(
                            candidates, key=lambda item: (item[0], item[1], tuple(-v for v in item[2].xyxy)))
                    diagnosis = "no_matching_candidate"
            diagnostic_confidence = "" if diagnostic_prediction is None else diagnostic_prediction.confidence
            if diagnosis in ("class_confusion", "poor_localization") and diagnostic_confidence >= config["high_confidence"]:
                high_confidence_fn_diagnostics += 1
            area = normalized_area(ground, width, height)
            false_negatives.append({"image_id": image_id, "validation_image_path": record["image_path"],
                "country": country, "gt_index": gi, "class_id": c, "class_code": CODES[c],
                "gt_xyxy": json.dumps(ground.xyxy), "gt_normalized_bbox_area": area,
                "gt_size_bucket": size_bucket(area, config), "diagnosis": diagnosis,
                "diagnostic_predicted_class": "" if diagnostic_prediction is None else CODES[diagnostic_prediction.class_id],
                "diagnostic_confidence": diagnostic_confidence, "diagnostic_iou": diagnostic_iou})

        for pi in residual_predictions:
            predicted_box = retained[pi]
            c = predicted_box.class_id
            class_counts[c]["fp"] += 1
            country_counts[(country, c)]["fp"] += 1
            if predicted_box.confidence >= config["high_confidence"]:
                class_counts[c]["high_confidence_fp"] += 1
            if pi in confusion_by_prediction:
                pair = confusion_by_prediction[pi]
                diagnosis, related_class, best_iou = "class_confusion", CODES[ground_truth[pair.ground_truth_index].class_id], pair.overlap_iou
            elif pi in localization_by_prediction:
                pair = localization_by_prediction[pi]
                diagnosis, related_class, best_iou = "poor_localization", CODES[ground_truth[pair.ground_truth_index].class_id], pair.overlap_iou
            else:
                overlaps = [(iou(predicted_box, ground), ground.class_id) for ground in ground_truth]
                best_iou, related_id = max(overlaps, default=(0., -1))
                related_class = "" if related_id < 0 else CODES[related_id]
                if not ground_truth:
                    diagnosis = "negative_image"
                elif best_iou >= config["matching_iou"]:
                    diagnosis = "duplicate_or_competing_overlap"
                elif best_iou >= config["localization_iou_floor"]:
                    diagnosis = "nearby_unmatched_pattern"
                else:
                    diagnosis = "background_or_unlabelled_pattern"
            false_positives.append({"image_id": image_id, "validation_image_path": record["image_path"],
                "country": country, "prediction_index": pi, "class_id": c, "class_code": CODES[c],
                "confidence": predicted_box.confidence, "prediction_xyxy": json.dumps(predicted_box.xyxy),
                "prediction_normalized_bbox_area": normalized_area(predicted_box, width, height),
                "prediction_size_bucket": size_bucket(normalized_area(predicted_box, width, height), config),
                "diagnosis": diagnosis, "related_gt_class": related_class, "best_gt_iou": best_iou,
                "high_confidence": predicted_box.confidence >= config["high_confidence"]})

        for gi in residual_gt - confusion_gt:
            confusion_counts[(CODES[ground_truth[gi].class_id], BACKGROUND)] += 1
        for pi in residual_predictions - confusion_predictions:
            confusion_counts[(BACKGROUND, CODES[retained[pi].class_id])] += 1

        image_fp = [r for r in false_positives if r["image_id"] == image_id]
        image_fn = [r for r in false_negatives if r["image_id"] == image_id]
        if image_fp or image_fn:
            image_errors[image_id] = {"image_id": image_id, "validation_image_path": record["image_path"],
                "country": country, "fp": len(image_fp), "fn": len(image_fn),
                "total_errors": len(image_fp) + len(image_fn),
                "max_fp_confidence": max((r["confidence"] for r in image_fp), default=0.),
                "diagnoses": ";".join(sorted({r["diagnosis"] for r in [*image_fp, *image_fn]}))}
        if not ground_truth and image_fp:
            negative_rows.append({"image_id": image_id, "validation_image_path": record["image_path"],
                "country": country, "false_positives": len(image_fp),
                "maximum_confidence": max(r["confidence"] for r in image_fp),
                "predicted_classes": ";".join(sorted({r["class_code"] for r in image_fp})),
                "prediction_indices": ";".join(str(r["prediction_index"]) for r in image_fp)})

    # Exact primary counts must still equal the approved frozen selected artifact.
    selected_counts = [[class_counts[c][key] for key in ("tp", "fp", "fn")] for c in range(4)]
    per_class_rows = []
    for c, code in enumerate(CODES):
        values = class_counts[c]
        per_class_rows.append({"class_id": c, "class_code": code, "class_name": config["classes"][str(c)],
            **values, **_metric(values["tp"], values["fp"], values["fn"])})

    country_rows = []
    for country in ("India", "Japan"):
        for c, code in enumerate(CODES):
            values = country_counts[(country, c)]
            warning = "LOW SUPPORT — no strong country-specific conclusion" if country == "India" and code == "D10" else ""
            country_rows.append({"country": country, "class_id": c, "class_code": code, **values,
                                 **_metric(values["tp"], values["fp"], values["fn"]), "warning": warning})
        class_rows = [country_counts[(country, c)] for c in range(4)]
        tp, fp, fn = (sum(row[k] for row in class_rows) for k in ("tp", "fp", "fn"))
        country_rows.append({"country": country, "class_id": "", "class_code": "ALL",
            "images": class_rows[0]["images"],
            "positive_images": sum(any(b.class_id == c for b in gt_by_id[name]["_boxes"] for c in range(4))
                                   for name in gt_by_id if gt_by_id[name]["country"] == country),
            "support": sum(row["support"] for row in class_rows),
            "predictions": sum(row["predictions"] for row in class_rows),
            **_metric(tp, fp, fn), "warning": ""})

    object_size_rows = []
    for c, code in enumerate(CODES):
        for bucket in config["size_buckets"]:
            values = size_counts[(c, bucket["name"])]
            object_size_rows.append({"class_id": c, "class_code": code, "size_bucket": bucket["name"],
                "minimum_normalized_bbox_area_inclusive": bucket["minimum_inclusive"],
                "maximum_normalized_bbox_area_exclusive": bucket["maximum_exclusive"], **values,
                "recall": values["detected_gt_objects"] / values["gt_objects"] if values["gt_objects"] else 0.,
                "weak_support": values["gt_objects"] < 30})

    confidence_rows = []
    for outcome in ("TP", "FP"):
        for c, code in enumerate(CODES):
            confidence_rows.append({"outcome": outcome, "class_id": c, "class_code": code,
                                    **_quantiles(confidence_values[(outcome, c)])})

    matrix_rows = []
    for truth_code in (*CODES, BACKGROUND):
        for predicted_code in (*CODES, BACKGROUND):
            matrix_rows.append({"true_class": truth_code, "predicted_class": predicted_code,
                                "count": confusion_counts[(truth_code, predicted_code)]})

    ranked = _rank_examples(config, false_negatives, false_positives, localization_errors,
                            negative_rows, image_errors)
    negatives = gt["counts"]["negative_images"]
    negative_fp = sum(r["false_positives"] for r in negative_rows)
    summary = {"validation_images": len(gt_by_id), "validation_targets": gt["counts"]["targets"],
        "negative_images": negatives, "negative_fp": negative_fp,
        "negative_fp_per_image": negative_fp / negatives if negatives else 0.,
        "negative_images_with_fp": len(negative_rows),
        "negative_images_with_fp_fraction": len(negative_rows) / negatives if negatives else 0.,
        "localization_errors": len(localization_errors),
        "class_confusion_pairs": sum(
            count for (truth_code, predicted_code), count in confusion_counts.items()
            if truth_code != BACKGROUND and predicted_code != BACKGROUND and truth_code != predicted_code),
        "high_confidence_false_positives": sum(r["high_confidence"] for r in false_positives),
        "high_confidence_fn_diagnostics": high_confidence_fn_diagnostics,
        "primary_counts": {CODES[c]: dict(zip(("tp", "fp", "fn"), selected_counts[c], strict=True)) for c in range(4)}}
    return {"summary": summary, "per_class_errors": per_class_rows, "country_errors": country_rows,
            "false_negatives": false_negatives, "false_positives": false_positives,
            "localization_errors": localization_errors, "negative_image_false_positives": negative_rows,
            "object_size_metrics": object_size_rows, "iou_distribution": iou_rows,
            "confidence_distributions": confidence_rows, "class_confusion_matrix": matrix_rows,
            "ranked_examples": ranked}


def _rank_examples(config: Mapping[str, Any], false_negatives: Sequence[dict[str, Any]],
                   false_positives: Sequence[dict[str, Any]], localization: Sequence[dict[str, Any]],
                   negatives: Sequence[dict[str, Any]], image_errors: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable validation-image queues; no image pixels are read or copied."""
    limit = config["ranked_examples_per_category"]
    groups: list[tuple[str, Iterable[dict[str, Any]], Any]] = []
    diagnosis_order = {"class_confusion": 0, "poor_localization": 1,
                       "below_frozen_confidence": 2, "no_matching_candidate": 3}
    for code in CODES:
        rows = [r for r in false_negatives if r["class_code"] == code]
        groups.append((f"{code}_FALSE_NEGATIVE", rows, lambda r: (
            diagnosis_order[r["diagnosis"]], -float(r["diagnostic_confidence"] or 0.),
            -r["diagnostic_iou"], r["image_id"], r["gt_index"])))
    groups.append(("HIGH_CONFIDENCE_FALSE_POSITIVE",
                   [r for r in false_positives if r["high_confidence"]],
                   lambda r: (-r["confidence"], r["image_id"], r["prediction_index"])))
    groups.append(("HIGH_CONFIDENCE_FN_DIAGNOSTIC",
                   [r for r in false_negatives
                    if r["diagnosis"] in ("class_confusion", "poor_localization")
                    and float(r["diagnostic_confidence"] or 0.) >= config["high_confidence"]],
                   lambda r: (-float(r["diagnostic_confidence"]), -r["diagnostic_iou"],
                              r["image_id"], r["gt_index"])))
    groups.append(("NEGATIVE_ROAD_FALSE_POSITIVE", negatives,
                   lambda r: (-r["maximum_confidence"], -r["false_positives"], r["image_id"])))
    groups.append(("POSSIBLE_CLASS_CONFUSION",
                   [r for r in false_positives if r["diagnosis"] == "class_confusion"],
                   lambda r: (-r["confidence"], -r["best_gt_iou"], r["image_id"], r["prediction_index"])))
    groups.append(("POOR_LOCALIZATION", localization,
                   lambda r: (-r["prediction_confidence"], r["iou"], r["image_id"])))
    for country in ("India", "Japan"):
        groups.append((f"{country.upper()}_ERROR",
                       [r for r in image_errors.values() if r["country"] == country],
                       lambda r: (-r["total_errors"], -r["max_fp_confidence"], r["image_id"])))
    result = []
    for category, records, key in groups:
        for rank, row in enumerate(sorted(records, key=key)[:limit], 1):
            result.append({"category": category, "rank": rank, "image_id": row["image_id"],
                "validation_image_path": row["validation_image_path"], "country": row["country"],
                "class_code": row.get("class_code", ""), "diagnosis": row.get("diagnosis", row.get("diagnoses", "")),
                "confidence": row.get("confidence", row.get("prediction_confidence", row.get("maximum_confidence", ""))),
                "iou": row.get("best_gt_iou", row.get("iou", row.get("diagnostic_iou", ""))),
                "fp": row.get("fp", row.get("false_positives", "")), "fn": row.get("fn", "")})
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ThresholdError(f"Refusing to write headerless empty analysis artifact: {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(analysis: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    summary = analysis["summary"]
    per_class = analysis["per_class_errors"]
    countries = {r["country"]: r for r in analysis["country_errors"] if r["class_code"] == "ALL"}
    country_classes = [r for r in analysis["country_errors"] if r["class_code"] != "ALL"]
    size_rows = analysis["object_size_metrics"]
    confidence = {(r["outcome"], r["class_code"]): r for r in analysis["confidence_distributions"]}
    iou_by_class = {code: [r["iou"] for r in analysis["iou_distribution"] if r["class_code"] == code]
                    for code in CODES}
    diagnosis_counts = defaultdict(int)
    for row in analysis["false_negatives"]:
        diagnosis_counts[(row["class_code"], row["diagnosis"])] += 1
    confusion_rows = sorted((r for r in analysis["class_confusion_matrix"]
                             if r["true_class"] != BACKGROUND and r["predicted_class"] != BACKGROUND
                             and r["true_class"] != r["predicted_class"] and r["count"]),
                            key=lambda r: (-r["count"], r["true_class"], r["predicted_class"]))
    lines = ["# YOLOv8s baseline validation-only error analysis", "",
        "Source: verified threshold-selection ground-truth and NMS-0.50 prediction caches only.",
        "No model inference, canonical image/label reads, internal-test examples, or teacher video were used.", "",
        f"Frozen operating point: confidence {config['confidence']:.2f}, NMS IoU {config['nms_iou']:.2f}, "
        f"class-aware matching IoU {config['matching_iou']:.2f}.", "",
        "## Per-class fixed-point errors", "", "| Class | Support | TP | FP | FN | Precision | Recall | F1 | Localization | Confusion | High-conf FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_class:
        lines.append(f"| {row['class_code']} | {row['support']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                     f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
                     f"{row['localization_errors']} | {row['confusions']} | {row['high_confidence_fp']} |")
    lines.extend(["", "D00 has the lowest validation recall. D20 remains the strongest class by F1. "
        "These statements describe validation only.", "", "## Country comparison", "",
        "Support is reported before performance; the split is leakage-reduced grouped, not route-independent.", "",
        "| Country | Images | Targets | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for country in ("India", "Japan"):
        row = countries[country]
        lines.append(f"| {country} | {row['images']} | {row['support']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                     f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")
    lines.extend(["", "| Country | Class | Targets | Positive images | TP | FP | FN | Precision | Recall | F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in country_classes:
        lines.append(f"| {row['country']} | {row['class_code']} | {row['support']} | {row['positive_images']} | "
                     f"{row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.4f} | "
                     f"{row['recall']:.4f} | {row['f1']:.4f} |")
    lines.extend(["", "India D10 remains LOW SUPPORT: 10 boxes / 10 validation images; "
        "no strong India-specific D10 conclusion is supported.", "",
        "## Localization, confusion, confidence and negatives", "",
        f"- Possible localization pairs (same class, IoU {config['localization_iou_floor']:.2f} to "
        f"<{config['matching_iou']:.2f}): {summary['localization_errors']}.",
        f"- Class-confusion pairs (wrong class, IoU >={config['matching_iou']:.2f}): {summary['class_confusion_pairs']}.",
        f"- High-confidence false positives (confidence >={config['high_confidence']:.2f}): "
        f"{summary['high_confidence_false_positives']}.",
        f"- Misses with a high-confidence wrong-class/localization diagnostic candidate: "
        f"{summary['high_confidence_fn_diagnostics']}. Ground-truth misses have no intrinsic confidence.",
        f"- Negative images: {summary['negative_images']}; FP: {summary['negative_fp']}; "
        f"FP/image: {summary['negative_fp_per_image']:.6f}; images with >=1 FP: "
        f"{summary['negative_images_with_fp']} ({summary['negative_images_with_fp_fraction']:.2%}).", "",
        "False-negative diagnostic counts (these are explanations from cached geometry, not mutually independent visual causes):", "",
        "| Class | Below 0.19 with IoU >=0.50 | Wrong-class overlap | Poor localization | No matching candidate |",
        "|---|---:|---:|---:|---:|"])
    for code in CODES:
        lines.append(f"| {code} | {diagnosis_counts[(code, 'below_frozen_confidence')]} | "
                     f"{diagnosis_counts[(code, 'class_confusion')]} | "
                     f"{diagnosis_counts[(code, 'poor_localization')]} | "
                     f"{diagnosis_counts[(code, 'no_matching_candidate')]} |")
    lines.extend(["", "Nonzero wrong-class overlap directions:", "",
                  "| True class | Predicted class | Pairs |", "|---|---|---:|"])
    for row in confusion_rows:
        lines.append(f"| {row['true_class']} | {row['predicted_class']} | {row['count']} |")
    lines.extend(["", "Prediction-confidence and true-positive IoU distributions:", "",
                  "| Class | TP confidence median | FP confidence median | TP IoU P25 | TP IoU median | TP IoU P75 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for code in CODES:
        overlaps = _quantiles(iou_by_class[code])
        lines.append(f"| {code} | {_format_optional_float(confidence[('TP', code)]['median'])} | "
                     f"{_format_optional_float(confidence[('FP', code)]['median'])} | "
                     f"{_format_optional_float(overlaps['p25'])} | "
                     f"{_format_optional_float(overlaps['median'])} | "
                     f"{_format_optional_float(overlaps['p75'])} |")
    lines.extend(["",
        "The cache can identify geometric confusion/localization candidates but cannot determine visual causes such as "
        "shadow, water, marking, manhole, blur or occlusion without manual validation-image review.", "",
        "## Object-size rule", "",
        "Normalized bounding-box footprint = box area / image area. Buckets: small <1%, medium 1% to <5%, "
        "large >=5%. This is detector-box footprint, not physical damage area or engineering severity. "
        "Rows with fewer than 30 GT objects are marked weak.", "",
        "| Class | Bucket | GT objects | Detected | Recall | Support warning |",
        "|---|---|---:|---:|---:|---|"])
    for row in size_rows:
        lines.append(f"| {row['class_code']} | {row['size_bucket']} | {row['gt_objects']} | "
                     f"{row['detected_gt_objects']} | {row['recall']:.4f} | "
                     f"{'WEAK (<30)' if row['weak_support'] else ''} |")
    lines.extend(["",
        "## Ranked manual-review queues", "",
        "ranked_examples.csv deterministically lists per-class false negatives, high-confidence false positives, "
        "India/Japan errors, negative-road false positives, possible class confusions and poor-localization candidates. "
        "It references validation paths but does not read or copy pixels.", "",
        "## Recommended Experiment 2", "",
        "Proceed with the planned pretrained YOLO26s experiment under the matched YOLOv8s training protocol. "
        "Validation shows broad class/country recall and localization limitations rather than evidence that one different "
        "pipeline or dataset change should precede the controlled architecture comparison. Keep dataset version, split, "
        "image size, seed, training schedule and evaluation code matched. Do not train as part of this phase.", "",
        "The completed internal-test result is not used to select this recommendation; it remains a locked aggregate record."])
    return "\n".join(lines) + "\n"


def run(root: Path = ROOT) -> Path:
    if Path(root).absolute() != ROOT.absolute():
        raise ThresholdError("Project-root override is not supported for the canonical analysis run.")
    config, gt, cache, selected = load_sources(root)
    output = shared.checked_path(Path(config["output"]), OUTPUT, root)
    if output.exists():
        raise ThresholdError(f"Refusing to overwrite existing error analysis: {output}")
    analysis = analyze(config, gt, cache)
    for code, expected in selected["per_class_metrics"].items():
        observed = next(row for row in analysis["per_class_errors"] if row["class_code"] == code)
        for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
            if observed[key] != expected[key]:
                raise ThresholdError("Detailed analysis does not reconcile with the frozen validation result.")
    output.mkdir(parents=True, exist_ok=False)
    csv_names = ("per_class_errors", "country_errors", "false_negatives", "false_positives",
                 "localization_errors", "negative_image_false_positives", "object_size_metrics",
                 "iou_distribution", "confidence_distributions", "class_confusion_matrix", "ranked_examples")
    for name in csv_names:
        _write_csv(output / f"{name}.csv", analysis[name])
    (output / "error_analysis_summary.md").write_text(_summary_markdown(analysis, config), encoding="utf-8")
    status = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"],
                            cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    commit = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    artifacts = {path.name: shared.digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    manifest = {"schema_version": config["schema_version"], "status": "complete", "validation_only": True,
        "internal_test_images_or_labels_accessed": False, "inference_executed": False,
        "teacher_video_used": False, "threshold_tuning_performed": False,
        "source_artifacts_sha256": {
            config[key]: config[sha] for key, sha in (("ground_truth_cache", "ground_truth_sha256"),
                ("prediction_cache", "prediction_sha256"), ("selected_operating_point", "selected_operating_point_sha256"),
                ("source_manifest", "source_manifest_sha256"))},
        "model_sha256": config["model_sha256"],
        "frozen_operating_point": {"confidence": .19, "nms_iou": .5, "matching_iou": .5},
        "source_threshold_selection_commit": selected["git_commit"], "analysis_git_commit": commit,
        "analysis_worktree_status": status, "analysis_config_sha256": shared.digest(root / CONFIG),
        "analysis_code_sha256": shared.digest(Path(__file__)), "counts": analysis["summary"],
        "artifacts": artifacts}
    (output / "error_analysis_manifest.json").write_bytes(shared.json_bytes(manifest))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args(argv)
    try:
        output = run()
    except (ThresholdError, OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"Validation error analysis stopped: {exc}", file=sys.stderr)
        return 2
    print(f"Validation-only error analysis complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
