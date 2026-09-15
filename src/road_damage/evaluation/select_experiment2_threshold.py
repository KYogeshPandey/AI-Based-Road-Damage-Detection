"""Select YOLO26s' global confidence on the frozen validation split only.

The installed YOLO26 end-to-end head is NMS-free.  This tool therefore caches
one native prediction pass and sweeps confidence only; NMS IoU is deliberately
absent from the protocol and predictor overrides.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.evaluation import select_threshold as shared  # noqa: E402
from road_damage.evaluation.threshold_metrics import Box, ThresholdError, sweep_cache  # noqa: E402


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path("configs/evaluation/experiment2_yolo26s_threshold_selection.yaml")
DATA_YAML = shared.DATA_YAML
EXPORT = shared.EXPORT
OUTPUT = Path("outputs/evaluation/experiment2_yolo26s_threshold_selection")
RECOMPUTE_OUTPUT = Path("outputs/evaluation/experiment2_yolo26s_threshold_selection_recomputed")
MODEL = Path(
    "outputs/training/experiment2_yolo26s_matched/"
    "yolo26s_rdd2022-india-japan-v1.1.0_640_seed42/weights/best.pt"
)
TRAINING_RESULTS = MODEL.parents[1] / "results.csv"
TRAINING_COMPLETION = MODEL.parents[1] / "completion.json"
CLASS_NAMES = shared.CLASS_NAMES
EXPECTED_MODEL_SHA256 = "99c03d56f4b27d6a9cc774dd3880f11807642058934199ed675dc3bf4b9f37a1"
EXPECTED_MODEL_SIZE_BYTES = 20_320_709
ULTRALYTICS_SOURCE_HASHES = {
    "ultralytics/nn/modules/head.py": "38a31163093ae01fe691a68d1eecdab74456151c98140b28bf1da1bc9a0bdc23",
    "ultralytics/utils/nms.py": "d336cff44861dd5c84e7b7020428d9f64b408d7c21cd4b3b17d772f4b455ef83",
    "ultralytics/models/yolo/detect/predict.py": "2f4d4e817782da56d9d6864084727a9576a5d276370cc1811435622941a2385d",
    "ultralytics/models/yolo/detect/val.py": "098a00582a88969eadbb9a32a5337a3a9b27388d85455a15ff9d0f892373863c",
}
EXPECTED_CONFIG: dict[str, Any] = {
    "schema_version": "experiment2_yolo26s.threshold_selection.v1",
    "model": MODEL.as_posix(),
    "model_size_bytes": EXPECTED_MODEL_SIZE_BYTES,
    "model_sha256": EXPECTED_MODEL_SHA256,
    "training_results": TRAINING_RESULTS.as_posix(),
    "training_results_sha256": "255a796a025a20c6c55d21e63abb7d1f07a528b70cb30c620881de997a308579",
    "training_completion": TRAINING_COMPLETION.as_posix(),
    "training_completion_sha256": "2c0c1e6afe8124211d8c5b0e3ee034a17f88a4eab6cf594c448301c22ab35343",
    "best_epoch": 63,
    "best_validation_metrics": {
        "precision": "0.55503", "recall": "0.46612", "map50": "0.47631", "map50_95": "0.21702"
    },
    "validation_yaml": DATA_YAML.as_posix(),
    "expected_validation": {
        "images": 2602, "positive_images": 1619, "negative_images": 983, "targets": 3377,
        "targets_by_class": {"0": 822, "1": 576, "2": 1188, "3": 791},
    },
    "classes": {str(key): value for key, value in CLASS_NAMES.items()},
    "architecture": {"family": "YOLO26", "variant": "s", "nc": 4, "end2end": True, "reg_max": 1},
    "native_postprocessing": {
        "mode": "ultralytics_8.4.130_native_end2end_nms_free",
        "prediction_branch": "one2one", "nms_applicable": False, "nms_iou_swept": False,
        "iou_argument_supplied": False, "source_sha256": ULTRALYTICS_SOURCE_HASHES,
    },
    "imgsz": 640, "device": 0, "batch": 4, "inference_confidence_floor_milli": 10,
    "confidence_grid_milli": {"start": 10, "stop": 800, "step": 1},
    "matching_iou": 0.50, "global_threshold_only": True, "augment": False, "half": False,
    "max_det": 300, "seed": 42, "progress_every": 100, "output": OUTPUT.as_posix(),
    "recompute_output": RECOMPUTE_OUTPUT.as_posix(),
}

BASELINE_COMPARISON = {
    "model": "YOLOv8s",
    "selected_global_confidence": 0.19,
    "selected_nms_iou": 0.50,
    "matching_iou": 0.50,
    "macro_f1": 0.5055312316193121,
    "per_class_metrics": {
        "D00": {"tp": 304, "fp": 302, "fn": 518, "f1": 0.42577},
        "D10": {"tp": 248, "fp": 242, "fn": 328, "f1": 0.46529},
        "D20": {"tp": 774, "fp": 414, "fn": 414, "f1": 0.651515},
        "D40": {"tp": 340, "fp": 287, "fn": 451, "f1": 0.479549},
    },
    "negative_image_metrics": {
        "negative_images": 983, "negative_fp": 121, "negative_fp_per_image": 0.12309,
        "negative_images_with_fp": 96, "negative_images_with_fp_fraction": 0.09766,
    },
}


def load_config(root: Path = ROOT) -> dict[str, Any]:
    """Load the literal frozen protocol; unknown or per-class options fail."""
    path = shared.checked_path(CONFIG, CONFIG, root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if shared.json_bytes(value) != shared.json_bytes(EXPECTED_CONFIG):
        raise ThresholdError("Frozen Experiment 2 threshold config differs; overrides are forbidden.")
    return value


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "--no-optional-locks", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ThresholdError(f"Git provenance check failed: {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def git_provenance(root: Path = ROOT) -> dict[str, Any]:
    """Record, rather than mutate or conceal, the explicitly uncommitted tooling state."""
    source_paths = [
        CONFIG,
        Path(__file__).resolve().relative_to(ROOT),
        Path("src/road_damage/evaluation/select_threshold.py"),
        Path("src/road_damage/evaluation/threshold_metrics.py"),
        Path("tests/test_experiment2_threshold_selection.py"),
        Path("road-damage-project-docs/EXPERIMENT2_YOLO26S_RUNBOOK.md"),
    ]
    source_files = []
    for relative in source_paths:
        path = root / relative
        if not path.is_file():
            raise ThresholdError(f"Required evaluation source is missing: {relative}")
        source_files.append({"path": relative.as_posix(), "sha256": shared.digest(path), "size_bytes": path.stat().st_size})
    status = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    return {
        "head_commit": _git(root, "rev-parse", "HEAD"),
        "worktree_clean": not status,
        "git_status_porcelain": status,
        "source_files": source_files,
        "source_fingerprint_sha256": hashlib.sha256(shared.json_bytes(source_files)).hexdigest(),
        "history_modified": False,
    }


def verify_ultralytics_semantics() -> dict[str, Any]:
    """Pin and inspect the installed native end-to-end prediction source."""
    package = importlib.util.find_spec("ultralytics")
    if package is None or not package.submodule_search_locations:
        raise ThresholdError("Installed Ultralytics package cannot be located.")
    base = Path(next(iter(package.submodule_search_locations)))
    sources: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    for relative, expected in ULTRALYTICS_SOURCE_HASHES.items():
        inner = Path(relative).relative_to("ultralytics")
        path = base / inner
        actual = shared.digest(path)
        if actual != expected:
            raise ThresholdError(f"Installed postprocessing source changed: {relative}")
        sources[relative] = {"sha256": actual, "size_bytes": path.stat().st_size}
        texts[relative] = path.read_text(encoding="utf-8")
    head = texts["ultralytics/nn/modules/head.py"]
    nms = texts["ultralytics/utils/nms.py"]
    predictor = texts["ultralytics/models/yolo/detect/predict.py"]
    validator = texts["ultralytics/models/yolo/detect/val.py"]
    evidence = {
        "one2one_inference_branch": 'preds["one2one"] if self.end2end else preds' in head,
        "native_topk_postprocess": "self.get_topk_index(scores, self.max_det)" in head,
        "end2end_bypasses_legacy_nms": "if prediction.shape[-1] == 6 or end2end" in nms,
        "predictor_forwards_end2end": 'end2end=getattr(self.model, "end2end", False)' in predictor,
        "validator_forwards_end2end": "end2end=self.end2end" in validator,
    }
    branch = nms.split("if prediction.shape[-1] == 6 or end2end", 1)[1].split("bs = prediction.shape[0]", 1)[0]
    evidence["end2end_branch_does_not_use_iou_threshold"] = "iou_thres" not in branch
    if not all(evidence.values()):
        raise ThresholdError(f"Installed YOLO26 end-to-end semantic evidence failed: {evidence}")
    return {
        "ultralytics_version": "8.4.130",
        "mode": EXPECTED_CONFIG["native_postprocessing"]["mode"],
        "prediction_branch": "one2one",
        "nms_applicable": False,
        "nms_iou_swept": False,
        "iou_argument_supplied": False,
        "source_files": sources,
        "source_evidence": evidence,
    }


def verify_training_identity(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the frozen checkpoint's selection row and completion provenance."""
    results_path = shared.checked_path(Path(config["training_results"]), TRAINING_RESULTS, root)
    completion_path = shared.checked_path(Path(config["training_completion"]), TRAINING_COMPLETION, root)
    if shared.digest(results_path) != config["training_results_sha256"]:
        raise ThresholdError("Frozen Experiment 2 results.csv SHA-256 mismatch.")
    if shared.digest(completion_path) != config["training_completion_sha256"]:
        raise ThresholdError("Frozen Experiment 2 completion receipt SHA-256 mismatch.")
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    chosen = [row for row in rows if int(row["epoch"]) == config["best_epoch"]]
    if len(chosen) != 1:
        raise ThresholdError("Frozen Experiment 2 best-epoch row is missing or duplicated.")
    row = chosen[0]
    columns = {
        "precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)", "map50_95": "metrics/mAP50-95(B)",
    }
    observed = {name: row[column] for name, column in columns.items()}
    if any(Decimal(observed[name]) != Decimal(expected)
           for name, expected in config["best_validation_metrics"].items()):
        raise ThresholdError(f"Frozen Experiment 2 best validation metrics differ: {observed}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (completion.get("status") != "COMPLETED" or completion.get("epochs_completed") != 83
            or completion.get("internal_test_files_accessed") is not False
            or completion.get("artifacts", {}).get("weights/best.pt") != config["model_sha256"]
            or completion.get("early_stopping", {}).get("best_epoch") != config["best_epoch"]):
        raise ThresholdError("Frozen Experiment 2 completion receipt does not identify the selected checkpoint.")
    return {
        "best_epoch": config["best_epoch"], "best_validation_metrics": observed,
        "results_path": config["training_results"], "results_sha256": config["training_results_sha256"],
        "completion_path": config["training_completion"],
        "completion_sha256": config["training_completion_sha256"], "training_status": "COMPLETED",
    }


def inventory_validation(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the approved validator, then enforce positive and per-class support."""
    inventory_config = dict(config)
    expected = config["expected_validation"]
    inventory_config["expected_counts"] = {
        key: expected[key] for key in ("images", "targets", "negative_images")
    }
    ground_truth = shared.inventory_validation(root, inventory_config)
    by_class = {str(class_id): 0 for class_id in CLASS_NAMES}
    for image in ground_truth["images"]:
        for target in image["ground_truth"]:
            by_class[str(target["class_id"])] += 1
    counts = {
        **ground_truth["counts"],
        "positive_images": len(ground_truth["images"]) - ground_truth["counts"]["negative_images"],
        "targets_by_class": by_class,
    }
    if counts != expected:
        raise ThresholdError(f"Frozen validation identity mismatch: expected {expected}, observed {counts}")
    ground_truth["schema_version"] = "experiment2_threshold_ground_truth.v1"
    ground_truth["counts"] = counts
    return ground_truth


def preflight(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify frozen inputs and output availability without model inference."""
    config = load_config(root)
    output = shared.checked_path(Path(config["output"]), OUTPUT, root)
    shared.ensure_output_available(output)
    model_path = shared.checked_path(Path(config["model"]), MODEL, root)
    shared.verify_model_identity(
        model_path, expected_sha256=config["model_sha256"], expected_size_bytes=config["model_size_bytes"]
    )
    training = verify_training_identity(root, config)
    validation = inventory_validation(root, config)
    semantics = verify_ultralytics_semantics()
    environment = shared.environment()
    provenance = git_provenance(root)
    report = {
        "passed": True, "inference_executed": False, "validation_only": True,
        "internal_test_files_accessed": False, "teacher_video_accessed": False,
        "training_images_accessed": False, "official_unlabelled_test_accessed": False,
        "model": {"path": config["model"], "size_bytes": config["model_size_bytes"],
                  "sha256": config["model_sha256"], "weights_modified": False},
        "training_identity": training, "validation_identity": validation["counts"],
        "validation_yaml_sha256": shared.digest(root / DATA_YAML),
        "ground_truth_snapshot_sha256": hashlib.sha256(shared.json_bytes(validation)).hexdigest(),
        "protocol_config_sha256": shared.digest(root / CONFIG),
        "native_postprocessing": semantics, "environment": environment, "git_provenance": provenance,
        "output_available": str(output),
    }
    return config, validation, report


def prediction_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Exact native predictor policy.  An IoU argument is intentionally absent."""
    return {
        "imgsz": config["imgsz"], "conf": config["inference_confidence_floor_milli"] / 1000,
        "device": config["device"], "batch": config["batch"], "augment": False, "half": False,
        "max_det": config["max_det"], "agnostic_nms": False, "classes": None,
        "save": False, "save_txt": False, "save_conf": False, "save_crop": False,
        "show": False, "verbose": False, "stream": True, "rect": False, "compile": False,
    }


def verify_runtime_architecture(model: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail before inference unless this checkpoint has the native four-class YOLO26 head."""
    head = model.model.model[-1]
    observed = {
        "family": "YOLO26", "variant": "s", "head_type": type(head).__name__,
        "nc": getattr(head, "nc", None), "end2end": getattr(head, "end2end", None),
        "reg_max": getattr(head, "reg_max", None), "max_det": getattr(head, "max_det", None),
        "names": getattr(model, "names", None),
    }
    required = config["architecture"]
    if (observed["head_type"] != "Detect" or observed["nc"] != required["nc"]
            or observed["end2end"] is not True or observed["reg_max"] != required["reg_max"]
            or observed["max_det"] != config["max_det"] or observed["names"] != CLASS_NAMES):
        raise ThresholdError(f"Checkpoint is not the exact four-class native YOLO26s architecture: {observed}")
    return observed


def _prediction_box(xyxy: Sequence[float], confidence: float, class_id: float) -> dict[str, Any]:
    if int(class_id) != class_id:
        raise ThresholdError("Model returned a non-integral class ID.")
    return shared.box_record(Box(int(class_id), tuple(float(value) for value in xyxy), float(confidence)))


def collect_predictions(model: Any, ground_truth: Mapping[str, Any], config: Mapping[str, Any],
                        root: Path = ROOT) -> tuple[dict[str, Any], float]:
    """Run native prediction in explicit small batches and build one validation cache."""
    started = time.perf_counter()
    expected = list(ground_truth["images"])
    kwargs = prediction_kwargs(config)
    if "iou" in kwargs:
        raise ThresholdError("NMS IoU must not be supplied to the native YOLO26 end-to-end predictor.")
    records = []
    total = len(expected)
    LOGGER.info("Native end-to-end validation prediction: 0/%d images", total)
    for batch_start in range(0, total, config["batch"]):
        batch_records = expected[batch_start:batch_start + config["batch"]]
        sources = [
            str(shared.checked_path(Path(row["image_path"]), EXPORT / "images/val" / row["image_id"], root))
            for row in batch_records
        ]
        results = model.predict(source=sources, **kwargs)
        for offset, (reference, result) in enumerate(zip(batch_records, results, strict=True), start=1):
            index = batch_start + offset
            actual_path = Path(str(result.path))
            if os.path.normcase(os.path.abspath(actual_path)) != os.path.normcase(
                    os.path.abspath(root / reference["image_path"])):
                raise ThresholdError("Predictor result order/path escaped the explicit validation allowlist.")
            if tuple(result.orig_shape) != (reference["height"], reference["width"]):
                raise ThresholdError(f"Predictor image dimensions changed: {reference['image_id']}")
            boxes = []
            if result.boxes is not None:
                boxes = [
                    _prediction_box(xyxy, confidence, class_id)
                    for xyxy, confidence, class_id in zip(
                        result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                        result.boxes.cls.cpu().tolist(), strict=True,
                    )
                ]
            floor = config["inference_confidence_floor_milli"] / 1000
            if any(box["confidence"] <= floor for box in boxes):
                raise ThresholdError("Native predictor returned a candidate at/below its strict confidence floor.")
            if len(boxes) >= config["max_det"]:
                raise ThresholdError("Native top-k/max_det cap reached; low-confidence candidates may be truncated.")
            boxes.sort(key=lambda box: (-box["confidence"], box["class_id"], box["xyxy"]))
            records.append({"image_id": reference["image_id"], "predictions": boxes})
            if index % config["progress_every"] == 0 or index == total:
                elapsed = time.perf_counter() - started
                LOGGER.info("Validation images processed %d/%d; elapsed %.1fs; ETA %.1fs",
                            index, total, elapsed, elapsed / index * (total - index))
    duration = time.perf_counter() - started
    cache = {
        "schema_version": "experiment2_yolo26s.native_predictions.v1",
        "validation_only": True, "internal_test_files_accessed": False,
        "model_sha256": config["model_sha256"],
        "confidence_floor": config["inference_confidence_floor_milli"] / 1000,
        "postprocessing": {
            "mode": config["native_postprocessing"]["mode"], "prediction_branch": "one2one",
            "nms_applicable": False, "nms_iou_swept": False, "iou_argument_supplied": False,
            "max_det": config["max_det"],
        },
        "images": records,
    }
    return cache, duration


def decoded_predictions(cache: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, list[Box]]:
    """Validate a persisted native cache without opening a model or dataset."""
    expected_header = {
        "mode": config["native_postprocessing"]["mode"], "prediction_branch": "one2one",
        "nms_applicable": False, "nms_iou_swept": False, "iou_argument_supplied": False,
        "max_det": config["max_det"],
    }
    if (cache.get("validation_only") is not True or cache.get("internal_test_files_accessed") is not False
            or cache.get("model_sha256") != config["model_sha256"]
            or cache.get("confidence_floor") != config["inference_confidence_floor_milli"] / 1000
            or cache.get("postprocessing") != expected_header):
        raise ThresholdError("Prediction cache identity or native end-to-end policy mismatch.")
    decoded: dict[str, list[Box]] = {}
    for record in cache.get("images", []):
        name = record.get("image_id")
        if name in decoded or not isinstance(name, str) or not __import__("re").fullmatch(r"(?:India|Japan)_\d+\.jpg", name):
            raise ThresholdError("Invalid or duplicate validation image in prediction cache.")
        boxes = [Box(box["class_id"], tuple(box["xyxy"]), box["confidence"])
                 for box in record.get("predictions", [])]
        floor = config["inference_confidence_floor_milli"] / 1000
        if any(box.confidence <= floor for box in boxes) or len(boxes) >= config["max_det"]:
            raise ThresholdError("Prediction cache floor/cap policy mismatch.")
        decoded[name] = boxes
    return decoded


def _aggregate_metrics(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if key != "nms_iou"}
    codes = ("D00", "D10", "D20", "D40")
    result["macro_precision"] = sum(result[f"{code}_precision"] for code in codes) / 4
    result["macro_recall"] = sum(result[f"{code}_recall"] for code in codes) / 4
    tp, fp, fn = result["total_tp"], result["total_fp"], result["total_fn"]
    result["micro_precision"] = tp / (tp + fp) if tp + fp else 0.0
    result["micro_recall"] = tp / (tp + fn) if tp + fn else 0.0
    result["micro_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    result["negative_images_with_fp_percent"] = 100 * result["negative_images_with_fp_fraction"]
    result["nms_applicable"] = False
    return result


def select_global_threshold(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Maximum exact macro-F1, then negative FP/image, then higher confidence."""
    if not rows:
        raise ThresholdError("No threshold rows to select.")
    if any(row.get("nms_applicable") is not False for row in rows):
        raise ThresholdError("Experiment 2 selection must contain no NMS parameter.")
    best_f1 = max(Fraction(row["macro_f1_exact"]) for row in rows)
    primary = [row for row in rows if Fraction(row["macro_f1_exact"]) == best_f1]
    best_negative_fp = min(row["negative_fp"] for row in primary)
    secondary = [row for row in primary if row["negative_fp"] == best_negative_fp]
    best_confidence = max(row["confidence"] for row in secondary)
    finalists = [row for row in secondary if row["confidence"] == best_confidence]
    if len(finalists) != 1:
        raise ThresholdError("Global threshold tie remained after the complete declared tie-break.")
    return {
        "metrics": dict(finalists[0]),
        "tie_break_reasoning": {
            "objective": "maximum exact rational macro-F1 across D00/D10/D20/D40",
            "best_macro_f1_exact": str(best_f1), "points_after_macro_f1": len(primary),
            "tie_break_1": "minimum false positives per negative-only validation image",
            "minimum_negative_fp": best_negative_fp, "points_after_negative_fp": len(secondary),
            "tie_break_2": "highest global confidence threshold",
            "highest_confidence": best_confidence, "points_after_confidence": len(finalists),
            "nms_tie_break": None,
        },
    }


def calculate_artifacts(ground_truth: Mapping[str, Any], cache: Mapping[str, Any],
                        config: Mapping[str, Any], identity: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic inference-free metric calculation from the persisted cache."""
    if ground_truth.get("counts") != config["expected_validation"]:
        raise ThresholdError("Ground-truth cache support differs from the frozen validation identity.")
    truth = shared.decoded_ground_truth(dict(ground_truth))
    predictions = decoded_predictions(cache, config)
    grid = config["confidence_grid_milli"]
    confidence_grid = [value / 1000 for value in range(grid["start"], grid["stop"] + 1, grid["step"])]
    raw_rows = sweep_cache(truth, predictions, 0.0, confidence_grid, config["matching_iou"])
    rows = [_aggregate_metrics(row) for row in raw_rows]
    selection = select_global_threshold(rows)
    point = selection["metrics"]
    per_class = {
        code: {metric: point[f"{code}_{metric}"] for metric in ("tp", "fp", "fn", "precision", "recall", "f1")}
        for code in ("D00", "D10", "D20", "D40")
    }
    negative = {key: value for key, value in point.items() if key.startswith("negative_")}
    selected = {
        "schema_version": "experiment2_yolo26s.selected_operating_point.v1",
        "evaluation_identity": dict(identity), "selection_split": "validation",
        "selected_global_confidence": point["confidence"], "global_threshold_only": True,
        "per_class_thresholds": None, "matching_iou": config["matching_iou"],
        "nms": {"applicable": False, "swept": False, "iou": None,
                "reason": "native Ultralytics 8.4.130 YOLO26 end-to-end NMS-free output preserved"},
        "macro_metrics": {metric: point[f"macro_{metric}"] for metric in ("precision", "recall", "f1")},
        "micro_metrics": {metric: point[f"micro_{metric}"] for metric in ("precision", "recall", "f1")},
        "per_class_metrics": per_class, "negative_image_metrics": negative,
        "counts": {key: point[key] for key in ("total_tp", "total_fp", "total_fn")},
        "tie_break_reasoning": selection["tie_break_reasoning"],
    }
    return rows, selected


def comparison(selected: Mapping[str, Any]) -> dict[str, Any]:
    """Compare both models only at their own validation-selected points."""
    exp2 = {
        "model": "YOLO26s", "selected_global_confidence": selected["selected_global_confidence"],
        "nms": selected["nms"], "matching_iou": selected["matching_iou"],
        "macro_f1": selected["macro_metrics"]["f1"],
        "per_class_metrics": selected["per_class_metrics"],
        "negative_image_metrics": selected["negative_image_metrics"],
    }
    return {
        "schema_version": "experiment2_yolo26s.validation_comparison.v1",
        "comparison_scope": "each model at its own validation-selected operating point",
        "internal_test_used": False, "yolov8s": BASELINE_COMPARISON, "yolo26s": exp2,
        "yolo26s_minus_yolov8s": {
            "macro_f1": exp2["macro_f1"] - BASELINE_COMPARISON["macro_f1"],
            "negative_fp_per_image": (
                exp2["negative_image_metrics"]["negative_fp_per_image"]
                - BASELINE_COMPARISON["negative_image_metrics"]["negative_fp_per_image"]
            ),
        },
    }


def write_scientific_results(output: Path, rows: Sequence[dict[str, Any]], selected: Mapping[str, Any]) -> list[str]:
    """Write deterministic result artifacts; caller verifies them before completion."""
    artifacts = {
        "threshold_sweep.json": rows,
        "selected_operating_point.json": selected,
        "per_class_selected_metrics.json": selected["per_class_metrics"],
        "negative_image_fp_metrics.json": selected["negative_image_metrics"],
        "comparison_to_yolov8s.json": comparison(selected),
    }
    for name, value in artifacts.items():
        (output / name).write_bytes(shared.json_bytes(value))
    with (output / "threshold_sweep.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Experiment 2 YOLO26s validation threshold selection", "",
        "Selection used only the frozen 2,602-image validation split. Internal test and teacher video were untouched.",
        "Native Ultralytics 8.4.130 YOLO26 end-to-end, one-to-one output was preserved; NMS IoU is not applicable.", "",
        f"Selected global confidence: {selected['selected_global_confidence']:.3f}",
        f"Matching IoU: {selected['matching_iou']:.2f}",
        f"Macro precision / recall / F1: {selected['macro_metrics']['precision']:.8f} / "
        f"{selected['macro_metrics']['recall']:.8f} / {selected['macro_metrics']['f1']:.8f}",
        f"Micro precision / recall / F1: {selected['micro_metrics']['precision']:.8f} / "
        f"{selected['micro_metrics']['recall']:.8f} / {selected['micro_metrics']['f1']:.8f}", "",
        "| Class | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for code, metrics in selected["per_class_metrics"].items():
        lines.append(
            f"| {code} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{metrics['precision']:.8f} | {metrics['recall']:.8f} | {metrics['f1']:.8f} |"
        )
    negative = selected["negative_image_metrics"]
    lines.extend([
        "", f"Negative-only images: {negative['negative_images']}",
        f"False positives on negative-only images: {negative['negative_fp']}",
        f"FP per negative-only image: {negative['negative_fp_per_image']:.8f}",
        f"Negative-only images with >=1 FP: {negative['negative_images_with_fp']} "
        f"({negative['negative_images_with_fp_percent']:.4f}%)", "",
        "Tie-break record:", "", "```json", json.dumps(selected["tie_break_reasoning"], indent=2), "```", "",
        "This is validation-only threshold selection, not internal-test evaluation.",
    ])
    (output / "threshold_selection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [*artifacts, "threshold_sweep.csv", "threshold_selection_summary.md"]


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(shared.json_bytes(value))
    os.replace(temporary, path)


def _verify_persisted_results(output: Path, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Reopen every scientific input/output and reproduce selection before completion."""
    for name, expected in manifest["artifacts"].items():
        path = output / name
        if not path.is_file() or shared.digest(path) != expected:
            raise ThresholdError(f"Persisted artifact failed SHA-256 verification: {name}")
    ground_truth = json.loads((output / "ground_truth_cache.json").read_text(encoding="utf-8"))
    cache = json.loads((output / "prediction_cache.json").read_text(encoding="utf-8"))
    rows, selected = calculate_artifacts(ground_truth, cache, config, manifest["evaluation_identity"])
    if shared.json_bytes(rows) != (output / "threshold_sweep.json").read_bytes():
        raise ThresholdError("Persisted threshold sweep is not reproducible from the cache.")
    if shared.json_bytes(selected) != (output / "selected_operating_point.json").read_bytes():
        raise ThresholdError("Persisted selected point is not reproducible from the cache.")


def run(root: Path = ROOT) -> None:
    """Run the one-pass validation prediction collection and offline confidence sweep."""
    started = time.perf_counter()
    config, ground_truth, preflight_report = preflight(root)
    output = shared.checked_path(Path(config["output"]), OUTPUT, root)
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "model_path": config["model"], "model_sha256": config["model_sha256"],
        "model_size_bytes": config["model_size_bytes"], "best_epoch": config["best_epoch"],
        "validation_only": True, "internal_test_files_accessed": False,
        "protocol_config_sha256": preflight_report["protocol_config_sha256"],
        "source_fingerprint_sha256": preflight_report["git_provenance"]["source_fingerprint_sha256"],
        "git_head": preflight_report["git_provenance"]["head_commit"],
    }
    manifest: dict[str, Any] = {
        "schema_version": "experiment2_yolo26s.evaluation_manifest.v1", "status": "STARTED",
        "started_utc": datetime.now(timezone.utc).isoformat(), "evaluation_identity": identity,
        "validation_only": True, "internal_test_files_accessed": False, "teacher_video_accessed": False,
        "training_executed": False, "native_postprocessing": preflight_report["native_postprocessing"],
        "config": config, "preflight": preflight_report, "artifacts": {},
    }
    manifest_path = output / "evaluation_manifest.json"
    _atomic_json(manifest_path, manifest)
    try:
        for name, value in (
            ("preflight_report.json", preflight_report),
            ("ground_truth_cache.json", ground_truth),
            ("model_identity.json", {**identity, "training_identity": preflight_report["training_identity"]}),
            ("validation_identity.json", {
                "counts": ground_truth["counts"], "validation_yaml_sha256": preflight_report["validation_yaml_sha256"],
                "ground_truth_snapshot_sha256": preflight_report["ground_truth_snapshot_sha256"],
            }),
            ("protocol_identity.json", {
                "config_sha256": preflight_report["protocol_config_sha256"],
                "matching_iou": config["matching_iou"], "confidence_grid_milli": config["confidence_grid_milli"],
                "global_threshold_only": True, "per_class_thresholds": None,
                "native_postprocessing": preflight_report["native_postprocessing"],
            }),
            ("environment.json", preflight_report["environment"]),
            ("source_provenance.json", preflight_report["git_provenance"]),
        ):
            path = output / name
            path.write_bytes(shared.json_bytes(value))
            manifest["artifacts"][name] = shared.digest(path)
        _atomic_json(manifest_path, manifest)

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from ultralytics import YOLO

        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model_path = shared.checked_path(Path(config["model"]), MODEL, root)
        shared.verify_model_identity(
            model_path, expected_sha256=config["model_sha256"],
            expected_size_bytes=config["model_size_bytes"],
        )
        model = YOLO(str(model_path), task="detect")
        architecture = verify_runtime_architecture(model, config)
        architecture_path = output / "runtime_architecture.json"
        architecture_path.write_bytes(shared.json_bytes(architecture))
        manifest["artifacts"][architecture_path.name] = shared.digest(architecture_path)
        _atomic_json(manifest_path, manifest)

        cache, inference_duration = collect_predictions(model, ground_truth, config, root)
        cache_path = output / "prediction_cache.json"
        cache_path.write_bytes(shared.json_bytes(cache))
        manifest["artifacts"][cache_path.name] = shared.digest(cache_path)
        _atomic_json(manifest_path, manifest)

        # Revalidate all allowed model/validation/source identities after inference.
        shared.verify_model_identity(
            model_path, expected_sha256=config["model_sha256"],
            expected_size_bytes=config["model_size_bytes"],
        )
        if inventory_validation(root, config) != ground_truth:
            raise ThresholdError("Validation data changed during prediction collection.")
        if git_provenance(root) != preflight_report["git_provenance"]:
            raise ThresholdError("Evaluation source/provenance changed during prediction collection.")

        offline_started = time.perf_counter()
        rows, selected = calculate_artifacts(ground_truth, cache, config, identity)
        offline_duration = time.perf_counter() - offline_started
        names = write_scientific_results(output, rows, selected)
        for name in names:
            manifest["artifacts"][name] = shared.digest(output / name)
        _verify_persisted_results(output, config, manifest)

        manifest.update({
            "status": "COMPLETED", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "inference_duration_seconds": inference_duration,
            "offline_sweep_duration_seconds": offline_duration,
            "total_evaluation_duration_seconds": time.perf_counter() - started,
            "selected_global_confidence": selected["selected_global_confidence"],
            "selected_macro_f1": selected["macro_metrics"]["f1"],
            "persisted_cache_reverified_before_completion": True,
        })
        _atomic_json(manifest_path, manifest)
        receipt = {
            "schema_version": "experiment2_yolo26s.threshold_selection_receipt.v1", "status": "COMPLETED",
            "validation_only": True, "internal_test_files_accessed": False, "training_executed": False,
            "model_sha256": config["model_sha256"], "evaluation_manifest_sha256": shared.digest(manifest_path),
            "prediction_cache_sha256": manifest["artifacts"]["prediction_cache.json"],
            "selected_operating_point_sha256": manifest["artifacts"]["selected_operating_point.json"],
            "selected_global_confidence": selected["selected_global_confidence"],
            "selected_macro_f1": selected["macro_metrics"]["f1"],
            "nms_applicable": False, "persisted_cache_reverified": True,
        }
        _atomic_json(output / "completion.json", receipt)
        LOGGER.info("Native validation inference duration: %.2fs", inference_duration)
        LOGGER.info("Offline threshold sweep duration: %.2fs", offline_duration)
        LOGGER.info("Total evaluation duration: %.2fs", manifest["total_evaluation_duration_seconds"])
        LOGGER.info("Selected global confidence %.3f; macro-F1 %.8f; NMS IoU not applicable",
                    selected["selected_global_confidence"], selected["macro_metrics"]["f1"])
    except BaseException as exc:
        manifest.update({"status": "FAILED_TECHNICAL", "error": str(exc),
                         "internal_test_files_accessed": False, "training_executed": False})
        _atomic_json(manifest_path, manifest)
        raise


def recompute(root: Path = ROOT) -> None:
    """Recompute from verified caches without model or dataset access."""
    config = load_config(root)
    source = shared.checked_path(OUTPUT, OUTPUT, root)
    manifest_path = shared.checked_path(source / "evaluation_manifest.json", OUTPUT / "evaluation_manifest.json", root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads((source / "completion.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "COMPLETED" or receipt.get("status") != "COMPLETED"
            or shared.digest(manifest_path) != receipt.get("evaluation_manifest_sha256")
            or manifest.get("config") != config):
        raise ThresholdError("Cache replay requires a complete, intact matching receipt.")
    for name in ("ground_truth_cache.json", "prediction_cache.json", "selected_operating_point.json"):
        if shared.digest(source / name) != manifest["artifacts"].get(name):
            raise ThresholdError(f"Cache replay SHA-256 mismatch: {name}")
    ground_truth = json.loads((source / "ground_truth_cache.json").read_text(encoding="utf-8"))
    cache = json.loads((source / "prediction_cache.json").read_text(encoding="utf-8"))
    rows, selected = calculate_artifacts(ground_truth, cache, config, manifest["evaluation_identity"])
    if shared.json_bytes(selected) != (source / "selected_operating_point.json").read_bytes():
        raise ThresholdError("Cache replay selected operating point differs.")
    output = shared.checked_path(Path(config["recompute_output"]), RECOMPUTE_OUTPUT, root)
    shared.ensure_output_available(output)
    output.mkdir(parents=True, exist_ok=False)
    write_scientific_results(output, rows, selected)
    LOGGER.info("Verified cache replay reproduced the selected operating point byte-for-byte: %s", output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Verify inputs/protocol only; no inference.")
    mode.add_argument("--recompute-cache", action="store_true", help="Offline verified cache replay.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.preflight:
            _, _, report = preflight()
            print(shared.json_bytes(report).decode("utf-8"))
        elif args.recompute_cache:
            recompute()
        else:
            run()
    except (ThresholdError, shared.MentorDemoError, OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        LOGGER.error("Experiment 2 threshold selection stopped: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Evaluation interrupted; partial output is not a completed result.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
