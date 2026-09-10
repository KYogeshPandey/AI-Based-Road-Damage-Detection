"""Verified persisted caches and offline reporting; no public inference API."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from road_damage.evaluation import select_threshold as shared
from road_damage.evaluation.threshold_metrics import Box, ThresholdError

CODES = ("D00", "D10", "D20", "D40")
STANDARD_PR_NOTICE = (
    "Branch A precision/recall are test-curve-derived descriptive statistics: the actual "
    "Ultralytics validator reports at the confidence-curve index maximizing smoothed mean F1. "
    "This reporting index is NOT a deployment threshold and changes no model/configuration. "
    "Branch B P/R/F1 at frozen confidence 0.19 and NMS 0.50 is the primary operating-point evaluation."
)
CONFUSION_NOTICE = (
    "Branch B diagnostic class-agnostic spatial confusion matrix: predictions retained at "
    "confidence >=0.19; native matching uses IoU >0.50. Rows=prediction, columns=truth. "
    "This visualization differs from the primary class-aware IoU >=0.50 matcher."
)


def tensor_boxes(value: Mapping[str, Any], *, confidence: bool) -> list[dict[str, Any]]:
    """Serialize actual validator tensors in their unchanged inference-image coordinates."""
    classes = value["cls"].detach().cpu().tolist()
    boxes = value["bboxes"].detach().cpu().tolist()
    scores = value["conf"].detach().cpu().tolist() if confidence else [1.] * len(classes)
    result = []
    for cls, xyxy, score in zip(classes, boxes, scores, strict=True):
        if int(cls) != cls:
            raise ThresholdError("Native validator returned a non-integer class.")
        result.append({"class_id": int(cls), "xyxy": xyxy, "confidence": score})
    return result


def _box_valid(value: dict[str, Any], *, native: bool, width: int, height: int, floor: float) -> None:
    if set(value) != {"class_id", "confidence", "xyxy"} or type(value["class_id"]) is not int or value["class_id"] not in range(4):
        raise ThresholdError("Invalid cached target class/schema.")
    score, coords = value["confidence"], value["xyxy"]
    if (type(score) not in (float, int) or not math.isfinite(score) or not floor <= score <= 1
            or not isinstance(coords, list) or len(coords) != 4
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in coords)):
        raise ThresholdError("Invalid cached confidence/coordinates.")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ThresholdError("Invalid cached box area.")
    # Native validator predictions are deliberately UNCLIPPED, including padding/out-of-frame boxes.
    if not native and (x1 < 0 or y1 < 0 or x2 > width or y2 > height):
        raise ThresholdError("Frozen prediction box outside native image bounds.")


def verify_cache(path: Path, branch: str, gt: dict[str, Any], identity_sha256: str) -> tuple[list[dict[str, Any]], str]:
    """Reopen sealed bytes; enforce schema, exact identities/coverage, and valid metric inputs."""
    if branch not in ("standard", "frozen"):
        raise ThresholdError("Unknown cache branch.")
    seal = json.loads(path.with_suffix(".seal.json").read_bytes())
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if (set(seal) != {"sha256", "records", "branch", "identity_sha256"} or sha != seal["sha256"]
            or seal["branch"] != branch or seal["identity_sha256"] != identity_sha256
            or seal["records"] != len(gt["images"]) or not payload.endswith(b"\n")):
        raise ThresholdError("Cache hash/seal/count/truncation check failed.")
    expected = {r["image_id"]: r for r in gt["images"]}
    if len(expected) != len(gt["images"]):
        raise ThresholdError("Duplicate ground-truth identities.")
    seen, rows = set(), []
    common_keys = {"schema_version", "branch", "identity_sha256", "image_id", "country",
                   "nms_iou", "confidence_floor", "multi_label", "predictions"}
    keys = common_keys | ({"input_shape", "original_shape", "ratio_pad", "ground_truth", "native_tp"} if branch == "standard" else set())
    for line in payload.splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise ThresholdError("Malformed/truncated prediction cache.") from exc
        if not isinstance(row, dict) or set(row) != keys:
            raise ThresholdError("Cache record schema mismatch.")
        name = row["image_id"]
        if (name not in expected or name in seen or row["schema_version"] != "internal_test_prediction.v2"
                or row["identity_sha256"] != identity_sha256 or row["branch"] != branch
                or row["country"] != expected[name]["country"] or row["nms_iou"] != .5
                or row["confidence_floor"] != (.001 if branch == "standard" else .19)
                or row["multi_label"] is not (branch == "standard")):
            raise ThresholdError("Cache image identity, duplicates, country, branch or frozen settings mismatch.")
        seen.add(name)
        r = expected[name]
        if not isinstance(row["predictions"], list) or len(row["predictions"]) >= 300:
            raise ThresholdError("Cache predictions missing or detection cap reached.")
        for box in row["predictions"]:
            _box_valid(box, native=branch == "standard", width=r["width"], height=r["height"], floor=row["confidence_floor"])
        if branch == "standard":
            shape = row["input_shape"]
            if (not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape)
                    or row["original_shape"] != [r["height"], r["width"]]):
                raise ThresholdError("Native cache image-shape mismatch.")
            ratio = row["ratio_pad"]
            if (not isinstance(ratio, list) or len(ratio) != 2
                    or any(not isinstance(pair, list) or len(pair) != 2 for pair in ratio)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for pair in ratio for v in pair)
                    or any(v <= 0 for v in ratio[0]) or any(v < 0 for v in ratio[1])):
                raise ThresholdError("Invalid native letterbox transform.")
            if sorted(b["class_id"] for b in row["ground_truth"]) != sorted(b["class_id"] for b in r["ground_truth"]):
                raise ThresholdError("Native cache target supports differ from immutable labels.")
            for box in row["ground_truth"]:
                _box_valid(box, native=True, width=shape[1], height=shape[0], floor=1.)
                x1, y1, x2, y2 = box["xyxy"]
                if x1 < -1e-3 or y1 < -1e-3 or x2 > shape[1] + 1e-3 or y2 > shape[0] + 1e-3:
                    raise ThresholdError("Native GT exceeds image bounds beyond export float32 rounding tolerance.")
            tp = row["native_tp"]
            if (not isinstance(tp, list) or len(tp) != len(row["predictions"])
                    or any(not isinstance(t, list) or len(t) != 10 or any(type(b) is not bool for b in t) for t in tp)):
                raise ThresholdError("Invalid native validator IoU-match matrix.")
            for c in range(4):
                support = sum(b["class_id"] == c for b in row["ground_truth"])
                if any(sum(t[j] for b, t in zip(row["predictions"], tp, strict=True) if b["class_id"] == c) > support for j in range(10)):
                    raise ThresholdError("Native TP matrix violates one-to-one target support.")
        rows.append(row)
    if len(rows) != len(expected) or seen != set(expected):
        raise ThresholdError("Missing/extra cache image records.")
    return rows, sha


def metric_values(metric: Any) -> dict[str, Any]:
    """Lossless JSON values from the actual framework's DetMetrics object."""
    p, r, ap50, ap = metric.mean_results()
    per_class = {}
    counts = {code: int(metric.nt_per_class[c]) for c, code in enumerate(CODES)}
    for i, c in enumerate(metric.ap_class_index):
        cp, cr, cap50, cap = metric.class_result(i)
        per_class[CODES[int(c)]] = {"precision": float(cp), "recall": float(cr),
                                  "AP50": float(cap50), "AP50_95": float(cap), "target_boxes": counts[CODES[int(c)]]}
    return {"precision": float(p), "recall": float(r), "mAP50": float(ap50),
            "mAP50_95": float(ap), "per_class": per_class, "target_boxes": counts}


def replay_standard(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Native validator TP matrices + ordered confidences -> the same framework AP routine.

    Matching was performed by the actual validator on its original device; preserve
    those masks to avoid changing borderline float32 IoUs during CPU cache replay.
    Exact class/order/support and byte hashes are verified before this function.
    """
    import numpy as np
    from ultralytics.utils.metrics import DetMetrics
    metric = DetMetrics(names=shared.CLASS_NAMES)
    for row in rows:  # retain framework traversal and prediction order, including confidence ties
        target = np.asarray([b["class_id"] for b in row["ground_truth"]], dtype=np.float32)
        metric.update_stats({"tp": np.asarray(row["native_tp"], dtype=bool).reshape(-1, 10),
                             "conf": np.asarray([b["confidence"] for b in row["predictions"]], dtype=np.float32),
                             "pred_cls": np.asarray([b["class_id"] for b in row["predictions"]], dtype=np.float32),
                             "target_cls": target, "target_img": np.unique(target), "im_name": row["image_id"]})
    metric.process(plot=False)
    values = metric_values(metric)
    return {"schema_version": "internal_test_standard_metrics.v2", **values, "native_metric_values": values,
            "branch": "A_actual_framework_validator", "ultralytics_version": config["ultralytics_version"],
            "procedure": "actual DetectionValidator; sealed native TP/confidence cache; exact DetMetrics replay reconciliation",
            "confidence_floor": .001, "nms_iou": .5, "multi_label": True,
            "matching_ious": config["standard_ap_matching_ious"],
            "coordinate_space": "unchanged validator inference-image coordinates, before prediction-mode clipping",
            "precision_recall_convention": STANDARD_PR_NOTICE, "deployment_threshold_selected": False}


def frozen_confusion(gt: dict[str, Any], predictions: Mapping[str, Sequence[Box]], output: Path) -> None:
    """Diagnostic plot only; primary P/R/F1 still use the approved class-aware matcher."""
    import torch
    from ultralytics.utils.metrics import ConfusionMatrix
    matrix = ConfusionMatrix(names=shared.CLASS_NAMES, task="detect")
    for r in gt["images"]:
        boxes = [b for b in predictions[r["image_id"]] if b.confidence >= .19]
        matrix.process_batch(
            {"cls": torch.tensor([b.class_id for b in boxes]), "conf": torch.tensor([b.confidence for b in boxes]),
             "bboxes": torch.tensor([b.xyxy for b in boxes], dtype=torch.float32).reshape(-1, 4)},
            {"cls": torch.tensor([b["class_id"] for b in r["ground_truth"]]),
             "bboxes": torch.tensor([b["xyxy"] for b in r["ground_truth"]], dtype=torch.float32).reshape(-1, 4)},
            conf=0., iou_thres=.5)
    plot_confusion(matrix.matrix, output)


def plot_confusion(matrix: Any, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    names = [*CODES, "background"]
    ax.set(xticks=range(5), yticks=range(5), xticklabels=names, yticklabels=names,
           xlabel="True class", ylabel="Predicted class", title="Frozen baseline: diagnostic confusion matrix\nconf >=0.19; spatial IoU >0.50")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_results(output: Path, gt: dict[str, Any], point: dict[str, Any], countries: dict[str, Any],
                  standard: dict[str, Any], config: dict[str, Any]) -> None:
    """Render only computed values. Validation-selection numbers stay separate."""
    frozen = {"schema_version": "internal_test_operating_point.v2", "model_sha256": config["model_sha256"],
              "split": "canonical_internal_test", "thresholds_frozen_before_test": True,
              "threshold_selection_commit": config["selection"]["commit"],
              "matching": "Validation implementation reused: confidence descending, class ID then xyxy ascending; GT highest IoU then xyxy/stable index; class-aware one-to-one; inclusive thresholds.",
              "overall": point, "countries": countries}
    (output / "internal_test_metrics.json").write_bytes(shared.json_bytes(standard))
    (output / "internal_test_operating_point.json").write_bytes(shared.json_bytes(frozen))
    per_class = [{"class_id": c, "class_name": shared.CLASS_NAMES[c], "target_boxes": gt["counts"]["class_boxes"][code],
                  **{f"standard_{k}": v for k, v in standard["per_class"].get(code, {}).items() if k != "target_boxes"},
                  **{f"frozen_{k}": point[f"{code}_{k}"] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}}
                 for c, code in enumerate(CODES)]
    write_csv(output / "per_class_metrics.csv", per_class)
    rows = []
    for country, data in countries.items():
        s, p = data["support"], data["operating_point"]
        for code in ("ALL", *CODES):
            rows.append({"country": country, "class": code, "images": s["images"],
                         "positive_images": s["positive_images"], "negative_images": s["negative_images"],
                         "class_positive_images": s["positive_images"] if code == "ALL" else s["class_positive_images"][code],
                         "target_boxes": s["targets"] if code == "ALL" else s["class_boxes"][code],
                         **{k: p[("total_" + k if k in ("tp", "fp", "fn") else k) if code == "ALL" else f"{code}_{k}"]
                            for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                         "warning": "LOW SUPPORT; no strong conclusions" if country == "India" and code == "D10" else ""})
    write_csv(output / "country_metrics.csv", rows)
    lines = ["# First/final YOLOv8s baseline internal-test evaluation", "",
             "Canonical public-data leakage-reduced grouped split; not route-independent or teacher-domain results.",
             "Internal test is now evaluated: it must no longer be called untouched after this run.",
             "No threshold/model selection from internal test; no teacher video or official unlabelled test used.", "",
             f"Frozen confidence **{config['confidence']:.2f}**, NMS IoU **{config['nms_iou']:.2f}**, matching IoU **{config['matching_iou']:.2f}**.",
             f"Model SHA-256: `{config['model_sha256']}`", "",
             f"Support: {gt['counts']['images']} images, {gt['counts']['targets']} targets; {gt['counts']['negative_images']} negatives.", "",
             "## Branch A: descriptive standard detector metrics", "",
             f"Ultralytics {config['ultralytics_version']}; AP floor {config['standard_ap_confidence_floor']}, fixed NMS {config['nms_iou']}; 640 square, batch 1, FP32, no augmentation.",
             f"mAP50: {standard['mAP50']:.10f}; mAP50-95: {standard['mAP50_95']:.10f}; precision: {standard['precision']:.10f}; recall: {standard['recall']:.10f}.",
             STANDARD_PR_NOTICE, "",
             "| Class | AP50 | AP50-95 | Standard P | Standard R |",
             "|---|---:|---:|---:|---:|"]
    for code, r in standard["per_class"].items():
        lines.append(f"| {code} | {r['AP50']:.8f} | {r['AP50_95']:.8f} | {r['precision']:.8f} | {r['recall']:.8f} |")
    lines.extend(["", "## Branch B: primary frozen deployment operating point", "", f"Macro-F1: {point['macro_f1']:.10f}.",
                  f"Total TP/FP/FN: {point['total_tp']} / {point['total_fp']} / {point['total_fn']}.",
                  f"Micro P/R/F1: {point['precision']:.10f} / {point['recall']:.10f} / {point['f1']:.10f}.", "",
                  "| Class | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for code in CODES:
        lines.append(f"| {code} | {point[code + '_tp']} | {point[code + '_fp']} | {point[code + '_fn']} | "
                     f"{point[code + '_precision']:.8f} | {point[code + '_recall']:.8f} | {point[code + '_f1']:.8f} |")
    lines.extend(["", f"Negative images: {point['negative_images']}; negative FP: {point['negative_fp']}; FP/image: {point['negative_fp_per_image']:.10f}.",
                  f"Negatives with >=1 FP: {point['negative_images_with_fp']} ({100 * point['negative_images_with_fp_fraction']:.6f}%).", "",
                  "## Country breakdown", "", "| Country | Images | Positive | Negative | Targets | TP/FP/FN | P | R | F1 |",
                  "|---|---:|---:|---:|---:|---|---:|---:|---:|"])
    for country, data in countries.items():
        s, p = data["support"], data["operating_point"]
        lines.append(f"| {country} | {s['images']} | {s['positive_images']} | {s['negative_images']} | {s['targets']} | "
                     f"{p['total_tp']}/{p['total_fp']}/{p['total_fn']} | {p['precision']:.8f} | {p['recall']:.8f} | {p['f1']:.8f} |")
    lines.extend(["", *countries["India"]["warnings"], "",
                  "Country per-class supports and P/R/F1 are in country_metrics.csv and internal_test_operating_point.json.",
                  "Classes without support use zero-denominator value 0, not evidence of failure/success; macro-F1 retains all four classes.", "",
                  CONFUSION_NOTICE, "",
                  "Prediction collection time is not application FPS. Timings, source/Git/environment identities, hashes, and final status are in evaluation_manifest.json.",
                  "Only a COMPLETED manifest validates this report. Preserve failed attempts; technical recovery requires identical scientific identities."])
    (output / "internal_test_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
