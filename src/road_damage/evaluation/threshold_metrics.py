"""Pure, deterministic operating-point metrics; no model or dataset access."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence


class ThresholdError(RuntimeError):
    """An operating-point input or integrity requirement was violated."""


@dataclass(frozen=True)
class Box:
    """Class-labelled, continuous xyxy pixel box, with prediction confidence."""

    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if type(self.class_id) is not int or self.class_id not in range(4):
            raise ThresholdError("Box class ID must be an integer in 0,1,2,3.")
        if len(self.xyxy) != 4 or not all(math.isfinite(v) for v in self.xyxy):
            raise ThresholdError("Box coordinates must be four finite values.")
        x1, y1, x2, y2 = self.xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ThresholdError("Invalid continuous xyxy box.")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ThresholdError("Prediction confidence must be finite and in [0,1].")


def iou(a: Box, b: Box) -> float:
    """Continuous-coordinate IoU, without the integer VOC +1 convention."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def match_image(
    ground_truth: Sequence[Box], predictions: Sequence[Box], matching_iou: float = 0.50
) -> list[tuple[Box, bool]]:
    """Greedy class-aware matching: confidence desc, xyxy asc; GT IoU desc, xyxy asc.

    Identical boxes retain stable input order. Identical predictions/GT are
    interchangeable, so their ordering cannot change aggregate metrics.
    """
    unmatched = {c: sorted((b for b in ground_truth if b.class_id == c), key=lambda b: b.xyxy)
                 for c in range(4)}
    matched = []
    for pred in sorted(predictions, key=lambda b: (-b.confidence, b.class_id, b.xyxy)):
        candidates = unmatched[pred.class_id]
        eligible = [(iou(pred, gt), index) for index, gt in enumerate(candidates)]
        eligible = [(overlap, index) for overlap, index in eligible if overlap >= matching_iou]
        if eligible:
            _, index = min(eligible, key=lambda pair: (-pair[0], candidates[pair[1]].xyxy, pair[1]))
            candidates.pop(index)
        matched.append((pred, bool(eligible)))
    return matched


def metrics_row(
    counts: Sequence[Sequence[int]], nms_iou: float, confidence: float,
    negative_images: int, negative_fp: int, negative_images_with_fp: int,
) -> dict[str, Any]:
    """Zero denominators yield 0; absent classes remain in the four-class mean."""
    row: dict[str, Any] = {"nms_iou": nms_iou, "confidence": confidence}
    exact_f1 = []
    for code, (tp, fp, fn) in zip(("D00", "D10", "D20", "D40"), counts, strict=True):
        f1 = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
        exact_f1.append(f1)
        row.update({f"{code}_tp": tp, f"{code}_fp": fp, f"{code}_fn": fn,
                    f"{code}_precision": tp / (tp + fp) if tp + fp else 0.0,
                    f"{code}_recall": tp / (tp + fn) if tp + fn else 0.0,
                    f"{code}_f1": float(f1)})
    macro = sum(exact_f1, Fraction(0)) / 4
    row.update({"macro_f1": float(macro), "macro_f1_exact": str(macro),
                "total_tp": sum(c[0] for c in counts), "total_fp": sum(c[1] for c in counts),
                "total_fn": sum(c[2] for c in counts), "negative_images": negative_images,
                "negative_predictions": negative_fp, "negative_fp": negative_fp,
                "negative_fp_per_image": negative_fp / negative_images if negative_images else 0.0,
                "negative_images_with_fp": negative_images_with_fp,
                "negative_images_with_fp_fraction": (
                    negative_images_with_fp / negative_images if negative_images else 0.0)})
    return row


def sweep_cache(
    ground_truth: Mapping[str, Sequence[Box]], predictions: Mapping[str, Sequence[Box]],
    nms_iou: float, confidence_grid: Sequence[float], matching_iou: float = 0.50,
) -> list[dict[str, Any]]:
    """Match once in confidence order, then retain prefixes for each threshold.

    Removing lower-confidence predictions cannot change earlier greedy matches.
    This is equivalent to repeating matching at each threshold, including ties.
    """
    if set(ground_truth) != set(predictions):
        raise ThresholdError("Prediction cache must include every validation image, including empty ones.")
    matched = {name: match_image(ground_truth[name], predictions[name], matching_iou)
               for name in sorted(ground_truth)}
    gt_counts = [sum(b.class_id == c for boxes in ground_truth.values() for b in boxes) for c in range(4)]
    negatives = {name for name, boxes in ground_truth.items() if not boxes}
    rows = []
    for threshold in confidence_grid:
        counts = [[0, 0, value] for value in gt_counts]
        negative_fp = 0
        negative_with_fp = 0
        for name, records in matched.items():
            retained = [(box, tp) for box, tp in records if box.confidence >= threshold]
            for box, tp in retained:
                counts[box.class_id][0 if tp else 1] += 1
                counts[box.class_id][2] -= int(tp)
            if name in negatives:
                negative_fp += len(retained)
                negative_with_fp += bool(retained)
        rows.append(metrics_row(counts, nms_iou, threshold, len(negatives), negative_fp, negative_with_fp))
    return rows


def select_operating_point(rows: Sequence[dict[str, Any]], nms_order: Sequence[float]) -> dict[str, Any]:
    """Rank using exact rational macro-F1; provide the actual tie-break survivor counts."""
    if not rows:
        raise ThresholdError("No operating points to select.")
    best_f1 = max(Fraction(r["macro_f1_exact"]) for r in rows)
    primary = [r for r in rows if Fraction(r["macro_f1_exact"]) == best_f1]
    best_fp = min(r["negative_fp"] for r in primary)
    secondary = [r for r in primary if r["negative_fp"] == best_fp]
    best_conf = max(r["confidence"] for r in secondary)
    tertiary = [r for r in secondary if r["confidence"] == best_conf]
    chosen = min(tertiary, key=lambda r: nms_order.index(r["nms_iou"]))
    return {"metrics": dict(chosen), "tie_break_reasoning": {
        "objective": "maximum exact rational macro-F1 across all four classes",
        "best_macro_f1_exact": str(best_f1), "points_after_macro_f1": len(primary),
        "minimum_negative_fp": best_fp, "points_after_negative_fp": len(secondary),
        "highest_confidence": best_conf, "points_after_confidence": len(tertiary),
        "final_fallback": "first candidate in declared NMS order",
        "nms_order": list(nms_order), "fallback_used": len(tertiary) > 1}}
