"""Prepare a small, unlabeled full-resolution frame pilot for human review."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.video.inspect_video import (
    VideoInspectionError,
    _capture_metadata,
    _load_opencv,
    _project_relative,
    _resolve_from_project_root,
    _sha256,
    build_metadata_validation,
    format_timestamp,
)
from road_damage.video.reconnaissance import (
    _empty_output,
    _region_masks,
    _safe_timestamp,
    _write_image,
    load_config as load_roi_config,
)


LOGGER = logging.getLogger("road_damage.dataset.adjudication_pilot")
MANIFEST_FILENAME = "pilot_manifest.json"
SUMMARY_FILENAME = "summary.json"
SOURCE_TYPES = ("candidate_interval", "normal_negative", "hard_negative")
QUALITY_FIELDS = (
    "full_frame_brightness",
    "road_roi_brightness",
    "road_roi_sharpness_laplacian_variance",
    "road_visibility_contrast_stddev",
    "road_visibility_dynamic_range_p10_p90",
    "previous_road_roi_mean_absolute_difference",
)


class AdjudicationPilotError(VideoInspectionError):
    """Raised when the adjudication pilot cannot be created safely."""


def parse_timestamp(value: str) -> float:
    """Parse HH:MM:SS[.sss] into seconds."""
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise AdjudicationPilotError(f"Invalid timestamp {value!r}; use HH:MM:SS.")
    try:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    except ValueError as exc:
        raise AdjudicationPilotError(f"Invalid timestamp: {value!r}.") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise AdjudicationPilotError(f"Timestamp is out of range: {value!r}.")
    return hours * 3600.0 + minutes * 60.0 + seconds


def timestamp_to_frame_index(timestamp_seconds: float, fps: float) -> int:
    """Map a timestamp to the nearest zero-based constant-FPS frame index."""
    if timestamp_seconds < 0 or not math.isfinite(timestamp_seconds):
        raise AdjudicationPilotError("Timestamp seconds must be finite and non-negative.")
    if fps <= 0 or not math.isfinite(fps):
        raise AdjudicationPilotError("FPS must be finite and positive.")
    return int(round(timestamp_seconds * fps))


def _load_json(path: Path, description: str) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_from_project_root(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdjudicationPilotError(f"{description} does not exist: {resolved}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AdjudicationPilotError(f"Could not read {description}: {resolved}") from exc
    if not isinstance(value, dict):
        raise AdjudicationPilotError(f"{description} must contain an object.")
    return resolved, value


def selection_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized interval specifications in deterministic config order."""
    settings = config["selection"]
    specs: list[dict[str, Any]] = []
    for interval in config["candidate_intervals"]:
        specs.append(
            {
                **interval,
                "type": "candidate_interval",
                "sampling_fps": float(settings["candidate_sampling_fps"]),
                "context_before_seconds": float(
                    settings["candidate_context_before_seconds"]
                ),
                "context_after_seconds": float(
                    settings["candidate_context_after_seconds"]
                ),
            }
        )
    for interval in config["negative_intervals"]:
        specs.append(
            {
                **interval,
                "sampling_fps": float(settings["negative_sampling_fps"]),
                "context_before_seconds": 0.0,
                "context_after_seconds": 0.0,
            }
        )
    return specs


def _interval_seconds(interval: dict[str, Any]) -> tuple[float, float]:
    start = parse_timestamp(str(interval["start"]))
    end = parse_timestamp(str(interval["end"]))
    if end < start:
        raise AdjudicationPilotError(
            f"Interval {interval.get('id', '<unnamed>')} ends before it starts."
        )
    return start, end


def _avoid_ranges(config: dict[str, Any]) -> list[tuple[float, float]]:
    return [_interval_seconds(interval) for interval in config["avoid_intervals"]]


def generate_candidate_indices(
    spec: dict[str, Any],
    fps: float,
    frame_count: int,
    avoid_ranges: Sequence[tuple[float, float]],
) -> list[int]:
    """Generate fixed-rate frame candidates inside one configured sampling window."""
    start, end = _interval_seconds(spec)
    window_start = max(0.0, start - float(spec["context_before_seconds"]))
    duration = (frame_count - 1) / fps
    window_end = min(duration, end + float(spec["context_after_seconds"]))
    step = max(1, round(fps / float(spec["sampling_fps"])))
    first = timestamp_to_frame_index(window_start, fps)
    last = min(frame_count - 1, timestamp_to_frame_index(window_end, fps))
    indices = []
    for index in range(first, last + 1, step):
        seconds = index / fps
        if not any(avoid_start <= seconds <= avoid_end for avoid_start, avoid_end in avoid_ranges):
            indices.append(index)
    return indices


def validate_pilot_config(config: dict[str, Any]) -> None:
    """Validate the dependency-free JSON-compatible pilot configuration."""
    try:
        selection = config["selection"]
        target = int(selection["target_count"])
        candidate_fps = float(selection["candidate_sampling_fps"])
        negative_fps = float(selection["negative_sampling_fps"])
        context_before = float(selection["candidate_context_before_seconds"])
        context_after = float(selection["candidate_context_after_seconds"])
        weights = selection["quality_weights"]
        quality_weights = [float(weights[name]) for name in (
            "roi_sharpness", "road_visibility_contrast", "temporal_change"
        )]
        frame_quality = int(config["frame_output"]["jpeg_quality"])
        sheet = config["contact_sheet"]
        dimensions = [int(sheet[name]) for name in (
            "thumbnail_width", "thumbnail_height", "columns", "rows", "label_height"
        )]
        sheet_quality = int(sheet["jpeg_quality"])
        parse_timestamp(str(config["mask_preview"]["timestamp"]))
        specs = selection_specs(config)
        avoid_ranges = _avoid_ranges(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise AdjudicationPilotError(
            "Pilot configuration is missing a required value or has an invalid type."
        ) from exc

    if target <= 0 or candidate_fps <= 0 or negative_fps <= 0:
        raise AdjudicationPilotError("Target and sampling rates must be positive.")
    if context_before < 0 or context_after < 0:
        raise AdjudicationPilotError("Context windows must be non-negative.")
    if any(weight < 0 for weight in quality_weights) or not math.isclose(
        sum(quality_weights), 1.0, abs_tol=1e-9
    ):
        raise AdjudicationPilotError("Quality weights must be non-negative and sum to 1.")
    if any(value <= 0 for value in dimensions):
        raise AdjudicationPilotError("Contact-sheet dimensions must be positive.")
    if not 1 <= frame_quality <= 100 or not 1 <= sheet_quality <= 100:
        raise AdjudicationPilotError("JPEG quality must be between 1 and 100.")
    if not specs or not avoid_ranges:
        raise AdjudicationPilotError("Selection and avoid intervals cannot be empty.")

    ids: set[str] = set()
    counts = {source_type: 0 for source_type in SOURCE_TYPES}
    for spec in specs:
        spec_id = str(spec.get("id", ""))
        source_type = str(spec.get("type", ""))
        quota = int(spec.get("quota", 0))
        _interval_seconds(spec)
        if not spec_id or spec_id in ids:
            raise AdjudicationPilotError(f"Duplicate or empty interval id: {spec_id!r}.")
        if source_type not in SOURCE_TYPES or quota <= 0:
            raise AdjudicationPilotError(f"Invalid type or quota for {spec_id}.")
        ids.add(spec_id)
        counts[source_type] += quota
    if sum(counts.values()) != target:
        raise AdjudicationPilotError(
            f"Configured quotas total {sum(counts.values())}, expected {target}."
        )
    if counts["normal_negative"] != counts["hard_negative"]:
        raise AdjudicationPilotError(
            "Normal and hard-negative quotas must be balanced in this pilot."
        )


def load_pilot_config(path: Path) -> dict[str, Any]:
    """Load and validate the pilot configuration."""
    _, config = _load_json(path, "adjudication pilot configuration")
    validate_pilot_config(config)
    return config


def _quality_metrics(
    frame: Any, road_mask: Any, previous_gray: Any | None, cv2: Any, np: Any
) -> tuple[dict[str, float | None], Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[road_mask > 0]
    if roi_pixels.size == 0:
        raise AdjudicationPilotError("The effective road ROI contains no pixels.")
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    low, high = np.percentile(roi_pixels, (10, 90))
    change = None
    if previous_gray is not None:
        change = cv2.mean(cv2.absdiff(gray, previous_gray), mask=road_mask)[0]
    return {
        "full_frame_brightness": round(float(gray.mean()), 4),
        "road_roi_brightness": round(float(roi_pixels.mean()), 4),
        "road_roi_sharpness_laplacian_variance": round(
            float(laplacian[road_mask > 0].var()), 4
        ),
        "road_visibility_contrast_stddev": round(float(roi_pixels.std()), 4),
        "road_visibility_dynamic_range_p10_p90": round(float(high - low), 4),
        "previous_road_roi_mean_absolute_difference": (
            round(float(change), 4) if change is not None else None
        ),
    }, gray


def _normalized_ranks(values: Sequence[float]) -> list[float]:
    if len(values) == 1:
        return [0.5]
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(ordered):
        result[index] = rank / (len(values) - 1)
    return result


def select_representatives(
    candidates: Sequence[dict[str, Any]],
    quota: int,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    """Select one representative per temporal bin using within-bin quality ranks."""
    ordered = sorted(candidates, key=lambda item: int(item["requested_frame_index"]))
    if quota <= 0 or len(ordered) < quota:
        raise AdjudicationPilotError(
            f"Cannot select quota {quota} from {len(ordered)} decoded candidates."
        )
    selected: list[dict[str, Any]] = []
    for bin_number in range(quota):
        left = bin_number * len(ordered) // quota
        right = (bin_number + 1) * len(ordered) // quota
        temporal_bin = ordered[left:right]
        sharpness = _normalized_ranks([
            float(item["road_roi_sharpness_laplacian_variance"])
            for item in temporal_bin
        ])
        visibility = _normalized_ranks([
            float(item["road_visibility_contrast_stddev"]) for item in temporal_bin
        ])
        change = _normalized_ranks([
            float(item["previous_road_roi_mean_absolute_difference"] or 0.0)
            for item in temporal_bin
        ])
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, item in enumerate(temporal_bin):
            score = (
                sharpness[index] * float(weights["roi_sharpness"])
                + visibility[index] * float(weights["road_visibility_contrast"])
                + change[index] * float(weights["temporal_change"])
            )
            scored.append((score, -int(item["requested_frame_index"]), item))
        score, _, winner = max(scored, key=lambda value: (value[0], value[1]))
        result = dict(winner)
        result["selection"] = {
            "temporal_bin_one_based": bin_number + 1,
            "temporal_bin_count": quota,
            "candidate_count_in_bin": len(temporal_bin),
            "score": round(score, 6),
            "criterion": (
                "highest weighted within-bin ranks for ROI sharpness, road "
                "contrast, and change from the previous sampled ROI"
            ),
            "tie_breaker": "earliest source frame index",
            "no_quality_threshold_applied": True,
        }
        selected.append(result)
    return selected


def _decode_metric_pool(
    capture: Any,
    indices: Sequence[int],
    spec: dict[str, Any],
    metadata: dict[str, Any],
    road_mask: Any,
    cv2: Any,
    np: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    previous_gray = None
    for frame_index in indices:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            warnings.append(f"{spec['id']}: could not seek to frame {frame_index}.")
            continue
        read_ok, frame = capture.read()
        if not read_ok or frame is None or frame.size == 0:
            warnings.append(f"{spec['id']}: could not decode frame {frame_index}.")
            continue
        if frame.shape[1] != metadata["width"] or frame.shape[0] != metadata["height"]:
            raise AdjudicationPilotError(
                f"Unexpected decoded size at frame {frame_index}: "
                f"{frame.shape[1]}x{frame.shape[0]}."
            )
        metrics, previous_gray = _quality_metrics(
            frame, road_mask, previous_gray, cv2, np
        )
        seconds = frame_index / float(metadata["fps"])
        candidates.append(
            {
                "requested_frame_index": frame_index,
                "timestamp_seconds": round(seconds, 6),
                "timestamp_ms": round(seconds * 1000),
                "timestamp_human": format_timestamp(seconds),
                **metrics,
            }
        )
    return candidates, warnings


def _summary_stats(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in QUALITY_FIELDS:
        values = [float(item[field]) for item in entries if item.get(field) is not None]
        result[field] = {"count": 0} if not values else {
            "count": len(values),
            "minimum": round(min(values), 4),
            "maximum": round(max(values), 4),
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
        }
    return result


def _write_png(path: Path, image: Any, cv2: Any) -> None:
    try:
        written = cv2.imwrite(str(path), image)
    except cv2.error as exc:
        raise AdjudicationPilotError(f"OpenCV could not write image: {path}") from exc
    if not written or not path.is_file():
        raise AdjudicationPilotError(f"Could not write image: {path}")


def _mask_artifacts(
    capture: Any,
    temporary: Path,
    output: Path,
    config: dict[str, Any],
    roi_config: dict[str, Any],
    metadata: dict[str, Any],
    road_mask: Any,
    overlay_mask: Any,
    cv2: Any,
) -> dict[str, Any]:
    masks_dir = temporary / "masks"
    masks_dir.mkdir()
    road_name = "effective_road_roi_mask.png"
    overlay_name = "overlay_exclusion_mask.png"
    preview_name = "roi_overlay_preview.jpg"
    _write_png(masks_dir / road_name, road_mask, cv2)
    _write_png(masks_dir / overlay_name, overlay_mask, cv2)

    preview_seconds = parse_timestamp(str(config["mask_preview"]["timestamp"]))
    preview_index = timestamp_to_frame_index(preview_seconds, float(metadata["fps"]))
    if not 0 <= preview_index < int(metadata["frame_count"]):
        raise AdjudicationPilotError("Configured mask-preview timestamp is outside video.")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, preview_index):
        raise AdjudicationPilotError("Could not seek to the mask-preview frame.")
    read_ok, preview = capture.read()
    if not read_ok or preview is None or preview.size == 0:
        raise AdjudicationPilotError("Could not decode the mask-preview frame.")

    height, width = preview.shape[:2]
    polygon = []
    for x_value, y_value in roi_config["valid_region"]["polygon_normalized"]:
        polygon.append((round(x_value * (width - 1)), round(y_value * (height - 1))))
    import numpy as np

    cv2.polylines(preview, [np.array(polygon, np.int32)], True, (0, 255, 0), 3)
    for exclusion in roi_config["overlay_exclusions"]:
        x_min, y_min, x_max, y_max = exclusion["xyxy_normalized"]
        first = (round(x_min * (width - 1)), round(y_min * (height - 1)))
        second = (round(x_max * (width - 1)), round(y_max * (height - 1)))
        cv2.rectangle(preview, first, second, (0, 0, 255), 2)
        label_y = min(height - 5, max(14, first[1] + 14))
        cv2.putText(
            preview,
            str(exclusion["name"]),
            (first[0] + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        preview,
        "green: road ROI | red: permanent overlay exclusions",
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    _write_image(masks_dir / preview_name, preview, 95, cv2)
    total_pixels = width * height
    return {
        "road_roi_name": roi_config["valid_region"]["name"],
        "road_roi_polygon_normalized": roi_config["valid_region"][
            "polygon_normalized"
        ],
        "overlay_exclusions": roi_config["overlay_exclusions"],
        "effective_road_roi_pixel_count": int(cv2.countNonZero(road_mask)),
        "effective_road_roi_fraction": round(
            cv2.countNonZero(road_mask) / total_pixels, 6
        ),
        "overlay_exclusion_pixel_count": int(cv2.countNonZero(overlay_mask)),
        "overlay_exclusion_fraction": round(
            cv2.countNonZero(overlay_mask) / total_pixels, 6
        ),
        "effective_road_roi_mask_path": _project_relative(output / "masks" / road_name),
        "overlay_exclusion_mask_path": _project_relative(output / "masks" / overlay_name),
        "preview_path": _project_relative(output / "masks" / preview_name),
        "preview_source_frame_index": preview_index,
        "preview_timestamp_human": format_timestamp(preview_seconds),
        "interpretation": (
            "The road mask is the configured polygon after permanent overlay "
            "exclusions; previews do not alter source frames."
        ),
    }


def _contact_sheet(entries: Sequence[tuple[Any, str]], config: dict[str, Any], cv2: Any, np: Any) -> Any:
    settings = config["contact_sheet"]
    width = int(settings["thumbnail_width"])
    height = int(settings["thumbnail_height"])
    columns = int(settings["columns"])
    rows = int(settings["rows"])
    label_height = int(settings["label_height"])
    sheet = np.full((rows * (height + label_height), columns * width, 3), 245, np.uint8)
    for index, (thumbnail, label) in enumerate(entries):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + label_height)
        sheet[y : y + height, x : x + width] = thumbnail
        cv2.putText(
            sheet, label, (x + 5, y + height + label_height - 9),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA,
        )
    return sheet


def _extract_selected(
    capture: Any,
    selected: Sequence[dict[str, Any]],
    temporary: Path,
    output: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    source: Path,
    source_sha256: str,
    cv2: Any,
    np: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    frames_dir = temporary / "frames"
    sheets_dir = temporary / "contact_sheets"
    frames_dir.mkdir()
    sheets_dir.mkdir()
    settings = config["contact_sheet"]
    page_size = int(settings["columns"]) * int(settings["rows"])
    page_entries: list[tuple[Any, str]] = []
    records: list[dict[str, Any]] = []
    sheet_paths: list[str] = []

    for pilot_number, item in enumerate(selected, start=1):
        frame_index = int(item["requested_frame_index"])
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise AdjudicationPilotError(f"Could not seek to selected frame {frame_index}.")
        read_ok, frame = capture.read()
        if not read_ok or frame is None or frame.size == 0:
            raise AdjudicationPilotError(f"Could not decode selected frame {frame_index}.")
        if frame.shape[1] != metadata["width"] or frame.shape[0] != metadata["height"]:
            raise AdjudicationPilotError(f"Selected frame {frame_index} is not full resolution.")
        reported = capture.get(cv2.CAP_PROP_POS_FRAMES)
        source_frame_index = frame_index
        if math.isfinite(reported) and reported >= 1 and abs(round(reported) - 1 - frame_index) <= 1:
            source_frame_index = round(reported) - 1

        short_type = {
            "candidate_interval": "candidate",
            "normal_negative": "normal",
            "hard_negative": "hard",
        }[item["candidate_source_type"]]
        filename = (
            f"pilot_{pilot_number:04d}_{short_type}_"
            f"t{_safe_timestamp(item['timestamp_human'])}_f{frame_index:08d}.jpg"
        )
        _write_image(
            frames_dir / filename,
            frame,
            int(config["frame_output"]["jpeg_quality"]),
            cv2,
        )
        page_number = (pilot_number - 1) // page_size + 1
        sheet_name = f"pilot_contact_sheet_{page_number:02d}.jpg"
        thumbnail = cv2.resize(
            frame,
            (int(settings["thumbnail_width"]), int(settings["thumbnail_height"])),
            interpolation=cv2.INTER_AREA,
        )
        page_entries.append(
            (thumbnail, f"#{pilot_number:03d} {short_type} {item['timestamp_human']}")
        )
        record = {
            "pilot_id": f"pilot_{pilot_number:04d}",
            "image_filename": filename,
            "image_path": _project_relative(output / "frames" / filename),
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0]),
            "requested_frame_index": frame_index,
            "source_frame_index": source_frame_index,
            "frame_number_one_based": source_frame_index + 1,
            "timestamp_ms": item["timestamp_ms"],
            "timestamp_seconds": item["timestamp_seconds"],
            "timestamp_human": item["timestamp_human"],
            "source_interval_id": item["source_interval_id"],
            "source_interval_start": item["source_interval_start"],
            "source_interval_end": item["source_interval_end"],
            "sampling_window_start_seconds": item["sampling_window_start_seconds"],
            "sampling_window_end_seconds": item["sampling_window_end_seconds"],
            "candidate_source_type": item["candidate_source_type"],
            "provisional_selection_basis": item.get("provisional_selection_basis"),
            "interval_sampling_fps": item["interval_sampling_fps"],
            **{field: item[field] for field in QUALITY_FIELDS},
            "selection": item["selection"],
            "contact_sheet_number": page_number,
            "contact_sheet_path": _project_relative(output / "contact_sheets" / sheet_name),
            "provenance": {
                "source_video_path": _project_relative(source),
                "source_video_sha256": source_sha256,
                "timestamp_basis": "requested zero-based frame index / detected FPS",
                "decoded_directly_from_original_video": True,
                "saved_as_full_resolution_jpeg": True,
            },
            "review_status": "unreviewed",
        }
        records.append(record)

        if len(page_entries) == page_size or pilot_number == len(selected):
            _write_image(
                sheets_dir / sheet_name,
                _contact_sheet(page_entries, config, cv2, np),
                int(settings["jpeg_quality"]),
                cv2,
            )
            sheet_paths.append(_project_relative(output / "contact_sheets" / sheet_name))
            page_entries = []
    return records, sheet_paths


def run_adjudication_pilot(
    input_path: Path,
    inspection_report_path: Path,
    reconnaissance_manifest_path: Path,
    roi_config_path: Path,
    pilot_config_path: Path,
    output_dir: Path,
) -> Path:
    """Create the deterministic Phase 1.6 review-only pilot."""
    source = _resolve_from_project_root(input_path)
    output = _resolve_from_project_root(output_dir)
    roi_config_path = _resolve_from_project_root(roi_config_path)
    pilot_config_path = _resolve_from_project_root(pilot_config_path)
    if not source.is_file():
        raise AdjudicationPilotError(f"Input video does not exist: {source}")
    if output == source or output == source.parent:
        raise AdjudicationPilotError("Output must be separate from the raw video.")
    _empty_output(output)

    inspection_path, inspection = _load_json(
        inspection_report_path, "Phase 1 inspection report"
    )
    reconnaissance_path, reconnaissance = _load_json(
        reconnaissance_manifest_path, "Phase 1.5 reconnaissance manifest"
    )
    roi_config = load_roi_config(roi_config_path)
    pilot_config = load_pilot_config(pilot_config_path)
    try:
        source_sha256 = str(inspection["source"]["sha256"])
        source_size = int(inspection["source"]["size_bytes"])
        reconnaissance_sha256 = str(
            reconnaissance["source"]["sha256_from_phase1_report"]
        )
        expected_video = inspection["video"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AdjudicationPilotError("Input manifests lack required provenance.") from exc
    if source_sha256 != reconnaissance_sha256:
        raise AdjudicationPilotError("Phase 1 and Phase 1.5 source hashes disagree.")
    before = source.stat()
    if before.st_size != source_size:
        raise AdjudicationPilotError("Raw video size differs from the Phase 1 report.")

    cv2 = _load_opencv()
    try:
        import numpy as np
    except ImportError as exc:
        raise AdjudicationPilotError(
            "NumPy is unavailable in the project environment."
        ) from exc
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise AdjudicationPilotError(f"OpenCV could not open: {source}")

    temporary: Path | None = None
    try:
        metadata = _capture_metadata(capture, cv2, source)
        expected = {
            field: expected_video.get(field)
            for field in ("width", "height", "fps", "frame_count", "duration_seconds", "container")
        }
        expected["codec"] = expected_video.get("codec_description")
        metadata_check = build_metadata_validation(metadata, expected)
        if metadata_check["discrepancies"]:
            raise AdjudicationPilotError(
                "Video differs from Phase 1: " + "; ".join(metadata_check["discrepancies"])
            )
        if int(metadata["width"]) != 720 or int(metadata["height"]) != 576:
            raise AdjudicationPilotError(
                f"Expected 720x576 source, detected {metadata['resolution']}."
            )

        road_mask, overlay_mask = _region_masks(
            int(metadata["width"]), int(metadata["height"]), roi_config, cv2, np
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".adjudication-pilot-", dir=output.parent))
        mask_details = _mask_artifacts(
            capture, temporary, output, pilot_config, roi_config,
            metadata, road_mask, overlay_mask, cv2,
        )

        warnings: list[str] = []
        selected: list[dict[str, Any]] = []
        avoid_ranges = _avoid_ranges(pilot_config)
        weights = pilot_config["selection"]["quality_weights"]
        specs = selection_specs(pilot_config)
        for spec_number, spec in enumerate(specs, start=1):
            indices = generate_candidate_indices(
                spec,
                float(metadata["fps"]),
                int(metadata["frame_count"]),
                avoid_ranges,
            )
            candidates, decode_warnings = _decode_metric_pool(
                capture, indices, spec, metadata, road_mask, cv2, np
            )
            warnings.extend(decode_warnings)
            chosen = select_representatives(
                candidates, int(spec["quota"]), weights
            )
            start, end = _interval_seconds(spec)
            window_start = max(0.0, start - float(spec["context_before_seconds"]))
            window_end = min(
                (int(metadata["frame_count"]) - 1) / float(metadata["fps"]),
                end + float(spec["context_after_seconds"]),
            )
            for item in chosen:
                item.update(
                    {
                        "source_interval_id": spec["id"],
                        "source_interval_start": spec["start"],
                        "source_interval_end": spec["end"],
                        "sampling_window_start_seconds": round(window_start, 6),
                        "sampling_window_end_seconds": round(window_end, 6),
                        "candidate_source_type": spec["type"],
                        "provisional_selection_basis": spec.get("selection_basis"),
                        "interval_sampling_fps": spec["sampling_fps"],
                    }
                )
            selected.extend(chosen)
            LOGGER.info(
                "Selected %d/%d from %s (%d/%d intervals)",
                len(chosen), len(candidates), spec["id"], spec_number, len(specs),
            )

        expected_count = int(pilot_config["selection"]["target_count"])
        if len(selected) != expected_count:
            raise AdjudicationPilotError(
                f"Selected {len(selected)} frames; expected {expected_count}."
            )
        frame_indices = [int(item["requested_frame_index"]) for item in selected]
        if len(frame_indices) != len(set(frame_indices)):
            raise AdjudicationPilotError("Selection contains duplicate frame indices.")
        selected.sort(key=lambda item: int(item["requested_frame_index"]))
        records, sheet_paths = _extract_selected(
            capture, selected, temporary, output, pilot_config, metadata,
            source, source_sha256, cv2, np,
        )
    except Exception:
        capture.release()
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        capture.release()

    after = source.stat()
    unchanged = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    if not unchanged:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise AdjudicationPilotError(
            "Raw video size or modification time changed; pilot outputs were discarded."
        )
    if temporary is None:
        raise AdjudicationPilotError("Internal error: temporary output was not created.")

    counts = {
        source_type: sum(
            record["candidate_source_type"] == source_type for record in records
        )
        for source_type in SOURCE_TYPES
    }
    manifest = {
        "schema_version": "1.0",
        "phase": pilot_config["phase"],
        "purpose": (
            "Unlabeled, full-resolution pilot for human adjudication of whether "
            "the source video contains sufficient reliable road-damage examples."
        ),
        "source": {
            "path": _project_relative(source),
            "size_bytes": before.st_size,
            "sha256_from_phase1_report": source_sha256,
            "metadata": metadata,
            "immutable_source_check": {
                "size_and_mtime_unchanged_during_run": unchanged,
                "opened_read_only_by_utility": True,
            },
        },
        "input_provenance": {
            "inspection_report_path": _project_relative(inspection_path),
            "reconnaissance_manifest_path": _project_relative(reconnaissance_path),
            "reconnaissance_manifest_sha256": _sha256(reconnaissance_path),
            "roi_config_path": _project_relative(roi_config_path),
            "roi_config_sha256": _sha256(roi_config_path),
            "pilot_config_path": _project_relative(pilot_config_path),
            "pilot_config_sha256": _sha256(pilot_config_path),
        },
        "selection_policy": {
            "configured_target_count": int(pilot_config["selection"]["target_count"]),
            "selected_count": len(records),
            "counts_by_candidate_source_type": counts,
            "candidate_pool_rate_fps": float(
                pilot_config["selection"]["candidate_sampling_fps"]
            ),
            "negative_pool_rate_fps": float(
                pilot_config["selection"]["negative_sampling_fps"]
            ),
            "candidate_context_before_seconds": float(
                pilot_config["selection"]["candidate_context_before_seconds"]
            ),
            "candidate_context_after_seconds": float(
                pilot_config["selection"]["candidate_context_after_seconds"]
            ),
            "selection_method": (
                "one frame per temporal bin, preferring within-bin ROI sharpness, "
                "road contrast, and temporal change; no quality rejection threshold"
            ),
            "avoid_intervals": pilot_config["avoid_intervals"],
            "class_labels_assigned": False,
            "dataset_splits_created": False,
        },
        "mask_configuration": mask_details,
        "image_output": {
            "format": "JPEG",
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "jpeg_quality": int(pilot_config["frame_output"]["jpeg_quality"]),
            "source": "decoded directly from original AVI, not reconnaissance thumbnails",
        },
        "quality_metric_definitions": {
            "brightness": "mean grayscale intensity on a 0-255 scale",
            "roi_sharpness": "variance of grayscale Laplacian within effective road ROI",
            "road_visibility": "ROI grayscale standard deviation and p90-p10 range",
            "temporal_change": "mean absolute grayscale difference in ROI from prior pool frame",
        },
        "quality_summary": _summary_stats(records),
        "contact_sheet_paths": sheet_paths,
        "frames": records,
        "warnings": warnings + [
            "Candidate/easy/hard source types are sampling strata, not road-damage labels.",
            "Human review is required before any annotation or dataset decision.",
            "Frames may contain faces, number plates, or location clues and remain private project data.",
        ],
        "runtime": {
            "python_version": sys.version.split()[0],
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        },
    }
    summary = {
        "schema_version": "1.0",
        "phase": pilot_config["phase"],
        "pilot_frame_count": len(records),
        "counts_by_candidate_source_type": counts,
        "contact_sheet_count": len(sheet_paths),
        "mask_artifact_count": 3,
        "quality_summary": manifest["quality_summary"],
        "source_unchanged": unchanged,
        "annotations_created": False,
        "dataset_splits_created": False,
        "final_dataset_created": False,
        "manual_review_required": True,
        "manifest_path": _project_relative(output / MANIFEST_FILENAME),
        "frames_directory": _project_relative(output / "frames"),
        "contact_sheets_directory": _project_relative(output / "contact_sheets"),
        "masks_directory": _project_relative(output / "masks"),
        "warnings": manifest["warnings"],
    }
    try:
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (temporary / SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if output.exists():
            output.rmdir()
        os.replace(temporary, output)
    except OSError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise AdjudicationPilotError(f"Could not commit outputs to: {output}") from exc

    manifest_path = output / MANIFEST_FILENAME
    LOGGER.info("Adjudication pilot completed: %s", _project_relative(manifest_path))
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an unlabeled full-resolution adjudication frame pilot."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--inspection-report", required=True, type=Path)
    parser.add_argument("--reconnaissance-manifest", required=True, type=Path)
    parser.add_argument("--roi-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_adjudication_pilot(
            args.input,
            args.inspection_report,
            args.reconnaissance_manifest,
            args.roi_config,
            args.pilot_config,
            args.output_dir,
        )
    except (AdjudicationPilotError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
