"""Create thumbnails, quality metrics, and contact sheets for reconnaissance."""

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
    build_metadata_validation,
    format_timestamp,
)


LOGGER = logging.getLogger("road_damage.video.reconnaissance")
MANIFEST_FILENAME = "reconnaissance_manifest.json"
QUALITY_FIELDS = (
    "full_frame_brightness",
    "full_frame_sharpness_laplacian_variance",
    "valid_region_brightness",
    "valid_region_sharpness_laplacian_variance",
    "previous_valid_region_mean_absolute_difference",
)


class ReconnaissanceError(VideoInspectionError):
    """Raised when reconnaissance cannot be completed safely."""


def interval_frame_indices(
    frame_count: int,
    fps: float,
    interval_seconds: float,
    edge_margin_seconds: float,
) -> list[int]:
    """Return deterministic zero-based indices at a fixed interval."""
    if frame_count < 3 or not math.isfinite(fps) or fps <= 0:
        raise ReconnaissanceError("Video frame count and FPS must be positive.")
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ReconnaissanceError("Sampling interval must be positive.")
    if not math.isfinite(edge_margin_seconds) or edge_margin_seconds < 0:
        raise ReconnaissanceError("Edge margin must be non-negative.")

    step = max(1, round(interval_seconds * fps))
    margin = max(1, round(edge_margin_seconds * fps))
    if margin >= frame_count - margin:
        raise ReconnaissanceError("The edge margin leaves no frames to sample.")
    return list(range(margin, frame_count - margin, step))


def expected_contact_sheet_count(
    candidate_count: int, columns: int, rows: int
) -> int:
    """Return the number of sheets needed for the configured grid."""
    if candidate_count < 0 or columns <= 0 or rows <= 0:
        raise ValueError("Counts and contact-sheet dimensions must be valid.")
    page_size = columns * rows
    return (candidate_count + page_size - 1) // page_size


def validate_config(config: dict[str, Any]) -> None:
    """Validate the JSON-compatible YAML configuration."""
    try:
        numeric_values = {
            "interval": float(config["sampling"]["interval_seconds"]),
            "margin": float(config["sampling"]["edge_margin_seconds"]),
            "thumb_width": int(config["thumbnail"]["width"]),
            "thumb_height": int(config["thumbnail"]["height"]),
            "thumb_quality": int(config["thumbnail"]["jpeg_quality"]),
            "columns": int(config["contact_sheet"]["columns"]),
            "rows": int(config["contact_sheet"]["rows"]),
            "label_height": int(config["contact_sheet"]["label_height"]),
            "sheet_quality": int(config["contact_sheet"]["jpeg_quality"]),
        }
        polygon = config["valid_region"]["polygon_normalized"]
        exclusions = config["overlay_exclusions"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconnaissanceError(
            "Configuration is missing a required value or has an invalid type."
        ) from exc

    if numeric_values["interval"] <= 0 or numeric_values["margin"] < 0:
        raise ReconnaissanceError("Sampling interval/margin values are invalid.")
    positive_names = ("thumb_width", "thumb_height", "columns", "rows", "label_height")
    if any(numeric_values[name] <= 0 for name in positive_names):
        raise ReconnaissanceError("Image and grid dimensions must be positive.")
    if any(
        not 1 <= numeric_values[name] <= 100
        for name in ("thumb_quality", "sheet_quality")
    ):
        raise ReconnaissanceError("JPEG quality must be between 1 and 100.")

    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ReconnaissanceError("The valid-region polygon needs at least 3 points.")
    for point in polygon:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(0.0 <= float(value) <= 1.0 for value in point)
        ):
            raise ReconnaissanceError(
                "Valid-region points must be normalized [x, y] pairs."
            )

    if not isinstance(exclusions, list):
        raise ReconnaissanceError("overlay_exclusions must be a list.")
    for exclusion in exclusions:
        try:
            rectangle = [float(value) for value in exclusion["xyxy_normalized"]]
            name = str(exclusion["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconnaissanceError(
                "Each overlay exclusion needs a name and normalized xyxy values."
            ) from exc
        if (
            not name
            or len(rectangle) != 4
            or not all(0.0 <= value <= 1.0 for value in rectangle)
            or rectangle[0] >= rectangle[2]
            or rectangle[1] >= rectangle[3]
        ):
            raise ReconnaissanceError(
                f"Invalid normalized overlay exclusion: {name or '<unnamed>'}."
            )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    resolved = _resolve_from_project_root(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReconnaissanceError(f"{description} does not exist: {resolved}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconnaissanceError(f"Could not read {description}: {resolved}") from exc
    if not isinstance(value, dict):
        raise ReconnaissanceError(f"{description} must contain an object.")
    return value


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate dependency-free, JSON-compatible YAML."""
    config = _load_json(config_path, "reconnaissance configuration")
    validate_config(config)
    return config


def _phase1_report(report_path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_from_project_root(report_path)
    report = _load_json(resolved, "Phase 1 inspection report")
    try:
        report["source"]["sha256"]
        report["source"]["size_bytes"]
        for field in ("width", "height", "fps", "frame_count", "duration_seconds"):
            report["video"][field]
    except (KeyError, TypeError) as exc:
        raise ReconnaissanceError(
            "Phase 1 report is missing required source/video metadata."
        ) from exc
    return resolved, report


def _pixel_point(point: Sequence[float], width: int, height: int) -> tuple[int, int]:
    return (
        min(width - 1, max(0, round(float(point[0]) * (width - 1)))),
        min(height - 1, max(0, round(float(point[1]) * (height - 1)))),
    )


def _region_masks(
    width: int, height: int, config: dict[str, Any], cv2: Any, np: Any
) -> tuple[Any, Any]:
    valid = np.zeros((height, width), dtype=np.uint8)
    overlays = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [_pixel_point(point, width, height) for point in config["valid_region"]["polygon_normalized"]],
        dtype=np.int32,
    )
    cv2.fillPoly(valid, [polygon], 255)
    for exclusion in config["overlay_exclusions"]:
        x_min, y_min, x_max, y_max = exclusion["xyxy_normalized"]
        cv2.rectangle(
            overlays,
            _pixel_point((x_min, y_min), width, height),
            _pixel_point((x_max, y_max), width, height),
            255,
            -1,
        )
    valid[overlays > 0] = 0
    if cv2.countNonZero(valid) == 0:
        raise ReconnaissanceError("The configured valid-region mask is empty.")
    return valid, overlays


def _quality_metrics(
    frame: Any, valid_mask: Any, previous_gray: Any | None, cv2: Any
) -> tuple[dict[str, float | None], Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    difference = None
    if previous_gray is not None:
        difference = cv2.mean(cv2.absdiff(gray, previous_gray), mask=valid_mask)[0]
    return {
        "full_frame_brightness": round(float(gray.mean()), 4),
        "full_frame_sharpness_laplacian_variance": round(float(laplacian.var()), 4),
        "valid_region_brightness": round(float(cv2.mean(gray, mask=valid_mask)[0]), 4),
        "valid_region_sharpness_laplacian_variance": round(
            float(laplacian[valid_mask > 0].var()), 4
        ),
        "previous_valid_region_mean_absolute_difference": (
            round(float(difference), 4) if difference is not None else None
        ),
    }, gray


def summarize_reconnaissance_quality(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize metrics without classifying or rejecting candidates."""
    summaries: dict[str, Any] = {}
    for field in QUALITY_FIELDS:
        values = [
            float(candidate[field])
            for candidate in candidates
            if candidate.get(field) is not None
        ]
        summaries[field] = {"count": 0} if not values else {
            "count": len(values),
            "minimum": round(min(values), 4),
            "maximum": round(max(values), 4),
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
        }
    return summaries


def _safe_timestamp(value: str) -> str:
    return value.replace(":", "-").replace(".", "-")


def _write_image(path: Path, image: Any, quality: int, cv2: Any) -> None:
    try:
        written = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except cv2.error as exc:
        raise ReconnaissanceError(f"OpenCV could not write image: {path}") from exc
    if not written or not path.is_file():
        raise ReconnaissanceError(f"Could not write image: {path}")


def _sheet(
    entries: Sequence[tuple[Any, str]], config: dict[str, Any], cv2: Any, np: Any
) -> Any:
    width = int(config["thumbnail"]["width"])
    height = int(config["thumbnail"]["height"])
    columns = int(config["contact_sheet"]["columns"])
    rows = int(config["contact_sheet"]["rows"])
    label_height = int(config["contact_sheet"]["label_height"])
    sheet = np.full((rows * (height + label_height), columns * width, 3), 245, np.uint8)
    for index, (thumbnail, label) in enumerate(entries):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + label_height)
        sheet[y : y + height, x : x + width] = thumbnail
        cv2.putText(
            sheet, label, (x + 6, y + height + label_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
        )
    return sheet


def _empty_output(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ReconnaissanceError(f"Output path is not a directory: {output_dir}")
    try:
        next(output_dir.iterdir())
    except StopIteration:
        return
    except OSError as exc:
        raise ReconnaissanceError(f"Could not inspect output: {output_dir}") from exc
    raise ReconnaissanceError(
        f"Output directory is not empty: {output_dir}. Move it before rerunning."
    )


def run_reconnaissance(
    input_path: Path,
    inspection_report_path: Path,
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Generate the Phase 1.5 manifest, thumbnails, masks, and contact sheets."""
    source = _resolve_from_project_root(input_path)
    output = _resolve_from_project_root(output_dir)
    config_path = _resolve_from_project_root(config_path)
    if not source.is_file():
        raise ReconnaissanceError(f"Input video does not exist: {source}")
    if output == source or output == source.parent:
        raise ReconnaissanceError("Output must be separate from the raw video.")
    _empty_output(output)

    config = load_config(config_path)
    phase1_path, phase1 = _phase1_report(inspection_report_path)
    before = source.stat()
    if before.st_size != int(phase1["source"]["size_bytes"]):
        raise ReconnaissanceError("Raw video size differs from the Phase 1 report.")

    cv2 = _load_opencv()
    try:
        import numpy as np
    except ImportError as exc:
        raise ReconnaissanceError("NumPy is unavailable in the project environment.") from exc

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ReconnaissanceError(f"OpenCV could not open: {source}")

    temporary: Path | None = None
    try:
        metadata = _capture_metadata(capture, cv2, source)
        expected = {field: phase1["video"].get(field) for field in (
            "width", "height", "fps", "frame_count", "duration_seconds", "container"
        )}
        expected["codec"] = phase1["video"].get("codec_description")
        metadata_check = build_metadata_validation(metadata, expected)
        if metadata_check["discrepancies"]:
            raise ReconnaissanceError(
                "Video differs from Phase 1: " + "; ".join(metadata_check["discrepancies"])
            )

        sampling = config["sampling"]
        indices = interval_frame_indices(
            int(metadata["frame_count"]), float(metadata["fps"]),
            float(sampling["interval_seconds"]), float(sampling["edge_margin_seconds"]),
        )
        valid_mask, overlay_mask = _region_masks(
            int(metadata["width"]), int(metadata["height"]), config, cv2, np
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".reconnaissance-", dir=output.parent))
        thumbnails_dir = temporary / "thumbnails"
        sheets_dir = temporary / "contact_sheets"
        thumbnails_dir.mkdir()
        sheets_dir.mkdir()
        if not cv2.imwrite(str(temporary / "valid_region_mask.png"), valid_mask):
            raise ReconnaissanceError("Could not write valid-region mask.")
        if not cv2.imwrite(str(temporary / "overlay_exclusion_mask.png"), overlay_mask):
            raise ReconnaissanceError("Could not write overlay mask.")

        thumb_width = int(config["thumbnail"]["width"])
        thumb_height = int(config["thumbnail"]["height"])
        thumb_quality = int(config["thumbnail"]["jpeg_quality"])
        columns = int(config["contact_sheet"]["columns"])
        rows = int(config["contact_sheet"]["rows"])
        sheet_quality = int(config["contact_sheet"]["jpeg_quality"])
        page_size = columns * rows
        candidates: list[dict[str, Any]] = []
        sheet_paths: list[str] = []
        page_entries: list[tuple[Any, str]] = []
        previous_gray = None
        successful_reads = 0

        for number, frame_index in enumerate(indices, start=1):
            seconds = frame_index / float(metadata["fps"])
            timestamp = format_timestamp(seconds)
            warnings: list[str] = []
            metrics = dict.fromkeys(QUALITY_FIELDS)
            source_frame_index = None
            thumbnail_path = None

            seek_ok = capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            read_ok, frame = capture.read() if seek_ok else (False, None)
            if not seek_ok:
                warnings.append(f"Could not seek to frame {frame_index}.")
            elif not read_ok or frame is None or frame.size == 0:
                warnings.append(f"Could not read frame {frame_index}.")

            if read_ok and frame is not None and frame.size > 0:
                if frame.shape[1] != metadata["width"] or frame.shape[0] != metadata["height"]:
                    raise ReconnaissanceError(f"Unexpected size at frame {frame_index}.")
                reported = capture.get(cv2.CAP_PROP_POS_FRAMES)
                source_frame_index = frame_index
                if math.isfinite(reported) and reported >= 1:
                    actual = round(reported) - 1
                    if abs(actual - frame_index) <= 1:
                        source_frame_index = actual
                    else:
                        warnings.append(f"Decoder reported frame {actual} after seeking to {frame_index}.")
                metrics, previous_gray = _quality_metrics(frame, valid_mask, previous_gray, cv2)
                thumbnail = cv2.resize(
                    frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA
                )
                filename = (
                    f"recon_{number:04d}_t{_safe_timestamp(timestamp)}_"
                    f"f{frame_index:08d}.jpg"
                )
                _write_image(thumbnails_dir / filename, thumbnail, thumb_quality, cv2)
                thumbnail_path = _project_relative(output / "thumbnails" / filename)
                successful_reads += 1
            else:
                thumbnail = np.full((thumb_height, thumb_width, 3), 225, np.uint8)
                cv2.putText(
                    thumbnail, "FRAME READ FAILED", (20, thumb_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 180), 2, cv2.LINE_AA,
                )

            sheet_number = ((number - 1) // page_size) + 1
            sheet_name = f"contact_sheet_{sheet_number:03d}.jpg"
            sheet_path = _project_relative(output / "contact_sheets" / sheet_name)
            page_entries.append(
                (thumbnail, f"#{number:04d}  {timestamp}  f{frame_index:08d}")
            )
            candidates.append({
                "candidate_number": number,
                "requested_frame_index": frame_index,
                "source_frame_index": source_frame_index,
                "timestamp_ms": round(seconds * 1000),
                "timestamp_seconds": round(seconds, 6),
                "timestamp_human": timestamp,
                **metrics,
                "thumbnail_path": thumbnail_path,
                "contact_sheet_number": sheet_number,
                "contact_sheet_path": sheet_path,
                "read_warnings": warnings,
                "review_status": "unreviewed",
            })

            if len(page_entries) == page_size or number == len(indices):
                _write_image(
                    sheets_dir / sheet_name,
                    _sheet(page_entries, config, cv2, np),
                    sheet_quality,
                    cv2,
                )
                sheet_paths.append(sheet_path)
                page_entries = []
            if number % 100 == 0 or number == len(indices):
                LOGGER.info("Processed %d/%d candidates", number, len(indices))

        if successful_reads == 0:
            raise ReconnaissanceError("No reconnaissance frames could be decoded.")
        after = source.stat()
        unchanged = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        if not unchanged:
            raise ReconnaissanceError("Raw video changed during reconnaissance.")

        pixel_count = int(metadata["width"]) * int(metadata["height"])
        valid_pixels = int(cv2.countNonZero(valid_mask))
        overlay_pixels = int(cv2.countNonZero(overlay_mask))
        step_frames = round(float(sampling["interval_seconds"]) * float(metadata["fps"]))
        manifest = {
            "schema_version": "1.0",
            "phase": "Phase 1.5 - dataset reconnaissance",
            "scope": "Reconnaissance only; this is not a final dataset.",
            "source": {
                "path": _project_relative(source),
                "size_bytes": before.st_size,
                "sha256_from_phase1_report": phase1["source"]["sha256"],
                "phase1_inspection_report": _project_relative(phase1_path),
                "size_and_mtime_unchanged_during_run": unchanged,
                "opened_read_only_by_utility": True,
            },
            "video": metadata,
            "phase1_metadata_validation": metadata_check,
            "configuration": {"path": _project_relative(config_path), "resolved": config},
            "sampling": {
                "interval_seconds_requested": float(sampling["interval_seconds"]),
                "interval_frames": step_frames,
                "actual_interval_seconds": round(step_frames / float(metadata["fps"]), 6),
                "candidate_count": len(candidates),
                "successful_thumbnail_count": successful_reads,
                "read_failure_count": len(candidates) - successful_reads,
                "automatic_rejection_applied": False,
            },
            "image_storage": {
                "metrics_computed_on": f"original decoded {metadata['resolution']} frame",
                "stored_thumbnail_size": [thumb_width, thumb_height],
                "thumbnail_directory": _project_relative(output / "thumbnails"),
                "full_resolution_frames_saved": False,
            },
            "region_masks": {
                "coordinate_space": "normalized_xy_or_xyxy",
                "valid_region_name": config["valid_region"]["name"],
                "valid_region_mask_path": _project_relative(output / "valid_region_mask.png"),
                "overlay_exclusion_mask_path": _project_relative(output / "overlay_exclusion_mask.png"),
                "valid_pixel_count": valid_pixels,
                "valid_pixel_fraction": round(valid_pixels / pixel_count, 6),
                "overlay_pixel_count": overlay_pixels,
                "overlay_pixel_fraction": round(overlay_pixels / pixel_count, 6),
                "hidden_pixels_reconstructed": False,
                "provisional_reconnaissance_region": True,
            },
            "quality_metrics": {
                "brightness": "mean grayscale intensity on a 0-255 scale",
                "sharpness": "variance of grayscale Laplacian",
                "adjacent_difference": (
                    "mean absolute valid-region grayscale difference from the "
                    "previous successfully decoded candidate"
                ),
                "quality_rejection_thresholds": None,
            },
            "quality_summary": summarize_reconnaissance_quality(candidates),
            "contact_sheets": {
                "directory": _project_relative(output / "contact_sheets"),
                "columns": columns,
                "rows": rows,
                "candidates_per_sheet": page_size,
                "expected_count": expected_contact_sheet_count(len(indices), columns, rows),
                "created_count": len(sheet_paths),
                "paths": sheet_paths,
            },
            "candidates": candidates,
            "warnings": [warning for item in candidates for warning in item["read_warnings"]],
            "runtime": {
                "python_version": sys.version.split()[0],
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
            },
        }
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (ReconnaissanceError, VideoInspectionError):
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    except Exception as exc:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise ReconnaissanceError(f"Reconnaissance failed: {exc}") from exc
    finally:
        capture.release()

    if temporary is None:
        raise ReconnaissanceError("Temporary output was not created.")
    try:
        if output.exists():
            output.rmdir()
        os.replace(temporary, output)
    except OSError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ReconnaissanceError(f"Could not commit output: {output}") from exc

    manifest_path = output / MANIFEST_FILENAME
    LOGGER.info("Reconnaissance completed: %s", _project_relative(manifest_path))
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--inspection-report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_reconnaissance(
            args.input, args.inspection_report, args.config, args.output_dir
        )
    except (ReconnaissanceError, VideoInspectionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
