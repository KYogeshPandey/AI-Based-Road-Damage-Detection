"""Inspect a source video and save a small, deterministic contact sample."""

from __future__ import annotations

import argparse
import hashlib
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


LOGGER = logging.getLogger("road_damage.video.inspect_video")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_FILENAME = "video_report.json"
REPORT_SCHEMA_VERSION = "1.0"


class VideoInspectionError(RuntimeError):
    """Raised when a video cannot be inspected safely or completely."""


def format_timestamp(timestamp_seconds: float) -> str:
    """Format non-negative seconds as ``HH:MM:SS.mmm``."""
    if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
        raise ValueError("timestamp_seconds must be a finite, non-negative value")

    total_ms = int(round(timestamp_seconds * 1000.0))
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    seconds, milliseconds = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def distributed_frame_indices(
    frame_count: int,
    fps: float,
    sample_count: int,
) -> list[int]:
    """Return evenly distributed zero-based frame indices away from boundaries."""
    if frame_count < 3:
        raise VideoInspectionError("At least three frames are required for sampling.")
    if not math.isfinite(fps) or fps <= 0:
        raise VideoInspectionError(f"Invalid FPS for sampling: {fps!r}")
    if sample_count <= 0:
        raise VideoInspectionError("Sample count must be greater than zero.")
    if sample_count > frame_count - 2:
        raise VideoInspectionError(
            f"Cannot extract {sample_count} unique non-boundary samples from "
            f"a {frame_count}-frame video."
        )

    last_frame_index = frame_count - 1
    one_second_margin = max(1, int(round(fps)))
    margin = min(one_second_margin, (last_frame_index - 1) // 2)
    start_frame = margin
    end_frame = last_frame_index - margin

    if end_frame - start_frame + 1 < sample_count:
        start_frame = 1
        end_frame = last_frame_index - 1

    if sample_count == 1:
        return [(start_frame + end_frame) // 2]

    interval_count = sample_count - 1
    span = end_frame - start_frame
    indices = [
        start_frame + (span * index + interval_count // 2) // interval_count
        for index in range(sample_count)
    ]
    if len(set(indices)) != sample_count:
        raise VideoInspectionError("Could not calculate unique sample frame indices.")
    return indices


def summarize_quality(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize brightness and sharpness values across sampled frames."""
    if not samples:
        return {"sample_count": 0}

    def summary(metric: str) -> dict[str, float]:
        values = [float(sample[metric]) for sample in samples]
        return {
            "minimum": round(min(values), 4),
            "maximum": round(max(values), 4),
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
        }

    return {
        "sample_count": len(samples),
        "brightness_mean_intensity": summary("brightness_mean_intensity"),
        "sharpness_laplacian_variance": summary(
            "sharpness_laplacian_variance"
        ),
    }


def build_metadata_validation(
    detected: dict[str, Any],
    expected: dict[str, Any],
    *,
    fps_tolerance: float = 0.01,
    duration_tolerance_seconds: float = 0.1,
) -> dict[str, Any]:
    """Compare detected metadata with caller-supplied reference values."""
    checks: dict[str, dict[str, Any]] = {}
    discrepancies: list[str] = []

    exact_fields = ("width", "height", "frame_count")
    for field in exact_fields:
        if expected.get(field) is None:
            continue
        detected_value = int(detected[field])
        expected_value = int(expected[field])
        difference = detected_value - expected_value
        status = "match" if difference == 0 else "discrepancy"
        checks[field] = {
            "detected": detected_value,
            "expected": expected_value,
            "difference": difference,
            "tolerance": 0,
            "status": status,
        }
        if status == "discrepancy":
            discrepancies.append(
                f"{field}: detected {detected_value}, expected {expected_value}"
            )

    approximate_fields = (
        ("fps", fps_tolerance),
        ("duration_seconds", duration_tolerance_seconds),
    )
    for field, tolerance in approximate_fields:
        if expected.get(field) is None:
            continue
        detected_value = float(detected[field])
        expected_value = float(expected[field])
        difference = detected_value - expected_value
        status = "match" if abs(difference) <= tolerance else "discrepancy"
        checks[field] = {
            "detected": round(detected_value, 6),
            "expected": round(expected_value, 6),
            "difference": round(difference, 6),
            "tolerance": tolerance,
            "status": status,
        }
        if status == "discrepancy":
            discrepancies.append(
                f"{field}: detected {detected_value:.6f}, "
                f"expected {expected_value:.6f} (tolerance {tolerance})"
            )

    if expected.get("container") is not None:
        detected_value = str(detected["container"])
        expected_value = str(expected["container"])
        status = (
            "match"
            if detected_value.casefold() == expected_value.casefold()
            else "discrepancy"
        )
        checks["container"] = {
            "detected": detected_value,
            "expected": expected_value,
            "status": status,
        }
        if status == "discrepancy":
            discrepancies.append(
                f"container: detected {detected_value}, expected {expected_value}"
            )

    if expected.get("codec") is not None:
        detected_value = str(detected["codec_description"])
        expected_value = str(expected["codec"])
        status = (
            "match"
            if detected_value.casefold() == expected_value.casefold()
            else "discrepancy"
        )
        checks["codec"] = {
            "detected": detected_value,
            "detected_fourcc": detected["codec_fourcc"],
            "expected": expected_value,
            "status": status,
        }
        if status == "discrepancy":
            discrepancies.append(
                f"codec: detected {detected_value} "
                f"({detected['codec_fourcc']}), expected {expected_value}"
            )

    if expected.get("pixel_format") is not None:
        checks["pixel_format"] = {
            "detected": None,
            "expected": str(expected["pixel_format"]),
            "status": "not_verifiable_with_opencv",
        }

    return {
        "checks": checks,
        "discrepancies": discrepancies,
        "all_verifiable_fields_match": not discrepancies,
    }


def _load_opencv() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise VideoInspectionError(
            "OpenCV is unavailable in this Python environment. Run the utility "
            "with the existing project interpreter: "
            ".\\.venv\\Scripts\\python.exe"
        ) from exc
    return cv2


def _resolve_from_project_root(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VideoInspectionError(
            f"Could not read the input video while calculating SHA-256: {path}"
        ) from exc
    return digest.hexdigest()


def _fourcc_text(value: float) -> str:
    integer_value = int(value)
    characters = [chr((integer_value >> (8 * index)) & 0xFF) for index in range(4)]
    return "".join(character for character in characters if character.isprintable()).strip()


def _codec_description(fourcc: str) -> str:
    mpeg4_part2_codes = {"DIVX", "DX50", "FMP4", "M4S2", "MP4V", "XVID"}
    return "MPEG-4 Part 2" if fourcc.upper() in mpeg4_part2_codes else "unknown"


def _capture_metadata(capture: Any, cv2: Any, input_path: Path) -> dict[str, Any]:
    width_value = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_value = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if not math.isfinite(width_value) or width_value <= 0:
        raise VideoInspectionError(f"Invalid video width reported: {width_value!r}")
    if not math.isfinite(height_value) or height_value <= 0:
        raise VideoInspectionError(f"Invalid video height reported: {height_value!r}")
    if not math.isfinite(fps) or fps <= 0:
        raise VideoInspectionError(f"Invalid video FPS reported: {fps!r}")
    if not math.isfinite(frame_count_value) or frame_count_value <= 0:
        raise VideoInspectionError(
            f"Invalid video frame count reported: {frame_count_value!r}"
        )

    width = int(round(width_value))
    height = int(round(height_value))
    frame_count = int(round(frame_count_value))
    duration_seconds = frame_count / fps
    fourcc = _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
    try:
        decoder_backend = capture.getBackendName()
    except (AttributeError, cv2.error):
        decoder_backend = "unknown"

    return {
        "container": input_path.suffix.lstrip(".").upper() or "unknown",
        "codec_fourcc": fourcc or "unknown",
        "codec_description": _codec_description(fourcc),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "aspect_ratio": round(width / height, 6),
        "orientation": "landscape" if width >= height else "portrait",
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 6),
        "duration_human": format_timestamp(duration_seconds),
        "duration_calculation": "frame_count / fps",
        "decoder_backend": decoder_backend,
    }


def _sample_filename(sample_number: int, frame_index: int, timestamp: str) -> str:
    safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
    return (
        f"sample_{sample_number:02d}_t{safe_timestamp}_"
        f"f{frame_index:08d}.jpg"
    )


def _extract_samples(
    capture: Any,
    cv2: Any,
    metadata: dict[str, Any],
    sample_count: int,
    temporary_output_dir: Path,
    final_output_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    frame_indices = distributed_frame_indices(
        int(metadata["frame_count"]), float(metadata["fps"]), sample_count
    )
    samples: list[dict[str, Any]] = []
    warnings: list[str] = []

    for sample_number, requested_frame_index in enumerate(frame_indices, start=1):
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, requested_frame_index):
            raise VideoInspectionError(
                f"Could not seek to frame {requested_frame_index} for sample "
                f"{sample_number}."
            )

        read_ok, frame_bgr = capture.read()
        if not read_ok or frame_bgr is None or frame_bgr.size == 0:
            raise VideoInspectionError(
                f"Could not decode frame {requested_frame_index} for sample "
                f"{sample_number}."
            )

        position_after_read = capture.get(cv2.CAP_PROP_POS_FRAMES)
        decoded_frame_index = requested_frame_index
        if math.isfinite(position_after_read) and position_after_read >= 1:
            reported_frame_index = int(round(position_after_read)) - 1
            if abs(reported_frame_index - requested_frame_index) > 1:
                warnings.append(
                    f"Sample {sample_number}: decoder reported frame "
                    f"{reported_frame_index} after seeking to "
                    f"{requested_frame_index}; the requested index is retained "
                    "for timestamp provenance."
                )
            else:
                decoded_frame_index = reported_frame_index

        frame_height, frame_width = frame_bgr.shape[:2]
        if frame_width != metadata["width"] or frame_height != metadata["height"]:
            warnings.append(
                f"Sample {sample_number}: decoded frame size was "
                f"{frame_width}x{frame_height}, while metadata reported "
                f"{metadata['resolution']}."
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        timestamp_seconds = requested_frame_index / float(metadata["fps"])
        timestamp_ms = int(round(timestamp_seconds * 1000.0))
        timestamp_human = format_timestamp(timestamp_seconds)
        filename = _sample_filename(
            sample_number, requested_frame_index, timestamp_human
        )
        temporary_path = temporary_output_dir / filename

        try:
            write_ok = cv2.imwrite(
                str(temporary_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
        except cv2.error as exc:
            raise VideoInspectionError(
                f"OpenCV failed while writing sample image: {temporary_path}"
            ) from exc
        if not write_ok or not temporary_path.is_file():
            raise VideoInspectionError(
                f"Could not write sample image: {temporary_path}"
            )

        samples.append(
            {
                "sample_number": sample_number,
                "requested_frame_index": requested_frame_index,
                "source_frame_index": decoded_frame_index,
                "frame_number_one_based": decoded_frame_index + 1,
                "timestamp_ms": timestamp_ms,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "timestamp_human": timestamp_human,
                "brightness_mean_intensity": round(brightness, 4),
                "sharpness_laplacian_variance": round(sharpness, 4),
                "saved_image_path": _project_relative(final_output_dir / filename),
            }
        )

    return samples, warnings


def inspect_video(
    input_path: Path,
    output_dir: Path,
    sample_count: int,
    expected_metadata: dict[str, Any],
) -> Path:
    """Inspect one video and return the generated JSON report path."""
    resolved_input = _resolve_from_project_root(input_path)
    resolved_output_dir = _resolve_from_project_root(output_dir)

    if not resolved_input.exists():
        raise VideoInspectionError(f"Input video does not exist: {resolved_input}")
    if not resolved_input.is_file():
        raise VideoInspectionError(f"Input video is not a file: {resolved_input}")
    if resolved_input == resolved_output_dir or resolved_input.parent == resolved_output_dir:
        raise VideoInspectionError(
            "The output directory must be separate from the raw video's directory."
        )

    cv2 = _load_opencv()

    if resolved_output_dir.exists():
        if not resolved_output_dir.is_dir():
            raise VideoInspectionError(
                f"Output path exists and is not a directory: {resolved_output_dir}"
            )
        try:
            next(resolved_output_dir.iterdir())
        except StopIteration:
            pass
        except OSError as exc:
            raise VideoInspectionError(
                f"Could not inspect output directory: {resolved_output_dir}"
            ) from exc
        else:
            raise VideoInspectionError(
                f"Output directory is not empty: {resolved_output_dir}. "
                "Move the existing inspection artifacts before running again."
            )

    source_stat_before = resolved_input.stat()
    source_sha256 = _sha256(resolved_input)
    capture = cv2.VideoCapture(str(resolved_input))
    if not capture.isOpened():
        capture.release()
        raise VideoInspectionError(
            f"OpenCV could not open the input video: {resolved_input}"
        )

    temporary_output_dir: Path | None = None
    try:
        try:
            metadata = _capture_metadata(capture, cv2, resolved_input)
            metadata_validation = build_metadata_validation(
                metadata, expected_metadata
            )

            try:
                resolved_output_dir.parent.mkdir(parents=True, exist_ok=True)
                temporary_output_dir = Path(
                    tempfile.mkdtemp(
                        prefix=".inspection-", dir=resolved_output_dir.parent
                    )
                )
            except OSError as exc:
                raise VideoInspectionError(
                    f"Could not create a temporary output directory under: "
                    f"{resolved_output_dir.parent}"
                ) from exc

            samples, sample_warnings = _extract_samples(
                capture,
                cv2,
                metadata,
                sample_count,
                temporary_output_dir,
                resolved_output_dir,
            )
        finally:
            capture.release()
    except Exception:
        if temporary_output_dir is not None:
            shutil.rmtree(temporary_output_dir, ignore_errors=True)
        raise

    source_stat_after = resolved_input.stat()
    source_unchanged = (
        source_stat_before.st_size == source_stat_after.st_size
        and source_stat_before.st_mtime_ns == source_stat_after.st_mtime_ns
    )
    if not source_unchanged:
        if temporary_output_dir is not None:
            shutil.rmtree(temporary_output_dir, ignore_errors=True)
        raise VideoInspectionError(
            "The input video's size or modification time changed during inspection; "
            "temporary outputs were discarded."
        )

    warnings = [
        *metadata_validation["discrepancies"],
        *sample_warnings,
        (
            "OpenCV does not independently report the encoded pixel format; "
            "the supplied reference pixel format is recorded but unverified."
            if expected_metadata.get("pixel_format")
            else ""
        ),
        (
            "Representative frames may contain faces, number plates, or location "
            "clues and must be treated as private project data."
        ),
    ]
    warnings = [warning for warning in warnings if warning]

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "inspection_phase": "Phase 1 - video inspection",
        "source": {
            "path": _project_relative(resolved_input),
            "filename": resolved_input.name,
            "size_bytes": source_stat_before.st_size,
            "sha256": source_sha256,
            "immutable_source_check": {
                "size_and_mtime_unchanged_during_run": source_unchanged,
                "opened_read_only_by_utility": True,
            },
        },
        "video": metadata,
        "reference_metadata": {
            key: value for key, value in expected_metadata.items() if value is not None
        },
        "metadata_validation": metadata_validation,
        "sampling": {
            "strategy": "evenly spaced from one second after start to one second before end",
            "requested_sample_count": sample_count,
            "extracted_sample_count": len(samples),
            "timestamp_source": "zero-based requested frame index / detected FPS",
            "image_format": "JPEG",
            "jpeg_quality": 95,
        },
        "quality_metrics": {
            "brightness": "mean grayscale intensity on a 0-255 scale",
            "sharpness": (
                "variance of the grayscale Laplacian; higher generally means "
                "sharper, but values are scene-dependent"
            ),
        },
        "quality_summary": summarize_quality(samples),
        "samples": samples,
        "warnings": warnings,
        "runtime": {
            "python_version": sys.version.split()[0],
            "opencv_version": cv2.__version__,
        },
    }

    if temporary_output_dir is None:
        raise VideoInspectionError("Internal error: temporary output directory was not created.")

    temporary_report_path = temporary_output_dir / REPORT_FILENAME
    try:
        temporary_report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if resolved_output_dir.exists():
            resolved_output_dir.rmdir()
        os.replace(temporary_output_dir, resolved_output_dir)
    except OSError as exc:
        shutil.rmtree(temporary_output_dir, ignore_errors=True)
        raise VideoInspectionError(
            f"Could not commit inspection outputs to: {resolved_output_dir}"
        ) from exc

    report_path = resolved_output_dir / REPORT_FILENAME
    LOGGER.info(
        "Video inspection completed",
        extra={
            "input_path": _project_relative(resolved_input),
            "report_path": _project_relative(report_path),
            "sample_count": len(samples),
        },
    )
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect video metadata and save deterministic representative frames "
            "with brightness and sharpness metrics."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Empty output directory"
    )
    parser.add_argument(
        "--samples", type=int, default=12, help="Number of representative frames"
    )
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--expected-frame-count", type=int)
    parser.add_argument("--expected-duration-seconds", type=float)
    parser.add_argument("--expected-container")
    parser.add_argument("--expected-codec")
    parser.add_argument("--expected-pixel-format")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    expected_metadata = {
        "width": args.expected_width,
        "height": args.expected_height,
        "fps": args.expected_fps,
        "frame_count": args.expected_frame_count,
        "duration_seconds": args.expected_duration_seconds,
        "container": args.expected_container,
        "codec": args.expected_codec,
        "pixel_format": args.expected_pixel_format,
    }

    try:
        report_path = inspect_video(
            args.input, args.output_dir, args.samples, expected_metadata
        )
    except (VideoInspectionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Report written to %s", _project_relative(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
