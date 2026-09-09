"""Streaming frozen-model video demo: shared annotations for encoding and preview."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.demo.mentor_demo import (  # noqa: E402
    CLASS_NAMES, FROZEN_MODEL_RELATIVE_PATH, FROZEN_MODEL_SHA256,
    FROZEN_MODEL_SIZE_BYTES, Detection, MentorDemoError, _annotate_image,
    _detections_from_result, _validate_model_class_names, sha256_file, verify_model_identity,
)


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/demo/video_demo.yaml")
OUTPUT_PATH = Path("outputs/demo/video")
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".webm", ".m4v"}
CSV_FIELDS = ("frame_index", "timestamp_seconds", "class_id", "class_name", "confidence",
              "x1", "y1", "x2", "y2")
PROVISIONAL_NOTICE = (
    "Provisional demo operating point. Final confidence/NMS thresholds will be "
    "selected through validation-only threshold optimization."
)
SCIENTIFIC_NOTICE = (
    "Raw frame detections are not unique physical damage events or ground-truth accuracy. "
    "The teacher video has no reliably confirmed V1-positive D00/D10/D20/D40 events; "
    "use it only as an application/domain/false-positive demonstration, not labelled "
    "positive evaluation or target-domain recall evidence."
)
WINDOW_NAME = "Road Damage Video Demo - q/Esc to stop (provisional thresholds)"


class VideoDemoError(RuntimeError):
    """Actionable video input, inference, or output failure."""


@dataclass(frozen=True)
class VideoConfig:
    model_path: Path
    confidence: float
    nms_iou: float
    max_detections: int
    codecs: tuple[tuple[str, str], ...]
    progress_every: int
    display_max_width: int
    display_max_height: int


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int | None


@dataclass(frozen=True)
class OutputPaths:
    run_dir: Path
    stem: str
    csv: Path
    summary: Path


def load_video_config(root: Path = ROOT) -> VideoConfig:
    """Keep thresholds in YAML while freezing model, device, image size and taxonomy."""
    try:
        raw = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
        model, inference = raw["model"], raw["inference"]
        if raw["schema_version"] != "video_demo.v1" or model != {
            "path": FROZEN_MODEL_RELATIVE_PATH.as_posix(),
            "sha256": FROZEN_MODEL_SHA256, "size_bytes": FROZEN_MODEL_SIZE_BYTES,
        }:
            raise VideoDemoError("Video demo requires the exact frozen best.pt identity.")
        if raw["classes"] != {str(k): v for k, v in CLASS_NAMES.items()}:
            raise VideoDemoError("Video demo class mapping must remain exactly D00/D10/D20/D40.")
        if (inference["imgsz"] != 640 or inference["device"] != 0
                or inference["augment"] is not False):
            raise VideoDemoError("Video demo requires imgsz=640, device=0, augment=False.")
        if raw["output_path"] != OUTPUT_PATH.as_posix() or raw["operating_point_status"] != "provisional_demo":
            raise VideoDemoError("Video output path or provisional operating-point declaration changed.")
        conf, nms = float(inference["confidence"]), float(inference["nms_iou"])
        if not (math.isfinite(conf) and 0 < conf <= 1 and math.isfinite(nms) and 0 < nms <= 1):
            raise VideoDemoError("Confidence and NMS IoU must be finite numbers in (0,1].")
        numeric_keys = [inference["max_detections"], raw["progress_every_frames"],
                        raw["display_max_width"], raw["display_max_height"]]
        if any(type(value) is not int or value <= 0 for value in numeric_keys):
            raise VideoDemoError("Detection limit, progress interval, and preview dimensions must be positive integers.")
        codecs = tuple((item["fourcc"], item["suffix"]) for item in raw["codec_candidates"])
        if codecs != (("mp4v", ".mp4"), ("MJPG", ".avi")):
            raise VideoDemoError("Codec candidates must be mp4v/MP4 then MJPG/AVI.")
        return VideoConfig(Path(model["path"]), conf, nms, numeric_keys[0], codecs,
                           numeric_keys[1], numeric_keys[2], numeric_keys[3])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise VideoDemoError(f"Cannot read video demo configuration: {root / CONFIG_PATH}") from exc


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def reject_redirects(path: Path) -> None:
    """Check ancestors before descendants, avoiding entry into symlink/junction targets."""
    for part in reversed([path, *path.parents]):
        if part.is_symlink() or part.is_junction():
            raise VideoDemoError(f"Redirected source/output/model paths are unsupported: {part}")


def validate_video_source(source: Path, root: Path = ROOT) -> Path:
    """Reject protected datasets lexically, before inspecting the supplied video path."""
    path = _absolute(source)
    parts = {part.casefold() for part in path.parts}
    if parts.intersection({"test", "internal_test", "internal-test", "test_images", "testimages"}):
        raise VideoDemoError("Internal/official test content is forbidden as a video source.")
    # Public raw/export/canonical datasets never serve video application inputs.
    for subtree in ("data/raw", "data/processed", "data/exports"):
        if path.is_relative_to(_absolute(root / subtree)):
            raise VideoDemoError("Dataset directories are forbidden as video application inputs.")
    if path.suffix.casefold() not in VIDEO_SUFFIXES:
        raise VideoDemoError(f"Unsupported video suffix: {path.suffix or '(none)'}")
    reject_redirects(path)
    if not path.is_file():
        raise VideoDemoError(f"Source video does not exist or is not a file: {path}")
    return path


def verify_video_model(config: VideoConfig, root: Path = ROOT) -> tuple[Path, str]:
    """Reuse the image demo's file-size and SHA verifier; never accept last.pt."""
    required = _absolute(root / FROZEN_MODEL_RELATIVE_PATH)
    candidate = _absolute(root / config.model_path)
    if candidate != required:
        raise VideoDemoError("Only this baseline run's frozen best.pt is permitted.")
    reject_redirects(candidate)
    return candidate, verify_model_identity(candidate, expected_sha256=FROZEN_MODEL_SHA256,
                                            expected_size_bytes=FROZEN_MODEL_SIZE_BYTES)


def read_video_info(capture: Any, cv2: Any) -> VideoInfo:
    """Read valid nominal FPS/dimensions; actual frame availability is checked by decoding."""
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width, height = (float(capture.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0:
        raise VideoDemoError("Video FPS must be finite and positive.")
    if any(not math.isfinite(v) or v <= 0 or not v.is_integer() for v in (width, height)):
        raise VideoDemoError("Video dimensions must be positive integers.")
    if int(width) % 2 or int(height) % 2:
        raise VideoDemoError("Odd frame dimensions are unsupported by these codecs without cropping; source preservation required.")
    count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoInfo(int(width), int(height), fps,
                     int(round(count)) if math.isfinite(count) and count > 0 else None)


def timestamp_ms(frame_index: int, fps: float) -> int:
    """Zero-based frame-index/nominal-FPS fallback, rounded to integer milliseconds."""
    if type(frame_index) is not int or frame_index < 0 or not math.isfinite(fps) or fps <= 0:
        raise VideoDemoError("Timestamp requires a non-negative frame index and valid FPS.")
    return round(frame_index * 1000 / fps)


def detection_rows(frame_index: int, fps: float, detections: Sequence[Detection]) -> list[dict[str, Any]]:
    seconds = f"{timestamp_ms(frame_index, fps) / 1000:.3f}"
    return [{"frame_index": frame_index, "timestamp_seconds": seconds,
             "class_id": item.class_id, "class_name": item.class_name,
             "confidence": format(item.confidence, ".9g"),
             **dict(zip(("x1", "y1", "x2", "y2"), item.bbox_xyxy_pixel, strict=True))}
            for item in detections]


def reserve_outputs(source: Path, root: Path = ROOT) -> OutputPaths:
    """Atomically reserve a unique run folder, with no source or previous-output overwrite."""
    base = _absolute(root / OUTPUT_PATH)
    reject_redirects(base)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:100] or "video"
    run_dir = base / f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if _absolute(source).is_relative_to(run_dir):
        raise VideoDemoError("Output would overlap the source.")
    run_dir.mkdir(parents=True, exist_ok=False)
    return OutputPaths(run_dir, stem, run_dir / f"{stem}_detections.csv", run_dir / f"{stem}_summary.json")


def open_writer(paths: OutputPaths, info: VideoInfo, config: VideoConfig, cv2: Any) -> tuple[Any, Path, str]:
    """Prefer MP4/mp4v; release failed candidates and fall back to AVI/MJPG."""
    failures = []
    for codec, suffix in config.codecs:
        path = paths.run_dir / f"{paths.stem}_detected{suffix}"
        if path.exists():
            raise VideoDemoError(f"Refusing to overwrite an output file: {path}")
        writer = None
        try:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), info.fps, (info.width, info.height))
            if writer.isOpened():
                return writer, path, codec
            failures.append(f"{codec} could not open")
        except Exception as exc:
            failures.append(f"{codec}: {exc}")
        except BaseException:
            if writer is not None:
                writer.release()
            raise
        if writer is not None:
            writer.release()
        LOGGER.warning("Codec %s unavailable; attempting fallback. Failed probe file may remain in %s", codec, paths.run_dir)
    raise VideoDemoError("No video writer available: " + "; ".join(failures))


def preview_frame(annotated: Any, config: VideoConfig, cv2: Any) -> Any:
    """Resize only the preview; the writer receives the original-resolution annotation."""
    height, width = annotated.shape[:2]
    scale = min(1., config.display_max_width / width, config.display_max_height / height)
    if scale == 1:
        return annotated
    return cv2.resize(annotated, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def check_cuda_device() -> None:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise VideoDemoError("CUDA device 0 is unavailable in the current environment.")


def provenance(root: Path, config: VideoConfig) -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in ("ultralytics", "torch", "torchvision", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    git = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        try:
            result = subprocess.run(["git", "--no-optional-locks", *args], cwd=root, capture_output=True, text=True)
            git[key] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        except OSError:
            git[key] = "unavailable"
    files = (root / CONFIG_PATH, root / "src/road_damage/demo/video_demo.py",
             root / "src/road_damage/demo/mentor_demo.py")
    return {"framework_versions": versions, "git": git,
            "implementation_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in files if p.is_file()},
            "resolved_config": {**asdict(config), "model_path": config.model_path.as_posix()}}


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_encoded_video(path: Path, info: VideoInfo, frames: int, cv2: Any) -> dict[str, Any]:
    """Reopen the finished output and check first-frame decode, dimensions/FPS/frame count."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoDemoError("Encoded output cannot be reopened.")
        actual = read_video_info(capture, cv2)
        ok, first = capture.read()
        if not ok or first is None or first.shape[:2] != (info.height, info.width):
            raise VideoDemoError("Encoded video dimensions or first-frame decode failed.")
        if (actual.width, actual.height) != (info.width, info.height) or not math.isclose(actual.fps, info.fps, rel_tol=1e-4, abs_tol=1e-3):
            raise VideoDemoError("Encoded output did not preserve source FPS/resolution.")
        if actual.frame_count != frames:
            raise VideoDemoError(f"Encoder frame count mismatch: wrote {frames}, encoded {actual.frame_count}.")
        return {"passed": True, "first_frame_decodes": True, "encoded_frames": actual.frame_count,
                "fps": actual.fps, "width": actual.width, "height": actual.height}
    finally:
        capture.release()


def run_video(
    source: Path, *, root: Path = ROOT, display: bool = False, max_frames: int | None = None,
    stop_event: threading.Event | None = None, yolo_factory: Callable[..., Any] | None = None,
    cv2_module: Any = None,
) -> dict[str, Any]:
    """One model, one sequential decode/predict/render/write loop, and a bounded working set.

    Injectable factory/backend permit synthetic tests. The CLI always loads the
    frozen model with the installed backend and provides no model override.
    """
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise VideoDemoError("--max-frames must be a positive integer.")
    config = load_video_config(root)
    source = validate_video_source(source, root)
    if cv2_module is None:
        import cv2 as cv2_module
    cv2 = cv2_module
    stop = stop_event if stop_event is not None else threading.Event()
    capture = writer = csv_handle = None
    paths = video_path = None
    processed = frames_with_detections = total_detections = 0
    counts = {name: 0 for name in CLASS_NAMES.values()}
    loop_started = None
    status, reason = "failed", "initialization_failed"
    failure: BaseException | None = None
    cleanup_errors = []
    summary: dict[str, Any] = {}
    display_active = display
    display_worked = display_failed = False
    loop_seconds = 0.
    LOGGER.info(PROVISIONAL_NOTICE)
    LOGGER.info(SCIENTIFIC_NOTICE)
    try:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoDemoError(f"Cannot open video: {source}")
        info = read_video_info(capture, cv2)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoDemoError("Video has no decodable frames.")
        if frame.shape[:2] != (info.height, info.width):
            raise VideoDemoError("Decoded frame dimensions differ from source metadata.")
        if stop.is_set():
            raise KeyboardInterrupt
        model_path, model_sha = verify_video_model(config, root)
        LOGGER.info("Frozen best.pt verified: %s", model_sha)
        check_cuda_device()
        LOGGER.info("Hashing source video for provenance: %s", source)
        source_before = source.stat()
        source_sha = sha256_file(source)
        paths = reserve_outputs(source, root)
        summary = {"schema_version": "video_demo.summary.v1", "run_id": paths.run_dir.name,
                   "started_utc": datetime.now(timezone.utc).isoformat(), "source_path": str(source),
                   "source_sha256": source_sha, "source_size_bytes": source_before.st_size,
                   "model_path": str(model_path), "model_sha256": model_sha,
                   "model_size_bytes": FROZEN_MODEL_SIZE_BYTES,
                   "input_resolution": {"width": info.width, "height": info.height},
                   "input_fps": info.fps, "frame_count": info.frame_count,
                   "duration_seconds": info.frame_count / info.fps if info.frame_count is not None else None,
                   "timestamp_policy": "zero-based frame_index/nominal FPS, rounded to integer milliseconds",
                   "max_frames": max_frames, "display_requested": display,
                   "demo_confidence": config.confidence, "demo_nms_iou": config.nms_iou,
                   "imgsz": 640, "device": 0, "augment": False,
                   "class_mapping": CLASS_NAMES, "operating_point_status": "provisional_demo",
                   "provisional_threshold_disclaimer": PROVISIONAL_NOTICE,
                   "scientific_limitations": SCIENTIFIC_NOTICE,
                   "audio_preserved": False, "csv_path": str(paths.csv),
                   **provenance(root, config)}
        write_summary(paths.summary, {**summary, "status": "running", "partial_output": True})
        writer, video_path, codec = open_writer(paths, info, config, cv2)
        summary.update({"output_path": str(video_path), "codec": codec,
                        "container": video_path.suffix.lstrip(".")})
        csv_handle = paths.csv.open("x", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        csv_writer.writeheader()
        if stop.is_set():
            raise KeyboardInterrupt
        if yolo_factory is None:
            from ultralytics import YOLO
            yolo_factory = YOLO
        model = yolo_factory(str(model_path), task="detect")
        _validate_model_class_names(model.names)
        loop_started = time.perf_counter()
        target = min(info.frame_count, max_frames) if info.frame_count and max_frames else (max_frames or info.frame_count)
        while True:
            if stop.is_set():
                status, reason = "interrupted", "ctrl_c"
                break
            if frame.shape[:2] != (info.height, info.width):
                raise VideoDemoError("Source resolution changed mid-video; encoding stopped.")
            prediction = model.predict(
                source=frame, imgsz=640, device=0, conf=config.confidence, iou=config.nms_iou,
                augment=False, max_det=config.max_detections, save=False, save_txt=False,
                save_crop=False, show=False, verbose=False, stream=False,
            )
            if len(prediction) != 1:
                raise VideoDemoError("Expected one model result for one source frame.")
            detections = _detections_from_result(prediction[0], info.width, info.height)
            annotated = _annotate_image(frame, detections, cv2)
            # Both consumers use this exact annotation; no second model call.
            writer.write(annotated)
            csv_writer.writerows(detection_rows(processed, info.fps, detections))
            for detection in detections:
                counts[detection.class_name] += 1
            total_detections += len(detections)
            frames_with_detections += bool(detections)
            processed += 1
            elapsed = time.perf_counter() - loop_started
            if processed == 1 or processed % config.progress_every == 0:
                csv_handle.flush()
                rate = processed / elapsed if elapsed else 0.
                eta = max(0, target - processed) / rate if target and rate else None
                LOGGER.info("Processing frame %d/%s; elapsed %.1fs; processing FPS %.2f; source FPS %.3f; ETA %s",
                            processed, target or "unknown", elapsed, rate, info.fps,
                            f"{eta:.1f}s" if eta is not None else "unknown")
            if display_active:
                try:
                    cv2.imshow(WINDOW_NAME, preview_frame(annotated, config, cv2))
                    display_worked = True
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        status, reason = "interrupted", "preview_quit"
                        break
                except (cv2.error, NotImplementedError) as exc:
                    display_active = False
                    display_failed = True
                    LOGGER.warning("Live preview unavailable; continuing to save annotated video: %s", exc)
            if stop.is_set():
                status, reason = "interrupted", "ctrl_c"
                break
            if max_frames is not None and processed >= max_frames:
                status, reason = "limited", "max_frames"
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                if info.frame_count is not None and processed < info.frame_count:
                    status, reason = "partial", "decode_ended_before_reported_frame_count"
                    LOGGER.warning("Decode ended after %d of %d reported frames.", processed, info.frame_count)
                else:
                    status, reason = "completed", "end_of_video"
                break
    except KeyboardInterrupt:
        status, reason = "interrupted", "ctrl_c"
    except Exception as exc:
        failure = exc
        status, reason = "failed", "processing_error"
    finally:
        if loop_started is not None:
            loop_seconds = time.perf_counter() - loop_started
        # Independent releases ensure one backend failure cannot skip other resources.
        for label, resource, method in (("VideoCapture", capture, "release"),
                                         ("VideoWriter", writer, "release"),
                                         ("detection CSV", csv_handle, "close")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as exc:
                    cleanup_errors.append(f"{label}: {exc}")
        if display:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                LOGGER.warning("GUI cleanup unavailable: %s", exc)
    if paths is not None:
        if cleanup_errors:
            failure = failure or VideoDemoError("Resource finalization failed: " + "; ".join(cleanup_errors))
            status, reason = "failed", "resource_finalization_error"
        encoding = {"passed": False, "reason": "no encoded frames"}
        if video_path is not None and processed:
            try:
                encoding = verify_encoded_video(video_path, info, processed, cv2)
            except Exception as exc:
                encoding = {"passed": False, "error": str(exc)}
                failure = failure or exc
                status, reason = "failed", "output_verification_error"
        try:
            source_after = source.stat()
            unchanged = (source_before.st_size, source_before.st_mtime_ns) == (source_after.st_size, source_after.st_mtime_ns)
        except OSError:
            unchanged = False
        if not unchanged:
            status, reason = "failed", "source_changed_during_processing"
            failure = failure or VideoDemoError("Source changed externally during processing.")
        try:
            _, model_after_sha = verify_video_model(config, root)
            model_unchanged = model_after_sha == model_sha
        except (VideoDemoError, MentorDemoError, OSError) as exc:
            model_unchanged = False
            failure = failure or exc
            status, reason = "failed", "model_integrity_error"
        summary.update({"status": status, "stop_reason": reason, "partial_output": status != "completed",
                        "processed_frames": processed, "output_duration_seconds": processed / info.fps,
                        "counts_by_class": counts, "total_detections": total_detections,
                        "frames_with_detections": frames_with_detections,
                        "processing_duration_seconds": loop_seconds,
                        "average_processing_fps": processed / loop_seconds if loop_seconds else 0.,
                        "processing_timing_scope": "frame loop: inference/warmup, annotation, encode submission, CSV, preview, sequential decode; excludes setup/hash/finalization",
                        "source_size_and_mtime_unchanged": unchanged,
                        "model_sha256_verified_after_processing": model_unchanged,
                        "display_worked": display_worked, "display_failed": display_failed,
                        "encoding_verification": encoding, "cleanup_errors": cleanup_errors,
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                        "error": str(failure) if failure else None})
        write_summary(paths.summary, summary)
        LOGGER.info("Video demo %s (%s): %d frames, %d raw detections; %.2f processing FPS.",
                    status, reason, processed, total_detections, summary["average_processing_fps"])
        LOGGER.info("Output: %s; CSV: %s; summary: %s", video_path, paths.csv, paths.summary)
        if status != "completed":
            LOGGER.warning("Partial output: only %d source frames were processed; see summary status.", processed)
    if failure is not None:
        raise VideoDemoError(f"Video demo failed: {failure}; summary: {paths.summary if paths else 'not created'}") from failure
    if not summary:
        return {"status": status, "stop_reason": reason, "partial_output": True, "processed_frames": 0,
                "output_path": None}
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Local video file; source is never overwritten.")
    parser.add_argument("--display", action="store_true", help="Preview the same annotated frames being saved; q/Esc stops.")
    parser.add_argument("--max-frames", type=int, help="Process only the first N frames; mark output as limited.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stop = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum: int, frame: Any) -> None:
        # Finish a frame's writer/CSV transaction instead of interrupting halfway.
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    try:
        summary = run_video(args.source, display=args.display, max_frames=args.max_frames, stop_event=stop)
        return 130 if summary["stop_reason"] == "ctrl_c" else (2 if summary["status"] == "partial" else 0)
    except (VideoDemoError, MentorDemoError, OSError, ValueError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted before video processing started.")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
