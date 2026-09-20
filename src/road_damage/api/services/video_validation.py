"""Lightweight upload validation with no model, checkpoint, or GPU access."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class StagedVideoValidationError(ValueError):
    """Raised when an uploaded file is not a conservatively decodable video."""


@dataclass(frozen=True, slots=True)
class StagedVideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int


def validate_staged_video(
    source: Path, *, cv2_module: Any | None = None
) -> StagedVideoMetadata:
    """Open an API-controlled path and decode one frame without model loading."""
    if cv2_module is None:
        try:
            import cv2 as cv2_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise StagedVideoValidationError("OpenCV video support is unavailable.") from exc

    capture = cv2_module.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise StagedVideoValidationError("Uploaded video could not be opened.")
        width = int(round(float(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT))))
        fps = float(capture.get(cv2_module.CAP_PROP_FPS))
        frame_count = int(round(float(capture.get(cv2_module.CAP_PROP_FRAME_COUNT))))
        if width <= 0 or height <= 0:
            raise StagedVideoValidationError("Uploaded video has invalid dimensions.")
        if not math.isfinite(fps) or fps <= 0:
            raise StagedVideoValidationError("Uploaded video has invalid frame rate metadata.")
        if frame_count <= 0:
            raise StagedVideoValidationError("Uploaded video has no readable frame count.")
        readable, frame = capture.read()
        if not readable or frame is None or getattr(frame, "size", 0) <= 0:
            raise StagedVideoValidationError("Uploaded video has no decodable frame.")
        return StagedVideoMetadata(width, height, fps, frame_count)
    finally:
        capture.release()
