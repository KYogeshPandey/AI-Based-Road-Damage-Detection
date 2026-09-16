"""Pure, deterministic temporal association for heuristic video damage events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


class AggregationError(ValueError):
    """A temporal-association input or invariant is invalid."""


@dataclass(frozen=True)
class AggregationConfig:
    """Versioned, interpretable association and confirmation thresholds."""

    minimum_iou: float
    maximum_center_distance_normalized: float
    minimum_area_ratio: float
    maximum_gap_seconds: float
    minimum_confirmed_observations: int

    def __post_init__(self) -> None:
        finite = (
            self.minimum_iou,
            self.maximum_center_distance_normalized,
            self.minimum_area_ratio,
            self.maximum_gap_seconds,
        )
        if not all(math.isfinite(value) for value in finite):
            raise AggregationError("Aggregation thresholds must be finite.")
        if not 0 < self.minimum_iou <= 1:
            raise AggregationError("minimum_iou must be in (0,1].")
        if not 0 < self.maximum_center_distance_normalized <= math.sqrt(2):
            raise AggregationError(
                "maximum_center_distance_normalized must be in (0,sqrt(2)]."
            )
        if not 0 < self.minimum_area_ratio <= 1:
            raise AggregationError("minimum_area_ratio must be in (0,1].")
        if self.maximum_gap_seconds < 0:
            raise AggregationError("maximum_gap_seconds must be non-negative.")
        if (
            type(self.minimum_confirmed_observations) is not int
            or self.minimum_confirmed_observations < 1
        ):
            raise AggregationError(
                "minimum_confirmed_observations must be a positive integer."
            )


@dataclass(frozen=True)
class DetectionObservation:
    """One direct detector observation in explicit pixel and normalized xyxy form."""

    detection_index: int
    frame_index: int
    timestamp_ms: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy_pixel: tuple[float, float, float, float]
    bbox_xyxy_normalized: tuple[float, float, float, float]
    bbox_area_pixels: float
    bbox_relative_area: float

    def __post_init__(self) -> None:
        if type(self.detection_index) is not int or self.detection_index < 0:
            raise AggregationError("detection_index must be a non-negative integer.")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise AggregationError("frame_index must be a non-negative integer.")
        if type(self.timestamp_ms) is not int or self.timestamp_ms < 0:
            raise AggregationError("timestamp_ms must be a non-negative integer.")
        if type(self.class_id) is not int or self.class_id not in range(4):
            raise AggregationError("class_id must be one of 0,1,2,3.")
        if not self.class_name:
            raise AggregationError("class_name must be non-empty.")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise AggregationError("confidence must be finite and in [0,1].")
        if len(self.bbox_xyxy_pixel) != 4 or not all(
            math.isfinite(value) for value in self.bbox_xyxy_pixel
        ):
            raise AggregationError("bbox_xyxy_pixel must contain four finite values.")
        x1, y1, x2, y2 = self.bbox_xyxy_pixel
        if min(x1, y1) < 0 or x2 < x1 or y2 < y1:
            raise AggregationError("bbox_xyxy_pixel is invalid.")
        if len(self.bbox_xyxy_normalized) != 4 or not all(
            math.isfinite(value) and 0 <= value <= 1
            for value in self.bbox_xyxy_normalized
        ):
            raise AggregationError(
                "bbox_xyxy_normalized must contain four values in [0,1]."
            )
        if (
            not math.isfinite(self.bbox_area_pixels)
            or self.bbox_area_pixels < 0
            or not math.isfinite(self.bbox_relative_area)
            or not 0 <= self.bbox_relative_area <= 1
        ):
            raise AggregationError("Bounding-box areas must be finite and non-negative.")

    def to_record(self, event_id: str, association: dict[str, Any]) -> dict[str, Any]:
        """Serialize one observation for the frame-level JSONL contract."""
        return {
            "detection_index": self.detection_index,
            "event_id": event_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy_pixel": list(self.bbox_xyxy_pixel),
            "bbox_xyxy_normalized": list(self.bbox_xyxy_normalized),
            "bbox_area_pixels": self.bbox_area_pixels,
            "bbox_relative_area": self.bbox_relative_area,
            "association": association,
        }


def bbox_iou_xyxy(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Continuous-coordinate IoU; degenerate boxes yield zero."""
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Euclidean center distance in the normalized image coordinate plane."""
    left_x, left_y = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    right_x, right_y = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    return math.hypot(left_x - right_x, left_y - right_y)


def bbox_area_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Smaller/larger normalized box-area ratio, or zero for degenerate boxes."""
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return min(left_area, right_area) / max(left_area, right_area) if max(left_area, right_area) else 0.0


@dataclass(frozen=True)
class AssociationDecision:
    """Auditable result linking exactly one detection to one event."""

    detection_index: int
    event_id: str
    association_type: str
    iou_with_previous: float | None
    center_distance_normalized: float | None
    area_ratio: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "type": self.association_type,
            "iou_with_previous": self.iou_with_previous,
            "center_distance_normalized": self.center_distance_normalized,
            "area_ratio": self.area_ratio,
        }


@dataclass
class _EventState:
    sequence: int
    event_id: str
    class_id: int
    class_name: str
    first_frame: int
    first_timestamp_ms: int
    last_frame: int
    last_timestamp_ms: int
    observation_count: int
    confidence_sum: float
    max_confidence: float
    relative_area_sum: float
    max_relative_area: float
    max_bbox_area_pixels: float
    last_bbox_xyxy_pixel: tuple[float, float, float, float]
    last_bbox_xyxy_normalized: tuple[float, float, float, float]
    representative: DetectionObservation
    matched_observations: int = 0
    iou_rule_matches: int = 0
    center_scale_rule_matches: int = 0

    # Long-video memory invariant: one active event retains only scalar aggregates,
    # its last box, and one representative observation. No append-only detection
    # or association history is retained, and finalization replaces this state with
    # one compact immutable summary.
    @classmethod
    def from_observation(
        cls, sequence: int, event_id: str, observation: DetectionObservation
    ) -> _EventState:
        return cls(
            sequence=sequence,
            event_id=event_id,
            class_id=observation.class_id,
            class_name=observation.class_name,
            first_frame=observation.frame_index,
            first_timestamp_ms=observation.timestamp_ms,
            last_frame=observation.frame_index,
            last_timestamp_ms=observation.timestamp_ms,
            observation_count=1,
            confidence_sum=observation.confidence,
            max_confidence=observation.confidence,
            relative_area_sum=observation.bbox_relative_area,
            max_relative_area=observation.bbox_relative_area,
            max_bbox_area_pixels=observation.bbox_area_pixels,
            last_bbox_xyxy_pixel=observation.bbox_xyxy_pixel,
            last_bbox_xyxy_normalized=observation.bbox_xyxy_normalized,
            representative=observation,
        )

    def update(self, observation: DetectionObservation, association_type: str) -> None:
        if observation.class_id != self.class_id:
            raise AggregationError("Different classes must never update one event.")
        if observation.frame_index <= self.last_frame:
            raise AggregationError("An event accepts at most one observation per frame.")
        if association_type not in ("matched_iou", "matched_center_scale"):
            raise AggregationError(f"Unknown matched association type: {association_type!r}.")
        self.last_frame = observation.frame_index
        self.last_timestamp_ms = observation.timestamp_ms
        self.last_bbox_xyxy_pixel = observation.bbox_xyxy_pixel
        self.last_bbox_xyxy_normalized = observation.bbox_xyxy_normalized
        self.observation_count += 1
        self.confidence_sum += observation.confidence
        self.max_confidence = max(self.max_confidence, observation.confidence)
        self.relative_area_sum += observation.bbox_relative_area
        self.max_relative_area = max(
            self.max_relative_area, observation.bbox_relative_area
        )
        self.max_bbox_area_pixels = max(
            self.max_bbox_area_pixels, observation.bbox_area_pixels
        )
        self.matched_observations += 1
        if association_type == "matched_iou":
            self.iou_rule_matches += 1
        else:
            self.center_scale_rule_matches += 1
        if _representative_key(observation) < _representative_key(
            self.representative
        ):
            self.representative = observation

    def finalize(
        self, minimum_confirmed_observations: int, reason: str
    ) -> _FinalizedEvent:
        status = (
            "confirmed"
            if self.observation_count >= minimum_confirmed_observations
            else "tentative"
        )
        return _FinalizedEvent(
            sequence=self.sequence,
            event_id=self.event_id,
            class_id=self.class_id,
            class_name=self.class_name,
            first_frame=self.first_frame,
            last_frame=self.last_frame,
            first_timestamp_ms=self.first_timestamp_ms,
            last_timestamp_ms=self.last_timestamp_ms,
            observation_count=self.observation_count,
            confidence_sum=self.confidence_sum,
            max_confidence=self.max_confidence,
            representative=self.representative,
            max_bbox_area_pixels=self.max_bbox_area_pixels,
            max_relative_area=self.max_relative_area,
            relative_area_sum=self.relative_area_sum,
            last_bbox_xyxy_pixel=self.last_bbox_xyxy_pixel,
            status=status,
            finalized_reason=reason,
            matched_observations=self.matched_observations,
            iou_rule_matches=self.iou_rule_matches,
            center_scale_rule_matches=self.center_scale_rule_matches,
        )


def _representative_key(
    observation: DetectionObservation,
) -> tuple[float, float, int, int]:
    """Preserve the original full-history representative ordering incrementally."""
    return (
        -observation.confidence,
        -observation.bbox_relative_area,
        observation.frame_index,
        observation.detection_index,
    )


@dataclass(frozen=True)
class _FinalizedEvent:
    """Compact immutable event summary; never owns raw observation history."""

    sequence: int
    event_id: str
    class_id: int
    class_name: str
    first_frame: int
    last_frame: int
    first_timestamp_ms: int
    last_timestamp_ms: int
    observation_count: int
    confidence_sum: float
    max_confidence: float
    representative: DetectionObservation
    max_bbox_area_pixels: float
    max_relative_area: float
    relative_area_sum: float
    last_bbox_xyxy_pixel: tuple[float, float, float, float]
    status: str
    finalized_reason: str
    matched_observations: int
    iou_rule_matches: int
    center_scale_rule_matches: int

    def to_record(self) -> dict[str, Any]:
        representative = self.representative
        return {
            "event_id": self.event_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_timestamp_ms": self.first_timestamp_ms,
            "last_timestamp_ms": self.last_timestamp_ms,
            "first_timestamp_seconds": self.first_timestamp_ms / 1000,
            "last_timestamp_seconds": self.last_timestamp_ms / 1000,
            "duration_seconds": (self.last_timestamp_ms - self.first_timestamp_ms)
            / 1000,
            "observation_count": self.observation_count,
            "mean_confidence": self.confidence_sum / self.observation_count,
            "max_confidence": self.max_confidence,
            "representative_frame": representative.frame_index,
            "representative_timestamp_ms": representative.timestamp_ms,
            "representative_timestamp_seconds": representative.timestamp_ms / 1000,
            "representative_bbox_xyxy_pixel": list(representative.bbox_xyxy_pixel),
            "representative_bbox_xyxy_normalized": list(
                representative.bbox_xyxy_normalized
            ),
            "maximum_bbox_area_pixels": self.max_bbox_area_pixels,
            "maximum_bbox_relative_area": self.max_relative_area,
            "mean_bbox_relative_area": self.relative_area_sum
            / self.observation_count,
            "last_bbox_xyxy_pixel": list(self.last_bbox_xyxy_pixel),
            "status": self.status,
            "lifecycle_status": "finalized",
            "finalized_reason": self.finalized_reason,
            "source_track_ids": [],
            "association_statistics": {
                "matched_observations": self.matched_observations,
                "iou_rule_matches": self.iou_rule_matches,
                "center_scale_rule_matches": self.center_scale_rule_matches,
            },
            "interpretation": "temporally aggregated damage event; not a ground-truth physical-world unique object",
        }


class TemporalDamageAggregator:
    """Associate direct detections into same-class, auditable temporal events."""

    def __init__(self, config: AggregationConfig, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise AggregationError("Aggregator FPS must be finite and positive.")
        self.config = config
        self.fps = fps
        self.max_gap_frames = max(
            0, int(math.floor(config.maximum_gap_seconds * fps + 1e-9))
        )
        self._active: dict[str, _EventState] = {}
        self._finalized: list[_FinalizedEvent] = []
        self._next_sequence = 1
        self._last_processed_frame: int | None = None

    def _new_event(self, observation: DetectionObservation) -> _EventState:
        event_id = f"RD{self._next_sequence:04d}"
        state = _EventState.from_observation(
            self._next_sequence, event_id, observation
        )
        self._next_sequence += 1
        self._active[event_id] = state
        return state

    def _finalize(self, event_id: str, reason: str) -> None:
        state = self._active.pop(event_id)
        self._finalized.append(
            state.finalize(self.config.minimum_confirmed_observations, reason)
        )

    def _finalize_expired(self, frame_index: int) -> None:
        expired = [
            state
            for state in self._active.values()
            if frame_index - state.last_frame - 1 > self.max_gap_frames
        ]
        for state in sorted(expired, key=lambda item: item.sequence):
            self._finalize(state.event_id, "maximum_temporal_gap_exceeded")

    def process_frame(
        self, frame_index: int, observations: Sequence[DetectionObservation]
    ) -> tuple[AssociationDecision, ...]:
        """Associate one frame using deterministic global greedy one-to-one matching."""
        if type(frame_index) is not int or frame_index < 0:
            raise AggregationError("frame_index must be a non-negative integer.")
        if self._last_processed_frame is not None and frame_index <= self._last_processed_frame:
            raise AggregationError("Frames must be processed in strictly increasing order.")
        if any(item.frame_index != frame_index for item in observations):
            raise AggregationError("Every observation must belong to the supplied frame.")
        indices = [item.detection_index for item in observations]
        if len(indices) != len(set(indices)):
            raise AggregationError("Detection indices must be unique within a frame.")

        self._finalize_expired(frame_index)
        edges: list[tuple[tuple[Any, ...], _EventState, DetectionObservation, str, float, float, float]] = []
        for state in sorted(self._active.values(), key=lambda item: item.sequence):
            for observation in sorted(observations, key=lambda item: item.detection_index):
                if state.class_id != observation.class_id:
                    continue
                overlap = bbox_iou_xyxy(
                    state.last_bbox_xyxy_normalized,
                    observation.bbox_xyxy_normalized,
                )
                distance = normalized_center_distance(
                    state.last_bbox_xyxy_normalized,
                    observation.bbox_xyxy_normalized,
                )
                area_ratio = bbox_area_ratio(
                    state.last_bbox_xyxy_normalized,
                    observation.bbox_xyxy_normalized,
                )
                if overlap >= self.config.minimum_iou:
                    association_type = "matched_iou"
                elif (
                    distance <= self.config.maximum_center_distance_normalized
                    and area_ratio >= self.config.minimum_area_ratio
                ):
                    association_type = "matched_center_scale"
                else:
                    continue
                rank = (
                    -overlap,
                    distance,
                    -area_ratio,
                    state.sequence,
                    observation.detection_index,
                )
                edges.append(
                    (rank, state, observation, association_type, overlap, distance, area_ratio)
                )

        assigned_events: set[str] = set()
        assigned_detections: set[int] = set()
        decisions: dict[int, AssociationDecision] = {}
        for _, state, observation, kind, overlap, distance, area_ratio in sorted(
            edges, key=lambda item: item[0]
        ):
            if (
                state.event_id in assigned_events
                or observation.detection_index in assigned_detections
            ):
                continue
            state.update(observation, kind)
            assigned_events.add(state.event_id)
            assigned_detections.add(observation.detection_index)
            decisions[observation.detection_index] = AssociationDecision(
                observation.detection_index,
                state.event_id,
                kind,
                overlap,
                distance,
                area_ratio,
            )

        for observation in sorted(observations, key=lambda item: item.detection_index):
            if observation.detection_index in assigned_detections:
                continue
            state = self._new_event(observation)
            decisions[observation.detection_index] = AssociationDecision(
                observation.detection_index,
                state.event_id,
                "new_event",
                None,
                None,
                None,
            )
        self._last_processed_frame = frame_index
        return tuple(decisions[index] for index in sorted(decisions))

    def finalize_all(self, reason: str = "processing_scope_ended") -> None:
        """Finalize every active event without discarding tentative events."""
        for state in sorted(self._active.values(), key=lambda item: item.sequence):
            self._finalize(state.event_id, reason)

    def event_records(self) -> list[dict[str, Any]]:
        """Return finalized events in stable numeric event-ID order."""
        if self._active:
            raise AggregationError("Finalize active events before requesting final records.")
        return [
            state.to_record()
            for state in sorted(self._finalized, key=lambda item: item.sequence)
        ]

    def active_event_ids(self) -> tuple[str, ...]:
        """Expose stable active IDs for progress reporting, not final counting."""
        return tuple(
            state.event_id
            for state in sorted(self._active.values(), key=lambda item: item.sequence)
        )
