"""Synthetic and pure-logic tests for the Phase 4 video-analysis pipeline."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_damage.aggregation.temporal import (  # noqa: E402
    AggregationConfig,
    AssociationDecision,
    DetectionObservation,
    TemporalDamageAggregator,
)
from road_damage.inference import video_analysis as tool  # noqa: E402


def make_video(root: Path, frames: int = 6, fps: float = 10.0) -> Path:
    path = root / "fixture.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create the synthetic Phase 4 fixture.")
    try:
        for index in range(frames):
            frame = np.full((120, 160, 3), 40 + index, dtype=np.uint8)
            cv2.line(frame, (20, 30), (140, 105), (10, 10, 10), 2)
            writer.write(frame)
    finally:
        writer.release()
    return path


class Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Boxes:
    def __init__(self, rows: list[tuple[int, float, tuple[float, float, float, float]]]):
        self.rows = rows
        self.xyxy = Values([list(row[2]) for row in rows])
        self.cls = Values([float(row[0]) for row in rows])
        self.conf = Values([row[1] for row in rows])

    def __len__(self):
        return len(self.rows)


def result(*rows: tuple[int, float, tuple[float, float, float, float]]):
    return SimpleNamespace(boxes=Boxes(list(rows)))


def model_with(results) -> MagicMock:
    model = MagicMock()
    model.names = dict(tool.CLASS_NAMES)
    model.task = "detect"
    model.model = SimpleNamespace(yaml={"nc": 4, "scale": "s", "yaml_file": "yolov8.yaml"})
    model.predict.side_effect = [
        value if isinstance(value, BaseException) else [value] for value in results
    ]
    return model


def prepare_root(root: Path) -> None:
    target = root / tool.CONFIG_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / tool.CONFIG_PATH).read_bytes())


def observation(
    frame: int,
    index: int = 0,
    class_id: int = 3,
    box: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    confidence: float = 0.8,
) -> DetectionObservation:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    return DetectionObservation(
        index,
        frame,
        frame * 100,
        class_id,
        tool.CLASS_NAMES[class_id],
        confidence,
        (x1 * 100, y1 * 100, x2 * 100, y2 * 100),
        box,
        area * 10_000,
        area,
    )


def aggregation_config(minimum_observations: int = 3) -> AggregationConfig:
    return AggregationConfig(.20, .12, .25, .20, minimum_observations)


def verification_fixture(root: Path) -> tuple[tool.RunPaths, dict]:
    """Persist a small internally consistent frame/event/summary contract."""
    root.mkdir(parents=True, exist_ok=True)
    paths = tool.RunPaths(
        root,
        root / "run_manifest.json",
        root / "summary.json",
        root / "events.json",
        root / "frame_detections.jsonl",
        root / "completion.json",
        root / "annotated_video.mp4",
        root / ".lock",
    )
    aggregator = TemporalDamageAggregator(aggregation_config(), 10.)
    rows = []
    for frame in range(3):
        observations = [observation(frame, confidence=.7 + frame * .1)]
        if frame == 2:
            observations.append(
                observation(frame, index=1, class_id=0, box=(.6, .6, .8, .8))
            )
        decisions = aggregator.process_frame(frame, observations)
        rows.append(tool.frame_record(frame, 10., 100, 100, observations, decisions))
    aggregator.finalize_all("video_end")
    events = aggregator.event_records()
    counts = tool._event_counts(events)
    raw_by_class = {name: 0 for name in tool.CLASS_NAMES.values()}
    raw_by_class[tool.CLASS_NAMES[3]] = 3
    raw_by_class[tool.CLASS_NAMES[0]] = 1
    summary = {
        "frames": {"inferred": 3},
        "raw_detection_observations": {"total": 4, "by_class": raw_by_class},
        "events": {
            "total": len(events),
            "by_class": counts["all"],
            "confirmed_total": 1,
            "confirmed_by_class": counts["confirmed"],
            "tentative_total": 1,
            "tentative_by_class": counts["tentative"],
        },
    }
    paths.frame_detections.write_bytes(b"".join(tool.jsonl_bytes(row) for row in rows))
    tool._write_json(
        paths.events,
        {"event_count": len(events), "events": events},
    )
    tool._write_json(paths.summary, summary)
    return paths, copy.deepcopy(summary)


def empty_verification_fixture(
    root: Path, frame_count: int
) -> tuple[tool.RunPaths, dict]:
    root.mkdir(parents=True, exist_ok=True)
    paths = tool.RunPaths(
        root,
        root / "run_manifest.json",
        root / "summary.json",
        root / "events.json",
        root / "frame_detections.jsonl",
        root / "completion.json",
        root / "annotated_video.mp4",
        root / ".lock",
    )
    rows = (
        {
            "frame_index": index,
            "timestamp_ms": index * 40,
            "timestamp_seconds": index * .04,
            "source_width": 1280,
            "source_height": 720,
            "detections": [],
        }
        for index in range(frame_count)
    )
    paths.frame_detections.write_bytes(b"".join(tool.jsonl_bytes(row) for row in rows))
    zero = {name: 0 for name in tool.CLASS_NAMES.values()}
    summary = {
        "frames": {"inferred": frame_count},
        "raw_detection_observations": {"total": 0, "by_class": zero},
        "events": {
            "total": 0,
            "by_class": zero,
            "confirmed_total": 0,
            "confirmed_by_class": zero,
            "tentative_total": 0,
            "tentative_by_class": zero,
        },
    }
    tool._write_json(paths.events, {"event_count": 0, "events": []})
    tool._write_json(paths.summary, summary)
    return paths, copy.deepcopy(summary)


class FrozenPolicyTests(unittest.TestCase):
    def test_config_model_class_threshold_and_inference_contract(self) -> None:
        config = tool.load_config(ROOT)
        self.assertEqual(config.model_path, Path(
            "outputs/training/baseline_public_v1/20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42/weights/best.pt"
        ))
        self.assertEqual(config.model_sha256,
                         "BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7")
        self.assertEqual(config.model_size_bytes, 22_524_074)
        self.assertEqual((config.confidence, config.nms_iou), (.19, .50))
        self.assertEqual((config.imgsz, config.max_detections, config.frame_stride), (640, 300, 1))
        self.assertEqual(tool.EXPECTED_CONFIG["classes"], {
            "0": "D00_longitudinal_crack", "1": "D10_transverse_crack",
            "2": "D20_alligator_crack", "3": "D40_pothole",
        })
        self.assertNotIn("model", {action.dest for action in tool._parser()._actions})
        self.assertNotIn("confidence", {action.dest for action in tool._parser()._actions})
        self.assertNotIn("nms_iou", {action.dest for action in tool._parser()._actions})

    def test_config_rejects_model_threshold_class_and_aggregation_drift(self) -> None:
        original = json.loads((ROOT / tool.CONFIG_PATH).read_text(encoding="utf-8"))
        mutations = (
            lambda value: value["model"].update(path="last.pt"),
            lambda value: value["model"].update(sha256="0" * 64),
            lambda value: value["operating_point"].update(confidence=.25),
            lambda value: value["operating_point"].update(nms_iou=.70),
            lambda value: value["classes"].update({"0": "wrong"}),
            lambda value: value["temporal_aggregation"].update(minimum_iou=.5),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                changed = copy.deepcopy(original)
                mutate(changed)
                path = root / tool.CONFIG_PATH
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(tool.VideoAnalysisError):
                    tool.load_config(root)

    def test_checkpoint_allowlist_sha_size_and_loaded_class_architecture_guards(self) -> None:
        config = tool.load_config(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / config.model_path
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"fixture")
            with patch.object(tool, "verify_model_identity", return_value=config.model_sha256) as verify:
                path, sha = tool.verify_frozen_checkpoint(config, root)
            self.assertEqual((path, sha), (expected, config.model_sha256))
            verify.assert_called_once_with(
                expected,
                expected_sha256=config.model_sha256,
                expected_size_bytes=22_524_074,
            )
            with self.assertRaises(tool.VideoAnalysisError):
                tool.verify_frozen_checkpoint(replace(config, model_path=Path("last.pt")), root)
        bad_names = model_with([])
        bad_names.names = {0: "D40_pothole"}
        with self.assertRaises(tool.VideoAnalysisError):
            tool.validate_loaded_model(bad_names)
        bad_scale = model_with([])
        bad_scale.model.yaml["scale"] = "n"
        with self.assertRaises(tool.VideoAnalysisError):
            tool.validate_loaded_model(bad_scale)

    def test_loaded_model_requires_explicit_detect_task(self) -> None:
        detected = model_with([])
        self.assertEqual(tool.validate_loaded_model(detected)["task"], "detect")

        missing = SimpleNamespace(
            names=dict(tool.CLASS_NAMES),
            model=SimpleNamespace(yaml={"nc": 4, "scale": "s"}),
        )
        with self.assertRaises(tool.VideoAnalysisError):
            tool.validate_loaded_model(missing)

        for task in (None, "segment", "classify", "pose", "obb", "anything_else"):
            rejected = model_with([])
            rejected.task = task
            with self.subTest(task=task), self.assertRaises(tool.VideoAnalysisError):
                tool.validate_loaded_model(rejected)

    def test_preflight_validates_without_model_construction_or_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root)
            output = root / "analysis"
            with patch.object(tool, "verify_frozen_checkpoint", return_value=(root / "best.pt", tool.FROZEN_MODEL_SHA256)), \
                    patch.object(tool, "resolve_device", return_value="cpu"):
                report = tool.preflight(source, output, root=root, cv2_module=cv2)
            self.assertEqual(report["status"], "PASSED")
            self.assertFalse(report["model_inference_executed"])
            self.assertFalse(report["internal_test_accessed"])
            self.assertFalse(output.exists())

    def test_internal_test_and_dataset_paths_are_rejected_before_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for source in (
                root / "internal_test/private.avi",
                root / "data/exports/rdd/images/test/private.avi",
            ):
                with patch.object(Path, "is_file", side_effect=AssertionError("must not access")):
                    with self.assertRaises(tool.VideoAnalysisError):
                        tool.inspect_video_input(source, root=root, cv2_module=cv2)
            with self.assertRaises(tool.VideoAnalysisError):
                tool.validate_output_path(root / "internal-test/run", root / "source.avi", root)


class SerializationAndInputTests(unittest.TestCase):
    def test_input_metadata_timestamp_and_detection_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_video(root, frames=4, fps=12.5)
            metadata = tool.inspect_video_input(source, root=root, cv2_module=cv2)
            self.assertEqual((metadata["width"], metadata["height"]), (160, 120))
            self.assertAlmostEqual(metadata["fps"], 12.5)
            self.assertEqual(metadata["frame_count"], 4)
            self.assertEqual(metadata["size_bytes"], source.stat().st_size)
            self.assertEqual(metadata["sha256"], tool.sha256_file(source))
        detected = tool.Detection(3, "D40_pothole", .8, (16, 24, 48, 48))
        item = tool.observation_from_detection(detected, 0, 5, 10., 160, 120)
        decision = SimpleNamespace(
            detection_index=0, event_id="RD0001", to_record=lambda: {
                "type": "new_event", "iou_with_previous": None,
                "center_distance_normalized": None, "area_ratio": None,
            }
        )
        row = tool.frame_record(5, 10., 160, 120, [item], [decision])
        self.assertEqual((row["frame_index"], row["timestamp_ms"], row["timestamp_seconds"]), (5, 500, .5))
        self.assertEqual(row["detections"][0]["bbox_xyxy_pixel"], [16., 24., 48., 48.])
        self.assertEqual(row["detections"][0]["bbox_xyxy_normalized"], [.1, .2, .3, .4])
        self.assertEqual(row["detections"][0]["event_id"], "RD0001")

    def test_jsonl_serialization_is_byte_deterministic(self) -> None:
        left = {"z": [2, 1], "a": {"value": .5}}
        right = {"a": {"value": .5}, "z": [2, 1]}
        self.assertEqual(tool.jsonl_bytes(left), tool.jsonl_bytes(right))
        self.assertEqual(tool.jsonl_bytes(left), b'{"a":{"value":0.5},"z":[2,1]}\n')

    def test_unreadable_video_and_output_collision_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "bad.avi"
            bad.write_bytes(b"not-video")
            with self.assertRaises(tool.VideoAnalysisError):
                tool.inspect_video_input(bad, root=root, cv2_module=cv2)
            source = make_video(root)
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "completion.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(tool.VideoAnalysisError):
                tool.validate_output_path(occupied, source, root)


class AnnotationRenderingTests(unittest.TestCase):
    def test_all_classes_render_friendly_name_confidence_and_event_id(self) -> None:
        config = tool.load_config(ROOT)
        observations = tuple(
            observation(0, index=class_id, class_id=class_id, confidence=.37)
            for class_id in range(4)
        )
        decisions = tuple(
            AssociationDecision(
                class_id,
                f"RD{class_id + 1:04d}",
                "new_event",
                None,
                None,
                None,
            )
            for class_id in range(4)
        )
        cv2_spy = SimpleNamespace(
            FILLED=cv2.FILLED,
            FONT_HERSHEY_SIMPLEX=cv2.FONT_HERSHEY_SIMPLEX,
            LINE_AA=cv2.LINE_AA,
            rectangle=MagicMock(side_effect=cv2.rectangle),
            getTextSize=MagicMock(side_effect=cv2.getTextSize),
            putText=MagicMock(side_effect=cv2.putText),
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        rendered = tool.render_frame(
            frame, observations, decisions, config, cv2_spy
        )

        labels = [call.args[1] for call in cv2_spy.putText.call_args_list]
        self.assertEqual(
            labels,
            [
                "Longitudinal Crack | 0.37 | RD0001",
                "Transverse Crack | 0.37 | RD0002",
                "Alligator Crack | 0.37 | RD0003",
                "Pothole | 0.37 | RD0004",
            ],
        )
        self.assertEqual(rendered.shape, frame.shape)
        self.assertEqual(
            [item.class_name for item in observations],
            list(tool.CLASS_NAMES.values()),
        )
        self.assertEqual(
            [item.event_id for item in decisions],
            ["RD0001", "RD0002", "RD0003", "RD0004"],
        )
        with self.assertRaises(TypeError):
            tool.ANNOTATION_DISPLAY_NAMES[0] = "Changed"

    def test_friendly_overlay_does_not_change_machine_readable_serialization(self) -> None:
        observations = tuple(
            observation(0, index=class_id, class_id=class_id, confidence=.37)
            for class_id in range(4)
        )
        decisions = tuple(
            AssociationDecision(
                class_id,
                f"RD{class_id + 1:04d}",
                "new_event",
                None,
                None,
                None,
            )
            for class_id in range(4)
        )
        row = tool.frame_record(0, 10., 160, 120, observations, decisions)
        self.assertEqual(
            [(item["class_id"], item["class_name"], item["event_id"])
             for item in row["detections"]],
            [(class_id, tool.CLASS_NAMES[class_id], f"RD{class_id + 1:04d}")
             for class_id in range(4)],
        )


class TemporalAggregationTests(unittest.TestCase):
    def test_same_class_stable_event_confirmation_and_all_events_preserved(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(), 10.)
        ids = []
        for frame in range(3):
            ids.append(aggregator.process_frame(frame, [observation(frame)])[0].event_id)
        aggregator.process_frame(3, [observation(3, class_id=0, box=(.6, .6, .8, .8))])
        aggregator.finalize_all()
        events = aggregator.event_records()
        self.assertEqual(ids, ["RD0001"] * 3)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["status"], "confirmed")
        self.assertEqual(events[0]["observation_count"], 3)
        self.assertEqual(events[1]["status"], "tentative")

    def test_different_classes_never_merge(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(1), 10.)
        first = aggregator.process_frame(0, [observation(0, class_id=0)])[0]
        second = aggregator.process_frame(1, [observation(1, class_id=1)])[0]
        self.assertNotEqual(first.event_id, second.event_id)
        aggregator.finalize_all()
        self.assertEqual([event["class_id"] for event in aggregator.event_records()], [0, 1])

    def test_short_gap_recovers_and_excessive_gap_closes_event(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(1), 10.)
        first = aggregator.process_frame(0, [observation(0)])[0]
        aggregator.process_frame(1, [])
        aggregator.process_frame(2, [])
        recovered = aggregator.process_frame(3, [observation(3)])[0]
        self.assertEqual(first.event_id, recovered.event_id)
        aggregator.process_frame(4, [])
        aggregator.process_frame(5, [])
        aggregator.process_frame(6, [])
        later = aggregator.process_frame(7, [observation(7)])[0]
        self.assertNotEqual(first.event_id, later.event_id)
        aggregator.finalize_all()
        events = aggregator.event_records()
        self.assertEqual(events[0]["finalized_reason"], "maximum_temporal_gap_exceeded")

    def test_25_fps_gap_boundary_remains_n_plus_6_allowed_n_plus_7_expired(self) -> None:
        reconnecting = TemporalDamageAggregator(aggregation_config(1), 25.)
        first = reconnecting.process_frame(0, [observation(0)])[0]
        for frame in range(1, 6):
            reconnecting.process_frame(frame, [])
        at_n_plus_6 = reconnecting.process_frame(6, [observation(6)])[0]
        self.assertEqual(first.event_id, at_n_plus_6.event_id)

        expiring = TemporalDamageAggregator(aggregation_config(1), 25.)
        first = expiring.process_frame(0, [observation(0)])[0]
        for frame in range(1, 7):
            expiring.process_frame(frame, [])
        at_n_plus_7 = expiring.process_frame(7, [observation(7)])[0]
        self.assertNotEqual(first.event_id, at_n_plus_7.event_id)

    def test_two_objects_one_to_one_assignment_and_deterministic_ties(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(1), 10.)
        created = aggregator.process_frame(0, [observation(0, 0), observation(0, 1)])
        tied = aggregator.process_frame(1, [observation(1, 0), observation(1, 1)])
        self.assertEqual([item.event_id for item in created], ["RD0001", "RD0002"])
        self.assertEqual([item.event_id for item in tied], ["RD0001", "RD0002"])
        aggregator.finalize_all()
        self.assertEqual([event["observation_count"] for event in aggregator.event_records()], [2, 2])

        one = TemporalDamageAggregator(aggregation_config(1), 10.)
        one.process_frame(0, [observation(0)])
        decisions = one.process_frame(1, [observation(1, 0), observation(1, 1)])
        self.assertEqual(len({item.event_id for item in decisions}), 2)
        one.finalize_all()
        self.assertEqual(len(one.event_records()), 2)

    def test_event_schema_representative_and_confidence_are_deterministic(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(2), 10.)
        aggregator.process_frame(0, [observation(0, confidence=.7)])
        aggregator.process_frame(1, [observation(1, confidence=.9)])
        aggregator.finalize_all("video_end")
        event = aggregator.event_records()[0]
        required = {
            "event_id", "class_id", "class_name", "first_frame", "last_frame",
            "first_timestamp_seconds", "last_timestamp_seconds", "duration_seconds",
            "observation_count", "max_confidence", "mean_confidence",
            "representative_frame", "representative_timestamp_seconds",
            "representative_bbox_xyxy_pixel", "status",
        }
        self.assertTrue(required.issubset(event))
        self.assertEqual((event["event_id"], event["representative_frame"], event["status"]),
                         ("RD0001", 1, "confirmed"))
        self.assertAlmostEqual(event["mean_confidence"], .8)

    def test_active_event_state_is_bounded_without_observation_history(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(), 25.)
        for frame in range(500):
            aggregator.process_frame(frame, [observation(frame)])
        state = next(iter(aggregator._active.values()))
        self.assertEqual(state.observation_count, 500)
        self.assertFalse(hasattr(state, "observations"))
        self.assertFalse(hasattr(state, "association_types"))
        self.assertFalse(any(isinstance(value, list) for value in vars(state).values()))

    def test_compact_representative_keeps_original_deterministic_order(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(1), 10.)
        aggregator.process_frame(
            0, [observation(0, confidence=.9, box=(.10, .10, .30, .30))]
        )
        aggregator.process_frame(
            1, [observation(1, confidence=.9, box=(.05, .05, .35, .35))]
        )
        aggregator.process_frame(
            2, [observation(2, confidence=.9, box=(.05, .05, .35, .35))]
        )
        aggregator.finalize_all()
        event = aggregator.event_records()[0]
        self.assertEqual(event["representative_frame"], 1)
        self.assertEqual(event["representative_timestamp_ms"], 100)

    def test_incremental_confidence_and_area_aggregates_match_observations(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(1), 10.)
        inputs = (
            observation(0, confidence=.2, box=(.10, .10, .20, .20)),
            observation(1, confidence=.6, box=(.05, .05, .25, .25)),
            observation(2, confidence=1., box=(.00, .00, .30, .30)),
        )
        for item in inputs:
            aggregator.process_frame(item.frame_index, [item])
        aggregator.finalize_all()
        event = aggregator.event_records()[0]
        self.assertAlmostEqual(event["mean_confidence"], .6)
        self.assertAlmostEqual(event["max_confidence"], 1.)
        self.assertAlmostEqual(event["mean_bbox_relative_area"], (0.01 + .04 + .09) / 3)
        self.assertAlmostEqual(event["maximum_bbox_relative_area"], .09)
        self.assertAlmostEqual(event["maximum_bbox_area_pixels"], 900.)

    def test_finalized_event_is_compact_immutable_and_serializes_no_history(self) -> None:
        aggregator = TemporalDamageAggregator(aggregation_config(1), 10.)
        for frame in range(20):
            aggregator.process_frame(frame, [observation(frame)])
        aggregator.finalize_all()
        finalized = aggregator._finalized[0]
        self.assertFalse(hasattr(finalized, "observations"))
        self.assertFalse(hasattr(finalized, "association_types"))
        self.assertFalse(any(isinstance(value, list) for value in vars(finalized).values()))
        with self.assertRaises(FrozenInstanceError):
            finalized.event_id = "RD9999"
        serialized = aggregator.event_records()[0]
        self.assertNotIn("observations", serialized)
        self.assertNotIn("association_types", serialized)
        self.assertEqual(serialized["observation_count"], 20)


class PersistedOutputVerificationTests(unittest.TestCase):
    @staticmethod
    def load(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def save(path: Path, value) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_jsonl_verification_uses_iteration_and_never_bulk_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            original_open = Path.open
            original_read_bytes = Path.read_bytes

            class StreamingOnly:
                def __init__(self, wrapped):
                    self.wrapped = wrapped

                def __enter__(self):
                    self.wrapped.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.wrapped.__exit__(*args)

                def __iter__(self):
                    return iter(self.wrapped)

                def read(self, *args, **kwargs):
                    raise AssertionError("bulk text read is forbidden")

                def readlines(self, *args, **kwargs):
                    raise AssertionError("bulk readlines is forbidden")

            def guarded_open(path, *args, **kwargs):
                handle = original_open(path, *args, **kwargs)
                mode = args[0] if args else kwargs.get("mode", "r")
                if Path(path) == paths.frame_detections and mode == "r":
                    return StreamingOnly(handle)
                return handle

            def guarded_read_bytes(path):
                if Path(path) == paths.frame_detections:
                    raise AssertionError("read_bytes is forbidden for JSONL verification")
                return original_read_bytes(path)

            with patch.object(Path, "open", new=guarded_open), patch.object(
                Path, "read_bytes", new=guarded_read_bytes
            ):
                hashes = tool._verify_persisted_outputs(paths, expected, False)
            self.assertEqual(set(hashes), {
                "summary.json", "events.json", "frame_detections.jsonl"
            })

    def test_large_synthetic_jsonl_is_streamed_and_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = empty_verification_fixture(Path(temporary), 2_000)
            hashes = tool._verify_persisted_outputs(paths, expected, False)
            self.assertEqual(len(hashes["frame_detections.jsonl"]), 64)

    def test_raw_detection_total_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            summary = self.load(paths.summary)
            summary["raw_detection_observations"]["total"] = 5
            self.save(paths.summary, summary)
            with self.assertRaises(tool.VideoAnalysisError):
                tool._verify_persisted_outputs(paths, expected, False)

    def test_raw_per_class_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            rows = [json.loads(line) for line in paths.frame_detections.read_text().splitlines()]
            rows[0]["detections"][0]["class_id"] = 2
            rows[0]["detections"][0]["class_name"] = tool.CLASS_NAMES[2]
            paths.frame_detections.write_bytes(
                b"".join(tool.jsonl_bytes(row) for row in rows)
            )
            with self.assertRaises(tool.VideoAnalysisError):
                tool._verify_persisted_outputs(paths, expected, False)

    def test_declared_event_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            events = self.load(paths.events)
            events["event_count"] += 1
            self.save(paths.events, events)
            with self.assertRaises(tool.VideoAnalysisError):
                tool._verify_persisted_outputs(paths, expected, False)

    def test_event_per_class_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            events = self.load(paths.events)
            events["events"][0]["class_id"] = 2
            events["events"][0]["class_name"] = tool.CLASS_NAMES[2]
            self.save(paths.events, events)
            with self.assertRaises(tool.VideoAnalysisError):
                tool._verify_persisted_outputs(paths, expected, False)

    def test_confirmed_tentative_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            events = self.load(paths.events)
            events["events"][0]["status"] = "tentative"
            self.save(paths.events, events)
            with self.assertRaises(tool.VideoAnalysisError):
                tool._verify_persisted_outputs(paths, expected, False)

    def test_duplicate_event_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            events = self.load(paths.events)
            events["events"][1]["event_id"] = events["events"][0]["event_id"]
            self.save(paths.events, events)
            with self.assertRaises(tool.VideoAnalysisError):
                tool._verify_persisted_outputs(paths, expected, False)

    def test_jsonl_unknown_event_reference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            rows = [json.loads(line) for line in paths.frame_detections.read_text().splitlines()]
            rows[0]["detections"][0]["event_id"] = "RD9999"
            paths.frame_detections.write_bytes(
                b"".join(tool.jsonl_bytes(row) for row in rows)
            )
            with self.assertRaises(tool.VideoAnalysisError):
                tool._verify_persisted_outputs(paths, expected, False)

    def test_invalid_event_id_and_class_identity_fail(self) -> None:
        for mutation in ("event_id", "class_id", "class_name"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                paths, expected = verification_fixture(Path(temporary))
                events = self.load(paths.events)
                event = events["events"][0]
                if mutation == "event_id":
                    event["event_id"] = "event-1"
                elif mutation == "class_id":
                    event["class_id"] = 4
                else:
                    event["class_name"] = "D40"
                self.save(paths.events, events)
                with self.assertRaises(tool.VideoAnalysisError):
                    tool._verify_persisted_outputs(paths, expected, False)

    def test_non_finite_detection_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, expected = verification_fixture(Path(temporary))
            rows = [json.loads(line) for line in paths.frame_detections.read_text().splitlines()]
            rows[0]["detections"][0]["confidence"] = float("nan")
            paths.frame_detections.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(tool.VideoAnalysisError):
                tool._verify_persisted_outputs(paths, expected, False)


class PipelineIntegrationTests(unittest.TestCase):
    def invoke(
        self,
        root: Path,
        source: Path,
        model: MagicMock,
        *,
        no_video: bool = True,
        stop_event: threading.Event | None = None,
        **kwargs,
    ):
        output = root / kwargs.pop("output_name", "analysis")
        cv2_backend = kwargs.pop("cv2_module", cv2)
        factory = MagicMock(return_value=model)
        with patch.object(
            tool,
            "verify_frozen_checkpoint",
            return_value=(root / "best.pt", tool.FROZEN_MODEL_SHA256),
        ), patch.object(tool, "resolve_device", return_value="cpu"):
            summary = tool.analyze_video(
                source,
                output,
                root=root,
                yolo_factory=factory,
                cv2_module=cv2_backend,
                no_annotated_video=no_video,
                stop_event=stop_event,
                **kwargs,
            )
        return summary, output, factory

    def test_empty_detection_video_summary_and_completion_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=4)
            summary, output, factory = self.invoke(
                root, source, model_with([result()] * 4)
            )
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["raw_detection_observations"]["total"], 0)
            self.assertEqual(summary["events"]["total"], 0)
            self.assertEqual(summary["frames"]["inferred"], 4)
            self.assertTrue((output / "completion.json").is_file())
            self.assertEqual(json.loads((output / "events.json").read_text())["events"], [])
            self.assertEqual(len((output / "frame_detections.jsonl").read_text().splitlines()), 4)
            factory.assert_called_once_with(str(root / "best.pt"), task="detect")

    def test_raw_counts_aggregate_counts_start_frame_and_frozen_predict_kwargs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=6)
            one = result((2, .75, (20., 30., 140., 105.)))
            model = model_with([one] * 3)
            summary, output, _ = self.invoke(
                root, source, model, start_frame=2, max_frames=3
            )
            self.assertEqual(summary["frames"]["inferred"], 3)
            self.assertEqual(summary["frames"]["skipped_before_start"], 2)
            self.assertFalse(summary["frames"]["full_source_processed"])
            self.assertEqual(summary["raw_detection_observations"]["by_class"]["D20_alligator_crack"], 3)
            self.assertEqual(summary["events"]["total"], 1)
            self.assertEqual(summary["events"]["confirmed_total"], 1)
            first = json.loads((output / "frame_detections.jsonl").read_text().splitlines()[0])
            self.assertEqual((first["frame_index"], first["timestamp_ms"]), (2, 200))
            for call in model.predict.call_args_list:
                self.assertEqual(
                    (call.kwargs["conf"], call.kwargs["iou"], call.kwargs["imgsz"],
                     call.kwargs["max_det"], call.kwargs["device"]),
                    (.19, .50, 640, 300, "cpu"),
                )
                self.assertFalse(call.kwargs["augment"])

    def test_annotated_video_preserves_dimensions_fps_and_same_inference_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=3, fps=10.)
            model = model_with([result((3, .8, (20., 30., 80., 90.)))] * 3)
            summary, output, _ = self.invoke(root, source, model, no_video=False)
            self.assertEqual(model.predict.call_count, 3)
            self.assertTrue(summary["encoding_verification"]["passed"])
            self.assertEqual(summary["encoding_verification"]["encoded_frames"], 3)
            self.assertTrue((output / "annotated_video.mp4").is_file())
            capture = cv2.VideoCapture(str(output / "annotated_video.mp4"))
            try:
                self.assertEqual(
                    (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                    (160, 120),
                )
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 10.)
            finally:
                capture.release()

    def test_model_failure_preserves_partial_diagnostics_without_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=4)
            model = model_with([result((3, .8, (20., 30., 80., 90.))), RuntimeError("synthetic failure")])
            output = root / "failed"
            with patch.object(tool, "verify_frozen_checkpoint", return_value=(root / "best.pt", tool.FROZEN_MODEL_SHA256)), \
                    patch.object(tool, "resolve_device", return_value="cpu"):
                with self.assertRaises(tool.VideoAnalysisError):
                    tool.analyze_video(source, output, root=root,
                                       yolo_factory=MagicMock(return_value=model),
                                       cv2_module=cv2, no_annotated_video=True)
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(json.loads((output / "run_manifest.json").read_text())["status"],
                             "FAILED_TECHNICAL")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["frames"]["inferred"], 1)
            self.assertEqual(summary["status"], "FAILED_TECHNICAL")

    def test_reconciliation_failure_never_writes_completion_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=2)
            with patch.object(
                tool,
                "_verify_persisted_outputs",
                side_effect=tool.VideoAnalysisError("synthetic reconciliation failure"),
            ):
                with self.assertRaises(tool.VideoAnalysisError):
                    self.invoke(root, source, model_with([result(), result()]))
            output = root / "analysis"
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["status"],
                "FAILED_TECHNICAL",
            )
            self.assertEqual(
                json.loads((output / "run_manifest.json").read_text())["status"],
                "FAILED_TECHNICAL",
            )

    def test_late_exception_after_receipt_is_failed_and_removes_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=2)
            output = root / "analysis"
            original_read_text = Path.read_text
            receipt_seen = False

            def fail_after_receipt(path, *args, **kwargs):
                nonlocal receipt_seen
                if Path(path) == output / "completion.json":
                    self.assertTrue(Path(path).is_file())
                    receipt_seen = True
                    raise RuntimeError("synthetic late publication failure")
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, "read_text", new=fail_after_receipt):
                with self.assertRaises(tool.VideoAnalysisError):
                    self.invoke(root, source, model_with([result(), result()]))
            self.assertTrue(receipt_seen)
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["status"],
                "FAILED_TECHNICAL",
            )
            self.assertEqual(
                json.loads((output / "run_manifest.json").read_text())["status"],
                "FAILED_TECHNICAL",
            )

    def test_late_keyboard_interrupt_after_receipt_invalidates_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=2)
            output = root / "analysis"
            original_read_text = Path.read_text
            receipt_seen = False

            class TrackedResource:
                def __init__(self, wrapped):
                    self.wrapped = wrapped
                    self.released = False

                def release(self):
                    self.released = True
                    return self.wrapped.release()

                def __getattr__(self, name):
                    return getattr(self.wrapped, name)

            class TrackingCV2:
                def __init__(self):
                    self.captures = []
                    self.writers = []

                def VideoCapture(self, *args, **kwargs):
                    resource = TrackedResource(cv2.VideoCapture(*args, **kwargs))
                    self.captures.append(resource)
                    return resource

                def VideoWriter(self, *args, **kwargs):
                    resource = TrackedResource(cv2.VideoWriter(*args, **kwargs))
                    self.writers.append(resource)
                    return resource

                def __getattr__(self, name):
                    return getattr(cv2, name)

            tracking_cv2 = TrackingCV2()

            def interrupt_after_receipt(path, *args, **kwargs):
                nonlocal receipt_seen
                if Path(path) == output / "completion.json":
                    self.assertTrue(Path(path).is_file())
                    receipt_seen = True
                    raise KeyboardInterrupt("synthetic late interrupt")
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, "read_text", new=interrupt_after_receipt):
                summary, returned_output, _ = self.invoke(
                    root,
                    source,
                    model_with([result(), result()]),
                    no_video=False,
                    cv2_module=tracking_cv2,
                )
            self.assertEqual(returned_output, output)
            self.assertTrue(receipt_seen)
            self.assertEqual(summary["status"], "INTERRUPTED")
            self.assertNotEqual(summary["status"], "COMPLETED")
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["status"],
                "INTERRUPTED",
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "INTERRUPTED")
            self.assertEqual(manifest["stop_reason"], "late_keyboard_interrupt")
            self.assertTrue(tracking_cv2.captures)
            self.assertTrue(tracking_cv2.writers)
            self.assertTrue(all(item.released for item in tracking_cv2.captures))
            self.assertTrue(all(item.released for item in tracking_cv2.writers))
            self.assertFalse((output / ".run.lock").exists())

    def test_late_interrupt_quarantines_receipt_when_direct_delete_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=1)
            output = root / "analysis"
            original_read_text = Path.read_text
            original_unlink = Path.unlink
            receipt_seen = False
            direct_delete_refused = False

            def interrupt_after_receipt(path, *args, **kwargs):
                nonlocal receipt_seen
                if Path(path) == output / "completion.json":
                    self.assertTrue(Path(path).is_file())
                    receipt_seen = True
                    raise KeyboardInterrupt("synthetic late interrupt")
                return original_read_text(path, *args, **kwargs)

            def refuse_direct_completion_delete(path, *args, **kwargs):
                nonlocal direct_delete_refused
                if Path(path) == output / "completion.json":
                    direct_delete_refused = True
                    raise PermissionError("synthetic completion delete failure")
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "read_text", new=interrupt_after_receipt), patch.object(
                Path, "unlink", new=refuse_direct_completion_delete
            ):
                summary, _, _ = self.invoke(
                    root, source, model_with([result()])
                )
            self.assertTrue(receipt_seen)
            self.assertTrue(direct_delete_refused)
            self.assertEqual(summary["status"], "INTERRUPTED")
            self.assertFalse((output / "completion.json").exists())
            self.assertFalse((output / "completion.invalidated.json").exists())
            self.assertEqual(
                json.loads((output / "run_manifest.json").read_text())["status"],
                "INTERRUPTED",
            )

    def test_keyboard_interrupt_before_receipt_is_interrupted_without_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=2)
            summary, output, _ = self.invoke(
                root,
                source,
                model_with([KeyboardInterrupt("synthetic inference interrupt")]),
            )
            self.assertEqual(summary["status"], "INTERRUPTED")
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(
                json.loads((output / "run_manifest.json").read_text())["status"],
                "INTERRUPTED",
            )

    def test_interruption_is_not_completed_and_preserves_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_root(root)
            source = make_video(root, frames=4)
            stop = threading.Event()
            model = model_with([])

            def predict(**kwargs):
                stop.set()
                return [result((0, .7, (20., 30., 80., 90.)))]

            model.predict.side_effect = predict
            summary, output, _ = self.invoke(root, source, model, stop_event=stop)
            self.assertEqual(summary["status"], "INTERRUPTED")
            self.assertEqual(summary["frames"]["inferred"], 1)
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(json.loads((output / "events.json").read_text())["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
