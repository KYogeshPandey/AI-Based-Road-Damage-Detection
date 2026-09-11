"""Synthetic video/encoder integration and mocked inference/GUI lifecycle tests."""

from __future__ import annotations

import csv
import hashlib
import json
import signal
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_damage.demo import video_demo as demo  # noqa: E402


def make_video(root: Path, frames: int = 6) -> Path:
    path = root / "fixture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.5, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("Existing OpenCV environment cannot encode synthetic MJPG fixture.")
    try:
        for index in range(frames):
            frame = np.full((240, 320, 3), 50 + index, dtype=np.uint8)
            cv2.line(frame, (50, 60), (280, 220), (20, 20, 20), 3)
            writer.write(frame)
    finally:
        writer.release()
    return path


def prediction() -> SimpleNamespace:
    boxes = MagicMock()
    boxes.__len__.return_value = 1
    boxes.xyxy.cpu.return_value.tolist.return_value = [[40, 50, 290, 225]]
    boxes.conf.cpu.return_value.tolist.return_value = [.75]
    boxes.cls.cpu.return_value.tolist.return_value = [2.0]
    return SimpleNamespace(boxes=boxes)


class Backend:
    """Delegate codecs/drawing to real OpenCV, intercept only GUI and resource handles."""

    def __init__(self, keys: list[int] | None = None, gui_error: bool = False) -> None:
        self.captures = []
        self.writers = []
        self.shown = []
        self.destroy_count = 0
        self.keys = iter(keys or [-1] * 100)
        self.gui_error = gui_error

    def __getattr__(self, name):
        return getattr(cv2, name)

    def VideoCapture(self, path):
        cap = cv2.VideoCapture(path)
        self.captures.append(cap)
        return cap

    def VideoWriter(self, *args):
        writer = cv2.VideoWriter(*args)
        self.writers.append(writer)
        return writer

    def imshow(self, name, frame):
        if self.gui_error:
            raise cv2.error("synthetic GUI unavailable")
        self.shown.append(frame.copy())

    def waitKey(self, delay):
        return next(self.keys, -1)

    def destroyAllWindows(self):
        self.destroy_count += 1


class VideoDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = demo.load_video_config(ROOT)

    def invoke(self, root: Path, source: Path, *, backend: Backend | None = None,
               model: MagicMock | None = None, **kwargs):
        model = model if model is not None else MagicMock()
        model.names = {0: "D00_longitudinal_crack", 1: "D10_transverse_crack",
                       2: "D20_alligator_crack", 3: "D40_pothole"}
        if model.predict.side_effect is None:
            model.predict.return_value = [prediction()]
        factory = MagicMock(return_value=model)
        backend = backend or Backend()
        with patch.object(demo, "load_video_config", return_value=self.config), \
                patch.object(demo, "verify_video_model", return_value=(root / "best.pt", demo.FROZEN_MODEL_SHA256)), \
                patch.object(demo, "check_cuda_device"):
            summary = demo.run_video(source, root=root, yolo_factory=factory, cv2_module=backend, **kwargs)
        return summary, model, factory, backend

    def test_frozen_model_mapping_and_default_demo_thresholds(self) -> None:
        self.assertEqual(self.config.model_path.as_posix(),
                         "outputs/training/baseline_public_v1/20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42/weights/best.pt")
        self.assertEqual((self.config.confidence, self.config.nms_iou), (.19, .50))
        self.assertEqual(self.config.operating_point_status, "frozen_validation_selected")
        self.assertIn("frozen_validation_selected", demo.FROZEN_OPERATING_POINT_NOTICE)
        self.assertIn("frozen_validation_selected", demo.WINDOW_NAME)
        self.assertEqual(demo.FROZEN_MODEL_SIZE_BYTES, 22524074)
        self.assertEqual(demo.FROZEN_MODEL_SHA256,
                         "BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7")

    def test_config_rejects_provisional_thresholds_and_status(self) -> None:
        original = json.loads((ROOT / demo.CONFIG_PATH).read_text(encoding="utf-8"))
        for field, value in (("confidence", .25), ("nms_iou", .70),
                             ("operating_point_status", "provisional_demo")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / demo.CONFIG_PATH
                path.parent.mkdir(parents=True)
                changed = json.loads(json.dumps(original))
                if field == "operating_point_status":
                    changed[field] = value
                else:
                    changed["inference"][field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(demo.VideoDemoError):
                    demo.load_video_config(root)

    def test_model_path_and_sha_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for model in ("last.pt", "other/best.pt", "yolo26s.pt"):
                with self.assertRaises(demo.VideoDemoError):
                    demo.verify_video_model(replace(self.config, model_path=Path(model)), root)
            file = root / "synthetic.pt"
            file.write_bytes(b"synthetic-frozen-bytes")
            sha = hashlib.sha256(file.read_bytes()).hexdigest()
            demo.verify_model_identity(file, expected_sha256=sha, expected_size_bytes=file.stat().st_size)
            with self.assertRaises(demo.MentorDemoError):
                demo.verify_model_identity(file, expected_sha256="0" * 64, expected_size_bytes=file.stat().st_size)

    def test_source_suffix_missing_directory_and_internal_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix in (".avi", ".MP4", ".mov", ".mkv"):
                source = root / f"source{suffix}"
                source.write_bytes(b"fixture")
                self.assertEqual(demo.validate_video_source(source, root), source)
            for path in (root / "missing.avi", root / "last.pt", root):
                with self.assertRaises(demo.VideoDemoError):
                    demo.validate_video_source(path, root)
            for path in (root / "data/exports/any/images/test/private.avi", root / "test/private.mp4",
                         root / "data/raw/official_test/clip.avi"):
                with patch.object(Path, "is_file", side_effect=AssertionError("must reject before file access")):
                    with self.assertRaises(demo.VideoDemoError):
                        demo.validate_video_source(path, root)

    def test_outputs_unique_and_never_overlap_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_video(root)
            before = source.read_bytes()
            a, b = demo.reserve_outputs(source, root), demo.reserve_outputs(source, root)
            self.assertNotEqual(a.run_dir, b.run_dir)
            self.assertTrue(a.run_dir.is_relative_to(root / "outputs/demo/video"))
            self.assertNotEqual(source, a.run_dir / "fixture_detected.avi")
            self.assertEqual(source.read_bytes(), before)

    def test_timestamp_and_detection_csv_serialization(self) -> None:
        self.assertEqual(demo.timestamp_ms(0, 25), 0)
        self.assertEqual(demo.timestamp_ms(10, 25), 400)
        self.assertEqual(demo.timestamp_ms(1, 29.97), 33)
        item = demo.Detection(3, "D40_pothole", .8, (1, 2, 30, 40))
        row = demo.detection_rows(10, 25, [item])[0]
        self.assertEqual(tuple(row), ("frame_index", "timestamp_seconds", "class_id", "class_name",
                                     "confidence", "x1", "y1", "x2", "y2"))
        self.assertEqual(row["timestamp_seconds"], "0.400")
        self.assertEqual(row["class_id"], 3)
        self.assertEqual(row["x2"], 30)
        with self.assertRaises(demo.VideoDemoError):
            demo.timestamp_ms(0, 0)

    def test_full_synthetic_video_and_summary_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_video(root)
            source_hash = demo.sha256_file(source)
            summary, model, factory, backend = self.invoke(root, source)
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["partial_output"])
            self.assertEqual(summary["processed_frames"], 6)
            self.assertEqual(summary["total_detections"], 6)
            self.assertEqual(summary["frames_with_detections"], 6)
            self.assertEqual(summary["counts_by_class"]["D20_alligator_crack"], 6)
            self.assertEqual(summary["input_resolution"], {"width": 320, "height": 240})
            self.assertAlmostEqual(summary["input_fps"], 12.5)
            self.assertEqual(summary["duration_seconds"], .48)
            self.assertTrue(summary["encoding_verification"]["passed"])
            self.assertEqual(summary["source_sha256"], source_hash)
            self.assertEqual(demo.sha256_file(source), source_hash)
            self.assertIn("not unique physical damage events", summary["scientific_limitations"])
            self.assertEqual(summary["operating_point_status"], "frozen_validation_selected")
            self.assertIn(
                "Final frozen validation-selected",
                summary["frozen_operating_point_declaration"],
            )
            self.assertIn(
                "completed internal test did not alter",
                summary["frozen_operating_point_declaration"],
            )
            self.assertEqual(model.predict.call_count, 6)
            factory.assert_called_once()
            for call in model.predict.call_args_list:
                self.assertEqual((call.kwargs["conf"], call.kwargs["iou"], call.kwargs["imgsz"], call.kwargs["device"]), (.19, .5, 640, 0))
                self.assertFalse(call.kwargs["augment"])
            with Path(summary["csv_path"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[-1]["frame_index"], "5")
            self.assertEqual(rows[-1]["timestamp_seconds"], "0.400")
            for capture in backend.captures:
                self.assertFalse(capture.isOpened())
            for writer in backend.writers:
                self.assertFalse(writer.isOpened())

    def test_max_frames_and_same_inference_for_live_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_video(root)
            summary, model, factory, backend = self.invoke(root, source, display=True, max_frames=3)
            self.assertEqual(summary["stop_reason"], "max_frames")
            self.assertTrue(summary["partial_output"])
            self.assertEqual(summary["processed_frames"], 3)
            self.assertEqual(model.predict.call_count, 3)
            self.assertEqual(len(backend.shown), 3)
            self.assertEqual(backend.destroy_count, 1)
            self.assertEqual(summary["encoding_verification"]["encoded_frames"], 3)
            # Intercept the encoder input too and compare it with the preview pixel-for-pixel.
            backend2 = Backend()
            real_constructor = backend2.VideoWriter
            encoded = []
            def recording_writer(*args):
                writer = real_constructor(*args)
                return SimpleNamespace(isOpened=writer.isOpened, release=writer.release,
                                       write=lambda frame: (encoded.append(frame.copy()), writer.write(frame)))
            backend2.VideoWriter = recording_writer
            self.invoke(root, source, backend=backend2, display=True, max_frames=2)
            self.assertEqual(len(encoded), 2)
            for saved, preview in zip(encoded, backend2.shown, strict=True):
                np.testing.assert_array_equal(saved, preview)

    def test_preview_resize_does_not_resize_saved_frame(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        preview = demo.preview_frame(frame, self.config, cv2)
        self.assertEqual(frame.shape, (1080, 1920, 3))
        self.assertEqual(preview.shape, (720, 1280, 3))

    def test_gui_failure_continues_all_frames_and_saves_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = Backend(gui_error=True)
            with self.assertLogs(demo.LOGGER, level="WARNING") as logs:
                summary, model, _, _ = self.invoke(root, make_video(root), backend=backend, display=True)
            self.assertEqual(summary["processed_frames"], 6)
            self.assertEqual(model.predict.call_count, 6)
            self.assertTrue(summary["display_failed"])
            self.assertTrue(summary["encoding_verification"]["passed"])
            self.assertIn("continuing to save", " ".join(logs.output))
            self.assertEqual(backend.destroy_count, 1)

    def test_q_and_escape_stop_cleanly(self) -> None:
        for key in (ord("q"), 27):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                summary, model, _, backend = self.invoke(root, make_video(root),
                                                         backend=Backend(keys=[key]), display=True)
                self.assertEqual(summary["stop_reason"], "preview_quit")
                self.assertEqual(summary["processed_frames"], 1)
                self.assertEqual(model.predict.call_count, 1)
                self.assertTrue(summary["partial_output"])
                self.assertTrue(all(not cap.isOpened() for cap in backend.captures))
                self.assertTrue(all(not writer.isOpened() for writer in backend.writers))

    def test_ctrl_c_event_finishes_current_frame_and_releases_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop = threading.Event()
            model = MagicMock()
            def stop_during_predict(**kwargs):
                stop.set()
                return [prediction()]
            model.predict.side_effect = stop_during_predict
            summary, _, _, backend = self.invoke(root, make_video(root), model=model, stop_event=stop, display=True)
            self.assertEqual(summary["stop_reason"], "ctrl_c")
            self.assertEqual(summary["processed_frames"], 1)
            self.assertTrue(summary["encoding_verification"]["passed"])
            self.assertTrue(all(not cap.isOpened() for cap in backend.captures))
            self.assertTrue(all(not writer.isOpened() for writer in backend.writers))
            self.assertEqual(backend.destroy_count, 1)

    def test_keyboard_interrupt_and_model_error_release_all_resources(self) -> None:
        for error in (KeyboardInterrupt(), RuntimeError("synthetic model error")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                backend = Backend()
                model = MagicMock()
                model.predict.side_effect = [[prediction()], error]
                if isinstance(error, KeyboardInterrupt):
                    summary, _, _, _ = self.invoke(root, make_video(root), model=model, backend=backend, display=True)
                    self.assertEqual(summary["stop_reason"], "ctrl_c")
                else:
                    with self.assertRaises(demo.VideoDemoError):
                        self.invoke(root, make_video(root), model=model, backend=backend, display=True)
                self.assertTrue(all(not cap.isOpened() for cap in backend.captures))
                self.assertTrue(all(not writer.isOpened() for writer in backend.writers))
                self.assertEqual(backend.destroy_count, 1)
                summary = json.loads(next((root / "outputs/demo/video").glob("*/*_summary.json")).read_text())
                self.assertTrue(summary["partial_output"])
                self.assertEqual(summary["processed_frames"], 1)

    def test_bad_metadata_and_unreadable_video_release_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_video(root)
            for prop, value in ((cv2.CAP_PROP_FPS, 0.), (cv2.CAP_PROP_FRAME_WIDTH, 0.),
                                (cv2.CAP_PROP_FRAME_HEIGHT, 241.)):
                backend = Backend()
                cap = MagicMock()
                cap.isOpened.return_value = True
                values = {cv2.CAP_PROP_FPS: 12.5, cv2.CAP_PROP_FRAME_WIDTH: 320.,
                          cv2.CAP_PROP_FRAME_HEIGHT: 240., cv2.CAP_PROP_FRAME_COUNT: 6., prop: value}
                cap.get.side_effect = lambda key: values[key]
                backend.VideoCapture = lambda path: cap
                with self.assertRaises(demo.VideoDemoError):
                    self.invoke(root, source, backend=backend)
                cap.release.assert_called_once()
            bad = root / "bad.avi"
            bad.write_bytes(b"not a video")
            backend = Backend()
            with self.assertRaises(demo.VideoDemoError):
                self.invoke(root, bad, backend=backend)
            self.assertTrue(all(not cap.isOpened() for cap in backend.captures))

    def test_codec_fallback_releases_failed_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = demo.reserve_outputs(root / "input.avi", root)
            backend = Backend()
            bad = MagicMock()
            bad.isOpened.return_value = False
            good = MagicMock()
            good.isOpened.return_value = True
            backend.VideoWriter = MagicMock(side_effect=[bad, good])
            writer, path, codec = demo.open_writer(paths, demo.VideoInfo(320, 240, 25., 10), self.config, backend)
            self.assertIs(writer, good)
            self.assertEqual((path.suffix, codec), (".avi", "MJPG"))
            bad.release.assert_called_once()

    def test_cli_ctrl_c_handler_and_exit_code(self) -> None:
        def run(source, **kwargs):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            self.assertTrue(kwargs["stop_event"].is_set())
            return {"status": "interrupted", "stop_reason": "ctrl_c"}
        previous = signal.getsignal(signal.SIGINT)
        with patch.object(demo, "run_video", side_effect=run):
            self.assertEqual(demo.main(["--source", "synthetic.avi", "--display", "--max-frames", "10"]), 130)
        self.assertIs(signal.getsignal(signal.SIGINT), previous)


if __name__ == "__main__":
    unittest.main()
