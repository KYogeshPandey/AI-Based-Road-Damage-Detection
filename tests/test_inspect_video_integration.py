"""Small integration test for inspection output generation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from road_damage.video.inspect_video import inspect_video  # noqa: E402


@unittest.skipIf(cv2 is None or np is None, "OpenCV/NumPy not available")
class VideoInspectionIntegrationTests(unittest.TestCase):
    def test_inspects_small_synthetic_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "fixture.avi"
            output_dir = root / "inspection"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            try:
                for frame_index in range(30):
                    frame = np.full(
                        (48, 64, 3), frame_index * 8, dtype=np.uint8
                    )
                    cv2.line(frame, (0, frame_index % 48), (63, 47), (255, 255, 255))
                    writer.write(frame)
            finally:
                writer.release()

            report_path = inspect_video(
                video_path,
                output_dir,
                3,
                {
                    "width": 64,
                    "height": 48,
                    "fps": 10.0,
                    "frame_count": 30,
                    "duration_seconds": 3.0,
                    "container": "AVI",
                },
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["sampling"]["extracted_sample_count"], 3)
            self.assertEqual(report["video"]["resolution"], "64x48")
            self.assertTrue(
                report["source"]["immutable_source_check"][
                    "size_and_mtime_unchanged_during_run"
                ]
            )
            self.assertEqual(len(list(output_dir.glob("*.jpg"))), 3)


if __name__ == "__main__":
    unittest.main()
