"""Unit tests for deterministic video-inspection helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.video.inspect_video import (  # noqa: E402
    VideoInspectionError,
    build_metadata_validation,
    distributed_frame_indices,
    format_timestamp,
    summarize_quality,
)


class TimestampTests(unittest.TestCase):
    def test_formats_timestamp_with_milliseconds(self) -> None:
        self.assertEqual(format_timestamp(3751.0), "01:02:31.000")
        self.assertEqual(format_timestamp(1.2346), "00:00:01.235")

    def test_rejects_negative_timestamp(self) -> None:
        with self.assertRaises(ValueError):
            format_timestamp(-0.001)


class SamplingTests(unittest.TestCase):
    def test_indices_span_video_without_using_boundaries(self) -> None:
        indices = distributed_frame_indices(93_801, 25.0, 12)

        self.assertEqual(len(indices), 12)
        self.assertEqual(indices[0], 25)
        self.assertEqual(indices[-1], 93_775)
        self.assertEqual(indices, sorted(set(indices)))

        gaps = [right - left for left, right in zip(indices, indices[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_rejects_too_many_non_boundary_samples(self) -> None:
        with self.assertRaises(VideoInspectionError):
            distributed_frame_indices(5, 25.0, 4)


class MetadataValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detected = {
            "width": 720,
            "height": 576,
            "fps": 25.0,
            "frame_count": 93_801,
            "duration_seconds": 3752.04,
            "container": "AVI",
            "codec_fourcc": "FMP4",
            "codec_description": "MPEG-4 Part 2",
        }

    def test_matching_reference_metadata(self) -> None:
        expected = {
            "width": 720,
            "height": 576,
            "fps": 25.0,
            "frame_count": 93_801,
            "duration_seconds": 3752.04,
            "container": "AVI",
            "codec": "MPEG-4 Part 2",
            "pixel_format": "yuv420p",
        }

        result = build_metadata_validation(self.detected, expected)

        self.assertTrue(result["all_verifiable_fields_match"])
        self.assertEqual(result["discrepancies"], [])
        self.assertEqual(
            result["checks"]["pixel_format"]["status"],
            "not_verifiable_with_opencv",
        )

    def test_reports_meaningful_discrepancy(self) -> None:
        result = build_metadata_validation(
            self.detected, {"frame_count": 93_800, "fps": 24.0}
        )

        self.assertFalse(result["all_verifiable_fields_match"])
        self.assertEqual(result["checks"]["frame_count"]["difference"], 1)
        self.assertEqual(len(result["discrepancies"]), 2)


class QualitySummaryTests(unittest.TestCase):
    def test_summarizes_sample_metrics(self) -> None:
        samples = [
            {
                "brightness_mean_intensity": 10.0,
                "sharpness_laplacian_variance": 100.0,
            },
            {
                "brightness_mean_intensity": 30.0,
                "sharpness_laplacian_variance": 300.0,
            },
        ]

        result = summarize_quality(samples)

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["brightness_mean_intensity"]["mean"], 20.0)
        self.assertEqual(result["sharpness_laplacian_variance"]["median"], 200.0)


if __name__ == "__main__":
    unittest.main()

