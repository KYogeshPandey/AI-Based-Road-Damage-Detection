"""Unit tests for the Phase 1.5 reconnaissance helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.video.reconnaissance import (  # noqa: E402
    ReconnaissanceError,
    expected_contact_sheet_count,
    interval_frame_indices,
    load_config,
    summarize_reconnaissance_quality,
    validate_config,
)


class IntervalSamplingTests(unittest.TestCase):
    def test_two_second_sampling_matches_verified_video(self) -> None:
        indices = interval_frame_indices(93_801, 25.0, 2.0, 1.0)

        self.assertEqual(len(indices), 1_876)
        self.assertEqual(indices[0], 25)
        self.assertEqual(indices[-1], 93_775)
        self.assertTrue(all(right - left == 50 for left, right in zip(indices, indices[1:])))

    def test_contact_sheet_count(self) -> None:
        self.assertEqual(expected_contact_sheet_count(1_876, 6, 5), 63)


class ConfigurationTests(unittest.TestCase):
    def test_project_configuration_is_valid(self) -> None:
        config = load_config(Path("configs/video/reconnaissance.yaml"))

        self.assertEqual(config["thumbnail"]["width"], 360)
        self.assertEqual(len(config["overlay_exclusions"]), 4)
        self.assertEqual(
            config["valid_region"]["name"],
            "provisional_conservative_highway_road_surface",
        )

    def test_rejects_invalid_normalized_rectangle(self) -> None:
        config = {
            "sampling": {"interval_seconds": 2.0, "edge_margin_seconds": 1.0},
            "thumbnail": {"width": 360, "height": 288, "jpeg_quality": 82},
            "contact_sheet": {
                "columns": 6,
                "rows": 5,
                "label_height": 28,
                "jpeg_quality": 85,
            },
            "valid_region": {
                "polygon_normalized": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            },
            "overlay_exclusions": [
                {"name": "bad", "xyxy_normalized": [0.8, 0.2, 0.1, 0.9]}
            ],
        }

        with self.assertRaises(ReconnaissanceError):
            validate_config(config)

    def test_loads_json_compatible_yaml_without_extra_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            project_config = load_config(Path("configs/video/reconnaissance.yaml"))
            config_path.write_text(json.dumps(project_config), encoding="utf-8")

            loaded = load_config(config_path)

            self.assertEqual(loaded, project_config)


class QualitySummaryTests(unittest.TestCase):
    def test_ignores_failed_candidates(self) -> None:
        candidates = [
            {
                "full_frame_brightness": 100.0,
                "full_frame_sharpness_laplacian_variance": 200.0,
                "valid_region_brightness": 90.0,
                "valid_region_sharpness_laplacian_variance": 180.0,
                "previous_valid_region_mean_absolute_difference": None,
            },
            {
                "full_frame_brightness": None,
                "full_frame_sharpness_laplacian_variance": None,
                "valid_region_brightness": None,
                "valid_region_sharpness_laplacian_variance": None,
                "previous_valid_region_mean_absolute_difference": None,
            },
        ]

        summary = summarize_reconnaissance_quality(candidates)

        self.assertEqual(summary["full_frame_brightness"]["count"], 1)
        self.assertEqual(
            summary["previous_valid_region_mean_absolute_difference"]["count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
