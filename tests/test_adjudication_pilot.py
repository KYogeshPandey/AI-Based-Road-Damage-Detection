"""Focused tests for Phase 1.6 adjudication-pilot selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.dataset.adjudication_pilot import (  # noqa: E402
    _interval_seconds,
    generate_candidate_indices,
    load_pilot_config,
    parse_timestamp,
    select_representatives,
    selection_specs,
    timestamp_to_frame_index,
)
from road_damage.video.reconnaissance import load_config as load_roi_config  # noqa: E402


class TimestampTests(unittest.TestCase):
    def test_parses_hour_timestamp_and_maps_to_frame(self) -> None:
        seconds = parse_timestamp("01:01:35")

        self.assertEqual(seconds, 3_695.0)
        self.assertEqual(timestamp_to_frame_index(seconds, 25.0), 92_375)

    def test_timestamp_frame_round_trip(self) -> None:
        frame_index = timestamp_to_frame_index(parse_timestamp("00:04:19.20"), 25.0)

        self.assertEqual(frame_index, 6_480)
        self.assertAlmostEqual(frame_index / 25.0, 259.2)


class ConfigurationTests(unittest.TestCase):
    def test_selection_quotas_are_balanced_and_total_180(self) -> None:
        config = load_pilot_config(Path("configs/video/adjudication_pilot.yaml"))
        specs = selection_specs(config)
        counts = {
            source_type: sum(
                int(spec["quota"])
                for spec in specs
                if spec["type"] == source_type
            )
            for source_type in (
                "candidate_interval",
                "normal_negative",
                "hard_negative",
            )
        }

        self.assertEqual(counts["candidate_interval"], 130)
        self.assertEqual(counts["normal_negative"], 25)
        self.assertEqual(counts["hard_negative"], 25)
        self.assertEqual(sum(counts.values()), 180)

    def test_conservative_roi_and_all_four_overlays_are_configured(self) -> None:
        config = load_roi_config(Path("configs/video/reconnaissance.yaml"))
        exclusion_names = {
            item["name"] for item in config["overlay_exclusions"]
        }

        self.assertEqual(
            config["valid_region"]["polygon_normalized"],
            [[0.31, 0.38], [0.69, 0.38], [0.94, 0.98], [0.06, 0.98]],
        )
        self.assertEqual(
            exclusion_names,
            {
                "csir_crri_logo_upper_left",
                "driver_camera_inset_lower_center",
                "circular_gauge_lower_right",
                "sats_telemetry_text_lower_right",
            },
        )
        telemetry = next(
            item
            for item in config["overlay_exclusions"]
            if item["name"] == "sats_telemetry_text_lower_right"
        )
        self.assertLessEqual(telemetry["xyxy_normalized"][1], 0.49)


class IntervalSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_pilot_config(Path("configs/video/adjudication_pilot.yaml"))
        self.avoid_ranges = [
            _interval_seconds(interval)
            for interval in self.config["avoid_intervals"]
        ]

    def test_candidate_pool_is_sampled_at_five_fps(self) -> None:
        spec = selection_specs(self.config)[0]
        indices = generate_candidate_indices(
            spec, 25.0, 93_801, self.avoid_ranges
        )

        self.assertEqual(indices[0], 6_450)
        self.assertEqual(indices[-1], 6_600)
        self.assertTrue(
            all(right - left == 5 for left, right in zip(indices, indices[1:]))
        )

    def test_all_pool_frames_stay_in_sampling_windows_and_out_of_avoid_ranges(self) -> None:
        for spec in selection_specs(self.config):
            interval_start, interval_end = _interval_seconds(spec)
            window_start = max(
                0.0, interval_start - float(spec["context_before_seconds"])
            )
            window_end = interval_end + float(spec["context_after_seconds"])
            indices = generate_candidate_indices(
                spec, 25.0, 93_801, self.avoid_ranges
            )

            self.assertGreaterEqual(len(indices), int(spec["quota"]), spec["id"])
            for index in indices:
                seconds = index / 25.0
                self.assertGreaterEqual(seconds, window_start)
                self.assertLessEqual(seconds, window_end)
                self.assertFalse(
                    any(start <= seconds <= end for start, end in self.avoid_ranges)
                )


class DeterministicSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_temporally_distributed(self) -> None:
        candidates = [
            {
                "requested_frame_index": index,
                "road_roi_sharpness_laplacian_variance": float((index * 7) % 11),
                "road_visibility_contrast_stddev": float((index * 5) % 13),
                "previous_road_roi_mean_absolute_difference": float((index * 3) % 17),
            }
            for index in range(40)
        ]
        weights = {
            "roi_sharpness": 0.4,
            "road_visibility_contrast": 0.3,
            "temporal_change": 0.3,
        }

        first = select_representatives(candidates, 8, weights)
        second = select_representatives(candidates, 8, weights)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(
            [item["selection"]["temporal_bin_one_based"] for item in first],
            list(range(1, 9)),
        )
        for bin_index, item in enumerate(first):
            self.assertGreaterEqual(item["requested_frame_index"], bin_index * 5)
            self.assertLess(item["requested_frame_index"], (bin_index + 1) * 5)


if __name__ == "__main__":
    unittest.main()
