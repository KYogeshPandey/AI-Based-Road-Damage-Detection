"""Focused tests for deterministic Phase 3B smoke-dataset preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.dataset.rdd2022_common import RDD2022Error  # noqa: E402
from road_damage.training.create_smoke_subset import (  # noqa: E402
    ALLOWED_CLASS_IDS,
    run_smoke_subset,
)


def _write_config(path: Path, source: Path, output: Path) -> None:
    value = {
        "schema_version": "3B.smoke_subset.v1",
        "seed": 42,
        "source_dataset_path": str(source),
        "output_path": str(output),
        "splits": {
            "train": {"positive": 80, "negative": 48},
            "val": {"positive": 40, "negative": 24},
        },
        "countries": ["India", "Japan"],
        "taxonomy": [
            {"id": 0, "name": "D00_longitudinal_crack"},
            {"id": 1, "name": "D10_transverse_crack"},
            {"id": 2, "name": "D20_alligator_crack"},
            {"id": 3, "name": "D40_pothole"},
        ],
        "minimum_india_d10_positive_images_per_split": 2,
        "copy_policy": "byte_identical_copy_no_hardlinks",
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_source(root: Path) -> Path:
    source = root / "source"
    specifications = {"train": (96, 60), "val": (52, 30)}
    for split, (positive_count, negative_count) in specifications.items():
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(positive_count):
            country = "India" if index < positive_count // 2 else "Japan"
            stem = f"{country}_{split}_positive_{index:04d}"
            (image_dir / f"{stem}.jpg").write_bytes(f"image:{split}:{index}".encode())
            class_id = index % 4
            (label_dir / f"{stem}.txt").write_bytes(
                f"{class_id} 0.50000000 0.50000000 0.10000000 0.10000000\n".encode()
            )
        for index in range(negative_count):
            country = "India" if index % 2 == 0 else "Japan"
            stem = f"{country}_{split}_negative_{index:04d}"
            (image_dir / f"{stem}.jpg").write_bytes(f"negative:{split}:{index}".encode())
            (label_dir / f"{stem}.txt").write_bytes(b"")
    validation = {
        "phase2c1_complete": True,
        "status": "passed",
        "exclusions": {
            "intersections": {"official_unlabelled_test": []},
            "teacher_reference_count": 0,
            "passed": True,
        },
    }
    (source / "validation_report.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )
    return source


class SmokeSubsetTests(unittest.TestCase):
    def test_selection_is_deterministic_and_required_counts_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _make_source(root)
            first = root / "smoke-a"
            second = root / "smoke-b"
            first_config = root / "first.json"
            second_config = root / "second.json"
            _write_config(first_config, source, first)
            _write_config(second_config, source, second)

            run_smoke_subset(first_config, project_root=root)
            run_smoke_subset(second_config, project_root=root)
            first_manifest = json.loads((first / "smoke_manifest.json").read_text())
            second_manifest = json.loads((second / "smoke_manifest.json").read_text())
            self.assertEqual(
                first_manifest["selection_sha256"], second_manifest["selection_sha256"]
            )
            self.assertEqual(
                [(row["smoke_split"], row["filename"]) for row in first_manifest["samples"]],
                [(row["smoke_split"], row["filename"]) for row in second_manifest["samples"]],
            )

            report = json.loads((first / "smoke_validation_report.json").read_text())
            self.assertTrue(report["passed"])
            for split, positive, negative in (("train", 80, 48), ("val", 40, 24)):
                split_report = report["splits"][split]
                self.assertEqual(split_report["images"], positive + negative)
                self.assertEqual(split_report["labels"], positive + negative)
                self.assertEqual(split_report["positive"], positive)
                self.assertEqual(split_report["negative"], negative)
                self.assertGreater(split_report["country_counts"]["India"], 0)
                self.assertGreater(split_report["country_counts"]["Japan"], 0)
                self.assertEqual(
                    {int(key) for key, value in split_report["class_image_occurrence_counts"].items() if value},
                    set(ALLOWED_CLASS_IDS),
                )
                self.assertGreaterEqual(split_report["india_d10_positive_image_count"], 2)
            self.assertTrue(report["hash_verification"]["byte_identical_to_source"])
            self.assertTrue(report["hash_verification"]["no_hardlinks"])
            self.assertNotIn("test:", (first / "data.yaml").read_text())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _make_source(root)
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("untouched", encoding="utf-8")
            config = root / "config.json"
            _write_config(config, source, output)
            with self.assertRaises(RDD2022Error):
                run_smoke_subset(config, project_root=root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
