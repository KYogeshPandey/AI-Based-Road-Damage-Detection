"""Focused tests for the Phase 2C.1 RDD2022 YOLO detection export."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.dataset.rdd2022_common import RDD2022Error  # noqa: E402
from road_damage.dataset.rdd2022_yolo_export import (  # noqa: E402
    CLASS_MAPPING,
    _content_tree_fingerprint,
    canonical_xyxy_pixel_to_yolo,
    format_yolo_label_line,
    run_export,
)
from road_damage.dataset.rdd2022_yolo_validate import validate_export  # noqa: E402


TAXONOMY = [
    {"id": 0, "raw_code": "D00", "name": "D00_longitudinal_crack"},
    {"id": 1, "raw_code": "D10", "name": "D10_transverse_crack"},
    {"id": 2, "raw_code": "D20", "name": "D20_alligator_crack"},
    {"id": 3, "raw_code": "D40", "name": "D40_pothole"},
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _make_fixture(root: Path) -> tuple[Path, Path, Path]:
    canonical = root / "canonical"
    raw = root / "raw"
    parent = root / "parent"
    parent.mkdir()
    raw.mkdir()
    source_specs = [
        ("train-1", "India", "India_000001.jpg", "train", True, b"train-image"),
        ("val-1", "Japan", "Japan_000001.jpg", "val", False, b"val-image"),
        ("test-1", "India", "India_000002.jpg", "test", False, b"test-image"),
    ]
    images: list[dict] = []
    assignments: list[dict] = []
    for canonical_id, country, filename, split, positive, content in source_specs:
        source = raw / filename
        source.write_bytes(content)
        images.append(
            {
                "canonical_image_id": canonical_id,
                "country": country,
                "original_filename": filename,
                "raw_relative_image_path": str(source),
                "image_sha256": _sha256(source),
                "decoded_width": 100,
                "decoded_height": 100,
                "derived_split": split,
                "inclusion_status": "split_assigned",
                "is_positive": positive,
                "image_category": (
                    "mixed_target_and_known_non_target" if positive else "empty_xml"
                ),
            }
        )
        assignments.append(
            {
                "canonical_image_id": canonical_id,
                "country": country,
                "original_filename": filename,
                "derived_split": split,
                "inclusion_status": "split_assigned",
            }
        )
    images.extend(
        [
            {
                "canonical_image_id": "guard-1",
                "original_filename": "India_088888.jpg",
                "inclusion_status": "guard_excluded",
                "derived_split": None,
            },
            {
                "canonical_image_id": "quarantine-1",
                "original_filename": "India_099999.jpg",
                "inclusion_status": "quarantined",
                "derived_split": None,
            },
        ]
    )
    assignments.extend(
        [
            {
                "canonical_image_id": "guard-1",
                "original_filename": "India_088888.jpg",
                "inclusion_status": "guard_excluded",
                "derived_split": None,
            },
            {
                "canonical_image_id": "quarantine-1",
                "original_filename": "India_099999.jpg",
                "inclusion_status": "quarantined",
                "derived_split": None,
            },
        ]
    )
    objects = [
        {
            "canonical_image_id": "train-1",
            "canonical_object_id": "train-1:object:0001",
            "object_index_one_based": 1,
            "original_raw_class_string": "D00",
            "canonical_numeric_class": 0,
            "canonical_xyxy_pixel_zero_based_half_open": [0, 0, 10, 20],
            "source_zero_edge_coordinate": True,
            "final_target_disposition": "retained",
        },
        {
            "canonical_image_id": "train-1",
            "canonical_object_id": "train-1:object:0002",
            "object_index_one_based": 2,
            "original_raw_class_string": "D01",
            "canonical_numeric_class": None,
            "canonical_xyxy_pixel_zero_based_half_open": [20, 20, 30, 30],
            "source_zero_edge_coordinate": False,
            "final_target_disposition": None,
        },
    ]
    _write_json(canonical / "dataset_card.json", {"dataset_version": "1.1.0", "taxonomy": TAXONOMY})
    _write_jsonl(canonical / "manifests" / "images.jsonl", images)
    _write_jsonl(canonical / "manifests" / "objects.jsonl", objects)
    _write_jsonl(canonical / "manifests" / "quarantine.jsonl", [{"original_filename": "India_099999.jpg"}])
    _write_jsonl(canonical / "manifests" / "official_unlabelled_test.jsonl", [{"original_filename": "Japan_test_1.jpg"}])
    _write_jsonl(canonical / "manifests" / "rejected_objects.jsonl", [])
    _write_jsonl(canonical / "splits" / "assignments.jsonl", assignments)
    _write_jsonl(canonical / "splits" / "guard_exclusions.jsonl", [{"original_filename": "India_088888.jpg"}])

    config = {
        "schema_version": "2C.1",
        "export_version": "test-1",
        "canonical_dataset_path": str(canonical),
        "canonical_dataset_version": "1.1.0",
        "parent_dataset_path": str(parent),
        "raw_dataset_path": str(raw),
        "output_path": str(root / "unused-output"),
        "taxonomy": TAXONOMY,
        "split_names": ["train", "val", "test"],
        "label_decimal_places": 8,
        "image_copy_policy": "byte_identical_copy_no_recompression_no_hardlinks",
        "copy_workers": 2,
        "label_write_workers": 4,
        "verify_canonical_artifact_checksums": False,
        "expected": {
            "split_image_counts": {"train": 1, "val": 1, "test": 1},
            "positive_images": 1,
            "negative_images": {"train": 0, "val": 1, "test": 1},
            "target_objects": {
                "train": {"D00": 1, "D10": 0, "D20": 0, "D40": 0},
                "val": {"D00": 0, "D10": 0, "D20": 0, "D40": 0},
                "test": {"D00": 0, "D10": 0, "D20": 0, "D40": 0},
            },
            "total_target_objects": {"D00": 1, "D10": 0, "D20": 0, "D40": 0},
            "total_target_object_count": 1,
            "official_unlabelled_test_images": 1,
            "negative_image_count": 2,
        },
        "visual_qa": {"enabled": False},
    }
    config_path = root / "config.json"
    _write_json(config_path, config)
    return config_path, canonical, raw


class ConversionTests(unittest.TestCase):
    def test_class_mapping_is_exact_and_ordered(self) -> None:
        self.assertEqual(
            list(CLASS_MAPPING.items()),
            [
                ("D00", (0, "D00_longitudinal_crack")),
                ("D10", (1, "D10_transverse_crack")),
                ("D20", (2, "D20_alligator_crack")),
                ("D40", (3, "D40_pothole")),
            ],
        )

    def test_zero_edge_canonical_box_converts_without_shift(self) -> None:
        self.assertEqual(
            canonical_xyxy_pixel_to_yolo([0, 0, 10, 20], 100, 200),
            (0.05, 0.05, 0.1, 0.1),
        )

    def test_conversion_rejects_invalid_boxes(self) -> None:
        with self.assertRaises(RDD2022Error):
            canonical_xyxy_pixel_to_yolo([-1, 0, 10, 20], 100, 100)
        with self.assertRaises(RDD2022Error):
            canonical_xyxy_pixel_to_yolo([1, 1, 1, 20], 100, 100)

    def test_label_has_fixed_precision(self) -> None:
        line = format_yolo_label_line(3, [0, 0, 10, 20], 100, 100, 8)
        self.assertEqual(line, "3 0.05000000 0.10000000 0.10000000 0.20000000")
        self.assertTrue(all(len(value.split(".")[1]) == 8 for value in line.split()[1:]))

    def test_unknown_class_id_is_rejected(self) -> None:
        with self.assertRaises(RDD2022Error):
            format_yolo_label_line(4, [0, 0, 10, 20], 100, 100, 8)


class SyntheticExportIntegrationTests(unittest.TestCase):
    def test_export_pairs_every_image_and_writes_empty_negative_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, canonical, _ = _make_fixture(root)
            output = run_export(config, canonical, root / "export", root)
            self.assertEqual(len(list((output / "images").rglob("*.jpg"))), 3)
            self.assertEqual(len(list((output / "labels").rglob("*.txt"))), 3)
            self.assertEqual(
                (output / "labels" / "train" / "India_000001.txt").read_text(),
                "0 0.05000000 0.10000000 0.10000000 0.20000000\n",
            )
            self.assertEqual(
                (output / "labels" / "val" / "Japan_000001.txt").read_bytes(), b""
            )
            self.assertEqual(
                (output / "labels" / "test" / "India_000002.txt").read_bytes(), b""
            )

    def test_non_target_objects_and_excluded_images_do_not_enter_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, canonical, _ = _make_fixture(root)
            output = run_export(config, canonical, root / "export", root)
            label = (output / "labels" / "train" / "India_000001.txt").read_text()
            self.assertEqual(len(label.splitlines()), 1)
            exported_names = {path.name for path in (output / "images").rglob("*.jpg")}
            self.assertNotIn("India_099999.jpg", exported_names)
            self.assertNotIn("India_088888.jpg", exported_names)
            self.assertNotIn("Japan_test_1.jpg", exported_names)

    def test_split_assignments_remain_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, canonical, _ = _make_fixture(root)
            output = run_export(config, canonical, root / "export", root)
            self.assertEqual(
                {path.name for path in (output / "images" / "train").iterdir()},
                {"India_000001.jpg"},
            )
            self.assertEqual(
                {path.name for path in (output / "images" / "val").iterdir()},
                {"Japan_000001.jpg"},
            )
            self.assertEqual(
                {path.name for path in (output / "images" / "test").iterdir()},
                {"India_000002.jpg"},
            )
            manifest = json.loads((output / "export_manifest.json").read_text())
            observed = [item["canonical_image_id"] for item in manifest["images"]]
            self.assertEqual(len(observed), len(set(observed)))

    def test_expected_object_reconciliation_blocks_a_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, canonical, _ = _make_fixture(root)
            config = json.loads(config_path.read_text())
            config["expected"]["target_objects"]["train"]["D00"] = 2
            _write_json(config_path, config)
            with self.assertRaises(RDD2022Error):
                run_export(config_path, canonical, root / "export", root)

    def test_export_is_deterministic_and_preserves_canonical_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, canonical, _ = _make_fixture(root)
            before = _content_tree_fingerprint(canonical)
            first = run_export(config, canonical, root / "export-a", root)
            second = run_export(config, canonical, root / "export-b", root)
            first_manifest = json.loads((first / "export_manifest.json").read_text())
            second_manifest = json.loads((second / "export_manifest.json").read_text())
            self.assertTrue(first_manifest["deterministic_rerun"]["match"])
            self.assertEqual(
                first_manifest["deterministic_rerun"]["first_plan_sha256"],
                second_manifest["deterministic_rerun"]["first_plan_sha256"],
            )
            self.assertEqual(
                first_manifest["generated_artifact_hashes"]["label_tree_sha256"],
                second_manifest["generated_artifact_hashes"]["label_tree_sha256"],
            )
            self.assertEqual(before, _content_tree_fingerprint(canonical))

    def test_independent_validator_checks_pairing_hashes_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, canonical, _ = _make_fixture(root)
            output = run_export(config, canonical, root / "export", root)
            report = validate_export(config, canonical, output, root)
            self.assertTrue(report["phase2c1_complete"])
            self.assertTrue(report["image_label_pairing"]["passed"])
            self.assertEqual(report["negative_empty_label_count"], 2)
            self.assertEqual(report["total_target_object_count"], 1)
            self.assertEqual(
                report["source_export_image_hashes"]["export_hash_mismatch_count"], 0
            )
            self.assertTrue(report["exclusions"]["passed"])
            finalized = json.loads((output / "export_report.json").read_text())
            self.assertEqual(finalized["status"], "export_validated")
            self.assertTrue(finalized["validation"]["passed"])

    def test_export_refuses_to_overwrite_an_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, canonical, _ = _make_fixture(root)
            output = root / "export"
            output.mkdir()
            with self.assertRaises(RDD2022Error):
                run_export(config, canonical, output, root)


if __name__ == "__main__":
    unittest.main()
