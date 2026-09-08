"""Focused tests for the frozen-model mentor image demo."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.demo.mentor_demo import (  # noqa: E402
    CLASS_NAMES,
    DEMO_OUTPUT_RELATIVE_PATH,
    FROZEN_MODEL_RELATIVE_PATH,
    FROZEN_MODEL_SHA256,
    FROZEN_MODEL_SIZE_BYTES,
    INTERNAL_TEST_IMAGES_RELATIVE_PATH,
    MentorDemoError,
    VALIDATION_IMAGES_RELATIVE_PATH,
    VALIDATION_LABELS_RELATIVE_PATH,
    frozen_model_path,
    inventory_validation_samples,
    load_demo_config,
    next_output_path,
    select_mentor_samples,
    validate_source_path,
    validate_validation_sample_directories,
    verify_model_identity,
)


EXPECTED_CLASSES = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}
EXPECTED_MODEL_PATH = Path(
    "outputs/training/baseline_public_v1/"
    "20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42/weights/best.pt"
)
EXPECTED_MODEL_SHA256 = (
    "BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7"
)


def _make_validation_inventory(root: Path) -> tuple[Path, Path]:
    images = root / VALIDATION_IMAGES_RELATIVE_PATH
    labels = root / VALIDATION_LABELS_RELATIVE_PATH
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for class_id in range(4):
        for index, area in enumerate((0.01, 0.04, 0.02)):
            stem = f"class_{class_id}_{index}"
            (images / f"{stem}.jpg").write_bytes(f"image:{stem}".encode())
            side = area ** 0.5
            (labels / f"{stem}.txt").write_text(
                f"{class_id} 0.5 0.5 {side} {side}\n", encoding="utf-8"
            )
    for index in range(3):
        stem = f"negative_{index}"
        (images / f"{stem}.jpg").write_bytes(f"image:{stem}".encode())
        (labels / f"{stem}.txt").write_bytes(b"")
    return images, labels


class MentorDemoTests(unittest.TestCase):
    def test_class_mapping_model_path_and_sha_are_frozen(self) -> None:
        self.assertEqual(CLASS_NAMES, EXPECTED_CLASSES)
        self.assertEqual(FROZEN_MODEL_RELATIVE_PATH, EXPECTED_MODEL_PATH)
        self.assertEqual(FROZEN_MODEL_SHA256, EXPECTED_MODEL_SHA256)
        self.assertEqual(FROZEN_MODEL_SIZE_BYTES, 22_524_074)
        self.assertEqual(frozen_model_path(PROJECT_ROOT), (PROJECT_ROOT / EXPECTED_MODEL_PATH).resolve())
        config = load_demo_config()
        self.assertEqual(dict(config.class_names), EXPECTED_CLASSES)
        self.assertEqual(config.model_path, EXPECTED_MODEL_PATH)
        self.assertEqual(config.model_sha256, EXPECTED_MODEL_SHA256)

    def test_source_validation_rejects_missing_unsupported_checkpoint_and_internal_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(MentorDemoError):
                validate_source_path(root / "missing.jpg", project_root=root)
            unsupported = root / "notes.txt"
            unsupported.write_text("not an image", encoding="utf-8")
            with self.assertRaises(MentorDemoError):
                validate_source_path(unsupported, project_root=root)
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaises(MentorDemoError):
                validate_source_path(checkpoint, project_root=root)
            internal = root / INTERNAL_TEST_IMAGES_RELATIVE_PATH / "forbidden.jpg"
            internal.parent.mkdir(parents=True)
            internal.write_bytes(b"image")
            with self.assertRaises(MentorDemoError):
                validate_source_path(internal, project_root=root)
            allowed = root / "explicit.jpg"
            allowed.write_bytes(b"image")
            self.assertEqual(validate_source_path(allowed, project_root=root), [allowed.resolve()])

            folder = root / "folder"
            folder.mkdir()
            (folder / "z.jpg").write_bytes(b"image-z")
            (folder / "a.png").write_bytes(b"image-a")
            (folder / "sample_manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                [path.name for path in validate_source_path(folder, project_root=root)],
                ["a.png", "z.jpg"],
            )

    def test_output_path_avoids_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / DEMO_OUTPUT_RELATIVE_PATH
            output.mkdir(parents=True)
            (output / "sample.jpg").write_bytes(b"first")
            (output / "sample_001.jpg").write_bytes(b"second")
            self.assertEqual(next_output_path(output, "sample.jpg"), output / "sample_002.jpg")
            with self.assertRaises(MentorDemoError):
                next_output_path(output, "last.pt")

    def test_model_identity_rejects_bad_sha_or_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "best.pt"
            payload = b"frozen-test-model"
            model.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest().upper()
            self.assertEqual(
                verify_model_identity(
                    model, expected_sha256=expected, expected_size_bytes=len(payload)
                ),
                expected,
            )
            with self.assertRaises(MentorDemoError):
                verify_model_identity(
                    model, expected_sha256="0" * 64, expected_size_bytes=len(payload)
                )
            with self.assertRaises(MentorDemoError):
                verify_model_identity(
                    model, expected_sha256=expected, expected_size_bytes=len(payload) + 1
                )

    def test_validation_only_selection_is_deterministic_and_gt_based(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels = _make_validation_inventory(root)
            inventory = inventory_validation_samples(images, labels, project_root=root)
            first = select_mentor_samples(inventory)
            second = select_mentor_samples(list(reversed(inventory)))
            self.assertEqual(
                [(item.bucket, item.sample.filename) for item in first],
                [(item.bucket, item.sample.filename) for item in second],
            )
            self.assertEqual(len(first), 10)
            self.assertEqual(sum(item.class_id is not None for item in first), 8)
            self.assertEqual(sum(item.class_id is None for item in first), 2)
            for class_id in range(4):
                assigned = [item for item in first if item.class_id == class_id]
                self.assertEqual(len(assigned), 2)
                self.assertIn(f"class_{class_id}_1.jpg", {item.sample.filename for item in assigned})
                self.assertTrue(all(class_id in item.sample.classes_present for item in assigned))
            self.assertTrue(all(item.sample.is_negative for item in first if item.class_id is None))

    def test_internal_test_and_teacher_paths_cannot_supply_project_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels = _make_validation_inventory(root)
            validate_validation_sample_directories(images, labels, project_root=root)
            internal_images = root / INTERNAL_TEST_IMAGES_RELATIVE_PATH
            internal_labels = internal_images.parent.parent / "labels" / "test"
            internal_images.mkdir(parents=True)
            internal_labels.mkdir(parents=True)
            with self.assertRaises(MentorDemoError):
                validate_validation_sample_directories(
                    internal_images, internal_labels, project_root=root
                )
            teacher_images = root / "outputs" / "teacher_video" / "positive_frames"
            teacher_labels = root / "outputs" / "teacher_video" / "labels"
            teacher_images.mkdir(parents=True)
            teacher_labels.mkdir(parents=True)
            with self.assertRaises(MentorDemoError):
                validate_validation_sample_directories(
                    teacher_images, teacher_labels, project_root=root
                )


if __name__ == "__main__":
    unittest.main()
