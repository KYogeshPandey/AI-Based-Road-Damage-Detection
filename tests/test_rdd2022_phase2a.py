"""Tests for country-scoped RDD2022 acquisition and raw-data auditing."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.dataset.rdd2022_acquire import (  # noqa: E402
    _download,
    _download_segmented,
    _existing_extraction,
    extract_country_from_full_archive,
    extract_zip_safely,
    inspect_full_archive,
    inspect_zip,
    query_figshare_metadata,
)
from road_damage.dataset.rdd2022_audit import (  # noqa: E402
    audit_country,
    discover_country_layout,
    parse_pascal_voc_xml,
)
from road_damage.dataset.rdd2022_common import (  # noqa: E402
    RDD2022Error,
    load_config,
    validate_config,
)
from road_damage.video.inspect_video import _load_opencv  # noqa: E402


class _MemoryResponse(io.BytesIO):
    def __init__(
        self, value: bytes, status: int = 200, content_range: str | None = None
    ) -> None:
        super().__init__(value)
        self.headers = {"Content-Length": str(len(value))}
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        self.status = status

    def __enter__(self) -> "_MemoryResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class ConfigurationTests(unittest.TestCase):
    def test_project_config_is_strictly_india_and_japan(self) -> None:
        _, config = load_config(Path("configs/dataset/rdd2022_phase2a.yaml"))

        self.assertEqual(config["allowed_countries"], ["India", "Japan"])
        self.assertEqual(config["target_classes"], ["D00", "D10", "D20", "D40"])
        self.assertEqual(
            config["dataset"]["licenses"]["project_conservative_interpretation"],
            "CC BY-SA 4.0",
        )
        self.assertEqual(
            config["dataset"]["licenses"]["discrepancy_status"], "unresolved"
        )
        self.assertEqual(config["acquisition_fallback"]["file_id"], 38030910)
        self.assertEqual(
            config["acquisition_fallback"]["size_bytes"], 13_264_172_619
        )
        self.assertEqual(
            config["acquisition_fallback"]["countries_to_extract"],
            ["India", "Japan"],
        )

    def test_rejects_an_extra_country(self) -> None:
        _, config = load_config(Path("configs/dataset/rdd2022_phase2a.yaml"))
        config = json.loads(json.dumps(config))
        config["allowed_countries"].append("Czech")

        with self.assertRaises(RDD2022Error):
            validate_config(config)


class AcquisitionTests(unittest.TestCase):
    def _zip(self, root: Path, members: dict[str, bytes]) -> Path:
        archive_path = root / "country.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in members.items():
                archive.writestr(name, value)
        return archive_path

    def test_download_commits_complete_response_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "RDD2022_India.zip"
            content = b"permission-safe fixture bytes"

            result = _download(
                "https://example.invalid/RDD2022_India.zip",
                destination,
                opener=lambda request, timeout: _MemoryResponse(content),
            )

            self.assertTrue(result["downloaded_this_run"])
            self.assertEqual(destination.read_bytes(), content)
            self.assertFalse(destination.with_name(f".{destination.name}.part").exists())

    def test_download_resumes_a_partial_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "archive.zip"
            partial = destination.with_name(f".{destination.name}.part")
            partial.write_bytes(b"first-")

            result = _download(
                "https://example.invalid/archive.zip",
                destination,
                opener=lambda request, timeout: _MemoryResponse(b"second", status=206),
                expected_size=12,
            )

            self.assertEqual(destination.read_bytes(), b"first-second")
            self.assertTrue(result["download_resumed"])
            self.assertEqual(result["resumed_from_bytes"], 6)

    def test_segmented_download_adopts_partial_and_assembles_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "archive.zip"
            destination.with_name(f".{destination.name}.part").write_bytes(b"abc")
            content = b"abcdefghijklmnopqrstuvwxyz"

            def opener(request: object, timeout: int) -> _MemoryResponse:
                range_header = request.headers["Range"]  # type: ignore[attr-defined]
                start_text, end_text = range_header.removeprefix("bytes=").split("-")
                start, end = int(start_text), int(end_text)
                return _MemoryResponse(
                    content[start:end + 1],
                    status=206,
                    content_range=f"bytes {start}-{end}/{len(content)}",
                )

            result = _download_segmented(
                "https://example.invalid/archive.zip",
                destination,
                expected_size=len(content),
                connections=3,
                opener=opener,
            )

            self.assertEqual(destination.read_bytes(), content)
            self.assertTrue(result["download_resumed"])
            self.assertEqual(result["adopted_legacy_sequential_partial_bytes"], 3)
            self.assertEqual(list(destination.parent.glob("*.part*")), [])

    def test_segmented_download_retries_failure_before_segment_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "archive.zip"
            content = b"retry-safe"
            attempts = 0

            def opener(request: object, timeout: int) -> _MemoryResponse:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise urllib.error.URLError("temporary fixture failure")
                range_header = request.headers["Range"]  # type: ignore[attr-defined]
                start_text, end_text = range_header.removeprefix("bytes=").split("-")
                start, end = int(start_text), int(end_text)
                return _MemoryResponse(
                    content[start:end + 1],
                    status=206,
                    content_range=f"bytes {start}-{end}/{len(content)}",
                )

            _download_segmented(
                "https://example.invalid/archive.zip",
                destination,
                expected_size=len(content),
                connections=1,
                opener=opener,
            )

            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(attempts, 2)

    def test_official_figshare_metadata_must_match_pinned_config(self) -> None:
        _, config = load_config(Path("configs/dataset/rdd2022_phase2a.yaml"))
        fallback = config["acquisition_fallback"]
        response = {
            "id": fallback["article_id"],
            "version": fallback["article_version"],
            "doi": config["dataset"]["figshare_doi"],
            "license": {
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            },
            "files": [
                {
                    "id": fallback["file_id"],
                    "name": fallback["archive_filename"],
                    "download_url": fallback["download_url"],
                    "size": fallback["size_bytes"],
                    "supplied_md5": fallback["supplied_md5"],
                    "computed_md5": fallback["supplied_md5"],
                }
            ],
        }

        metadata = query_figshare_metadata(
            config,
            opener=lambda request, timeout: _MemoryResponse(
                json.dumps(response).encode("utf-8")
            ),
        )

        self.assertTrue(metadata["matches_pinned_config"])
        self.assertEqual(metadata["file_id"], 38030910)

    def test_safe_zip_is_verified_and_extracted_without_renaming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = self._zip(
                root,
                {
                    "RDD2022/India/train/images/India_000001.jpg": b"jpeg-fixture",
                    "RDD2022/India/train/annotations/xmls/India_000001.xml": b"<annotation />",
                },
            )

            inspection = inspect_zip(archive, "India")
            extraction = extract_zip_safely(archive, "India", root / "raw" / "India")

            self.assertTrue(inspection["zip_crc_test_passed"])
            self.assertEqual(extraction["extracted_file_count"], 2)
            self.assertTrue(
                (root / "raw" / "India" / "RDD2022" / "India" / "train" / "images" / "India_000001.jpg").is_file()
            )

    def test_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = self._zip(
                Path(temporary_directory),
                {"RDD2022/India/../../escape.txt": b"unsafe"},
            )

            with self.assertRaises(RDD2022Error):
                inspect_zip(archive, "India")

    def test_rejects_other_country_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = self._zip(
                Path(temporary_directory),
                {
                    "RDD2022/India/train/images/a.jpg": b"india",
                    "RDD2022/Japan/train/images/b.jpg": b"japan",
                },
            )

            with self.assertRaises(RDD2022Error):
                inspect_zip(archive, "India")

    def test_full_archive_is_validated_but_only_approved_countries_are_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = self._zip(
                root,
                {
                    "RDD2022/India/train/images/India_000001.jpg": b"india",
                    "RDD2022/India/train/annotations/xmls/India_000001.xml": b"<annotation />",
                    "RDD2022/Japan/train/images/Japan_000001.jpg": b"japan",
                    "RDD2022/Czech/train/images/Czech_000001.jpg": b"czech",
                },
            )

            inspection = inspect_full_archive(archive, ["India", "Japan"])
            india = extract_country_from_full_archive(
                archive, "India", root / "raw" / "India"
            )
            japan = extract_country_from_full_archive(
                archive, "Japan", root / "raw" / "Japan"
            )

            self.assertTrue(inspection["zip_crc_test_passed"])
            self.assertIn("Czech", inspection["countries_present_by_exact_path_segment"])
            self.assertEqual(india["extracted_file_count"], 2)
            self.assertEqual(japan["extracted_file_count"], 1)
            self.assertTrue(
                (root / "raw" / "India" / "train" / "images" / "India_000001.jpg").is_file()
            )
            self.assertFalse((root / "raw" / "Czech").exists())

    def test_nested_country_archives_are_crc_checked_and_selectively_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def nested_zip(members: dict[str, bytes]) -> bytes:
                value = io.BytesIO()
                with zipfile.ZipFile(value, "w", zipfile.ZIP_DEFLATED) as nested:
                    for name, content in members.items():
                        nested.writestr(name, content)
                return value.getvalue()

            archive = self._zip(
                root,
                {
                    "RDD2022/India.zip": nested_zip(
                        {
                            "India/train/images/India_1.jpg": b"india",
                            "India/train/annotations/xmls/India_1.xml": b"<annotation />",
                        }
                    ),
                    "RDD2022/Japan.zip": nested_zip(
                        {"Japan/train/images/Japan_1.jpg": b"japan"}
                    ),
                    "RDD2022/Czech.zip": nested_zip(
                        {"Czech/train/images/Czech_1.jpg": b"czech"}
                    ),
                },
            )

            inspection = inspect_full_archive(archive, ["India", "Japan"])
            india = extract_country_from_full_archive(
                archive, "India", root / "raw" / "India"
            )
            japan = extract_country_from_full_archive(
                archive, "Japan", root / "raw" / "Japan"
            )

            self.assertEqual(
                inspection["countries_present_by_exact_path_segment"],
                ["Czech", "India", "Japan"],
            )
            self.assertEqual(india["nested_archive_member"], "RDD2022/India.zip")
            self.assertTrue(india["country_archive_integrity"]["zip_crc_test_passed"])
            self.assertTrue(japan["country_archive_integrity"]["zip_crc_test_passed"])
            self.assertTrue(
                (root / "raw" / "India" / "train" / "images" / "India_1.jpg").is_file()
            )
            self.assertFalse((root / "raw" / "Czech").exists())

    def test_existing_nonempty_raw_extraction_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_country = Path(temporary_directory) / "raw" / "India"
            source_file = raw_country / "India" / "train" / "images" / "image.jpg"
            source_file.parent.mkdir(parents=True)
            source_file.write_bytes(b"raw-source-fixture")

            result = _existing_extraction(raw_country)

            self.assertIsNotNone(result)
            self.assertTrue(result["reused_existing_extraction"])
            self.assertEqual(result["extracted_file_count"], 1)


class RawAuditTests(unittest.TestCase):
    def _country_tree(self, root: Path) -> tuple[Path, dict[str, object]]:
        country_root = root / "raw" / "India"
        country = country_root / "wrapper" / "India"
        images = country / "train" / "images"
        xmls = country / "train" / "annotations" / "xmls"
        test_images = country / "test" / "images"
        images.mkdir(parents=True)
        xmls.mkdir(parents=True)
        test_images.mkdir(parents=True)
        cv2 = _load_opencv()
        import numpy as np

        image = np.full((12, 16, 3), 127, np.uint8)
        self.assertTrue(cv2.imwrite(str(images / "India_000001.jpg"), image))
        self.assertTrue(cv2.imwrite(str(images / "India_000002.jpg"), image))
        self.assertTrue(cv2.imwrite(str(test_images / "India_test_0001.jpg"), image))
        (xmls / "India_000001.xml").write_text(
            """<annotation><filename>India_000001.jpg</filename>
            <size><width>16</width><height>12</height></size>
            <object><name>D00</name></object><object><name>D50</name></object>
            </annotation>""",
            encoding="utf-8",
        )
        (xmls / "orphan.xml").write_text(
            "<annotation><object><name>D40</name></object></annotation>",
            encoding="utf-8",
        )
        country_config: dict[str, object] = {
            "country": "India",
            "expected_image_dimensions": [[16, 12]],
            "expected_target_object_counts": {
                "D00": 1,
                "D10": 0,
                "D20": 0,
                "D40": 1,
            },
        }
        return country_root, country_config

    def test_layout_discovery_and_pascal_voc_class_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            country_root, _ = self._country_tree(Path(temporary_directory))

            layout = discover_country_layout(country_root, "India")
            parsed = parse_pascal_voc_xml(
                layout.train_xmls / "India_000001.xml"
            )

            self.assertEqual(parsed["classes"], ["D00", "D50"])
            self.assertEqual(layout.dataset_country_root.name, "India")

    def test_audit_reports_pairs_classes_and_image_metadata_without_remapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            country_root, country_config = self._country_tree(Path(temporary_directory))

            audit = audit_country(country_root, country_config, _load_opencv())

            self.assertEqual(audit["counts"]["train_images"], 2)
            self.assertEqual(audit["counts"]["test_images"], 1)
            self.assertEqual(audit["counts"]["annotation_xmls"], 2)
            self.assertEqual(len(audit["pair_integrity"]["train_images_missing_xml"]), 1)
            self.assertEqual(len(audit["pair_integrity"]["xmls_missing_train_image"]), 1)
            self.assertEqual(audit["unexpected_class_names"], ["D50"])
            self.assertEqual(audit["target_object_counts"]["D00"], 1)
            self.assertEqual(audit["target_object_counts"]["D40"], 1)
            self.assertEqual(
                audit["image_metadata"]["dimension_distribution"], {"16x12": 3}
            )
            self.assertEqual(audit["image_metadata"]["unreadable_images"], [])


if __name__ == "__main__":
    unittest.main()
