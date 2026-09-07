"""Read-only integrity, class, and image-metadata audit for raw RDD2022 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_common import (
    RDD2022Error,
    TARGET_CLASSES,
    load_config,
    project_relative,
    resolve_project_path,
    sha256_file,
    write_json_atomic,
)
from road_damage.video.inspect_video import _load_opencv


LOGGER = logging.getLogger("road_damage.dataset.rdd2022_audit")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class CountryLayout:
    """Discovered raw paths for one country archive."""

    country_root: Path
    dataset_country_root: Path
    train_images: Path
    train_xmls: Path
    test_images: Path | None


def discover_country_layout(country_root: Path, country: str) -> CountryLayout:
    """Find the single untouched country tree regardless of archive wrapper folders."""
    if not country_root.is_dir():
        raise RDD2022Error(f"Raw {country} directory does not exist: {country_root}")
    candidates: list[CountryLayout] = []
    directories = [country_root, *(path for path in country_root.rglob("*") if path.is_dir())]
    for directory in directories:
        train = directory / "train"
        train_images = train / "images"
        train_xmls = train / "annotations" / "xmls"
        if train_images.is_dir() and train_xmls.is_dir():
            test_images = directory / "test" / "images"
            candidates.append(
                CountryLayout(
                    country_root=country_root,
                    dataset_country_root=directory,
                    train_images=train_images,
                    train_xmls=train_xmls,
                    test_images=test_images if test_images.is_dir() else None,
                )
            )
    if len(candidates) != 1:
        raise RDD2022Error(
            f"Expected one {country} train/images + train/annotations/xmls layout; "
            f"found {len(candidates)} under {country_root}."
        )
    return candidates[0]


def _files(path: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        (file for file in path.rglob("*") if file.is_file() and file.suffix.casefold() in extensions),
        key=lambda file: file.as_posix().casefold(),
    )


def _duplicates_by_filename(paths: Sequence[Path]) -> list[dict[str, Any]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[path.name.casefold()].append(path)
    return [
        {
            "filename": items[0].name,
            "occurrences": len(items),
            "paths": [project_relative(item) for item in items],
        }
        for _, items in sorted(groups.items())
        if len(items) > 1
    ]


def _stem_map(paths: Sequence[Path]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        result[path.stem.casefold()].append(path)
    return result


def parse_pascal_voc_xml(path: Path) -> dict[str, Any]:
    """Parse class strings and declared image metadata without modifying XML."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RDD2022Error(f"Malformed or unreadable XML {path}: {exc}") from exc
    classes: list[str] = []
    for object_node in root.findall(".//object"):
        name = object_node.findtext("name")
        classes.append(name.strip() if name and name.strip() else "<missing>")
    size = root.find("size")
    declared_size = None
    if size is not None:
        try:
            declared_size = [int(size.findtext("width", "")), int(size.findtext("height", ""))]
        except ValueError:
            declared_size = None
    return {
        "classes": classes,
        "declared_filename": (root.findtext("filename") or "").strip() or None,
        "declared_size": declared_size,
    }


def _image_format(path: Path) -> str:
    try:
        with path.open("rb") as image_file:
            header = image_file.read(12)
    except OSError:
        return "unreadable"
    if header.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    return "unknown"


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        stat_result = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(
            f"{relative}\0{stat_result.st_size}\0{stat_result.st_mtime_ns}\n".encode("utf-8")
        )
        file_count += 1
        total_bytes += stat_result.st_size
    return {
        "metadata_fingerprint_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _audit_images(
    image_paths: Sequence[Path], expected_dimensions: set[tuple[int, int]], cv2: Any
) -> dict[str, Any]:
    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    orientations: Counter[str] = Counter()
    unreadable: list[str] = []
    unexpected: list[dict[str, Any]] = []
    for number, path in enumerate(image_paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        formats[_image_format(path)] += 1
        if image is None or image.size == 0 or image.ndim < 2:
            unreadable.append(project_relative(path))
            continue
        height, width = image.shape[:2]
        dimensions[f"{width}x{height}"] += 1
        orientation = "square" if width == height else (
            "landscape" if width > height else "portrait"
        )
        orientations[orientation] += 1
        if (width, height) not in expected_dimensions:
            unexpected.append(
                {"path": project_relative(path), "width": width, "height": height}
            )
        if number % 2_000 == 0:
            LOGGER.info("Read image metadata for %d/%d files", number, len(image_paths))
    return {
        "image_count": len(image_paths),
        "readable_image_count": len(image_paths) - len(unreadable),
        "unreadable_images": unreadable,
        "format_distribution": dict(sorted(formats.items())),
        "dimension_distribution": dict(sorted(dimensions.items())),
        "orientation_distribution": dict(sorted(orientations.items())),
        "expected_dimensions": [list(value) for value in sorted(expected_dimensions)],
        "unexpected_resolution_count": len(unexpected),
        "unexpected_resolutions": unexpected,
    }


def audit_country(
    country_root: Path, country_config: dict[str, Any], cv2: Any
) -> dict[str, Any]:
    """Audit one extracted country tree read-only."""
    country = str(country_config["country"])
    layout = discover_country_layout(country_root, country)
    train_images = _files(layout.train_images, IMAGE_EXTENSIONS)
    test_images = _files(layout.test_images, IMAGE_EXTENSIONS) if layout.test_images else []
    all_images = sorted([*train_images, *test_images])
    xmls = _files(layout.train_xmls, {".xml"})
    train_stems = _stem_map(train_images)
    xml_stems = _stem_map(xmls)
    missing_xml = sorted(
        project_relative(path)
        for stem in train_stems.keys() - xml_stems.keys()
        for path in train_stems[stem]
    )
    xml_missing_image = sorted(
        project_relative(path)
        for stem in xml_stems.keys() - train_stems.keys()
        for path in xml_stems[stem]
    )

    class_objects: Counter[str] = Counter()
    class_images: Counter[str] = Counter()
    malformed_xml: list[dict[str, str]] = []
    declared_filename_mismatches: list[dict[str, Any]] = []
    for number, path in enumerate(xmls, start=1):
        try:
            parsed = parse_pascal_voc_xml(path)
        except RDD2022Error as exc:
            malformed_xml.append({"path": project_relative(path), "error": str(exc)})
            continue
        classes = parsed["classes"]
        class_objects.update(classes)
        class_images.update(set(classes))
        declared = parsed["declared_filename"]
        matching_images = train_stems.get(path.stem.casefold(), [])
        if declared and matching_images and all(
            image.name.casefold() != declared.casefold() for image in matching_images
        ):
            declared_filename_mismatches.append(
                {
                    "xml_path": project_relative(path),
                    "declared_filename": declared,
                    "matched_by_stem": [project_relative(image) for image in matching_images],
                }
            )
        if number % 2_000 == 0:
            LOGGER.info("Parsed annotations for %d/%d %s XML files", number, len(xmls), country)

    expected_dimensions = {
        (int(item[0]), int(item[1]))
        for item in country_config["expected_image_dimensions"]
    }
    image_metadata = _audit_images(all_images, expected_dimensions, cv2)
    zero_byte_files = sorted(
        project_relative(path) for path in country_root.rglob("*")
        if path.is_file() and path.stat().st_size == 0
    )
    expected_counts = {
        key: int(value)
        for key, value in country_config["expected_target_object_counts"].items()
    }
    target_counts = {code: int(class_objects.get(code, 0)) for code in TARGET_CLASSES}
    discrepancies = {
        code: target_counts[code] - expected_counts[code] for code in TARGET_CLASSES
    }
    inventory = [
        {
            "class_name": class_name,
            "country": country,
            "object_count": int(class_objects[class_name]),
            "image_count_containing_class": int(class_images[class_name]),
            "is_project_target_class": class_name in TARGET_CLASSES,
        }
        for class_name in sorted(class_objects)
    ]
    return {
        "country": country,
        "layout": {
            "raw_country_root": project_relative(country_root),
            "dataset_country_root": project_relative(layout.dataset_country_root),
            "train_images_directory": project_relative(layout.train_images),
            "train_xml_directory": project_relative(layout.train_xmls),
            "test_images_directory": (
                project_relative(layout.test_images) if layout.test_images else None
            ),
        },
        "counts": {
            "train_images": len(train_images),
            "test_images": len(test_images),
            "all_images": len(all_images),
            "annotation_xmls": len(xmls),
        },
        "pair_integrity": {
            "train_images_missing_xml": missing_xml,
            "xmls_missing_train_image": xml_missing_image,
            "duplicate_image_filenames": _duplicates_by_filename(all_images),
            "duplicate_xml_filenames": _duplicates_by_filename(xmls),
            "declared_filename_mismatches": declared_filename_mismatches,
        },
        "malformed_xml": malformed_xml,
        "zero_byte_files": zero_byte_files,
        "class_inventory": inventory,
        "unexpected_class_names": [
            item["class_name"] for item in inventory
            if not item["is_project_target_class"]
        ],
        "target_object_counts": target_counts,
        "expected_target_object_counts": expected_counts,
        "target_count_discrepancies_observed_minus_expected": discrepancies,
        "image_metadata": image_metadata,
    }


def run_audit(
    config_path: Path, dataset_root: Path, acquisition_manifest_path: Path
) -> dict[str, Path]:
    """Audit India and Japan and write four read-only-derived JSON reports."""
    config_path, config = load_config(config_path)
    root = resolve_project_path(dataset_root)
    acquisition_path = resolve_project_path(acquisition_manifest_path)
    try:
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RDD2022Error(f"Could not read acquisition manifest: {acquisition_path}") from exc
    if acquisition.get("status") != "completed":
        raise RDD2022Error(
            "Acquisition manifest is not completed; raw audit cannot start safely."
        )
    if tuple(acquisition.get("allowed_countries", [])) != ("India", "Japan"):
        raise RDD2022Error("Acquisition manifest is outside the approved country scope.")

    cv2 = _load_opencv()
    before = _tree_fingerprint(root / "raw")
    country_audits = [
        audit_country(root / "raw" / item["country"], item, cv2)
        for item in config["countries"]
    ]
    after = _tree_fingerprint(root / "raw")
    if before != after:
        raise RDD2022Error("Raw data metadata changed during the read-only audit.")

    class_inventory = {
        "schema_version": "1.0",
        "source_acquisition_manifest": project_relative(acquisition_path),
        "target_classes": list(TARGET_CLASSES),
        "inventory": [
            row for audit in country_audits for row in audit["class_inventory"]
        ],
        "unexpected_class_names_by_country": {
            audit["country"]: audit["unexpected_class_names"]
            for audit in country_audits
        },
        "annotations_remapped_or_converted": False,
    }
    integrity_report = {
        "schema_version": "1.0",
        "source_acquisition_manifest": project_relative(acquisition_path),
        "raw_tree_before": before,
        "raw_tree_after": after,
        "raw_tree_unchanged_during_audit": before == after,
        "countries": [
            {
                key: audit[key]
                for key in (
                    "country", "layout", "counts", "pair_integrity",
                    "malformed_xml", "zero_byte_files", "image_metadata",
                )
            }
            for audit in country_audits
        ],
    }
    combined_observed = {
        code: sum(audit["target_object_counts"][code] for audit in country_audits)
        for code in TARGET_CLASSES
    }
    combined_expected = {
        code: sum(audit["expected_target_object_counts"][code] for audit in country_audits)
        for code in TARGET_CLASSES
    }
    dataset_summary = {
        "schema_version": "1.0",
        "audit_completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_acquisition_manifest": project_relative(acquisition_path),
        "countries": [
            {
                "country": audit["country"],
                "counts": audit["counts"],
                "target_object_counts": audit["target_object_counts"],
                "expected_target_object_counts": audit[
                    "expected_target_object_counts"
                ],
                "target_count_discrepancies_observed_minus_expected": audit[
                    "target_count_discrepancies_observed_minus_expected"
                ],
                "unexpected_class_names": audit["unexpected_class_names"],
                "dimension_distribution": audit["image_metadata"][
                    "dimension_distribution"
                ],
            }
            for audit in country_audits
        ],
        "combined": {
            "all_images": sum(audit["counts"]["all_images"] for audit in country_audits),
            "train_images": sum(audit["counts"]["train_images"] for audit in country_audits),
            "test_images": sum(audit["counts"]["test_images"] for audit in country_audits),
            "annotation_xmls": sum(audit["counts"]["annotation_xmls"] for audit in country_audits),
            "target_object_counts": combined_observed,
            "expected_target_object_counts": combined_expected,
            "target_count_discrepancies_observed_minus_expected": {
                code: combined_observed[code] - combined_expected[code]
                for code in TARGET_CLASSES
            },
            "raw_disk_usage_bytes": after["total_bytes"],
        },
        "scope_confirmations": {
            "annotations_converted": False,
            "class_names_remapped": False,
            "dataset_split_created": False,
            "raw_files_modified": False,
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "opencv_version": cv2.__version__,
        },
    }
    now = datetime.now(timezone.utc)
    run_id = f"rdd2022_audit_{now.strftime('%Y%m%dT%H%M%SZ')}"
    reports = {
        "class_inventory.json": class_inventory,
        "integrity_report.json": integrity_report,
        "dataset_summary.json": dataset_summary,
    }
    output_paths: dict[str, Path] = {}
    for filename, report in reports.items():
        latest = root / "manifests" / filename
        run_path = root / "manifests" / "runs" / run_id / filename
        write_json_atomic(run_path, report)
        write_json_atomic(latest, report)
        output_paths[filename] = latest
    return output_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit untouched RDD2022 India/Japan Pascal VOC source data."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--acquisition-manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        reports = run_audit(
            args.config, args.dataset_root, args.acquisition_manifest
        )
    except RDD2022Error as exc:
        LOGGER.error("%s", exc)
        return 1
    for name, path in reports.items():
        LOGGER.info("%s: %s", name, project_relative(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
