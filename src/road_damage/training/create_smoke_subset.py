"""Create and independently validate the deterministic Phase 3B smoke subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_common import (  # noqa: E402
    PROJECT_ROOT,
    RDD2022Error,
    sha256_file,
    write_json_atomic,
)


LOGGER = logging.getLogger(__name__)
ALLOWED_SPLITS = ("train", "val")
ALLOWED_COUNTRIES = ("India", "Japan")
ALLOWED_CLASS_IDS = (0, 1, 2, 3)
CLASS_NAMES = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}
EXPECTED_COUNTS = {
    "train": {"positive": 80, "negative": 48},
    "val": {"positive": 40, "negative": 24},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATA_YAML = """train: images/train
val: images/val

names:
  0: D00_longitudinal_crack
  1: D10_transverse_crack
  2: D20_alligator_crack
  3: D40_pothole
"""


@dataclass(frozen=True)
class SourceSample:
    """One paired source image and label eligible for smoke selection."""

    filename: str
    source_split: str
    country: str
    image_path: Path
    label_path: Path
    is_positive: bool
    classes_present: tuple[int, ...]
    class_object_counts: tuple[tuple[int, int], ...]


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RDD2022Error(f"Could not read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise RDD2022Error(f"{description.capitalize()} must be a JSON object: {path}")
    return value


def load_smoke_config(path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load and strictly validate the dependency-free Phase 3B config."""
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    config = _read_json(resolved, "Phase 3B smoke-subset config")
    expected_taxonomy = [
        {"id": class_id, "name": CLASS_NAMES[class_id]}
        for class_id in ALLOWED_CLASS_IDS
    ]
    if config.get("schema_version") != "3B.smoke_subset.v1":
        raise RDD2022Error("Phase 3B smoke config schema_version must be 3B.smoke_subset.v1.")
    if config.get("seed") != 42:
        raise RDD2022Error("Phase 3B smoke selection seed must be exactly 42.")
    if config.get("splits") != EXPECTED_COUNTS:
        raise RDD2022Error(f"Phase 3B smoke split counts must be exactly {EXPECTED_COUNTS}.")
    if config.get("countries") != list(ALLOWED_COUNTRIES):
        raise RDD2022Error("Phase 3B smoke countries must be exactly India and Japan.")
    if config.get("taxonomy") != expected_taxonomy:
        raise RDD2022Error("Phase 3B smoke taxonomy or numeric class order changed.")
    if config.get("minimum_india_d10_positive_images_per_split") != 2:
        raise RDD2022Error("Each smoke split must require at least two India D10-positive images.")
    if config.get("copy_policy") != "byte_identical_copy_no_hardlinks":
        raise RDD2022Error("Smoke copy policy must require byte-identical copies without hardlinks.")
    for key in ("source_dataset_path", "output_path"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise RDD2022Error(f"Phase 3B smoke config requires a non-empty {key}.")
    config["_config_path"] = str(resolved)
    return config


def _country_from_filename(filename: str) -> str:
    matches = [country for country in ALLOWED_COUNTRIES if filename.startswith(f"{country}_")]
    if len(matches) != 1:
        raise RDD2022Error(f"Cannot determine approved country from filename: {filename}")
    return matches[0]


def _parse_yolo_label_bytes(data: bytes, path: Path) -> tuple[tuple[int, ...], Counter[int]]:
    if data == b"":
        return (), Counter()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RDD2022Error(f"YOLO label is not UTF-8: {path}") from exc
    counts: Counter[int] = Counter()
    lines = text.splitlines()
    if not lines:
        raise RDD2022Error(f"Non-empty label has no target rows: {path}")
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise RDD2022Error(f"Invalid YOLO row at {path}:{line_number}")
        try:
            class_id = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise RDD2022Error(f"Invalid YOLO value at {path}:{line_number}") from exc
        if class_id not in ALLOWED_CLASS_IDS:
            raise RDD2022Error(f"Disallowed class ID {class_id} at {path}:{line_number}")
        if not all(math.isfinite(value) for value in coordinates):
            raise RDD2022Error(f"Non-finite YOLO value at {path}:{line_number}")
        center_x, center_y, width, height = coordinates
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            raise RDD2022Error(f"YOLO center is outside [0,1] at {path}:{line_number}")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise RDD2022Error(f"YOLO size is outside (0,1] at {path}:{line_number}")
        counts[class_id] += 1
    if not counts:
        raise RDD2022Error(f"Non-empty label has no target rows: {path}")
    return tuple(sorted(counts)), counts


def _unique_files_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise RDD2022Error(f"Required source directory is missing: {directory}")
    result: dict[str, Path] = {}
    try:
        candidates = sorted(
            (path for path in directory.iterdir() if path.is_file()),
            key=lambda path: path.name.casefold(),
        )
    except OSError as exc:
        raise RDD2022Error(f"Could not enumerate source directory: {directory}") from exc
    for path in candidates:
        if path.suffix.casefold() not in suffixes:
            continue
        key = path.stem.casefold()
        if key in result:
            raise RDD2022Error(f"Duplicate filename stem in {directory}: {path.stem}")
        result[key] = path
    return result


def inventory_source_split(source_root: Path, split: str) -> list[SourceSample]:
    """Inventory exactly one approved source split; test is never an accepted value."""
    if split not in ALLOWED_SPLITS:
        raise RDD2022Error(f"Smoke construction may inventory only {ALLOWED_SPLITS}, not {split!r}.")
    images = _unique_files_by_stem(source_root / "images" / split, IMAGE_SUFFIXES)
    labels = _unique_files_by_stem(source_root / "labels" / split, {".txt"})
    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    if missing_labels or orphan_labels:
        raise RDD2022Error(
            f"Source {split} image/label pairing failed: "
            f"missing_labels={missing_labels[:5]}, orphan_labels={orphan_labels[:5]}"
        )
    samples: list[SourceSample] = []
    for stem in sorted(images):
        label_path = labels[stem]
        classes, object_counts = _parse_yolo_label_bytes(label_path.read_bytes(), label_path)
        samples.append(
            SourceSample(
                filename=images[stem].name,
                source_split=split,
                country=_country_from_filename(images[stem].name),
                image_path=images[stem].resolve(),
                label_path=label_path.resolve(),
                is_positive=bool(classes),
                classes_present=classes,
                class_object_counts=tuple(sorted(object_counts.items())),
            )
        )
    return samples


def _selection_rank(sample: SourceSample, seed: int) -> tuple[str, str]:
    token = f"phase3b-smoke-v1\0{seed}\0{sample.source_split}\0{sample.filename}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest(), sample.filename.casefold()


def select_smoke_split(
    samples: Sequence[SourceSample],
    split: str,
    positive_count: int,
    negative_count: int,
    seed: int = 42,
    minimum_india_d10: int = 2,
) -> list[SourceSample]:
    """Select one split deterministically while satisfying all coverage constraints."""
    if split not in ALLOWED_SPLITS or any(sample.source_split != split for sample in samples):
        raise RDD2022Error("Smoke selection received a disallowed or mixed source split.")
    positives = sorted((sample for sample in samples if sample.is_positive), key=lambda item: _selection_rank(item, seed))
    negatives = sorted((sample for sample in samples if not sample.is_positive), key=lambda item: _selection_rank(item, seed))
    if len(positives) < positive_count or len(negatives) < negative_count:
        raise RDD2022Error(
            f"Source {split} lacks smoke capacity: positives={len(positives)}, negatives={len(negatives)}."
        )

    selected_positive: list[SourceSample] = []
    selected_names: set[str] = set()

    def add_positive(sample: SourceSample) -> None:
        if sample.filename not in selected_names:
            selected_positive.append(sample)
            selected_names.add(sample.filename)

    india_d10 = [
        sample for sample in positives
        if sample.country == "India" and 1 in sample.classes_present
    ]
    if len(india_d10) < minimum_india_d10:
        raise RDD2022Error(
            f"Source {split} has only {len(india_d10)} India D10-positive images; "
            f"{minimum_india_d10} are required."
        )
    for sample in india_d10[:minimum_india_d10]:
        add_positive(sample)

    missing_classes = set(ALLOWED_CLASS_IDS) - {
        class_id for sample in selected_positive for class_id in sample.classes_present
    }
    while missing_classes:
        covering = [
            sample for sample in positives
            if sample.filename not in selected_names
            and missing_classes.intersection(sample.classes_present)
        ]
        if not covering:
            raise RDD2022Error(f"Source {split} cannot cover classes {sorted(missing_classes)}.")
        chosen = min(
            covering,
            key=lambda sample: (
                -len(missing_classes.intersection(sample.classes_present)),
                *_selection_rank(sample, seed),
            ),
        )
        add_positive(chosen)
        missing_classes.difference_update(chosen.classes_present)

    present_countries = {sample.country for sample in selected_positive}
    for country in ALLOWED_COUNTRIES:
        if country in present_countries:
            continue
        candidates = [
            sample for sample in positives
            if sample.country == country and sample.filename not in selected_names
        ]
        if candidates and len(selected_positive) < positive_count:
            add_positive(candidates[0])
            present_countries.add(country)

    for sample in positives:
        if len(selected_positive) == positive_count:
            break
        add_positive(sample)
    if len(selected_positive) != positive_count:
        raise RDD2022Error(f"Could not select exactly {positive_count} positives for {split}.")

    selected_negative: list[SourceSample] = []
    negative_names: set[str] = set()
    present_countries = {sample.country for sample in selected_positive}
    for country in ALLOWED_COUNTRIES:
        if country in present_countries:
            continue
        candidates = [sample for sample in negatives if sample.country == country]
        if not candidates:
            raise RDD2022Error(f"Source {split} cannot represent country {country}.")
        selected_negative.append(candidates[0])
        negative_names.add(candidates[0].filename)
        present_countries.add(country)
    for sample in negatives:
        if len(selected_negative) == negative_count:
            break
        if sample.filename not in negative_names:
            selected_negative.append(sample)
            negative_names.add(sample.filename)
    if len(selected_negative) != negative_count:
        raise RDD2022Error(f"Could not select exactly {negative_count} negatives for {split}.")

    selected = selected_positive + selected_negative
    if {sample.country for sample in selected} != set(ALLOWED_COUNTRIES):
        raise RDD2022Error(f"Selected {split} does not contain both approved countries.")
    if {class_id for sample in selected for class_id in sample.classes_present} != set(ALLOWED_CLASS_IDS):
        raise RDD2022Error(f"Selected {split} does not contain all four classes.")
    return sorted(selected, key=lambda sample: sample.filename.casefold())


def _copy_and_hash(source: Path, destination: Path) -> tuple[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    try:
        with source.open("rb") as source_handle, destination.open("xb") as target_handle:
            for chunk in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
                source_digest.update(chunk)
                target_handle.write(chunk)
    except OSError as exc:
        raise RDD2022Error(f"Could not byte-copy {source} to {destination}") from exc
    if os.path.samefile(source, destination):
        raise RDD2022Error(f"Smoke output unexpectedly shares file identity with source: {destination}")
    source_hash = source_digest.hexdigest()
    copied_hash = sha256_file(destination)
    if source_hash != copied_hash:
        raise RDD2022Error(f"Byte-copy hash mismatch: {destination}")
    return source_hash, copied_hash


def _load_upstream_validation(source_root: Path) -> dict[str, Any]:
    report = _read_json(source_root / "validation_report.json", "approved export validation report")
    exclusions = report.get("exclusions", {})
    intersections = exclusions.get("intersections", {}) if isinstance(exclusions, dict) else {}
    if (
        report.get("phase2c1_complete") is not True
        or report.get("status") != "passed"
        or exclusions.get("passed") is not True
        or intersections.get("official_unlabelled_test") != []
        or exclusions.get("teacher_reference_count") != 0
    ):
        raise RDD2022Error(
            "Approved source validation does not prove exclusion of official unlabelled-test "
            "and teacher-video content."
        )
    return report


def _selection_fingerprint(selected: dict[str, list[SourceSample]], seed: int) -> str:
    payload = {
        "seed": seed,
        "samples": [
            {
                "filename": sample.filename,
                "source_split": sample.source_split,
                "country": sample.country,
                "classes_present": list(sample.classes_present),
            }
            for split in ALLOWED_SPLITS
            for sample in selected[split]
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_dataset_files(
    staging: Path,
    final_output: Path,
    source_root: Path,
    selected: dict[str, list[SourceSample]],
    seed: int,
    project_root: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        for sample in selected[split]:
            copied_image = staging / "images" / split / sample.filename
            copied_label = staging / "labels" / split / sample.label_path.name
            source_image_hash, copied_image_hash = _copy_and_hash(sample.image_path, copied_image)
            source_label_hash, copied_label_hash = _copy_and_hash(sample.label_path, copied_label)
            records.append(
                {
                    "filename": sample.filename,
                    "source_split": sample.source_split,
                    "smoke_split": split,
                    "country": sample.country,
                    "sample_type": "positive" if sample.is_positive else "negative",
                    "is_positive": sample.is_positive,
                    "classes_present": list(sample.classes_present),
                    "source_image_path": _display_path(sample.image_path, project_root),
                    "source_label_path": _display_path(sample.label_path, project_root),
                    "copied_image_path": _display_path(final_output / "images" / split / sample.filename, project_root),
                    "copied_label_path": _display_path(final_output / "labels" / split / sample.label_path.name, project_root),
                    "source_image_sha256": source_image_hash,
                    "copied_image_sha256": copied_image_hash,
                    "source_label_sha256": source_label_hash,
                    "copied_label_sha256": copied_label_hash,
                }
            )
    try:
        (staging / "data.yaml").write_text(DATA_YAML, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise RDD2022Error(f"Could not write smoke data.yaml in {staging}") from exc
    manifest = {
        "schema_version": "3B.smoke_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "selection_algorithm": "constraint_first_then_sha256_rank_v1",
        "selection_sha256": _selection_fingerprint(selected, seed),
        "source_dataset_path": _display_path(source_root, project_root),
        "smoke_dataset_path": _display_path(final_output, project_root),
        "source_split_policy": {"train": "train", "val": "val"},
        "source_test_split_accessed": False,
        "copy_policy": "byte_identical_copy_no_hardlinks_no_move_no_recompression",
        "taxonomy": [
            {"id": class_id, "name": CLASS_NAMES[class_id]}
            for class_id in ALLOWED_CLASS_IDS
        ],
        "samples": records,
    }
    write_json_atomic(staging / "smoke_manifest.json", manifest)
    return manifest


def validate_smoke_dataset(
    dataset_root: Path,
    source_root: Path,
    final_output: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Independently inspect copied files, labels, lineage, and hashes."""
    dataset_root = dataset_root.resolve()
    source_root = source_root.resolve()
    final_output = (final_output or dataset_root).resolve()
    manifest = _read_json(dataset_root / "smoke_manifest.json", "smoke manifest")
    records = manifest.get("samples")
    if not isinstance(records, list):
        raise RDD2022Error("Smoke manifest samples must be a list.")
    upstream = _load_upstream_validation(source_root)
    errors: list[str] = []
    split_reports: dict[str, dict[str, Any]] = {}
    hash_mismatches: list[str] = []
    shared_file_identities: list[str] = []
    observed_class_ids: set[int] = set()
    lineage_only_approved_splits = True
    unique_rows: set[tuple[str, str]] = set()

    for split in ALLOWED_SPLITS:
        split_records = [record for record in records if record.get("smoke_split") == split]
        images = _unique_files_by_stem(dataset_root / "images" / split, IMAGE_SUFFIXES)
        labels = _unique_files_by_stem(dataset_root / "labels" / split, {".txt"})
        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))
        manifest_filenames = {str(record.get("filename")) for record in split_records}
        actual_filenames = {path.name for path in images.values()}
        if missing_labels:
            errors.append(f"{split}: missing labels for {missing_labels[:5]}")
        if orphan_labels:
            errors.append(f"{split}: orphan labels for {orphan_labels[:5]}")
        if manifest_filenames != actual_filenames:
            errors.append(f"{split}: manifest and copied image filenames differ")

        positive = 0
        negative = 0
        countries: Counter[str] = Counter()
        class_image_counts: Counter[int] = Counter()
        class_object_counts: Counter[int] = Counter()
        india_d10 = 0
        for record in split_records:
            filename = str(record.get("filename"))
            row_key = (split, filename.casefold())
            if row_key in unique_rows:
                errors.append(f"{split}: duplicate manifest row for {filename}")
            unique_rows.add(row_key)
            source_split = record.get("source_split")
            if source_split != split or source_split not in ALLOWED_SPLITS:
                lineage_only_approved_splits = False
                errors.append(f"{split}: disallowed source split for {filename}")
                continue
            try:
                country = _country_from_filename(filename)
            except RDD2022Error as exc:
                errors.append(str(exc))
                continue
            if record.get("country") != country:
                errors.append(f"{split}: country mismatch for {filename}")
            countries[country] += 1
            source_image = (source_root / "images" / split / filename).resolve()
            source_label = (source_root / "labels" / split / f"{Path(filename).stem}.txt").resolve()
            copied_image = dataset_root / "images" / split / filename
            copied_label = dataset_root / "labels" / split / f"{Path(filename).stem}.txt"
            expected_record_paths = {
                "source_image_path": _display_path(source_image, project_root),
                "source_label_path": _display_path(source_label, project_root),
                "copied_image_path": _display_path(final_output / "images" / split / filename, project_root),
                "copied_label_path": _display_path(final_output / "labels" / split / f"{Path(filename).stem}.txt", project_root),
            }
            for key, expected in expected_record_paths.items():
                if record.get(key) != expected:
                    errors.append(f"{split}: {key} mismatch for {filename}")
            if not all(path.is_file() for path in (source_image, source_label, copied_image, copied_label)):
                errors.append(f"{split}: missing source or copied pair for {filename}")
                continue
            source_image_hash = sha256_file(source_image)
            copied_image_hash = sha256_file(copied_image)
            source_label_hash = sha256_file(source_label)
            copied_label_hash = sha256_file(copied_label)
            observed_hashes = {
                "source_image_sha256": source_image_hash,
                "copied_image_sha256": copied_image_hash,
                "source_label_sha256": source_label_hash,
                "copied_label_sha256": copied_label_hash,
            }
            if source_image_hash != copied_image_hash or source_label_hash != copied_label_hash:
                hash_mismatches.append(f"{split}/{filename}")
            for key, observed in observed_hashes.items():
                if record.get(key) != observed:
                    hash_mismatches.append(f"{split}/{filename}:{key}")
            if os.path.samefile(source_image, copied_image) or os.path.samefile(source_label, copied_label):
                shared_file_identities.append(f"{split}/{filename}")
            try:
                copied_classes, copied_objects = _parse_yolo_label_bytes(copied_label.read_bytes(), copied_label)
                source_classes, _ = _parse_yolo_label_bytes(source_label.read_bytes(), source_label)
            except RDD2022Error as exc:
                errors.append(str(exc))
                continue
            if copied_classes != source_classes or list(copied_classes) != record.get("classes_present"):
                errors.append(f"{split}: class presence mismatch for {filename}")
            is_positive = bool(copied_classes)
            if record.get("is_positive") is not is_positive:
                errors.append(f"{split}: positive/negative manifest mismatch for {filename}")
            if record.get("sample_type") != ("positive" if is_positive else "negative"):
                errors.append(f"{split}: sample_type mismatch for {filename}")
            if is_positive:
                positive += 1
            else:
                negative += 1
            for class_id in copied_classes:
                class_image_counts[class_id] += 1
                observed_class_ids.add(class_id)
            class_object_counts.update(copied_objects)
            if country == "India" and 1 in copied_classes:
                india_d10 += 1

        expected = EXPECTED_COUNTS[split]
        if len(images) != expected["positive"] + expected["negative"]:
            errors.append(f"{split}: image count is {len(images)}")
        if len(labels) != expected["positive"] + expected["negative"]:
            errors.append(f"{split}: label count is {len(labels)}")
        if positive != expected["positive"] or negative != expected["negative"]:
            errors.append(f"{split}: positive/negative counts are {positive}/{negative}")
        if set(countries) != set(ALLOWED_COUNTRIES):
            errors.append(f"{split}: both countries are not represented")
        if set(class_image_counts) != set(ALLOWED_CLASS_IDS):
            errors.append(f"{split}: all four classes are not represented")
        if india_d10 < 2:
            errors.append(f"{split}: only {india_d10} India D10-positive images")
        split_reports[split] = {
            "images": len(images),
            "labels": len(labels),
            "positive": positive,
            "negative": negative,
            "country_counts": {country: countries[country] for country in ALLOWED_COUNTRIES},
            "class_image_occurrence_counts": {
                str(class_id): class_image_counts[class_id] for class_id in ALLOWED_CLASS_IDS
            },
            "class_object_counts": {
                str(class_id): class_object_counts[class_id] for class_id in ALLOWED_CLASS_IDS
            },
            "india_d10_positive_image_count": india_d10,
            "missing_image_label_pairs": len(missing_labels),
            "orphan_labels": len(orphan_labels),
        }

    unexpected_manifest_splits = sorted(
        {str(record.get("smoke_split")) for record in records} - set(ALLOWED_SPLITS)
    )
    if unexpected_manifest_splits:
        errors.append(f"Unexpected smoke splits in manifest: {unexpected_manifest_splits}")
    try:
        data_yaml = (dataset_root / "data.yaml").read_text(encoding="utf-8")
    except OSError as exc:
        raise RDD2022Error(f"Could not read smoke data.yaml: {dataset_root / 'data.yaml'}") from exc
    data_yaml_exact = data_yaml == DATA_YAML and not any(
        line.lstrip().startswith("test:") for line in data_yaml.splitlines()
    )
    if not data_yaml_exact:
        errors.append("data.yaml is not the exact train/val-only four-class mapping")
    if hash_mismatches:
        errors.append(f"Byte hash mismatches: {hash_mismatches[:5]}")
    if shared_file_identities:
        errors.append(f"Hardlink/shared identity detected: {shared_file_identities[:5]}")
    if observed_class_ids - set(ALLOWED_CLASS_IDS):
        errors.append(f"Disallowed class IDs found: {sorted(observed_class_ids)}")

    upstream_exclusions = upstream["exclusions"]
    exclusions_passed = (
        lineage_only_approved_splits
        and upstream_exclusions["intersections"]["official_unlabelled_test"] == []
        and upstream_exclusions["teacher_reference_count"] == 0
    )
    if not exclusions_passed:
        errors.append("Official unlabelled-test or teacher-video exclusion evidence failed")
    report = {
        "schema_version": "3B.smoke_validation.v1",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "errors": errors,
        "splits": split_reports,
        "source_lineage": {
            "only_source_train_to_smoke_train_and_source_val_to_smoke_val": lineage_only_approved_splits,
            "source_test_split_accessed": False,
            "no_source_test_image_included": lineage_only_approved_splits,
        },
        "exclusions": {
            "no_official_unlabelled_test_image_included": exclusions_passed,
            "no_teacher_video_content_included": exclusions_passed,
            "evidence": "selected files are exact members of approved source train/val; source Phase 2C.1 validation exclusions passed",
        },
        "pairing": {
            "no_missing_image_label_pair": all(
                value["missing_image_label_pairs"] == 0 for value in split_reports.values()
            ),
            "no_orphan_image_or_label": all(
                value["missing_image_label_pairs"] == 0 and value["orphan_labels"] == 0
                for value in split_reports.values()
            ),
        },
        "hash_verification": {
            "files_checked": len(records) * 2,
            "mismatch_count": len(hash_mismatches),
            "byte_identical_to_source": not hash_mismatches,
            "shared_file_identity_count": len(shared_file_identities),
            "no_hardlinks": not shared_file_identities,
        },
        "labels": {
            "negative_labels_exactly_empty": not errors or not any("negative" in error for error in errors),
            "positive_labels_non_empty": not errors or not any("positive" in error for error in errors),
            "only_class_ids": sorted(observed_class_ids),
            "only_allowed_class_ids": not (observed_class_ids - set(ALLOWED_CLASS_IDS)),
        },
        "data_yaml_train_val_only": data_yaml_exact,
    }
    return report


def run_smoke_subset(
    config_path: Path,
    source_root: Path | None = None,
    output_root: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Create the immutable Phase 3B smoke dataset and refuse overwrite."""
    config = load_smoke_config(config_path, project_root)
    source_root = (
        source_root.resolve()
        if source_root is not None
        else (project_root / config["source_dataset_path"]).resolve()
    )
    output_root = (
        output_root.resolve()
        if output_root is not None
        else (project_root / config["output_path"]).resolve()
    )
    if output_root.exists():
        raise RDD2022Error(f"Refusing to overwrite existing smoke dataset: {output_root}")
    _load_upstream_validation(source_root)
    inventories = {
        split: inventory_source_split(source_root, split) for split in ALLOWED_SPLITS
    }
    selected = {
        split: select_smoke_split(
            inventories[split],
            split,
            EXPECTED_COUNTS[split]["positive"],
            EXPECTED_COUNTS[split]["negative"],
            seed=config["seed"],
            minimum_india_d10=config["minimum_india_d10_positive_images_per_split"],
        )
        for split in ALLOWED_SPLITS
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        _write_dataset_files(
            staging, output_root, source_root, selected, config["seed"], project_root
        )
        report = validate_smoke_dataset(
            staging, source_root, final_output=output_root, project_root=project_root
        )
        write_json_atomic(staging / "smoke_validation_report.json", report)
        if not report["passed"]:
            raise RDD2022Error(
                "Independent smoke validation failed: " + "; ".join(report["errors"][:5])
            )
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the deterministic train/val-only Phase 3B smoke dataset."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output = run_smoke_subset(args.config, args.source_root, args.output_root)
    except RDD2022Error as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Phase 3B smoke dataset created: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
