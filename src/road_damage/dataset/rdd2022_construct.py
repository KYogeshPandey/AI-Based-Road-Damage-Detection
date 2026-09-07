"""Construct the reference-only RDD2022 India/Japan Phase 2B canonical dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_audit import (  # noqa: E402
    IMAGE_EXTENSIONS,
    _image_format,
    _tree_fingerprint,
    discover_country_layout,
)
from road_damage.dataset.rdd2022_canonical import (  # noqa: E402
    NON_TARGET_ACTION,
    SPLIT_NAMES,
    TARGET_ACTION,
    UNKNOWN_ACTION,
    REJECT_INVALID_STATUS,
    REJECT_TINY_STATUS,
    apply_boundary_guards,
    apply_special_case_policy,
    assign_capture_groups,
    assign_exact_duplicate_groups,
    classify_labelled_image,
    generate_near_duplicate_candidates,
    load_phase2b_config,
    parse_voc_annotation,
    plan_grouped_split,
    read_exif_timestamp,
    split_conflict_audit,
    stable_id,
)
from road_damage.dataset.rdd2022_common import (  # noqa: E402
    PROJECT_ROOT,
    RDD2022Error,
    TARGET_CLASSES,
    project_relative,
    resolve_project_path,
    sha256_file,
    write_json_atomic,
)
from road_damage.video.inspect_video import _load_opencv  # noqa: E402


LOGGER = logging.getLogger("road_damage.dataset.rdd2022_construct")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _files(path: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        (
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.casefold() in extensions
        ),
        key=lambda item: item.as_posix().casefold(),
    )


def _canonical_image_id(country: str, partition: str, filename: str) -> str:
    return stable_id("rdd2022v1", country.casefold(), partition, filename.casefold(), length=24)


def _dhash64(gray_image: Any, cv2: Any) -> str:
    resized = cv2.resize(gray_image, (9, 8), interpolation=cv2.INTER_AREA)
    differences = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in differences.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _similarity_thumbnail(gray_image: Any, cv2: Any) -> bytes:
    return cv2.resize(gray_image, (32, 32), interpolation=cv2.INTER_AREA).tobytes()


def _read_image_metadata(path: Path, cv2: Any) -> tuple[dict[str, Any], bytes]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RDD2022Error(f"Previously validated source image is unreadable: {path}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    exif = read_exif_timestamp(path)
    metadata = {
        "image_sha256": sha256_file(path),
        "decoded_width": int(width),
        "decoded_height": int(height),
        "decoded_channels": int(image.shape[2]),
        "image_format": _image_format(path),
        "perceptual_hash_dhash64": _dhash64(gray, cv2),
        **exif,
    }
    return metadata, _similarity_thumbnail(gray, cv2)


def _target_counts(objects: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        item["original_raw_class_string"]
        for item in objects if item.get("object_status") == TARGET_ACTION
    )
    return {name: int(counts[name]) for name in TARGET_CLASSES}


def _scan_labelled_country(
    country: str,
    country_root: Path,
    config: dict[str, Any],
    cv2: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layout = discover_country_layout(country_root, country)
    images = _files(layout.train_images, IMAGE_EXTENSIONS)
    xmls = _files(layout.train_xmls, {".xml"})
    xml_by_stem = {item.stem.casefold(): item for item in xmls}
    if len(xml_by_stem) != len(xmls):
        raise RDD2022Error(f"Duplicate XML stems prevent canonical indexing for {country}.")
    records: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []

    def scan_one(image_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        xml_path = xml_by_stem.get(image_path.stem.casefold())
        if xml_path is None:
            raise RDD2022Error(f"Labelled image has no matching XML: {image_path}")
        image_id = _canonical_image_id(country, "train", image_path.name)
        image_metadata, thumbnail = _read_image_metadata(image_path, cv2)
        parsed = parse_voc_annotation(
            xml_path,
            image_id,
            image_metadata["decoded_width"],
            image_metadata["decoded_height"],
            config,
        )
        apply_special_case_policy(parsed, image_path.name, config)
        classification = classify_labelled_image(parsed, config)
        for item in parsed["objects"]:
            item.update(
                {
                    "dataset_name": config["source_dataset"],
                    "dataset_version": config["source_dataset_version"],
                    "country": country,
                    "original_partition": "train",
                    "original_filename": image_path.name,
                    "raw_relative_xml_path": project_relative(xml_path),
                }
            )
        record = {
            "canonical_image_id": image_id,
            "dataset_name": config["source_dataset"],
            "dataset_version": config["source_dataset_version"],
            "country": country,
            "original_partition": "train",
            "original_filename": image_path.name,
            "raw_relative_image_path": project_relative(image_path),
            "raw_relative_xml_path": project_relative(xml_path),
            "xml_sha256": sha256_file(xml_path),
            "xml_declared_width": parsed["xml_width"],
            "xml_declared_height": parsed["xml_height"],
            "xml_declared_depth": parsed["xml_depth"],
            "xml_image_dimension_mismatch": parsed["xml_image_dimension_mismatch"],
            "xml_parse_error": parsed["xml_parse_error"],
            **image_metadata,
            **classification,
            "target_class_counts": _target_counts(parsed["objects"]),
            "capture_group": None,
            "grouping_method": None,
            "grouping_confidence": None,
            "exact_duplicate_group": None,
            "near_duplicate_candidate_group": None,
            "confirmed_near_duplicate_group": None,
            "derived_split": None,
            "_similarity_gray_bytes": thumbnail,
        }
        return record, parsed["objects"]

    worker_count = max(1, int(config.get("source_scan_worker_count", 1)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(scan_one, images)
        for number, (record, parsed_objects) in enumerate(results, start=1):
            records.append(record)
            objects.extend(parsed_objects)
            if number % 2_000 == 0:
                LOGGER.info("Canonicalized %d/%d labelled %s images", number, len(images), country)
    if len(xmls) != len(images):
        raise RDD2022Error(
            f"{country} labelled source count mismatch: {len(images)} images, {len(xmls)} XMLs."
        )
    return records, objects


def _scan_official_test_country(
    country: str, country_root: Path, config: dict[str, Any], cv2: Any
) -> list[dict[str, Any]]:
    layout = discover_country_layout(country_root, country)
    if layout.test_images is None:
        raise RDD2022Error(f"Official unlabelled test directory is absent for {country}.")
    images = _files(layout.test_images, IMAGE_EXTENSIONS)

    def scan_one(image_path: Path) -> dict[str, Any]:
        image_id = _canonical_image_id(country, "official_unlabelled_test", image_path.name)
        metadata, _ = _read_image_metadata(image_path, cv2)
        metadata.pop("perceptual_hash_dhash64")
        return {
            "canonical_image_id": image_id,
            "dataset_name": config["source_dataset"],
            "dataset_version": config["source_dataset_version"],
            "country": country,
            "original_partition": "official_unlabelled_test",
            "original_filename": image_path.name,
            "raw_relative_image_path": project_relative(image_path),
            "raw_relative_xml_path": None,
            "xml_sha256": None,
            **metadata,
            "target_object_count": None,
            "non_target_object_count": None,
            "total_object_count": None,
            "inclusion_status": "official_unlabelled_test_excluded",
            "inclusion_exclusion_reason": [
                "official_unlabelled_test_excluded_from_all_development_splits"
            ],
            "negative_subtype": None,
            "capture_group": None,
            "grouping_method": None,
            "grouping_confidence": None,
            "exact_duplicate_group": None,
            "near_duplicate_candidate_group": None,
            "derived_split": None,
        }

    worker_count = max(1, int(config.get("source_scan_worker_count", 1)))
    records = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for number, record in enumerate(executor.map(scan_one, images), start=1):
            records.append(record)
            if number % 2_000 == 0:
                LOGGER.info("Registered %d/%d official-test %s images", number, len(images), country)
    return records


def _clean_image_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items()
        if not key.startswith("_") and key != "exif_timestamp_sort_ms"
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                output.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RDD2022Error(f"Could not write JSONL artifact: {path}") from exc


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _group_size_statistics(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[int]] = defaultdict(list)
    for group in groups:
        by_method[group["grouping_method"]].append(int(group["image_count"]))

    def percentile(values: list[int], fraction: float) -> int:
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        method: {
            "groups": len(values),
            "images": sum(values),
            "minimum": min(values),
            "p10": percentile(values, 0.10),
            "median": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "maximum": max(values),
            "singleton_groups": sum(value == 1 for value in values),
            "groups_with_at_most_five_images": sum(value <= 5 for value in values),
        }
        for method, values in sorted(by_method.items())
    }


def _summarize_images(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    country = Counter(item["country"] for item in records)
    resolution = Counter(
        f"{item['decoded_width']}x{item['decoded_height']}" for item in records
    )
    categories = Counter(item["image_category"] for item in records)
    target_counts = Counter()
    country_targets: dict[str, Counter[str]] = defaultdict(Counter)
    for item in records:
        target_counts.update(item["target_class_counts"])
        country_targets[item["country"]].update(item["target_class_counts"])
    return {
        "images": len(records),
        "countries": _counter_dict(country),
        "positive_images": sum(bool(item["is_positive"]) for item in records),
        "negative_images": sum(bool(item["is_negative"]) for item in records),
        "image_categories": _counter_dict(categories),
        "target_object_counts": {
            name: int(target_counts[name]) for name in TARGET_CLASSES
        },
        "target_object_counts_by_country": {
            name: {code: int(country_targets[name][code]) for code in TARGET_CLASSES}
            for name in sorted(country_targets)
        },
        "resolution_distribution": _counter_dict(resolution),
    }


def _annotation_qa(objects: Sequence[dict[str, Any]], images: Sequence[dict[str, Any]]) -> dict[str, Any]:
    flags_all: Counter[str] = Counter()
    flags_target: Counter[str] = Counter()
    invalid_targets: list[dict[str, Any]] = []
    projected_areas: list[float] = []
    aspect_ratios: list[float] = []
    raw_classes: Counter[str] = Counter()
    object_statuses: Counter[str] = Counter()
    zero_edge_objects: list[str] = []
    rejected_objects: list[dict[str, Any]] = []
    for item in objects:
        raw_classes[item["original_raw_class_string"]] += 1
        object_statuses[item["object_status"]] += 1
        flags_all.update(item["validation_flags"])
        if item.get("source_zero_edge_coordinate"):
            zero_edge_objects.append(item["canonical_object_id"])
        if item["object_status"] in {REJECT_INVALID_STATUS, REJECT_TINY_STATUS}:
            rejected_objects.append(
                {
                    "canonical_object_id": item["canonical_object_id"],
                    "canonical_image_id": item["canonical_image_id"],
                    "original_filename": item["original_filename"],
                    "raw_class": item["original_raw_class_string"],
                    "object_status": item["object_status"],
                    "reason": item["object_status_reason"],
                }
            )
        if item["canonical_action"] == TARGET_ACTION:
            flags_target.update(item["validation_flags"])
            if item["invalid_target_flags"]:
                invalid_targets.append(
                    {
                        "canonical_object_id": item["canonical_object_id"],
                        "canonical_image_id": item["canonical_image_id"],
                        "country": item["country"],
                        "flags": item["invalid_target_flags"],
                    }
                )
            projected = item["projected_640"]
            if projected is not None:
                projected_areas.append(float(projected["bbox_area_pixels"]))
            aspect = item["aspect_ratio_width_over_height"]
            if aspect is not None:
                aspect_ratios.append(float(aspect))

    def distribution(values: list[float], thresholds: list[float]) -> dict[str, int]:
        buckets: Counter[str] = Counter()
        for value in values:
            placed = False
            lower = 0.0
            for upper in thresholds:
                if value < upper:
                    buckets[f"[{lower},{upper})"] += 1
                    placed = True
                    break
                lower = upper
            if not placed:
                buckets[f"[{lower},inf)"] += 1
        return _counter_dict(buckets)

    return {
        "raw_class_object_inventory": _counter_dict(raw_classes),
        "object_status_inventory": _counter_dict(object_statuses),
        "source_zero_edge_coordinate_count": len(zero_edge_objects),
        "source_zero_edge_object_ids": sorted(zero_edge_objects),
        "rejected_target_object_count": len(rejected_objects),
        "rejected_target_objects": rejected_objects,
        "validation_flags_all_objects": _counter_dict(flags_all),
        "validation_flags_target_objects": _counter_dict(flags_target),
        "invalid_target_annotation_count": len(invalid_targets),
        "invalid_target_annotations": invalid_targets,
        "xml_image_dimension_mismatch_image_count": sum(
            item["xml_image_dimension_mismatch"] is True for item in images
        ),
        "projected_640_target_box_area_distribution_pixels": distribution(
            projected_areas, [16, 64, 256, 1024, 4096, 16384, 65536]
        ),
        "target_box_aspect_ratio_distribution": distribution(
            aspect_ratios, [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 20]
        ),
        "raw_coordinates_preserved": True,
        "questionable_annotations_silently_repaired": False,
    }


def _content_tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        file_digest = sha256_file(path)
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{file_digest}\n".encode("utf-8"))
        file_count += 1
        total_bytes += path.stat().st_size
    return {
        "content_tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _apply_final_target_dispositions(
    objects: Sequence[dict[str, Any]], images: Sequence[dict[str, Any]]
) -> None:
    by_id = {item["canonical_image_id"]: item for item in images}
    for item in objects:
        if item["canonical_action"] != TARGET_ACTION:
            item["final_target_disposition"] = None
            continue
        if item["object_status"] in {REJECT_INVALID_STATUS, REJECT_TINY_STATUS}:
            item["final_target_disposition"] = item["object_status"]
            continue
        image = by_id[item["canonical_image_id"]]
        if image["inclusion_status"] == "quarantined":
            item["final_target_disposition"] = "whole_image_quarantine"
        elif image["inclusion_status"] == "guard_excluded":
            item["final_target_disposition"] = "guard_excluded"
        elif image["inclusion_status"] == "split_assigned":
            item["final_target_disposition"] = "retained"
        else:
            raise RDD2022Error(
                f"Target object has no final disposition: {item['canonical_object_id']}"
            )


def _target_disposition_reconciliation(
    objects: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in objects:
        if item["canonical_action"] != TARGET_ACTION:
            continue
        counts[item["original_raw_class_string"]][item["final_target_disposition"]] += 1
    by_class = {
        class_name: _counter_dict(counts[class_name]) for class_name in TARGET_CLASSES
    }
    totals = Counter()
    for class_counts in counts.values():
        totals.update(class_counts)
    return {
        "by_class": by_class,
        "totals": _counter_dict(totals),
        "raw_target_total": int(sum(totals.values())),
        "unexplained_target_object_loss": 0,
    }


def _india_d10_support(
    objects: Sequence[dict[str, Any]], images: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    by_image = {item["canonical_image_id"]: item for item in images}
    boxes: Counter[str] = Counter()
    image_ids: dict[str, set[str]] = defaultdict(set)
    for item in objects:
        if item["country"] != "India" or item["original_raw_class_string"] != "D10":
            continue
        image = by_image[item["canonical_image_id"]]
        if item["final_target_disposition"] == "retained":
            category = str(image["derived_split"])
        else:
            category = str(item["final_target_disposition"])
        boxes[category] += 1
        image_ids[category].add(item["canonical_image_id"])
    categories = (*SPLIT_NAMES, "guard_excluded", "whole_image_quarantine", "reject_invalid", "reject_tiny")
    return {
        category: {"boxes": int(boxes[category]), "images": len(image_ids[category])}
        for category in categories
    }


def _guard_loss_summary(
    guard_images: Sequence[dict[str, Any]], eligible_before_guards: int
) -> dict[str, Any]:
    targets = Counter()
    for item in guard_images:
        targets.update(item["target_class_counts"])
    return {
        "images": len(guard_images),
        "percentage_of_eligible_labelled_images": (
            100.0 * len(guard_images) / eligible_before_guards
            if eligible_before_guards else 0.0
        ),
        "by_country": _counter_dict(Counter(item["country"] for item in guard_images)),
        "by_target_class": {name: int(targets[name]) for name in TARGET_CLASSES},
        "positive_images": sum(bool(item["is_positive"]) for item in guard_images),
        "negative_images": sum(bool(item["is_negative"]) for item in guard_images),
        "by_grouping_evidence": _counter_dict(
            Counter(item["grouping_method"] for item in guard_images)
        ),
    }


def _country_share_audit(
    assigned_by_split: dict[str, list[dict[str, Any]]],
    eligible: Sequence[dict[str, Any]],
    tolerance_points: float,
) -> dict[str, Any]:
    overall = 100.0 * sum(item["country"] == "India" for item in eligible) / len(eligible)
    splits: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        records = assigned_by_split[split_name]
        india = sum(item["country"] == "India" for item in records)
        share = 100.0 * india / len(records) if records else 0.0
        deviation = share - overall
        splits[split_name] = {
            "india_images": india,
            "japan_images": len(records) - india,
            "india_share_percent": share,
            "deviation_from_overall_percentage_points": deviation,
            "within_tolerance": abs(deviation) <= tolerance_points,
        }
    return {
        "overall_eligible_india_share_percent": overall,
        "tolerance_percentage_points": tolerance_points,
        "splits": splits,
        "all_splits_within_tolerance": all(
            item["within_tolerance"] for item in splits.values()
        ),
    }


def _enrich_capture_groups(
    capture_groups: list[dict[str, Any]], records: Sequence[dict[str, Any]]
) -> None:
    by_id = {item["canonical_image_id"]: item for item in records}
    for group in capture_groups:
        members = [by_id[image_id] for image_id in group["image_ids"]]
        group["status_counts"] = _counter_dict(
            Counter(item["inclusion_status"] for item in members)
        )
        group["derived_splits"] = sorted({
            item["derived_split"] for item in members if item["derived_split"] is not None
        })
        targets = Counter()
        for item in members:
            targets.update(item["target_class_counts"])
        group["target_object_counts"] = {
            name: int(targets[name]) for name in TARGET_CLASSES
        }
        numeric_ids = [
            item["filename_numeric_id"] for item in members
            if item.get("filename_numeric_id") is not None
        ]
        timestamps = [
            item["exif_timestamp_value"] for item in members
            if item.get("exif_timestamp_value") is not None
        ]
        group["filename_numeric_id_min"] = min(numeric_ids) if numeric_ids else None
        group["filename_numeric_id_max"] = max(numeric_ids) if numeric_ids else None
        group["exif_timestamp_min"] = min(timestamps) if timestamps else None
        group["exif_timestamp_max"] = max(timestamps) if timestamps else None


def _contact_sheet(
    samples: Sequence[tuple[dict[str, Any], str]],
    title: str,
    destination: Path,
    config: dict[str, Any],
    cv2: Any,
) -> None:
    import numpy as np

    review = config["review_artifacts"]
    thumb_width = int(review["thumbnail_width"])
    thumb_height = int(review["thumbnail_height"])
    columns = int(review["contact_sheet_columns"])
    rows = max(1, math_ceil(len(samples) / columns))
    label_height = 54
    header_height = 42
    canvas = np.full(
        (header_height + rows * (thumb_height + label_height), columns * thumb_width, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(canvas, title[:110], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 1, cv2.LINE_AA)
    for index, (record, label) in enumerate(samples):
        image_path = resolve_project_path(Path(record["raw_relative_image_path"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        row, column = divmod(index, columns)
        scale = min(thumb_width / image.shape[1], thumb_height / image.shape[0])
        resized_width = max(1, int(round(image.shape[1] * scale)))
        resized_height = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        x = column * thumb_width + (thumb_width - resized_width) // 2
        y = header_height + row * (thumb_height + label_height) + (thumb_height - resized_height) // 2
        canvas[y:y + resized_height, x:x + resized_width] = resized
        for review_box in record.get("_review_boxes", []):
            xyxy = review_box.get("canonical_xyxy_pixel_zero_based_half_open")
            if xyxy is None:
                continue
            color = tuple(review_box.get("color_bgr", (0, 0, 220)))
            x1 = x + int(round(float(xyxy[0]) * scale))
            y1 = y + int(round(float(xyxy[1]) * scale))
            x2 = x + int(round(float(xyxy[2]) * scale))
            y2 = y + int(round(float(xyxy[3]) * scale))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        base_y = header_height + row * (thumb_height + label_height) + thumb_height + 18
        lines = [record["original_filename"], label]
        for line_index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line[:38],
                (column * thumb_width + 5, base_y + 20 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination),
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, int(review["jpeg_quality"])],
    ):
        raise RDD2022Error(f"Could not write review contact sheet: {destination}")


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if value == integer else integer + 1


def _deterministic_sample(
    records: Sequence[dict[str, Any]], count: int, seed: int, salt: str
) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: hashlib.sha256(
            f"{seed}:{salt}:{item['canonical_image_id']}".encode()
        ).hexdigest(),
    )[:count]


def _create_review_artifacts(
    records: Sequence[dict[str, Any]],
    objects: Sequence[dict[str, Any]],
    near_candidates: Sequence[dict[str, Any]],
    root: Path,
    config: dict[str, Any],
    cv2: Any,
) -> dict[str, Any]:
    review_root = root / "reports" / "review_contact_sheets"
    review_root.mkdir(parents=True, exist_ok=True)
    page_size = int(config["review_artifacts"]["contact_sheet_page_size"])
    seed = int(config["split"]["seed"])
    by_id = {item["canonical_image_id"]: item for item in records}
    objects_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        objects_by_image[item["canonical_image_id"]].append(item)
    artifacts: list[dict[str, Any]] = []

    def with_boxes(
        record: dict[str, Any], selected_objects: Sequence[dict[str, Any]], color: tuple[int, int, int]
    ) -> dict[str, Any]:
        review_record = dict(record)
        review_record["_review_boxes"] = [
            {
                "canonical_xyxy_pixel_zero_based_half_open": item[
                    "canonical_xyxy_pixel_zero_based_half_open"
                ],
                "color_bgr": color,
            }
            for item in selected_objects
        ]
        return review_record

    def write_pages(category: str, samples: list[tuple[dict[str, Any], str]]) -> None:
        for page_index in range(0, len(samples), page_size):
            page = samples[page_index:page_index + page_size]
            destination = review_root / f"{category}_{page_index // page_size + 1:03d}.jpg"
            _contact_sheet(page, category.replace("_", " ").title(), destination, config, cv2)
            artifacts.append(
                {
                    "category": category,
                    "path": destination.relative_to(root).as_posix(),
                    "image_ids": [item[0]["canonical_image_id"] for item in page],
                }
            )

    quarantine = []
    for item in records:
        if item["inclusion_status"] != "quarantined":
            continue
        target_objects = [
            value for value in objects_by_image[item["canonical_image_id"]]
            if value["canonical_action"] == TARGET_ACTION
        ]
        quarantine.append(
            (with_boxes(item, target_objects, (0, 0, 220)), ";".join(item["inclusion_exclusion_reason"]))
        )
    write_pages("quarantine_all_cases", quarantine)

    confirmed = [item for item in near_candidates if item["hard_split_constraint"]]
    warnings = [item for item in near_candidates if not item["hard_split_constraint"]]
    strongest = warnings[:int(config["near_duplicates"]["maximum_review_pairs"])]
    near_samples: list[tuple[dict[str, Any], str]] = []
    for pair_index, pair in enumerate(strongest, start=1):
        label = (
            f"pair {pair_index} h={pair['dhash_hamming_distance']} "
            f"mae={pair['gray_32x32_mean_absolute_difference_normalized']:.4f}"
        )
        near_samples.append((by_id[pair["first_image_id"]], label))
        near_samples.append((by_id[pair["second_image_id"]], label))
    write_pages("strongest_unconfirmed_near_duplicate_candidates", near_samples)

    confirmed_samples: list[tuple[dict[str, Any], str]] = []
    for pair_index, pair in enumerate(confirmed, start=1):
        label = (
            f"confirmed {pair_index} h={pair['dhash_hamming_distance']} "
            f"mae={pair['gray_32x32_mean_absolute_difference_normalized']:.4f}"
        )
        confirmed_samples.append((by_id[pair["first_image_id"]], label))
        confirmed_samples.append((by_id[pair["second_image_id"]], label))
    write_pages("confirmed_near_duplicate_relationships", confirmed_samples)

    zero_edge_samples: list[tuple[dict[str, Any], str]] = []
    rejected_samples: list[tuple[dict[str, Any], str]] = []
    aspect_objects = [
        item for item in objects
        if item["canonical_action"] == TARGET_ACTION
        and "unusual_aspect_ratio" in item["validation_flags"]
    ]
    for item in objects:
        image = by_id[item["canonical_image_id"]]
        if item.get("source_zero_edge_coordinate"):
            zero_edge_samples.append(
                (
                    with_boxes(image, [item], (0, 160, 0)),
                    f"{item['original_raw_class_string']} {item['object_status']}",
                )
            )
        if item["object_status"] in {REJECT_INVALID_STATUS, REJECT_TINY_STATUS}:
            rejected_samples.append(
                (
                    with_boxes(image, [item], (0, 0, 220)),
                    f"{item['original_raw_class_string']} {item['object_status']}",
                )
            )
    write_pages("zero_edge_normalized_objects", zero_edge_samples)
    write_pages("rejected_target_objects", rejected_samples)
    aspect_sample = _deterministic_sample(
        aspect_objects,
        int(config["review_artifacts"].get("representative_aspect_ratio_count", 16)),
        seed,
        "aspect_ratio",
    )
    write_pages(
        "representative_aspect_ratio_warnings",
        [
            (
                with_boxes(by_id[item["canonical_image_id"]], [item], (200, 80, 0)),
                f"{item['original_raw_class_string']} aspect={item['aspect_ratio_width_over_height']:.3f}",
            )
            for item in aspect_sample
        ],
    )

    negatives = [item for item in records if item.get("derived_split") and item["is_negative"]]
    negative_sample = _deterministic_sample(
        negatives,
        int(config["review_artifacts"]["representative_negative_count"]),
        seed,
        "negative",
    )
    write_pages(
        "representative_negatives",
        [(item, item["negative_subtype"] or "negative") for item in negative_sample],
    )

    target_samples: list[tuple[dict[str, Any], str]] = []
    per_class = int(config["review_artifacts"]["representative_per_class_country"])
    for country in config["countries"]:
        for class_name in TARGET_CLASSES:
            choices = [
                item for item in records
                if item.get("derived_split")
                and item["country"] == country
                and item["target_class_counts"][class_name] > 0
            ]
            for item in _deterministic_sample(choices, per_class, seed, f"{country}:{class_name}"):
                matching = [
                    value for value in objects_by_image[item["canonical_image_id"]]
                    if value["original_raw_class_string"] == class_name
                    and value["object_status"] == TARGET_ACTION
                ]
                target_samples.append(
                    (with_boxes(item, matching, (0, 160, 0)), f"{country} {class_name}")
                )
    write_pages("representative_targets_by_class_country", target_samples)

    for split_name in SPLIT_NAMES:
        choices = [item for item in records if item.get("derived_split") == split_name]
        selected = _deterministic_sample(
            choices,
            int(config["review_artifacts"]["representative_per_eval_split"]),
            seed,
            f"split:{split_name}",
        )
        write_pages(
            f"representative_{split_name}",
            [(item, f"{split_name} {item['country']}") for item in selected],
        )
        stratified: list[tuple[dict[str, Any], str]] = []
        for country in config["countries"]:
            for class_name in TARGET_CLASSES:
                class_records = [
                    item for item in choices
                    if item["country"] == country
                    and item["target_class_counts"][class_name] > 0
                ]
                for item in _deterministic_sample(
                    class_records, 1, seed, f"{split_name}:{country}:{class_name}"
                ):
                    matching = [
                        value for value in objects_by_image[item["canonical_image_id"]]
                        if value["original_raw_class_string"] == class_name
                        and value["object_status"] == TARGET_ACTION
                    ]
                    stratified.append(
                        (
                            with_boxes(item, matching, (0, 160, 0)),
                            f"{split_name} {country} {class_name}",
                        )
                    )
        write_pages(f"representative_{split_name}_by_country_class", stratified)
    index = {
        "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
        "thumbnail_only": True,
        "raw_images_resized_or_modified": False,
        "artifact_count": len(artifacts),
        "confirmed_relationships_rendered": len(confirmed),
        "warning_relationships_rendered": len(strongest),
        "zero_edge_objects_rendered": len(zero_edge_samples),
        "rejected_target_objects_rendered": len(rejected_samples),
        "artifacts": artifacts,
    }
    write_json_atomic(review_root / "index.json", index)
    return index


def _artifact_sha256s(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _resolve_configured_path(value: str, override: Path | None) -> Path:
    return resolve_project_path(override if override is not None else Path(value))


def run_construction(
    config_path: Path,
    source_root_override: Path | None = None,
    output_root_override: Path | None = None,
) -> Path:
    """Build and atomically publish the immutable Phase 2B canonical layer."""
    resolved_config = resolve_project_path(config_path)
    config = load_phase2b_config(resolved_config)
    source_root = _resolve_configured_path(config["source_root"], source_root_override)
    output_root = _resolve_configured_path(config["output_root"], output_root_override)
    manifests_root = resolve_project_path(Path(config["phase2a_manifests_root"]))
    parent_root = (
        resolve_project_path(Path(config["parent_dataset_root"]))
        if config.get("parent_dataset_root") else None
    )
    teacher_path = (
        resolve_project_path(Path(config["external_teacher_video"]["path"]))
        if config.get("external_teacher_video") else None
    )
    if output_root.exists():
        raise RDD2022Error(
            f"Canonical dataset version already exists and will not be overwritten: {output_root}"
        )
    if not source_root.is_dir():
        raise RDD2022Error(f"Immutable RDD2022 raw source is missing: {source_root}")
    if parent_root is not None and not parent_root.is_dir():
        raise RDD2022Error(f"Immutable parent dataset is missing: {parent_root}")
    if parent_root is not None and output_root == parent_root:
        raise RDD2022Error("Phase 2B.1 output must not overwrite its parent dataset.")
    if teacher_path is not None and not teacher_path.is_file():
        raise RDD2022Error(f"External teacher video is missing: {teacher_path}")
    parent_before = _content_tree_fingerprint(parent_root) if parent_root else None
    teacher_before = sha256_file(teacher_path) if teacher_path else None
    expected_teacher_sha256 = (
        str(config["external_teacher_video"].get("expected_sha256", ""))
        if teacher_path else ""
    )
    if expected_teacher_sha256 and teacher_before != expected_teacher_sha256:
        raise RDD2022Error("Teacher AVI checksum differs from the approved Phase 2B.1 value.")
    acquisition_path = manifests_root / "acquisition_manifest.json"
    integrity_path = manifests_root / "integrity_report.json"
    try:
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        phase2a_integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RDD2022Error("Could not read completed Phase 2A provenance reports.") from exc
    if acquisition.get("status") != "completed":
        raise RDD2022Error("Phase 2A acquisition is not completed.")
    raw_before = _tree_fingerprint(source_root)
    if raw_before != phase2a_integrity.get("raw_tree_after"):
        raise RDD2022Error("Raw RDD2022 fingerprint differs from the completed Phase 2A audit.")

    now = _utc_now()
    phase_name = "phase2b1" if config.get("dataset_version") == "1.1.0" else "phase2b"
    run_id = f"rdd2022_{phase_name}_{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    staging = output_root.parent / f".{output_root.name}-{run_id}"
    if staging.exists():
        raise RDD2022Error(f"Unexpected Phase 2B staging path already exists: {staging}")
    staging.mkdir(parents=True)
    cv2 = _load_opencv()
    if int(config.get("source_scan_worker_count", 1)) > 1:
        cv2.setNumThreads(1)
    try:
        labelled: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []
        official_test: list[dict[str, Any]] = []
        for country in config["countries"]:
            country_records, country_objects = _scan_labelled_country(
                country, source_root / country, config, cv2
            )
            labelled.extend(country_records)
            objects.extend(country_objects)
            official_test.extend(
                _scan_official_test_country(country, source_root / country, config, cv2)
            )
        labelled.sort(key=lambda item: item["canonical_image_id"])
        objects.sort(key=lambda item: item["canonical_object_id"])
        official_test.sort(key=lambda item: item["canonical_image_id"])

        labelled_country_counts = Counter(item["country"] for item in labelled)
        official_country_counts = Counter(item["country"] for item in official_test)
        raw_target_counts = Counter(
            item["original_raw_class_string"]
            for item in objects if item["canonical_action"] == TARGET_ACTION
        )
        expected = config["expected_source"]
        if dict(labelled_country_counts) != expected["labelled_images_by_country"]:
            raise RDD2022Error(
                f"Labelled source counts differ from the approved Phase 2B values: "
                f"{dict(labelled_country_counts)}"
            )
        if dict(official_country_counts) != expected["official_unlabelled_test_images_by_country"]:
            raise RDD2022Error(
                "Official unlabelled-test counts differ from the approved Phase 2B values."
            )
        if {name: raw_target_counts[name] for name in TARGET_CLASSES} != expected["target_objects"]:
            raise RDD2022Error("Raw target-object counts differ from approved Phase 2B values.")

        capture_groups = assign_capture_groups(labelled, config["grouping"])
        exact_groups = assign_exact_duplicate_groups([*labelled, *official_test])
        near_input = [item for item in labelled if item["inclusion_status"] == "retained"]
        near_candidates, near_groups, near_audit = generate_near_duplicate_candidates(
            near_input, config
        )
        parent_confirmed_relationships: set[tuple[str, str]] = set()
        if parent_root is not None:
            parent_candidates_path = parent_root / "groups" / "near_duplicate_candidates.jsonl"
            try:
                for line in parent_candidates_path.read_text(encoding="utf-8").splitlines():
                    relation = json.loads(line)
                    if relation.get("hard_split_constraint"):
                        parent_confirmed_relationships.add(
                            tuple(sorted((relation["first_image_id"], relation["second_image_id"])))
                        )
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                raise RDD2022Error("Could not verify parent near-duplicate constraints.") from exc
        current_confirmed_relationships = {
            tuple(sorted((item["first_image_id"], item["second_image_id"])))
            for item in near_candidates if item["hard_split_constraint"]
        }
        missing_parent_confirmed = sorted(
            parent_confirmed_relationships - current_confirmed_relationships
        )
        if missing_parent_confirmed:
            raise RDD2022Error(
                f"Phase 2B.1 lost {len(missing_parent_confirmed)} confirmed parent relationships."
            )
        near_audit["parent_confirmed_relationship_count"] = len(
            parent_confirmed_relationships
        )
        near_audit["all_parent_confirmed_relationships_retained"] = not missing_parent_confirmed
        split_plan_first = plan_grouped_split(
            labelled, near_candidates, config["split"], config["grouping"]
        )
        guards_first = apply_boundary_guards(
            labelled,
            capture_groups,
            split_plan_first["image_assignments"],
            config["grouping"],
        )
        split_plan_second = plan_grouped_split(
            labelled, near_candidates, config["split"], config["grouping"]
        )
        guards_second = apply_boundary_guards(
            labelled,
            capture_groups,
            split_plan_second["image_assignments"],
            config["grouping"],
        )
        deterministic_first = json.dumps(
            [split_plan_first["image_assignments"], guards_first], sort_keys=True
        ).encode()
        deterministic_second = json.dumps(
            [split_plan_second["image_assignments"], guards_second], sort_keys=True
        ).encode()
        determinism = {
            "planning_rerun_count": 2,
            "first_sha256": hashlib.sha256(deterministic_first).hexdigest(),
            "second_sha256": hashlib.sha256(deterministic_second).hexdigest(),
            "identical": deterministic_first == deterministic_second,
        }
        if not determinism["identical"]:
            raise RDD2022Error("Deterministic grouped-split planning rerun differed.")

        for item in labelled:
            image_id = item["canonical_image_id"]
            if item["inclusion_status"] == "quarantined":
                item["derived_split"] = None
            elif image_id in guards_first:
                item["inclusion_status"] = "guard_excluded"
                item["inclusion_exclusion_reason"] = guards_first[image_id]
                item["derived_split"] = None
            else:
                item["derived_split"] = split_plan_first["image_assignments"][image_id]
                item["inclusion_status"] = "split_assigned"
                item["inclusion_exclusion_reason"] = [
                    f"assigned_complete_group_to:{item['derived_split']}"
                ]
        _enrich_capture_groups(capture_groups, labelled)
        conflicts = split_conflict_audit(labelled, near_candidates)
        hard_conflict_count = sum(
            len(conflicts[name])
            for name in (
                "capture_group_conflicts",
                "exact_duplicate_group_conflicts",
                "confirmed_near_duplicate_conflicts",
            )
        )
        if hard_conflict_count:
            raise RDD2022Error("A hard grouping/duplicate split conflict was detected.")
        if not conflicts["all_required_countries_in_every_split"]:
            raise RDD2022Error("India and Japan are not both represented in every split.")
        if not conflicts["all_target_classes_in_every_split"]:
            raise RDD2022Error("All four target classes are not represented in every split.")

        assigned_by_split = {
            split_name: [item for item in labelled if item["derived_split"] == split_name]
            for split_name in SPLIT_NAMES
        }
        assigned_total = sum(len(items) for items in assigned_by_split.values())
        split_summaries = {
            split_name: _summarize_images(items)
            for split_name, items in assigned_by_split.items()
        }
        tolerance = float(config["split"]["ratio_tolerance_percentage_points"])
        ratio_audit = {}
        for split_name in SPLIT_NAMES:
            actual = len(assigned_by_split[split_name]) / assigned_total if assigned_total else 0.0
            target = float(config["split"]["ratios"][split_name])
            difference_points = (actual - target) * 100
            ratio_audit[split_name] = {
                "target_ratio": target,
                "actual_ratio": actual,
                "difference_percentage_points": difference_points,
                "within_configured_tolerance": abs(difference_points) <= tolerance,
            }

        _apply_final_target_dispositions(objects, labelled)
        target_reconciliation = _target_disposition_reconciliation(objects)
        india_d10_support = _india_d10_support(objects, labelled)
        annotation_qa = _annotation_qa(objects, labelled)
        quarantine_records = [
            {
                "canonical_image_id": item["canonical_image_id"],
                "country": item["country"],
                "original_filename": item["original_filename"],
                "raw_relative_image_path": item["raw_relative_image_path"],
                "raw_relative_xml_path": item["raw_relative_xml_path"],
                "reasons": item["inclusion_exclusion_reason"],
                "target_object_count": item["target_object_count"],
                "non_target_object_count": item["non_target_object_count"],
                "unknown_object_count": item["unknown_object_count"],
            }
            for item in labelled if item["inclusion_status"] == "quarantined"
        ]
        guard_records = [
            {
                "canonical_image_id": item["canonical_image_id"],
                "country": item["country"],
                "original_filename": item["original_filename"],
                "capture_group": item["capture_group"],
                "grouping_method": item["grouping_method"],
                "is_positive": item["is_positive"],
                "is_negative": item["is_negative"],
                "target_class_counts": item["target_class_counts"],
                "reasons": item["inclusion_exclusion_reason"],
            }
            for item in labelled if item["inclusion_status"] == "guard_excluded"
        ]
        rejected_object_records = [
            {
                "canonical_object_id": item["canonical_object_id"],
                "canonical_image_id": item["canonical_image_id"],
                "country": item["country"],
                "original_filename": item["original_filename"],
                "raw_relative_xml_path": item["raw_relative_xml_path"],
                "object_index_one_based": item["object_index_one_based"],
                "original_raw_class_string": item["original_raw_class_string"],
                "raw_voc_coordinates_exact": item["raw_voc_coordinates_exact"],
                "canonical_xyxy_pixel_zero_based_half_open": item[
                    "canonical_xyxy_pixel_zero_based_half_open"
                ],
                "object_status": item["object_status"],
                "reason": item["object_status_reason"],
            }
            for item in objects
            if item["object_status"] in {REJECT_INVALID_STATUS, REJECT_TINY_STATUS}
        ]
        assignments = [
            {
                "canonical_image_id": item["canonical_image_id"],
                "country": item["country"],
                "original_filename": item["original_filename"],
                "inclusion_status": item["inclusion_status"],
                "derived_split": item["derived_split"],
                "capture_group": item["capture_group"],
                "exact_duplicate_group": item["exact_duplicate_group"],
                "confirmed_near_duplicate_group": item["confirmed_near_duplicate_group"],
                "reason": item["inclusion_exclusion_reason"],
            }
            for item in labelled
        ]

        manifests_directory = staging / "manifests"
        groups_directory = staging / "groups"
        splits_directory = staging / "splits"
        reports_directory = staging / "reports"
        _write_jsonl(
            manifests_directory / "images.jsonl",
            (_clean_image_record(item) for item in labelled),
        )
        _write_jsonl(manifests_directory / "objects.jsonl", objects)
        _write_jsonl(
            manifests_directory / "official_unlabelled_test.jsonl", official_test
        )
        _write_jsonl(manifests_directory / "quarantine.jsonl", quarantine_records)
        _write_jsonl(
            manifests_directory / "rejected_objects.jsonl", rejected_object_records
        )
        _write_jsonl(groups_directory / "capture_groups.jsonl", capture_groups)
        _write_jsonl(
            groups_directory / "duplicate_groups.jsonl",
            sorted([*exact_groups, *near_groups], key=lambda item: item["duplicate_group"]),
        )
        _write_jsonl(
            groups_directory / "near_duplicate_candidates.jsonl", near_candidates
        )
        _write_jsonl(splits_directory / "assignments.jsonl", assignments)
        _write_jsonl(splits_directory / "guard_exclusions.jsonl", guard_records)
        for split_name in SPLIT_NAMES:
            text_path = splits_directory / f"{split_name}.txt"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(
                "".join(
                    f"{item['canonical_image_id']}\n"
                    for item in sorted(
                        assigned_by_split[split_name],
                        key=lambda value: value["canonical_image_id"],
                    )
                ),
                encoding="utf-8",
                newline="\n",
            )

        eligible_before_guards = [
            item for item in labelled if item["inclusion_status"] != "quarantined"
        ]
        guard_images = [
            item for item in labelled if item["inclusion_status"] == "guard_excluded"
        ]
        guard_summary = _guard_loss_summary(guard_images, len(eligible_before_guards))
        country_share_audit = _country_share_audit(
            assigned_by_split,
            eligible_before_guards,
            float(config["split"].get("country_share_tolerance_percentage_points", 100.0)),
        )
        grouping_audit = {
            "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
            "grouping_wording": config["split"]["description"],
            "capture_group_count": len(capture_groups),
            "capture_groups_by_method": _counter_dict(
                Counter(item["grouping_method"] for item in capture_groups)
            ),
            "images_by_grouping_method": _counter_dict(
                Counter(item["grouping_method"] for item in labelled)
            ),
            "images_by_grouping_confidence": _counter_dict(
                Counter(item["grouping_confidence"] for item in labelled)
            ),
            "group_size_statistics": _group_size_statistics(capture_groups),
            "numeric_block_proxy_is_route_capture_session_or_location_proof": False,
            "numeric_block_proxy_meaning": "low-confidence numeric grouping constraint",
            "timestamp_guard_seconds": int(config["grouping"]["timestamp_guard_seconds"]),
            "numeric_block_guard_ids": int(config["grouping"]["filename_guard_numeric_ids"]),
            "exact_duplicate_cluster_count": len(exact_groups),
            "near_duplicate_audit": near_audit,
            "guard_excluded_image_count": len(guard_records),
            "guard_loss": guard_summary,
            "split_optimization": split_plan_first["optimization_audit"],
            "hard_split_conflicts": conflicts,
        }
        write_json_atomic(groups_directory / "grouping_audit.json", grouping_audit)

        source_summary = _summarize_images(labelled)
        retained_before_guards = eligible_before_guards
        assigned_records = [item for item in labelled if item["derived_split"]]
        construction_report = {
            "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
            "run_id": run_id,
            "constructed_utc": now.isoformat(),
            "source_labelled_images_examined": len(labelled),
            "source_official_unlabelled_images_registered": len(official_test),
            "source_objects_examined": len(objects),
            "raw_target_objects_examined": {
                name: int(raw_target_counts[name]) for name in TARGET_CLASSES
            },
            "raw_target_objects_examined_total": sum(raw_target_counts.values()),
            "target_object_disposition_reconciliation": target_reconciliation,
            "india_d10_support": india_d10_support,
            "zero_edge_objects_normalized": annotation_qa[
                "source_zero_edge_coordinate_count"
            ],
            "object_level_rejections": len(rejected_object_records),
            "source_image_summary": source_summary,
            "retained_before_boundary_guards": _summarize_images(retained_before_guards),
            "split_assigned_after_guards": _summarize_images(assigned_records),
            "quarantined_images": len(quarantine_records),
            "guard_excluded_images": len(guard_records),
            "guard_loss": guard_summary,
            "canonical_layer_references_raw_files": True,
            "source_images_or_xml_copied": False,
            "raw_images_resized": False,
            "yolo_export_created": False,
        }
        split_report = {
            "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
            "split_description": config["split"]["description"],
            "seed": int(config["split"]["seed"]),
            "split_summaries": split_summaries,
            "ratio_audit": ratio_audit,
            "guard_excluded_image_count": len(guard_records),
            "guard_loss": guard_summary,
            "capture_group_component_count": len(split_plan_first["components"]),
            "optimization_audit": split_plan_first["optimization_audit"],
            "country_share_audit": country_share_audit,
            "india_d10_support": india_d10_support,
            "conflict_audit": conflicts,
            "internal_test_assignment_frozen": True,
            "natural_country_distribution_preserved": country_share_audit[
                "all_splits_within_tolerance"
            ],
            "balancing_or_weighting_applied": bool(config["split"].get("optimization")),
        }

        raw_after = _tree_fingerprint(source_root)
        parent_after = _content_tree_fingerprint(parent_root) if parent_root else None
        teacher_after = sha256_file(teacher_path) if teacher_path else None
        parent_unchanged = parent_before == parent_after
        teacher_unchanged = teacher_before == teacher_after
        guard_loss_target = float(config["split"].get("guard_loss_max_percent", 100.0))
        guard_loss_within_target = (
            guard_summary["percentage_of_eligible_labelled_images"] <= guard_loss_target
        )
        validation_report = {
            "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
            "source_gates": {
                "exactly_18212_labelled_pairs_examined": len(labelled) == 18_212,
                "exactly_23301_raw_target_boxes_accounted_for": sum(raw_target_counts.values()) == 23_301,
                "country_labelled_counts": _counter_dict(labelled_country_counts),
                "official_unlabelled_counts": _counter_dict(official_country_counts),
            },
            "annotation_qa": annotation_qa,
            "grouping_qa": grouping_audit,
            "split_qa": split_report,
            "official_unlabelled_test_exclusion": {
                "registered_count": len(official_test),
                "all_derived_splits_null": all(item["derived_split"] is None for item in official_test),
                "excluded_from_training_validation_internal_test": True,
            },
            "teacher_video_exclusion": {
                "teacher_paths_referenced": False,
                "teacher_data_integrated": False,
                "baseline_scope": "public_data_only",
                "path": project_relative(teacher_path) if teacher_path else None,
                "sha256_before": teacher_before,
                "sha256_after": teacher_after,
                "unchanged": teacher_unchanged,
            },
            "parent_dataset_immutability": {
                "path": project_relative(parent_root) if parent_root else None,
                "before": parent_before,
                "after": parent_after,
                "byte_content_unchanged": parent_unchanged,
            },
            "raw_tree_before": raw_before,
            "raw_tree_after": raw_after,
            "raw_tree_unchanged": raw_before == raw_after,
            "deterministic_rerun": determinism,
            "target_object_reconciliation": target_reconciliation,
            "country_share_acceptance": country_share_audit,
            "guard_loss_acceptance": {
                "target_max_percent": guard_loss_target,
                "actual_percent": guard_summary[
                    "percentage_of_eligible_labelled_images"
                ],
                "within_target": guard_loss_within_target,
            },
            "phase2b_complete": (
                raw_before == raw_after
                and parent_unchanged
                and teacher_unchanged
                and determinism["identical"]
                and all(item["within_configured_tolerance"] for item in ratio_audit.values())
                and country_share_audit["all_splits_within_tolerance"]
                and guard_loss_within_target
                and target_reconciliation["raw_target_total"] == sum(raw_target_counts.values())
                and hard_conflict_count == 0
            ),
        }
        if raw_before != raw_after:
            raise RDD2022Error("Raw RDD2022 tree changed during canonical construction.")
        if not all(item["within_configured_tolerance"] for item in ratio_audit.values()):
            raise RDD2022Error(f"Grouped split ratios exceed configured tolerance: {ratio_audit}")
        if not parent_unchanged:
            raise RDD2022Error("The immutable parent dataset changed during construction.")
        if not teacher_unchanged:
            raise RDD2022Error("The external teacher AVI changed during construction.")
        write_json_atomic(reports_directory / "construction_report.json", construction_report)
        write_json_atomic(reports_directory / "split_report.json", split_report)
        write_json_atomic(reports_directory / "validation_report.json", validation_report)

        review_index = _create_review_artifacts(
            labelled, objects, near_candidates, staging, config, cv2
        )
        source_index_payload = "\n".join(
            f"{item['canonical_image_id']}\0{item['image_sha256']}\0{item.get('xml_sha256') or ''}"
            for item in sorted([*labelled, *official_test], key=lambda value: value["canonical_image_id"])
        ).encode()
        source_checksums = {
            "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
            "phase2a_acquisition_manifest": {
                "path": project_relative(acquisition_path),
                "sha256": sha256_file(acquisition_path),
            },
            "phase2a_integrity_report": {
                "path": project_relative(integrity_path),
                "sha256": sha256_file(integrity_path),
            },
            "raw_tree_metadata_fingerprint": raw_after,
            "canonical_source_content_index_sha256": hashlib.sha256(source_index_payload).hexdigest(),
            "phase2b_config": {
                "path": project_relative(resolved_config),
                "sha256": sha256_file(resolved_config),
            },
            "parent_dataset": {
                "path": project_relative(parent_root) if parent_root else None,
                "content_fingerprint": parent_after,
            },
            "external_teacher_video_not_used": {
                "path": project_relative(teacher_path) if teacher_path else None,
                "sha256": teacher_after,
            },
        }
        write_json_atomic(staging / "source_checksums.json", source_checksums)
        write_json_atomic(
            staging / "taxonomy.yaml",
            {
                "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
                "coordinate_note": "Class IDs apply only to retained target objects.",
                "classes": config["taxonomy"],
            },
        )
        write_json_atomic(
            staging / "class_policy.yaml",
            {
                "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
                **config["class_policy"],
                "object_status_policy": config.get("object_status_policy"),
                "frozen_special_case_policy": config.get("policy_revision"),
                "d01_mapped_to_d00": False,
                "d11_mapped_to_d10": False,
                "d0w0_meaning_inferred": False,
            },
        )
        dataset_card = {
            "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
            "dataset_id": config["dataset_id"],
            "dataset_version": config["dataset_version"],
            "parent_version": config.get("parent_version"),
            "parent_dataset_path": project_relative(parent_root) if parent_root else None,
            "run_id": run_id,
            "description": (
                "Reference-only canonical representation of the immutable RDD2022 India and "
                "Japan labelled data with a leakage-reduced grouped split."
            ),
            "source_dataset": config["source_dataset"],
            "source_dataset_version": config["source_dataset_version"],
            "countries": config["countries"],
            "canonical_image_storage": "raw_relative_path_reference_only",
            "taxonomy": config["taxonomy"],
            "class_policy_file": "class_policy.yaml",
            "grouping_statement": (
                "Low-confidence numeric-block constraints reduce accidental filename-adjacent "
                "splitting but do not identify routes, capture sessions, locations, or independent "
                "roads. This dataset claims only a leakage-reduced grouped split."
            ),
            "revision_summary": config.get("revision_summary"),
            "official_unlabelled_test_policy": (
                "Registered separately and excluded from train, validation, internal labelled test, "
                "threshold selection, pseudo-labelling, and debugging."
            ),
            "teacher_video_policy": "Excluded; Phase 2B baseline remains public-data-only.",
            "license_statements": acquisition["dataset"]["licenses"],
            "environment": {
                "python_used": sys.version.split()[0],
                "opencv_used": cv2.__version__,
                "authoritative_docs_python_target": "3.11",
                "environment_documentation_mismatch_reported_not_changed": True,
            },
            "counts": {
                "source_labelled_images": len(labelled),
                "official_unlabelled_test_images": len(official_test),
                "source_objects": len(objects),
                "raw_target_objects": sum(raw_target_counts.values()),
                "target_object_dispositions": target_reconciliation["totals"],
                "zero_edge_objects_normalized": annotation_qa[
                    "source_zero_edge_coordinate_count"
                ],
                "rejected_target_objects": len(rejected_object_records),
                "split_assigned_images": assigned_total,
                "quarantined_images": len(quarantine_records),
                "guard_excluded_images": len(guard_records),
            },
            "review_contact_sheet_count": review_index["artifact_count"],
            "limitations": [
                "Numeric blocks are low-confidence grouping constraints, not route/location evidence.",
                "Perceptual similarity generates candidates; uncertain candidates are warnings only.",
                "India-specific D10 validation and internal-test support is expected to remain very low.",
                "No image resizing, YOLO export, augmentation, sampling balance, or model work is part of this version.",
            ],
        }
        write_json_atomic(staging / "dataset_card.json", dataset_card)
        artifacts = _artifact_sha256s(staging)
        write_json_atomic(
            reports_directory / "artifact_checksums.json",
            {
                "schema_version": "1.1" if config.get("dataset_version") == "1.1.0" else "1.0",
                "checksums_excluding_this_file": artifacts,
            },
        )
        os.replace(staging, output_root)
    except Exception as exc:
        if staging.exists() and staging.is_relative_to(output_root.parent):
            shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, RDD2022Error):
            raise
        raise RDD2022Error(f"Phase 2B construction failed: {exc}") from exc
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the reference-only RDD2022 India/Japan canonical dataset and "
            "leakage-reduced grouped split."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output = run_construction(args.config, args.source_root, args.output_root)
    except RDD2022Error as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Canonical Phase 2B dataset: %s", project_relative(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
