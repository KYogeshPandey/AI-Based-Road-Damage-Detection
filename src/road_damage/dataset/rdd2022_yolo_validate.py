"""Independent validation for the derived RDD2022 v1.1 YOLO detection export."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_common import (  # noqa: E402
    PROJECT_ROOT,
    RDD2022Error,
    project_relative,
    sha256_file,
    write_json_atomic,
)


LOGGER = logging.getLogger("road_damage.dataset.rdd2022_yolo_validate")
SPLITS = ("train", "val", "test")
CLASS_IDS = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
CLASS_NAMES = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RDD2022Error(f"Could not read required JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RDD2022Error(f"JSON artifact must contain an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                values.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RDD2022Error(f"Could not read {path} at line {line_number}.") from exc
    return values


def _resolve(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _content_tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        file_hash = sha256_file(path)
        digest.update(
            f"{path.relative_to(root).as_posix()}\0{file_hash}\n".encode("utf-8")
        )
        count += 1
        total += path.stat().st_size
    return {"content_tree_sha256": digest.hexdigest(), "file_count": count, "total_bytes": total}


def _metadata_tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat_result = path.stat()
        digest.update(
            (
                f"{path.relative_to(root).as_posix()}\0{stat_result.st_size}\0"
                f"{stat_result.st_mtime_ns}\n"
            ).encode("utf-8")
        )
        count += 1
        total += stat_result.st_size
    return {
        "metadata_fingerprint_sha256": digest.hexdigest(),
        "file_count": count,
        "total_bytes": total,
    }


def _expected_data_yaml(config: dict[str, Any]) -> str:
    return (
        "# Generated from frozen RDD2022 India/Japan canonical dataset v1.1.0.\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        + "".join(
            f"  {item['id']}: {item['name']}\n" for item in config["taxonomy"]
        )
    )


def _expected_label_line(
    item: dict[str, Any], width: int, height: int, precision: int
) -> str:
    x1, y1, x2, y2 = (
        float(value) for value in item["canonical_xyxy_pixel_zero_based_half_open"]
    )
    values = (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
    return (
        f"{int(item['canonical_numeric_class'])} "
        + " ".join(f"{value:.{precision}f}" for value in values)
    )


def _aggregate_hash(items: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(items):
        digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _split_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['canonical_image_id']}\0{record['image_sha256']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _taxonomy_fingerprint(taxonomy: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        taxonomy, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plan_fingerprint(
    assignments: Sequence[dict[str, Any]],
    image_by_id: dict[str, dict[str, Any]],
    objects_by_image: dict[str, list[dict[str, Any]]],
    data_yaml: str,
    precision: int,
) -> str:
    digest = hashlib.sha256(data_yaml.encode("utf-8"))
    for assignment in assignments:
        image = image_by_id[assignment["canonical_image_id"]]
        split = assignment["derived_split"]
        filename = image["original_filename"]
        label_relative = f"labels/{split}/{Path(filename).stem}.txt"
        image_relative = f"images/{split}/{filename}"
        digest.update(
            (
                f"{image['canonical_image_id']}\0{split}\0{image_relative}\0"
                f"{label_relative}\0{image['image_sha256']}\n"
            ).encode("utf-8")
        )
        lines = [
            _expected_label_line(
                item, int(image["decoded_width"]), int(image["decoded_height"]), precision
            )
            for item in objects_by_image.get(image["canonical_image_id"], [])
        ]
        digest.update(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return digest.hexdigest()


def _validate_label_text(
    content: str,
    expected_content: str,
    precision: int,
) -> tuple[int, list[str]]:
    errors: list[str] = []
    invalid_lines = 0
    if content != expected_content:
        errors.append("content_does_not_exactly_match_canonical_conversion")
    if content and not content.endswith("\n"):
        errors.append("missing_final_newline")
    tolerance = 0.5 * 10 ** (-precision) + 1e-12
    for line_number, line in enumerate(content.splitlines(), start=1):
        parts = line.split()
        line_errors: list[str] = []
        if len(parts) != 5:
            line_errors.append("field_count")
        else:
            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                line_errors.append("non_numeric")
            else:
                if class_id not in CLASS_NAMES:
                    line_errors.append("class_id")
                if not all(math.isfinite(value) for value in values):
                    line_errors.append("non_finite")
                else:
                    x_center, y_center, width, height = values
                    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
                        line_errors.append("center_bounds")
                    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
                        line_errors.append("size_bounds")
                    if (
                        x_center - width / 2.0 < -tolerance
                        or y_center - height / 2.0 < -tolerance
                        or x_center + width / 2.0 > 1.0 + tolerance
                        or y_center + height / 2.0 > 1.0 + tolerance
                    ):
                        line_errors.append("box_bounds")
                if any(
                    "." not in value or len(value.rsplit(".", 1)[1]) != precision
                    for value in parts[1:]
                ):
                    line_errors.append("precision")
        if line_errors:
            invalid_lines += 1
            errors.append(f"line_{line_number}:" + ",".join(line_errors))
    return invalid_lines, errors


def _hash_source_and_export(
    expected: dict[str, Any], project_root: Path, export_root: Path
) -> dict[str, Any]:
    source = _resolve(str(expected["raw_relative_image_path"]), project_root)
    exported = export_root / expected["exported_image_path"]
    source_hash = sha256_file(source) if source.is_file() else None
    export_hash = sha256_file(exported) if exported.is_file() else None
    same_identity = False
    if source.is_file() and exported.is_file():
        try:
            same_identity = os.path.samefile(source, exported)
        except OSError:
            same_identity = False
    return {
        "canonical_image_id": expected["canonical_image_id"],
        "source_hash": source_hash,
        "export_hash": export_hash,
        "expected_hash": expected["image_sha256"],
        "same_identity": same_identity,
    }


def _files_with_extensions(root: Path, extensions: set[str]) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.suffix.casefold() in extensions
    }


def _directory_size(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def finalize_export_report(export_root: Path) -> dict[str, Any]:
    """Mark a construction report complete after a passing independent validation."""
    validation_path = export_root / "validation_report.json"
    validation = _read_json(validation_path)
    if validation.get("phase2c1_complete") is not True or validation.get("status") != "passed":
        raise RDD2022Error("Cannot finalize an export without a passing validation report.")
    report_path = export_root / "export_report.json"
    report = _read_json(report_path)
    report["status"] = "export_validated"
    report["validation"] = {
        "report_path": "validation_report.json",
        "report_sha256": sha256_file(validation_path),
        "validated_utc": validation["validated_utc"],
        "passed": True,
    }
    write_json_atomic(report_path, report)
    return report


def validate_export(
    config_path: Path,
    canonical_root: Path | None = None,
    export_root: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Independently validate every image, label, count, hash, and exclusion."""
    started = datetime.now(timezone.utc)
    config_path = (
        config_path.resolve()
        if config_path.is_absolute()
        else (PROJECT_ROOT / config_path).resolve()
    )
    config = _read_json(config_path)
    canonical = (
        canonical_root.resolve()
        if canonical_root is not None
        else _resolve(config["canonical_dataset_path"], project_root)
    )
    export = (
        export_root.resolve()
        if export_root is not None
        else _resolve(config["output_path"], project_root)
    )
    if not export.is_dir():
        raise RDD2022Error(f"YOLO export does not exist: {export}")
    validation_path = export / "validation_report.json"
    if validation_path.exists():
        raise RDD2022Error(f"Refusing to overwrite validation report: {validation_path}")

    errors: list[str] = []
    images = _read_jsonl(canonical / "manifests" / "images.jsonl")
    objects = _read_jsonl(canonical / "manifests" / "objects.jsonl")
    assignment_records = _read_jsonl(canonical / "splits" / "assignments.jsonl")
    assignments = [
        item
        for item in assignment_records
        if item.get("inclusion_status") == "split_assigned"
        and item.get("derived_split") in SPLITS
    ]
    quarantines = _read_jsonl(canonical / "manifests" / "quarantine.jsonl")
    guards = _read_jsonl(canonical / "splits" / "guard_exclusions.jsonl")
    official = _read_jsonl(canonical / "manifests" / "official_unlabelled_test.jsonl")
    image_by_id = {item["canonical_image_id"]: item for item in images}
    if len(image_by_id) != len(images):
        errors.append("duplicate_canonical_image_ids")
    assignment_ids = [item["canonical_image_id"] for item in assignments]
    if len(set(assignment_ids)) != len(assignment_ids):
        errors.append("duplicate_split_assignments")

    expected_taxonomy = [
        {"id": class_id, "raw_code": code, "name": CLASS_NAMES[class_id]}
        for code, class_id in CLASS_IDS.items()
    ]
    if config.get("taxonomy") != expected_taxonomy:
        errors.append("invalid_config_taxonomy")
    precision = int(config["label_decimal_places"])
    retained: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        if item.get("final_target_disposition") == "retained":
            code = str(item.get("original_raw_class_string"))
            if code not in CLASS_IDS or int(item.get("canonical_numeric_class", -1)) != CLASS_IDS[code]:
                errors.append(f"invalid_retained_class:{item.get('canonical_object_id')}")
            retained[item["canonical_image_id"]].append(item)
    for values in retained.values():
        values.sort(key=lambda item: (int(item["object_index_one_based"]), item["canonical_object_id"]))

    expected_records: list[dict[str, Any]] = []
    expected_image_paths: set[str] = set()
    expected_label_paths: set[str] = set()
    for assignment in assignments:
        canonical_id = assignment["canonical_image_id"]
        image = image_by_id.get(canonical_id)
        if image is None:
            errors.append(f"assignment_missing_image:{canonical_id}")
            continue
        split = str(assignment["derived_split"])
        if split not in SPLITS or image.get("derived_split") != split:
            errors.append(f"split_mismatch:{canonical_id}")
        filename = str(image["original_filename"])
        image_relative = f"images/{split}/{filename}"
        label_relative = f"labels/{split}/{Path(filename).stem}.txt"
        expected_image_paths.add(image_relative)
        expected_label_paths.add(label_relative)
        image_objects = retained.get(canonical_id, [])
        lines = [
            _expected_label_line(
                item,
                int(image["decoded_width"]),
                int(image["decoded_height"]),
                precision,
            )
            for item in image_objects
        ]
        expected_records.append(
            {
                **image,
                "split": split,
                "exported_image_path": image_relative,
                "exported_label_path": label_relative,
                "expected_label_text": "\n".join(lines) + ("\n" if lines else ""),
                "objects": image_objects,
            }
        )

    actual_images = _files_with_extensions(export / "images", IMAGE_EXTENSIONS)
    actual_images = {f"images/{value}" for value in actual_images}
    actual_labels = _files_with_extensions(export / "labels", {".txt"})
    actual_labels = {f"labels/{value}" for value in actual_labels}
    missing_images = sorted(expected_image_paths - actual_images)
    unexpected_images = sorted(actual_images - expected_image_paths)
    missing_labels = sorted(expected_label_paths - actual_labels)
    orphan_labels = sorted(actual_labels - expected_label_paths)
    if missing_images:
        errors.append(f"missing_images:{len(missing_images)}")
    if unexpected_images:
        errors.append(f"unexpected_images:{len(unexpected_images)}")
    if missing_labels:
        errors.append(f"missing_labels:{len(missing_labels)}")
    if orphan_labels:
        errors.append(f"orphan_labels:{len(orphan_labels)}")

    label_file_mismatches: list[str] = []
    invalid_label_lines = 0
    empty_labels = 0
    class_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for record in expected_records:
        label_path = export / record["exported_label_path"]
        if not label_path.is_file():
            continue
        content = label_path.read_text(encoding="utf-8")
        invalid, label_errors = _validate_label_text(
            content, record["expected_label_text"], precision
        )
        invalid_label_lines += invalid
        if label_errors:
            label_file_mismatches.append(
                f"{record['exported_label_path']}:{'|'.join(label_errors)}"
            )
        if not content:
            empty_labels += 1
        for item in record["objects"]:
            class_counts[record["split"]][item["original_raw_class_string"]] += 1
    if label_file_mismatches:
        errors.append(f"label_file_mismatches:{len(label_file_mismatches)}")

    hash_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(config.get("copy_workers", 4))) as executor:
        for result in executor.map(
            lambda record: _hash_source_and_export(record, project_root, export),
            expected_records,
        ):
            hash_results.append(result)
    source_hash_mismatches = [
        item["canonical_image_id"]
        for item in hash_results
        if item["source_hash"] != item["expected_hash"]
    ]
    export_hash_mismatches = [
        item["canonical_image_id"]
        for item in hash_results
        if item["export_hash"] != item["expected_hash"]
    ]
    shared_file_identities = [
        item["canonical_image_id"] for item in hash_results if item["same_identity"]
    ]
    if source_hash_mismatches:
        errors.append(f"source_image_hash_mismatches:{len(source_hash_mismatches)}")
    if export_hash_mismatches:
        errors.append(f"export_image_hash_mismatches:{len(export_hash_mismatches)}")
    if shared_file_identities:
        errors.append(f"shared_source_export_file_identities:{len(shared_file_identities)}")

    manifest = _read_json(export / "export_manifest.json")
    manifest_rows = manifest.get("images", [])
    if not isinstance(manifest_rows, list) or len(manifest_rows) != len(expected_records):
        errors.append("export_manifest_image_count")
        manifest_rows = []
    manifest_by_id = {
        item.get("canonical_image_id"): item
        for item in manifest_rows
        if isinstance(item, dict)
    }
    if len(manifest_by_id) != len(manifest_rows):
        errors.append("export_manifest_duplicate_image_ids")
    for record, hash_result in zip(expected_records, hash_results):
        row = manifest_by_id.get(record["canonical_image_id"])
        if row is None:
            continue
        if (
            row.get("derived_split") != record["split"]
            or row.get("exported_image_path") != record["exported_image_path"]
            or row.get("exported_label_path") != record["exported_label_path"]
            or row.get("source_image_sha256") != record["image_sha256"]
            or row.get("exported_image_sha256") != hash_result["export_hash"]
            or row.get("label_sha256")
            != sha256_file(export / record["exported_label_path"])
        ):
            errors.append(f"export_manifest_trace_mismatch:{record['canonical_image_id']}")

    split_counts = Counter(record["split"] for record in expected_records)
    positive_count = sum(bool(record["objects"]) for record in expected_records)
    observed_class_counts = {
        split: {code: class_counts[split][code] for code in CLASS_IDS} for split in SPLITS
    }
    total_class_counts = {
        code: sum(observed_class_counts[split][code] for split in SPLITS)
        for code in CLASS_IDS
    }
    expected = config["expected"]
    if {split: split_counts[split] for split in SPLITS} != expected["split_image_counts"]:
        errors.append("split_image_count_reconciliation")
    if observed_class_counts != expected["target_objects"]:
        errors.append("split_object_count_reconciliation")
    if total_class_counts != expected["total_target_objects"]:
        errors.append("total_object_count_reconciliation")
    if empty_labels != sum(int(value) for value in expected["negative_images"].values()):
        errors.append("negative_empty_label_count_reconciliation")
    if positive_count != int(expected["positive_images"]):
        errors.append("positive_image_count_reconciliation")

    exported_filenames = {record["original_filename"] for record in expected_records}
    quarantine_names = {item["original_filename"] for item in quarantines}
    guard_names = {item["original_filename"] for item in guards}
    official_names = {item["original_filename"] for item in official}
    exclusion_intersections = {
        "quarantine": sorted(exported_filenames & quarantine_names),
        "guard": sorted(exported_filenames & guard_names),
        "official_unlabelled_test": sorted(exported_filenames & official_names),
    }
    if any(exclusion_intersections.values()):
        errors.append("excluded_images_present")
    manifest_text = (export / "export_manifest.json").read_text(encoding="utf-8").casefold()
    data_yaml_text = (export / "data.yaml").read_text(encoding="utf-8")
    teacher_references = sum(
        token in manifest_text for token in ("dashcam_raw", "data/videos", "teacher_video")
    )
    if teacher_references:
        errors.append("teacher_reference_present")
    if data_yaml_text != _expected_data_yaml(config):
        errors.append("data_yaml_content")

    source_split_fingerprints = {
        split: _split_fingerprint(
            record for record in expected_records if record["split"] == split
        )
        for split in SPLITS
    }
    if manifest.get("source_split_fingerprints") != source_split_fingerprints:
        errors.append("source_split_fingerprints")
    taxonomy_hash = _taxonomy_fingerprint(config["taxonomy"])
    if manifest.get("taxonomy_sha256") != taxonomy_hash:
        errors.append("taxonomy_fingerprint")
    deterministic_plan_hash = _plan_fingerprint(
        assignments,
        image_by_id,
        retained,
        data_yaml_text,
        precision,
    )
    deterministic_record = manifest.get("deterministic_rerun", {})
    deterministic_ok = (
        deterministic_record.get("match") is True
        and deterministic_record.get("first_plan_sha256") == deterministic_plan_hash
        and deterministic_record.get("second_plan_sha256") == deterministic_plan_hash
    )
    if not deterministic_ok:
        errors.append("deterministic_rerun")

    image_hash_items = [
        (record["exported_image_path"], result["export_hash"] or "")
        for record, result in zip(expected_records, hash_results)
    ]
    label_hash_items = [
        (record["exported_label_path"], sha256_file(export / record["exported_label_path"]))
        for record in expected_records
        if (export / record["exported_label_path"]).is_file()
    ]
    generated = manifest.get("generated_artifact_hashes", {})
    if generated.get("image_tree_sha256") != _aggregate_hash(image_hash_items):
        errors.append("image_tree_fingerprint")
    if generated.get("label_tree_sha256") != _aggregate_hash(label_hash_items):
        errors.append("label_tree_fingerprint")
    if generated.get("data_yaml_sha256") != sha256_file(export / "data.yaml"):
        errors.append("data_yaml_fingerprint")

    visual_errors: list[str] = []
    visual_coverage: dict[str, Any] = {}
    visual_index_path = export / "visual_qa" / "index.json"
    if not config.get("visual_qa", {}).get("enabled", False):
        visual_coverage = {"disabled_by_config": True}
    elif not visual_index_path.is_file():
        visual_errors.append("missing_visual_qa_index")
    else:
        visual_index = _read_json(visual_index_path)
        sheets = visual_index.get("sheets", [])
        if visual_index.get("contact_sheet_count") != 4 or len(sheets) != 4:
            visual_errors.append("contact_sheet_count")
        all_samples = [
            sample
            for sheet in sheets
            for sample in sheet.get("samples", [])
        ]
        tags = {tag for sample in all_samples for tag in sample.get("tags", [])}
        representative_combinations = {
            (sample.get("split"), sample.get("country"), tag.split(":", 1)[1])
            for sample in all_samples
            for tag in sample.get("tags", [])
            if tag.startswith("class:")
        }
        required_combinations = {
            (split, country, code)
            for split in SPLITS
            for country in ("India", "Japan")
            for code in CLASS_IDS
        }
        required_tags = {
            "small_box", "large_box", "edge_touching", "zero_edge",
            "mixed_target_non_target_source", "negative_empty_label",
        }
        if representative_combinations != required_combinations:
            visual_errors.append("target_split_country_class_coverage")
        if not required_tags.issubset(tags):
            visual_errors.append("special_case_coverage")
        for sheet in sheets:
            sheet_path = export / sheet["path"]
            if not sheet_path.is_file() or sha256_file(sheet_path) != sheet.get("sha256"):
                visual_errors.append(f"sheet_hash:{sheet.get('path')}")
        if sha256_file(visual_index_path) != generated.get("visual_qa_index_sha256"):
            visual_errors.append("index_hash")
        visual_coverage = {
            "representative_split_country_class_combinations": len(representative_combinations),
            "required_representative_combinations": len(required_combinations),
            "tags": sorted(tags),
            "zero_edge_sample_count": sum("zero_edge" in sample.get("tags", []) for sample in all_samples),
        }
    if visual_errors:
        errors.extend(f"visual_qa:{value}" for value in visual_errors)

    canonical_now = _content_tree_fingerprint(canonical)
    parent = _resolve(config["parent_dataset_path"], project_root)
    raw_root = _resolve(config["raw_dataset_path"], project_root)
    parent_now = _content_tree_fingerprint(parent) if parent.is_dir() else None
    raw_now = _metadata_tree_fingerprint(raw_root) if raw_root.is_dir() else None
    protected_checks = {
        "canonical_v1_1": {
            "observed": canonical_now,
            "expected": config.get("canonical_content_fingerprint"),
            "unchanged": canonical_now == config.get("canonical_content_fingerprint"),
        },
        "canonical_v1_0": {
            "observed": parent_now,
            "expected": config.get("parent_content_fingerprint"),
            "unchanged": parent_now == config.get("parent_content_fingerprint"),
        },
        "raw_rdd2022": {
            "observed": raw_now,
            "expected": config.get("raw_metadata_fingerprint"),
            "unchanged": raw_now == config.get("raw_metadata_fingerprint"),
        },
    }
    for name, check in protected_checks.items():
        if check["expected"] is not None and not check["unchanged"]:
            errors.append(f"protected_source_changed:{name}")
    if manifest.get("canonical_content_fingerprint") != canonical_now:
        errors.append("manifest_canonical_fingerprint")

    report_file_count, report_prewrite_bytes = _directory_size(export)
    passed = not errors
    report = {
        "schema_version": "2C.1",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(
            (datetime.now(timezone.utc) - started).total_seconds(), 3
        ),
        "phase2c1_complete": passed,
        "status": "passed" if passed else "failed",
        "errors": errors,
        "image_label_pairing": {
            "expected_images": len(expected_image_paths),
            "actual_images": len(actual_images),
            "expected_labels": len(expected_label_paths),
            "actual_labels": len(actual_labels),
            "missing_images": missing_images[:100],
            "unexpected_images": unexpected_images[:100],
            "missing_labels": missing_labels[:100],
            "orphan_labels": orphan_labels[:100],
            "passed": not (missing_images or unexpected_images or missing_labels or orphan_labels),
        },
        "split_image_counts": {split: split_counts[split] for split in SPLITS},
        "positive_image_count": positive_count,
        "negative_empty_label_count": empty_labels,
        "target_object_counts": observed_class_counts,
        "total_target_object_counts": total_class_counts,
        "total_target_object_count": sum(total_class_counts.values()),
        "label_validation": {
            "invalid_yolo_label_line_count": invalid_label_lines,
            "label_file_mismatch_count": len(label_file_mismatches),
            "mismatch_samples": label_file_mismatches[:100],
            "only_class_ids": [0, 1, 2, 3],
            "fixed_decimal_places": precision,
        },
        "source_export_image_hashes": {
            "images_checked": len(hash_results),
            "source_hash_mismatch_count": len(source_hash_mismatches),
            "export_hash_mismatch_count": len(export_hash_mismatches),
            "shared_file_identity_count": len(shared_file_identities),
            "all_source_and_export_hashes_match": not (
                source_hash_mismatches or export_hash_mismatches
            ),
        },
        "exclusions": {
            "intersections": exclusion_intersections,
            "teacher_reference_count": teacher_references,
            "known_non_target_classes_exported_as_targets": [],
            "passed": not any(exclusion_intersections.values()) and teacher_references == 0,
        },
        "deterministic_rerun": {
            "independently_recomputed_plan_sha256": deterministic_plan_hash,
            "manifest_record": deterministic_record,
            "passed": deterministic_ok,
        },
        "visual_qa": {
            "index_path": "visual_qa/index.json",
            "coverage": visual_coverage,
            "errors": visual_errors,
            "passed": not visual_errors,
        },
        "protected_sources": protected_checks,
        "export_disk_usage_before_validation_report": {
            "file_count": report_file_count,
            "total_bytes": report_prewrite_bytes,
        },
    }
    write_json_atomic(validation_path, report)
    if not passed:
        raise RDD2022Error(
            "YOLO export validation failed: " + ", ".join(errors[:20])
        )
    finalize_export_report(export)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Phase 2C.1 YOLO export.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--export-root", type=Path)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Finalize export_report.json from an already passing validation report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.finalize_existing:
            config = _read_json(args.config.resolve())
            export_root = (
                args.export_root.resolve()
                if args.export_root is not None
                else _resolve(config["output_path"], PROJECT_ROOT)
            )
            finalized = finalize_export_report(export_root)
            LOGGER.info("Finalized export report: %s", project_relative(export_root))
            return 0 if finalized["status"] == "export_validated" else 1
        report = validate_export(args.config, args.canonical_root, args.export_root)
    except RDD2022Error as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info(
        "YOLO export validation passed for %d images.",
        report["image_label_pairing"]["expected_images"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
