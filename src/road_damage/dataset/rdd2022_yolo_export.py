"""Deterministic YOLO detection export from the frozen RDD2022 v1.1 layer."""

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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from road_damage.video.inspect_video import _load_opencv  # noqa: E402


LOGGER = logging.getLogger("road_damage.dataset.rdd2022_yolo_export")
SPLIT_NAMES = ("train", "val", "test")
CLASS_MAPPING = {
    "D00": (0, "D00_longitudinal_crack"),
    "D10": (1, "D10_transverse_crack"),
    "D20": (2, "D20_alligator_crack"),
    "D40": (3, "D40_pothole"),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ExportEntry:
    """One canonical image and its deterministic derived export payload."""

    canonical_image_id: str
    country: str
    original_filename: str
    source_image_path: Path
    source_image_sha256: str
    split: str
    exported_image_relative_path: str
    exported_label_relative_path: str
    width: int
    height: int
    image_category: str
    is_positive: bool
    label_text: str
    target_objects: tuple[dict[str, Any], ...]


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RDD2022Error(f"Required JSON artifact does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RDD2022Error(f"Could not read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RDD2022Error(f"JSON artifact must contain an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                records.append(value)
    except FileNotFoundError as exc:
        raise RDD2022Error(f"Required JSONL artifact does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RDD2022Error(
            f"Could not read JSONL artifact {path} at line {line_number}."
        ) from exc
    return records


def _resolve_from_project(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_export_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the dependency-free Phase 2C.1 config."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    config = _read_json(resolved)
    expected_taxonomy = [
        {"id": class_id, "raw_code": code, "name": name}
        for code, (class_id, name) in CLASS_MAPPING.items()
    ]
    try:
        split_names = tuple(config["split_names"])
        precision = int(config["label_decimal_places"])
        taxonomy = config["taxonomy"]
        copy_policy = config["image_copy_policy"]
        workers = int(config["copy_workers"])
        label_workers = int(config.get("label_write_workers", workers))
        expected = config["expected"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RDD2022Error("Phase 2C.1 configuration is missing required fields.") from exc
    if config.get("schema_version") != "2C.1":
        raise RDD2022Error("Phase 2C.1 schema_version must be '2C.1'.")
    if split_names != SPLIT_NAMES:
        raise RDD2022Error("YOLO export splits must be train, val, test in order.")
    if taxonomy != expected_taxonomy:
        raise RDD2022Error("YOLO export must preserve the exact four-class V1 taxonomy.")
    if precision < 6 or precision > 12:
        raise RDD2022Error("YOLO label precision must be between 6 and 12 decimals.")
    if copy_policy != "byte_identical_copy_no_recompression_no_hardlinks":
        raise RDD2022Error("YOLO images must be independent byte-identical copies.")
    if not 1 <= workers <= 16:
        raise RDD2022Error("copy_workers must be between 1 and 16.")
    if not 1 <= label_workers <= 32:
        raise RDD2022Error("label_write_workers must be between 1 and 32.")
    if set(expected["split_image_counts"]) != set(SPLIT_NAMES):
        raise RDD2022Error("Expected image counts must cover train, val, and test.")
    if set(expected["target_objects"]) != set(SPLIT_NAMES):
        raise RDD2022Error("Expected object counts must cover train, val, and test.")
    for split in SPLIT_NAMES:
        if set(expected["target_objects"][split]) != set(CLASS_MAPPING):
            raise RDD2022Error(f"Expected {split} counts must use exactly four classes.")
    if set(expected["total_target_objects"]) != set(CLASS_MAPPING):
        raise RDD2022Error("Expected total object counts must use exactly four classes.")
    return config


def canonical_xyxy_pixel_to_yolo(
    xyxy_pixel: Sequence[int | float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert canonical zero-based half-open pixels to normalized YOLO xywh."""
    if len(xyxy_pixel) != 4 or image_width <= 0 or image_height <= 0:
        raise RDD2022Error("Canonical box conversion received invalid dimensions.")
    x1, y1, x2, y2 = (float(value) for value in xyxy_pixel)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise RDD2022Error("Canonical box contains a non-finite coordinate.")
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        raise RDD2022Error("Canonical box exceeds the decoded image bounds.")
    if x2 <= x1 or y2 <= y1:
        raise RDD2022Error("Canonical box must have positive width and height.")
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    values = (x_center, y_center, box_width, box_height)
    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < box_width <= 1.0
        and 0.0 < box_height <= 1.0
    ):
        raise RDD2022Error("Converted YOLO box is outside normalized bounds.")
    tolerance = 1e-12
    if (
        x_center - box_width / 2.0 < -tolerance
        or y_center - box_height / 2.0 < -tolerance
        or x_center + box_width / 2.0 > 1.0 + tolerance
        or y_center + box_height / 2.0 > 1.0 + tolerance
    ):
        raise RDD2022Error("Converted YOLO box extends outside the image.")
    return values


def format_yolo_label_line(
    class_id: int,
    xyxy_pixel: Sequence[int | float],
    image_width: int,
    image_height: int,
    decimal_places: int,
) -> str:
    """Format one deterministic YOLO detection label line."""
    if class_id not in {0, 1, 2, 3}:
        raise RDD2022Error(f"Unsupported YOLO class ID: {class_id}")
    values = canonical_xyxy_pixel_to_yolo(xyxy_pixel, image_width, image_height)
    formatted = " ".join(f"{value:.{decimal_places}f}" for value in values)
    return f"{class_id} {formatted}"


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


def _metadata_tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat_result = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(
            f"{relative}\0{stat_result.st_size}\0{stat_result.st_mtime_ns}\n".encode(
                "utf-8"
            )
        )
        file_count += 1
        total_bytes += stat_result.st_size
    return {
        "metadata_fingerprint_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _check_canonical_artifacts(canonical_root: Path) -> None:
    checksums = _read_json(canonical_root / "reports" / "artifact_checksums.json").get(
        "checksums_excluding_this_file"
    )
    if not isinstance(checksums, dict):
        raise RDD2022Error("Canonical artifact checksum inventory is missing.")
    mismatches = [
        relative
        for relative, expected in checksums.items()
        if not (canonical_root / relative).is_file()
        or sha256_file(canonical_root / relative) != expected
    ]
    if mismatches:
        raise RDD2022Error(
            "Frozen canonical artifacts failed checksum validation: "
            + ", ".join(mismatches[:10])
        )


def _data_yaml(config: dict[str, Any]) -> str:
    lines = [
        "# Generated from frozen RDD2022 India/Japan canonical dataset v1.1.0.",
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {item['id']}: {item['name']}" for item in config["taxonomy"])
    return "\n".join(lines) + "\n"


def _split_fingerprint(entries: Iterable[ExportEntry]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry.canonical_image_id}\0{entry.source_image_sha256}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _taxonomy_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config["taxonomy"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plan_fingerprint(entries: Sequence[ExportEntry], data_yaml: str) -> str:
    digest = hashlib.sha256(data_yaml.encode("utf-8"))
    for entry in entries:
        digest.update(
            (
                f"{entry.canonical_image_id}\0{entry.split}\0"
                f"{entry.exported_image_relative_path}\0"
                f"{entry.exported_label_relative_path}\0"
                f"{entry.source_image_sha256}\n"
            ).encode("utf-8")
        )
        digest.update(entry.label_text.encode("utf-8"))
    return digest.hexdigest()


def build_export_plan(
    config: dict[str, Any], canonical_root: Path, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Build and fully reconcile an in-memory export plan without writing files."""
    card = _read_json(canonical_root / "dataset_card.json")
    if card.get("dataset_version") != config.get("canonical_dataset_version"):
        raise RDD2022Error("Canonical dataset version does not match the export config.")
    if card.get("taxonomy") != config["taxonomy"]:
        raise RDD2022Error("Canonical and export taxonomies differ.")

    images = _read_jsonl(canonical_root / "manifests" / "images.jsonl")
    assignment_records = _read_jsonl(canonical_root / "splits" / "assignments.jsonl")
    assignments = [
        item
        for item in assignment_records
        if item.get("inclusion_status") == "split_assigned"
        and item.get("derived_split") in SPLIT_NAMES
    ]
    objects = _read_jsonl(canonical_root / "manifests" / "objects.jsonl")
    image_by_id = {item["canonical_image_id"]: item for item in images}
    if len(image_by_id) != len(images):
        raise RDD2022Error("Canonical image IDs are not unique.")
    expected_assigned_ids = {
        item["canonical_image_id"]
        for item in images
        if item.get("inclusion_status") == "split_assigned"
    }
    assignment_ids = [item["canonical_image_id"] for item in assignments]
    if len(set(assignment_ids)) != len(assignment_ids):
        raise RDD2022Error("Canonical split assignments contain duplicate image IDs.")
    if set(assignment_ids) != expected_assigned_ids:
        raise RDD2022Error("Canonical split assignments do not reconcile with images.jsonl.")

    objects_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        if item.get("final_target_disposition") != "retained":
            continue
        code = str(item.get("original_raw_class_string"))
        if code not in CLASS_MAPPING:
            raise RDD2022Error(f"Retained object has unsupported target class: {code}")
        expected_id = CLASS_MAPPING[code][0]
        if int(item.get("canonical_numeric_class", -1)) != expected_id:
            raise RDD2022Error(f"Canonical class ID mismatch for {item['canonical_object_id']}.")
        objects_by_image[item["canonical_image_id"]].append(item)
    for image_objects in objects_by_image.values():
        image_objects.sort(
            key=lambda item: (int(item["object_index_one_based"]), item["canonical_object_id"])
        )

    precision = int(config["label_decimal_places"])
    entries: list[ExportEntry] = []
    collision_keys: set[tuple[str, str]] = set()
    for assignment in assignments:
        image = image_by_id[assignment["canonical_image_id"]]
        split = str(assignment["derived_split"])
        if split not in SPLIT_NAMES or image.get("derived_split") != split:
            raise RDD2022Error(f"Invalid split assignment for {assignment['canonical_image_id']}.")
        filename = str(image["original_filename"])
        if Path(filename).name != filename or Path(filename).suffix.casefold() not in IMAGE_EXTENSIONS:
            raise RDD2022Error(f"Unsafe or unsupported canonical image filename: {filename}")
        collision_key = (split, filename.casefold())
        if collision_key in collision_keys:
            raise RDD2022Error(f"Case-insensitive export filename collision: {split}/{filename}")
        collision_keys.add(collision_key)
        source_path = _resolve_from_project(str(image["raw_relative_image_path"]), project_root)
        if not source_path.is_file():
            raise RDD2022Error(f"Canonical source image is missing: {source_path}")
        width = int(image["decoded_width"])
        height = int(image["decoded_height"])
        target_objects = tuple(objects_by_image.get(image["canonical_image_id"], []))
        lines = [
            format_yolo_label_line(
                int(item["canonical_numeric_class"]),
                item["canonical_xyxy_pixel_zero_based_half_open"],
                width,
                height,
                precision,
            )
            for item in target_objects
        ]
        label_text = "\n".join(lines) + ("\n" if lines else "")
        if bool(image["is_positive"]) != bool(target_objects):
            raise RDD2022Error(
                f"Positive/negative status conflicts with retained objects: {filename}"
            )
        entries.append(
            ExportEntry(
                canonical_image_id=str(image["canonical_image_id"]),
                country=str(image["country"]),
                original_filename=filename,
                source_image_path=source_path,
                source_image_sha256=str(image["image_sha256"]),
                split=split,
                exported_image_relative_path=f"images/{split}/{filename}",
                exported_label_relative_path=f"labels/{split}/{Path(filename).stem}.txt",
                width=width,
                height=height,
                image_category=str(image["image_category"]),
                is_positive=bool(image["is_positive"]),
                label_text=label_text,
                target_objects=target_objects,
            )
        )

    image_counts = Counter(entry.split for entry in entries)
    negative_counts = Counter(entry.split for entry in entries if not entry.target_objects)
    object_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLIT_NAMES
    }
    for entry in entries:
        object_counts[entry.split].update(
            str(item["original_raw_class_string"]) for item in entry.target_objects
        )
    observed_images = {split: image_counts[split] for split in SPLIT_NAMES}
    observed_negatives = {split: negative_counts[split] for split in SPLIT_NAMES}
    observed_objects = {
        split: {code: object_counts[split][code] for code in CLASS_MAPPING}
        for split in SPLIT_NAMES
    }
    total_objects = {
        code: sum(observed_objects[split][code] for split in SPLIT_NAMES)
        for code in CLASS_MAPPING
    }
    expected = config["expected"]
    if observed_images != expected["split_image_counts"]:
        raise RDD2022Error(f"Split image reconciliation failed: {observed_images}")
    if observed_negatives != expected["negative_images"]:
        raise RDD2022Error(f"Negative-image reconciliation failed: {observed_negatives}")
    if observed_objects != expected["target_objects"]:
        raise RDD2022Error(f"Target-object reconciliation failed: {observed_objects}")
    if total_objects != expected["total_target_objects"]:
        raise RDD2022Error(f"Total target-object reconciliation failed: {total_objects}")
    if sum(total_objects.values()) != int(expected["total_target_object_count"]):
        raise RDD2022Error("Combined target-object count is incorrect.")
    positive_count = sum(bool(entry.target_objects) for entry in entries)
    if positive_count != int(expected["positive_images"]):
        raise RDD2022Error("Positive-image count is incorrect.")

    data_yaml = _data_yaml(config)
    return {
        "entries": entries,
        "data_yaml": data_yaml,
        "deterministic_plan_sha256": _plan_fingerprint(entries, data_yaml),
        "source_split_fingerprints": {
            split: _split_fingerprint(entry for entry in entries if entry.split == split)
            for split in SPLIT_NAMES
        },
        "taxonomy_sha256": _taxonomy_fingerprint(config),
        "counts": {
            "split_images": observed_images,
            "positive_images": positive_count,
            "negative_images": observed_negatives,
            "target_objects": observed_objects,
            "total_target_objects": total_objects,
            "total_target_object_count": sum(total_objects.values()),
        },
    }


def _copy_and_hash(entry: ExportEntry, staging: Path) -> tuple[str, str]:
    destination = staging / entry.exported_image_relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with entry.source_image_path.open("rb") as source, destination.open("xb") as target:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
    except OSError as exc:
        raise RDD2022Error(
            f"Could not copy source image for {entry.canonical_image_id}: {entry.source_image_path}"
        ) from exc
    if os.path.samefile(entry.source_image_path, destination):
        raise RDD2022Error(f"Export unexpectedly shares a file identity: {destination}")
    exported_hash = digest.hexdigest()
    if exported_hash != entry.source_image_sha256:
        raise RDD2022Error(f"Copied image hash mismatch: {destination}")
    return entry.canonical_image_id, exported_hash


def _write_label_and_hash(entry: ExportEntry, staging: Path) -> tuple[str, str]:
    destination = staging / entry.exported_label_relative_path
    payload = entry.label_text.encode("utf-8")
    try:
        with destination.open("xb") as target:
            target.write(payload)
    except OSError as exc:
        raise RDD2022Error(f"Could not write YOLO label: {destination}") from exc
    return entry.exported_label_relative_path, hashlib.sha256(payload).hexdigest()


def _aggregate_hash(items: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(items):
        digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _render_contact_sheet(
    title: str,
    samples: Sequence[dict[str, Any]],
    staging: Path,
    destination: Path,
    visual_config: dict[str, Any],
    cv2: Any,
    np: Any,
) -> None:
    thumb_width = int(visual_config["thumbnail_width"])
    thumb_height = int(visual_config["thumbnail_height"])
    columns = int(visual_config["columns"])
    caption_height = 64
    heading_height = 44
    rows = max(1, math.ceil(len(samples) / columns))
    canvas = np.full(
        (heading_height + rows * (thumb_height + caption_height), columns * thumb_width, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    colors = {0: (255, 80, 40), 1: (40, 180, 255), 2: (70, 220, 70), 3: (30, 30, 230)}
    for index, sample in enumerate(samples):
        row, column = divmod(index, columns)
        left = column * thumb_width
        top = heading_height + row * (thumb_height + caption_height)
        image_path = staging / sample["exported_image_path"]
        label_path = staging / sample["exported_label_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RDD2022Error(f"Visual QA could not decode exported image: {image_path}")
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_text, x_text, y_text, w_text, h_text = line.split()
            class_id = int(class_text)
            x_center, y_center, box_width, box_height = map(
                float, (x_text, y_text, w_text, h_text)
            )
            x1 = int(round((x_center - box_width / 2.0) * width))
            y1 = int(round((y_center - box_height / 2.0) * height))
            x2 = int(round((x_center + box_width / 2.0) * width))
            y2 = int(round((y_center + box_height / 2.0) * height))
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_id], 2)
            cv2.putText(
                image,
                str(class_id),
                (max(0, x1), max(14, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colors[class_id],
                1,
                cv2.LINE_AA,
            )
        scale = min(thumb_width / width, thumb_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
        image_left = left + (thumb_width - resized_width) // 2
        image_top = top + (thumb_height - resized_height) // 2
        canvas[image_top:image_top + resized_height, image_left:image_left + resized_width] = resized
        cv2.putText(
            canvas,
            sample["original_filename"][:42],
            (left + 6, top + thumb_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        caption = f"{sample['split']} {sample['country']} {'/'.join(sample['tags'])}"
        cv2.putText(
            canvas,
            caption[:52],
            (left + 6, top + thumb_height + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination),
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, int(visual_config["jpeg_quality"])],
    ):
        raise RDD2022Error(f"Could not write visual QA contact sheet: {destination}")


def _visual_sample(
    entry: ExportEntry, tags: Sequence[str], focus_object_id: str | None = None
) -> dict[str, Any]:
    return {
        "canonical_image_id": entry.canonical_image_id,
        "original_filename": entry.original_filename,
        "country": entry.country,
        "split": entry.split,
        "exported_image_path": entry.exported_image_relative_path,
        "exported_label_path": entry.exported_label_relative_path,
        "focus_object_id": focus_object_id,
        "tags": list(tags),
    }


def _generate_visual_qa(
    entries: Sequence[ExportEntry], staging: Path, config: dict[str, Any]
) -> dict[str, Any]:
    visual_config = config["visual_qa"]
    if not visual_config.get("enabled", False):
        return {"enabled": False, "contact_sheet_count": 0, "sheets": []}
    cv2 = _load_opencv()
    import numpy as np

    try:
        cv2.setNumThreads(1)
    except AttributeError:
        pass
    ordered = sorted(entries, key=lambda entry: (entry.split, entry.country, entry.canonical_image_id))
    representative: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        for country in ("India", "Japan"):
            for code in CLASS_MAPPING:
                match = next(
                    (
                        entry
                        for entry in ordered
                        if entry.split == split
                        and entry.country == country
                        and any(
                            item["original_raw_class_string"] == code
                            for item in entry.target_objects
                        )
                    ),
                    None,
                )
                if match is None:
                    raise RDD2022Error(
                        f"Visual QA has no {split}/{country}/{code} representative."
                    )
                representative.append(
                    _visual_sample(match, [f"class:{code}", "representative"])
                )

    flattened = [
        (
            (item["canonical_xyxy_pixel_zero_based_half_open"][2]
             - item["canonical_xyxy_pixel_zero_based_half_open"][0])
            * (item["canonical_xyxy_pixel_zero_based_half_open"][3]
               - item["canonical_xyxy_pixel_zero_based_half_open"][1]),
            entry,
            item,
        )
        for entry in ordered
        for item in entry.target_objects
    ]
    flattened.sort(key=lambda value: (value[0], value[2]["canonical_object_id"]))
    size_count = int(visual_config["size_examples_per_extreme"])
    size_samples = [
        _visual_sample(entry, [tag], item["canonical_object_id"])
        for tag, selected in (("small_box", flattened[:size_count]), ("large_box", flattened[-size_count:]))
        for _, entry, item in selected
    ]
    edge_values = [
        (entry, item)
        for _, entry, item in flattened
        if (
            item["canonical_xyxy_pixel_zero_based_half_open"][0] == 0
            or item["canonical_xyxy_pixel_zero_based_half_open"][1] == 0
            or item["canonical_xyxy_pixel_zero_based_half_open"][2] == entry.width
            or item["canonical_xyxy_pixel_zero_based_half_open"][3] == entry.height
        )
    ][: int(visual_config["edge_examples"])]
    size_samples.extend(
        _visual_sample(entry, ["edge_touching"], item["canonical_object_id"])
        for entry, item in edge_values
    )

    zero_values = [
        (entry, item)
        for entry in ordered
        for item in entry.target_objects
        if item.get("source_zero_edge_coordinate")
    ][: int(visual_config["zero_edge_examples"])]
    zero_samples = [
        _visual_sample(entry, ["zero_edge"], item["canonical_object_id"])
        for entry, item in zero_values
    ]
    mixed = [
        entry for entry in ordered
        if entry.image_category == "mixed_target_and_known_non_target"
    ][: int(visual_config["mixed_examples"])]
    negatives = [entry for entry in ordered if not entry.target_objects][
        : int(visual_config["negative_examples"])
    ]
    mixed_negative_samples = [
        *(_visual_sample(entry, ["mixed_target_non_target_source"]) for entry in mixed),
        *(_visual_sample(entry, ["negative_empty_label"]) for entry in negatives),
    ]

    sheet_specs = [
        ("targets_by_split_country_class", "Targets by Split, Country, and Class", representative),
        ("small_large_edge_boxes", "Small, Large, and Edge-Touching Boxes", size_samples),
        ("zero_edge_cases", "Corrected Zero-Edge Cases", zero_samples),
        ("mixed_and_negative_images", "Mixed-Source and Negative Images", mixed_negative_samples),
    ]
    sheets: list[dict[str, Any]] = []
    for stem, title, samples in sheet_specs:
        if not samples:
            raise RDD2022Error(f"Visual QA category has no samples: {stem}")
        relative = f"visual_qa/{stem}.jpg"
        _render_contact_sheet(
            title, samples, staging, staging / relative, visual_config, cv2, np
        )
        sheets.append(
            {
                "category": stem,
                "path": relative,
                "sha256": sha256_file(staging / relative),
                "sample_count": len(samples),
                "samples": samples,
            }
        )
    index = {
        "schema_version": "2C.1",
        "purpose": "YOLO label reconstruction visual QA",
        "contact_sheet_count": len(sheets),
        "sheets": sheets,
    }
    write_json_atomic(staging / "visual_qa" / "index.json", index)
    index["index_sha256"] = sha256_file(staging / "visual_qa" / "index.json")
    return index


def _directory_size(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def run_export(
    config_path: Path,
    canonical_root: Path | None = None,
    output_root: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Create a new, immutable-source YOLO export and provenance reports."""
    started = datetime.now(timezone.utc)
    config = load_export_config(config_path)
    canonical = (
        canonical_root.resolve()
        if canonical_root is not None
        else _resolve_from_project(config["canonical_dataset_path"], project_root)
    )
    output = (
        output_root.resolve()
        if output_root is not None
        else _resolve_from_project(config["output_path"], project_root)
    )
    parent = _resolve_from_project(config["parent_dataset_path"], project_root)
    raw_root = _resolve_from_project(config["raw_dataset_path"], project_root)
    if output.exists():
        raise RDD2022Error(f"Refusing to overwrite existing YOLO export: {output}")
    if not canonical.is_dir():
        raise RDD2022Error(f"Canonical dataset does not exist: {canonical}")
    if config.get("verify_canonical_artifact_checksums", True):
        _check_canonical_artifacts(canonical)

    canonical_before = _content_tree_fingerprint(canonical)
    parent_before = _content_tree_fingerprint(parent) if parent.is_dir() else None
    raw_before = _metadata_tree_fingerprint(raw_root) if raw_root.is_dir() else None
    for key, observed in (
        ("canonical_content_fingerprint", canonical_before),
        ("parent_content_fingerprint", parent_before),
        ("raw_metadata_fingerprint", raw_before),
    ):
        expected_fingerprint = config.get(key)
        if expected_fingerprint is not None and observed != expected_fingerprint:
            raise RDD2022Error(f"Pinned source fingerprint changed: {key}")

    first_plan = build_export_plan(config, canonical, project_root)
    second_plan = build_export_plan(config, canonical, project_root)
    deterministic = (
        first_plan["deterministic_plan_sha256"]
        == second_plan["deterministic_plan_sha256"]
    )
    if not deterministic:
        raise RDD2022Error("Independent export planning passes were not deterministic.")
    entries: list[ExportEntry] = first_plan["entries"]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    ).resolve()
    if staging.parent != output.parent.resolve():
        raise RDD2022Error("Export staging directory escaped the output parent.")
    try:
        for split in SPLIT_NAMES:
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
        (staging / "data.yaml").write_text(first_plan["data_yaml"], encoding="utf-8")
        with ThreadPoolExecutor(
            max_workers=int(config.get("label_write_workers", config["copy_workers"]))
        ) as executor:
            label_hashes = list(
                executor.map(lambda entry: _write_label_and_hash(entry, staging), entries)
            )

        exported_hash_by_id: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=int(config["copy_workers"])) as executor:
            for canonical_id, exported_hash in executor.map(
                lambda entry: _copy_and_hash(entry, staging), entries
            ):
                exported_hash_by_id[canonical_id] = exported_hash
        if len(exported_hash_by_id) != len(entries):
            raise RDD2022Error("Image copy results did not cover every export entry.")

        visual_qa = _generate_visual_qa(entries, staging, config)
        image_hashes = [
            (entry.exported_image_relative_path, exported_hash_by_id[entry.canonical_image_id])
            for entry in entries
        ]
        label_hash_by_path = dict(label_hashes)
        manifest_rows = [
            {
                "canonical_image_id": entry.canonical_image_id,
                "country": entry.country,
                "original_filename": entry.original_filename,
                "raw_relative_image_path": project_relative(entry.source_image_path),
                "derived_split": entry.split,
                "exported_image_path": entry.exported_image_relative_path,
                "exported_label_path": entry.exported_label_relative_path,
                "source_image_sha256": entry.source_image_sha256,
                "exported_image_sha256": exported_hash_by_id[entry.canonical_image_id],
                "label_sha256": label_hash_by_path[entry.exported_label_relative_path],
                "target_object_count": len(entry.target_objects),
                "target_class_counts": dict(
                    sorted(
                        Counter(
                            str(item["original_raw_class_string"])
                            for item in entry.target_objects
                        ).items()
                    )
                ),
                "is_positive": bool(entry.target_objects),
            }
            for entry in entries
        ]
        manifest = {
            "schema_version": "2C.1",
            "export_version": config["export_version"],
            "created_utc": _utc_now_text(),
            "canonical_source_version": config["canonical_dataset_version"],
            "canonical_source_path": project_relative(canonical),
            "canonical_content_fingerprint": canonical_before,
            "source_split_fingerprints": first_plan["source_split_fingerprints"],
            "taxonomy": config["taxonomy"],
            "taxonomy_sha256": first_plan["taxonomy_sha256"],
            "coordinate_convention": (
                "canonical_zero_based_half_open_xyxy_pixels_to_"
                "normalized_yolo_xy_center_y_center_width_height"
            ),
            "label_decimal_places": int(config["label_decimal_places"]),
            "image_copy_policy": config["image_copy_policy"],
            "counts": first_plan["counts"],
            "deterministic_rerun": {
                "first_plan_sha256": first_plan["deterministic_plan_sha256"],
                "second_plan_sha256": second_plan["deterministic_plan_sha256"],
                "match": deterministic,
            },
            "generated_artifact_hashes": {
                "data_yaml_sha256": sha256_file(staging / "data.yaml"),
                "image_tree_sha256": _aggregate_hash(image_hashes),
                "label_tree_sha256": _aggregate_hash(label_hashes),
                "visual_qa_index_sha256": visual_qa.get("index_sha256"),
                "visual_qa_contact_sheets": {
                    item["path"]: item["sha256"] for item in visual_qa.get("sheets", [])
                },
            },
            "images": manifest_rows,
        }
        write_json_atomic(staging / "export_manifest.json", manifest)

        canonical_after = _content_tree_fingerprint(canonical)
        parent_after = _content_tree_fingerprint(parent) if parent.is_dir() else None
        raw_after = _metadata_tree_fingerprint(raw_root) if raw_root.is_dir() else None
        immutable = (
            canonical_before == canonical_after
            and parent_before == parent_after
            and raw_before == raw_after
        )
        if not immutable:
            raise RDD2022Error("A protected source fingerprint changed during export.")
        file_count, bytes_before_report = _directory_size(staging)
        report = {
            "schema_version": "2C.1",
            "export_version": config["export_version"],
            "created_utc": _utc_now_text(),
            "duration_seconds": round(
                (datetime.now(timezone.utc) - started).total_seconds(), 3
            ),
            "status": "export_created_validation_pending",
            "output_path": project_relative(output),
            "counts": first_plan["counts"],
            "image_label_pairs_written": len(entries),
            "empty_label_files_written": sum(not entry.target_objects for entry in entries),
            "image_hashes_matching_canonical": len(exported_hash_by_id),
            "image_hash_mismatches": 0,
            "hardlinks_used": False,
            "images_recompressed_or_resized": False,
            "model_framework_installed_by_export": False,
            "model_weights_downloaded": False,
            "training_or_inference_performed": False,
            "protected_sources": {
                "canonical_v1_1_before": canonical_before,
                "canonical_v1_1_after": canonical_after,
                "canonical_v1_1_unchanged": canonical_before == canonical_after,
                "canonical_v1_0_before": parent_before,
                "canonical_v1_0_after": parent_after,
                "canonical_v1_0_unchanged": parent_before == parent_after,
                "raw_rdd2022_before": raw_before,
                "raw_rdd2022_after": raw_after,
                "raw_rdd2022_unchanged": raw_before == raw_after,
            },
            "deterministic_rerun": manifest["deterministic_rerun"],
            "visual_qa": {
                "enabled": visual_qa.get("enabled", True),
                "contact_sheet_count": visual_qa.get("contact_sheet_count", 0),
                "index_path": "visual_qa/index.json" if visual_qa.get("sheets") else None,
            },
            "pre_report_artifact_count": file_count,
            "pre_report_disk_bytes": bytes_before_report,
        }
        write_json_atomic(staging / "export_report.json", report)
        os.replace(staging, output)
    except Exception as exc:
        if staging.exists() and staging.parent == output.parent.resolve():
            shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, RDD2022Error):
            raise
        raise RDD2022Error(f"YOLO export failed: {exc}") from exc
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export frozen RDD2022 v1.1 canonical records to YOLO detection format."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output = run_export(args.config, args.canonical_root, args.output_root)
    except RDD2022Error as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("YOLO detection export: %s", project_relative(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
