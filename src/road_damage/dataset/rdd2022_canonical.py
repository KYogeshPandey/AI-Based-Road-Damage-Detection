"""Pure canonicalization, grouping, duplicate, and split logic for RDD2022 Phase 2B."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

from road_damage.dataset.rdd2022_common import RDD2022Error, TARGET_CLASSES


SPLIT_NAMES = ("train", "val", "test")
TARGET_ACTION = "retain_target"
NON_TARGET_ACTION = "known_non_target"
UNKNOWN_ACTION = "quarantine_unknown"
REJECT_INVALID_STATUS = "reject_invalid"
REJECT_TINY_STATUS = "reject_tiny"
QUARANTINE_AMBIGUOUS_STATUS = "quarantine_ambiguous"
WHOLE_IMAGE_QUARANTINE_STATUS = "whole_image_quarantine"
FILENAME_NUMBER = re.compile(r"^(.*?)(\d+)$")


class UnionFind:
    """Small deterministic disjoint-set helper for grouping constraints."""

    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        lower, higher = sorted((first_root, second_root))
        self.parent[higher] = lower


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:length]}"


def load_phase2b_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RDD2022Error(f"Phase 2B configuration does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RDD2022Error(f"Could not read Phase 2B configuration: {path}") from exc
    validate_phase2b_config(config)
    return config


def validate_phase2b_config(config: dict[str, Any]) -> None:
    try:
        taxonomy = config["taxonomy"]
        policy = config["class_policy"]
        ratios = config["split"]["ratios"]
        countries = tuple(config["countries"])
    except (KeyError, TypeError) as exc:
        raise RDD2022Error("Phase 2B configuration is missing required fields.") from exc
    expected_taxonomy = [
        (0, "D00", "D00_longitudinal_crack"),
        (1, "D10", "D10_transverse_crack"),
        (2, "D20", "D20_alligator_crack"),
        (3, "D40", "D40_pothole"),
    ]
    observed_taxonomy = [
        (int(item["id"]), str(item["raw_code"]), str(item["name"]))
        for item in taxonomy
    ]
    if observed_taxonomy != expected_taxonomy:
        raise RDD2022Error("Phase 2B taxonomy must preserve the fixed four V1 labels.")
    if countries != ("India", "Japan"):
        raise RDD2022Error("Phase 2B source countries must be India and Japan only.")
    if tuple(policy["direct_target_classes"]) != TARGET_CLASSES:
        raise RDD2022Error("Direct target-class order must be D00/D10/D20/D40.")
    if set(policy["known_non_target_classes"]) != {"D01", "D11", "D43", "D44", "D50"}:
        raise RDD2022Error("Known non-target class policy changed unexpectedly.")
    if tuple(policy["quarantine_unknown_classes"]) != ("D0w0",):
        raise RDD2022Error("D0w0 must remain an unguessed quarantine class.")
    if tuple(ratios) != SPLIT_NAMES or not math.isclose(
        sum(float(ratios[name]) for name in SPLIT_NAMES), 1.0, abs_tol=1e-9
    ):
        raise RDD2022Error("Split ratios must be ordered train/val/test and sum to one.")
    if config["split"].get("description") != "leakage-reduced grouped split":
        raise RDD2022Error("Dataset wording must remain 'leakage-reduced grouped split'.")
    if config.get("dataset_version") == "1.1.0":
        revision = config.get("policy_revision", {})
        grouping = config.get("grouping", {})
        near = config.get("near_duplicates", {})
        if str(config.get("parent_version")) != "1.0":
            raise RDD2022Error("Phase 2B.1 must record parent_version 1.0.")
        if not config["annotation_validation"].get("allow_zero_lower_edge"):
            raise RDD2022Error("Phase 2B.1 must enable deterministic zero-edge handling.")
        if not 1 <= int(config.get("source_scan_worker_count", 0)) <= 16:
            raise RDD2022Error("Phase 2B.1 source scan workers must be between one and sixteen.")
        if int(grouping.get("filename_guard_numeric_ids", -1)) != 0:
            raise RDD2022Error("Phase 2B.1 filename/numeric-block guard must be zero.")
        if int(grouping.get("timestamp_guard_seconds", -1)) != 10:
            raise RDD2022Error("Phase 2B.1 timestamp guard must remain ten seconds.")
        if grouping.get("numeric_block_method_name") != "numeric_block_proxy":
            raise RDD2022Error("Phase 2B.1 must use numeric_block_proxy terminology.")
        if not near.get("evaluate_candidates_before_display_cap"):
            raise RDD2022Error("Near-duplicate candidates must be evaluated before display capping.")
        if float(config["split"].get("country_share_tolerance_percentage_points", -1)) != 5.0:
            raise RDD2022Error("Phase 2B.1 country-share tolerance must be five points.")
        if float(config["split"].get("ratio_tolerance_percentage_points", -1)) != 2.0:
            raise RDD2022Error("Phase 2B.1 split-ratio tolerance must be two points.")
        if float(config["split"].get("guard_loss_max_percent", -1)) != 5.0:
            raise RDD2022Error("Phase 2B.1 guard-loss target must be five percent.")
        expected_whole_image = {
            "India_006389.jpg", "Japan_012185.jpg", "Japan_004342.jpg",
            "India_005123.jpg", "India_009013.jpg",
        }
        if set(revision.get("whole_image_quarantine_filenames", {})) != expected_whole_image:
            raise RDD2022Error("Frozen Phase 2B.1 whole-image quarantines changed.")
        expected_zero_edge = {
            "Japan_003421.jpg", "Japan_011029.jpg", "Japan_011979.jpg",
            "Japan_006792.jpg", "Japan_012313.jpg", "Japan_000375.jpg",
            "Japan_000275.jpg", "Japan_007628.jpg", "Japan_002087.jpg",
            "Japan_003238.jpg", "Japan_010217.jpg", "Japan_010631.jpg",
            "Japan_007834.jpg",
        }
        if set(revision.get("approved_zero_edge_filenames", [])) != expected_zero_edge:
            raise RDD2022Error("Frozen Phase 2B.1 accepted zero-edge cases changed.")
        expected_rejections = {
            ("Japan_011217.jpg", 2, "D20", REJECT_INVALID_STATUS),
            ("Japan_001265.jpg", 4, "D20", REJECT_TINY_STATUS),
        }
        observed_rejections = {
            (
                str(item.get("original_filename")),
                int(item.get("object_index_one_based", -1)),
                str(item.get("raw_class")),
                str(item.get("object_status")),
            )
            for item in revision.get("rejected_objects", [])
        }
        if observed_rejections != expected_rejections:
            raise RDD2022Error("Frozen Phase 2B.1 object rejections changed.")


def canonical_action(raw_class: str, config: dict[str, Any]) -> str:
    policy = config["class_policy"]
    if raw_class in policy["direct_target_classes"]:
        return TARGET_ACTION
    if raw_class in policy["known_non_target_classes"]:
        return NON_TARGET_ACTION
    return UNKNOWN_ACTION


def _parse_integer_coordinate(value: str | None) -> tuple[int | None, list[str]]:
    if value is None or not value.strip():
        return None, ["malformed_coordinate"]
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation:
        return None, ["malformed_coordinate"]
    if not parsed.is_finite():
        return None, ["malformed_coordinate"]
    if parsed != parsed.to_integral_value():
        return None, ["non_integer_coordinate"]
    return int(parsed), []


def validate_voc_box(
    raw_xyxy: dict[str, str | None],
    image_width: int,
    image_height: int,
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate one one-based inclusive VOC box and derive lossless-safe forms."""
    parsed: dict[str, int | None] = {}
    flags: list[str] = []
    for name in ("xmin", "ymin", "xmax", "ymax"):
        parsed[name], coordinate_flags = _parse_integer_coordinate(raw_xyxy.get(name))
        flags.extend(coordinate_flags)
    canonical_xyxy: list[int] | None = None
    normalized_xyxy: list[float] | None = None
    projected: dict[str, Any] | None = None
    width_inclusive: int | None = None
    height_inclusive: int | None = None
    aspect_ratio: float | None = None
    source_zero_edge_coordinate = False
    if all(value is not None for value in parsed.values()):
        xmin = int(parsed["xmin"])
        ymin = int(parsed["ymin"])
        xmax = int(parsed["xmax"])
        ymax = int(parsed["ymax"])
        allow_zero_lower_edge = bool(validation_config.get("allow_zero_lower_edge", False))
        if any(value < 0 for value in (xmin, ymin, xmax, ymax)):
            flags.append("negative_coordinate")
        source_zero_edge_coordinate = allow_zero_lower_edge and (xmin == 0 or ymin == 0)
        if source_zero_edge_coordinate:
            flags.append("source_zero_edge_coordinate")
        if (
            xmax == 0
            or ymax == 0
            or ((xmin == 0 or ymin == 0) and not allow_zero_lower_edge)
        ):
            flags.append("zero_coordinate")
        if xmax < xmin or ymax < ymin:
            flags.append("inverted_box")
        canonical_xmin = 0 if xmin == 0 and allow_zero_lower_edge else xmin - 1
        canonical_ymin = 0 if ymin == 0 and allow_zero_lower_edge else ymin - 1
        width_inclusive = xmax - canonical_xmin
        height_inclusive = ymax - canonical_ymin
        if width_inclusive <= 0 or height_inclusive <= 0:
            flags.append("zero_area_box")
        minimum_lower = 0 if allow_zero_lower_edge else 1
        if (
            xmin < minimum_lower
            or ymin < minimum_lower
            or xmax > image_width
            or ymax > image_height
        ):
            flags.append("out_of_bounds")
        hard_geometry_flags = {
            "negative_coordinate", "zero_coordinate", "inverted_box",
            "zero_area_box", "out_of_bounds",
        }
        if not hard_geometry_flags.intersection(flags):
            canonical_xyxy = [canonical_xmin, canonical_ymin, xmax, ymax]
            normalized_xyxy = [
                canonical_xyxy[0] / image_width,
                canonical_xyxy[1] / image_height,
                canonical_xyxy[2] / image_width,
                canonical_xyxy[3] / image_height,
            ]
            aspect_ratio = width_inclusive / height_inclusive
            minimum = float(validation_config["unusual_aspect_ratio_min"])
            maximum = float(validation_config["unusual_aspect_ratio_max"])
            if aspect_ratio < minimum or aspect_ratio > maximum:
                flags.append("unusual_aspect_ratio")
            canvas = int(validation_config["projected_model_canvas_size"])
            scale = canvas / max(image_width, image_height)
            resized_width = image_width * scale
            resized_height = image_height * scale
            pad_x = (canvas - resized_width) / 2.0
            pad_y = (canvas - resized_height) / 2.0
            projected_width = width_inclusive * scale
            projected_height = height_inclusive * scale
            projected_area = projected_width * projected_height
            if projected_area < float(validation_config["tiny_projected_area_pixels"]):
                flags.append("extremely_tiny")
            projected = {
                "canvas_width": canvas,
                "canvas_height": canvas,
                "scale": scale,
                "padding_left": pad_x,
                "padding_top": pad_y,
                "bbox_xyxy_pixel_zero_based_half_open": [
                    canonical_xyxy[0] * scale + pad_x,
                    canonical_xyxy[1] * scale + pad_y,
                    canonical_xyxy[2] * scale + pad_x,
                    canonical_xyxy[3] * scale + pad_y,
                ],
                "bbox_width_pixels": projected_width,
                "bbox_height_pixels": projected_height,
                "bbox_area_pixels": projected_area,
            }
    return {
        "parsed_voc_xyxy_pixel_one_based_inclusive": parsed,
        "canonical_xyxy_pixel_zero_based_half_open": canonical_xyxy,
        "canonical_xyxy_normalized": normalized_xyxy,
        "source_width_pixels_inclusive": width_inclusive,
        "source_height_pixels_inclusive": height_inclusive,
        "aspect_ratio_width_over_height": aspect_ratio,
        "projected_640": projected,
        "source_zero_edge_coordinate": source_zero_edge_coordinate,
        "validation_flags": sorted(set(flags)),
    }


def parse_voc_annotation(
    path: Path,
    canonical_image_id: str,
    decoded_width: int,
    decoded_height: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Parse all original VOC objects while retaining raw class and coordinate strings."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return {
            "xml_parse_error": str(exc),
            "xml_width": None,
            "xml_height": None,
            "xml_depth": None,
            "xml_image_dimension_mismatch": None,
            "objects": [],
        }
    size = root.find("size")

    def integer_text(node: ET.Element | None, name: str) -> int | None:
        if node is None:
            return None
        value = node.findtext(name)
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    xml_width = integer_text(size, "width")
    xml_height = integer_text(size, "height")
    xml_depth = integer_text(size, "depth")
    dimension_mismatch = (
        xml_width is None
        or xml_height is None
        or xml_width != decoded_width
        or xml_height != decoded_height
    )
    target_map = {
        str(item["raw_code"]): int(item["id"]) for item in config["taxonomy"]
    }
    quarantine_flags = set(
        config["annotation_validation"]["quarantine_target_flags"]
    )
    object_records: list[dict[str, Any]] = []
    for index, object_node in enumerate(root.findall("object"), start=1):
        raw_class = (object_node.findtext("name") or "").strip() or "<missing>"
        action = canonical_action(raw_class, config)
        box = object_node.find("bndbox")
        raw_coordinates = {
            name: ((box.findtext(name) or "").strip() or None) if box is not None else None
            for name in ("xmin", "ymin", "xmax", "ymax")
        }
        validation = validate_voc_box(
            raw_coordinates,
            decoded_width,
            decoded_height,
            config["annotation_validation"],
        )
        if dimension_mismatch:
            validation["validation_flags"] = sorted(
                set(validation["validation_flags"]) | {"xml_image_dimension_mismatch"}
            )
        invalid_target_flags = sorted(
            quarantine_flags.intersection(validation["validation_flags"])
        ) if action == TARGET_ACTION else []
        if action == TARGET_ACTION:
            object_status = (
                QUARANTINE_AMBIGUOUS_STATUS if invalid_target_flags else TARGET_ACTION
            )
            object_status_reason = (
                "structurally_invalid_target_annotation"
                if invalid_target_flags else "valid_target_annotation"
            )
        elif action == NON_TARGET_ACTION:
            object_status = NON_TARGET_ACTION
            object_status_reason = "known_non_target_background_context"
        else:
            object_status = UNKNOWN_ACTION
            object_status_reason = "unknown_raw_class_not_guessed"
        flags = {
            name: (object_node.findtext(name) or "").strip() or None
            for name in ("difficult", "truncated", "occluded", "pose")
        }
        object_records.append(
            {
                "canonical_object_id": f"{canonical_image_id}:object:{index:04d}",
                "canonical_image_id": canonical_image_id,
                "object_index_one_based": index,
                "original_raw_class_string": raw_class,
                "canonical_action": action,
                "canonical_numeric_class": target_map.get(raw_class)
                if action == TARGET_ACTION else None,
                "source_coordinate_convention": config["annotation_validation"][
                    "source_voc_coordinate_convention"
                ],
                "raw_voc_coordinates_exact": raw_coordinates,
                **validation,
                "is_valid_target_annotation": (
                    action == TARGET_ACTION and not invalid_target_flags
                ),
                "invalid_target_flags": invalid_target_flags,
                "object_status": object_status,
                "object_status_reason": object_status_reason,
                "difficult": flags["difficult"],
                "truncated": flags["truncated"],
                "additional_voc_flags": {
                    "occluded": flags["occluded"], "pose": flags["pose"]
                },
            }
        )
    return {
        "xml_parse_error": None,
        "xml_width": xml_width,
        "xml_height": xml_height,
        "xml_depth": xml_depth,
        "xml_image_dimension_mismatch": dimension_mismatch,
        "objects": object_records,
    }


def apply_special_case_policy(
    parsed: dict[str, Any], original_filename: str, config: dict[str, Any]
) -> None:
    """Apply frozen version-specific dispositions without altering raw annotation fields."""
    revision = config.get("policy_revision")
    if not revision:
        return
    objects = parsed["objects"]
    indexed = {int(item["object_index_one_based"]): item for item in objects}
    for rule in revision.get("rejected_objects", []):
        if rule["original_filename"] != original_filename:
            continue
        index = int(rule["object_index_one_based"])
        item = indexed.get(index)
        if item is None or item["original_raw_class_string"] != rule["raw_class"]:
            raise RDD2022Error(
                f"Frozen object disposition no longer matches source XML: "
                f"{original_filename} object {index}."
            )
        item["object_status"] = str(rule["object_status"])
        item["object_status_reason"] = str(rule["reason"])
        item["is_valid_target_annotation"] = False
    whole_image = revision.get("whole_image_quarantine_filenames", {})
    if original_filename in whole_image:
        reason = str(whole_image[original_filename])
        for item in objects:
            if item["canonical_action"] == TARGET_ACTION:
                item["object_status"] = WHOLE_IMAGE_QUARANTINE_STATUS
                item["object_status_reason"] = reason


def classify_labelled_image(
    parsed: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    objects = parsed["objects"]
    target_objects = [item for item in objects if item["canonical_action"] == TARGET_ACTION]
    known_objects = [item for item in objects if item["canonical_action"] == NON_TARGET_ACTION]
    unknown_objects = [item for item in objects if item["canonical_action"] == UNKNOWN_ACTION]
    reasons: list[str] = []
    if parsed["xml_parse_error"] is not None:
        reasons.append("malformed_or_unreadable_xml")
    if unknown_objects:
        unknown_names = sorted({item["original_raw_class_string"] for item in unknown_objects})
        reasons.extend(f"unknown_raw_class:{name}" for name in unknown_names)
    quarantine_targets = [
        item for item in target_objects
        if item["object_status"] in {
            QUARANTINE_AMBIGUOUS_STATUS, WHOLE_IMAGE_QUARANTINE_STATUS
        }
    ]
    for item in quarantine_targets:
        if not config.get("policy_revision") and item["invalid_target_flags"]:
            reasons.append(
                f"invalid_target_annotation:{item['canonical_object_id']}:"
                + ",".join(item["invalid_target_flags"])
            )
        else:
            reasons.append(
                f"target_object_requires_whole_image_quarantine:"
                f"{item['canonical_object_id']}:{item['object_status_reason']}"
            )
    quarantined = bool(reasons)
    valid_target_count = sum(
        item["object_status"] == TARGET_ACTION for item in target_objects
    )
    rejected_target_count = sum(
        item["object_status"] in {REJECT_INVALID_STATUS, REJECT_TINY_STATUS}
        for item in target_objects
    )
    if quarantined:
        category = "quarantined"
        negative_subtype = None
    elif valid_target_count > 0 and known_objects:
        category = "mixed_target_and_known_non_target"
        negative_subtype = None
    elif valid_target_count > 0:
        category = "target_only"
        negative_subtype = None
    elif known_objects:
        category = "known_non_target_only"
        negative_subtype = "known_non_target_hard_negative"
    else:
        category = "empty_xml"
        negative_subtype = "empty_xml_background"
    return {
        "target_object_count": len(target_objects),
        "valid_target_object_count": valid_target_count,
        "rejected_target_object_count": rejected_target_count,
        "quarantined_target_object_count": len(quarantine_targets),
        "non_target_object_count": len(known_objects),
        "unknown_object_count": len(unknown_objects),
        "total_object_count": len(objects),
        "image_category": category,
        "is_positive": not quarantined and valid_target_count > 0,
        "is_negative": not quarantined and valid_target_count == 0,
        "negative_subtype": negative_subtype,
        "inclusion_status": "quarantined" if quarantined else "retained",
        "inclusion_exclusion_reason": reasons if reasons else ["eligible_after_annotation_validation"],
    }


def _read_ifd_value(
    payload: bytes, endian: str, value_type: int, count: int, field_offset: int
) -> bytes | None:
    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1}
    size = type_sizes.get(value_type, 0) * count
    if size <= 0:
        return None
    if size <= 4:
        return payload[field_offset:field_offset + size]
    try:
        data_offset = struct.unpack_from(f"{endian}I", payload, field_offset)[0]
    except struct.error:
        return None
    if data_offset < 0 or data_offset + size > len(payload):
        return None
    return payload[data_offset:data_offset + size]


def _parse_ifd(payload: bytes, endian: str, offset: int) -> dict[int, tuple[int, int, int]]:
    if offset < 0 or offset + 2 > len(payload):
        return {}
    try:
        count = struct.unpack_from(f"{endian}H", payload, offset)[0]
    except struct.error:
        return {}
    entries: dict[int, tuple[int, int, int]] = {}
    for index in range(count):
        entry_offset = offset + 2 + 12 * index
        if entry_offset + 12 > len(payload):
            break
        try:
            tag, value_type, value_count = struct.unpack_from(
                f"{endian}HHI", payload, entry_offset
            )
        except struct.error:
            continue
        entries[tag] = (value_type, value_count, entry_offset + 8)
    return entries


def _ascii_ifd_value(
    payload: bytes, endian: str, entry: tuple[int, int, int] | None
) -> str | None:
    if entry is None:
        return None
    value_type, count, offset = entry
    raw = _read_ifd_value(payload, endian, value_type, count, offset)
    if raw is None:
        return None
    return raw.rstrip(b"\0").decode("ascii", errors="replace").strip() or None


def read_exif_timestamp(path: Path) -> dict[str, Any]:
    """Read DateTimeOriginal from JPEG EXIF using only the standard library."""
    try:
        with path.open("rb") as source:
            if source.read(2) != b"\xff\xd8":
                return _empty_exif_result(False)
            while True:
                marker_start = source.read(1)
                if not marker_start:
                    return _empty_exif_result(False)
                if marker_start != b"\xff":
                    continue
                marker = source.read(1)
                while marker == b"\xff":
                    marker = source.read(1)
                if not marker or marker in (b"\xda", b"\xd9"):
                    return _empty_exif_result(False)
                length_bytes = source.read(2)
                if len(length_bytes) != 2:
                    return _empty_exif_result(False)
                length = int.from_bytes(length_bytes, "big")
                if length < 2:
                    return _empty_exif_result(False)
                segment = source.read(length - 2)
                if marker == b"\xe1" and segment.startswith(b"Exif\0\0"):
                    return _parse_exif_payload(segment[6:])
    except OSError:
        return _empty_exif_result(False)


def _empty_exif_result(exif_present: bool) -> dict[str, Any]:
    return {
        "exif_present": exif_present,
        "exif_timestamp_available": False,
        "exif_timestamp_raw": None,
        "exif_timestamp_value": None,
        "exif_timestamp_trustworthy": False,
        "exif_timestamp_sort_ms": None,
    }


def _parse_exif_payload(payload: bytes) -> dict[str, Any]:
    if len(payload) < 8 or payload[:2] not in (b"II", b"MM"):
        return _empty_exif_result(True)
    endian = "<" if payload[:2] == b"II" else ">"
    try:
        ifd0_offset = struct.unpack_from(f"{endian}I", payload, 4)[0]
    except struct.error:
        return _empty_exif_result(True)
    ifd0 = _parse_ifd(payload, endian, ifd0_offset)
    original = None
    offset_original = None
    exif_pointer = ifd0.get(0x8769)
    if exif_pointer:
        raw_pointer = _read_ifd_value(payload, endian, *exif_pointer)
        if raw_pointer and len(raw_pointer) >= 4:
            exif_offset = struct.unpack_from(f"{endian}I", raw_pointer, 0)[0]
            exif_ifd = _parse_ifd(payload, endian, exif_offset)
            original = _ascii_ifd_value(payload, endian, exif_ifd.get(0x9003))
            offset_original = _ascii_ifd_value(payload, endian, exif_ifd.get(0x9011))
    if original is None:
        original = _ascii_ifd_value(payload, endian, ifd0.get(0x0132))
    result = _empty_exif_result(True)
    result["exif_timestamp_available"] = original is not None
    result["exif_timestamp_raw"] = original
    if original is None:
        return result
    try:
        parsed = datetime.strptime(original, "%Y:%m:%d %H:%M:%S")
        if offset_original and re.fullmatch(r"[+-]\d{2}:\d{2}", offset_original):
            sign = 1 if offset_original[0] == "+" else -1
            hours, minutes = map(int, offset_original[1:].split(":"))
            offset_seconds = sign * (hours * 3600 + minutes * 60)
            epoch_ms = int(parsed.replace(tzinfo=timezone.utc).timestamp() * 1000) - offset_seconds * 1000
            value = parsed.strftime("%Y-%m-%dT%H:%M:%S") + offset_original
        else:
            epoch_ms = int(parsed.replace(tzinfo=timezone.utc).timestamp() * 1000)
            value = parsed.strftime("%Y-%m-%dT%H:%M:%S")
        result.update(
            {
                "exif_timestamp_value": value,
                "exif_timestamp_trustworthy": True,
                "exif_timestamp_sort_ms": epoch_ms,
            }
        )
    except (ValueError, OverflowError):
        pass
    return result


def filename_components(filename: str) -> tuple[str, int | None]:
    stem = Path(filename).stem
    match = FILENAME_NUMBER.match(stem)
    if match is None:
        return stem, None
    return match.group(1), int(match.group(2))


def assign_capture_groups(
    records: list[dict[str, Any]], grouping_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Assign timestamp groups where available, else low-confidence numeric blocks."""
    timestamp_sequences: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    fallback: list[dict[str, Any]] = []
    for record in records:
        prefix, numeric_id = filename_components(record["original_filename"])
        record["filename_prefix"] = prefix
        record["filename_numeric_id"] = numeric_id
        if record.get("exif_timestamp_trustworthy") and record.get(
            "exif_timestamp_sort_ms"
        ) is not None:
            timestamp_sequences[(record["country"], prefix)].append(record)
        else:
            fallback.append(record)
    metadata: dict[str, dict[str, Any]] = {}
    gap_ms = int(grouping_config["timestamp_gap_seconds"]) * 1000
    span_ms = int(grouping_config["timestamp_group_seconds"]) * 1000
    for (country, prefix), sequence in sorted(timestamp_sequences.items()):
        sequence.sort(key=lambda item: (item["exif_timestamp_sort_ms"], item["canonical_image_id"]))
        run_index = 0
        run_start = int(sequence[0]["exif_timestamp_sort_ms"])
        previous = run_start
        for record in sequence:
            timestamp_ms = int(record["exif_timestamp_sort_ms"])
            if timestamp_ms - previous > gap_ms:
                run_index += 1
                run_start = timestamp_ms
            subgroup = (timestamp_ms - run_start) // span_ms
            sequence_key = f"{country}|{prefix}|timestamp|run:{run_index}"
            group_id = stable_id("cg", sequence_key, subgroup)
            record.update(
                {
                    "capture_group": group_id,
                    "grouping_method": "timestamp",
                    "grouping_confidence": grouping_config["timestamp_confidence"],
                    "group_sequence_key": sequence_key,
                    "group_sequence_index": int(subgroup),
                }
            )
            metadata.setdefault(
                group_id,
                {
                    "capture_group": group_id,
                    "country": country,
                    "filename_prefix": prefix,
                    "grouping_method": "timestamp",
                    "grouping_confidence": grouping_config["timestamp_confidence"],
                    "sequence_key": sequence_key,
                    "sequence_index": int(subgroup),
                    "run_index": run_index,
                    "filename_bucket": None,
                    "image_ids": [],
                },
            )["image_ids"].append(record["canonical_image_id"])
            previous = timestamp_ms
    group_size = int(grouping_config["filename_numeric_group_size"])
    numeric_method = str(grouping_config.get("numeric_block_method_name", "filename_proxy"))
    numeric_confidence = grouping_config.get(
        "numeric_block_confidence", grouping_config.get("filename_proxy_confidence", "low")
    )
    for record in sorted(fallback, key=lambda item: item["canonical_image_id"]):
        prefix = record["filename_prefix"]
        numeric_id = record["filename_numeric_id"]
        if numeric_id is None:
            bucket: int | str = record["canonical_image_id"]
            sequence_index = 0
            sequence_key = f"{record['country']}|{prefix}|{numeric_method}_unparsed"
        else:
            bucket = numeric_id // group_size
            sequence_index = int(bucket)
            sequence_key = f"{record['country']}|{prefix}|{numeric_method}"
        group_id = stable_id("cg", sequence_key, bucket)
        record.update(
            {
                "capture_group": group_id,
                "grouping_method": numeric_method,
                "grouping_confidence": numeric_confidence,
                "group_sequence_key": sequence_key,
                "group_sequence_index": sequence_index,
            }
        )
        metadata.setdefault(
            group_id,
            {
                "capture_group": group_id,
                "country": record["country"],
                "filename_prefix": prefix,
                "grouping_method": numeric_method,
                "grouping_confidence": numeric_confidence,
                "sequence_key": sequence_key,
                "sequence_index": sequence_index,
                "run_index": None,
                "filename_bucket": int(bucket) if isinstance(bucket, int) else None,
                "image_ids": [],
            },
        )["image_ids"].append(record["canonical_image_id"])
    for item in metadata.values():
        item["image_ids"].sort()
        item["image_count"] = len(item["image_ids"])
    return [metadata[key] for key in sorted(metadata)]


def assign_exact_duplicate_groups(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hash[record["image_sha256"]].append(record)
        record["exact_duplicate_group"] = None
    groups: list[dict[str, Any]] = []
    for image_hash, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        group_id = stable_id("exact", image_hash)
        for member in members:
            member["exact_duplicate_group"] = group_id
        groups.append(
            {
                "duplicate_type": "exact_sha256",
                "duplicate_group": group_id,
                "image_sha256": image_hash,
                "member_image_ids": sorted(item["canonical_image_id"] for item in members),
                "member_count": len(members),
                "partitions": sorted({item["original_partition"] for item in members}),
                "hard_split_constraint": True,
            }
        )
    return groups


def generate_near_duplicate_candidates(
    records: Sequence[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Generate conservative perceptual candidates; confirmation needs gray-MAE evidence."""
    near = config["near_duplicates"]
    band_count = int(near["lsh_band_count"])
    band_bits = 64 // band_count
    maximum_bucket = int(near["maximum_bucket_size"])
    neighbor_window = int(near["maximum_neighbors_per_oversized_bucket_item"])
    by_id = {item["canonical_image_id"]: item for item in records}
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for item in records:
        value = int(item["perceptual_hash_dhash64"], 16)
        for band in range(band_count):
            shift = band * band_bits
            mask = (1 << band_bits) - 1
            buckets[(band, (value >> shift) & mask)].append(item["canonical_image_id"])
    pairs: set[tuple[str, str]] = set()
    oversized: list[dict[str, Any]] = []
    for key, member_ids in sorted(buckets.items()):
        unique_ids = sorted(set(member_ids))
        if len(unique_ids) > maximum_bucket:
            oversized.append({"band": key[0], "value": key[1], "member_count": len(unique_ids)})
            for index, first in enumerate(unique_ids):
                for second in unique_ids[index + 1:index + 1 + neighbor_window]:
                    pairs.add((first, second))
        else:
            for index, first in enumerate(unique_ids):
                for second in unique_ids[index + 1:]:
                    pairs.add((first, second))
    evaluated: list[dict[str, Any]] = []
    hamming_max = int(near["dhash_hamming_candidate_max"])
    for first_id, second_id in sorted(pairs):
        first = by_id[first_id]
        second = by_id[second_id]
        if first["image_sha256"] == second["image_sha256"]:
            continue
        distance = (
            int(first["perceptual_hash_dhash64"], 16)
            ^ int(second["perceptual_hash_dhash64"], 16)
        ).bit_count()
        if distance > hamming_max:
            continue
        first_gray = first["_similarity_gray_bytes"]
        second_gray = second["_similarity_gray_bytes"]
        mae = sum(abs(a - b) for a, b in zip(first_gray, second_gray)) / (
            255.0 * len(first_gray)
        )
        confirmed = (
            distance <= int(near["confirmed_hamming_max"])
            and mae <= float(near["confirmed_gray_mae_max"])
            and first["decoded_width"] == second["decoded_width"]
            and first["decoded_height"] == second["decoded_height"]
        )
        evaluated.append(
            {
                "first_image_id": first_id,
                "second_image_id": second_id,
                "dhash_hamming_distance": distance,
                "gray_32x32_mean_absolute_difference_normalized": mae,
                "relationship": "confirmed_near_duplicate" if confirmed else "candidate_warning",
                "hard_split_constraint": confirmed,
            }
        )
    evaluated.sort(
        key=lambda item: (
            item["dhash_hamming_distance"],
            item["gray_32x32_mean_absolute_difference_normalized"],
            item["first_image_id"],
            item["second_image_id"],
        )
    )
    limit = int(near["maximum_candidates_per_image"])
    if near.get("evaluate_candidates_before_display_cap", False):
        selected = evaluated
    else:
        per_image: Counter[str] = Counter()
        selected = []
        for item in evaluated:
            first_id = item["first_image_id"]
            second_id = item["second_image_id"]
            if per_image[first_id] >= limit or per_image[second_id] >= limit:
                continue
            selected.append(item)
            per_image[first_id] += 1
            per_image[second_id] += 1
    candidate_union = UnionFind(by_id)
    confirmed_union = UnionFind(by_id)
    for item in selected:
        candidate_union.union(item["first_image_id"], item["second_image_id"])
        if item["hard_split_constraint"]:
            confirmed_union.union(item["first_image_id"], item["second_image_id"])
    candidate_members: dict[str, list[str]] = defaultdict(list)
    confirmed_members: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(by_id):
        candidate_members[candidate_union.find(image_id)].append(image_id)
        confirmed_members[confirmed_union.find(image_id)].append(image_id)
    group_records: list[dict[str, Any]] = []
    for members in candidate_members.values():
        if len(members) < 2:
            by_id[members[0]]["near_duplicate_candidate_group"] = None
            continue
        group_id = stable_id("near_candidate", *members)
        for image_id in members:
            by_id[image_id]["near_duplicate_candidate_group"] = group_id
        group_records.append(
            {
                "duplicate_type": "near_duplicate_candidate_component",
                "duplicate_group": group_id,
                "member_image_ids": members,
                "member_count": len(members),
                "hard_split_constraint": False,
            }
        )
    for image_id in by_id:
        by_id[image_id].setdefault("near_duplicate_candidate_group", None)
        by_id[image_id]["confirmed_near_duplicate_group"] = None
    for members in confirmed_members.values():
        if len(members) < 2:
            continue
        group_id = stable_id("near_confirmed", *members)
        for image_id in members:
            by_id[image_id]["confirmed_near_duplicate_group"] = group_id
        group_records.append(
            {
                "duplicate_type": "confirmed_near_duplicate_component",
                "duplicate_group": group_id,
                "member_image_ids": members,
                "member_count": len(members),
                "hard_split_constraint": True,
            }
        )
    audit = {
        "method": near["method"],
        "lsh_pair_count_before_hamming_filter": len(pairs),
        "candidate_count_before_per_image_limit": len(evaluated),
        "total_candidates": len(evaluated),
        "automatically_evaluated_candidates": len(evaluated),
        "recorded_candidate_count": len(selected),
        "confirmed_candidate_count": sum(item["hard_split_constraint"] for item in selected),
        "warning_only_candidate_count": sum(
            not item["hard_split_constraint"] for item in selected
        ),
        "candidate_evaluation_preceded_display_cap": bool(
            near.get("evaluate_candidates_before_display_cap", False)
        ),
        "display_contact_sheet_pair_limit": int(near["maximum_review_pairs"]),
        "oversized_lsh_buckets": oversized,
        "per_image_candidate_limit": limit,
        "automatic_removal_or_merge_performed": False,
    }
    return selected, sorted(group_records, key=lambda item: item["duplicate_group"]), audit


def _feature_counts(records: Sequence[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in records:
        counts["images"] += 1
        counts[f"country:{item['country']}"] += 1
        counts["positive" if item["is_positive"] else "negative"] += 1
        for class_name, count in item["target_class_counts"].items():
            counts[f"class:{class_name}"] += int(count)
        if item["country"] == "Japan":
            counts[f"japan_resolution:{item['decoded_width']}x{item['decoded_height']}"] += 1
    return counts


def _feature_weight(name: str) -> float:
    if name == "images":
        return 12.0
    if name.startswith("class:"):
        return 4.0
    if name.startswith("country:"):
        return 3.0
    if name in ("positive", "negative"):
        return 2.0
    return 0.75


def plan_grouped_split(
    records: Sequence[dict[str, Any]],
    confirmed_candidates: Sequence[dict[str, Any]],
    split_config: dict[str, Any],
    grouping_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign complete capture/duplicate components with deterministic feature balancing."""
    eligible = sorted(
        (item for item in records if item["inclusion_status"] == "retained"),
        key=lambda item: item["canonical_image_id"],
    )
    by_id = {item["canonical_image_id"]: item for item in eligible}
    union = UnionFind(by_id)
    by_capture: dict[str, list[str]] = defaultdict(list)
    by_exact: dict[str, list[str]] = defaultdict(list)
    for item in eligible:
        by_capture[item["capture_group"]].append(item["canonical_image_id"])
        if item.get("exact_duplicate_group"):
            by_exact[item["exact_duplicate_group"]].append(item["canonical_image_id"])
    for groups in (by_capture, by_exact):
        for members in groups.values():
            for image_id in members[1:]:
                union.union(members[0], image_id)
    for relation in confirmed_candidates:
        if not relation["hard_split_constraint"]:
            continue
        first = relation["first_image_id"]
        second = relation["second_image_id"]
        if first in by_id and second in by_id:
            union.union(first, second)
    component_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_id, item in by_id.items():
        component_members[union.find(image_id)].append(item)
    components: list[dict[str, Any]] = []
    for members in component_members.values():
        member_ids = sorted(item["canonical_image_id"] for item in members)
        components.append(
            {
                "component_id": stable_id("split_component", *member_ids),
                "member_image_ids": member_ids,
                "capture_groups": sorted({item["capture_group"] for item in members}),
                "features": _feature_counts(members),
            }
        )
    seed = int(split_config["seed"])
    components.sort(
        key=lambda item: (
            -item["features"]["images"],
            hashlib.sha256(f"{seed}:{item['component_id']}".encode()).hexdigest(),
        )
    )
    ratios = {name: float(split_config["ratios"][name]) for name in SPLIT_NAMES}
    totals = _feature_counts(eligible)
    current = {name: Counter() for name in SPLIT_NAMES}
    component_assignment: dict[str, str] = {}

    def delta_score(split_name: str, features: Counter[str]) -> float:
        score = 0.0
        for feature_name, total in totals.items():
            if total <= 0:
                continue
            target = total * ratios[split_name]
            before = current[split_name][feature_name]
            after = before + features[feature_name]
            denominator = max(target, 1.0)
            score += _feature_weight(feature_name) * (
                ((after - target) / denominator) ** 2
                - ((before - target) / denominator) ** 2
            )
        return score

    for component in components:
        scored = [
            (
                delta_score(split_name, component["features"]),
                hashlib.sha256(
                    f"{seed}:{component['component_id']}:{split_name}".encode()
                ).hexdigest(),
                split_name,
            )
            for split_name in SPLIT_NAMES
        ]
        selected = min(scored)[2]
        component_assignment[component["component_id"]] = selected
        current[selected].update(component["features"])

    optimization_config = split_config.get("optimization")
    optimization_audit: dict[str, Any] = {
        "enabled": False,
        "moves_applied": 0,
        "passes_completed": 0,
    }
    if optimization_config and grouping_config is not None:
        component_by_image = {
            image_id: component["component_id"]
            for component in components
            for image_id in component["member_image_ids"]
        }
        timestamp_groups: dict[str, dict[str, Any]] = {}
        for item in eligible:
            if item.get("grouping_method") != "timestamp":
                continue
            group = timestamp_groups.setdefault(
                item["capture_group"],
                {
                    "sequence_key": item["group_sequence_key"],
                    "sequence_index": int(item["group_sequence_index"]),
                    "image_ids": [],
                },
            )
            group["image_ids"].append(item["canonical_image_id"])
        by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group in timestamp_groups.values():
            by_sequence[group["sequence_key"]].append(group)
        timestamp_guard_ms = int(grouping_config["timestamp_guard_seconds"]) * 1000
        boundary_edges: list[dict[str, Any]] = []
        for groups in by_sequence.values():
            groups.sort(key=lambda item: item["sequence_index"])
            for lower, upper in zip(groups, groups[1:]):
                if upper["sequence_index"] != lower["sequence_index"] + 1:
                    continue
                lower_component = component_by_image[lower["image_ids"][0]]
                upper_component = component_by_image[upper["image_ids"][0]]
                if lower_component == upper_component:
                    continue
                lower_edge = max(
                    int(by_id[image_id]["exif_timestamp_sort_ms"])
                    for image_id in lower["image_ids"]
                )
                upper_edge = min(
                    int(by_id[image_id]["exif_timestamp_sort_ms"])
                    for image_id in upper["image_ids"]
                )
                guarded = {
                    image_id for image_id in lower["image_ids"]
                    if int(by_id[image_id]["exif_timestamp_sort_ms"])
                    >= lower_edge - timestamp_guard_ms
                }
                guarded.update(
                    image_id for image_id in upper["image_ids"]
                    if int(by_id[image_id]["exif_timestamp_sort_ms"])
                    <= upper_edge + timestamp_guard_ms
                )
                boundary_edges.append(
                    {
                        "lower_component": lower_component,
                        "upper_component": upper_component,
                        "guard_image_ids": guarded,
                    }
                )

        weights = optimization_config["weights"]
        ratio_tolerance = float(split_config["ratio_tolerance_percentage_points"])
        country_tolerance = float(
            split_config["country_share_tolerance_percentage_points"]
        )
        global_india_share = (
            100.0 * totals["country:India"] / totals["images"]
            if totals["images"] else 0.0
        )

        edges_by_component: dict[str, list[int]] = defaultdict(list)
        active_guard_memberships: Counter[str] = Counter()
        transition_count = 0
        for edge_index, edge in enumerate(boundary_edges):
            edges_by_component[edge["lower_component"]].append(edge_index)
            edges_by_component[edge["upper_component"]].append(edge_index)
            if (
                component_assignment[edge["lower_component"]]
                != component_assignment[edge["upper_component"]]
            ):
                transition_count += 1
                active_guard_memberships.update(edge["guard_image_ids"])
        projected_guard_images = sum(value > 0 for value in active_guard_memberships.values())

        def objective(
            feature_counts: dict[str, Counter[str]],
            transitions: int,
            guarded_image_count: int,
        ) -> dict[str, Any]:
            ratio_deviation: dict[str, float] = {}
            country_deviation: dict[str, float] = {}
            hard_excess = 0.0
            score = 0.0
            for split_name in SPLIT_NAMES:
                image_count = feature_counts[split_name]["images"]
                ratio_points = (
                    100.0 * image_count / totals["images"]
                    - 100.0 * ratios[split_name]
                )
                ratio_deviation[split_name] = ratio_points
                india_share = (
                    100.0 * feature_counts[split_name]["country:India"] / image_count
                    if image_count else 0.0
                )
                country_points = india_share - global_india_share
                country_deviation[split_name] = country_points
                score += float(weights["split_ratio_pp_squared"]) * ratio_points ** 2
                score += float(weights["country_share_pp_squared"]) * country_points ** 2
                hard_excess += max(0.0, abs(ratio_points) - ratio_tolerance) ** 2
                hard_excess += max(0.0, abs(country_points) - country_tolerance) ** 2
            for feature_name, total in totals.items():
                if total <= 0:
                    continue
                if feature_name.startswith("class:"):
                    weight = float(weights["class_ratio_pp_squared"])
                elif feature_name in ("positive", "negative"):
                    weight = float(weights["positive_negative_pp_squared"])
                else:
                    continue
                for split_name in SPLIT_NAMES:
                    points = (
                        100.0 * feature_counts[split_name][feature_name] / total
                        - 100.0 * ratios[split_name]
                    )
                    score += weight * points ** 2
            score += float(weights["timestamp_transition"]) * transitions
            score += float(weights["projected_timestamp_guard_image"]) * guarded_image_count
            score += float(weights["tolerance_violation_pp_squared"]) * hard_excess
            return {
                "score": score,
                "split_ratio_deviation_percentage_points": ratio_deviation,
                "india_share_deviation_percentage_points": country_deviation,
                "timestamp_split_transitions": transitions,
                "projected_timestamp_guard_images": guarded_image_count,
            }

        def boundary_move_effect(
            component_id: str, new_split: str
        ) -> tuple[int, int, Counter[str]]:
            transition_delta = 0
            guard_deltas: Counter[str] = Counter()
            for edge_index in edges_by_component.get(component_id, []):
                edge = boundary_edges[edge_index]
                lower = edge["lower_component"]
                upper = edge["upper_component"]
                before_crosses = component_assignment[lower] != component_assignment[upper]
                lower_split = new_split if lower == component_id else component_assignment[lower]
                upper_split = new_split if upper == component_id else component_assignment[upper]
                after_crosses = lower_split != upper_split
                if before_crosses == after_crosses:
                    continue
                delta = 1 if after_crosses else -1
                transition_delta += delta
                for image_id in edge["guard_image_ids"]:
                    guard_deltas[image_id] += delta
            unique_delta = 0
            for image_id, delta in guard_deltas.items():
                before_count = active_guard_memberships[image_id]
                after_count = before_count + delta
                if before_count <= 0 < after_count:
                    unique_delta += 1
                elif before_count > 0 >= after_count:
                    unique_delta -= 1
            return transition_delta, unique_delta, guard_deltas

        before = objective(current, transition_count, projected_guard_images)
        current_objective = before
        moves = 0
        passes = 0
        ordered_components = sorted(components, key=lambda item: item["component_id"])
        for _ in range(int(optimization_config["maximum_local_search_passes"])):
            passes += 1
            changed = False
            for component in ordered_components:
                component_id = component["component_id"]
                original_split = component_assignment[component_id]
                best_split = original_split
                best_objective = current_objective
                best_boundary_effect = (0, 0, Counter())
                for split_name in SPLIT_NAMES:
                    if split_name == original_split:
                        continue
                    features = component["features"]
                    current[original_split].subtract(features)
                    current[split_name].update(features)
                    transition_delta, unique_delta, guard_deltas = boundary_move_effect(
                        component_id, split_name
                    )
                    candidate = objective(
                        current,
                        transition_count + transition_delta,
                        projected_guard_images + unique_delta,
                    )
                    current[split_name].subtract(features)
                    current[original_split].update(features)
                    if candidate["score"] + 1e-9 < best_objective["score"]:
                        best_split = split_name
                        best_objective = candidate
                        best_boundary_effect = (
                            transition_delta, unique_delta, guard_deltas
                        )
                if best_split != original_split:
                    features = component["features"]
                    current[original_split].subtract(features)
                    current[best_split].update(features)
                    component_assignment[component_id] = best_split
                    transition_delta, unique_delta, guard_deltas = best_boundary_effect
                    transition_count += transition_delta
                    projected_guard_images += unique_delta
                    active_guard_memberships.update(guard_deltas)
                    current_objective = best_objective
                    moves += 1
                    changed = True
            if not changed:
                break
        optimization_audit = {
            "enabled": True,
            "moves_applied": moves,
            "passes_completed": passes,
            "timestamp_boundary_edge_count": len(boundary_edges),
            "before": before,
            "after": current_objective,
        }
    image_assignment: dict[str, str] = {}
    group_assignment: dict[str, str] = {}
    for component in components:
        split_name = component_assignment[component["component_id"]]
        for image_id in component["member_image_ids"]:
            image_assignment[image_id] = split_name
        for group_id in component["capture_groups"]:
            previous = group_assignment.setdefault(group_id, split_name)
            if previous != split_name:
                raise RDD2022Error("A capture group received conflicting split assignments.")
    return {
        "image_assignments": image_assignment,
        "capture_group_assignments": group_assignment,
        "components": [
            {
                **component,
                "features": dict(sorted(component["features"].items())),
                "assigned_split": component_assignment[component["component_id"]],
            }
            for component in sorted(components, key=lambda item: item["component_id"])
        ],
        "pre_guard_feature_counts": {
            name: dict(sorted(current[name].items())) for name in SPLIT_NAMES
        },
        "optimization_audit": optimization_audit,
    }


def apply_boundary_guards(
    records: Sequence[dict[str, Any]],
    capture_groups: Sequence[dict[str, Any]],
    image_assignments: dict[str, str],
    grouping_config: dict[str, Any],
) -> dict[str, list[str]]:
    """Return deterministic exclusions around neighboring groups assigned differently."""
    by_id = {item["canonical_image_id"]: item for item in records}
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in capture_groups:
        if group["capture_group"] in {
            by_id[image_id]["capture_group"] for image_id in image_assignments
            if image_id in by_id
        }:
            by_sequence[group["sequence_key"]].append(group)
    exclusions: dict[str, list[str]] = defaultdict(list)
    numeric_guard = int(grouping_config["filename_guard_numeric_ids"])
    timestamp_guard_ms = int(grouping_config["timestamp_guard_seconds"]) * 1000
    group_size = int(grouping_config["filename_numeric_group_size"])
    for groups in by_sequence.values():
        groups.sort(key=lambda item: (item["sequence_index"], item["capture_group"]))
        for lower, upper in zip(groups, groups[1:]):
            if upper["sequence_index"] != lower["sequence_index"] + 1:
                continue
            lower_split = next(
                (image_assignments[image_id] for image_id in lower["image_ids"] if image_id in image_assignments),
                None,
            )
            upper_split = next(
                (image_assignments[image_id] for image_id in upper["image_ids"] if image_id in image_assignments),
                None,
            )
            if lower_split is None or upper_split is None or lower_split == upper_split:
                continue
            reason = (
                f"boundary_guard_between:{lower['capture_group']}:{lower_split}:"
                f"{upper['capture_group']}:{upper_split}"
            )
            if lower["grouping_method"] != "timestamp":
                boundary = (int(lower["filename_bucket"]) + 1) * group_size
                for image_id in lower["image_ids"]:
                    numeric_id = by_id[image_id].get("filename_numeric_id")
                    if image_id in image_assignments and numeric_id is not None and numeric_id >= boundary - numeric_guard:
                        exclusions[image_id].append(reason)
                for image_id in upper["image_ids"]:
                    numeric_id = by_id[image_id].get("filename_numeric_id")
                    if image_id in image_assignments and numeric_id is not None and numeric_id < boundary + numeric_guard:
                        exclusions[image_id].append(reason)
            else:
                lower_times = [
                    int(by_id[image_id]["exif_timestamp_sort_ms"])
                    for image_id in lower["image_ids"] if image_id in image_assignments
                ]
                upper_times = [
                    int(by_id[image_id]["exif_timestamp_sort_ms"])
                    for image_id in upper["image_ids"] if image_id in image_assignments
                ]
                if not lower_times or not upper_times:
                    continue
                lower_edge = max(lower_times)
                upper_edge = min(upper_times)
                for image_id in lower["image_ids"]:
                    timestamp = by_id[image_id].get("exif_timestamp_sort_ms")
                    if image_id in image_assignments and timestamp is not None and timestamp >= lower_edge - timestamp_guard_ms:
                        exclusions[image_id].append(reason)
                for image_id in upper["image_ids"]:
                    timestamp = by_id[image_id].get("exif_timestamp_sort_ms")
                    if image_id in image_assignments and timestamp is not None and timestamp <= upper_edge + timestamp_guard_ms:
                        exclusions[image_id].append(reason)
    return {image_id: sorted(set(reasons)) for image_id, reasons in sorted(exclusions.items())}


def split_conflict_audit(
    records: Sequence[dict[str, Any]],
    confirmed_candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    assigned = {
        item["canonical_image_id"]: item["derived_split"]
        for item in records if item.get("derived_split") in SPLIT_NAMES
    }

    def group_conflicts(field: str) -> list[dict[str, Any]]:
        groups: dict[str, set[str]] = defaultdict(set)
        for item in records:
            group = item.get(field)
            split_name = assigned.get(item["canonical_image_id"])
            if group and split_name:
                groups[group].add(split_name)
        return [
            {"group": group, "splits": sorted(splits)}
            for group, splits in sorted(groups.items()) if len(splits) > 1
        ]

    near_conflicts = []
    for relation in confirmed_candidates:
        if not relation["hard_split_constraint"]:
            continue
        first_split = assigned.get(relation["first_image_id"])
        second_split = assigned.get(relation["second_image_id"])
        if first_split and second_split and first_split != second_split:
            near_conflicts.append(relation)
    country_coverage = {
        split_name: sorted({
            item["country"] for item in records if item.get("derived_split") == split_name
        }) for split_name in SPLIT_NAMES
    }
    class_coverage = {
        split_name: sorted({
            class_name
            for item in records if item.get("derived_split") == split_name
            for class_name, count in item["target_class_counts"].items() if count > 0
        }) for split_name in SPLIT_NAMES
    }
    return {
        "capture_group_conflicts": group_conflicts("capture_group"),
        "exact_duplicate_group_conflicts": group_conflicts("exact_duplicate_group"),
        "confirmed_near_duplicate_conflicts": near_conflicts,
        "country_coverage_by_split": country_coverage,
        "class_coverage_by_split": class_coverage,
        "all_required_countries_in_every_split": all(
            set(countries) == {"India", "Japan"} for countries in country_coverage.values()
        ),
        "all_target_classes_in_every_split": all(
            set(classes) == set(TARGET_CLASSES) for classes in class_coverage.values()
        ),
    }
