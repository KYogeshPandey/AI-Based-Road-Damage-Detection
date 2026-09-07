"""Shared configuration and provenance helpers for the RDD2022 raw layer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COUNTRIES = ("India", "Japan")
TARGET_CLASSES = ("D00", "D10", "D20", "D40")


class RDD2022Error(RuntimeError):
    """Raised for an actionable RDD2022 acquisition or integrity error."""


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    """Return a portable project-relative path when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    """Load and validate the dependency-free JSON-compatible YAML config."""
    resolved = resolve_project_path(path)
    try:
        config = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RDD2022Error(f"RDD2022 configuration does not exist: {resolved}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RDD2022Error(f"Could not read RDD2022 configuration: {resolved}") from exc
    if not isinstance(config, dict):
        raise RDD2022Error("RDD2022 configuration must contain an object.")
    validate_config(config)
    return resolved, config


def validate_config(config: dict[str, Any]) -> None:
    """Enforce the approved India/Japan-only Phase 2A scope."""
    try:
        allowed = tuple(str(value) for value in config["allowed_countries"])
        countries = config["countries"]
        target_classes = tuple(str(value) for value in config["target_classes"])
        licenses = config["dataset"]["licenses"]
        figshare_doi = str(config["dataset"]["figshare_doi"])
        fallback = config["acquisition_fallback"]
    except (KeyError, TypeError) as exc:
        raise RDD2022Error("RDD2022 configuration is missing required fields.") from exc
    if allowed != COUNTRIES or target_classes != TARGET_CLASSES:
        raise RDD2022Error(
            "Phase 2A must contain only India/Japan and D00/D10/D20/D40 in order."
        )
    if not isinstance(countries, list) or len(countries) != 2:
        raise RDD2022Error("Exactly two country entries are required.")
    if tuple(str(item.get("country")) for item in countries) != COUNTRIES:
        raise RDD2022Error("Country entries must be India followed by Japan.")
    required_country_fields = {
        "archive_filename",
        "source_url",
        "published_approximate_size_mb",
        "expected_image_dimensions",
        "expected_target_object_counts",
    }
    for country in countries:
        missing = required_country_fields.difference(country)
        if missing:
            raise RDD2022Error(
                f"{country['country']} config is missing: {', '.join(sorted(missing))}."
            )
        if set(country["expected_target_object_counts"]) != set(TARGET_CLASSES):
            raise RDD2022Error(
                f"{country['country']} expected class counts must use four target codes."
            )
        if Path(str(country["archive_filename"])).name != country["archive_filename"]:
            raise RDD2022Error("Archive filenames must not contain directories.")
        if not str(country["source_url"]).startswith("https://"):
            raise RDD2022Error("Archive source URLs must use HTTPS.")
    if figshare_doi != "10.6084/m9.figshare.21431547.v1":
        raise RDD2022Error("The fixed RDD2022 Figshare v1 DOI must be preserved.")
    if (
        int(fallback.get("article_id", 0)) != 21431547
        or int(fallback.get("article_version", 0)) != 1
        or int(fallback.get("file_id", 0)) != 38030910
        or fallback.get("source_type")
        != "official_figshare_full_archive_selective_extraction"
        or fallback.get("archive_filename")
        != "RDD2022_released_through_CRDDC2022.zip"
        or fallback.get("download_url")
        != "https://ndownloader.figshare.com/files/38030910"
        or int(fallback.get("size_bytes", 0)) != 13_264_172_619
        or str(fallback.get("supplied_md5", "")).casefold()
        != "b62bd51d2ffcfaa76c60f234f0cc2bb3"
        or tuple(fallback.get("countries_to_extract", [])) != COUNTRIES
        or not 1 <= int(fallback.get("parallel_download_connections", 0)) <= 16
        or int(fallback.get("minimum_extraction_and_temporary_reserve_bytes", 0))
        < 5_000_000_000
    ):
        raise RDD2022Error("Official Figshare file metadata or extraction scope changed.")
    excluded = set(fallback.get("countries_intentionally_not_extracted", []))
    if excluded != {
        "China_Drone", "China_MotorBike", "Czech", "Norway", "United_States"
    }:
        raise RDD2022Error("All non-approved RDD2022 countries must remain excluded.")
    if (
        licenses.get("figshare", {}).get("stated_license") != "CC BY 4.0"
        or licenses.get("authors_repository", {}).get("stated_license")
        != "CC BY-SA 4.0"
        or licenses.get("project_conservative_interpretation") != "CC BY-SA 4.0"
        or licenses.get("discrepancy_status") != "unresolved"
    ):
        raise RDD2022Error("Both license statements and the unresolved conflict are required.")


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RDD2022Error(f"Could not hash file: {path}") from exc
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 without modifying a file."""
    return _hash_file(path, "sha256")


def md5_file(path: Path) -> str:
    """Calculate MD5 only for comparison with the checksum supplied by Figshare."""
    return _hash_file(path, "md5")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON artifact atomically."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise RDD2022Error(f"Could not write JSON artifact: {path}") from exc
