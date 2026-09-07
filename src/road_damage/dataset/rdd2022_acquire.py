"""Acquire and safely extract the approved India/Japan RDD2022 raw archives."""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import secrets
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_common import (
    RDD2022Error,
    load_config,
    md5_file,
    project_relative,
    resolve_project_path,
    sha256_file,
    write_json_atomic,
)


LOGGER = logging.getLogger("road_damage.dataset.rdd2022_acquire")
MANIFEST_FILENAME = "acquisition_manifest.json"
BUFFER_SIZE = 8 * 1024 * 1024
HTTP_RANGE_REQUEST_SIZE = 8 * 1024 * 1024
KNOWN_COUNTRY_NAMES = {
    "china_drone", "china_motorbike", "czech", "india", "japan", "norway",
    "united_states",
}
COUNTRY_DISPLAY_NAMES = {
    "china_drone": "China_Drone",
    "china_motorbike": "China_MotorBike",
    "czech": "Czech",
    "india": "India",
    "japan": "Japan",
    "norway": "Norway",
    "united_states": "United_States",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _download(
    source_url: str,
    destination: Path,
    opener: Any = urllib.request.urlopen,
    expected_size: int | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """Resume into a partial file and atomically commit a complete download."""
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RDD2022Error(f"Existing archive path is not a usable file: {destination}")
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise RDD2022Error(
                f"Existing archive size is {destination.stat().st_size}, expected {expected_size}."
            )
        return {
            "downloaded_this_run": False,
            "download_resumed": False,
            "http_content_length": None,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    initial_size = temporary.stat().st_size if temporary.exists() else 0
    if expected_size is not None and initial_size > expected_size:
        raise RDD2022Error(
            f"Partial download is larger than the official file: {temporary}"
        )
    last_error: Exception | None = None
    content_length: str | None = None
    for attempt in range(1, retries + 1):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "AI-Road-Damage-Project/Phase2A"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(source_url, headers=headers)
        try:
            with opener(request, timeout=60) as response:
                status = getattr(response, "status", None)
                content_length = response.headers.get("Content-Length")
                append = offset > 0 and status == 206
                mode = "ab" if append else "wb"
                starting_bytes = offset if append else 0
                with temporary.open(mode) as output:
                    downloaded = _copy_stream(response, output, starting_bytes)
            if expected_size is not None and downloaded != expected_size:
                raise RDD2022Error(
                    f"Download has {downloaded} bytes; official size is {expected_size}."
                )
            if expected_size is None and content_length is not None:
                expected_response_size = int(content_length) + starting_bytes
                if downloaded != expected_response_size:
                    raise RDD2022Error(
                        f"Download length mismatch: {downloaded} vs {expected_response_size}."
                    )
            os.replace(temporary, destination)
            break
        except (
            OSError,
            http.client.HTTPException,
            urllib.error.URLError,
            RDD2022Error,
        ) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429) and exc.code < 500:
                break
            if attempt < retries:
                LOGGER.warning("Download attempt %d/%d failed; retrying: %s", attempt, retries, exc)
                time.sleep(2 ** (attempt - 1))
    if not destination.exists():
        raise RDD2022Error(
            f"Could not download official archive {source_url}: {last_error}"
        ) from last_error
    return {
        "downloaded_this_run": True,
        "download_resumed": initial_size > 0,
        "resumed_from_bytes": initial_size,
        "http_content_length": int(content_length) if content_length else None,
    }


def _copy_stream(
    source: BinaryIO, destination: BinaryIO, starting_bytes: int = 0
) -> int:
    downloaded = starting_bytes
    next_log = ((downloaded // (256 * 1024 * 1024)) + 1) * 256 * 1024 * 1024
    while True:
        chunk = source.read(BUFFER_SIZE)
        if not chunk:
            break
        destination.write(chunk)
        downloaded += len(chunk)
        if downloaded >= next_log:
            LOGGER.info("Downloaded %.1f MiB", downloaded / (1024 * 1024))
            next_log += 256 * 1024 * 1024
    return downloaded


def _segment_path(part_path: Path, index: int) -> Path:
    return part_path.with_name(f"{part_path.name}.{index:03d}")


def _download_segment(
    source_url: str,
    destination: Path,
    start: int,
    end: int,
    opener: Any,
    retries: int,
) -> dict[str, Any]:
    expected_length = end - start + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected_length:
        raise RDD2022Error(f"Download segment is too large: {destination}")
    initial_size = existing
    if existing == expected_length:
        return {
            "path": project_relative(destination),
            "range_start": start,
            "range_end": end,
            "size_bytes": existing,
            "reused_complete_segment": True,
        }
    next_log = ((existing // (256 * 1024 * 1024)) + 1) * 256 * 1024 * 1024
    while existing < expected_length:
        existing = destination.stat().st_size if destination.exists() else 0
        request_start = start + existing
        request_end = min(request_start + HTTP_RANGE_REQUEST_SIZE - 1, end)
        expected_chunk_length = request_end - request_start + 1
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            request = urllib.request.Request(
                source_url,
                headers={
                    "Range": f"bytes={request_start}-{request_end}",
                    "User-Agent": "AI-Road-Damage-Project/Phase2A",
                },
            )
            try:
                with opener(request, timeout=60) as response:
                    status = getattr(response, "status", None)
                    content_range = str(response.headers.get("Content-Range", ""))
                    if status != 206 or not content_range.startswith(
                        f"bytes {request_start}-{request_end}/"
                    ):
                        raise RDD2022Error(
                            "Official server did not honor bounded range "
                            f"{request_start}-{request_end}; status={status}, "
                            f"Content-Range={content_range!r}."
                        )
                    with destination.open("ab") as output:
                        while True:
                            chunk = response.read(BUFFER_SIZE)
                            if not chunk:
                                break
                            output.write(chunk)
                size = destination.stat().st_size
                if size != existing + expected_chunk_length:
                    raise RDD2022Error(
                        f"Range {request_start}-{request_end} produced "
                        f"{size - existing} bytes; expected {expected_chunk_length}."
                    )
                existing = size
                if existing >= next_log:
                    LOGGER.info(
                        "Segment %s downloaded %.1f MiB / %.1f MiB",
                        destination.name,
                        existing / (1024 * 1024),
                        expected_length / (1024 * 1024),
                    )
                    next_log += 256 * 1024 * 1024
                break
            except (
                OSError,
                http.client.HTTPException,
                urllib.error.URLError,
                RDD2022Error,
            ) as exc:
                last_error = exc
                if destination.exists():
                    try:
                        with destination.open("r+b") as output:
                            output.truncate(existing)
                    except OSError as truncate_error:
                        raise RDD2022Error(
                            f"Could not restore incomplete segment {destination}: "
                            f"{truncate_error}"
                        ) from truncate_error
                if attempt < retries:
                    LOGGER.warning(
                        "Range %d-%d attempt %d/%d failed; retrying: %s",
                        request_start,
                        request_end,
                        attempt,
                        retries,
                        exc,
                    )
                    time.sleep(2 ** (attempt - 1))
        else:
            raise RDD2022Error(
                f"Could not download official archive range "
                f"{request_start}-{request_end}: {last_error}"
            ) from last_error
    LOGGER.info("Completed download segment %s (%d bytes)", destination.name, existing)
    return {
        "path": project_relative(destination),
        "range_start": start,
        "range_end": end,
        "size_bytes": existing,
        "resumed_from_bytes": initial_size,
        "reused_complete_segment": False,
    }


def _download_segmented(
    source_url: str,
    destination: Path,
    expected_size: int,
    connections: int,
    opener: Any = urllib.request.urlopen,
    retries: int = 5,
) -> dict[str, Any]:
    """Download fixed byte ranges concurrently, resume them, and commit atomically."""
    if expected_size <= 0:
        raise RDD2022Error("The official archive size must be positive.")
    if not 1 <= connections <= 16:
        raise RDD2022Error("Parallel download connections must be between 1 and 16.")
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != expected_size:
            raise RDD2022Error(
                f"Existing archive is not the official size ({expected_size}): {destination}"
            )
        return {
            "downloaded_this_run": False,
            "download_resumed": False,
            "parallel_connections": connections,
            "segments": [],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f".{destination.name}.part")
    segment_size = (expected_size + connections - 1) // connections
    ranges = [
        (index, index * segment_size, min((index + 1) * segment_size, expected_size) - 1)
        for index in range(connections)
        if index * segment_size < expected_size
    ]
    first_segment = _segment_path(part_path, 0)
    adopted_sequential_bytes = 0
    if part_path.exists():
        if first_segment.exists():
            raise RDD2022Error(
                f"Both legacy and segmented partial files exist: {part_path}"
            )
        adopted_sequential_bytes = part_path.stat().st_size
        first_expected = ranges[0][2] - ranges[0][1] + 1
        if adopted_sequential_bytes > first_expected:
            raise RDD2022Error(
                "Existing sequential partial file is larger than the first parallel segment; "
                "resume once with the previous single-stream utility or remove only that partial."
            )
        try:
            os.replace(part_path, first_segment)
        except OSError as exc:
            raise RDD2022Error(
                f"Could not adopt the existing sequential partial file {part_path}: {exc}"
            ) from exc
    segment_paths = [_segment_path(part_path, index) for index, _, _ in ranges]
    bytes_before = sum(path.stat().st_size for path in segment_paths if path.exists())
    with ThreadPoolExecutor(max_workers=connections) as executor:
        futures = [
            executor.submit(
                _download_segment,
                source_url,
                _segment_path(part_path, index),
                start,
                end,
                opener,
                retries,
            )
            for index, start, end in ranges
        ]
        segment_results = [future.result() for future in futures]
    assembling = destination.with_name(f".{destination.name}.assembling")
    try:
        assembling.unlink(missing_ok=True)
        with assembling.open("xb") as output:
            for segment in segment_paths:
                with segment.open("rb") as source:
                    shutil.copyfileobj(source, output, BUFFER_SIZE)
        assembled_size = assembling.stat().st_size
        if assembled_size != expected_size:
            raise RDD2022Error(
                f"Assembled archive has {assembled_size} bytes; expected {expected_size}."
            )
        os.replace(assembling, destination)
    except Exception as exc:
        assembling.unlink(missing_ok=True)
        if isinstance(exc, RDD2022Error):
            raise
        raise RDD2022Error(f"Could not assemble archive segments: {exc}") from exc
    for segment in segment_paths:
        segment.unlink(missing_ok=True)
    return {
        "downloaded_this_run": True,
        "download_resumed": bytes_before > 0,
        "resumed_from_bytes": bytes_before,
        "adopted_legacy_sequential_partial_bytes": adopted_sequential_bytes,
        "parallel_connections": connections,
        "segments": segment_results,
    }


def _member_destination(root: Path, member_name: str) -> Path:
    portable = PurePosixPath(member_name.replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts or any(
        ":" in part for part in portable.parts
    ):
        raise RDD2022Error(f"Unsafe ZIP member path: {member_name!r}.")
    destination = root.joinpath(*portable.parts).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise RDD2022Error(f"ZIP member escapes extraction root: {member_name!r}.")
    return destination


def _make_staging_directory(parent: Path, prefix: str) -> Path:
    """Create an atomic staging directory that inherits the parent ACL."""
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        candidate = parent / f".{prefix}-{secrets.token_hex(8)}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise RDD2022Error(f"Could not create staging directory: {candidate}") from exc
    raise RDD2022Error(f"Could not allocate a unique staging directory under {parent}.")


def _member_country(member_name: str) -> tuple[str | None, int | None]:
    parts = PurePosixPath(member_name.replace("\\", "/")).parts
    for index, part in enumerate(parts):
        normalized = part.casefold()
        if normalized.endswith(".zip"):
            normalized = PurePosixPath(normalized).stem
        if normalized in KNOWN_COUNTRY_NAMES:
            return COUNTRY_DISPLAY_NAMES[normalized], index
    return None, None


def _inspect_country_zip_handle(
    archive: zipfile.ZipFile, country: str, archive_label: str
) -> dict[str, Any]:
    members = archive.infolist()
    if not members:
        raise RDD2022Error(f"Archive is empty: {archive_label}")
    country_seen = False
    zero_byte_members: list[str] = []
    encrypted_members: list[str] = []
    other_country_members: list[str] = []
    for member in members:
        _member_destination(Path.cwd() / ".zip-path-probe", member.filename)
        mode = (member.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise RDD2022Error(
                f"Symbolic-link ZIP member is not allowed: {member.filename}"
            )
        member_country, _ = _member_country(member.filename)
        country_seen = country_seen or member_country == country
        if member_country is not None and member_country != country:
            other_country_members.append(member.filename)
        if not member.is_dir() and member.file_size == 0:
            zero_byte_members.append(member.filename)
        if member.flag_bits & 0x1:
            encrypted_members.append(member.filename)
    if other_country_members:
        raise RDD2022Error(
            f"{archive_label} contains another-country path; refusing extraction."
        )
    if not country_seen:
        raise RDD2022Error(
            f"{archive_label} does not contain a recognizable {country} path."
        )
    if encrypted_members:
        raise RDD2022Error(
            f"{archive_label} contains encrypted members and cannot be audited."
        )
    corrupt_member = archive.testzip()
    if corrupt_member is not None:
        raise RDD2022Error(
            f"ZIP CRC/decompression check failed at member: {corrupt_member}"
        )
    return {
        "archive_opens_successfully": True,
        "zip_crc_test_passed": True,
        "member_count": len(members),
        "file_member_count": sum(not member.is_dir() for member in members),
        "directory_member_count": sum(member.is_dir() for member in members),
        "zero_byte_file_members": zero_byte_members,
        "encrypted_members": encrypted_members,
        "other_country_members": other_country_members,
    }


def query_figshare_metadata(
    config: dict[str, Any], opener: Any = urllib.request.urlopen
) -> dict[str, Any]:
    """Fetch and verify the pinned official Figshare article/file metadata."""
    fallback = config["acquisition_fallback"]
    api_url = str(config["dataset"]["figshare_api_url"])
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/json", "User-Agent": "AI-Road-Damage-Project/Phase2A"},
    )
    try:
        with opener(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.URLError) as exc:
        raise RDD2022Error(
            f"Could not query official Figshare metadata at {api_url}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RDD2022Error("Official Figshare API response was not a JSON object.")
    files = payload.get("files")
    if not isinstance(files, list):
        raise RDD2022Error("Official Figshare API response has no file list.")
    file_id = int(fallback["file_id"])
    matches = [item for item in files if isinstance(item, dict) and item.get("id") == file_id]
    if len(matches) != 1:
        raise RDD2022Error(
            f"Official Figshare response contains {len(matches)} matches for file ID {file_id}."
        )
    file_metadata = matches[0]
    license_metadata = payload.get("license")
    if not isinstance(license_metadata, dict):
        raise RDD2022Error("Official Figshare API response has no license metadata.")
    observed = {
        "article_id": int(payload.get("id", 0)),
        "article_version": int(payload.get("version", 0)),
        "doi": str(payload.get("doi", "")),
        "license": {
            "name": str(license_metadata.get("name", "")),
            "url": str(license_metadata.get("url", "")),
        },
        "file_id": int(file_metadata.get("id", 0)),
        "filename": str(file_metadata.get("name", "")),
        "download_url": str(file_metadata.get("download_url", "")),
        "size_bytes": int(file_metadata.get("size", 0)),
        "supplied_md5": str(file_metadata.get("supplied_md5", "")).casefold(),
        "computed_md5": str(file_metadata.get("computed_md5", "")).casefold() or None,
        "metadata_api_url": api_url,
        "metadata_verified_utc": _utc_now().isoformat(),
    }
    expected = {
        "article_id": int(fallback["article_id"]),
        "article_version": int(fallback["article_version"]),
        "doi": str(config["dataset"]["figshare_doi"]),
        "license_name": "CC BY 4.0",
        "file_id": file_id,
        "filename": str(fallback["archive_filename"]),
        "download_url": str(fallback["download_url"]),
        "size_bytes": int(fallback["size_bytes"]),
        "supplied_md5": str(fallback["supplied_md5"]).casefold(),
    }
    comparisons = {
        "article_id": observed["article_id"] == expected["article_id"],
        "article_version": observed["article_version"] == expected["article_version"],
        "doi": observed["doi"] == expected["doi"],
        "license_name": observed["license"]["name"] == expected["license_name"],
        "file_id": observed["file_id"] == expected["file_id"],
        "filename": observed["filename"] == expected["filename"],
        "download_url": observed["download_url"] == expected["download_url"],
        "size_bytes": observed["size_bytes"] == expected["size_bytes"],
        "supplied_md5": observed["supplied_md5"] == expected["supplied_md5"],
    }
    failed = [name for name, matches_expected in comparisons.items() if not matches_expected]
    if failed:
        raise RDD2022Error(
            "Official Figshare metadata differs from the pinned Phase 2A config: "
            + ", ".join(failed)
            + "."
        )
    observed["matches_pinned_config"] = True
    return observed


def inspect_full_archive(
    archive_path: Path, required_countries: Sequence[str]
) -> dict[str, Any]:
    """Validate every member and CRC while inventorying country path segments."""
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            if not members:
                raise RDD2022Error(f"Archive is empty: {archive_path}")
            zero_byte_members: list[str] = []
            encrypted_members: list[str] = []
            country_member_counts: dict[str, int] = {
                display: 0 for display in COUNTRY_DISPLAY_NAMES.values()
            }
            for member in members:
                _member_destination(archive_path.parent / ".zip-path-probe", member.filename)
                mode = (member.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise RDD2022Error(
                        f"Symbolic-link ZIP member is not allowed: {member.filename}"
                    )
                country, _ = _member_country(member.filename)
                if country is not None and not member.is_dir():
                    country_member_counts[country] += 1
                if not member.is_dir() and member.file_size == 0:
                    zero_byte_members.append(member.filename)
                if member.flag_bits & 0x1:
                    encrypted_members.append(member.filename)
            if encrypted_members:
                raise RDD2022Error(
                    f"{archive_path.name} contains encrypted members and cannot be audited."
                )
            absent = [
                country for country in required_countries
                if country_member_counts.get(country, 0) == 0
            ]
            if absent:
                raise RDD2022Error(
                    "Full archive does not contain required country paths: "
                    + ", ".join(absent)
                    + "."
                )
            LOGGER.info("Running full ZIP CRC/decompression validation")
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise RDD2022Error(
                    f"ZIP CRC/decompression check failed at member: {corrupt_member}"
                )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise RDD2022Error(f"Archive cannot be opened as a valid ZIP: {archive_path}") from exc
    countries_present = [
        country for country, count in country_member_counts.items() if count > 0
    ]
    return {
        "archive_opens_successfully": True,
        "zip_crc_test_passed": True,
        "member_count": len(members),
        "file_member_count": sum(not member.is_dir() for member in members),
        "directory_member_count": sum(member.is_dir() for member in members),
        "zero_byte_file_members": zero_byte_members,
        "encrypted_members": encrypted_members,
        "countries_present_by_exact_path_segment": countries_present,
        "country_file_member_counts": {
            country: count for country, count in country_member_counts.items() if count > 0
        },
    }


def extract_country_from_full_archive(
    archive_path: Path, country: str, destination: Path
) -> dict[str, Any]:
    """Extract one direct or nested country ZIP subtree, omitting transport wrappers."""
    if destination.exists():
        raise RDD2022Error(f"Raw country output already exists: {destination}")
    temporary = _make_staging_directory(destination.parent, country.lower())
    extracted_files = 0
    extracted_bytes = 0
    country_archive_integrity: dict[str, Any] | None = None
    nested_archive_member: str | None = None

    def extract_members(country_archive: zipfile.ZipFile) -> None:
        nonlocal extracted_files, extracted_bytes
        for member in country_archive.infolist():
            member_country, country_index = _member_country(member.filename)
            if member_country != country or country_index is None:
                continue
            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            relative_parts = parts[country_index + 1:]
            if not relative_parts:
                continue
            relative_name = PurePosixPath(*relative_parts).as_posix()
            target = _member_destination(temporary, relative_name)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with country_archive.open(member, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, BUFFER_SIZE)
            extracted_files += 1
            extracted_bytes += member.file_size

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            nested_members = [
                member for member in archive.infolist()
                if not member.is_dir()
                and PurePosixPath(member.filename.replace("\\", "/")).suffix.casefold()
                == ".zip"
                and _member_country(member.filename)[0] == country
            ]
            if len(nested_members) > 1:
                raise RDD2022Error(
                    f"Found multiple nested {country} archives in {archive_path.name}."
                )
            if nested_members:
                nested = nested_members[0]
                nested_archive_member = nested.filename
                with archive.open(nested, "r") as nested_source:
                    with zipfile.ZipFile(nested_source, "r") as country_archive:
                        LOGGER.info("Validating nested %s ZIP CRC", country)
                        country_archive_integrity = _inspect_country_zip_handle(
                            country_archive, country, nested.filename
                        )
                        extract_members(country_archive)
            else:
                corrupt_member = archive.testzip()
                if corrupt_member is not None:
                    raise RDD2022Error(
                        f"ZIP CRC/decompression check failed at member: {corrupt_member}"
                    )
                country_archive_integrity = {
                    "archive_opens_successfully": True,
                    "zip_crc_test_passed": True,
                    "country_files_selected_from_shared_archive": True,
                }
                extract_members(archive)
        if extracted_files == 0:
            raise RDD2022Error(f"No {country} files were selected from the full archive.")
        zero_byte_files = sorted(
            project_relative(path) for path in temporary.rglob("*")
            if path.is_file() and path.stat().st_size == 0
        )
        os.replace(temporary, destination)
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(exc, RDD2022Error):
            raise
        raise RDD2022Error(
            f"Could not selectively extract {country} from {archive_path}: {exc}"
        ) from exc
    return {
        "raw_root": project_relative(destination),
        "extracted_file_count": extracted_files,
        "extracted_logical_bytes": extracted_bytes,
        "zero_byte_files": zero_byte_files,
        "archive_wrapper_removed": True,
        "nested_archive_member": nested_archive_member,
        "country_archive_integrity": country_archive_integrity,
        "original_filenames_and_country_subdirectories_preserved": True,
        "immutable_raw_layer_policy": True,
        "files_marked_read_only": False,
        "reused_existing_extraction": False,
    }


def inspect_zip(archive_path: Path, country: str) -> dict[str, Any]:
    """Verify ZIP readability, scope, member paths, encryption, and zero-byte files."""
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            return _inspect_country_zip_handle(archive, country, archive_path.name)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise RDD2022Error(f"Archive cannot be opened as a valid ZIP: {archive_path}") from exc


def extract_zip_safely(archive_path: Path, country: str, destination: Path) -> dict[str, Any]:
    """Extract into a new country root after validating every member path."""
    if destination.exists():
        raise RDD2022Error(f"Raw country output already exists: {destination}")
    temporary = _make_staging_directory(destination.parent, country.lower())
    extracted_files = 0
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                target = _member_destination(temporary, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, BUFFER_SIZE)
                extracted_files += 1
        zero_byte_files = [
            project_relative(path) for path in temporary.rglob("*")
            if path.is_file() and path.stat().st_size == 0
        ]
        if zero_byte_files:
            raise RDD2022Error(
                f"Extracted {country} data contains {len(zero_byte_files)} zero-byte files."
            )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "raw_root": project_relative(destination),
        "extracted_file_count": extracted_files,
        "zero_byte_files": [],
        "immutable_raw_layer_policy": True,
        "files_marked_read_only": False,
    }


def _existing_extraction(destination: Path) -> dict[str, Any] | None:
    if not destination.exists():
        return None
    if not destination.is_dir():
        raise RDD2022Error(f"Raw country output is not a directory: {destination}")
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    if not files:
        raise RDD2022Error(f"Existing raw country output is empty: {destination}")
    zero_byte_files = [project_relative(path) for path in files if path.stat().st_size == 0]
    return {
        "raw_root": project_relative(destination),
        "extracted_file_count": len(files),
        "extracted_logical_bytes": sum(path.stat().st_size for path in files),
        "zero_byte_files": zero_byte_files,
        "immutable_raw_layer_policy": True,
        "files_marked_read_only": False,
        "reused_existing_extraction": True,
    }


def run_acquisition(config_path: Path, dataset_root: Path) -> tuple[Path, bool]:
    """Acquire the official transport archive and extract only India and Japan."""
    config_path, config = load_config(config_path)
    root = resolve_project_path(dataset_root)
    archives_dir = root / "archives"
    raw_dir = root / "raw"
    manifests_dir = root / "manifests"
    now = _utc_now()
    run_id = f"rdd2022_phase2a_{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    fallback = config["acquisition_fallback"]
    archive_path = archives_dir / str(fallback["archive_filename"])
    part_path = archive_path.with_name(f".{archive_path.name}.part")
    stale_sources = [
        {
            "country": item["country"],
            "url": item["source_url"],
            "observed_result": "HTTP 403 AccessDenied",
            "used_for_download": False,
        }
        for item in config["countries"]
    ]
    acquisition: dict[str, Any] = {
        "status": "pending",
        "archive_path": project_relative(archive_path),
    }
    disk_before = shutil.disk_usage(root if root.exists() else root.parent)
    completed = False
    error: str | None = None
    try:
        figshare = query_figshare_metadata(config)
        expected_size = int(figshare["size_bytes"])
        already_present = archive_path.stat().st_size if archive_path.is_file() else 0
        partial_paths = [part_path, *sorted(part_path.parent.glob(f"{part_path.name}.*"))]
        partial_present = sum(
            path.stat().st_size for path in partial_paths
            if path.is_file() and not path.name.endswith(".assembling")
        )
        if already_present:
            download_bytes_remaining = 0
        else:
            download_bytes_remaining = max(expected_size - partial_present, 0)
        reserve = int(fallback["minimum_extraction_and_temporary_reserve_bytes"])
        assembly_bytes = 0 if already_present else expected_size
        required_free = download_bytes_remaining + assembly_bytes + reserve
        acquisition["storage_preflight"] = {
            "drive_total_bytes": disk_before.total,
            "drive_used_bytes": disk_before.used,
            "drive_free_bytes": disk_before.free,
            "archive_bytes_already_present": already_present,
            "partial_download_bytes_already_present": partial_present,
            "download_bytes_remaining": download_bytes_remaining,
            "assembly_temporary_bytes_required": assembly_bytes,
            "minimum_extraction_and_temporary_reserve_bytes": reserve,
            "required_free_bytes": required_free,
            "passed": disk_before.free >= required_free,
        }
        if disk_before.free < required_free:
            raise RDD2022Error(
                f"Insufficient free space: {disk_before.free} bytes available, "
                f"at least {required_free} bytes required."
            )
        acquisition["official_figshare_metadata"] = figshare
        acquisition["download"] = _download_segmented(
            str(figshare["download_url"]),
            archive_path,
            expected_size,
            int(fallback["parallel_download_connections"]),
        )
        actual_size = archive_path.stat().st_size
        actual_md5 = md5_file(archive_path)
        if actual_size != expected_size:
            raise RDD2022Error(
                f"Archive size is {actual_size}; official size is {expected_size}."
            )
        if actual_md5.casefold() != str(figshare["supplied_md5"]).casefold():
            raise RDD2022Error(
                "Archive MD5 does not match the checksum supplied by Figshare."
            )
        acquisition["archive"] = {
            "filename": archive_path.name,
            "path": project_relative(archive_path),
            "size_bytes": actual_size,
            "md5": actual_md5,
            "md5_matches_figshare": True,
            "sha256": sha256_file(archive_path),
            "retained_after_extraction": True,
            "content_modified": False,
        }
        acquisition["archive_integrity"] = inspect_full_archive(
            archive_path, config["allowed_countries"]
        )
        extractions: list[dict[str, Any]] = []
        for country in config["allowed_countries"]:
            raw_country = raw_dir / str(country)
            extraction = _existing_extraction(raw_country)
            if extraction is None:
                LOGGER.info("Selectively extracting %s", country)
                extraction = extract_country_from_full_archive(
                    archive_path, str(country), raw_country
                )
            extraction["country"] = country
            extractions.append(extraction)
        excluded_paths_found = [
            project_relative(raw_dir / country)
            for country in fallback["countries_intentionally_not_extracted"]
            if (raw_dir / country).exists()
        ]
        if excluded_paths_found:
            raise RDD2022Error(
                "Non-scope country directories exist in the raw layer: "
                + ", ".join(excluded_paths_found)
            )
        acquisition["extractions"] = extractions
        acquisition["countries_extracted"] = list(config["allowed_countries"])
        acquisition["countries_intentionally_not_extracted"] = list(
            fallback["countries_intentionally_not_extracted"]
        )
        acquisition["excluded_raw_paths_found"] = excluded_paths_found
        acquisition["status"] = "completed"
        completed = True
    except Exception as exc:
        error = str(exc) if isinstance(exc, RDD2022Error) else (
            f"Unexpected {type(exc).__name__}: {exc}"
        )
        acquisition["status"] = "failed"
        acquisition["error"] = error
        if isinstance(exc, RDD2022Error):
            LOGGER.error("Phase 2A acquisition failed: %s", exc)
        else:
            LOGGER.exception("Unexpected Phase 2A acquisition failure")

    disk_after = shutil.disk_usage(root if root.exists() else root.parent)
    archive_bytes = archive_path.stat().st_size if archive_path.is_file() else 0
    partial_bytes = sum(
        path.stat().st_size
        for path in [part_path, *part_path.parent.glob(f"{part_path.name}.*")]
        if path.is_file()
    ) if part_path.parent.exists() else 0
    raw_files = list(raw_dir.rglob("*")) if raw_dir.exists() else []
    extracted_bytes = sum(path.stat().st_size for path in raw_files if path.is_file())
    manifest = {
        "schema_version": "1.0",
        "phase": "Phase 2A - RDD2022 data acquisition and integrity audit",
        "run_id": run_id,
        "retrieval_attempt_utc": now.isoformat(),
        "status": "completed" if completed else "failed",
        "config_path": project_relative(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset": config["dataset"],
        "allowed_countries": config["allowed_countries"],
        "source_transition": {
            "original_country_specific_sources": stale_sources,
            "transition_reason": fallback["reason"],
            "fallback_source_type": fallback["source_type"],
            "unofficial_mirrors_used": False,
        },
        "acquisition": acquisition,
        "storage_after_run": {
            "archive_logical_bytes": archive_bytes,
            "partial_download_logical_bytes": partial_bytes,
            "extracted_india_japan_logical_bytes": extracted_bytes,
            "drive_total_bytes": disk_after.total,
            "drive_used_bytes": disk_after.used,
            "drive_free_bytes": disk_after.free,
        },
        "scope_confirmations": {
            "rdd2020_downloaded": False,
            "full_transport_archive_contains_non_scope_countries": True,
            "countries_other_than_india_japan_extracted": False,
            "countries_other_than_india_japan_used": False,
            "annotations_converted": False,
            "dataset_split_created": False,
            "model_downloaded": False,
            "training_performed": False,
            "teacher_video_data_mixed": False,
        },
    }
    if error is not None:
        manifest["error"] = error
    run_manifest = manifests_dir / "runs" / run_id / MANIFEST_FILENAME
    latest_manifest = manifests_dir / MANIFEST_FILENAME
    write_json_atomic(run_manifest, manifest)
    write_json_atomic(latest_manifest, manifest)
    return latest_manifest, completed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire the official RDD2022 Figshare archive and selectively extract "
            "the India/Japan raw data."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        manifest_path, completed = run_acquisition(args.config, args.dataset_root)
    except RDD2022Error as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Acquisition manifest: %s", project_relative(manifest_path))
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
