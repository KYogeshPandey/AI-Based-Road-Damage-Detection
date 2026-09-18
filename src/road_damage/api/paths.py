"""Confinement helpers for future upload and artifact paths."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID


class UnsafeApiPathError(ValueError):
    """Raised when an untrusted path could escape API-controlled storage."""


def validate_public_filename(value: str) -> str:
    """Accept one display/upload filename, never a client-supplied server path."""
    if not isinstance(value, str):
        raise UnsafeApiPathError("Filename must be text.")
    filename = value.strip()
    if not filename or filename in {".", ".."} or "\x00" in filename:
        raise UnsafeApiPathError("Filename is empty or reserved.")
    windows = PureWindowsPath(filename)
    posix = PurePosixPath(filename)
    if (
        windows.is_absolute()
        or bool(windows.drive)
        or posix.is_absolute()
        or "/" in filename
        or "\\" in filename
        or ":" in filename
        or any(part == ".." for part in windows.parts + posix.parts)
    ):
        raise UnsafeApiPathError("Only a base filename is accepted; paths are forbidden.")
    return filename


def resolve_job_directory(output_root: Path, job_id: UUID | str) -> Path:
    """Return a UUID-named directory confined to the configured output root."""
    try:
        canonical_job_id = str(UUID(str(job_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise UnsafeApiPathError("Job ID must be a valid UUID.") from exc
    root = output_root.resolve(strict=False)
    candidate = (root / canonical_job_id).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # Defensive even though UUID canonicalization is strict.
        raise UnsafeApiPathError("Job directory escaped the configured output root.") from exc
    return candidate


def resolve_job_artifact(
    output_root: Path, job_id: UUID | str, artifact_name: str
) -> Path:
    """Resolve one internally named artifact without creating it."""
    safe_name = validate_public_filename(artifact_name)
    job_directory = resolve_job_directory(output_root, job_id)
    candidate = (job_directory / safe_name).resolve(strict=False)
    try:
        candidate.relative_to(job_directory)
    except ValueError as exc:
        raise UnsafeApiPathError("Artifact escaped its job directory.") from exc
    return candidate
