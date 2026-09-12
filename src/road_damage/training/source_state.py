"""Deterministic checksum manifest for baseline-relevant project source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_common import PROJECT_ROOT, sha256_file, write_json_atomic  # noqa: E402
from road_damage.training.baseline_support import BaselineTrainingError  # noqa: E402


INCLUDED_DIRECTORIES = ("src", "configs", "tests", "road-damage-project-docs")
INCLUDED_ROOT_FILES = ("AGENTS.md", "requirements.txt", ".gitignore")
MANIFEST_RELATIVE_PATH = Path("reproducibility") / "baseline_public_v1_source_state_manifest.json"
SOURCE_STATE_MODE_FILESYSTEM = "filesystem"
SOURCE_STATE_MODE_COMMITTED_GIT_TREE = "committed_git_tree"


def _is_included_file(path: Path) -> bool:
    if "__pycache__" in path.parts:
        return False
    if path.suffix.casefold() in {".pyc", ".pyo", ".tmp", ".part", ".partial"}:
        return False
    return path.is_file()


def _filesystem_files(project_root: Path) -> list[dict[str, Any]]:
    """Preserve the historical worktree-recursion semantics used by the baseline."""
    paths: list[Path] = []
    for name in INCLUDED_DIRECTORIES:
        directory = project_root / name
        if not directory.is_dir():
            raise BaselineTrainingError(f"Source-state directory is missing: {directory}")
        paths.extend(path for path in directory.rglob("*") if _is_included_file(path))
    for name in INCLUDED_ROOT_FILES:
        path = project_root / name
        if not path.is_file():
            raise BaselineTrainingError(f"Source-state file is missing: {path}")
        paths.append(path)
    unique = sorted(set(paths), key=lambda path: path.relative_to(project_root).as_posix())
    return [
        {
            "path": path.relative_to(project_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique
    ]


def _run_git_bytes(project_root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise BaselineTrainingError(
            f"Could not read committed Git source state with: git {' '.join(args)}"
        ) from exc


def _is_included_committed_path(relative: PurePosixPath) -> bool:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False
    if "__pycache__" in relative.parts or any(part.startswith(".venv") for part in relative.parts):
        return False
    if relative.suffix.casefold() in {".pyc", ".pyo", ".tmp", ".part", ".partial"}:
        return False
    value = relative.as_posix()
    return value in INCLUDED_ROOT_FILES or relative.parts[0] in INCLUDED_DIRECTORIES


def _committed_git_tree_files(project_root: Path) -> tuple[str, list[dict[str, Any]]]:
    top_level = Path(
        _run_git_bytes(project_root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if top_level != project_root:
        raise BaselineTrainingError(
            f"Git top-level directory is {top_level}, expected {project_root}."
        )
    commit = _run_git_bytes(project_root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    records = _run_git_bytes(project_root, "ls-tree", "-r", "-z", "--full-tree", commit)
    entries: list[tuple[PurePosixPath, str]] = []
    for record in records.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            _mode, object_type, object_id = metadata.split(b" ", 2)
            relative = PurePosixPath(raw_path.decode("utf-8", errors="surrogateescape"))
        except (ValueError, UnicodeError) as exc:
            raise BaselineTrainingError("Could not parse the committed Git tree.") from exc
        if not _is_included_committed_path(relative):
            continue
        if object_type != b"blob":
            raise BaselineTrainingError(f"Committed source path is not a Git blob: {relative.as_posix()}")
        entries.append((relative, object_id.decode("ascii")))

    observed_paths = {relative.as_posix() for relative, _object_id in entries}
    for name in INCLUDED_DIRECTORIES:
        if not any(path.startswith(f"{name}/") for path in observed_paths):
            raise BaselineTrainingError(f"Source-state directory is missing from committed HEAD: {name}")
    for name in INCLUDED_ROOT_FILES:
        if name not in observed_paths:
            raise BaselineTrainingError(f"Source-state file is missing from committed HEAD: {name}")

    files = []
    for relative, object_id in sorted(entries, key=lambda item: item[0].as_posix()):
        content = _run_git_bytes(project_root, "cat-file", "blob", object_id)
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return commit, files


def build_source_state_manifest(
    project_root: Path = PROJECT_ROOT,
    *,
    mode: str = SOURCE_STATE_MODE_FILESYSTEM,
) -> dict[str, Any]:
    """Hash source scope from the historical filesystem or exact committed HEAD tree."""
    project_root = project_root.resolve()
    mode_metadata: dict[str, Any] = {}
    if mode == SOURCE_STATE_MODE_FILESYSTEM:
        files = _filesystem_files(project_root)
    elif mode == SOURCE_STATE_MODE_COMMITTED_GIT_TREE:
        commit, files = _committed_git_tree_files(project_root)
        mode_metadata = {"mode": mode, "git_commit": commit}
    else:
        raise BaselineTrainingError(f"Unsupported source-state mode: {mode}")

    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "baseline_public_v1.source_state.v1",
        "algorithm": "sha256",
        **mode_metadata,
        "included_directories": list(INCLUDED_DIRECTORIES),
        "included_root_files": list(INCLUDED_ROOT_FILES),
        "excluded_generated_content": [
            "data", "models", "outputs", "runs", "weights", ".venv*",
            "__pycache__", "*.pyc", "temporary files",
        ],
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "source_tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def verify_source_state_manifest(
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Require the committed secondary manifest to match current source exactly."""
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineTrainingError(f"Could not read source-state manifest: {manifest_path}") from exc
    observed = build_source_state_manifest(project_root)
    if recorded != observed:
        raise BaselineTrainingError(
            "Source-state checksum manifest is stale; regenerate and commit it before training."
        )
    return observed


def write_source_state_manifest(
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Generate the deterministic secondary source-state manifest."""
    manifest = build_source_state_manifest(project_root)
    write_json_atomic(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the deterministic baseline source-state manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / MANIFEST_RELATIVE_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = write_source_state_manifest(args.output.resolve())
    except BaselineTrainingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: manifest[key] for key in ("file_count", "total_bytes", "source_tree_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
