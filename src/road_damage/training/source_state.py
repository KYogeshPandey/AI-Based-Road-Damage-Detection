"""Deterministic checksum manifest for baseline-relevant project source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.dataset.rdd2022_common import PROJECT_ROOT, sha256_file, write_json_atomic  # noqa: E402
from road_damage.training.baseline_support import BaselineTrainingError  # noqa: E402


INCLUDED_DIRECTORIES = ("src", "configs", "tests", "road-damage-project-docs")
INCLUDED_ROOT_FILES = ("AGENTS.md", "requirements.txt", ".gitignore")
MANIFEST_RELATIVE_PATH = Path("reproducibility") / "baseline_public_v1_source_state_manifest.json"


def _is_included_file(path: Path) -> bool:
    if "__pycache__" in path.parts:
        return False
    if path.suffix.casefold() in {".pyc", ".pyo", ".tmp", ".part", ".partial"}:
        return False
    return path.is_file()


def build_source_state_manifest(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Hash the exact deterministic source/config/test/documentation scope."""
    project_root = project_root.resolve()
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
    files = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique
    ]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "baseline_public_v1.source_state.v1",
        "algorithm": "sha256",
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
