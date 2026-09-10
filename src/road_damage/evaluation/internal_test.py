"""Explicit user-run, first/final frozen baseline test; inference-free preflight."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.evaluation import select_threshold as shared  # noqa: E402
from road_damage.evaluation.threshold_metrics import Box, ThresholdError, sweep_cache  # noqa: E402

ROOT = shared.ROOT
CONFIG = Path("configs/evaluation/baseline_public_v1_frozen_operating_point.yaml")
OUTPUT = Path("outputs/evaluation/baseline_public_v1_internal_test")
RECEIPT = Path("outputs/evaluation/baseline_public_v1_internal_test_attempt.json")
SELECTION_COMMIT = "027f5f41c386f558319799fa8784b32cf3dcefdb"
CODES = ("D00", "D10", "D20", "D40")
LOGGER = logging.getLogger(__name__)
# Independent safety contract: editing YAML alone cannot retune the final test.
EXPECTED_CONFIG = {
    "schema_version": "baseline_public_v1.frozen_internal_test.v2",
    "frozen_at_utc": "2026-09-09T17:33:29+00:00",
    "model": shared.FROZEN_MODEL_RELATIVE_PATH.as_posix(),
    "model_size_bytes": shared.FROZEN_MODEL_SIZE_BYTES, "model_sha256": shared.FROZEN_MODEL_SHA256,
    "confidence": 0.19, "nms_iou": 0.5, "matching_iou": 0.5,
    "classes": {str(k): v for k, v in shared.CLASS_NAMES.items()},
    "selection": {
        "validation_only": True, "internal_test_untouched": True,
        "commit": SELECTION_COMMIT, "timestamp": "2026-09-09T17:16:17.442283+00:00",
        "macro_f1": 0.5055312316193121,
        "per_class_f1": {"D00": 0.4257703081232493, "D10": 0.4652908067542214,
                         "D20": 0.6515151515151515, "D40": 0.4795486600846262},
        "negative_images": 983, "negative_fp": 121, "negative_fp_per_image": 0.12309257375381485,
        "negative_images_with_fp": 96, "negative_images_with_fp_fraction": 0.09766022380467955,
        "selected_artifact_sha256": "5220c1ebcc17b99b5e1d46076024a4c592a5f636fe072a42759431a29f0eea45",
        "evaluation_manifest_sha256": "8950cd975469d89344a6d26903dce3d48314eaded4ee93bdf10649ddfffc5bdb",
    },
    "export": shared.EXPORT.as_posix(),
    "approved_metadata_sha256": {
        "export_manifest.json": "cf6c46454d20d7e6a033b5f47a09c77b971c620517d0f4b4b24104939aa0e1c6",
        "export_report.json": "f6054fd350fb9a2702f252062c7238d0cdb0dee4e3626d33bb532c8e9c3a9e14",
        "validation_report.json": "4291d4f213a3e1e22b8a4728bbefcf8c6d6ad9e24ec4b9efe6c9ce4e534411a2",
    },
    "expected_counts": {"images": 2775, "positive_images": 1712, "negative_images": 1063,
                        "targets": 3535, "countries": {"India": 1116, "Japan": 1659},
                        "class_boxes": {"D00": 841, "D10": 557, "D20": 1297, "D40": 840}},
    "imgsz": 640, "device": 0, "standard_ap_confidence_floor": 0.001,
    "standard_ap_matching_ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "ultralytics_version": "8.4.130", "batch": 1, "rect": False, "half": False, "augment": False,
    "max_det": 300, "seed": 42, "progress_every": 100, "output": OUTPUT.as_posix(),
    "attempt_receipt": RECEIPT.as_posix(),
    "evaluation_branches": ["framework_validator", "frozen_prediction_mode"],
    "validator_multi_label": True, "validator_workers": 0,
    "policy": "one_frozen_identity_technical_recovery_only_completed_lock",
}


def load_config(root: Path = ROOT) -> dict[str, Any]:
    """Only the reviewed record is allowed; no alternate model or threshold options."""
    path = shared.checked_path(CONFIG, CONFIG, root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if shared.json_bytes(value) != shared.json_bytes(EXPECTED_CONFIG):
        raise ThresholdError("Frozen internal-test config changed; overrides/unknown keys are forbidden.")
    return value


def hashed_json(root: Path, relative: Path, expected_hash: str) -> dict[str, Any]:
    path = shared.checked_path(relative, relative, root)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ThresholdError(f"Approved artifact SHA-256 mismatch: {relative}")
    return json.loads(content)


def selection_provenance(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Verify completed selection artifacts, not another validation run or sweep."""
    expected = config["selection"]
    selected = hashed_json(root, shared.OUTPUT / "selected_operating_point.json", expected["selected_artifact_sha256"])
    manifest = hashed_json(root, shared.OUTPUT / "evaluation_manifest.json", expected["evaluation_manifest_sha256"])
    if (manifest["status"] != "complete" or selected["git_commit"] != SELECTION_COMMIT
            or selected["validation_only"] is not True or selected["internal_test_untouched"] is not True
            or selected["model_sha256"] != config["model_sha256"]
            or selected["selected_global_confidence"] != config["confidence"]
            or selected["selected_nms_iou"] != config["nms_iou"]
            or selected["matching_iou"] != config["matching_iou"]
            or selected["macro_f1"] != expected["macro_f1"]):
        raise ThresholdError("Validation-only selection identity or frozen operating point mismatch.")
    for name, sha in manifest["artifacts"].items():
        if Path(name).name != name:
            raise ThresholdError("Selection artifact must be a local filename.")
        path = shared.checked_path(shared.OUTPUT / name, shared.OUTPUT / name, root)
        if shared.digest(path) != sha:
            raise ThresholdError(f"Validation selection artifact changed: {name}")
    return {**expected, "artifacts_verified": manifest["artifacts"]}


def support(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count actual cached targets and country support, independent of expected totals."""
    return {"images": len(records), "positive_images": sum(bool(r["ground_truth"]) for r in records),
            "negative_images": sum(not r["ground_truth"] for r in records),
            "targets": sum(len(r["ground_truth"]) for r in records),
            "countries": {country: sum(r["country"] == country for r in records) for country in ("India", "Japan")},
            "class_boxes": {code: sum(b["class_id"] == c for r in records for b in r["ground_truth"])
                            for c, code in enumerate(CODES)}}


def verify_counts(records: Sequence[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    actual = support(records)
    if actual != expected:
        raise ThresholdError(f"Internal-test count mismatch: expected {expected}; observed {actual}")
    return actual


def approved_metadata(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Only read the pinned metadata; raw official-test paths are never followed."""
    values = {name: hashed_json(root, shared.EXPORT / name, sha)
              for name, sha in config["approved_metadata_sha256"].items()}
    manifest, report, validation = (values[n] for n in ("export_manifest.json", "export_report.json", "validation_report.json"))
    taxonomy = [{"id": c, "raw_code": CODES[c], "name": name} for c, name in shared.CLASS_NAMES.items()]
    if (manifest["taxonomy"] != taxonomy or report["status"] != "export_validated"
            or validation["status"] != "passed" or validation["exclusions"]["passed"] is not True
            or any(validation["exclusions"]["intersections"].values())
            or validation["exclusions"]["teacher_reference_count"] != 0):
        raise ThresholdError("Approved export taxonomy/exclusion evidence failed.")
    records = sorted((r for r in manifest["images"] if r["derived_split"] == "test"), key=lambda r: r["original_filename"])
    names = set()
    for r in records:
        name, country = r["original_filename"], r["country"]
        if (country not in ("India", "Japan") or not re.fullmatch(rf"{country}_\d+\.jpg", name)
                or name in names or r["exported_image_path"] != f"images/test/{name}"
                or r["exported_label_path"] != f"labels/test/{Path(name).stem}.txt"
                or r["raw_relative_image_path"] != f"data/external/rdd2022/raw/{country}/train/images/{name}"
                or r["source_image_sha256"] != r["exported_image_sha256"]):
            raise ThresholdError("Non-canonical/official-test/teacher source or duplicate in test metadata.")
        names.add(name)
    metadata_records = [{"country": r["country"], "ground_truth": [
        {"class_id": c} for c, code in enumerate(CODES) for _ in range(r["target_class_counts"].get(code, 0))]}
        for r in records]
    verify_counts(metadata_records, config["expected_counts"])
    if (report["counts"]["split_images"]["test"] != len(records)
            or validation["split_image_counts"]["test"] != len(records)
            or validation["target_object_counts"]["test"] != config["expected_counts"]["class_boxes"]):
        raise ThresholdError("Approved reports disagree on internal-test counts.")
    return records


def inventory_internal_test(capability: object, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Authorized evaluation only: verify allowlisted pairs; NEVER called by preflight."""
    # This must remain the first operation. An unauthorized caller must not reach
    # imports, path resolution, enumeration, stat/header reads, hashing or opens.
    _require_inventory_capability(capability, root, config)
    from PIL import Image

    approved = approved_metadata(root, config)
    images = shared.checked_path(shared.EXPORT / "images/test", shared.EXPORT / "images/test", root)
    labels = shared.checked_path(shared.EXPORT / "labels/test", shared.EXPORT / "labels/test", root)
    expected_names = {r["original_filename"] for r in approved}
    if ({p.name for p in images.iterdir()} != expected_names
            or {p.name for p in labels.iterdir()} != {Path(n).stem + ".txt" for n in expected_names}):
        raise ThresholdError("Internal-test missing/orphan/unexpected image or label.")
    records = []
    for r in approved:
        image_path = shared.checked_path(shared.EXPORT / r["exported_image_path"], shared.EXPORT / r["exported_image_path"], root)
        label_path = shared.checked_path(shared.EXPORT / r["exported_label_path"], shared.EXPORT / r["exported_label_path"], root)
        image_sha, label_sha = shared.digest(image_path), shared.digest(label_path)
        if image_sha != r["exported_image_sha256"] or label_sha != r["label_sha256"]:
            raise ThresholdError(f"Canonical exported pair changed: {r['original_filename']}")
        with Image.open(image_path) as image:
            width, height = image.size
        raw = label_path.read_bytes()
        gt = []
        for line in raw.decode("utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5 or fields[0] not in {"0", "1", "2", "3"}:
                raise ThresholdError(f"Invalid target label: {label_path}")
            cx, cy, w, h = map(float, fields[1:])
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ThresholdError("Invalid normalized test box.")
            xyxy = ((cx - w / 2) * width, (cy - h / 2) * height,
                    (cx + w / 2) * width, (cy + h / 2) * height)
            if xyxy[0] < -1e-4 or xyxy[1] < -1e-4 or xyxy[2] > width + 1e-4 or xyxy[3] > height + 1e-4:
                raise ThresholdError("Test box extends beyond export rounding tolerance.")
            gt.append(shared.box_record(Box(int(fields[0]), (max(0., xyxy[0]), max(0., xyxy[1]),
                                                            min(width, xyxy[2]), min(height, xyxy[3])))))
        counts = {code: sum(b["class_id"] == c for b in gt) for c, code in enumerate(CODES)}
        if (bool(gt) != r["is_positive"] or len(gt) != r["target_object_count"] or (raw and not gt)
                or counts != {code: r["target_class_counts"].get(code, 0) for code in CODES}):
            raise ThresholdError("Test label content does not reconcile with approved metadata.")
        records.append({"image_id": r["original_filename"], "country": r["country"],
                        "image_path": image_path.relative_to(root).as_posix(),
                        "label_path": label_path.relative_to(root).as_posix(), "width": width, "height": height,
                        "image_sha256": image_sha, "label_sha256": label_sha, "ground_truth": gt})
    return {"schema_version": "internal_test_ground_truth.v1", "split": "canonical_internal_test",
            "counts": verify_counts(records, config["expected_counts"]), "images": records}


def git_provenance(root: Path) -> dict[str, Any]:
    provenance = shared.git_provenance(root)
    def git(*args: str) -> str:
        result = subprocess.run(["git", "--no-optional-locks", *args], cwd=root, capture_output=True, text=True)
        if result.returncode:
            raise ThresholdError(f"Internal-test Git provenance failed: {result.stderr.strip()}")
        return result.stdout.strip()
    git("merge-base", "--is-ancestor", SELECTION_COMMIT, "HEAD")
    if provenance["evaluation_commit"] == SELECTION_COMMIT:
        raise ThresholdError("Review and commit frozen internal-test tooling before user preflight/evaluation.")
    for path in (CONFIG.as_posix(), "src/road_damage/evaluation/internal_test.py",
                 "src/road_damage/evaluation/internal_test_metrics.py", "tests/test_internal_test.py",
                 "road-damage-project-docs/INTERNAL_TEST_RUNBOOK.md"):
        git("ls-files", "--error-unmatch", path)
    provenance["threshold_selection_commit"] = SELECTION_COMMIT
    return provenance


def operating_point(gt: dict[str, Any], predictions: Mapping[str, Sequence[Box]]) -> dict[str, Any]:
    """Exactly one frozen point, sharing the original validation matching implementation."""
    truth = {r["image_id"]: [Box(b["class_id"], tuple(b["xyxy"])) for b in r["ground_truth"]] for r in gt["images"]}
    if len(truth) != len(gt["images"]):
        raise ThresholdError("Duplicate test cache image identifier.")
    row = sweep_cache(truth, predictions, 0.5, [0.19], 0.5)[0]
    tp, fp, fn = (row[f"total_{k}"] for k in ("tp", "fp", "fn"))
    row.update({"matching_iou": 0.5, "precision": tp / (tp + fp) if tp + fp else 0.,
                "recall": tp / (tp + fn) if tp + fn else 0.,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.})
    return row


def country_metrics(gt: dict[str, Any], predictions: Mapping[str, Sequence[Box]]) -> dict[str, Any]:
    results = {}
    for country in ("India", "Japan"):
        records = [r for r in gt["images"] if r["country"] == country]
        counts = support(records)
        counts["class_positive_images"] = {code: sum(any(b["class_id"] == c for b in r["ground_truth"]) for r in records)
                                           for c, code in enumerate(CODES)}
        results[country] = {"support": counts, "operating_point": operating_point(
            {"images": records}, {r["image_id"]: predictions[r["image_id"]] for r in records}),
            "warnings": ([f"India D10 LOW SUPPORT: {counts['class_boxes']['D10']} boxes / "
                          f"{counts['class_positive_images']['D10']} images; no strong country-specific conclusions."]
                         if country == "India" else [])}
    return results


RECEIPT_SCHEMA = "frozen_test_attempts.v2"
_ACTIVE_CONTEXT: _EvaluationContext | None = None
_ACTIVE_INVENTORY_CAPABILITY: tuple[object, Path, Path, bytes, bytes] | None = None
RECOVERY_IDENTITY_FIELDS = frozenset({
    "schema_version", "project_root", "model_path", "model_sha256", "model_size_bytes",
    "git_commit", "config_sha256", "confidence", "nms_iou", "matching_iou",
    "class_mapping", "evaluation_code", "framework", "dataset_metadata_sha256",
    "test_records_sha256", "threshold_selection",
})


def _atomic_json(path: Path, value: Any) -> None:
    """Flush a complete replacement before publishing it atomically."""
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        handle.write(shared.json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _write_once(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(shared.json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _durable_artifact_hash(path: Path) -> str:
    """Flush report/cache bytes to storage before a completion manifest can cite them."""
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return shared.digest(path)


def _framework_identity() -> dict[str, Any]:
    """Version and implementation hashes, without importing a model/framework validator."""
    result = shared.environment()
    distribution = importlib.metadata.distribution("ultralytics")
    names = sorted(str(p).replace("\\", "/") for p in distribution.files or ()
                   if str(p).replace("\\", "/").startswith("ultralytics/")
                   and p.suffix in (".py", ".yaml", ".yml"))
    if not names or "ultralytics/models/yolo/detect/val.py" not in names:
        raise ThresholdError("Installed framework source inventory is unavailable; identity cannot be frozen.")
    result["ultralytics_source_sha256"] = {
        name: shared.digest(Path(distribution.locate_file(name))) for name in names}
    return result


def _metadata_support(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [{"country": r["country"], "ground_truth": [
        {"class_id": c} for c, code in enumerate(CODES)
        for _ in range(r["target_class_counts"].get(code, 0))]} for r in records]
    return support(rows)


def _recovery_identity(root: Path, config: dict[str, Any], records: Sequence[dict[str, Any]],
                       provenance: dict[str, Any], framework: dict[str, Any],
                       selection: dict[str, Any]) -> dict[str, Any]:
    """Canonical all-fields-critical identity used for new attempts and recovery."""
    identity = {
        "schema_version": config["schema_version"], "project_root": str(root.absolute()),
        "model_path": config["model"], "model_sha256": config["model_sha256"],
        "model_size_bytes": config["model_size_bytes"], "git_commit": provenance["evaluation_commit"],
        "config_sha256": shared.digest(root / CONFIG),
        "confidence": config["confidence"], "nms_iou": config["nms_iou"], "matching_iou": config["matching_iou"],
        "class_mapping": config["classes"], "evaluation_code": provenance,
        "framework": framework, "dataset_metadata_sha256": config["approved_metadata_sha256"],
        "test_records_sha256": hashlib.sha256(shared.json_bytes(records)).hexdigest(),
        "threshold_selection": selection,
    }
    if set(identity) != RECOVERY_IDENTITY_FIELDS:
        raise ThresholdError("Internal recovery identity schema is incomplete or contains an unreviewed field.")
    return identity


def _prepare_metadata(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Shared gate: metadata/config/model/Git only. NEVER inventory test directories."""
    config = load_config(root)
    provenance = git_provenance(root)
    selection = selection_provenance(root, config)
    model = shared.checked_path(Path(config["model"]), shared.FROZEN_MODEL_RELATIVE_PATH, root)
    shared.verify_model_identity(model, expected_sha256=config["model_sha256"], expected_size_bytes=config["model_size_bytes"])
    framework = _framework_identity()
    records = approved_metadata(root, config)
    identity = _recovery_identity(root, config, records, provenance, framework, selection)
    return config, records, identity


def _receipt(root: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    """Fail closed on unknown state, changed identity, or ANY completed marker."""
    output = shared.checked_path(OUTPUT, OUTPUT, root)
    receipt_path = shared.checked_path(RECEIPT, RECEIPT, root)
    if (output / "evaluation_manifest.json").exists():
        # This name is exclusively the final publication marker.
        raise ThresholdError("COMPLETED/final publication exists; all further baseline evaluation is refused.")
    if not receipt_path.exists():
        if output.exists():
            raise ThresholdError("Unrecognized existing output without receipt; refusing overwrite.")
        return None
    value = json.loads(receipt_path.read_bytes())
    if value.get("schema_version") != RECEIPT_SCHEMA:
        raise ThresholdError("Unknown/legacy receipt: no automatic migration or overwrite.")
    if value.get("status") == "COMPLETED":
        raise ThresholdError("COMPLETED baseline evaluation is permanently locked.")
    if value.get("status") not in ("STARTED", "FAILED_TECHNICAL") or not value.get("attempts"):
        raise ThresholdError("Invalid attempt state.")
    if shared.json_bytes(value["identity"]) != shared.json_bytes(identity):
        raise ThresholdError("Recovery rejected: frozen evaluation identity changed.")
    if hashlib.sha256(shared.json_bytes(identity)).hexdigest() != value["identity_sha256"]:
        raise ThresholdError("Receipt identity hash mismatch.")
    pinned = shared.checked_path(OUTPUT / "identity.json", OUTPUT / "identity.json", root)
    if pinned.exists() and pinned.read_bytes() != shared.json_bytes(identity):
        raise ThresholdError("Immutable evaluation identity changed or missing.")
    for index, attempt in enumerate(value["attempts"], 1):
        relative = OUTPUT / "attempts" / f"{index:04d}"
        if attempt["path"] != relative.as_posix() or attempt["number"] != index:
            raise ThresholdError("Invalid attempt path/sequence.")
        if attempt.get("status") == "COMPLETED":
            raise ThresholdError("Completed ledger attempt exists; baseline evaluation is locked.")
        if attempt.get("status") not in ("STARTED", "FAILED_TECHNICAL"):
            raise ThresholdError("Unknown attempt state; refusing automatic recovery.")
        directory = shared.checked_path(relative, relative, root)
        if (directory / "COMPLETED.json").exists():
            raise ThresholdError("A completed attempt exists; baseline evaluation is locked.")
        if (directory / "INCOMPLETE.json").exists():
            raise ThresholdError("Integrity/scientific failure is not eligible for technical recovery.")
        if (directory / "evaluation_manifest.json").exists():
            saved = json.loads((directory / "evaluation_manifest.json").read_bytes())
            if saved.get("status") == "COMPLETED":
                raise ThresholdError("Completed attempt manifest exists; baseline evaluation is locked.")
    return value


def _require_inventory_capability(capability: object, root: Path, config: dict[str, Any]) -> None:
    """Authorize real test-file access only for the one process-local STARTED attempt."""
    active = _ACTIVE_INVENTORY_CAPABILITY
    if active is None or capability is not active[0]:
        raise ThresholdError("Internal-test inventory requires an active opaque evaluation capability.")
    _, authorized_root, authorized_output, identity_json, config_json = active
    if (Path(root).absolute() != authorized_root or shared.json_bytes(config) != config_json):
        raise ThresholdError("Internal-test inventory capability does not match this root/configuration.")
    identity = json.loads(identity_json)
    state = _receipt(root, identity)
    if (state is None or state["status"] != "STARTED"
            or root / state["attempts"][-1]["path"] != authorized_output):
        raise ThresholdError("Internal-test inventory capability is not bound to the active STARTED attempt.")


def preflight(root: Path = ROOT) -> dict[str, Any]:
    """Metadata-only: no test image/label opening, hashing, header reading or enumeration."""
    config, records, identity = _prepare_metadata(root)
    state = _receipt(root, identity)
    output = shared.checked_path(OUTPUT, OUTPUT, root)
    if state is None:
        shared.ensure_output_available(output)
    return {"passed": True, "metadata_only": True, "inference_executed": False,
            "test_images_or_labels_accessed": False, "counts": _metadata_support(records),
            "countries": {c: _metadata_support([r for r in records if r["country"] == c]) for c in ("India", "Japan")},
            "identity": identity, "output": str(output),
            "attempt_state": None if state is None else state["status"],
            "technical_recovery_requires_explicit_flag": state is not None,
            "note": "Prior non-evaluation is user-attested and supported by selection provenance; not a forensic guarantee."}


@contextlib.contextmanager
def _process_lock(root: Path):
    """OS-held nonblocking lock. Process death releases it; never delete the lock file."""
    path = shared.checked_path(Path(str(RECEIPT) + ".lock"), Path(str(RECEIPT) + ".lock"), root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ThresholdError("Another baseline evaluation process holds the lock.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _start_attempt(root: Path, identity: dict[str, Any], recover: bool) -> tuple[Path, dict[str, Any]]:
    """Called only while holding the OS lock; preserve each failed attempt permanently."""
    value = _receipt(root, identity)
    if value is not None and not recover:
        raise ThresholdError("Prior incomplete attempt exists. Use explicit --recover-technical with identical identity.")
    if value is None and recover:
        raise ThresholdError("No prior attempt exists to recover.")
    output = shared.checked_path(OUTPUT, OUTPUT, root)
    now = datetime.now(timezone.utc).isoformat()
    if value is None:
        value = {"schema_version": RECEIPT_SCHEMA, "identity": identity,
                 "identity_sha256": hashlib.sha256(shared.json_bytes(identity)).hexdigest(),
                 "status": "STARTED", "attempts": [],
                 "user_attests_previously_unevaluated": True}
    elif value["status"] == "STARTED":
        # A live owner cannot reach here: the nonblocking OS lock was acquired.
        previous = value["attempts"][-1]
        previous_path = root / previous["path"]
        previous_path.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILED_TECHNICAL", "at_utc": now,
                   "reason": "Prior process exited without completion; exclusive OS lock reacquired.",
                   "identity_sha256": value["identity_sha256"]}
        if not (previous_path / "FAILED_TECHNICAL.json").exists():
            _write_once(previous_path / "FAILED_TECHNICAL.json", failure)
        previous.update(failure)
        value["status"] = "FAILED_TECHNICAL"
        _atomic_json(root / RECEIPT, value)
    number = len(value["attempts"]) + 1
    relative = OUTPUT / "attempts" / f"{number:04d}"
    attempt = {"number": number, "path": relative.as_posix(), "status": "STARTED", "at_utc": now}
    value["attempts"].append(attempt)
    value["status"] = "STARTED"
    (root / RECEIPT).parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / RECEIPT, value)  # identity is durable BEFORE any test content access
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "identity.json").exists():
        _write_once(output / "identity.json", identity)
    directory = shared.checked_path(relative, relative, root)
    directory.mkdir(parents=True, exist_ok=False)
    _write_once(directory / "STARTED.json", {**attempt, "identity_sha256": value["identity_sha256"]})
    return directory, value


@dataclass(frozen=True)
class _EvaluationContext:
    root: Path
    output: Path
    identity_json: bytes
    config_json: bytes
    ground_truth_sha256: str
    inventory_capability: object


def _require_context(context: _EvaluationContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process capability plus durable identity gates; no caller-supplied model/config."""
    if context is not _ACTIVE_CONTEXT or not isinstance(context, _EvaluationContext):
        raise ThresholdError("Inference requires the active authorized frozen evaluation context.")
    _require_inventory_capability(context.inventory_capability, context.root, json.loads(context.config_json))
    identity = json.loads(context.identity_json)
    state = _receipt(context.root, identity)
    if state is None or state["status"] != "STARTED" or context.root / state["attempts"][-1]["path"] != context.output:
        raise ThresholdError("Attempt is not authorized and STARTED.")
    config, _, current_identity = _prepare_metadata(context.root)
    if shared.json_bytes(current_identity) != context.identity_json:
        raise ThresholdError("Authorized evaluation identity changed before inference.")
    if shared.json_bytes(config) != context.config_json:
        raise ThresholdError("Frozen configuration changed.")
    if shared.digest(context.output / "ground_truth_cache.json") != context.ground_truth_sha256:
        raise ThresholdError("Authorized ground-truth snapshot changed.")
    shared.verify_model_identity(context.root / config["model"], expected_sha256=config["model_sha256"],
                                 expected_size_bytes=config["model_size_bytes"])
    gt = json.loads((context.output / "ground_truth_cache.json").read_bytes())
    verify_counts(gt["images"], config["expected_counts"])
    return config, gt


class _CacheWriter:
    """Hash bytes as written, close+fsync, then publish an independent cache seal."""
    def __init__(self, path: Path, branch: str, identity_sha256: str):
        self.path, self.branch, self.identity = path, branch, identity_sha256
        self.sha = hashlib.sha256()
        self.count = 0

    def __enter__(self):
        self.handle = self.path.open("xb")
        return self

    def write(self, value: dict[str, Any]) -> None:
        row = {"schema_version": "internal_test_prediction.v2", "branch": self.branch,
               "identity_sha256": self.identity, **value}
        payload = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self.handle.write(payload)
        self.handle.flush()
        self.sha.update(payload)
        self.count += 1

    def __exit__(self, exc_type, exc, tb):
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.handle.close()
        if exc_type is None:
            _write_once(self.path.with_suffix(".seal.json"), {
                "sha256": self.sha.hexdigest(), "records": self.count,
                "branch": self.branch, "identity_sha256": self.identity})


def _snapshot_inputs(root: Path, output: Path, gt: dict[str, Any]) -> Path:
    """Byte-copy verified pairs; framework caches stay inside this attempt, never export."""
    snapshot = output / "inputs"
    for r in gt["images"]:
        for kind, key, sha_key in (("images", "image_path", "image_sha256"), ("labels", "label_path", "label_sha256")):
            source = shared.checked_path(Path(r[key]), shared.EXPORT / kind / "test" / Path(r[key]).name, root)
            dest = snapshot / kind / "test" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise ThresholdError("Refusing to overwrite snapshot input.")
            shutil.copyfile(source, dest)
            if shared.digest(dest) != r[sha_key]:
                raise ThresholdError("Snapshot pair does not match the approved immutable source.")
    yaml_path = snapshot / "data.yaml"
    # Framework schema requires train/val keys. All aliases refer ONLY to the same internal test.
    import yaml
    with yaml_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump({"path": snapshot.as_posix(), "train": "images/test", "val": "images/test",
                        "test": "images/test", "names": shared.CLASS_NAMES}, handle, sort_keys=False)
    return yaml_path


def _verify_snapshot(output: Path, gt: dict[str, Any]) -> None:
    for r in gt["images"]:
        for kind, key, sha_key in (("images", "image_path", "image_sha256"), ("labels", "label_path", "label_sha256")):
            path = output / "inputs" / kind / "test" / Path(r[key]).name
            if shared.digest(path) != r[sha_key]:
                raise ThresholdError("Framework snapshot input changed (including repair/deduplication); completion blocked.")


def _set_determinism(config: dict[str, Any]) -> None:
    import random
    import numpy as np
    import torch
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _validator_settings(context: _EvaluationContext, config: dict[str, Any]) -> dict[str, Any]:
    return {"model": str(context.root / config["model"]), "data": str(context.output / "inputs/data.yaml"),
            "task": "detect", "mode": "val", "split": "test", "imgsz": config["imgsz"],
            "conf": config["standard_ap_confidence_floor"], "iou": config["nms_iou"],
            "batch": 1, "device": config["device"], "workers": config["validator_workers"],
            "rect": False, "quantize": None, "augment": False, "max_det": config["max_det"],
            "single_cls": False, "agnostic_nms": False, "classes": None, "fraction": 1.,
            "plots": False, "save_json": False, "save_txt": False, "save_conf": False,
            "save": False, "visualize": False, "verbose": False, "cache": False,
            "compile": False, "dnn": False, "channels_last": False, "seed": config["seed"],
            "deterministic": True, "project": str(context.output), "name": "framework_validator", "exist_ok": False}


def _run_standard(context: _EvaluationContext) -> None:
    """Actual DetectionValidator execution; hooks only capture native metric inputs."""
    config, gt = _require_context(context)
    from unittest.mock import patch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.data.utils import check_det_dataset
    from road_damage.evaluation.internal_test_metrics import metric_values
    expected = {r["image_id"]: r for r in gt["images"]}
    settings = _validator_settings(context, config)
    identity_sha = hashlib.sha256(context.identity_json).hexdigest()
    started = time.perf_counter()

    class AuditedValidator(DetectionValidator):
        def __call__(self, *args, **kwargs):
            _require_context(context)
            return super().__call__(*args, **kwargs)

        def init_metrics(self, model):
            super().init_metrics(model)
            if self.names != shared.CLASS_NAMES:
                raise ThresholdError("Validator model class mapping differs from the frozen baseline.")

        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                name = Path(batch["im_file"][si]).name
                if name not in expected:
                    raise ThresholdError("Validator read an unexpected image identity.")
                r = expected[name]
                prepared = self._prepare_batch(si, batch)
                target_classes = prepared["cls"].cpu().tolist()
                if sorted(target_classes) != sorted(b["class_id"] for b in r["ground_truth"]):
                    raise ThresholdError("Framework removed/changed target labels; scientific integrity blocked.")
                if len(pred["cls"]) >= config["max_det"]:
                    raise ThresholdError("Standard validator detection cap reached; no configuration change/retry permitted.")
                from road_damage.evaluation.internal_test_metrics import tensor_boxes
                cache.write({"image_id": name, "country": r["country"], "nms_iou": .5,
                    "confidence_floor": .001, "multi_label": True,
                    "input_shape": list(batch["img"].shape[2:]), "original_shape": list(prepared["ori_shape"]),
                    "ratio_pad": prepared["ratio_pad"],
                    "ground_truth": tensor_boxes(prepared, confidence=False),
                    "predictions": tensor_boxes(pred, confidence=True),
                    "native_tp": self._process_batch(pred, prepared)["tp"].tolist()})
                if cache.count == 1 or cache.count % config["progress_every"] == 0:
                    elapsed = time.perf_counter() - started
                    LOGGER.info("Branch A validator %d/%d; elapsed %.1fs; ETA %.1fs",
                                cache.count, len(expected), elapsed, elapsed / cache.count * (len(expected) - cache.count))
            super().update_metrics(preds, batch)  # the genuine framework metric path, unchanged

    def safe_dataset(path, *args, **kwargs):
        if Path(path) != context.output / "inputs/data.yaml":
            raise ThresholdError("Validator dataset path escaped the authorized snapshot.")
        return check_det_dataset(path, autodownload=False, split="test")

    # Disable only automatic dataset/font downloads, never framework metric/postprocessing logic.
    with _CacheWriter(context.output / "standard_predictions.jsonl", "standard", identity_sha) as cache:
        with patch("ultralytics.engine.validator.check_det_dataset", safe_dataset), \
             patch("ultralytics.data.utils.check_font", lambda *a, **k: None):
            validator = AuditedValidator(args=settings)
            validator(model=str(context.root / config["model"]))
        if cache.count != len(expected) or validator.seen != len(expected):
            raise ThresholdError("Framework did not process every internal-test image.")
        live = metric_values(validator.metrics)
        if live["target_boxes"] != config["expected_counts"]["class_boxes"]:
            raise ThresholdError("Framework target support does not match approved metadata.")
        _write_once(context.output / "standard_live_metrics.json", live)
        _write_once(context.output / "validator_settings.json", {
            "requested": settings, "resolved": vars(validator.args), "multi_label": True,
            "metric_class": "ultralytics.models.yolo.detect.val.DetectionValidator",
            "automatic_dataset_and_font_downloads": False})
    LOGGER.info("Branch A validator completed %d/%d", len(expected), len(expected))


def _run_frozen(context: _EvaluationContext) -> None:
    """Exact prediction-mode branch; only an authorized capability is accepted."""
    config, gt = _require_context(context)
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(context.root / config["model"]), task="detect")
    if model.names != shared.CLASS_NAMES:
        raise ThresholdError("Frozen model class mapping mismatch.")
    args = shared.prediction_kwargs({**config, "inference_confidence_floor": .19}, .5)
    started = time.perf_counter()
    identity_sha = hashlib.sha256(context.identity_json).hexdigest()
    with _CacheWriter(context.output / "frozen_predictions.jsonl", "frozen", identity_sha) as cache:
        for r in gt["images"]:
            path = context.output / "inputs/images/test" / r["image_id"]
            if shared.digest(path) != r["image_sha256"]:
                raise ThresholdError("Snapshot image changed before prediction.")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or tuple(image.shape[:2]) != (r["height"], r["width"]):
                raise OSError("Snapshot image could not be decoded.")
            result = model.predict(source=image, **args)
            if len(result) != 1:
                raise ThresholdError("Expected one prediction result per image.")
            boxes = []
            if result[0].boxes is not None:
                for xyxy, conf, cls in zip(result[0].boxes.xyxy.cpu().tolist(), result[0].boxes.conf.cpu().tolist(),
                                           result[0].boxes.cls.cpu().tolist(), strict=True):
                    if int(cls) != cls:
                        raise ThresholdError("Non-integer prediction class.")
                    b = Box(int(cls), tuple(xyxy), float(conf))
                    boxes.append(shared.box_record(b))
            if len(boxes) >= config["max_det"]:
                raise ThresholdError("Frozen branch detection cap reached.")
            cache.write({"image_id": r["image_id"], "country": r["country"], "nms_iou": .5,
                         "confidence_floor": .19, "multi_label": False,
                         "predictions": boxes})
            if cache.count == 1 or cache.count % config["progress_every"] == 0 or cache.count == len(gt["images"]):
                elapsed = time.perf_counter() - started
                LOGGER.info("Branch B frozen %d/%d; elapsed %.1fs; ETA %.1fs", cache.count, len(gt["images"]),
                            elapsed, elapsed / cache.count * (len(gt["images"]) - cache.count))
    _write_once(context.output / "prediction_settings.json", args)


def _failure(root: Path, output: Path, state: dict[str, Any], manifest: dict[str, Any], exc: BaseException) -> None:
    technical = isinstance(exc, (KeyboardInterrupt, OSError, RuntimeError)) and not isinstance(
        exc, (ThresholdError, shared.MentorDemoError))
    status = "FAILED_TECHNICAL" if technical else "STARTED"
    failure = {"status": status, "incomplete": True, "recovery_permitted": technical,
               "error_type": type(exc).__name__, "error": str(exc),
               "at_utc": datetime.now(timezone.utc).isoformat()}
    marker = "FAILED_TECHNICAL.json" if technical else "INCOMPLETE.json"
    _write_once(output / marker, failure)
    manifest.update(failure)
    _atomic_json(output / "evaluation_manifest.json", manifest)
    state["status"] = status
    state["attempts"][-1].update(failure)
    _atomic_json(root / RECEIPT, state)


def _finalize(context: _EvaluationContext, state: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Only verified disk artifacts can be used to generate final metrics and publish COMPLETED."""
    finalization_started = time.perf_counter()
    from road_damage.evaluation import internal_test_metrics as metrics
    config, gt = _require_context(context)
    identity = json.loads(context.identity_json)
    identity_sha = hashlib.sha256(context.identity_json).hexdigest()
    standard_rows, standard_sha = metrics.verify_cache(context.output / "standard_predictions.jsonl", "standard", gt, identity_sha)
    frozen_rows, frozen_sha = metrics.verify_cache(context.output / "frozen_predictions.jsonl", "frozen", gt, identity_sha)
    standard = metrics.replay_standard(standard_rows, config)
    live = json.loads((context.output / "standard_live_metrics.json").read_bytes())
    if shared.json_bytes(standard["native_metric_values"]) != shared.json_bytes(live):
        raise ThresholdError("Persisted standard cache does not reproduce the actual validator metrics.")
    predictions = {r["image_id"]: [Box(b["class_id"], tuple(b["xyxy"]), b["confidence"])
                                  for b in r["predictions"]] for r in frozen_rows}
    point, countries = operating_point(gt, predictions), country_metrics(gt, predictions)
    reports = context.output / "reports"
    reports.mkdir(exist_ok=False)
    metrics.write_results(reports, gt, point, countries, standard, config)
    metrics.frozen_confusion(gt, predictions, reports)
    # Verify all inputs and scientific identities once more; this is inside authorized evaluation only.
    if inventory_internal_test(context.inventory_capability, context.root, config) != gt:
        raise ThresholdError("Canonical test input changed during evaluation.")
    _verify_snapshot(context.output, gt)
    _, _, after_identity = _prepare_metadata(context.root)
    if shared.json_bytes(after_identity) != context.identity_json:
        raise ThresholdError("Evaluation identity changed during execution.")
    if (shared.digest(context.output / "ground_truth_cache.json") != context.ground_truth_sha256
            or shared.digest(context.output / "standard_predictions.jsonl") != standard_sha
            or shared.digest(context.output / "frozen_predictions.jsonl") != frozen_sha):
        raise ThresholdError("Verified cache changed during report generation.")
    required = ["internal_test_metrics.json", "internal_test_operating_point.json", "internal_test_summary.md",
                "confusion_matrix.png", "per_class_metrics.csv", "country_metrics.csv"]
    if any(not (reports / name).is_file() or not (reports / name).stat().st_size for name in required):
        raise ThresholdError("Required final report is absent or empty.")
    # Ignore large disposable input copies; original/copy identities are in ground_truth_cache.
    artifacts = {p.relative_to(context.output).as_posix(): _durable_artifact_hash(p)
                 for p in sorted(context.output.rglob("*")) if p.is_file()
                 and "inputs" not in p.relative_to(context.output).parts
                 and p.name not in ("evaluation_manifest.json", "evaluation_manifest.json.tmp")}
    manifest.update({"status": "COMPLETED", "incomplete": False, "artifacts": artifacts,
        "cache_sha256": {"standard_predictions.jsonl": standard_sha, "frozen_predictions.jsonl": frozen_sha},
        "standard_metrics_reproduced_from_persisted_cache": True,
        "branch_images": {"standard": len(standard_rows), "frozen": len(frozen_rows)},
        "offline_verification_and_report_seconds": time.perf_counter() - finalization_started,
        "total_evaluation_seconds": manifest["pre_finalization_elapsed_seconds"] + time.perf_counter() - finalization_started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    _atomic_json(context.output / "evaluation_manifest.json", manifest)
    # Either completed attempt manifest or final marker is sufficient to permanently lock.
    _write_once(context.output / "COMPLETED.json", {"status": "COMPLETED", "identity_sha256": identity_sha,
                                                  "manifest_sha256": shared.digest(context.output / "evaluation_manifest.json")})
    _atomic_json(context.root / OUTPUT / "evaluation_manifest.json", {
        **manifest, "completed_attempt": context.output.relative_to(context.root).as_posix(),
        "attempt_manifest_sha256": shared.digest(context.output / "evaluation_manifest.json")})
    state["status"] = "COMPLETED"
    state["attempts"][-1]["status"] = "COMPLETED"
    _atomic_json(context.root / RECEIPT, state)


def run_evaluation(root: Path = ROOT, *, acknowledged: bool = False, recover: bool = False) -> None:
    """The only supported public inference API; no model/config/threshold/output overrides."""
    global _ACTIVE_CONTEXT, _ACTIVE_INVENTORY_CAPABILITY
    if not acknowledged:
        raise ThresholdError("Explicit first/final frozen evaluation acknowledgement is required.")
    if Path(root).absolute() != ROOT.absolute():
        raise ThresholdError("Evaluation root override is forbidden; the project ledger cannot be bypassed.")
    if _ACTIVE_CONTEXT is not None or _ACTIVE_INVENTORY_CAPABILITY is not None:
        raise ThresholdError("An evaluation is already active in this process.")
    started = time.perf_counter()
    # This is NOT the public preflight command. Both share metadata-only integrity gates.
    config, _, identity = _prepare_metadata(root)
    with _process_lock(root):
        output, state = _start_attempt(root, identity, recover)
        manifest = {"schema_version": config["schema_version"], "status": "STARTED", "incomplete": True,
                    "identity": identity, "identity_sha256": state["identity_sha256"],
                    "scientific_evaluation": "one frozen configuration; two fixed branches",
                    "threshold_tuning_performed": False, "model_selection_performed": False,
                    "teacher_video_used": False, "attempt_number": state["attempts"][-1]["number"]}
        _atomic_json(output / "evaluation_manifest.json", manifest)
        identity_json, config_json = shared.json_bytes(identity), shared.json_bytes(config)
        inventory_capability = object()
        # Minted only here, after _start_attempt returned with durable receipt and
        # STARTED marker. No public constructor/factory or boolean bypass exists.
        _ACTIVE_INVENTORY_CAPABILITY = (
            inventory_capability, Path(root).absolute(), output, identity_json, config_json)
        try:
            gt = inventory_internal_test(inventory_capability, root, config)
            _write_once(output / "ground_truth_cache.json", gt)
            _snapshot_inputs(root, output, gt)
            context = _EvaluationContext(root, output, identity_json, config_json,
                                         shared.digest(output / "ground_truth_cache.json"), inventory_capability)
            _ACTIVE_CONTEXT = context
            _set_determinism(config)
            for label, function in (("standard", _run_standard), ("frozen", _run_frozen)):
                branch_started = time.perf_counter()
                _require_context(context)
                function(context)
                manifest[label + "_branch_seconds"] = time.perf_counter() - branch_started
                _atomic_json(output / "evaluation_manifest.json", manifest)
            manifest["pre_finalization_elapsed_seconds"] = time.perf_counter() - started
            _finalize(context, state, manifest)
        except BaseException as exc:
            # Never downgrade a successfully published completion if a later ledger write failed.
            completed_path = output / "evaluation_manifest.json"
            completed = completed_path.exists() and json.loads(completed_path.read_bytes()).get("status") == "COMPLETED"
            if not completed:
                _failure(root, output, state, manifest, exc)
            raise
        finally:
            _ACTIVE_CONTEXT = None
            _ACTIVE_INVENTORY_CAPABILITY = None
        LOGGER.info("COMPLETED frozen baseline evaluation: %s (%.2fs total)", output, time.perf_counter() - started)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Metadata-only checks; NO test image/label access or inference.")
    mode.add_argument("--evaluate-once", action="store_true", help="Run two fixed branches under one immutable identity.")
    parser.add_argument("--acknowledge-first-final-test", action="store_true")
    parser.add_argument("--recover-technical", action="store_true", help="Explicit recovery of an identical incomplete evaluation, never a completed test.")
    args = parser.parse_args(argv)
    if args.preflight and (args.acknowledge_first_final_test or args.recover_technical):
        parser.error("Evaluation acknowledgement/recovery flags cannot be used with preflight.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.preflight:
            sys.stdout.write(shared.json_bytes(preflight()).decode("utf-8"))
        else:
            run_evaluation(acknowledged=args.acknowledge_first_final_test, recover=args.recover_technical)
    except KeyboardInterrupt:
        LOGGER.warning("Technical interruption recorded. Recovery requires unchanged identity and an explicit command.")
        return 130
    except (ThresholdError, shared.MentorDemoError, OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        LOGGER.error("Internal-test evaluation stopped: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
