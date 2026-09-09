"""User-run validation-only threshold selection and inference-free cache replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_damage.demo.mentor_demo import (  # noqa: E402
    CLASS_NAMES, FROZEN_MODEL_RELATIVE_PATH, FROZEN_MODEL_SHA256,
    FROZEN_MODEL_SIZE_BYTES, MentorDemoError, verify_model_identity,
)
from road_damage.evaluation.threshold_metrics import (  # noqa: E402
    Box, ThresholdError, select_operating_point, sweep_cache,
)


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path("configs/evaluation/baseline_public_v1_threshold_selection.yaml")
DATA_YAML = Path("configs/training/baseline_public_v1_train_val.yaml")
EXPORT = Path("data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1")
OUTPUT = Path("outputs/evaluation/baseline_public_v1_threshold_selection")
MENTOR_COMMIT = "702a916e88af54504feb1f25b5f8af8ffef8621e"
TRAINING_PROVENANCE = {
    "original_training_commit": "80916642e45fe00bcc9a6054dea2cce3ea55e6a7",
    "resume_hotfix_commit": "451f785b2653ff5a6e0ef6653860b57dd856e65c",
    "original_source_fingerprint": "1e94e69c6a7e4b1532d764d41a05118c2cd78e6a3aa40aab2bb1829e512858da",
    "training_hotfix_source_fingerprint": "dc631672a1cc4171c6ec6876cdc629157c5e006b599926dd0339dde6660bf43b",
    "mentor_demo_commit": MENTOR_COMMIT,
}
EXPECTED_CONFIG = {
    "schema_version": "baseline_public_v1.threshold_selection.v1",
    "model": FROZEN_MODEL_RELATIVE_PATH.as_posix(), "model_size_bytes": FROZEN_MODEL_SIZE_BYTES,
    "model_sha256": FROZEN_MODEL_SHA256, "validation_yaml": DATA_YAML.as_posix(),
    "expected_counts": {"images": 2602, "targets": 3377, "negative_images": 983},
    "classes": {str(k): v for k, v in CLASS_NAMES.items()}, "imgsz": 640, "device": 0,
    "inference_confidence_floor": 0.01,
    "confidence_grid": {"start_percent": 1, "stop_percent": 80, "step_percent": 1},
    "nms_iou_candidates": [0.5, 0.6, 0.7], "matching_iou": 0.5,
    "global_threshold_only": True, "augment": False, "half": False, "max_det": 300,
    "seed": 42, "progress_every": 100, "output": OUTPUT.as_posix(),
    "recompute_output": OUTPUT.as_posix() + "_recomputed",
}


def json_bytes(value: Any) -> bytes:
    """Stable serialization used for hashes and replay-identical selection artifacts."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked_path(candidate: Path, expected: Path, root: Path) -> Path:
    """Check the lexical allowlist before touching a path; reject redirected ancestors."""
    actual = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
    required = Path(os.path.abspath(root / expected))
    if actual != required:
        raise ThresholdError(f"Path outside the approved allowlist: {candidate}; required: {expected}")
    for part in reversed([actual, *actual.parents]):
        if part == root or root not in part.parents:
            continue
        if part.is_symlink() or part.is_junction():
            raise ThresholdError(f"Redirected path is forbidden: {part}")
    return actual


def load_config(root: Path = ROOT) -> dict[str, Any]:
    config_path = checked_path(CONFIG, CONFIG, root)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if json_bytes(value) != json_bytes(EXPECTED_CONFIG):
        raise ThresholdError("Frozen threshold config differs: unknown keys/per-class thresholds are forbidden.")
    return value


def validation_paths(root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    """Parse only the training/validation YAML; never call framework dataset discovery."""
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader: Any, node: Any) -> dict[Any, Any]:
        pairs = loader.construct_pairs(node, deep=True)
        result: dict[Any, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ThresholdError(f"Duplicate YAML key: {key}")
            result[key] = value
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    path = checked_path(Path(config["validation_yaml"]), DATA_YAML, root)
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    if not isinstance(data, dict) or set(data) != {"train", "val", "names"}:
        raise ThresholdError("Validation YAML must contain only train, val, names; no test key.")
    if data["names"] != CLASS_NAMES or any(type(key) is not int for key in data["names"]):
        raise ThresholdError("Validation YAML must retain the exact four-class mapping.")
    # Compare train as text only: do not stat, resolve, enumerate, or read that directory.
    expected_train = os.path.normcase(os.path.abspath(root / EXPORT / "images/train"))
    if not isinstance(data["train"], str) or os.path.normcase(os.path.abspath(data["train"])) != expected_train:
        raise ThresholdError("Unexpected training path in approved train/val YAML.")
    images = checked_path(Path(data["val"]), EXPORT / "images/val", root)
    labels = checked_path(EXPORT / "labels/val", EXPORT / "labels/val", root)
    return images, labels


def box_record(box: Box) -> dict[str, Any]:
    return {"class_id": box.class_id, "confidence": box.confidence, "xyxy": list(box.xyxy)}


def verify_counts(images: Sequence[dict[str, Any]], expected: dict[str, int]) -> dict[str, int]:
    actual = {"images": len(images), "targets": sum(len(r["ground_truth"]) for r in images),
              "negative_images": sum(not r["ground_truth"] for r in images)}
    if actual != expected:
        raise ThresholdError(f"Validation count mismatch: expected {expected}, observed {actual}")
    return actual


def inventory_validation(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Hash and read only approved validation pairs, retaining GT for offline replay."""
    from PIL import Image

    images, labels = validation_paths(root, config)
    image_files = sorted(images.iterdir(), key=lambda p: p.name)
    label_files = sorted(labels.iterdir(), key=lambda p: p.name)
    if any(not re.fullmatch(r"(?:India|Japan)_\d+\.jpg", p.name) for p in image_files):
        raise ThresholdError("Unexpected filename in validation image directory.")
    if {p.stem for p in image_files} != {p.stem for p in label_files} or any(
        p.suffix != ".txt" for p in label_files
    ):
        raise ThresholdError("Validation image/label pairing or orphan check failed.")
    records = []
    for image_path in image_files:
        image_path = checked_path(image_path, EXPORT / "images/val" / image_path.name, root)
        label_path = checked_path(labels / (image_path.stem + ".txt"),
                                  EXPORT / "labels/val" / (image_path.stem + ".txt"), root)
        with Image.open(image_path) as image:
            width, height = image.size
        label_bytes = label_path.read_bytes()
        gt = []
        if label_bytes:
            for line in label_bytes.decode("utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] not in {"0", "1", "2", "3"}:
                    raise ThresholdError(f"Malformed label: {label_path}")
                cx, cy, w, h = map(float, fields[1:])
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ThresholdError(f"Invalid normalized box: {label_path}")
                coords = ((cx - w / 2) * width, (cy - h / 2) * height,
                          (cx + w / 2) * width, (cy + h / 2) * height)
                # Export has eight decimal places; tolerate only edge rounding error.
                if coords[0] < -1e-4 or coords[1] < -1e-4 or coords[2] > width + 1e-4 or coords[3] > height + 1e-4:
                    raise ThresholdError(f"Out-of-image box: {label_path}")
                bounded = (max(0., coords[0]), max(0., coords[1]), min(width, coords[2]), min(height, coords[3]))
                gt.append(box_record(Box(int(fields[0]), bounded)))
            if not gt:
                raise ThresholdError(f"Negative label must be exactly empty: {label_path}")
        records.append({"image_id": image_path.name, "width": width, "height": height,
                        "image_path": image_path.relative_to(root).as_posix(),
                        "label_path": label_path.relative_to(root).as_posix(),
                        "image_sha256": digest(image_path),
                        "label_sha256": hashlib.sha256(label_bytes).hexdigest(), "ground_truth": gt})
    counts = verify_counts(records, config["expected_counts"])
    return {"schema_version": "threshold_ground_truth.v1", "validation_only": True,
            "count_source": "direct approved validation image/label tally",
            "counts": counts, "images": records}


def git_provenance(root: Path) -> dict[str, Any]:
    """Require a clean post-demo commit and preserve all earlier history identities."""
    def git(*args: str) -> str:
        result = subprocess.run(["git", "--no-optional-locks", *args], cwd=root,
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise ThresholdError(f"Git provenance check failed: {' '.join(args)}: {result.stderr.strip()}")
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    git("merge-base", "--is-ancestor", MENTOR_COMMIT, "HEAD")
    if status or head == MENTOR_COMMIT:
        raise ThresholdError("Review and commit the new evaluation tooling before preflight/sweep; clean post-demo commit required.")
    for commit_key, fingerprint_key in (("original_training_commit", "original_source_fingerprint"),
                                        ("resume_hotfix_commit", "training_hotfix_source_fingerprint")):
        commit = TRAINING_PROVENANCE[commit_key]
        git("merge-base", "--is-ancestor", commit, "HEAD")
        historic = json.loads(git("show", f"{commit}:reproducibility/baseline_public_v1_source_state_manifest.json"))
        if historic["source_tree_sha256"] != TRAINING_PROVENANCE[fingerprint_key]:
            raise ThresholdError("Historical training source fingerprint mismatch.")
    files = [CONFIG, Path(__file__).relative_to(ROOT),
             Path("src/road_damage/evaluation/threshold_metrics.py")]
    for path in files:
        git("ls-files", "--error-unmatch", path.as_posix())
    tracked = git("ls-files", "src", "configs", "tests", "road-damage-project-docs", "reproducibility",
                  "AGENTS.md", "requirements.txt", ".gitignore").splitlines()
    hashes = [{"path": name, "sha256": digest(root / name)} for name in sorted(tracked)]
    return {"evaluation_commit": head, "worktree_clean": True,
            "post_training_source_fingerprint": hashlib.sha256(json_bytes(hashes)).hexdigest(),
            "source_files": hashes, "training_history": TRAINING_PROVENANCE}


def environment() -> dict[str, Any]:
    """Inspect package and CUDA state without loading a model or running inference."""
    import torch

    versions = {name: importlib.metadata.version(name)
                for name in ("ultralytics", "torch", "torchvision", "numpy", "Pillow", "PyYAML", "matplotlib")}
    if platform.python_version() != "3.12.10" or versions["ultralytics"] != "8.4.130":
        raise ThresholdError("Use the existing approved Python 3.12.10 / Ultralytics 8.4.130 environment.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ThresholdError("CUDA device 0 is unavailable; GPU execution is required.")
    return {"python": platform.python_version(), "packages": versions,
            "cuda": torch.version.cuda, "device": 0, "gpu": torch.cuda.get_device_name(0)}


def ensure_output_available(output: Path) -> None:
    """Refuse overwrite and probe write permission without leaving output artifacts."""
    if output.exists():
        raise ThresholdError(f"Refusing to overwrite existing output: {output}")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ThresholdError(f"Output ancestor is not a directory: {ancestor}")
    with tempfile.TemporaryFile(dir=ancestor) as probe:
        probe.write(b"threshold-output-preflight")


def preflight(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Complete input/environment/provenance verification, with no YOLO instantiation."""
    config = load_config(root)
    output = checked_path(Path(config["output"]), OUTPUT, root)
    ensure_output_available(output)
    model = checked_path(Path(config["model"]), FROZEN_MODEL_RELATIVE_PATH, root)
    verify_model_identity(model, expected_sha256=config["model_sha256"],
                          expected_size_bytes=config["model_size_bytes"])
    gt = inventory_validation(root, config)
    env = environment()
    provenance = git_provenance(root)
    report = {"passed": True, "inference_executed": False, "validation_only": True,
              "internal_test_untouched": True, "model_path": config["model"],
              "model_sha256": config["model_sha256"], "model_size_bytes": config["model_size_bytes"],
              "counts": gt["counts"], "class_mapping": config["classes"],
              "validation_yaml_sha256": digest(root / DATA_YAML),
              "evaluation_config_sha256": digest(root / CONFIG),
              "ground_truth_snapshot_sha256": hashlib.sha256(json_bytes(gt)).hexdigest(),
              "output_available": str(output), "environment": env, "provenance": provenance}
    return config, gt, report


def prediction_kwargs(config: dict[str, Any], nms_iou: float) -> dict[str, Any]:
    """Explicit predict settings; no dataset YAML is passed to Ultralytics."""
    return {"imgsz": config["imgsz"], "conf": config["inference_confidence_floor"],
            "iou": nms_iou, "device": config["device"], "augment": False, "half": False,
            "max_det": config["max_det"], "agnostic_nms": False, "classes": None,
            "save": False, "save_txt": False, "save_conf": False, "save_crop": False,
            "show": False, "verbose": False, "stream": False, "rect": False,
            "batch": 1, "compile": False}


def collect_predictions(model: Any, gt: dict[str, Any], config: dict[str, Any], nms_iou: float,
                        candidate_index: int, root: Path) -> dict[str, Any]:
    """One predict call per validation image for this NMS value; immutable sources."""
    import cv2

    started = time.perf_counter()
    records = []
    total = len(gt["images"])
    LOGGER.info("NMS candidate %d/3: %.2f", candidate_index, nms_iou)
    for index, record in enumerate(gt["images"], start=1):
        path = checked_path(Path(record["image_path"]), EXPORT / "images/val" / record["image_id"], root)
        if digest(path) != record["image_sha256"]:
            raise ThresholdError(f"Validation image changed since preflight: {record['image_id']}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or tuple(image.shape[:2]) != (record["height"], record["width"]):
            raise ThresholdError(f"Image decode or size mismatch: {path}")
        predictions = model.predict(source=image, **prediction_kwargs(config, nms_iou))
        if len(predictions) != 1:
            raise ThresholdError("Expected exactly one prediction result per validation image.")
        result = predictions[0]
        boxes = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist(), strict=True):
                if int(class_id) != class_id:
                    raise ThresholdError("Non-integral model class ID.")
                box = Box(int(class_id), tuple(xyxy), float(confidence))
                if box.confidence < config["inference_confidence_floor"]:
                    raise ThresholdError("Model returned a prediction below the declared floor.")
                boxes.append(box_record(box))
        if len(boxes) >= config["max_det"]:
            raise ThresholdError("Prediction cap reached; do not select thresholds from truncated candidates.")
        records.append({"image_id": record["image_id"], "nms_iou": nms_iou,
                        "predictions": sorted(boxes, key=lambda b: (-b["confidence"], b["class_id"], b["xyxy"]))})
        if index % config["progress_every"] == 0 or index == total:
            elapsed = time.perf_counter() - started
            LOGGER.info("NMS %d/3: validation images processed %d/%d; elapsed %.1fs; ETA %.1fs",
                        candidate_index, index, total, elapsed, elapsed / index * (total - index))
    return {"schema_version": "threshold_predictions.v1", "validation_only": True,
            "nms_iou": nms_iou, "confidence_floor": config["inference_confidence_floor"],
            "inference_duration_seconds": time.perf_counter() - started, "images": records}


def decoded_ground_truth(gt: dict[str, Any]) -> dict[str, list[Box]]:
    if gt.get("validation_only") is not True:
        raise ThresholdError("Ground truth cache is not validation-only.")
    result = {}
    for row in gt["images"]:
        name = row["image_id"]
        if name in result or not re.fullmatch(r"(?:India|Japan)_\d+\.jpg", name):
            raise ThresholdError("Invalid or duplicated cache image identifier.")
        if row["image_path"] != (EXPORT / "images/val" / name).as_posix():
            raise ThresholdError("Ground truth cache references a forbidden image source.")
        if row["label_path"] != (EXPORT / "labels/val" / (Path(name).stem + ".txt")).as_posix():
            raise ThresholdError("Ground truth cache references a forbidden label source.")
        result[name] = [Box(b["class_id"], tuple(b["xyxy"])) for b in row["ground_truth"]]
    return result


def decoded_predictions(cache: dict[str, Any], nms_iou: float) -> dict[str, list[Box]]:
    if cache.get("validation_only") is not True or cache.get("nms_iou") != nms_iou or cache.get("confidence_floor") != 0.01:
        raise ThresholdError("Prediction cache policy mismatch.")
    result = {}
    for row in cache["images"]:
        if row["image_id"] in result or row["nms_iou"] != nms_iou:
            raise ThresholdError("Duplicate image or NMS mismatch in cache.")
        boxes = [Box(b["class_id"], tuple(b["xyxy"]), b["confidence"]) for b in row["predictions"]]
        if any(b.confidence < 0.01 for b in boxes):
            raise ThresholdError("Cache prediction below declared floor.")
        result[row["image_id"]] = boxes
    return result


def calculate_artifacts(gt: dict[str, Any], caches: Sequence[dict[str, Any]],
                        config: dict[str, Any], identity: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inference-free replay; same cache and identity yield byte-identical selected JSON."""
    truth = decoded_ground_truth(gt)
    verify_counts(gt["images"], config["expected_counts"])
    nms_values = config["nms_iou_candidates"]
    if len(caches) != len(nms_values):
        raise ThresholdError("Require one prediction cache for each NMS candidate.")
    rows = []
    grid = config["confidence_grid"]
    confidences = [p / 100 for p in range(grid["start_percent"], grid["stop_percent"] + 1, grid["step_percent"])]
    for nms_iou, cache in zip(nms_values, caches, strict=True):
        rows.extend(sweep_cache(truth, decoded_predictions(cache, nms_iou), nms_iou,
                                confidences, config["matching_iou"]))
    selection = select_operating_point(rows, nms_values)
    point = selection["metrics"]
    selected = {**identity, "matching_iou": config["matching_iou"],
                "selected_nms_iou": point["nms_iou"], "selected_global_confidence": point["confidence"],
                "macro_f1": point["macro_f1"], "per_class_metrics": {
                    code: {metric: point[f"{code}_{metric}"] for metric in ("tp", "fp", "fn", "precision", "recall", "f1")}
                    for code in ("D00", "D10", "D20", "D40")},
                "negative_image_metrics": {key: value for key, value in point.items() if key.startswith("negative_")},
                **selection}
    return rows, selected


def write_results(output: Path, rows: list[dict[str, Any]], selected: dict[str, Any], config: dict[str, Any]) -> None:
    """Render only measured results supplied by the real sweep or verified cache replay."""
    (output / "selected_operating_point.json").write_bytes(json_bytes(selected))
    with (output / "threshold_sweep.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Validation operating-threshold selection", "",
             f"Selected global confidence: {selected['selected_global_confidence']:.2f}",
             f"Selected NMS IoU: {selected['selected_nms_iou']:.2f}",
             f"Macro-F1: {selected['macro_f1']:.8f}", "",
             "| Class | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    for code, metrics in selected["per_class_metrics"].items():
        lines.append(f"| {code} | {metrics['precision']:.8f} | {metrics['recall']:.8f} | {metrics['f1']:.8f} |")
    negative = selected["negative_image_metrics"]
    lines.extend(["", f"Negative FP/image: {negative['negative_fp_per_image']:.8f}",
                  f"Negative images with >=1 FP: {100 * negative['negative_images_with_fp_fraction']:.4f}%",
                  "", "| NMS IoU | Best confidence | Macro-F1 | Negative FP |",
                  "|---:|---:|---:|---:|"])
    for nms in config["nms_iou_candidates"]:
        best = select_operating_point([r for r in rows if r["nms_iou"] == nms], [nms])["metrics"]
        lines.append(f"| {nms:.2f} | {best['confidence']:.2f} | {best['macro_f1']:.8f} | {best['negative_fp']} |")
    lines.extend(["", "Exact tie-break reasoning (survivors after each criterion):", "", "```json",
                  json.dumps(selected["tie_break_reasoning"], indent=2), "```", "",
                  f"Model SHA-256: {selected['model_sha256']}", "",
                  "Validation-only: all selection uses the approved 2,602 validation images.",
                  "Internal test remains untouched. Teacher video was not used.",
                  "No mAP objective, per-class thresholds, or application FPS claim."])
    (output / "threshold_selection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for nms in config["nms_iou_candidates"]:
        curve = [r for r in rows if r["nms_iou"] == nms]
        ax.plot([r["confidence"] for r in curve], [r["macro_f1"] for r in curve], label=f"NMS IoU {nms:.2f}")
    ax.set(xlabel="Global confidence threshold", ylabel="Macro-F1 (four classes)", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "macro_f1_vs_confidence.png", dpi=150)
    plt.close(fig)


def run_sweep(root: Path = ROOT) -> None:
    """The user invokes this explicitly after review, commit, and successful preflight."""
    started = time.perf_counter()
    config, gt, report = preflight(root)
    output = checked_path(Path(config["output"]), OUTPUT, root)
    output.mkdir(parents=True, exist_ok=False)
    identity = {"schema_version": config["schema_version"], "model_path": config["model"],
                "model_sha256": config["model_sha256"], "validation_only": True,
                "internal_test_untouched": True, "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_config_identity": {k: report[k] for k in (
                    "validation_yaml_sha256", "evaluation_config_sha256", "ground_truth_snapshot_sha256")},
                "git_commit": report["provenance"]["evaluation_commit"],
                "environment": report["environment"], "training_provenance": TRAINING_PROVENANCE}
    manifest = {"status": "running", "identity": identity, "config": config,
                "preflight": report, "artifacts": {}, "prediction_cache_sha256": {},
                "matching_ties": "prediction confidence descending, xyxy ascending; GT IoU descending, xyxy ascending",
                "source_scope": "approved validation only; no framework dataset discovery"}
    manifest_path = output / "evaluation_manifest.json"
    def checkpoint_manifest() -> None:
        temporary = output / "evaluation_manifest.json.tmp"
        temporary.write_bytes(json_bytes(manifest))
        os.replace(temporary, manifest_path)
    checkpoint_manifest()
    try:
        gt_path = output / "ground_truth_cache.json"
        gt_path.write_bytes(json_bytes(gt))
        manifest["artifacts"][gt_path.name] = digest(gt_path)
        checkpoint_manifest()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from ultralytics import YOLO
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        verify_model_identity(root / config["model"], expected_sha256=config["model_sha256"],
                              expected_size_bytes=config["model_size_bytes"])
        model = YOLO(str(root / config["model"]), task="detect")
        if model.names != CLASS_NAMES:
            raise ThresholdError("Frozen checkpoint class names mismatch.")
        caches = []
        durations = {}
        for index, nms in enumerate(config["nms_iou_candidates"], start=1):
            cache = collect_predictions(model, gt, config, nms, index, root)
            path = output / f"predictions_nms_{int(round(nms * 100)):02d}.json"
            path.write_bytes(json_bytes(cache))
            manifest["prediction_cache_sha256"][path.name] = digest(path)
            manifest["artifacts"][path.name] = digest(path)
            durations[f"{nms:.2f}"] = cache["inference_duration_seconds"]
            checkpoint_manifest()
            caches.append(cache)
        # Recheck all input hashes and the commit before accepting any metrics.
        if inventory_validation(root, config) != gt or git_provenance(root) != report["provenance"]:
            raise ThresholdError("Validation input or source provenance changed during evaluation.")
        verify_model_identity(root / config["model"], expected_sha256=config["model_sha256"],
                              expected_size_bytes=config["model_size_bytes"])
        offline_started = time.perf_counter()
        rows, selected = calculate_artifacts(gt, caches, config, identity)
        offline_duration = time.perf_counter() - offline_started
        write_results(output, rows, selected, config)
        for name in ("threshold_sweep.csv", "selected_operating_point.json", "threshold_selection_summary.md", "macro_f1_vs_confidence.png"):
            manifest["artifacts"][name] = digest(output / name)
        manifest.update({"status": "complete", "inference_duration_per_nms_seconds": durations,
                         "offline_sweep_duration_seconds": offline_duration,
                         "total_evaluation_duration_seconds": time.perf_counter() - started})
        checkpoint_manifest()
        LOGGER.info("Inference collection seconds per NMS (includes decode/hash): %s", durations)
        LOGGER.info("Offline sweep %.2fs; total evaluation %.2fs", offline_duration,
                    manifest["total_evaluation_duration_seconds"])
        LOGGER.info("Selected confidence %.2f, NMS %.2f, macro-F1 %.8f; results: %s",
                    selected["selected_global_confidence"], selected["selected_nms_iou"], selected["macro_f1"], output)
    except BaseException as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        checkpoint_manifest()
        raise


def recompute(root: Path = ROOT) -> None:
    """Verify caches and reproduce selection without importing YOLO or opening datasets."""
    config = load_config(root)
    source = checked_path(OUTPUT, OUTPUT, root)
    manifest_path = checked_path(source / "evaluation_manifest.json", OUTPUT / "evaluation_manifest.json", root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("config") != config:
        raise ThresholdError("Replay requires a complete matching evaluation manifest.")
    if manifest["identity"]["model_sha256"] != FROZEN_MODEL_SHA256 or manifest["identity"]["validation_only"] is not True:
        raise ThresholdError("Replay model/data identity mismatch.")
    names = ["ground_truth_cache.json"] + [f"predictions_nms_{int(round(n * 100)):02d}.json" for n in config["nms_iou_candidates"]]
    values = []
    for name in names:
        path = checked_path(source / name, OUTPUT / name, root)
        if digest(path) != manifest["artifacts"].get(name):
            raise ThresholdError(f"Replay artifact SHA-256 mismatch: {name}")
        if name.startswith("predictions_") and digest(path) != manifest["prediction_cache_sha256"].get(name):
            raise ThresholdError(f"Replay prediction SHA-256 mismatch: {name}")
        values.append(json.loads(path.read_text(encoding="utf-8")))
    rows, selected = calculate_artifacts(values[0], values[1:], config, manifest["identity"])
    selected_bytes = json_bytes(selected)
    if hashlib.sha256(selected_bytes).hexdigest() != manifest["artifacts"].get("selected_operating_point.json"):
        raise ThresholdError("Recomputed selection differs from the original selected artifact.")
    output_relative = Path(config["recompute_output"])
    output = checked_path(output_relative, Path(str(OUTPUT) + "_recomputed"), root)
    ensure_output_available(output)
    output.mkdir(parents=True, exist_ok=False)
    write_results(output, rows, selected, config)
    LOGGER.info("Cache replay matched selected_operating_point.json byte-for-byte: %s", output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Check inputs/CUDA/provenance; no inference or results.")
    mode.add_argument("--recompute-cache", action="store_true", help="Replay completed caches offline; no model/data access.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.preflight:
            _, _, report = preflight()
            print(json_bytes(report).decode("utf-8"))
        elif args.recompute_cache:
            recompute()
        else:
            run_sweep()
    except (ThresholdError, MentorDemoError, OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        LOGGER.error("Threshold selection stopped: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Evaluation interrupted; partial results must not be used for selection.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
