"""Run the frozen YOLOv8s baseline on images for the mentor presentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo" / "mentor_demo.yaml"

FROZEN_MODEL_RELATIVE_PATH = Path(
    "outputs/training/baseline_public_v1/"
    "20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42/weights/best.pt"
)
FROZEN_MODEL_SHA256 = (
    "BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7"
)
FROZEN_MODEL_SIZE_BYTES = 22_524_074
FROZEN_CONFIDENCE = 0.19
FROZEN_NMS_IOU = 0.50
FROZEN_OPERATING_POINT_STATUS = "frozen_validation_selected"
CLASS_NAMES: dict[int, str] = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VALIDATION_IMAGES_RELATIVE_PATH = Path(
    "data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/images/val"
)
VALIDATION_LABELS_RELATIVE_PATH = Path(
    "data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/labels/val"
)
INTERNAL_TEST_IMAGES_RELATIVE_PATH = Path(
    "data/exports/rdd2022_india_japan_v1_1/yolo_detection_v1/images/test"
)
DEMO_OUTPUT_RELATIVE_PATH = Path("outputs/demo/mentor_live")
SAMPLE_OUTPUT_RELATIVE_PATH = Path("outputs/demo/mentor_samples")
FROZEN_OPERATING_POINT_NOTICE = (
    "Operating-point status: frozen_validation_selected. Final frozen "
    "validation-selected YOLOv8s baseline operating point: "
    "confidence 0.19, NMS IoU 0.50. The completed internal test did not alter "
    "these values. Demo execution is application inference only; its outputs "
    "are not new model evaluation results."
)
IMAGE_WINDOW_NAME = (
    "AI Road Damage Detection — frozen_validation_selected 0.19/0.50 — Esc/q to stop"
)


class MentorDemoError(RuntimeError):
    """Raised when a demo safety or input check fails."""


@dataclass(frozen=True)
class DemoConfig:
    """Strict, presentation-specific inference settings."""

    model_path: Path
    model_sha256: str
    model_size_bytes: int
    imgsz: int
    confidence: float
    nms_iou: float
    max_detections: int
    device: int
    class_names: Mapping[int, str]
    operating_point_status: str
    output_path: Path
    validation_images: Path
    validation_labels: Path
    sample_output_path: Path
    images_per_class: int
    negative_images: int
    sample_seed: int


@dataclass(frozen=True)
class GroundTruthSample:
    """One validation image and the class information in its YOLO label."""

    filename: str
    image_path: Path
    label_path: Path
    classes_present: tuple[int, ...]
    class_object_counts: tuple[tuple[int, int], ...]
    class_max_normalized_areas: tuple[tuple[int, float], ...]

    @property
    def is_negative(self) -> bool:
        return not self.classes_present


@dataclass(frozen=True)
class SelectedSample:
    """A validation sample selected for one presentation bucket."""

    bucket: str
    class_id: int | None
    sample: GroundTruthSample


@dataclass(frozen=True)
class Detection:
    """One rendered model detection in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy_pixel: tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageDemoResult:
    """Terminal and output details for one processed image."""

    source_path: Path
    output_path: Path
    width: int
    height: int
    inference_time_ms: float
    detections: tuple[Detection, ...]


def _resolve_project_path(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _project_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    """Hash a file without loading the complete file into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MentorDemoError(f"Could not hash file: {path}") from exc
    return digest.hexdigest().upper()


def load_demo_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
) -> DemoConfig:
    """Load the dependency-free config and enforce every frozen demo value."""
    resolved_config = _resolve_project_path(config_path, project_root)
    try:
        raw = json.loads(resolved_config.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MentorDemoError(f"Mentor demo config does not exist: {resolved_config}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MentorDemoError(f"Could not read mentor demo config: {resolved_config}") from exc
    try:
        model = raw["model"]
        inference = raw["inference"]
        samples = raw["mentor_samples"]
        class_names = {int(key): str(value) for key, value in raw["classes"].items()}
        config = DemoConfig(
            model_path=Path(str(model["path"])),
            model_sha256=str(model["sha256"]).upper(),
            model_size_bytes=int(model["size_bytes"]),
            imgsz=int(inference["imgsz"]),
            confidence=float(inference["confidence"]),
            nms_iou=float(inference["nms_iou"]),
            max_detections=int(inference["max_detections"]),
            device=int(inference["device"]),
            class_names=class_names,
            operating_point_status=str(raw["operating_point_status"]),
            output_path=Path(str(raw["output_path"])),
            validation_images=Path(str(samples["validation_images"])),
            validation_labels=Path(str(samples["validation_labels"])),
            sample_output_path=Path(str(samples["output_path"])),
            images_per_class=int(samples["images_per_class"]),
            negative_images=int(samples["negative_images"]),
            sample_seed=int(samples["seed"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise MentorDemoError("Mentor demo config is missing or has invalid fields.") from exc

    expected = {
        "schema_version": "mentor_demo.v1",
        "model_path": FROZEN_MODEL_RELATIVE_PATH.as_posix(),
        "model_sha256": FROZEN_MODEL_SHA256,
        "model_size_bytes": FROZEN_MODEL_SIZE_BYTES,
        "imgsz": 640,
        "confidence": FROZEN_CONFIDENCE,
        "nms_iou": FROZEN_NMS_IOU,
        "max_detections": 300,
        "device": 0,
        "class_names": CLASS_NAMES,
        "operating_point_status": FROZEN_OPERATING_POINT_STATUS,
        "output_path": DEMO_OUTPUT_RELATIVE_PATH.as_posix(),
        "validation_images": VALIDATION_IMAGES_RELATIVE_PATH.as_posix(),
        "validation_labels": VALIDATION_LABELS_RELATIVE_PATH.as_posix(),
        "sample_output_path": SAMPLE_OUTPUT_RELATIVE_PATH.as_posix(),
        "images_per_class": 2,
        "negative_images": 2,
        "sample_seed": 42,
    }
    actual = {
        "schema_version": raw.get("schema_version"),
        "model_path": config.model_path.as_posix(),
        "model_sha256": config.model_sha256,
        "model_size_bytes": config.model_size_bytes,
        "imgsz": config.imgsz,
        "confidence": config.confidence,
        "nms_iou": config.nms_iou,
        "max_detections": config.max_detections,
        "device": config.device,
        "class_names": dict(config.class_names),
        "operating_point_status": config.operating_point_status,
        "output_path": config.output_path.as_posix(),
        "validation_images": config.validation_images.as_posix(),
        "validation_labels": config.validation_labels.as_posix(),
        "sample_output_path": config.sample_output_path.as_posix(),
        "images_per_class": config.images_per_class,
        "negative_images": config.negative_images,
        "sample_seed": config.sample_seed,
    }
    if actual != expected:
        changed = sorted(key for key in expected if actual[key] != expected[key])
        raise MentorDemoError(
            "Mentor demo config changed a frozen value: " + ", ".join(changed)
        )
    return config


def frozen_model_path(project_root: Path = PROJECT_ROOT) -> Path:
    """Return the only checkpoint accepted by this demo."""
    return (project_root / FROZEN_MODEL_RELATIVE_PATH).resolve()


def verify_model_identity(
    model_path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> str:
    """Verify model size and SHA-256, returning the actual SHA-256."""
    try:
        size_bytes = model_path.stat().st_size
    except FileNotFoundError as exc:
        raise MentorDemoError(f"Frozen best.pt does not exist: {model_path}") from exc
    except OSError as exc:
        raise MentorDemoError(f"Could not inspect frozen best.pt: {model_path}") from exc
    if not model_path.is_file():
        raise MentorDemoError(f"Frozen model path is not a file: {model_path}")
    if size_bytes != expected_size_bytes:
        raise MentorDemoError(
            f"Frozen best.pt size mismatch: expected {expected_size_bytes}, got {size_bytes}."
        )
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_sha256.upper():
        raise MentorDemoError(
            "Frozen best.pt SHA-256 mismatch: "
            f"expected {expected_sha256.upper()}, got {actual_sha256}."
        )
    return actual_sha256


def verify_frozen_model(
    candidate: Path,
    config: DemoConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Reject every checkpoint path except the configured frozen best.pt."""
    required = frozen_model_path(project_root)
    if candidate.resolve() != required:
        raise MentorDemoError(
            f"The mentor demo accepts only the frozen best.pt at: {required}"
        )
    if _resolve_project_path(config.model_path, project_root) != required:
        raise MentorDemoError("Configured model path is not the frozen mentor-demo best.pt.")
    return verify_model_identity(
        required,
        expected_sha256=config.model_sha256,
        expected_size_bytes=config.model_size_bytes,
    )


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def validate_source_path(
    source: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[Path]:
    """Return supported input images while refusing the protected internal test split."""
    resolved = source.resolve()
    internal_test = (project_root / INTERNAL_TEST_IMAGES_RELATIVE_PATH).resolve()
    if _is_within(resolved, internal_test):
        raise MentorDemoError("The mentor demo must not access the internal test split.")
    if not resolved.exists():
        raise MentorDemoError(f"Source does not exist: {resolved}")
    if resolved.is_file():
        if resolved.suffix.casefold() in {".pt", ".pth", ".ckpt"}:
            raise MentorDemoError("A model checkpoint is not a supported image source.")
        if resolved.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
            raise MentorDemoError(f"Unsupported image file: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise MentorDemoError(f"Source is neither an image nor a directory: {resolved}")
    try:
        images = sorted(
            (
                path.resolve()
                for path in resolved.iterdir()
                if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
    except OSError as exc:
        raise MentorDemoError(f"Could not enumerate source directory: {resolved}") from exc
    if not images:
        raise MentorDemoError(f"Source directory contains no supported images: {resolved}")
    if any(_is_within(image, internal_test) for image in images):
        raise MentorDemoError("The mentor demo must not access the internal test split.")
    return images


def next_output_path(output_directory: Path, source_filename: str) -> Path:
    """Choose a non-existing output filename without destructive overwrite."""
    filename = Path(source_filename).name
    if not filename or Path(filename).suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
        raise MentorDemoError(f"Cannot create an image output for: {source_filename!r}")
    candidate = output_directory / filename
    if not candidate.exists():
        return candidate
    for index in range(1, 10_000):
        candidate = output_directory / f"{Path(filename).stem}_{index:03d}{Path(filename).suffix}"
        if not candidate.exists():
            return candidate
    raise MentorDemoError(f"Could not allocate a safe output filename in: {output_directory}")


def _parse_yolo_label(label_path: Path) -> GroundTruthSample:
    try:
        data = label_path.read_bytes()
    except OSError as exc:
        raise MentorDemoError(f"Could not read validation label: {label_path}") from exc
    counts: dict[int, int] = {}
    max_areas: dict[int, float] = {}
    if data:
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise MentorDemoError(f"Validation label is not UTF-8: {label_path}") from exc
        if not lines:
            raise MentorDemoError(f"Non-empty validation label has no rows: {label_path}")
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise MentorDemoError(f"Invalid YOLO row at {label_path}:{line_number}")
            try:
                class_id = int(fields[0])
                center_x, center_y, width, height = (float(value) for value in fields[1:])
            except ValueError as exc:
                raise MentorDemoError(
                    f"Invalid YOLO value at {label_path}:{line_number}"
                ) from exc
            if class_id not in CLASS_NAMES:
                raise MentorDemoError(
                    f"Unexpected class ID {class_id} at {label_path}:{line_number}"
                )
            if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
                raise MentorDemoError(
                    f"YOLO center is outside [0,1] at {label_path}:{line_number}"
                )
            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                raise MentorDemoError(
                    f"YOLO size is outside (0,1] at {label_path}:{line_number}"
                )
            counts[class_id] = counts.get(class_id, 0) + 1
            max_areas[class_id] = max(max_areas.get(class_id, 0.0), width * height)
    return GroundTruthSample(
        filename="",
        image_path=Path(),
        label_path=label_path,
        classes_present=tuple(sorted(counts)),
        class_object_counts=tuple(sorted(counts.items())),
        class_max_normalized_areas=tuple(sorted(max_areas.items())),
    )


def validate_validation_sample_directories(
    images_directory: Path,
    labels_directory: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path]:
    """Allow sample construction from exactly the approved validation directories."""
    images = images_directory.resolve()
    labels = labels_directory.resolve()
    required_images = (project_root / VALIDATION_IMAGES_RELATIVE_PATH).resolve()
    required_labels = (project_root / VALIDATION_LABELS_RELATIVE_PATH).resolve()
    if images != required_images or labels != required_labels:
        raise MentorDemoError(
            "Mentor samples may be selected only from the approved validation image/label directories; "
            "internal-test, teacher-video, quarantine, guard, and other sources are forbidden."
        )
    if not images.is_dir() or not labels.is_dir():
        raise MentorDemoError("Approved validation image or label directory is missing.")
    return images, labels


def inventory_validation_samples(
    images_directory: Path,
    labels_directory: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[GroundTruthSample]:
    """Inventory paired validation images and labels without reading any other split."""
    images_directory, labels_directory = validate_validation_sample_directories(
        images_directory, labels_directory, project_root=project_root
    )
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    try:
        for path in images_directory.iterdir():
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES:
                key = path.stem.casefold()
                if key in images:
                    raise MentorDemoError(f"Duplicate validation image stem: {path.stem}")
                images[key] = path.resolve()
        for path in labels_directory.iterdir():
            if path.is_file() and path.suffix.casefold() == ".txt":
                key = path.stem.casefold()
                if key in labels:
                    raise MentorDemoError(f"Duplicate validation label stem: {path.stem}")
                labels[key] = path.resolve()
    except OSError as exc:
        raise MentorDemoError("Could not inventory the approved validation split.") from exc
    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    if missing_labels or orphan_labels:
        raise MentorDemoError(
            "Validation image/label pairing failed: "
            f"missing_labels={missing_labels[:5]}, orphan_labels={orphan_labels[:5]}"
        )
    samples: list[GroundTruthSample] = []
    for stem in sorted(images):
        parsed = _parse_yolo_label(labels[stem])
        samples.append(
            GroundTruthSample(
                filename=images[stem].name,
                image_path=images[stem],
                label_path=labels[stem],
                classes_present=parsed.classes_present,
                class_object_counts=parsed.class_object_counts,
                class_max_normalized_areas=parsed.class_max_normalized_areas,
            )
        )
    return samples


def _seeded_rank(filename: str, seed: int) -> str:
    token = f"mentor-demo-samples-v1\0{seed}\0{filename}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def select_mentor_samples(
    samples: Sequence[GroundTruthSample],
    *,
    images_per_class: int = 2,
    negative_images: int = 2,
    seed: int = 42,
) -> list[SelectedSample]:
    """Select clear GT examples deterministically, never using model predictions."""
    if images_per_class <= 0 or negative_images <= 0:
        raise MentorDemoError("Mentor sample counts must be positive.")
    selected: list[SelectedSample] = []
    used: set[str] = set()
    for class_id, class_name in CLASS_NAMES.items():
        def class_rank(sample: GroundTruthSample) -> tuple[float, str, str]:
            area = dict(sample.class_max_normalized_areas)[class_id]
            return (-area, _seeded_rank(sample.filename, seed), sample.filename.casefold())

        candidates = sorted(
            (
                sample
                for sample in samples
                if class_id in sample.classes_present and sample.filename not in used
            ),
            key=class_rank,
        )
        if len(candidates) < images_per_class:
            raise MentorDemoError(
                f"Validation split lacks {images_per_class} unique {class_name} samples."
            )
        for sample in candidates[:images_per_class]:
            selected.append(SelectedSample(class_name, class_id, sample))
            used.add(sample.filename)

    negatives = sorted(
        (sample for sample in samples if sample.is_negative and sample.filename not in used),
        key=lambda sample: (_seeded_rank(sample.filename, seed), sample.filename.casefold()),
    )
    if len(negatives) < negative_images:
        raise MentorDemoError(
            f"Validation split lacks {negative_images} exactly-empty negative labels."
        )
    for sample in negatives[:negative_images]:
        selected.append(SelectedSample("negative", None, sample))
        used.add(sample.filename)
    return selected


def prepare_mentor_samples(
    config: DemoConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, tuple[SelectedSample, ...]]:
    """Copy the deterministic validation-only mentor samples and write provenance."""
    images_directory = _resolve_project_path(config.validation_images, project_root)
    labels_directory = _resolve_project_path(config.validation_labels, project_root)
    output_directory = _resolve_project_path(config.sample_output_path, project_root)
    validate_validation_sample_directories(
        images_directory, labels_directory, project_root=project_root
    )
    if output_directory.exists():
        raise MentorDemoError(f"Refusing to overwrite existing mentor samples: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=".mentor_samples.", dir=output_directory.parent)
    )
    try:
        inventory = inventory_validation_samples(
            images_directory, labels_directory, project_root=project_root
        )
        selected = select_mentor_samples(
            inventory,
            images_per_class=config.images_per_class,
            negative_images=config.negative_images,
            seed=config.sample_seed,
        )
        manifest_rows: list[dict[str, Any]] = []
        for item in selected:
            sample = item.sample
            copied_image = temporary_directory / sample.filename
            shutil.copyfile(sample.image_path, copied_image)
            source_image_sha256 = sha256_file(sample.image_path)
            copied_image_sha256 = sha256_file(copied_image)
            if source_image_sha256 != copied_image_sha256:
                raise MentorDemoError(f"Copied sample hash mismatch: {sample.filename}")
            manifest_rows.append(
                {
                    "filename": sample.filename,
                    "selection_bucket": item.bucket,
                    "selection_class_id": item.class_id,
                    "classes_present": list(sample.classes_present),
                    "class_names_present": [CLASS_NAMES[value] for value in sample.classes_present],
                    "class_object_counts": {
                        str(key): value for key, value in sample.class_object_counts
                    },
                    "class_max_normalized_bbox_areas": {
                        str(key): value for key, value in sample.class_max_normalized_areas
                    },
                    "source_image_path": _project_relative(sample.image_path, project_root),
                    "source_label_path": _project_relative(sample.label_path, project_root),
                    "copied_image_path": _project_relative(
                        output_directory / sample.filename, project_root
                    ),
                    "source_image_sha256": source_image_sha256,
                    "copied_image_sha256": copied_image_sha256,
                    "source_label_sha256": sha256_file(sample.label_path),
                }
            )
        manifest = {
            "schema_version": "mentor_demo.sample_manifest.v1",
            "source_policy": "validation_only_ground_truth_no_model_predictions",
            "source_validation_images": _project_relative(images_directory, project_root),
            "source_validation_labels": _project_relative(labels_directory, project_root),
            "selection_rule": (
                "For classes 0,1,2,3 in order, choose two unused images by descending "
                "maximum normalized ground-truth box area for that class; break ties with "
                "SHA-256(seed=42, filename). Choose two exactly-empty-label negatives by "
                "the same seeded filename rank."
            ),
            "seed": config.sample_seed,
            "class_mapping": {str(key): value for key, value in CLASS_NAMES.items()},
            "counts": {
                "total": len(selected),
                "positive": len(selected) - config.negative_images,
                "negative": config.negative_images,
                "assigned_per_class": {
                    str(key): sum(item.class_id == key for item in selected)
                    for key in CLASS_NAMES
                },
            },
            "samples": manifest_rows,
        }
        manifest_path = temporary_directory / "sample_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary_directory, output_directory)
        return output_directory / "sample_manifest.json", tuple(selected)
    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        raise


def _validate_model_class_names(model_names: Any) -> None:
    try:
        if isinstance(model_names, Mapping):
            actual = {int(key): str(value) for key, value in model_names.items()}
        else:
            actual = {index: str(value) for index, value in enumerate(model_names)}
    except (TypeError, ValueError) as exc:
        raise MentorDemoError("Could not read class mapping from frozen best.pt.") from exc
    if actual != CLASS_NAMES:
        raise MentorDemoError(
            f"Frozen model class mapping mismatch: expected {CLASS_NAMES}, got {actual}."
        )


def _detections_from_result(result: Any, width: int, height: int) -> tuple[Detection, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ()
    try:
        xyxy_rows = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise MentorDemoError("Could not decode detections returned by YOLO.") from exc
    detections: list[Detection] = []
    for xyxy, confidence, raw_class_id in zip(
        xyxy_rows, confidence_values, class_values, strict=True
    ):
        class_id = int(raw_class_id)
        if class_id not in CLASS_NAMES:
            raise MentorDemoError(f"Model returned unsupported class ID: {class_id}")
        x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
        bbox = (
            max(0, min(width - 1, x1)),
            max(0, min(height - 1, y1)),
            max(0, min(width - 1, x2)),
            max(0, min(height - 1, y2)),
        )
        detections.append(
            Detection(class_id, CLASS_NAMES[class_id], float(confidence), bbox)
        )
    return tuple(
        sorted(
            detections,
            key=lambda item: (-item.confidence, item.class_id, item.bbox_xyxy_pixel),
        )
    )


def _annotate_image(image: Any, detections: Sequence[Detection], cv2: Any) -> Any:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    thickness = max(2, int(round(min(width, height) / 320)))
    font_scale = max(0.5, min(0.9, width / 1100.0))
    colors = {
        0: (0, 220, 255),
        1: (255, 180, 0),
        2: (0, 210, 0),
        3: (30, 30, 255),
    }
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy_pixel
        color = colors[detection.class_id]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_top = max(0, y1 - text_height - baseline - 8)
        label_right = min(width - 1, x1 + text_width + 8)
        cv2.rectangle(
            annotated, (x1, label_top), (label_right, y1), color, cv2.FILLED
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(text_height + 2, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def format_image_summary(result: ImageDemoResult) -> list[str]:
    """Format the concise, presentation-safe terminal summary."""
    noun = "object" if len(result.detections) == 1 else "objects"
    lines = [
        "",
        f"Image: {result.source_path}",
        f"Resolution: {result.width}x{result.height}",
        f"Model inference time: {result.inference_time_ms:.2f} ms (not application FPS)",
        f"Detected {len(result.detections)} road-damage {noun}",
    ]
    for detection in result.detections:
        lines.append(
            f"class_id={detection.class_id} name={detection.class_name} "
            f"confidence={detection.confidence:.3f} "
            f"bbox_xyxy_pixel={detection.bbox_xyxy_pixel}"
        )
    lines.extend(
        [f"Annotated result saved to: {result.output_path}", FROZEN_OPERATING_POINT_NOTICE]
    )
    return lines


def run_demo(
    source: Path,
    config: DemoConfig,
    *,
    project_root: Path = PROJECT_ROOT,
    display: bool = True,
    write_line: Callable[[str], None] | None = None,
) -> tuple[ImageDemoResult, ...]:
    """Run one frozen-model inference session over an image or flat image folder."""
    sources = validate_source_path(source, project_root=project_root)
    model_path = frozen_model_path(project_root)
    actual_sha256 = verify_frozen_model(model_path, config, project_root=project_root)
    output_directory = _resolve_project_path(config.output_path, project_root)
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MentorDemoError(
            f"Could not create mentor-demo output directory: {output_directory}"
        ) from exc
    if not output_directory.is_dir():
        raise MentorDemoError(
            f"Mentor-demo output path is not a directory: {output_directory}"
        )

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise MentorDemoError(
            "The approved demo environment must provide OpenCV and Ultralytics."
        ) from exc

    model = YOLO(str(model_path), task="detect")
    _validate_model_class_names(model.names)
    if write_line is not None:
        write_line(f"Frozen model SHA-256 verified: {actual_sha256}")
        write_line(FROZEN_OPERATING_POINT_NOTICE)

    results: list[ImageDemoResult] = []
    display_available = display
    try:
        for image_path in sources:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise MentorDemoError(f"OpenCV could not decode image: {image_path}")
            height, width = image.shape[:2]
            predictions = model.predict(
                source=image,
                imgsz=config.imgsz,
                conf=config.confidence,
                iou=config.nms_iou,
                max_det=config.max_detections,
                device=config.device,
                save=False,
                verbose=False,
            )
            if len(predictions) != 1:
                raise MentorDemoError(
                    f"Expected one YOLO result for {image_path}, got {len(predictions)}."
                )
            prediction = predictions[0]
            detections = _detections_from_result(prediction, width, height)
            annotated = _annotate_image(image, detections, cv2)
            output_path = next_output_path(output_directory, image_path.name)
            try:
                written = cv2.imwrite(str(output_path), annotated)
            except cv2.error as exc:
                raise MentorDemoError(f"Could not save annotated image: {output_path}") from exc
            if not written:
                raise MentorDemoError(f"Could not save annotated image: {output_path}")
            speed = getattr(prediction, "speed", {})
            inference_time_ms = float(speed.get("inference", 0.0))
            result = ImageDemoResult(
                source_path=image_path,
                output_path=output_path,
                width=width,
                height=height,
                inference_time_ms=inference_time_ms,
                detections=detections,
            )
            results.append(result)
            if write_line is not None:
                for line in format_image_summary(result):
                    write_line(line)
            if display_available:
                try:
                    cv2.imshow(IMAGE_WINDOW_NAME, annotated)
                    key = cv2.waitKey(0) & 0xFF
                except cv2.error as exc:
                    display_available = False
                    LOGGER.warning(
                        "OpenCV GUI display is unavailable; saved output remains valid: %s", exc
                    )
                else:
                    if key in (27, ord("q"), ord("Q")):
                        break
    finally:
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    return tuple(results)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen YOLOv8s road-damage baseline on an image or folder."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--source", type=Path, help="Image file or flat image folder.")
    action.add_argument(
        "--prepare-samples",
        action="store_true",
        help="Create the deterministic validation-only mentor sample folder once.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save and summarize detections without opening an OpenCV window.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_demo_config()
        if args.prepare_samples:
            manifest_path, selected = prepare_mentor_samples(config)
            print(f"Prepared {len(selected)} validation-only mentor images.")
            for item in selected:
                print(f"{item.bucket}: {item.sample.filename}")
            print(f"Sample manifest: {manifest_path}")
            return 0
        run_demo(
            args.source,
            config,
            display=not args.no_display,
            write_line=print,
        )
    except MentorDemoError as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Demo stopped by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
