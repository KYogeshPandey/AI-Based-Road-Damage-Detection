# Technology Stack

## 1. Selection principle

Use one Python-centered stack until the complete offline pipeline works. A separate React/Node frontend, microservices, Kubernetes, and cloud deployment would add engineering work without improving the main research questions.

## 2. Approved stack

| Layer | Selection | Purpose |
|---|---|---|
| Language | Python 3.12.10 | Verified environment for the approved public YOLOv8s baseline |
| Deep learning | PyTorch through Ultralytics | Detector training and inference |
| Baseline detector | YOLOv8s | Literature-aligned baseline |
| Proposed detector | YOLO26s | Main detector; use `n` only for compute fallback |
| Tracking | BoT-SORT and ByteTrack | Primary and baseline moving-camera association |
| Video | OpenCV + FFmpeg/ffprobe | Decode, metadata, overlays, and reliable encoding |
| Annotation | CVAT | Video/image boxes, review, and YOLO/COCO export |
| Augmentation | Ultralytics pipeline; Albumentations only when needed | Training-time road-scene augmentation |
| Data | NumPy, Pandas | Geometry, manifests, reports, and analysis |
| Plots | Matplotlib and Seaborn | Metrics and dissertation-ready figures |
| Dashboard | Streamlit | Fast academic demo using the same Python pipeline |
| Config | YAML + typed validation | Reproducible component and experiment settings |
| Testing | pytest | Unit and integration tests |
| Quality | Ruff + mypy or Pyright | Formatting/linting and static type checks |
| Environment | `venv`/uv or Conda; one lock strategy | Reproducible installation |
| Version control | Git + GitHub | Code, documentation, and experiment metadata |
| Training compute | Local NVIDIA CUDA, Colab, or Kaggle | GPU training based on availability |

## 3. Version policy

Do not write unbounded dependency names into the final environment. During initial setup:

1. use the approved Python 3.12.10 baseline environment;
2. run a GPU/CPU smoke test;
3. capture exact versions in a lock file;
4. save `pip freeze` or equivalent in experiment manifests;
5. update dependencies only in a dedicated change with tests.

Because the vision ecosystem changes quickly, this document intentionally does not pretend that today's newest package version will remain appropriate for the whole project.

The approved public-baseline environment is Python 3.12.10 with `torch==2.13.0+cu126`, `torchvision==0.28.0+cu126`, and `ultralytics==8.4.130`. These versions must be recorded with the experiment and changed only through a reviewed, separately tested environment update.

## 4. Detector decision

### Baseline: YOLOv8s

YOLOv8 appears throughout the project's initial literature and provides an understandable comparison point.

### Proposed: YOLO26s

YOLO26 is the current Ultralytics family and supports detection and tracking workflows. Use the small model as the intended accuracy/compute compromise. If available compute cannot train it reliably, use YOLO26n and record the reason rather than hiding the change.

Do not compare detectors unfairly. Keep the dataset version, split, evaluation code, and image size constant unless an experiment intentionally studies one of those settings.

## 5. Tracker decision

### Primary: BoT-SORT

BoT-SORT supports global motion compensation and optional appearance-based association, making it the planned primary tracker for moving dashcam footage.

### Baseline: ByteTrack

ByteTrack is efficient and widely used. It provides a meaningful speed and association baseline.

Tracker choice remains an experimental question. The final report must use measured results, not assume that BoT-SORT always wins.

## 6. Annotation format

Maintain one archival export that preserves annotation provenance and one generated training export.

- Archival: CVAT/Datumaro or COCO with project/task metadata.
- Training: Ultralytics YOLO detection format.
- Tracking evaluation subset: a track-capable CVAT/MOT-style export when identities are manually labelled.

The ordinary legacy YOLO format does not preserve all video track metadata. Do not treat it as the only archive.

## 7. Dashboard decision

Use Streamlit after CLI completion because it allows upload/configuration, charts, event tables, video review, and downloads without introducing a second application stack. If a production service becomes a later goal, FastAPI plus a separate frontend can be considered as a post-project extension.

## 8. Tools deliberately excluded from V1

| Tool/approach | Reason deferred |
|---|---|
| TensorFlow/Keras | Duplicates the PyTorch stack |
| Mask R-CNN/Detectron2 | Requires mask annotations and more integration work |
| React/Node/MongoDB | Not needed for a single-user offline research prototype |
| Docker | Useful after the local workflow stabilizes, not a prerequisite |
| MLflow/W&B | Optional; local immutable run folders are sufficient initially |
| Automatic road segmentation | Manual ROI is adequate for V1 |
| Depth/monocular 3D model | Cannot support reliable physical claims without calibration |
| Cloud deployment | Privacy, cost, and licensing questions should be resolved first |

## 9. Minimum development environment

- 64-bit Windows or Linux;
- Python 3.12.10 for the approved baseline experiment;
- Git;
- FFmpeg/ffprobe available on `PATH`;
- 16 GB RAM recommended;
- sufficient storage for source video, extracted JPEGs, public data, weights, and outputs;
- NVIDIA CUDA GPU or an approved cloud GPU for training; CPU-only local tooling remains supported.

## 10. Reproducibility files expected in the future repository

```text
pyproject.toml
dependency lock file
.python-version (optional)
.env.example (only if required)
configs/base.yaml
configs/dataset.yaml
configs/detector/*.yaml
configs/tracker/*.yaml
```

## 11. Licensing note

Ultralytics documents AGPL-3.0 and enterprise licensing options. For the planned academic project, the simplest path is a fully open-source, license-compatible repository. Before any proprietary, closed-source, or commercial deployment, re-check the current terms and obtain appropriate guidance. See [Ultralytics licensing](https://www.ultralytics.com/license).
