# AI-Based Road Damage Detection

An academic computer-vision project for detecting road damage in images and
dashcam video. The repository currently contains a reproducible RDD2022 data
pipeline, two completed detector experiments, validation-only operating-point
selection, error analysis, presentation demos, the Phase 4 video/event
pipeline, and a non-executing Phase 5A FastAPI foundation. Analysis-job
execution, result storage, the dashboard, and deployment remain future work.

## Problem Statement

Manual road inspection is slow and difficult to repeat consistently. This
project studies automated detection of four visible pavement-damage classes and
is being extended toward a video system that can stabilize detections, avoid
counting the same defect repeatedly across adjacent frames, and produce
reviewable reports. It is an academic screening prototype, not a certified
pavement-engineering instrument, and it does not claim physical depth, area,
volume, repair cost, or engineering severity from monocular imagery.

## Current Project Status

| Area | Status |
|---|---|
| RDD2022 India + Japan dataset pipeline and audited YOLO export | Complete |
| YOLOv8s public baseline training and evaluation | Complete |
| Matched YOLO26s comparison experiment | Complete |
| Validation-only threshold selection for both detectors | Complete |
| Validation-only error analysis and detector selection | Complete |
| Primary detector selection | YOLOv8s selected |
| Full video inference and heuristic temporal event aggregation | Complete (Phase 4) |
| FastAPI contracts, health/system endpoints, and service boundary | Complete (Phase 5A; execution disabled) |
| Analysis-job execution, result storage, dashboard, and deployment | Next phase |

The repository includes image/video utilities and stable backend contracts, but
it does not yet expose video submission or execute analysis through the API.

## Damage Classes

The numeric order is fixed throughout the project.

| Class ID | Code | Damage type |
|---:|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

## Dataset

The detector experiments use the India and Japan portion of
[RDD2022 on Figshare](https://doi.org/10.6084/m9.figshare.21431547.v1). The
canonical processed dataset is `rdd2022_india_japan_v1_1`, version 1.1.0. Its
split is described as **leakage-reduced and grouped**: related source groups are
assigned together rather than randomly distributing adjacent or related images.
This is not claimed to be a route-independent split.

Raw RDD2022 images, downloaded archives, processed datasets, YOLO image/label
trees, and the private teacher dashcam video are intentionally excluded from
Git. The tracked repository contains configurations, source code, tests, and
reproducibility metadata—not the full dataset.

## Dataset Split

| Split | Images | Development use |
|---|---:|---|
| Train | 12,620 | Model fitting |
| Validation | 2,602 | Checkpoint selection, thresholds, and error analysis |
| Internal test | 2,775 | Protected final evaluation only |

The internal test split is isolated from development and threshold tuning. This
README intentionally does not expose individual internal-test samples or
provide a command to rerun that evaluation.

## Model Experiments

The following are the best standard Ultralytics validation summaries recorded
within each completed training run.

| Model | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8s | 50 | 0.53738 | 0.46688 | 0.47537 | 0.21631 |
| YOLO26s | 63 | 0.55503 | 0.46612 | 0.47631 | 0.21702 |

Training mAP alone was not used to select the primary detector. Each checkpoint
was also assessed at a frozen validation-selected operating point using the same
four-class macro-F1 objective.

## Operating-Point Comparison

| Model | Confidence | Post-processing | Matching IoU | Macro-F1 |
|---|---:|---|---:|---:|
| YOLOv8s | 0.19 | NMS IoU 0.50 | 0.50 | 0.5055312316193121 |
| YOLO26s | 0.193 | Native end-to-end, NMS-free | 0.50 | 0.4954746679 |

YOLOv8s is the selected primary detector because it provides the better balanced
validation macro-F1. YOLO26s remains a useful controlled comparison: it has
cleaner behavior on negative-only road images, stronger D40 F1, and slightly
higher standard validation mAP.

## Key Error-Analysis Findings

- YOLO26s gains some true positives but adds more false positives within images
  that contain labelled damage.
- YOLO26s produces fewer false positives on negative-only road images at its
  selected operating point.
- Small D20 objects are difficult for both models.
- Higher target density is descriptively associated with lower recall; this is
  not presented as a causal result.
- Current evidence shows a broad precision/class trade-off and does not justify
  a new Experiment 3.

## Project Architecture

The completed research path is:

```text
RDD2022 India + Japan
  -> acquisition and integrity audit
  -> grouped canonical dataset
  -> deterministic YOLO export
  -> YOLOv8s / YOLO26s training
  -> validation threshold selection
  -> validation error analysis
  -> YOLOv8s primary detector
```

The application continuation is:

```text
selected detector
  -> video inference
  -> tracking and temporal damage-event aggregation
  -> annotated video and structured report
  -> Phase 5A FastAPI contracts (execution disabled)
  -> analysis jobs and result/database layer
  -> frontend dashboard
  -> deployment
```

Tracking IDs will remain intermediate associations. Final unique counts are
planned to come from the project-owned `DamageEvent` aggregation layer.

## Repository Structure

```text
AI_Road_Damage_Project/
|-- configs/
|   |-- dataset/       # acquisition, canonicalization, and YOLO export policies
|   |-- demo/          # frozen mentor and video demo settings
|   |-- evaluation/    # operating-point and error-analysis protocols
|   |-- training/      # frozen baseline and YOLO26s experiment configs
|   `-- video/         # inspection and reconnaissance settings
|-- src/road_damage/
|   |-- api/            # Phase 5A HTTP contracts; analysis execution disabled
|   |-- aggregation/    # heuristic temporal damage-event aggregation
|   |-- dataset/       # RDD2022 acquisition, audit, construction, and export
|   |-- demo/          # image and video presentation demos
|   |-- evaluation/    # threshold selection and validation analyses
|   |-- inference/      # frozen Phase 4 video application service
|   |-- training/      # baseline, Experiment 2, checkpoint, and resume tooling
|   `-- video/         # metadata inspection and reconnaissance
|-- tests/              # unit and integration tests
|-- reproducibility/    # source-state and pretrained-asset identities
|-- road-damage-project-docs/
|                       # PRD, architecture, runbooks, governance, and roadmap
|-- data/                # local/ignored raw, processed, and exported datasets
|-- outputs/             # local/ignored generated runs and evaluation artifacts
|-- models/, weights/    # local/ignored model assets
|-- AGENTS.md
|-- requirements-backend.txt
|-- requirements.txt
`-- README.md
```

Generated trees are shown only to explain local layout; they are not intended
for Git tracking.

## Environment

The completed experiments were run with:

- Windows
- Python 3.12.10
- PyTorch 2.13.0+cu126
- torchvision 0.28.0+cu126
- Ultralytics 8.4.130
- NumPy 2.5.2
- OpenCV 5.0.0.93
- FFmpeg/ffprobe on `PATH`
- NVIDIA GeForce RTX 2050 used for training

The tracked `requirements.txt` is the currently declared Phase 2C dependency
subset (`numpy` and headless OpenCV); it is not a complete lock file for the
frozen GPU training environment.

## Setup

From PowerShell in the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ffmpeg -version
ffprobe -version
```

These commands install only the dependencies declared by the repository. For
the exact ML environment used by the completed experiments, review
[Technology Stack](road-damage-project-docs/TECH_STACK.md) and the relevant run
manifest before installing CUDA/PyTorch/Ultralytics packages. Do not silently
substitute versions for a scientific reproduction.

Large/private inputs and trained weights are not downloaded by setup. Their
source, permission, and checksum must be verified separately.

## Testing

The full local suite is run with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The recorded project state at the time of this release-preparation pass is
**300 tests passed locally**. This is a recorded result, not a guarantee for a
different machine or dependency set. See the
[Manual Testing Guide](road-damage-project-docs/MANUAL_TESTING_GUIDE.md) for
safe environment, checksum, artifact, and CLI-help checks.

## Reproducibility

The project preserves:

- frozen dataset, training, demo, and evaluation configurations;
- checkpoint sizes and SHA-256 identities;
- canonical dataset and YOLO-export fingerprints;
- source-state manifests and Git commit identities;
- immutable run manifests and completion receipts;
- deterministic seeds where applicable;
- validation-only checkpoint/threshold selection;
- explicit native end-to-end/NMS-free handling for YOLO26s.

Generated outputs and large weights may be absent from a clone. Their identity
records and runbooks remain the reference for authorized local reconstruction.

## Data and Model Availability

- RDD2022 raw/downloaded assets and all processed/exported image trees are not
  stored in Git.
- The teacher dashcam video and user-recorded road videos are private local
  assets and are not stored in Git.
- Training/evaluation outputs and model weights are ignored because of size,
  privacy, and reproducibility policy.
- Dataset acquisition policy is recorded in
  [`configs/dataset/rdd2022_phase2a.yaml`](configs/dataset/rdd2022_phase2a.yaml)
  and the implementation is documented by the
  [Dataset Guidelines](road-damage-project-docs/DATASET_GUIDELINES.md).
- Frozen model procedures are documented in the
  [YOLOv8s baseline runbook](road-damage-project-docs/BASELINE_PUBLIC_V1_RUNBOOK.md)
  and [YOLO26s runbook](road-damage-project-docs/EXPERIMENT2_YOLO26S_RUNBOOK.md).

Do not redistribute dataset images, private video, or model weights without
reviewing their source-specific permissions and license obligations.

## Limitations

- Training and validation use only the RDD2022 India + Japan subset.
- The detector supports only D00, D10, D20, and D40.
- The target-domain teacher video does not have reliable object-level ground
  truth and is a demonstration input, not an accuracy benchmark.
- Country comparisons can be confounded by camera, road, annotation, and domain
  differences; they are descriptive rather than causal.
- India D10 has low validation support (10 boxes in 10 images).
- Fine or distant cracks and small D20 regions remain difficult.
- API submission/execution, durable job storage, dashboard, and deployment are
  still under development; Phase 5A endpoints are read-only capabilities.

## Roadmap

1. Connect the Phase 5A service boundary to a reviewed asynchronous analysis-job lifecycle.
2. Add result storage/database records.
3. Build the frontend dashboard.
4. Run the teacher dashcam demonstration without treating it as ground-truth
   evaluation.
5. Package and deploy the end-to-end application.

## Documentation

- [Product and Research Requirements](road-damage-project-docs/PROJECT_PRD.md)
- [System Architecture](road-damage-project-docs/ARCHITECTURE.md)
- [Dataset and Annotation Guidelines](road-damage-project-docs/DATASET_GUIDELINES.md)
- [Technology Stack](road-damage-project-docs/TECH_STACK.md)
- [Project Workflow](road-damage-project-docs/WORKFLOW.md)
- [Evaluation Plan](road-damage-project-docs/EVALUATION_PLAN.md)
- [Project Roadmap](road-damage-project-docs/PROJECT_ROADMAP.md)
- [YOLOv8s Baseline Runbook](road-damage-project-docs/BASELINE_PUBLIC_V1_RUNBOOK.md)
- [Threshold-Selection Runbook](road-damage-project-docs/THRESHOLD_SELECTION_RUNBOOK.md)
- [Validation Error-Analysis Runbook](road-damage-project-docs/ERROR_ANALYSIS_RUNBOOK.md)
- [YOLO26s Experiment Runbook](road-damage-project-docs/EXPERIMENT2_YOLO26S_RUNBOOK.md)
- [Phase 5A FastAPI Backend](road-damage-project-docs/FASTAPI_BACKEND.md)
- [Mentor Demo Runbook](road-damage-project-docs/MENTOR_DEMO_RUNBOOK.md)
- [Video Demo Runbook](road-damage-project-docs/VIDEO_DEMO_RUNBOOK.md)
- [Ethics, Privacy, and Licensing](road-damage-project-docs/ETHICS_AND_LICENSE.md)
- [Manual Testing Guide](road-damage-project-docs/MANUAL_TESTING_GUIDE.md)

## License / Attribution

RDD2022 is attributed to its original authors and distribution record. The
Figshare v1 record states **CC BY 4.0**, while the authors' repository states
**CC BY-SA 4.0**. The project records this discrepancy as unresolved and uses
CC BY-SA 4.0 as its conservative interpretation. Review the
[RDD2022 Figshare record](https://doi.org/10.6084/m9.figshare.21431547.v1), the
[RoadDamageDetector repository](https://github.com/sekilab/RoadDamageDetector),
and the project's [licensing governance](road-damage-project-docs/ETHICS_AND_LICENSE.md)
before redistributing dataset-derived assets.

Ultralytics software and model usage is subject to its own current licensing
terms. This repository does **not currently contain a root software license**;
no project-wide reuse grant should be inferred. Selecting and adding an
appropriate repository license is required before a public release.

## Authors / Academic Context

This repository supports a student final-year project and faculty evaluation.
No institution or additional team-member information is asserted here; use the
project PRD and Git history as the authoritative authorship record.
