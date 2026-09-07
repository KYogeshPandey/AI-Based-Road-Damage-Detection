# Product and Research Requirements Document

## 1. Document status

| Field | Value |
|---|---|
| Project | AI-Based Road Damage Detection and Analysis from Dashcam Video |
| Owner | Vansh Tyagi |
| Status | Approved development baseline |
| Version | 1.0 |
| Last updated | 2026-09-02 |
| Source of truth | This document |

## 2. Project statement

Build a reproducible system that accepts forward-facing dashcam video, detects four standard road-damage classes, maintains stable tracks, converts observations into uniquely counted physical-damage events, estimates relative visual severity in a controlled image zone, and generates an annotated video plus structured reports and an interactive dashboard.

The project is an academic prototype for road-condition screening. It is not a certified pavement-engineering instrument and must not claim physical pothole depth, crack width, repair cost, or safety certification.

## 3. Why this project is needed

Manual pavement inspection is slow, expensive, inconsistent, and difficult to repeat frequently. Image detectors can reduce manual work, but frame-by-frame video detection introduces a serious error: one physical defect appears in many consecutive frames and can be counted repeatedly. Moving-camera footage also causes scale change, blur, camera shake, occlusion, and rapidly changing perspective.

The project's core research value is therefore not only object detection. It is the integration and evaluation of:

- detection on Indian dashcam footage;
- moving-camera tracking;
- temporal smoothing and missed-frame hold;
- confirmed event aggregation and duplicate control;
- standardized visual-severity estimation;
- reproducible reporting and error analysis.

## 4. Users and use cases

### Primary users

- Student researcher: prepares data, trains models, runs experiments, and analyzes errors.
- Faculty evaluator: verifies methodology, reproducibility, limitations, and measured results.
- Demo user: uploads or selects a video, runs analysis, views counts and timestamps, and downloads outputs.

### Core use case

Given an MP4 dashcam video and a trained model, the user runs one command or uses the dashboard. The system validates the video, processes it, and returns:

- annotated MP4 with stable labels and event IDs;
- one row per unique `DamageEvent` in CSV and JSON;
- summary counts by class and visual severity;
- timestamps for reviewing each event;
- run manifest containing model, configuration, environment, and performance information.

## 5. Scope

### V1 must include

1. Video metadata validation and readable error messages.
2. Configurable frame sampling for dataset preparation.
3. Blur, brightness, and near-duplicate filtering for candidate frames.
4. Four-class object detection:
   - D00: longitudinal crack;
   - D10: transverse crack;
   - D20: alligator crack;
   - D40: pothole.
5. YOLOv8s baseline and YOLO26s proposed-detector experiments.
6. Manually configured trapezoidal road ROI.
7. ByteTrack baseline and BoT-SORT primary-tracker experiments.
8. Exponential moving-average box/confidence smoothing.
9. Short missed-frame display hold to reduce flicker.
10. Track confirmation and `DamageEvent` aggregation.
11. Unique counting based on confirmed events.
12. Three-level relative visual severity: low, medium, high.
13. Timestamped CSV and JSON output.
14. Annotated MP4 output.
15. Streamlit dashboard after CLI completion.
16. Reproducible configs, tests, logs, and experiment manifests.

### Deferred until V1 is complete

- GPS/geospatial maps;
- automatic road segmentation;
- instance or semantic segmentation;
- physical size, depth, or volume estimation;
- patch/repair, edge damage, broken-surface, or miscellaneous classes;
- mobile or edge-device deployment;
- cloud API or MERN frontend;
- real-time performance guarantees.

Deferred items require a written scope change and must not silently enter V1.

## 6. Research questions

| ID | Question | Planned comparison |
|---|---|---|
| RQ1 | Does fine-tuning on custom Indian dashcam frames improve generalization to the teacher-provided video? | Public-only model vs public + custom fine-tuning |
| RQ2 | Does YOLO26s provide a better accuracy/speed trade-off than the literature baseline? | YOLOv8s vs YOLO26s |
| RQ3 | Which tracker better supports a moving dashcam? | ByteTrack vs BoT-SORT |
| RQ4 | How much does event aggregation reduce over-counting? | Frame count vs track count vs confirmed event count |
| RQ5 | Does temporal smoothing improve visual stability without hiding genuine short events? | Raw overlay vs smoothing + hold |

## 7. Functional requirements

### FR-01 Video validation

The system shall accept common video files, read resolution, FPS, frame count, duration, and codec where available, and fail with an actionable error if decoding is not possible.

### FR-02 Dataset frame generation

The system shall extract timestamped candidate frames at a configurable rate without modifying the source video. Each candidate must retain `video_id`, `frame_index`, `timestamp_ms`, and `segment_id`.

### FR-03 Candidate quality filtering

The system shall compute quality metadata including blur score and brightness. It shall support near-duplicate filtering using a deterministic similarity method and preserve a manifest of accepted and rejected candidates with rejection reasons.

### FR-04 Taxonomy enforcement

Training and inference shall use exactly four V1 classes in this order:

```text
0: D00_longitudinal_crack
1: D10_transverse_crack
2: D20_alligator_crack
3: D40_pothole
```

### FR-05 Detection

The detector shall return normalized and pixel bounding boxes, class ID, class name, confidence, frame index, and timestamp for each observation.

### FR-06 Road ROI

The system shall support a normalized polygonal road ROI stored in configuration. Detections shall be accepted or rejected using a documented point rule, initially the bottom-center point of the box.

### FR-07 Tracking

The system shall support selectable tracker configurations. BoT-SORT is primary; ByteTrack is the baseline. Raw tracker IDs shall remain internal observations and shall not automatically equal final damage IDs.

### FR-08 Temporal stabilization

For active tracks, the system shall smooth box coordinates and confidence using configurable exponential moving averages. A configurable short hold may render a predicted/smoothed box across brief missed detections, but held frames must be flagged and must not increase detection-confidence statistics.

### FR-09 Track confirmation

A candidate track shall become confirmed only after meeting validation rules such as minimum observed frames, mean confidence, class consistency, ROI validity, and plausible trajectory. All starting thresholds are provisional and must be tuned using validation clips.

### FR-10 Damage-event aggregation

The system shall aggregate observations into a `DamageEvent`. An event shall be finalized after its track exits, is lost beyond the allowed gap, or the video ends. Final event IDs must be stable within the run and formatted like `RD0001`.

### FR-11 Cross-track duplicate control

The system shall provide a second-stage merge mechanism for fragments believed to represent the same physical defect. The first version may use class agreement, time gap, trajectory/position compatibility, and box appearance statistics. Merge evidence must be recorded for auditability.

### FR-12 Visual severity

Severity shall be calculated only when an event's representative observation intersects a configured measurement zone. The score shall use box area divided by road-ROI area at that zone. Thresholds shall be class-aware where evidence supports it, tuned on validation data, and frozen before test evaluation.

### FR-13 Reporting

The CSV and JSON reports shall contain at minimum:

```text
event_id
class_id
class_name
first_frame
last_frame
representative_frame
first_timestamp_ms
last_timestamp_ms
representative_timestamp_ms
observed_frame_count
mean_confidence
max_confidence
severity_label
severity_score
source_track_ids
roi_valid
```

### FR-14 Annotated video

The output video shall retain the input dimensions and playback rate unless configuration explicitly requests otherwise. Overlays shall show class, event/track identifier, confidence, and severity when available. Held predictions must be visually distinguishable from direct detections.

### FR-15 Dashboard

The dashboard shall display processing status, annotated video, total unique events, class distribution, severity distribution, event table, and downloadable reports. The dashboard shall call the same pipeline used by the CLI rather than duplicate inference logic.

### FR-16 Run manifest

Every evaluation or final-video run shall record configuration, random seed, model checksum/path, software versions, input-video fingerprint, start/end time, device, elapsed time, processed frames, and average processing FPS.

## 8. Non-functional requirements

- Reproducibility: identical inputs, config, and compatible environment should produce equivalent event output.
- Modularity: video, detection, tracking, aggregation, severity, rendering, and reporting must remain separate modules.
- Configurability: thresholds and paths must live in versioned YAML configuration, not hard-coded source.
- Testability: pure logic such as ROI checks, smoothing, event confirmation, severity, and report serialization must have unit tests.
- Recoverability: long-video processing should support periodic checkpoints or resumable intermediate records before the final integration milestone.
- Observability: logs must explain skipped frames, rejected tracks, merges, and failure reasons.
- Privacy: raw footage, number plates, faces, and precise locations must not be published without authorization.
- Resource awareness: training may use a cloud GPU; dataset tools and short-video inference must remain usable locally.

## 9. Corrected design decisions

### Four classes, not eight

The earlier report mixed benchmark categories and proposed custom categories. V1 now uses the four established Road Damage Dataset categories. Extra classes would require enough examples, a precise annotation policy, and a new experiment version.

### Event counting, not ID counting

A tracker ID is an association hypothesis. It can fragment, switch, or reappear. The final count is created by a separate event aggregator with confirmation, finalization, and optional cross-track merging.

### Controlled visual severity

The largest box is not a physical-size estimate because box area grows as the vehicle approaches. The V1 score is taken in a fixed measurement zone and is explicitly named **relative visual severity**.

### Honest tracking evaluation

IDF1, HOTA, MOTA, and ID-switch values require track-level ground truth. They may only be reported on manually identity-labelled clips. When that ground truth is not available, use event-count error, duplicate rate, fragmentation, and manual review instead.

### Leakage-safe data split

Adjacent frames shall never be randomly distributed across train, validation, and test sets. Splits are assigned by temporal segment or route before frame selection and augmentation.

## 10. Provisional algorithm parameters

These values are starting points, not final scientific results:

| Parameter | Initial value | Tuning rule |
|---|---:|---|
| Dataset candidate sampling | 1 frame/second | Increase only in damage-dense clips |
| Detection confidence | 0.25 | Tune on validation precision-recall curve |
| Track confirmation observations | 3 | Compare 3, 5, and 8 |
| Maximum missed-frame hold | 3 frames | Tune for flicker without ghost boxes |
| EMA alpha | 0.6 | Compare 0.4, 0.6, and 0.8 |
| Tracker gap | Config-dependent | Tune per video FPS and processing stride |
| Severity thresholds | Not fixed initially | Derive from labelled validation examples |

## 11. Acceptance criteria

### Data gates

- Every labelled image has provenance and a segment-based split.
- No source segment appears in more than one split.
- A near-duplicate audit finds no prohibited cross-split duplicates.
- At least 10% of annotated frames are purposeful negatives, subject to observed data distribution.
- A second-pass review is completed for all test annotations and a stratified sample of training annotations.

### Pipeline gates

- A clean setup can process a supplied short sample using one documented command.
- The same core command can process the full one-hour video without manual code edits.
- One finalized event produces exactly one CSV row and one JSON event object.
- Invalid video, missing model, and unwritable output paths fail clearly.
- Unit and integration tests pass.

### Initial model targets

Targets are goals, not values to fabricate:

- overall custom-test mAP@0.5 target: at least 0.60;
- custom-test recall target: at least 0.60 overall;
- no V1 class should be hidden by reporting only the macro average;
- if a target is missed, the result remains valid research when accompanied by error analysis and a documented improvement attempt.

### Research success gates

- Event-based counting reduces absolute counting error relative to raw frame-by-frame counting on manually counted evaluation clips.
- Tracker comparison reports accuracy and speed trade-offs using the same detections and clips.
- Public-only and custom-fine-tuned models are evaluated on the same untouched custom test set.
- All final tables are generated from saved experiment artifacts.

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Too few examples for a class | Low recall or unstable training | Inspect video first; merge only through a formally versioned taxonomy change; supplement with licensed public data |
| Adjacent-frame leakage | Inflated metrics | Segment-first split, guard gaps, duplicate audit |
| Shadows/water/manholes cause false positives | Unreliable demo | Hard-negative mining and error-labelled evaluation |
| Track fragmentation | Duplicate events | BoT-SORT, gap tolerance, event-level merge, manual tuning clips |
| Severity changes with perspective | Misleading labels | Fixed measurement zone and relative-only terminology |
| One-hour processing fails late | Lost time | Short smoke tests, checkpoints, free-space checks, deterministic manifests |
| GPU limits | Delayed training | Start with `s`; fall back to `n`; use cloud GPU; keep inference configurable |
| Scope growth | Incomplete project | Enforce V1/deferred boundary and milestone gates |
| Licensing conflict | Cannot publish/deploy as planned | Keep academic repository open under compatible terms or reassess framework licensing before proprietary use |

## 13. Deliverables

- curated dataset manifest and annotation export;
- trained YOLOv8s and YOLO26s checkpoints or reproducible instructions where redistribution is restricted;
- detector evaluation artifacts;
- tracking and event-count evaluation artifacts;
- source code, tests, configuration, and environment lock;
- annotated sample and final video;
- CSV and JSON event reports;
- Streamlit dashboard;
- final dissertation/report using actual results;
- demonstration slides and reproducible demo instructions.

## 14. Change control

Any change to taxonomy, primary models, severity meaning, data split, or success metrics must be recorded in this document before implementation. Experimental branches may test alternatives, but they must not silently redefine the main project.

