# Project Workflow

## 1. Working principle

Build the project as a sequence of verifiable vertical slices. Do not begin full model training or dashboard development until the input video and dataset plan are understood. Each milestone must produce artifacts that the next milestone consumes.

## 2. Branch and change workflow

For each feature:

1. Choose one milestone issue.
2. Create a short-lived branch.
3. Implement the smallest complete behavior.
4. Add tests and example configuration.
5. Run the documented checks.
6. Review generated artifacts, not only terminal success.
7. Merge only when the milestone's definition of done is met.

Suggested branch names:

```text
feat/video-inspection
feat/frame-sampler
feat/event-aggregator
exp/yolo26s-custom-v1
fix/severity-zone-selection
docs/annotation-policy
```

## 3. Phase 0 - Repository and environment

### Tasks

- Create repository structure from `ARCHITECTURE.md`.
- Copy `AGENTS.md` to the repository root and the other files to `docs/`.
- Add `pyproject.toml`, lock strategy, `.gitignore`, base config, and CLI skeleton.
- Add a tiny video fixture or generate one within tests.
- Verify Python, FFmpeg, OpenCV, PyTorch, and device availability.

### Output

- clean install instructions;
- environment check command;
- passing test skeleton;
- no raw data or weights in Git.

### Gate

A new environment can install the project and run the test suite plus `road-damage --help`.

## 4. Phase 1 - Video inspection

### Tasks

- Run metadata inspection on the one-hour dashcam video.
- Generate a low-rate contact sample across the full duration.
- Record camera angle, road ROI, lighting changes, vibration, and privacy risks.
- Estimate actual class presence without yet promising final dataset counts.
- Select representative short clips for development and manual evaluation.

### Output

- `video_inspection.json`;
- contact sheets or sample-frame index;
- proposed ROI and measurement-zone config;
- list of evaluation clips with start/end timestamps.

### Gate

The team can explain what is visible, what classes are feasible, and which clips cover normal, difficult, and negative conditions.

## 5. Phase 2 - Sampling and dataset tooling

### Tasks

- Implement timestamped frame extraction.
- Implement blur/brightness metadata.
- Implement deterministic near-duplicate grouping.
- Build manifest generation.
- Add segment-first split assignment and leakage audit.
- Export selected frames with unchanged provenance.

### Output

- candidate manifest;
- selected-frame manifest;
- train/val/test segment map;
- automated validation report.

### Gate

Re-running with the same configuration and seed gives the same selected frame IDs and split assignments.

## 6. Phase 3 - Pilot annotation

### Tasks

- Create CVAT project with the frozen four classes.
- Annotate 100-200 pilot frames covering all observed conditions.
- Review confusing examples with the teacher if domain judgement is needed.
- Refine only the wording/examples of the annotation policy.
- Annotate the first full batch and export archival + training formats.
- Run label validation.

### Output

- dataset `v0.1.0`;
- annotation review log;
- class and split statistics;
- known-ambiguity list.

### Gate

No unresolved label-policy question affects a large fraction of the dataset, and all validation/test labels have been reviewed.

## 7. Phase 4 - Baseline detector

### Tasks

- Train YOLOv8s with a versioned config.
- Evaluate detection metrics on validation data.
- Perform error analysis by class, scale, condition, and false-positive type.
- Mine high-value hard negatives and missed examples from training/unlabelled segments.
- Add data in a new dataset version rather than modifying an old release.
- Freeze the chosen baseline checkpoint using validation results.

### Output

- baseline run directory;
- weights/checksum;
- metrics, confusion matrix, PR curves, qualitative examples;
- error taxonomy.

### Gate

The model can detect at least some examples of every present class, results are reproducible, and failure modes are understood well enough to guide the proposed-model experiment.

## 8. Phase 5 - Proposed detector

### Tasks

- Train YOLO26s under a matched experiment configuration.
- Use the same dataset version and evaluation code as the baseline.
- Compare accuracy, class-level recall, latency, model size, and resource use.
- Use YOLO26n only if the fallback is documented before test evaluation.
- Freeze the proposed checkpoint based on validation results.

### Output

- proposed-model run directory;
- fair comparison table;
- selected detector with justification.

### Gate

The detector decision is based on measured trade-offs, not novelty alone.

## 9. Phase 6 - Tracking and temporal stabilization

### Tasks

- Build detector-independent tracker adapters.
- Tune ByteTrack and BoT-SORT on validation clips.
- Add bbox/confidence smoothing and missed-frame hold.
- Render side-by-side qualitative clips.
- Manually label identities on a small evaluation subset if formal MOT metrics will be reported.

### Output

- tracker config files;
- association and runtime results;
- flicker and ID-fragment review artifacts.

### Gate

Both trackers run from the same detection input, held observations are distinguished from direct detections, and the comparison can be reproduced.

## 10. Phase 7 - Unique event counting

### Tasks

- Implement track confirmation and rejection.
- Implement finalization at loss/exit/end-of-video.
- Implement conservative cross-track fragment merging.
- Create manual event ground truth for selected clips.
- Compare frame count, raw track count, confirmed-track count, and merged-event count.

### Output

- `DamageEvent` JSON/CSV;
- merge audit log;
- count-error comparison;
- unit and integration tests.

### Gate

Event-based counting lowers error compared with frame counting on the selected labelled clips, or the failure is clearly diagnosed with evidence.

## 11. Phase 8 - Visual severity

### Tasks

- Calibrate road ROI and measurement zone.
- Select representative observations consistently.
- Label a validation subset as low/medium/high with an explicit visual rubric.
- Tune class-aware thresholds only on validation examples.
- Freeze thresholds and evaluate agreement on the test subset.

### Output

- severity config;
- labelled calibration set;
- severity confusion matrix/agreement report;
- limitation statement.

### Gate

No output or UI calls the estimate physical severity, depth, or physical area.

## 12. Phase 9 - Reporting and full-video pipeline

### Tasks

- Generate annotated MP4, CSV, JSON, summary, and run manifest.
- Add checkpointing and end-of-video finalization.
- Test on a short clip, then a medium clip, then the full hour.
- Verify playback duration, FPS, resolution, row counts, timestamp seek points, and disk use.

### Output

- final processed video;
- final event reports;
- processing-performance profile;
- failure/recovery log if applicable.

### Gate

The full video completes without code edits and every reported event can be reviewed at its representative timestamp.

## 13. Phase 10 - Dashboard

### Tasks

- Wrap the existing application service in Streamlit.
- Support validated video/model/config selection.
- Show progress and safe errors.
- Display summary cards, plots, video, and event table.
- Add report downloads.

### Gate

CLI and dashboard produce equivalent event data for the same model/config/input.

## 14. Phase 11 - Final evaluation and academic outputs

### Tasks

- Run the frozen test protocol once for final tables.
- Generate figures from saved artifacts.
- Write limitations and threats to validity.
- Prepare report, slides, demo script, and reproducibility instructions.
- Archive configs, manifests, code commit, and result checksums.

### Gate

Every number in the dissertation can be traced to a saved run artifact.

## 15. Standard run-directory structure

```text
experiments/<run_id>/
  resolved_config.yaml
  run_manifest.json
  environment.txt
  metrics.json
  logs/
  plots/
  predictions/
  qualitative/
  weights/
```

Use an informative ID such as:

```text
2026-09-15_yolo26s_custom-v01_seed42
```

## 16. Pull-request checklist

- PRD requirement/milestone linked.
- No V1 scope expansion.
- Tests added or reason documented.
- Targeted checks pass.
- Config and docs match behavior.
- No raw/private data, large outputs, weights, or secrets added.
- No invented metrics or unverified performance claim.
- Generated artifacts manually inspected where relevant.

