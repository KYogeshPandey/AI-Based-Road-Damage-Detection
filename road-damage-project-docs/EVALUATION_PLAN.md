# Experiment and Evaluation Plan

## 1. Goal

Evaluate detection, association, unique counting, temporal stability, visual severity, and system performance separately. A single mAP value cannot establish that the video system works.

## 2. Evaluation datasets

### Detection test set

Untouched custom frames assigned by source segment. Report public-dataset results separately from custom-domain results.

### Tracking subset

Several short, representative clips with manually assigned persistent identities. Include:

- low and high damage density;
- smooth and shaky camera motion;
- near and distant defects;
- temporary occlusion;
- confusing negatives;
- at least one clip with neighbouring defects.

### Event-count subset

Clips manually reviewed for the number and class of physical damage events. An event annotation must include first visible time, last visible time, representative time, class, and a stable ground-truth ID.

### Severity subset

Events independently labelled using the project's relative visual rubric. If possible, use two reviewers and adjudicate disagreements.

## 3. Detection metrics

Report:

- precision;
- recall;
- F1 at the selected operating threshold;
- AP per class;
- mAP@0.5;
- mAP@0.5:0.95;
- confusion matrix;
- false positives per 1,000 evaluated frames;
- inference time/FPS on named hardware.

Break results down by class and useful conditions such as near/far, bright/shadow, dry/wet, and blur when sample sizes allow.

## 4. Tracking metrics

Only report IDF1, HOTA, MOTA, or ID switches when track-level ground truth exists. The primary project-oriented measures are:

```text
track fragmentation rate = ground-truth events represented by >1 predicted track / matched ground-truth events

track merge rate = predicted tracks that combine >1 ground-truth event / matched predicted tracks

event coverage = matched ground-truth events / total ground-truth events
```

Also report processing FPS and average active tracks.

## 5. Counting metrics

For each clip and class:

```text
count_error = predicted_unique_count - ground_truth_unique_count

absolute_count_error = abs(count_error)

relative_count_error = abs(predicted - truth) / max(truth, 1)

duplicate_rate = duplicate_predicted_events / max(predicted_events, 1)

miss_rate = missed_ground_truth_events / max(ground_truth_events, 1)
```

Across clips, report MAE and totals. Do not use accuracy alone when the true count is small.

Compare four methods:

1. sum of frame detections;
2. number of raw tracker IDs;
3. confirmed tracks;
4. finalized/merged `DamageEvent` count.

## 6. Temporal stability metrics

Possible measurable indicators:

- visible flicker transitions per matched event;
- average number of one-frame disappearances;
- bbox center/size jitter on continuously visible tracks;
- ghost duration after the ground-truth event disappears.

The smoothing experiment must verify both benefit and cost. A display that never flickers but leaves false boxes on screen is not an improvement.

## 7. Severity metrics

Because severity is categorical and relative, report:

- confusion matrix;
- macro F1 or balanced accuracy;
- weighted Cohen's kappa when sample size permits;
- percentage labelled `unknown`;
- agreement before and after reviewer adjudication, if multiple reviewers participate.

Do not evaluate physical depth or dimensions without suitable sensors/calibration and ground truth.

## 8. System metrics

- video duration processed;
- total wall time;
- average processing FPS;
- decode, inference, tracker, render, and encode time where available;
- peak RAM and GPU memory;
- output size;
- failure/recovery behavior;
- input/output duration difference;
- successful timestamp seek verification rate.

## 9. Required experiments

### E1 - Domain adaptation

| Variable | Public-only | Public + custom fine-tune |
|---|---|---|
| Detector architecture | Same | Same |
| Custom test set | Same untouched set | Same untouched set |
| Purpose | Domain baseline | Measure benefit of local data |

### E2 - Detector comparison

YOLOv8s vs YOLO26s using the same dataset version, split, image size, metric code, and named hardware. Report both quality and speed.

### E3 - Tracker comparison

ByteTrack vs BoT-SORT using cached detections from the same detector whenever possible, so the experiment isolates association differences.

### E4 - Temporal stabilization

No smoothing/hold vs EMA + hold. Compare flicker, jitter, ghost duration, and viewer examples.

### E5 - Counting pipeline

Frame count vs raw tracks vs confirmed tracks vs `DamageEvent` aggregation.

### Optional E6 - ROI ablation

No ROI vs manual ROI, measuring false positives and missed true defects.

## 10. Hyperparameter protocol

- Tune only on validation clips/frames.
- Maintain a small, declared search space.
- Select one operating point per final system.
- Freeze all thresholds before test evaluation.
- Save every trial's resolved configuration and results.
- Do not repeatedly inspect test failures and then present the same test set as untouched.

## 11. Statistical and reporting cautions

- Report the number of images, boxes, clips, and ground-truth events behind every table.
- Use confidence intervals or bootstrap intervals when enough independent clips exist.
- Treat adjacent frames as correlated; do not pretend each frame is an independent experimental sample.
- Keep exploratory, validation, and final-test tables visibly separate.
- Report negative or inconclusive results.

## 12. Result table templates

### Detection

| Model | Training data | mAP50 | mAP50-95 | Precision | Recall | FPS | Hardware |
|---|---|---:|---:|---:|---:|---:|---|
| YOLOv8s | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLO26s | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Per-class detection

| Model | D00 AP50 | D10 AP50 | D20 AP50 | D40 AP50 |
|---|---:|---:|---:|---:|
| YOLOv8s | TBD | TBD | TBD | TBD |
| YOLO26s | TBD | TBD | TBD | TBD |

### Tracking and count

| Tracker/system | Fragment rate | Merge rate | ID switches* | Count MAE | Duplicate rate | FPS |
|---|---:|---:|---:|---:|---:|---:|
| ByteTrack raw IDs | TBD | TBD | TBD | TBD | TBD | TBD |
| BoT-SORT raw IDs | TBD | TBD | TBD | TBD | TBD | TBD |
| BoT-SORT + events | TBD | TBD | TBD | TBD | TBD | TBD |

`*` Only with identity ground truth.

### Severity

| Method | Labelled events | Balanced accuracy | Macro F1 | Kappa | Unknown rate |
|---|---:|---:|---:|---:|---:|
| Measurement-zone boxes | TBD | TBD | TBD | TBD | TBD |

## 13. Error-analysis taxonomy

Tag failures using consistent categories:

```text
FP_shadow
FP_water
FP_manhole
FP_patch
FP_marking
FN_small_or_distant
FN_blur
FN_occlusion
FN_low_contrast
CLASS_D00_D10
CLASS_D00_D20
TRACK_fragment
TRACK_merge
EVENT_duplicate
EVENT_miss
SEVERITY_perspective
```

Examples should be saved by tag so the final report can explain not only how often the system fails, but why.

## 14. Reproducibility checklist

- dataset version and manifest checksum recorded;
- split map frozen;
- code commit recorded;
- exact package versions recorded;
- model/config checksum recorded;
- seed and hardware recorded;
- test command stored;
- metrics generated from prediction artifacts;
- plots generated by scripts, not edited values;
- final artifact checksums saved.

