# YOLOv8s baseline validation-only error analysis

This phase explains the frozen public-data YOLOv8s baseline using only the
completed validation threshold-selection caches. It does not train a model,
run inference, read canonical image/label bytes, or inspect internal-test
examples. The completed internal-test result remains a locked aggregate record.

## Fixed inputs

The configuration at
`configs/evaluation/baseline_public_v1_error_analysis.yaml` pins SHA-256 values
for exactly four files under
`outputs/evaluation/baseline_public_v1_threshold_selection/`:

- `evaluation_manifest.json`
- `ground_truth_cache.json`
- `predictions_nms_50.json`
- `selected_operating_point.json`

The loader rejects changed hashes, incomplete/non-validation provenance,
noncanonical validation paths, duplicate/missing image identities, malformed
boxes, altered model identity, or frozen-point drift. It reproduces the selected
validation metrics byte-for-byte before analysis. It does not discover or open
the image/label paths recorded in the ground-truth cache.

Frozen values remain confidence 0.19, NMS IoU 0.50 and class-aware matching IoU
0.50. Predictions below 0.19 remain available only to diagnose a miss as a
below-frozen-confidence candidate; they do not enter fixed-point metrics and do
not support threshold retuning.

## Deterministic diagnostics

Primary TP/FP/FN use an indexed implementation of the already-approved matcher:
prediction confidence descending, class ID and xyxy ascending; eligible ground
truth by IoU descending, xyxy and stable index; class-aware one-to-one matching
at IoU >=0.50.

After primary matching, residual errors receive secondary diagnostic pairings:

- `class_confusion`: wrong-class prediction/GT overlap at IoU >=0.50.
- `poor_localization`: same-class overlap from IoU 0.10 to <0.50.
- `below_frozen_confidence`: same-class overlap at IoU >=0.50 from the pinned
  cache, but prediction confidence is below 0.19.
- `no_matching_candidate`: no stronger cached geometric explanation.

These diagnostic pairings never change primary TP/FP/FN. Visual explanations
such as shadow, water, road marking, manhole, patch, blur, occlusion or low
contrast require later manual review of the ranked validation examples; the
cache tool does not invent those labels.

High-confidence FP means confidence >=0.50. A false negative has no intrinsic
confidence. A separate diagnostic count/list identifies FNs paired with a
high-confidence wrong-class or poorly localized prediction.

Object size is normalized bounding-box footprint (`bbox area / image area`):

- small: <1%
- medium: 1% to <5%
- large: >=5%

This is not physical damage area or engineering severity. Every class/bucket
reports GT support before recall; support below 30 boxes is explicitly weak.

Country is derived from the allowlisted `India_*.jpg` / `Japan_*.jpg` validation
identifier. Country tables report image, positive-image and target support before
metrics. India D10 remains explicitly marked LOW SUPPORT.

## Command and output policy

From the project root, run once:

```powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\error_analysis.py
```

The command refuses to overwrite
`outputs/evaluation/baseline_public_v1_error_analysis/`. It writes:

- `error_analysis_summary.md`
- `per_class_errors.csv`
- `country_errors.csv`
- `false_negatives.csv`
- `false_positives.csv`
- `localization_errors.csv`
- `negative_image_false_positives.csv`
- `object_size_metrics.csv`
- `iou_distribution.csv`
- `confidence_distributions.csv`
- `class_confusion_matrix.csv`
- `ranked_examples.csv`
- `error_analysis_manifest.json`

The ranked file contains deterministic manual-review queues for every class's
false negatives, high-confidence FPs, India/Japan errors, negative-road FPs,
class-confusion candidates, localization candidates and high-confidence FN
diagnostics. It records validation paths but does not read or copy pixels.

The manifest records all pinned input hashes, frozen values, model SHA, source
selection commit, current Git state, code/config hashes, computed counts and
every generated artifact hash. It explicitly records no inference, no threshold
tuning, no teacher-video use and no internal-test image/label access.

## Interpretation and next experiment

All conclusions and Experiment 2 recommendations must come from the training
design, validation caches, these validation diagnostics and general model-design
reasoning. Individual internal-test examples must never be used for development.
The planned next controlled experiment remains pretrained YOLO26s under the
matched YOLOv8s protocol unless validation evidence establishes a stronger
single change. This runbook does not authorize that training.
