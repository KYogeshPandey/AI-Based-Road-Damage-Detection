# Frozen baseline: first/final internal-test evaluation

Implements the detector evaluation milestone, PRD FR-04/FR-16 and the
Evaluation Plan. No training, resume, threshold sweep, model selection,
teacher-video evaluation, or canonical dataset edits are authorized by this
tool. The earlier image/video demos and historical training records are unchanged.

## Frozen record and review gate

The JSON-compatible YAML record at
configs/evaluation/baseline_public_v1_frozen_operating_point.yaml preserves:

- Confidence 0.19; class-aware NMS IoU 0.50; operating-point matching IoU 0.50.
- Validation-only selection commit 027f5f41c386f558319799fa8784b32cf3dcefdb.
- Validation macro-F1 0.5055312316193121, NOT an internal-test result.
- Approved baseline model:
  outputs/training/baseline_public_v1/20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42/weights/best.pt.
- Model size 22,524,074 bytes; SHA-256
  BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7.
- Exact ordered classes: 0 D00_longitudinal_crack, 1 D10_transverse_crack,
  2 D20_alligator_crack, 3 D40_pothole.
- Selection timestamps, selection-artifact hashes, approved export metadata
  hashes and expected support counts.

An independent code contract rejects config edits, unknown keys, alternate
models/paths, per-class thresholds and CLI confidence/NMS/matching overrides.
Schema v2 records two fixed branches and identity-locked technical recovery;
it does not change any scientific operating-point value.

The user must review and commit all five internal-test tooling files before
real preflight or evaluation. A clean descendant of the selection commit is
required. Historical training/hotfix commits and fingerprints remain separate;
do not rewrite history or regenerate the original training source manifest.

## Future commands (PowerShell, project root)

After implementation review and the new commit, the user runs metadata-only
preflight first:

~~~powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\internal_test.py --preflight
~~~

After successful preflight and explicit approval, the user alone runs:

~~~powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\internal_test.py --evaluate-once --acknowledge-first-final-test
~~~

The acknowledgement attests that no completed baseline internal-test evaluation
preceded this scientific execution. The pinned validation-only record and local
ledger support that assertion; they are not forensic proof about other tools.

Only after a purely technical interruption, with every identity unchanged:

~~~powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\internal_test.py --evaluate-once --acknowledge-first-final-test --recover-technical
~~~

Recovery is not permission to inspect partial results and retune, choose another
checkpoint, edit the code/environment/config, or change the dataset. Do not
delete receipts, lock files or prior attempts to obtain another evaluation.

## Metadata-only preflight

Preflight never opens or hashes internal-test image bytes, reads test label
files, inspects image headers, enumerates test directories, constructs a model,
or invokes a framework dataset loader. It reads only:

- Frozen config, pinned approved export manifests/reports and selection artifacts.
- Exact local model path, size and SHA-256.
- Clean Git state, committed evaluation-source hashes and historical provenance.
- Existing Python 3.12.10, Ultralytics 8.4.130, package/CUDA/device identity.
- Installed Ultralytics Python/YAML implementation hashes.
- Output/attempt metadata and availability.

For a fresh output, an automatically removed temporary file probes write
permission in an existing output ancestor. No evaluation output/receipt or
test pair is created/read by preflight.

Expected support is derived from approved metadata, not from test files:

| Scope | Images | Positive | Negative |
|---|---:|---:|---:|
| Overall | 2,775 | 1,712 | 1,063 |
| India | 1,116 | 459 | 657 |
| Japan | 1,659 | 1,253 | 406 |

Targets: D00 841, D10 557, D20 1,297, D40 840; total 3,535.
India D10: LOW SUPPORT — 10 boxes / 9 images. No strong India-specific D10
conclusion is justified.

Only the explicitly acknowledged evaluation, after durable STARTED publication,
reads actual manifest-allowlisted pairs. It verifies exact pairing, image/label
hashes, dimensions, classes, empty negatives and every expected support count
BEFORE any model execution. Official unlabelled-test and teacher paths are
rejected using approved source provenance.

The real inventory function requires an opaque process-local capability as its
first gate. The evaluation workflow mints that object only after the receipt and
numbered STARTED marker have both been durably published. Preflight has no code
path that creates, receives or returns this capability. A missing or forged
object is rejected before path discovery, enumeration, stat/header inspection,
hashing or opening an image/label. The capability stays bound to the exact root,
config, identity and current STARTED attempt, and is cleared on every exit.

## Two fixed branches inside one scientific evaluation

Both branches use the same frozen best.pt, image size 640, square letterboxing
(rect=False), batch 1, CUDA 0, FP32, no augmentation, class-aware NMS 0.50,
max_det 300, seed 42 and deterministic PyTorch settings. Reaching the detection
cap blocks completion rather than silently accepting truncated statistics.

Verified pairs are byte-copied into the attempt's inputs directory before
execution. No source is moved, hardlinked, resized or relabelled. The native
validator may create its own label cache only beside these disposable copies.
Snapshot/source hashes are checked again before completion; any framework repair
or content change blocks acceptance. Budget disk space for one full byte-copy
per technical attempt.

The generated snapshot YAML has train/val/test aliases all pointing to that
same internal-test snapshot solely to satisfy the framework's dataset schema.
The validator explicitly uses split=test; it cannot traverse canonical train
or validation data through those aliases. Dataset/font automatic downloads are
disabled by a narrow I/O adapter; metric and postprocessing code are unchanged.

### Branch A: actual standard detector evaluation

The real installed Ultralytics 8.4.130 DetectionValidator executes the normal
model.val-compatible evaluation path, not model.predict. Its native
postprocessing uses multi_label=True at confidence floor 0.001 and NMS 0.50.
The floor is for standard AP integration, NOT a deployment confidence choice.

Native preprocessing, inference-image coordinate matching (before prediction
clipping), IoU 0.50:0.95 matching, and DetMetrics produce mAP50, mAP50-95,
per-class AP and standard precision/recall. Exact requested and resolved
validator settings are saved, including FP32 quantize=None, workers=0 and
disabled auxiliary export/plot/cache options.

Standard P/R are explicitly labelled "test-curve-derived descriptive
statistics": the framework reports them at its smoothed mean-F1 confidence-curve
index. That index is NOT a deployment threshold, is not applied to Branch B,
and changes no model/configuration/checkpoint. These P/R are NOT the frozen
deployment P/R.

Audit hooks capture native predictions, transformed ground truth, image shape,
letterbox transform and actual native per-prediction IoU TP masks before calling
the unchanged framework metric update. Native prediction coordinates remain
unclipped, including padding/out-of-frame predictions. The cache is not a
prediction-mode approximation of validator output.

### Branch B: primary frozen deployment operating point

A separate normal prediction-mode pass runs directly at confidence 0.19 and
NMS 0.50. This intentionally preserves prediction-mode single-label behavior
and original-image clipping. It does not reuse Branch A's post-NMS output.

The unchanged validation-threshold module's class-aware one-to-one matcher
computes only the singleton frozen point, using confidence >=0.19 and IoU >=0.50.
Prediction ties: confidence descending, then class ID and xyxy ascending.
Ground-truth ties: highest IoU, then xyxy ascending and stable input index.

Report per-class TP/FP/FN/P/R/F1, four-class macro-F1, totals and micro P/R/F1.
Negatives mean zero retained D00/D10/D20/D40 ground-truth objects. Report negative
image count, negative FP, FP/negative image, images with >=1 FP and that fraction.
India/Japan reports show support before performance and retain the India D10
low-support warning. Zero denominators produce 0, not evidence of success.

The diagnostic confusion plot uses native class-agnostic spatial matching
with IoU >0.50 on predictions retained at >=0.19. It is labelled separately:
its off-diagonal matching and totals need not match the primary class-aware
IoU >=0.50 metrics.

Two fixed passes are intentional: validator multi-label and coordinate
semantics differ from ordinary prediction mode. No threshold sweep is performed.

## Durable technical recovery and API gates

The receipt at outputs/evaluation/baseline_public_v1_internal_test_attempt.json
contains the immutable identity and append-preserved numbered attempt history:

1. STARTED is durably flushed before any test content access or inference.
2. Keyboard interruption or a technical I/O/runtime failure records
   FAILED_TECHNICAL and preserves all partial artifacts.
3. Explicit recovery acquires the exclusive nonblocking OS lock and requires
   exact identity equality, then creates a new numbered attempt.
4. A dead process may leave STARTED. Once its OS lock is released, identical
   explicit recovery records the prior attempt as FAILED_TECHNICAL.
5. Scientific/integrity failures remain incomplete with INCOMPLETE.json and
   are not eligible for technical recovery.
6. COMPLETED permanently blocks further baseline evaluation.

Recovery must match model path/SHA/size, project root, Git commit, complete frozen
config hash, confidence/NMS/matching IoU, ordered classes, evaluation-source
identity, package/CUDA/device and framework implementation identity, dataset
metadata/record hashes and validation-selection provenance. Any difference
blocks recovery. No --force, alternate output, arbitrary model/config, or
scientific override exists.

Every field and nested leaf in the canonical recovery identity is critical;
there are no informational exceptions. Production validates the exact top-level
identity schema when constructing it, and regression tests derive their mutation
matrix by recursively walking that canonical record. This prevents a handwritten
test list from silently falling behind when a nested provenance field is added.

The sibling .lock file is held by the OS for the whole attempt; process death
releases ownership. Do not remove the file. A live attempt cannot be recovered
concurrently. An unrecognized existing output or malformed/legacy ledger fails
closed.

The only supported public inference entry point is run_evaluation, with explicit
acknowledgement and optional identical-identity technical recovery. Both private
inference branches require the exact active process capability, durable STARTED
attempt, current frozen identity/model/config/dataset snapshot and authorized
output. An alternate project root is rejected so a caller cannot evade the
ledger by redirecting the same run to another directory. Caller-supplied models
or scientific settings are not accepted. Pure
metric functions remain reusable without inference.

These are application safeguards, not protection against intentional Python
monkeypatching or deletion of every local ledger. Preserve/back up the evidence.

## Persisted caches, reports and atomic completion

Each branch writes one JSONL record for every image, including empty predictions.
Common fields include schema/branch/identity SHA, image ID, country, NMS IoU,
confidence floor, multi-label mode and class/confidence/xyxy detections.
Branch A additionally saves native geometry/targets and native TP masks.

Bytes are incrementally hashed while writing, flushed, fsynced and closed before
an independent SHA/count/identity seal is published. Finalization reopens both
caches and validates their schema, hashes, exact expected identities, unique
coverage of all 2,775 images, counts, classes, scores and finite valid geometry.
Truncated, corrupted, missing or duplicate records block completion.

Branch A replays the saved native TP masks/confidences through the installed
DetMetrics implementation and requires exact JSON numeric reconciliation with
the real validator's live metrics. Saving native masks avoids silently changing
borderline GPU float32 IoUs during CPU replay. Framework traversal and prediction
order, including confidence ties, are preserved. Branch B metrics are computed
from the verified persisted prediction records with the approved matcher.
Identical verified cache input gives identical numeric metric results; no GPU
pass is needed for audit recomputation.

After reporting, caches, canonical pairs, copies and provenance are rechecked.
All six required reports must exist and be nonempty, and important artifacts
must be flushed to storage and hashed before the attempt completion manifest is atomically
published. Until then partial reports are staging artifacts, not final results.

~~~text
outputs/evaluation/baseline_public_v1_internal_test/
    identity.json
    evaluation_manifest.json           # published only after full completion
    attempts/
        0001/
            STARTED.json
            FAILED_TECHNICAL.json       # only for technical failure
            INCOMPLETE.json             # only for non-recoverable failure
            COMPLETED.json              # only after successful finalization
            evaluation_manifest.json
            ground_truth_cache.json
            standard_predictions.jsonl
            standard_predictions.seal.json
            frozen_predictions.jsonl
            frozen_predictions.seal.json
            standard_live_metrics.json
            validator_settings.json
            prediction_settings.json
            inputs/
            framework_validator/
            reports/
                internal_test_metrics.json
                internal_test_operating_point.json
                internal_test_summary.md
                confusion_matrix.png
                per_class_metrics.csv
                country_metrics.csv
~~~

A completed attempt manifest/marker, final root manifest, or COMPLETED receipt
independently locks further evaluation. If completion succeeds but later root
publication/receipt updating fails, the completed attempt is never downgraded
or rerun. Preserve it and review the publication failure.

Manifests record separate standard/frozen branch durations, offline verification
and reporting duration, total evaluation duration, provenance and artifact hashes.
Progress logs show processed/total images, elapsed time and ETA. These durations
are not application FPS. Following a real evaluation, the internal test must
no longer be described as untouched.

## Implementation verification only

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_internal_test.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

Tests use temporary synthetic images/labels/manifests, mocked model execution
and installed CPU validator/postprocessing/metric functions. No baseline model
is loaded, no model forward pass is performed, and actual internal-test pair
access is forbidden. Metadata-only preflight tests additionally deny synthetic
test image/label reads, header inspection and directory enumeration while
exercising the real preflight logic against mock authority metadata.

Implementation and tests do not authorize real preflight, internal-test access,
inference, training, validation threshold selection, teacher-video use or a Git
commit. Stop after reporting the implementation/test results.
