# Validation-only operating-threshold selection

This is post-training evaluation tooling for PRD FR-04/FR-05 and Evaluation Plan section 10. Review it and create its own Git commit before running preflight or the real sweep. The tool requires a clean descendant of mentor-demo commit `702a916e88af54504feb1f25b5f8af8ffef8621e`. It verifies the historical training commits/fingerprints and records a separate evaluation source fingerprint; it never regenerates the training source manifest or rewrites history.

From `E:\AI_Road_Damage_Project`, using the existing Python 3.12.10 environment:

```powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\select_threshold.py --preflight
```

Preflight checks the exact frozen best.pt path, 22,524,074-byte size and SHA-256; the existing train/val YAML, exact class mapping and absence of any test key; approved validation paths and redirects; 2,602 paired images, 3,377 label targets, 983 exactly-empty negative labels; CUDA device 0; dependencies; output write permission; and clean post-training Git provenance. Counts are tallied directly from approved validation labels rather than assumed from metadata. It hashes validation images/labels and reads image dimensions but never constructs YOLO or runs inference. It leaves no result directory. During implementation a dirty worktree is expected to fail the commit gate.

After review, commit, and successful preflight, the USER runs:

```powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\select_threshold.py
```

The sweep repeats preflight, reserves `outputs/evaluation/baseline_public_v1_threshold_selection/`, and refuses any existing output directory. A failed run remains marked failed, with available caches retained for diagnosis; it is not a completed selection and is not automatically resumed or overwritten.

## Algorithm

- Load only the frozen best.pt, verify its class names, and use CUDA device 0, imgsz 640, full precision, batch 1, square letterboxing, class-aware NMS, no augmentation/TTA, and seed 42. Framework dataset discovery is not called. Only approved validation images are passed as decoded arrays; training, internal test, official test, and teacher video are never enumerated or read.
- For each NMS IoU in `[0.50, 0.60, 0.70]`, collect predictions at **conf=0.01**. Save every image identifier, including empty predictions, and all retained class IDs/confidences/continuous xyxy pixel boxes plus NMS IoU. The maximum detections is 300; reaching that cap aborts selection rather than silently accepting truncated predictions. Framework internal NMS limits and numerical GPU behavior still apply.
- For each image, order predictions by confidence descending, then class ID and xyxy coordinates ascending. Match only to the same class, greedily choosing the unmatched GT with highest IoU >=0.50. Equal IoU uses GT xyxy ascending, then stable index for identical boxes. Identical boxes are interchangeable for aggregate counts. Coordinates use continuous xyxy (no VOC +1); only <=0.0001-pixel export edge rounding is clamped and original label SHA is preserved.
- Sweep the **single global confidence grid 0.01–0.80 inclusive, step 0.01**, retaining confidence >= threshold. Matching once at the floor and retaining confidence prefixes is equivalent to rerunning greedy matching for every threshold: lower-confidence predictions cannot alter earlier matches. Each GT/prediction participates in at most one match.
- Per class: precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Every zero denominator yields zero. Macro-F1 is the mean of **all four** class F1 values, including absent classes. Exact rational F1 is used to rank candidates, avoiding rounding-dependent ties. This objective is not mAP or micro-F1.
- Select highest macro-F1, then fewest total FP on negative images, then highest confidence. A complete remaining tie uses the declared NMS order. Record the number of candidates surviving every criterion and whether the fallback was needed.
- For the 983 zero-target images, all retained predictions are FP. Report total negative predictions/FP, FP/983, count with >=1 FP, and that count/983 (also percent in the summary).

## Artifacts and replay

The real user-run sweep creates `ground_truth_cache.json`, `predictions_nms_50.json`, `predictions_nms_60.json`, `predictions_nms_70.json`, `threshold_sweep.csv` (240 operating points), `selected_operating_point.json`, `evaluation_manifest.json`, `threshold_selection_summary.md`, and `macro_f1_vs_confidence.png`.

The manifest records SHA-256 for each GT/prediction cache and result artifact, frozen model identity, configuration/YAML hashes, image/label hashes, source fingerprint, evaluation Git commit, preserved training provenance, package/CUDA/GPU versions, collection duration per NMS candidate, offline sweep duration, and total evaluation duration. Collection progress reports candidate number, X/2602 images, elapsed time, and ETA. Collection duration includes decoding/hashing/forward pass/NMS; it is not application FPS. Total duration additionally includes preflight, integrity checks, and reports.

To recompute from a completed run without opening any model or dataset:

```powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\select_threshold.py --recompute-cache
```

Replay verifies cache hashes and uses the original run identity/timestamp so `selected_operating_point.json` must match byte-for-byte. It writes to the separate `outputs/evaluation/baseline_public_v1_threshold_selection_recomputed/`, refusing overwrite. Replay leaves the original artifacts unchanged. A new code change that changes metrics will fail the comparison instead of silently replacing the selected point. Repeated metric calculations from identical caches are deterministic; independent GPU inference runs are not promised byte-identical across hardware/framework versions.

No real sweep or model inference is performed as part of implementing or testing this tool. Synthetic unit tests use generated images and a mocked model in temporary folders. The mentor demo remains at its explicitly provisional thresholds until a separately approved change adopts measured selection results. Internal test remains untouched throughout this phase.
