# Dataset and Annotation Guidelines

## 1. Purpose

This document defines how the one-hour teacher-provided dashcam video and public road-damage datasets must be converted into a defensible dataset. It prevents annotation inconsistency, temporal leakage, duplicate-heavy training data, and misleading evaluation.

## 2. Fixed V1 taxonomy

| Class ID | Code | Label | Operational definition |
|---:|---|---|---|
| 0 | D00 | Longitudinal crack | Linear crack predominantly parallel to the road/travel direction |
| 1 | D10 | Transverse crack | Linear crack predominantly across/perpendicular to the road/travel direction |
| 2 | D20 | Alligator crack | Interconnected fatigue-crack network forming polygon/block patterns |
| 3 | D40 | Pothole | Localized loss/depression of pavement with visible broken boundary or depth/texture cues |

Do not label repair patches, manholes, water, shadows, lane markings, edge erosion, broken surface, or ambiguous discoloration as one of these classes unless the object clearly satisfies the chosen definition.

## 3. Before sampling

Create a video-inspection record containing:

- source filename and cryptographic hash;
- owner/permission note;
- duration, FPS, frame count, resolution, codec, and orientation;
- day/night, weather, camera height/angle if known;
- estimated road-visible area;
- blur/vibration and windshield obstruction notes;
- approximate occurrence of each V1 class;
- privacy risks such as faces, plates, homes, or GPS overlays.

Do not decide final ROI coordinates or sampling density until the video has been inspected.

## 4. Segment-first sampling strategy

At 30 FPS, a one-hour video has about 108,000 frames. Do not annotate every frame.

### Stage A: candidate pool

- Start at approximately 1 frame/second: about 3,600 candidates.
- Add denser sampling only around damage-rich or difficult intervals.
- Assign each frame to a source segment before filtering. A useful starting segment duration is 10-30 seconds.
- Preserve a manifest row even when a frame is rejected.

### Stage B: automated filtering

Record:

- Laplacian-based blur score or equivalent;
- mean/percentile brightness;
- perceptual hash or embedding similarity;
- segment ID and nearest accepted frame;
- accepted/rejected status and reason.

Filtering should remove unusable and redundant frames, not systematically remove hard but valid conditions. Keep some motion blur, shadow, low-light, and occlusion examples when the defect remains labelable.

### Stage C: first annotation batch

- Target roughly 600-1,000 diverse frames for the first model.
- Include positives across class, scale, lighting, road texture, and traffic conditions.
- Include intentional negatives.
- Train a first model, mine its errors, and then expand toward about 1,200-2,000 high-quality frames only if evidence justifies it.

These are planning ranges, not mandatory dataset statistics.

## 5. Split policy

### Prohibited

- random image-level splitting of adjacent video frames;
- choosing the best-looking test segment after seeing model results;
- augmentation before split assignment;
- frames from the same short source segment in multiple splits.

### Required

1. Define source segments or route groups first.
2. Assign each complete group to `train`, `val`, or `test`.
3. Insert guard intervals between neighbouring split blocks when using one continuous video.
4. Perform frame selection and augmentation within each split.
5. Freeze the test split before model development.
6. Audit cross-split near duplicates.

Preferred split: different recordings/routes for train, validation, and test. If only one video exists, use separated temporal/route blocks and document the weaker generalization claim.

A 70/15/15 allocation is a starting ratio, not a substitute for class coverage. The test split must contain enough examples to evaluate each reported class; otherwise the limitation must be stated.

## 6. Bounding-box policy

- Draw the tightest box that contains the visible damage without excessive road background.
- Use one box for one visually continuous physical defect.
- Label a partially visible defect only when its class is still identifiable; mark difficult/occluded status in the archival format if supported.
- Do not infer hidden extent outside the image.
- Avoid multiple boxes for fragments that clearly form one continuous alligator-crack region.
- Keep clearly separate defects as separate instances even if their boxes are close.
- When longitudinal/transverse orientation is ambiguous because of road curvature or perspective, flag for review rather than guessing.
- If an example remains unresolved after review, exclude it from the supervised dataset and record the reason.

## 7. Class-specific rules

### D00 longitudinal crack

- The dominant crack direction follows the road direction in the local perspective.
- Do not use D00 for lane markings, tar joints, shadows, or road edges.
- A visible gap may separate instances; define a consistent maximum gap during the pilot annotation review.

### D10 transverse crack

- The dominant direction crosses the road direction.
- Separate multiple distinct transverse cracks when visible pavement exists between them.
- Do not label speed-breaker edges or painted lines.

### D20 alligator crack

- Requires an interconnected network rather than a single isolated line.
- Box the coherent network region.
- Do not label normal block paving or patterned repair as alligator damage.

### D40 pothole

- Require pavement loss/depression cues, broken edges, or a clearly irregular damaged cavity.
- Exclude water puddles, manhole covers, dark repairs, shadows, loose soil, and stains unless a pothole is independently visible.
- If water hides whether a pothole exists, treat the image as ambiguous/hard negative rather than inventing a label.

## 8. Hard-negative policy

Purposefully include frames containing:

- water and reflections;
- shadows from trees, poles, vehicles, and flyovers;
- manholes and drain covers;
- patches and resurfaced asphalt;
- lane markings, arrows, and zebra crossings;
- speed breakers;
- tyre marks, oil stains, and dark aggregate;
- gravel, mud, leaves, and roadside debris;
- vehicle parts/windshield artifacts;
- intact textured or block-paved roads.

Negative images contain no label files or valid empty label files according to the training format. Never add a fake `background` bounding box.

## 9. Annotation workflow in CVAT

1. Create one project with the frozen four-label taxonomy.
2. Create tasks along source-segment boundaries, not arbitrary mixed uploads.
3. Include annotator instructions and examples in the project.
4. Annotate the pilot batch.
5. Hold a calibration review and resolve confusing examples.
6. Update only clarifying instructions; do not change existing class meaning silently.
7. Annotate remaining batches.
8. Review all validation/test labels and a stratified sample of training labels.
9. Export an archival format preserving metadata.
10. Generate the Ultralytics YOLO detection export used for training.

CVAT supports Ultralytics YOLO detection export, but preserve a richer archival export because plain training labels do not retain all review and tracking metadata. See [CVAT Ultralytics YOLO format](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/).

## 10. Quality-control checklist

### Automated validation

- every image is readable;
- every label uses class ID 0-3;
- normalized coordinates are within valid range;
- box width and height are positive;
- image-label pairs match;
- no exact duplicate files across splits;
- no forbidden source-segment overlap;
- class and split counts are reported;
- extremely tiny/large boxes are flagged for review.

### Human review

- 100% of test labels;
- 100% of validation labels where feasible;
- at least 10% stratified training sample;
- all ambiguous flags;
- samples from every annotator, class, split, and difficult condition.

For a multi-annotator project, double-label a shared calibration set and report agreement or adjudication rate.

## 11. Dataset manifest

Minimum columns:

```text
image_id
relative_path
video_id
source_hash
frame_index
timestamp_ms
segment_id
split
width
height
blur_score
brightness_score
duplicate_group
selection_reason
annotation_status
review_status
```

Never rely only on filenames for provenance.

## 12. Public dataset integration

- Verify the exact source, version, class definitions, annotation quality, and license before downloading or merging.
- Map only compatible D00/D10/D20/D40 labels.
- Keep `source_dataset` in the manifest.
- Do not allow near-identical public images to cross splits.
- Prefer evaluating the final domain claim on the untouched custom-video test set.
- Compare public-only training with public + custom fine-tuning to measure domain adaptation.

The RoadDamageDetector repository lists the four core damage codes used here: [official repository](https://github.com/sekilab/RoadDamageDetector).

## 13. Augmentation rules

Permitted starting augmentations include moderate brightness/contrast, scale, crop, noise, compression, and motion blur. Use horizontal flip only after confirming it does not undermine class interpretation or camera-specific geometry.

Avoid unrealistic rotation, aggressive perspective warping, or augmentations that erase thin cracks. Augmentation is training-only and must never modify validation/test images.

## 14. Dataset versioning

Each frozen dataset release needs:

- semantic version such as `custom_dashcam_v0.1.0`;
- manifest checksum;
- taxonomy version;
- split-assignment file;
- annotation export checksum;
- class/split statistics;
- known issues and excluded intervals;
- creation date and responsible reviewer.

Training runs must reference an immutable dataset version.

