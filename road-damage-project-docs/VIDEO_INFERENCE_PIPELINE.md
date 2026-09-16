# Phase 4 Video Inference and Temporal Damage Events

## Purpose and scope

This is the first production/application service around the frozen primary
YOLOv8s detector. It validates a road video, performs one inference pass per
source frame, records direct detection observations, groups spatially
continuous same-class observations into heuristic video-level events, and
writes a stable machine-readable run contract.

The service does **not** provide geospatial identity or ground-truth unique
damage counts. Its outputs must be described as **temporally aggregated damage
events**, **heuristic damage-event counts**, or **video-level damage events**.
A tracker/event identifier is not proof that an event is one physical defect.

This phase does not implement severity, ByteTrack/BoT-SORT comparison,
FastAPI, or a frontend. The callable `analyze_video()` service is the future
backend integration boundary; a backend must not duplicate its inference or
aggregation logic.

## Frozen detector identity

- Family: YOLOv8s object detector
- Checkpoint:
  `outputs/training/baseline_public_v1/20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42/weights/best.pt`
- Size: `22,524,074` bytes
- SHA-256:
  `BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7`
- Input size: `640`
- Confidence: `0.19`
- NMS IoU: `0.50`
- Maximum detections per frame: `300`
- Frame stride: `1` (every frame)
- Class order: D00, D10, D20, D40

The config is [video_analysis_yolov8s.yaml](../configs/inference/video_analysis_yolov8s.yaml).
The loader compares the complete config against the approved literal contract,
so unknown fields and model/threshold/class/aggregation drift fail closed.
Normal CLI arguments cannot override the checkpoint, classes, input size,
confidence, NMS IoU, maximum detections, or association thresholds.

Before inference the service verifies the exact path, size, and SHA-256. After
constructing the already-verified checkpoint it requires the loaded model to
expose an explicit task value equal to `detect`; a missing, null, classification,
segmentation, pose, OBB, or other task fails closed. It also validates the
four-class names and checks exposed architecture metadata (`nc=4`, scale `s`)
when Ultralytics provides it. There is no download or fallback path.

## Commands

Preflight only (hash/model/video/output/device validation, no YOLO construction
and no inference):

```powershell
python src\road_damage\inference\video_analysis.py `
  --input data\videos\dashcam_raw.avi `
  --output-dir outputs\inference\teacher_preflight_target `
  --preflight
```

Safe bounded smoke run after review:

```powershell
python src\road_damage\inference\video_analysis.py `
  --input data\videos\dashcam_raw.avi `
  --output-dir outputs\inference\teacher_smoke_001 `
  --max-frames 100
```

JSON-only bounded run:

```powershell
python src\road_damage\inference\video_analysis.py `
  --input data\videos\dashcam_raw.avi `
  --output-dir outputs\inference\teacher_smoke_json_001 `
  --max-frames 100 `
  --no-annotated-video
```

`--start-frame` is a zero-based original source-frame index. Event timestamps
always use the original frame index and nominal source FPS. `--max-frames`
limits inferred frames and is required practice for initial smoke runs. A
limited run may complete successfully, but its summary sets
`full_source_processed=false` and warns that it is not a full-video result.

The output directory must be nonexistent or exactly empty. Existing run
content is never overwritten.

## Frame detection contract

`frame_detections.jsonl` contains one deterministic JSON object for every
inferred frame, including frames with zero detections. Each record includes:

- schema version;
- zero-based original `frame_index`;
- `timestamp_ms` and `timestamp_seconds`;
- source width and height;
- deterministically ordered detections.

Each direct detection contains:

- `detection_index`;
- assigned `event_id`;
- class ID and class name;
- confidence;
- `bbox_xyxy_pixel` and `bbox_xyxy_normalized`;
- pixel area and relative image area;
- association type and measurable association values.

Individual source frames are not extracted by default.

The JSONL is written incrementally. Completion verification also streams it
line by line and retains only counters, per-class counts, and referenced event
IDs; it never loads the complete long-video frame record into memory.

## Temporal association heuristic

The initial lightweight aggregator is deliberately interpretable and has no
new tracking dependency. It is an operational starting point, not a
validation-tuned tracking claim.

An active event and a new detection are candidates only when their class IDs
are identical. A candidate is accepted when either:

1. last-box IoU is at least `0.20`; or
2. normalized center distance is at most `0.12` **and** smaller/larger box-area
   ratio is at least `0.25`.

All accepted event/detection edges are ranked deterministically by:

1. IoU descending;
2. center distance ascending;
3. area ratio descending;
4. existing numeric event sequence;
5. detection index.

A global greedy pass then makes one-to-one assignments. Consequently one
detection cannot update two events, and one event cannot consume two
simultaneous detections. Unmatched detections create new IDs formatted
`RD0001`, `RD0002`, and so on.

The maximum gap is configured in seconds (`0.20`) and converted once using
`floor(maximum_gap_seconds * source_fps)`. A detection can reconnect when the
number of intervening missing frames does not exceed that limit. Once it does,
the event is finalized; a later detection starts another event. All direct
observations remain in the frame JSONL regardless of event status.

Events with at least three direct observations are marked `confirmed`; shorter
events are `tentative`. Both are persisted and counted separately. Confirmation
is only a persistence label—not physical-world ground truth.

Active events use bounded memory: each retains first/last metadata, the last
box needed for association, running confidence/area aggregates, association
counters, and one incrementally selected representative observation. No event
retains an append-only observation history. Finalization replaces active state
with a compact immutable event summary, so memory growth is limited to active
state and compact finalized summaries rather than all raw detections.

## Event schema

`events.json` uses the `road_damage.video_analysis.v1.events` schema. Every
event includes:

- event/class identity;
- first, last, and representative source frame;
- integer millisecond and second timestamps;
- duration and observation count;
- mean and maximum direct-detection confidence;
- representative pixel and normalized xyxy box;
- maximum/mean relative detector-box area;
- tentative/confirmed status and finalization reason;
- auditable association-rule counts;
- an explicit heuristic interpretation statement.

The representative observation is selected by confidence descending, relative
area descending, frame ascending, then detection index. Detector-box area is
not physical road-damage area or severity.

## Run artifacts

Each dedicated run directory contains:

```text
run_manifest.json
summary.json
events.json
frame_detections.jsonl
completion.json                 # successful runs only
annotated_video.mp4             # unless --no-annotated-video
```

`run_manifest.json` begins as `STARTED`, includes progress fields suitable for
job polling, and finishes as `COMPLETED`, `INTERRUPTED`, or
`FAILED_TECHNICAL`. It records the resolved configuration, source/config/code
hashes, Git/environment provenance, input fingerprint, device, processing
scope, and aggregation thresholds.

`summary.json` provides input metadata, frozen model identity, operating point,
decoded/inferred/skipped frame counts, raw detections by class, aggregated and
confirmed/tentative events by class, timing, output paths, and limitations.
`events.json` and the summary can be loaded directly by a future FastAPI job
endpoint; event timestamps can drive video seeking.

The final `COMPLETED` receipt is written only after active events are finalized;
the capture, writer, and JSONL resources close; events and summary are
persisted; and the annotated video is reopened and verified when enabled.
Completion reconciliation streams all frame rows and cross-checks raw totals
and per-class counts, event totals and confirmed/tentative distributions,
class identities, unique/well-formed event IDs, JSONL event references, and
finite numeric fields against `events.json`, `summary.json`, and in-memory run
state. Hashes are computed only from closed files, and the frozen checkpoint is
reverified afterward. Any contradiction fails closed and removes/withholds
`completion.json`; interruption or failure preserves partial diagnostics where
possible. The receipt represents a fully returned successful analysis: a late
`KeyboardInterrupt` or other finalization error after publication invalidates
the receipt and rewrites the run state as `INTERRUPTED` or `FAILED_TECHNICAL`
before the service returns or raises.

## Annotated video

The annotation uses the same inference result that is recorded and aggregated;
the model is not called a second time. Human-facing labels use
`<Friendly Damage Type> | <confidence> | <event ID>` (for example,
`Alligator Crack | 0.37 | RD0012`). The friendly overlay names do not alter the
canonical `class_id`, D00/D10/D20/D40 `class_name`, or `event_id` stored in JSON.
`annotated_video.mp4` uses the source width, height, and FPS and is reopened after
encoding to verify frame count and metadata. Phase 4 does not copy source audio,
which is recorded as a limitation.

## Teacher-video and scientific limitations

The teacher dashcam video has no reliable road-damage ground truth. It may be
used as an application demonstration, domain-behavior inspection source, or
hard-negative/real-world example. Results from it must never be presented as
accuracy, recall, F1, mAP, target-domain performance, or true unique damage
counts.

Motion, viewpoint change, occlusion, missed detections, nearby same-class
defects, and re-entry after the configured gap can fragment or incorrectly
associate heuristic events. Track/event performance requires separately
reviewed clips with appropriate event/identity ground truth before scientific
claims are possible.
