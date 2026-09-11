# Mentor video application demo

This implements the video input/annotated output portions of PRD FR-01, FR-04, FR-05, FR-14 and FR-16. It reuses the mentor image demo's frozen-model checksum verification, class mapping, detection conversion and annotation renderer. Each source frame is decoded, inferred once, annotated once, written to video/CSV, and optionally shown. The preview uses the same annotation as the encoder. Memory does not grow with video length; the model is loaded once.

## Commands

Run these from `E:\AI_Road_Damage_Project` in the existing project environment. Normal saved-video inference (no GUI):

```powershell
.\.venv\Scripts\python.exe src\road_damage\demo\video_demo.py --source data\videos\dashcam_raw.avi
```

Live-preview inference, also saving the annotated video:

```powershell
.\.venv\Scripts\python.exe src\road_damage\demo\video_demo.py --source data\videos\dashcam_raw.avi --display
```

Limited preview (first 100 frames):

```powershell
.\.venv\Scripts\python.exe src\road_damage\demo\video_demo.py --source data\videos\dashcam_raw.avi --display --max-frames 100
```

Short saved-video run without a GUI (first 100 frames):

```powershell
.\.venv\Scripts\python.exe src\road_damage\demo\video_demo.py --source data\videos\dashcam_raw.avi --max-frames 100
```

`--max-frames` also works without `--display`. Press **q** or **Esc** in the preview to stop. **Ctrl+C** requests a stop at the next safe frame boundary, allowing the current encoder/CSV update to finish, and returns exit code 130. Capture, writer, CSV and GUI resources are released on completion, limits, interrupts, and errors. A forced process kill, power loss, or second external termination cannot guarantee a playable MP4.

## Outputs and timing

Each run reserves a new folder:

```text
outputs/demo/video/dashcam_raw_<UTC-run-id>/
    dashcam_raw_detected.mp4
    dashcam_raw_detections.csv
    dashcam_raw_summary.json
```

Default encoding is **MP4/mp4v (MPEG-4 Part 2)**. If unavailable, the tool warns and uses **AVI/MJPG** (`dashcam_raw_detected.avi`); no packages or codecs are installed. A failed codec probe file may remain in that run folder. Both failures stop with an actionable error. The summary records the actual codec/container and verifies output reopening, first-frame decode, frame count, FPS and dimensions after the writer closes.

The saved video retains nominal source FPS and original width/height. Preview alone may shrink to at most 1280×800. No source frames are resized for encoding, and no source file is overwritten. Odd pixel dimensions are rejected because the current codecs may silently crop them. Audio is not copied. Output is constant-frame-rate; variable-frame-rate source timing is approximated by nominal FPS. An unknown source frame count is reported as unknown; a decoded first frame is required. If decoding stops before the advertised frame count, the output is explicitly marked partial.

Progress reports processed/target frames, elapsed time, processing FPS, nominal source FPS and ETA. Processing FPS measures the application frame loop (including first-inference warmup, rendering, writer submission, CSV and enabled preview); it excludes initial model/source hashing, loading, final writer flush and output verification. It is not a real-time guarantee or an isolated model-speed benchmark. Preview is updated as each frame completes; slow inference therefore makes the preview slower than source playback.

## CSV and summary

CSV has one row per detection:

```text
frame_index,timestamp_seconds,class_id,class_name,confidence,x1,y1,x2,y2
```

Frame indices are zero-based. Timestamps use the documented frame-index/nominal-FPS fallback rounded to integer milliseconds and serialized as seconds with three decimals. Boxes are the same integer pixel xyxy coordinates used for rendering. No-detection frames contribute no CSV rows but still count toward processed frames.

Summary JSON records source path/hash/size, frozen model path/SHA/size, input resolution/FPS/frame count/duration, processed frames, output duration/path/codec, CSV path, the frozen validation-selected operating point and status, class mapping and raw counts, frames with detections, timing and average processing FPS, package versions, Git commit/status, code/config hashes, resolved configuration, output encoding checks, source size/mtime checks, model hash recheck, GUI status, and interruption/error state. Local source paths stay in ignored generated outputs.

Status is `completed`, `limited` (`--max-frames`), `interrupted` (q/Esc/Ctrl+C), `partial` (early decode stop), or `failed`. Anything other than `completed` has `partial_output: true`; retained output covers only processed frames and must not be presented as a complete-video run. A running marker remains if the process is forcibly killed. Encoder errors or disk exhaustion may still make a partial file unusable; check `encoding_verification` before presenting it.

## Frozen model and scientific scope

The only permitted model is this baseline run's `best.pt`, size **22,524,074 bytes**, SHA-256 **BEC3A297EAF3D9D2B5553D6D2D7550646D31B9EDF5FE41437E073975A5B1FCF7**. There is no CLI model override. `last.pt`, other checkpoints, altered hashes, test/dataset sources and redirected paths are rejected. The model uses device 0, imgsz 640, augment=False, and four labels in order: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole.

Thresholds live centrally in `configs/demo/video_demo.yaml`: confidence **0.19**, NMS IoU **0.50**, status **`frozen_validation_selected`**. They are the final frozen YOLOv8s baseline deployment operating point selected using validation data only. The completed internal-test evaluation did not alter these values. Any later change requires a newly reviewed config/status update. The current tool logs, displays and records:

> Operating-point status: `frozen_validation_selected`. Final frozen validation-selected YOLOv8s baseline operating point: confidence 0.19, NMS IoU 0.50. The completed internal test did not alter these values. Demo execution is application inference only; its outputs are not new model evaluation results.

The teacher video is permitted as **application input**. Earlier adjudication found no reliably confirmed V1-positive D00/D10/D20/D40 events. Its detections demonstrate deployment behavior/domain mismatch/potential false positives; they do not establish ground-truth accuracy, target-domain recall, or labelled-positive performance. Raw detection counts can count the same defect in many adjacent frames and are not unique physical-damage events. Tracking, event aggregation, severity, operating-point changes and dataset/model changes are outside this demo. The completed baseline internal-test result remains locked and is not accessed or reevaluated by the demo.
