# Mentor Live Image Demo

This demo uses only the frozen YOLOv8s `best.pt`. It verifies the checkpoint's expected 22,524,074-byte size and SHA-256 before inference, uses `imgsz=640`, confidence `0.25`, and NMS IoU `0.70`, and saves annotated images under `outputs/demo/mentor_live/` without Ultralytics `runs/detect/` nesting.

> **Demo threshold only — final confidence/NMS thresholds have not yet been selected through the planned validation-only threshold sweep.**

## Commands

From the repository root, run one image:

```powershell
.\.venv\Scripts\python.exe src\road_damage\demo\mentor_demo.py --source <image-path>
```

Run the prepared validation-only sample folder sequentially:

```powershell
.\.venv\Scripts\python.exe src\road_damage\demo\mentor_demo.py --source outputs\demo\mentor_samples
```

Press any key to continue to the next image; press `Esc` or `q` to stop. If OpenCV cannot open a GUI window, inference continues and annotated images are still saved. Add `--no-display` when a window is not wanted.

## Classes

| ID | Class | Meaning |
|---:|---|---|
| 0 | `D00_longitudinal_crack` | Crack running broadly along the road direction |
| 1 | `D10_transverse_crack` | Crack running broadly across the road direction |
| 2 | `D20_alligator_crack` | Interconnected, fatigue/alligator-style cracking |
| 3 | `D40_pothole` | Visible pothole/open pavement cavity |

These are visual detection classes. The demo does not estimate physical depth, area, volume, engineering severity, unique events, or tracking IDs.

## Completed baseline result

The model trained on 12,620 images and used 2,602 validation images. The best checkpoint was epoch 50; training stopped at epoch 70 by early stopping with patience 20.

| Scope | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Overall | 0.540 | 0.466 | 0.475 | 0.216 |
| D00 | 0.470 | 0.360 | 0.362 | 0.164 |
| D10 | 0.507 | 0.418 | 0.420 | 0.161 |
| D20 | 0.629 | 0.653 | 0.669 | 0.340 |
| D40 | 0.552 | 0.434 | 0.450 | 0.199 |

These are validation results, not internal-test or teacher-video performance claims. The terminal's model inference time is not a full-application FPS measurement.

## If asked what remains

The next scientific steps are the planned validation-only threshold sweep, frozen-checkpoint evaluation on the untouched internal test split, error analysis, comparison with the proposed detector, and then video tracking, event aggregation, relative visual-severity evaluation, reporting, and dashboard integration. The teacher video has not been used here as positive performance evidence.
