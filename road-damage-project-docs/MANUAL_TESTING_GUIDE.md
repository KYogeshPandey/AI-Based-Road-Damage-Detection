# Manual Testing Guide

This guide provides a beginner-friendly, read-only verification sequence for
the repository. It does not train a model, resume a run, execute inference, or
run the protected internal-test evaluation.

Run commands from the project root in PowerShell. If the repository is cloned
elsewhere, replace the example location with the actual local path.

## A. Environment Check

```powershell
Set-Location E:\AI_Road_Damage_Project
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import cv2, numpy; print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__)"
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch', torch.__version__); print('CUDA runtime', torch.version.cuda); print('CUDA available', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
ffmpeg -version
ffprobe -version
```

Expected for the completed experiments: Python 3.12.10, PyTorch
2.13.0+cu126, CUDA 12.6, OpenCV 5.0.0.93, and the locally recorded NVIDIA
GeForce RTX 2050. A CPU-only machine can inspect documentation and run many
tests, but it does not reproduce the approved GPU environment.

The tracked `requirements.txt` contains only the declared Phase 2C dependency
subset. It is not a complete lock file for the frozen training environment.

## B. Repository Test Suite

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The release-preparation baseline records 300 passing tests locally. A different
machine may report environment-specific failures; preserve the complete output
before changing dependencies.

## C. Safe CLI Help Checks

The following commands request help only. They do not authorize training,
resume, prediction, evaluation, or data acquisition.

```powershell
.\.venv\Scripts\python.exe src\road_damage\video\inspect_video.py --help
.\.venv\Scripts\python.exe src\road_damage\video\reconnaissance.py --help
.\.venv\Scripts\python.exe src\road_damage\training\train_baseline.py --help
.\.venv\Scripts\python.exe src\road_damage\training\experiment2.py --help
.\.venv\Scripts\python.exe src\road_damage\evaluation\select_threshold.py --help
.\.venv\Scripts\python.exe src\road_damage\evaluation\select_experiment2_threshold.py --help
.\.venv\Scripts\python.exe src\road_damage\evaluation\error_analysis.py --help
.\.venv\Scripts\python.exe src\road_damage\evaluation\experiment2_error_analysis.py --help
.\.venv\Scripts\python.exe src\road_damage\demo\mentor_demo.py --help
.\.venv\Scripts\python.exe src\road_damage\demo\video_demo.py --help
```

Do not add or run an internal-test command as part of routine verification.

## D. Dataset and Configuration Existence Checks

These checks report presence only. They do not enumerate protected samples or
read image/label contents.

```powershell
$requiredPaths = @(
  'configs\dataset\rdd2022_phase2a.yaml',
  'configs\dataset\rdd2022_phase2b_v1_1.yaml',
  'configs\dataset\rdd2022_phase2c1_yolo_detection_v1.yaml',
  'configs\training\baseline_public_v1_yolov8s.yaml',
  'configs\training\experiment2_yolo26s_matched.yaml',
  'configs\evaluation\baseline_public_v1_frozen_operating_point.yaml',
  'configs\evaluation\experiment2_yolo26s_threshold_selection.yaml',
  'data\processed\rdd2022_india_japan_v1_1',
  'data\exports\rdd2022_india_japan_v1_1\yolo_detection_v1'
)

$requiredPaths | ForEach-Object {
  [pscustomobject]@{ Path = $_; Exists = Test-Path -LiteralPath $_ }
} | Format-Table -AutoSize
```

The two `data` paths are intentionally absent from a normal Git clone. They
exist only after authorized local dataset preparation.

## E. Frozen Checkpoint Existence and Hash Checks

The trained checkpoints are Git-ignored. Missing files are therefore expected
in a public clone. If authorized local copies exist, verify them without loading
the models:

```powershell
$checkpoints = [ordered]@{
  'outputs\training\baseline_public_v1\20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42\weights\best.pt' = 'bec3a297eaf3d9d2b5553d6d2d7550646d31b9edf5fe41437e073975a5b1fcf7'
  'outputs\training\experiment2_yolo26s_matched\yolo26s_rdd2022-india-japan-v1.1.0_640_seed42\weights\best.pt' = '99c03d56f4b27d6a9cc774dd3880f11807642058934199ed675dc3bf4b9f37a1'
}

@(
  foreach ($entry in $checkpoints.GetEnumerator()) {
    if (Test-Path -LiteralPath $entry.Key -PathType Leaf) {
      $actual = (Get-FileHash -LiteralPath $entry.Key -Algorithm SHA256).Hash.ToLowerInvariant()
      [pscustomobject]@{
        Path = $entry.Key
        Status = if ($actual -eq $entry.Value) { 'MATCH' } else { 'MISMATCH - STOP' }
        SHA256 = $actual
      }
    } else {
      [pscustomobject]@{ Path = $entry.Key; Status = 'MISSING'; SHA256 = $null }
    }
  }
) | Format-Table -AutoSize
```

Never rename another checkpoint to satisfy a missing-file check. A mismatch
requires provenance review.

## F. Safe Validation-Artifact Checks

Only approved validation artifacts are listed here. This section does not touch
the internal-test output tree.

```powershell
$validationArtifacts = @(
  'outputs\evaluation\baseline_public_v1_threshold_selection\completion.json',
  'outputs\evaluation\baseline_public_v1_error_analysis\error_analysis_manifest.json',
  'outputs\evaluation\experiment2_yolo26s_threshold_selection\completion.json',
  'outputs\evaluation\experiment2_yolo26s_error_analysis\analysis_manifest.json',
  'outputs\evaluation\experiment2_yolo26s_error_analysis\completion.json'
)

$validationArtifacts | ForEach-Object {
  [pscustomobject]@{ Path = $_; Exists = Test-Path -LiteralPath $_ -PathType Leaf }
} | Format-Table -AutoSize

$predictionCache = 'outputs\evaluation\experiment2_yolo26s_threshold_selection\prediction_cache.json'
if (Test-Path -LiteralPath $predictionCache -PathType Leaf) {
  $actual = (Get-FileHash -LiteralPath $predictionCache -Algorithm SHA256).Hash.ToLowerInvariant()
  [pscustomobject]@{
    Path = $predictionCache
    Expected = 'b48aebb298bd4f1881cddd10602de332f8cc0e560f05fd9b2f9dbb212b8b98ca'
    Actual = $actual
    Matches = $actual -eq 'b48aebb298bd4f1881cddd10602de332f8cc0e560f05fd9b2f9dbb212b8b98ca'
  } | Format-List
}
```

Generated validation artifacts are ignored by Git and may be absent from a
clone. Do not substitute internal-test artifacts when they are missing.

## G. Git Cleanliness Check

```powershell
git status --short
git diff --check
```

Before a public push, inspect every listed path. Dataset, video, checkpoints,
generated outputs, environments, secrets, and downloaded papers must not be
staged.

## H. Common Failure Symptoms

| Symptom | Meaning and safe response |
|---|---|
| `.venv\Scripts\python.exe` is missing | Create the environment using Python 3.12.10; do not point scientific commands at an arbitrary interpreter. |
| `pip check` reports conflicts | Preserve the output and compare installed versions with `TECH_STACK.md`; do not upgrade packages casually. |
| `CUDA available False` | Documentation and many unit tests may still work, but this is not the approved training environment. Check the NVIDIA driver and PyTorch build. |
| `ffmpeg` or `ffprobe` is not recognized | Install FFmpeg and add its binaries to `PATH`; no video processing should start until both commands work. |
| A dataset directory is missing | Expected in a public clone. Follow the approved acquisition/governance process; never copy private data into Git. |
| A checkpoint is missing | Expected in a public clone because `*.pt` is ignored. Obtain an authorized, checksum-verified copy. |
| A checkpoint/cache hash differs | Stop. Do not regenerate or relabel it as the approved artifact. Investigate provenance. |
| Frozen training YAML contains a local absolute path | It is a historical experiment record, not a portable template. Do not edit it silently; use a separately reviewed local reproduction config. |
| Validation outputs are missing | Expected when generated outputs were not distributed. Do not rerun the internal test or use it as a replacement. |
| Tests fail while the host environment previously passed | Capture the exact interpreter and package versions, then separate environment failures from code failures. |
| `git status --short` lists data, video, weights, or outputs | Stop before staging and verify `.gitignore` plus the exact path. |

## Release Safety Boundary

This guide intentionally contains no command that trains, resumes, performs
model inference, accesses protected internal-test samples, or modifies frozen
scientific artifacts.
