# YOLOv8s Public Baseline Runbook

## Scope

This runbook controls one logical public-data YOLOv8s experiment with a maximum target of 100 epochs. The intended manual sessions are approximately epochs 1–20, 21–40, 41–60, 61–80, and 81–100. They are not separate training jobs.

The launcher exposes only the approved train and validation directories through `configs/training/baseline_public_v1_train_val.yaml`. The internal test split is deliberately absent and must remain unavailable to Ultralytics during training.

## Frozen run

- Run directory: `outputs/training/baseline_public_v1/20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42`
- Initial weights: `models/pretrained/yolov8s.pt`
- Target: at most 100 epochs with patience 20
- Batch and image size: 4 and 640
- Seed: 42 with deterministic mode enabled
- Recovery checkpoint interval: every 5 epochs, in addition to normal `last.pt` updates

The fresh launcher refuses to use a run directory that already exists. It passes absolute `project`, `data`, model, and `save_dir` paths so Ultralytics writes directly beneath `outputs/training/baseline_public_v1/`; it cannot fall back to `runs/detect/outputs/...`.

Before calling Ultralytics, the launcher verifies the frozen config, train/val-only dataset YAML, pretrained checksum, CUDA, package versions, and the existing automatic AMP-check artifact. It then records the requested and resolved configs, package freeze, hardware, checksums, export fingerprints, and session history in the run's `reproducibility/` directory.

The launcher also requires a clean local Git commit and an up-to-date deterministic source-state checksum manifest. Git remains the authoritative source identity; the checksum manifest is a secondary audit artifact and does not replace the commit.

Early-stopping continuity is project-owned because Ultralytics 8.4.130 does not restore its `EarlyStopping.best_epoch` or `best_fitness` across Python processes. After every completed epoch, an `on_model_save` callback atomically writes `state/early_stopping_state.json` only after `last.pt` has been written. On resume, an `on_pretrain_routine_end` callback restores patience 20, best epoch, best fitness, and the possible-stop condition after Ultralytics has created its stopper and loaded the checkpoint, but before `on_train_start` and the first resumed epoch. Missing, stale, cross-run, exhausted, or inconsistent state blocks resume.

Ultralytics 8.4.130 derives `save_period=5` filenames from its zero-based internal epoch index. Periodic checkpoint filenames can therefore correspond to human epochs 1, 6, 11, 16, 21, and so on. Their naming is not part of the scientific experiment; `weights/last.pt` remains the authoritative resume checkpoint after every completed epoch.

## Fresh start

Before the first run, initialize Git yourself, review `git status`, and create a local commit containing the source, configs, tests, documentation, `.gitignore`, and `reproducibility/baseline_public_v1_source_state_manifest.json`. Do not add data, outputs, environments, or weights. The launcher refuses to start without a clean commit matching the prepared source state.

From the project root in PowerShell, run exactly:

```powershell
.\.venv\Scripts\python.exe src\road_damage\training\train_baseline.py --config configs\training\baseline_public_v1_yolov8s.yaml
```

Do not set `epochs=20`. The experiment must start with the frozen `epochs=100` target.

## Safe manual interruption

At each intended boundary (20, 40, 60, and 80):

1. Allow that epoch's training, validation, and checkpoint write to finish.
2. Preferably wait until the next epoch begins.
3. Press `Ctrl+C` once to interrupt Python.
4. Do not close PowerShell, kill the process, suspend, or remove power while `last.pt` is being written.
5. Run the read-only checkpoint inspection command below.
6. Shut down or restart only after inspection reports `appears_resumable: true`.

If early stopping naturally ends the experiment before epoch 100, do not resume merely to force 100 epochs. A normally finalized Ultralytics checkpoint has its optimizer state stripped and should be reported as non-resumable.

## Checkpoint inspection

After each interruption, run:

```powershell
.\.venv\Scripts\python.exe src\road_damage\training\inspect_checkpoint.py --checkpoint outputs\training\baseline_public_v1\20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42\weights\last.pt
```

The command loads the trusted checkpoint on CPU and reports its hash, zero-based stored epoch, human-readable completed and next epoch numbers, target epochs, optimizer/scaler/train-argument presence, run identity, state path, persisted best epoch/fitness, patience, and checkpoint/config/run consistency. It hashes and stats both checkpoint and state before and after loading to verify that inspection was read-only. Exit code `0` means the checkpoint and early-stopping state appear resumable; exit code `2` means they do not.

## Resume the same run

Only after the inspection passes, run:

```powershell
.\.venv\Scripts\python.exe src\road_damage\training\train_baseline.py --config configs\training\baseline_public_v1_yolov8s.yaml --resume outputs\training\baseline_public_v1\20260907_yolov8s_rdd2022-india-japan-v1.1.0_640_seed42\weights\last.pt
```

The launcher accepts only that exact run's `weights/last.pt`. It checks that the checkpoint remains resumable, targets 100 epochs, retains the original scientific parameters, matches the run/output identity, and refers to unchanged dataset/export fingerprints. It then calls Ultralytics with only `resume=<absolute last.pt>`; it does not start another run or replace resume parameters.

Ultralytics 8.4.130 rewrites `train_args.model` to the authoritative `last.pt` path during a genuine resume and saves that value into subsequent checkpoints. Resume validation therefore accepts exactly two model-field values: the configured original `models/pretrained/yolov8s.pt`, or this experiment's exact `weights/last.pt`. Any other run, smoke checkpoint, architecture, or arbitrary weight path remains prohibited. The original pretrained checkpoint is independently verified against both its pinned SHA-256 and the fresh-start run manifest.

Human epochs 1-46 were trained under Git commit `80916642e45fe00bcc9a6054dea2cce3ea55e6a7` and source fingerprint `1e94e69c6a7e4b1532d764d41a05118c2cd78e6a3aa40aab2bb1829e512858da`. The non-scientific multi-resume launcher correction is governed by `reproducibility/baseline_public_v1_resume_hotfix_policy.json`. On its first use, the launcher requires the byte-identical epoch-46 checkpoint approved by that policy. It then records the user's clean hotfix commit, both source fingerprints, unchanged config/model/dataset identities, the pre-resume checkpoint hash, reason, and exact hotfix file list in the session record. Later resumes must use the same hotfix commit. There is no force or generic Git-bypass option.

Confirm in the Ultralytics console that a checkpoint interrupted after completed epoch 20 continues at epoch 21. Repeat the same procedure after completed epochs 40, 60, and 80. Do not change scientific training parameters between sessions.

## Operational cautions

Smoke training reported slow image access. The baseline deliberately retains `cache: false`, does not duplicate or move the full dataset, and does not enable RAM caching. Monitor full-run throughput before considering a separately approved storage change.

`weights/yolo26n.pt` is an automatic Ultralytics AMP-compatibility-check artifact. It is recorded but is not the proposed YOLO26 experiment, a baseline training input, or an evaluated model. The launcher requires the existing pinned file so the AMP check does not download another weight.

Do not run standalone validation, inference, or internal-test evaluation as part of this manual training workflow.
