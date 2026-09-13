# Experiment 2: matched pretrained YOLO26s

Status: preparation tooling with four preserved failed smoke attempts. No smoke
completion has been approved, and full training has not been run. This implements
the PRD's V1 detector comparison / RQ2 and EVALUATION_PLAN detector experiment.
The preceding reviewed project commit is
`95df7f1e85c8101b2fd9b1f0a4538a431234f557`.

## Scientific design

Compare the completed pretrained YOLOv8s baseline with pretrained YOLO26s on
RDD2022 India + Japan v1.1, using its unchanged leakage-reduced grouped split.
This split is not claimed to be route-independent. Development and model
selection use train and validation only. The completed baseline internal test
remains a locked final record and must not become an iterative benchmark.

The motivation comes from the completed **validation-only** cache analysis:
D00 recall 0.3698; India pooled F1 0.4278 versus Japan 0.5672; 325 geometric
localization candidates; weak small-object recall (particularly D20, 8/57);
D20 strongest by F1 (0.6515); 121 FPs across 983 negative validation images.
India D10 has only 10 validation targets in 10 images. These findings justify
the planned controlled architecture comparison, not a simultaneous change to
data composition, class balancing, augmentation, or resolution.

## Exact architecture and unavoidable differences

Installed Ultralytics **8.4.130** resolves virtual `yolo26s.yaml` to
`ultralytics/cfg/models/26/yolo26.yaml` with scale `s`:
depth 0.50, width 0.50, maximum channels 1024. It is a P3/8–P5/32 detection model.
The local architecture SHA-256 is pinned in the experiment config.
The expected pretrained checkpoint is the publisher's 80-class COCO detection
model; transfer learning replaces the class head for the four road-damage classes.

| Aspect | YOLOv8s baseline | YOLO26s Experiment 2 | Interpretation |
|---|---|---|---|
| Head | conventional detection / NMS | native end-to-end dual assignment (`end2end=true`) | architecture-specific loss and output semantics |
| Box regression | DFL, `reg_max=16` | DFL-free normalized L1-style regression, `reg_max=1` | `dfl=1.5` remains an active gain on YOLO26's L1-style regression term: matched hyperparameter value, model-specific loss semantics |
| Criterion | `v8DetectionLoss` | `E2ELoss` with one-to-many and one-to-one components | native loss/assignment and scheduled branch weighting are part of this model-family comparison |
| Validation NMS IoU argument | 0.70 | 0.70 passed explicitly | native end-to-end output does not use ordinary NMS in the same way |
| Optimizer | explicitly SGD | explicitly SGD | no `auto` optimizer and no switch to MuSGD |
| Pretraining | official YOLOv8s COCO checkpoint | official YOLO26s COCO checkpoint | pretrained model families include different upstream training histories |

The result is a **matched fine-tuning protocol comparing pretrained model
families**, not an isolated backbone-only ablation. Forcing YOLO26 into a
non-native head/loss would be a different experiment. No change to common
training hyperparameters is needed for compatibility. Later validation tooling
must explicitly accommodate native end-to-end predictions; do not reuse a
YOLOv8s NMS sweep and imply its NMS IoU has identical meaning. No new evaluation
tooling or test comparison is implemented here.

The `box=7.5`, `cls=0.5`, and `dfl=1.5` argument values match the completed
YOLOv8s baseline. Their effective loss interpretation is not identical. In
Ultralytics 8.4.130, YOLO26's `reg_max=1` disables Distribution Focal Loss and
uses normalized L1-style box regression in the third loss slot; `hyp.dfl` still
multiplies that term. Consequently, `dfl=1.5` is active for YOLO26 as an
L1-style regression-loss gain even though it is no longer a Distribution Focal
Loss gain. This is a **matched hyperparameter value with model-specific loss
semantics**, and the experiment is not an exact loss-function ablation.

References: [official YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
and [publisher release v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0).
Compatibility decisions are based on the installed 8.4.130 source, especially
`nn/tasks.py`, `cfg/models/26/yolo26.yaml`, `utils/loss.py`, and `engine/trainer.py`.

## Matched settings

`configs/training/experiment2_yolo26s_matched.yaml` explicitly pins the values.
The loader verifies its hash and compares them with the immutable baseline
config. Preflight additionally verifies the SHA of the completed baseline's
`args.yaml` and compares its effective recorded values.

| Setting | YOLOv8s | YOLO26s full run |
|---|---|---|
| Dataset / train / val | v1.1 / 12,620 / 2,602 | same |
| Classes | 0 D00, 1 D10, 2 D20, 3 D40 | same names and numeric order |
| Image / batch / nominal batch | 640 / 4 / 64 | same |
| Device / workers / cache | 0 / 2 / false | same |
| AMP / compile | true / false | same; abort if AMP is disabled |
| Maximum epochs / patience | 100 / 20 | same |
| SGD LR / final factor / cosine | 0.01 / 0.01 / true | same |
| Momentum / weight decay | 0.937 / 0.0005 | same |
| Warmup epochs / momentum / bias LR | 3 / 0.8 / 0.1 | same |
| Seed / deterministic | 42 / true | same |
| HSV H / S / V | 0.015 / 0.40 / 0.30 | same |
| Translate / scale / horizontal flip | 0.05 / 0.25 / 0.50 | same |
| Mosaic / close mosaic | 0.25 / 15 | same |
| Degrees / vertical flip / shear / perspective | all zero | same |
| Mixup / cutmix / copy-paste / BGR | all zero | same |
| Fraction / rectangular / multiscale | 1.0 / false / 0.0 | same |
| Box / class / third regression gain | 7.5 / 0.5 / DFL 1.5 | 7.5 / 0.5 / L1-style term 1.5; matched argument values, model-specific loss semantics |
| Class weighting / class remap | 0.0 / true | same |
| Validation / max detections / IoU | true / 300 / 0.70 | same settings; native head caveat above |
| Save / save period / plots | true / 5 / true | same |

Rotations and vertical flips remain disabled because D00/D10 orientation has
semantic meaning. Common settings such as `single_cls=false`, `freeze=null`,
`conf=null`, `agnostic_nms=false`, `classes=null`, `distill_model=null`,
`profile=false`, and `channels_last=false` are explicit too.

## Environment

Use only `E:\AI_Road_Damage_Project\.venv\Scripts\python.exe`, Python 3.12.10,
torch 2.13.0+cu126, torchvision 0.28.0+cu126, Ultralytics 8.4.130,
NumPy 2.5.2 and CUDA 12.6 on device 0, NVIDIA GeForce RTX 2050 (4 GB).
No installation or environment repair is part of these commands.

The pre-existing `weights/yolo26n.pt` remains only the verified framework AMP
check artifact. It cannot replace YOLO26s as the training input. Native AMP
checking runs only during a user-started smoke/full training operation, never
during preflight. Framework downloads, automatic installation, and external
integration callbacks are disabled during weight inspection and training.
Missing local resources fail rather than being acquired inside training.

## Controlled acquisition — USER command, separate operation

From `E:\AI_Road_Damage_Project`:

```powershell
.\.venv\Scripts\python.exe src\road_damage\training\experiment2.py --acquire-pretrained
```

This command only acquires/freezes the pretrained model. It uses the fixed
GitHub `ultralytics/assets` release `v8.4.0`, requires exactly `yolo26s.pt`,
checks its exact download URL, and requires the publisher API's SHA-256 and size.
It downloads to a new `.partial` file, verifies size/hash **before** checkpoint
deserialization, then performs a restricted CPU checkpoint-structure inspection
with no forward pass. An existing local weight can be adopted only if its bytes
match that same publisher identity. Existing identity records are never replaced.
If publisher metadata is missing or different, stop for review; there is no
alternate URL, architecture, random initialization, or latest-release fallback.
Interrupted or invalid partial files remain diagnostic evidence and are not
silently overwritten.

The model is stored at `models/pretrained/yolo26s.pt`. The command creates
`reproducibility/experiment2_yolo26s_pretrained_identity.json` with family,
variant, task, pretrained source URL, path, size, SHA-256, publisher asset ID and
digest, framework version, architecture hash and acquisition timestamp.
The official model uses the 80-class COCO head; it is not a trained RDD model.

Inspect and commit the identity JSON together with the reviewed Experiment 2
tooling before execution. Keep weight bytes ignored. This task creates no Git
commit, and acquisition never proceeds automatically to preflight or training.

## Preflight — USER command, no training or inference

```powershell
.\.venv\Scripts\python.exe src\road_damage\training\experiment2.py --preflight
```

Preflight prints JSON and exits 0 only if its checks pass, otherwise 2 with
specific blockers. It verifies:

- exact interpreter, framework versions, NumPy, CUDA, GPU and AMP-check artifact;
- installed small detection architecture, acquired identity, actual weight
  size/hash and loaded checkpoint structure (no prediction/forward pass);
- baseline config and effective recorded argument parity;
- approved export metadata hashes and the unchanged train/val-only YAML;
- actual train/val directory name counts and pairing, 12,620 / 2,602, with
  exact four-class mapping and no YAML test entry;
- a fresh full-run output path and writable ancestor;
- clean Git state, ancestry after `95df7f1...`, tracked tooling/identity, and a
  deterministic source fingerprint built from blobs in the recorded committed
  Git `HEAD` tree. Only committed files in the approved source/config/tests/docs
  scope enter this Experiment 2 fingerprint; ignored or untracked local files,
  including literature and research-paper artifacts, do not enter it. The commit
  and fingerprint are bound into smoke/full-training manifests and receipts.

Preflight does not read image or label contents, create output directories,
download resources, or call `model.train`, `model.predict`, or `model.val`.
It can read shared export metadata containing aggregate test information; it
does not enumerate or open internal-test image/label directories. Its name
inventory loops are literal `train` and `val`. Symlink/junction redirects fail.
The preflight is expected to report missing-identity and/or uncommitted-source
blockers until the user completes acquisition and the review/commit step.

Before a user-started smoke/full run, additional byte verification follows
only train/val records in the pinned export manifest. Internal-test records
are skipped before their paths are resolved. Dataset YAML has exactly `train`,
`val`, and `names`. There is no user-supplied dataset or model override.

## Smoke test — USER command, three-epoch protocol prepared but not run

```powershell
.\.venv\Scripts\python.exe src\road_damage\training\experiment2.py --smoke
```

Reuses the existing deterministic Phase 3B subset in
`outputs/training/phase3b_smoke/dataset/`. It does not sample or extract anything.
The pinned manifest and YAML are checked, then the existing independent smoke
validator rechecks byte identity, train-to-train / val-to-val lineage, class
mapping, countries, counts, positives/negatives and exclusions.

The smoke run is exactly **three epochs**, 128 train images (80 positive / 48
negative), 64 validation images (40 positive / 24 negative), batch 4, 640 pixels,
device 0, seed 42, AMP, SGD. The training `fraction` is type-pinned to float
`1.0`, meaning the complete 128-image training split; Ultralytics interprets
integer `1` as a one-image count, so that representation is explicitly rejected.
Validation remains the complete 64-image split. Overrides are restricted to three epochs,
`close_mosaic=0`, `save_period=1`, `plots=false`. It cannot accept an epoch,
model, dataset, output, resume, or force override. It runs exactly 32 batches per
epoch and exactly 96 training batches in total. Full training remains 100 epochs with
the unchanged matched scientific settings.

The fourth failed smoke completed two epochs / 64 batches with eight natural
AMP optimizer attempts, all skipped by GradScaler. It recorded zero successful
SGD updates and empty optimizer state. The reviewed scaler evidence shows that
scale 256 was reached only after the final attempt, so no attempt tested that
scale. Under the reviewed installed Ultralytics 8.4.130 behavior, the three-epoch
bound provides 14 natural attempts, at global one-based batches
`1, 2, 3, 5, 7, 10, 14, 19, 26, 35, 47, 62, 78, 94`. After the observed eight
backoff skips, the ninth attempt naturally tests scale 256. This is an opportunity
for normal dynamic scaling; it does not establish that scale 256 will succeed.
No AMP workaround is introduced: the initial scale, scaler state, optimizer
steps, accumulation settings and scientific hyperparameters are unchanged.
Success still requires at least one actual SGD step and independent optimizer
state evidence. All four historical failed attempts remain preserved.

This is a technical test of loading, CUDA, forward/backward optimization,
four-class mapping, memory feasibility and checkpoint writing. Its losses or
validation metrics are not an accuracy experiment. `optimizer_step_attempts`
records accumulated optimizer/AMP attempts observed through Ultralytics' EMA
update counter; `optimizer_updates` counts only successful underlying SGD
`step()` calls through a scoped PyTorch optimizer post-hook. Successful smoke
completion requires at least one such SGD update plus non-empty momentum-SGD
state. An attempted step skipped naturally by AMP is diagnostic evidence, not a
successful update, and the smoke fails if every attempt is skipped. The run also
records finite-loss batch completion, native AMP/CUDA/scaler state, epoch state,
checkpoint hashes, and peak allocated/reserved CUDA memory. A failed/OOM or
partial run cannot publish `completion.json`. Do not reduce batch/resolution,
switch to nano, or disable AMP silently; those changes would need a separately
reviewed experiment design.

The smoke directory is:
`outputs/training/experiment2_yolo26s_matched/smoke_yolo26s_640_batch4_seed42/`.
An existing directory, even partial, is refused. Keep failed evidence and resolve
the specific cause in a reviewed correction before retrying.

## Future full training — DOCUMENT ONLY, not executed in this phase

```powershell
.\.venv\Scripts\python.exe src\road_damage\training\experiment2.py --train
```

This fresh-run command repeats preflight and train/val byte checks, and requires
a completed three-epoch / 96-batch smoke receipt with identical model/config/source/dataset
identity and unchanged persisted checkpoint/results hashes. The smoke
`completion.json` is bound to both the exact verified clean Git `HEAD` commit and
the committed-tree source hash. Full training rejects the receipt when either
value differs, including when a later commit changes only an out-of-scope path
and therefore happens to preserve the approved-scope source hash. After any
commit change, the previous smoke receipt cannot authorize full training. It
then uses:

`outputs/training/experiment2_yolo26s_matched/yolo26s_rdd2022-india-japan-v1.1.0_640_seed42/`.

The run snapshots an absolute train/val-only YAML, resolved arguments, source
state, Git commit, environment, pretrained identity, metadata fingerprints and
smoke validation. `run_manifest.json` and `completion.json` record the same
verified `git_commit` and `source_tree_sha256` values. Successful completion
copies `internal_test_files_accessed: false` from the run manifest state. Smoke
receipt validation requires this explicit JSON boolean `false` independently in
both records; a missing field, `true`, or a non-boolean value fails the gate.
Existing run directories
are refused. Source labels/images are not copied or edited. Ultralytics dataset
cache disk writes are suppressed, while required in-memory cache metadata such
as the dataset cache version is preserved so a cache rescan remains valid.
Generated cache metadata is not part of the experiment identity.
Completion requires finite contiguous results and both best/last checkpoints.

This launcher supports **fresh full training only**. Native within-process early
stopping uses patience 20; epoch state records stopper/optimizer/scaler evidence
for any future reviewed resume extension. `last.pt` should normally exist after
at least one successfully saved training epoch, but it may be absent if training
fails or is interrupted before the first checkpoint save. The presence of
`last.pt` does **not** authorize an improvised resume command. The current
Experiment 2 launcher exposes no approved resume workflow and deliberately does
not reuse the baseline's hardcoded resume policy or mislabel Experiment 2 as
Experiment 1. If full training is interrupted, **STOP** and preserve the partial
run. Resume support requires separate reviewed tooling and provenance checks
before use; neither a generic resume command nor restarting in the same directory
is supported here.

After completion, checkpoint selection and operating-point selection must use
validation only, with the same metric philosophy as the baseline and explicit
end-to-end output semantics. No internal-test evaluation is authorized here.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_experiment2.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use synthetic files and mocked training calls. They must not acquire
real weights, run model inference, or access individual internal-test data.
Experiment 1 configs, runner, model, provenance, source-state manifest and
completed internal-test outputs remain immutable. The source-state builder is
reused in memory; this task does not regenerate the baseline manifest.
