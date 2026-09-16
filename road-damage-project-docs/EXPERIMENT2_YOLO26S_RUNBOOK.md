# Experiment 2: matched pretrained YOLO26s

Status: full training completed by early stopping after epoch 83. The frozen
validation-selected checkpoint is `weights/best.pt` from epoch 63, SHA-256
`99c03d56f4b27d6a9cc774dd3880f11807642058934199ed675dc3bf4b9f37a1`.
Earlier smoke attempts and interrupted resume segments remain historical evidence. This
implements the PRD's V1 detector comparison / RQ2 and EVALUATION_PLAN experiment.
The original full-training commit is
`39070294088a725edf9092d861e5da4d750acd25`.

## Validation-only operating-threshold selection

The dedicated selector is
`src/road_damage/evaluation/select_experiment2_threshold.py`, governed by
`configs/evaluation/experiment2_yolo26s_threshold_selection.yaml`. It verifies
the frozen epoch-63 checkpoint and its recorded validation row, then uses only
the approved 2,602-image validation split (1,619 positive, 983 negative-only;
3,377 targets: D00 822, D10 576, D20 1,188, D40 791).

Installed Ultralytics 8.4.130 runs this YOLO26 `end2end=true` head through its
one-to-one branch. The head selects its native top-k outputs and the predictor's
end-to-end branch applies the confidence filter without legacy NMS. Therefore
NMS IoU is **not applicable and is not supplied or swept**. The selector runs
one logical native prediction pass in explicit four-image batches at confidence floor 0.010, caches all retained
validation predictions, and evaluates one global confidence from 0.010 through
0.800 in increments of 0.001. Matching is class-aware at IoU >= 0.50. Selection
maximizes exact four-class macro-F1, then minimizes false positives per
negative-only image, then chooses the higher global confidence. Per-class
thresholds are forbidden.

Preflight (no inference):

```powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\select_experiment2_threshold.py --preflight
```

Real validation-only cache and sweep:

```powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\select_experiment2_threshold.py
```

Offline cache replay (no model or dataset access):

```powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\select_experiment2_threshold.py --recompute-cache
```

The run refuses to overwrite either output directory. A completion receipt is
published only after the saved prediction/ground-truth caches, metric sweep and
selected point are reopened, hash-verified and reproduced. Internal-test data,
official unlabelled test images, training images and teacher video are excluded.
The baseline comparison uses each model's own validation-selected operating
point; YOLO26 is not forced to reuse YOLOv8s confidence or NMS settings.

## Validation-only error analysis

The cache-only error analysis is implemented by
`src/road_damage/evaluation/experiment2_error_analysis.py` with the frozen
protocol in `configs/evaluation/experiment2_yolo26s_error_analysis.yaml`:

```powershell
.\.venv\Scripts\python.exe src\road_damage\evaluation\experiment2_error_analysis.py
```

It verifies and replays the approved YOLO26 prediction cache at global
confidence 0.193 and matching IoU 0.50. It does not import Ultralytics, load the
model, or read canonical dataset files. Every FN and FP receives one
deterministic cache-observable category. FN priority distinguishes
below-confidence correct-class candidates, same-class matching competition,
two localization IoU bands, retained wrong-class overlap, and exactly “no
stronger cached candidate above the 0.010 inference floor.” The last phrase
must not be interpreted as “the model saw nothing.”

Object-size, aspect-ratio, target-density and country summaries are descriptive
associations only. Bounding-box footprint is not physical damage area or
engineering severity. India D10 conclusions remain explicitly limited by its
10 targets in 10 validation images. The approved YOLOv8 error-analysis manifest
and artifacts are hash-verified and ingested for comparison; YOLOv8 inference
is not repeated. YOLO26 same-class overlapping unmatched predictions are
reported as observable native end-to-end output, without inventing legacy-NMS
duplicate semantics.

The output is `outputs/evaluation/experiment2_yolo26s_error_analysis/`. It is
never overwritten, and `completion.json` is written only after every persisted
machine-readable artifact has been reread and hash-verified. Internal-test
data, official unlabelled test data, teacher video, training and threshold
tuning remain outside this analysis.

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
YOLOv8s NMS sweep and imply its NMS IoU has identical meaning. Validation
threshold-selection and validation error-analysis tooling now implement this
native end-to-end treatment.

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
checking runs only during a user-started smoke/full/resumed training operation, never
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

## Fresh full training — refuses the existing run

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

`--train` remains fresh-run only and rejects the current existing directory.
`--smoke` remains exactly three epochs / 96 batches with its existing gates.
Only the dedicated, narrowly pinned `--resume` operation below may reuse the
interrupted full-run directory. Raw Ultralytics resume commands are forbidden.

## Controlled repeated resume — USER operation after review/commit

```powershell
.\.venv\Scripts\python.exe src\road_damage\training\experiment2.py --resume
```

Do not run this command until the durable-state correction has been reviewed and committed.
The CLI accepts no checkpoint, model, output, dataset, epoch, LR, augmentation,
or other scientific override. Every invocation resolves only the canonical current:

`outputs/training/experiment2_yolo26s_matched/yolo26s_rdd2022-india-japan-v1.1.0_640_seed42/weights/last.pt`

Current authoritative epoch-60 checkpoint SHA-256:
`335a4b825d1dd30f80caf68d844e12f023ad3a5e873bb9dd1d851ec9a946e5e4`.
Size: **40,377,697 bytes**. Stored epoch is 59 (zero-based), so the next human
epoch is **61/100**, not a new 40-epoch experiment.

Historical segments are immutable:

- epochs 1–33: original training, commit `39070294088a725edf9092d861e5da4d750acd25`;
- epochs 34–60: first controlled resume, commit `0312871fb275725699913e08baaacab570c7296f`,
  source `7740bc7d2c1516969f081885ea4fe317ef3c9237cb5ce5612b01e4474e44d28a`;
- partial epoch 61: interrupted, not a completed epoch and not a new result row.

The old resume callback wrote immutable per-epoch records but did not advance
the root `epoch_state.json`. The root file consequently still describes epoch 33.
The correction does **not** manually rewrite this real file during implementation.
For this reviewed migration only, the launcher verifies the exact epoch-60
checkpoint, 60 contiguous complete results rows, the byte-pinned first-attempt
records and epoch-60 journal, and the known stale root-state bytes. It reconciles
checkpoint epoch 59, optimizer/scaler/EMA state, and counters before accepting
epoch 60 in memory. No other stale state is silently repaired.

Durable progress at this boundary is **60 epochs / 189,300 batches / 13,038
optimizer attempts / 13,026 successful updates**. The interruption observed
189,556 batches / 13,054 attempts / 13,042 updates. The extra **256 batches and
16 optimizer operations** in partial epoch 61 are discarded; epoch 61 is replayed
from batch 1. Failure/observed counters never initialize durable progress.

The native call uses the explicit canonical checkpoint as the sole `resume`
argument. Installed Ultralytics 8.4.130 trainer/model/torch-helper/metrics/validator sources are
also hash-checked. Native resume loads the original SGD state (366 momentum
buffers in three parameter groups), GradScaler (currently scale 512, growth tracker 864),
EMA network and EMA counter. The project compares those restored states with the
checkpoint before the first resumed epoch. It does not reset or manually step
the optimizer, scaler, or EMA. The successful-step post-hook retains its existing
meaning and is removed when the invocation ends.

Ultralytics reconstructs the cosine schedule with the original 100-epoch target,
sets `start_epoch=60`, then `scheduler.last_epoch=59`. The next resumed epoch
advances the scheduler to zero-based epoch 60; LR does not restart at epoch 1.
Project callbacks verify the scheduler position and LR without overriding them.
At `on_pretrain_routine_end`, after native checkpoint restoration and before the
first resumed epoch, the project restores EarlyStopping from the complete results.
Installed detection fitness uses mAP50–95 alone, weights `[0, 0, 0, 1]`; the
training validator rounds this and the recorded metric to five decimal places.
The installed stopper accepts strictly greater fitness, except while its best
fitness is zero. Equal nonzero values do not restart patience.
Through epoch 60 this gives **best_epoch=59, best_fitness=0.21557, patience=20,
epochs_without_improvement=1, patience_exhausted=false, possible_stop=false**.
Checkpoint best fitness and epoch journals must agree with this reconstruction.

The original segment remains attributed to commit
`39070294088a725edf9092d861e5da4d750acd25` and source fingerprint
`a33facf065a4004fd32f673df716c05b1f5783dbd5899481ef99c957ac1b2b57`.
The frozen config remains
`0ae18ab9fa2d1098eefed24b2b75f5ed6f94b9e2bce773396d6402339cdaafe9`.
Resume requires a clean descendant of the first-resume commit, changing only the
explicit resume tooling/docs/tests allowlist. The new commit and source fingerprint
apply only to its new segment, beginning at epoch 61 for the current migration.

An OS-held, nonblocking run lock prevents concurrent resume processes and releases
on process death. Attempts are numbered `0002`, `0003`, etc., created exclusively;
earlier attempts are never reused or overwritten. Each new `STARTED.json` records
the complete prior segment list, hashes of prior history, input identities and
discarded partial progress. Original input metadata/checkpoint bytes are copied
into the new attempt's `input_snapshot/`; native args are saved in that attempt.
The original manifest, failure, args and first-attempt history remain immutable.

After each native `on_model_save`, the callback requires a complete batch count,
finite contiguous CSV rows, a full matching checkpoint, and matching stopper state.
It writes/fsyncs a new immutable `epochs/epoch_NNN.json`, then writes/fsyncs a
temporary root-state file and atomically replaces `epoch_state.json`. That pointer
records the completed epoch, checkpoint SHA/size, results SHA, optimizer counters,
optimizer/scaler presence, EarlyStopping fields, source/config identities, and
epoch-journal path/SHA. Durable in-memory progress advances only after publication.
Partial epochs cannot publish this state.

Subsequent pauses use the latest fully committed project state, without another
hard-coded checkpoint approval. Checkpoint epoch + 1, complete CSV row count,
last CSV epoch, root state, immutable journal, and checkpoint SHA must all agree.
Unknown/gapped history, stripped/missing state, exhausted patience, scientific
drift, or a completed receipt fails closed. An interrupted publication leaves a
detectable disagreement and requires review; there is no force mode or arbitrary
stale-state repair. An abrupt process exit is resumable only when this complete
durable evidence remains consistent; unfinished history is retained explicitly.

An interruption/error appends `INTERRUPTED.json`/`FAILED.json` with separate
observed and durable progress. Completion requires the target or exhausted
patience, finite complete evidence, verified resulting checkpoints, matching root
state and every saved epoch record. `COMPLETED.json` and the canonical receipt
retain all training segments and epoch-record hashes. Native final checkpoint
stripping occurs only on normal completion; completed runs cannot resume.
If final receipt publication fails after the attempt's completion record is
durable, preserve the evidence and stop for review rather than rerun training.

Before execution, the original config, pretrained lineage, train/val YAML and
export metadata hashes are verified, followed by the same train/val-only byte
verification. Dataset/cache files remain immutable. Runtime paths and class
mapping are checked before dataset loading and before the first resumed epoch.
Every resume record declares `internal_test_files_accessed=false`; individual
internal-test data remains locked and is never used for training, validation,
inference, debugging, or threshold selection.

Checkpoint selection and the operating-point selection above use validation
only, with the same metric philosophy as the baseline and explicit native
end-to-end output semantics. No internal-test evaluation is authorized here.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_experiment2*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use synthetic files and mocked training calls. They must not acquire
real weights, run model inference, or access individual internal-test data.
Experiment 1 configs, runner, model, provenance, source-state manifest and
completed internal-test outputs remain immutable. The source-state builder is
reused in memory; this task does not regenerate the baseline manifest.
