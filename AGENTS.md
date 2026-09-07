# Codex and Contributor Rules

These instructions apply to the entire future project repository. Copy this file to the repository root before coding so Codex discovers it automatically.

## 1. Authority

Read `docs/PROJECT_PRD.md` before making architectural or behavioral changes. If requirements conflict, follow the priority stated in `docs/README.md`. Do not expand V1 scope without updating the PRD.

## 2. Non-negotiable project rules

1. Use exactly four V1 labels and preserve their numeric order:
   - `0 D00_longitudinal_crack`
   - `1 D10_transverse_crack`
   - `2 D20_alligator_crack`
   - `3 D40_pothole`
2. Do not claim physical pothole depth, area, volume, or engineering severity from monocular video.
3. A tracker ID is not a final damage ID. Final counts come only from the `DamageEvent` aggregator.
4. Never randomly split adjacent video frames. Assign the split at the source-segment level.
5. Never fabricate metrics, result tables, dataset sizes, or performance claims.
6. Do not use test data to tune thresholds, select checkpoints, or debug model behavior.
7. Do not commit raw videos, public datasets, secrets, large weights, generated outputs, or personal-location metadata to Git.
8. Do not duplicate inference logic in the dashboard. Both CLI and UI must call the same application service.
9. Configuration values belong in versioned YAML files; environment-specific paths belong in environment variables or CLI arguments.
10. Preserve provenance for every generated frame, label, model, report, and experiment.

## 3. Engineering conventions

- Target Python 3.12.10 for the approved baseline experiment; do not change the experiment interpreter without a reviewed environment update.
- Use type hints for public functions and dataclasses/Pydantic-style models for boundary data.
- Keep modules focused: video, dataset, detection, tracking, aggregation, severity, rendering, reporting, and UI.
- Prefer pure functions for geometry and decision rules.
- Use `pathlib.Path`; do not hard-code Windows- or Linux-specific path separators.
- Use structured logging. Do not rely on ad-hoc `print` statements in library code.
- Raise actionable domain errors at boundaries and preserve original exceptions as causes.
- Make random behavior seedable.
- Store timestamps as integer milliseconds and frame indices as zero-based integers.
- Store bounding boxes in an explicit format; names must include `xyxy`, `xywh`, `pixel`, or `normalized` where ambiguity is possible.
- Never silently discard a detection or event. Record the rejection reason at debug/audit level.

## 4. Required workflow for each change

1. Identify the PRD requirement or milestone being implemented.
2. Inspect existing code and tests before editing.
3. Make the smallest coherent change.
4. Add or update tests.
5. Run targeted tests, then the broader relevant suite.
6. Run lint/type checks when configured.
7. Update documentation and example config if behavior changed.
8. Summarize what changed, commands run, results, limitations, and any unresolved risk.

## 5. Testing requirements

Unit tests are mandatory for:

- timestamp/frame conversions;
- ROI point-in-polygon decisions;
- bbox smoothing;
- class-vote logic;
- track confirmation and rejection;
- event finalization and fragment merging;
- severity-zone selection and thresholds;
- CSV/JSON serialization;
- configuration validation.

Integration tests must include:

- a tiny synthetic or permission-safe sample video;
- mocked detector results so event logic is testable without GPU weights;
- end-to-end generation of a report and output video;
- failure cases for unreadable video and missing model.

Tests must not download large models or datasets by default.

## 6. Data safety

- Treat uploaded dashcam video as private by default.
- Keep raw data immutable.
- Write derivatives to separate directories.
- Never rename frames in a way that loses their source timestamp or manifest mapping.
- Verify output paths before long runs and avoid overwriting earlier experiment results.
- Store credentials only in ignored environment files; provide `.env.example` without values if credentials later become necessary.

## 7. Experiment discipline

- Every run receives a unique `run_id` and immutable output directory.
- Save the resolved configuration, Git commit, package versions, model identity/checksum, seed, device, input fingerprint, and metrics.
- Compare models using the same dataset version, split, image size, and evaluation code unless the experiment explicitly studies one of those factors.
- Tune on validation data. Evaluate the test set only after choices are frozen.
- Mark exploratory results as exploratory; do not copy them into the final results table.

## 8. Definition of done for code tasks

A task is complete only when behavior works, relevant tests pass, configuration/docs are updated, outputs are validated, and no metric or capability is claimed without evidence.
