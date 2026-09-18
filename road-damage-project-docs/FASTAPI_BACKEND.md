# Phase 5A FastAPI Backend Foundation

## Status and boundary

Phase 5A provides a versioned HTTP boundary around the completed Phase 4
application contracts. It does **not** accept analysis submissions, run model
inference, load a checkpoint, inspect CUDA, or read datasets. The execution
service is intentionally disabled until Phase 5B connects a reviewed job
runner to the existing `analyze_video(...)` application service.

The API is a local application foundation, not a cloud deployment. It does not
change the V1 scientific scope or any frozen Phase 4 setting.

## Package structure

```text
src/road_damage/api/
|-- app.py                 # create_app() and importable ASGI app
|-- config.py              # strict backend-only environment settings
|-- dependencies.py        # request-scoped service/settings providers
|-- errors.py              # stable public error envelope and handlers
|-- paths.py               # filename and output-root confinement
|-- routes/
|   |-- health.py
|   |-- system.py
|   `-- analyses.py
|-- schemas/
|   |-- common.py
|   |-- health.py
|   `-- analysis.py
`-- services/
    `-- analysis_service.py
```

Route handlers contain only HTTP translation and dependency calls. The
analysis-service protocol is the application boundary; persistence and
execution implementations belong behind that interface.

## Install and start

Keep backend dependencies separate from the frozen detector requirements:

```powershell
python -m pip install -r requirements-backend.txt
```

No dependency installation is performed by the application itself. From the
repository root, the exact development start command is:

```powershell
python -m uvicorn road_damage.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `/docs` and the schema at
`/openapi.json` when `ROAD_DAMAGE_API_DOCS_ENABLED` is true (the development
default).

## Versioning and endpoints

The API version is `1.0.0`; all application routes use `/api/v1`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Cheap process liveness; no model, GPU, checkpoint, or data access |
| `GET` | `/api/v1/system` | Safe public capabilities and the frozen four-class contract |
| `GET` | `/api/v1/analyses/capabilities` | Explicitly reports that submission and execution are disabled |

There is deliberately no `POST /api/v1/analyses` route and no fake completed
job. Requests to that unregistered path receive the standard safe `404
NOT_FOUND` envelope; `405 METHOD_NOT_ALLOWED` would imply that the exact path
is registered for another method.

### Health response

```json
{
  "status": "ok",
  "service": "AI-Based Road Damage Detection API",
  "api_version": "1.0.0"
}
```

### System response

The system response exposes only stable application facts: API version, the
four canonical class IDs/names plus friendly display names, model family
`YOLOv8s`, operating-point status `frozen_validation_selected`, Phase 4 output
artifact names, and the disabled Phase 5A execution capability. It never
returns absolute paths, host information, secrets, checkpoint bytes, dataset
details, or CUDA information.

## Job lifecycle contract

The future application job lifecycle is intentionally distinct from the
Phase 4 pipeline-run status:

```text
QUEUED -> RUNNING -> COMPLETED
                  -> FAILED
                  -> INTERRUPTED
```

Job responses have a UUID, status, timezone-aware timestamps, progress,
display-only input filename, artifact-availability flags, and an optional
safe error. Progress uses non-negative frame counts; unknown totals and
percentages remain `null` rather than being fabricated.

Lifecycle validation rejects contradictory records. `QUEUED` has no start,
completion, or error; `RUNNING` has a start but no completion or error;
`COMPLETED` has both timestamps and no error; and `FAILED` has both timestamps
plus an error. `INTERRUPTED` has both timestamps and may optionally include
safe error metadata, preserving interruption details when available. Timestamp
ordering is `created_at <= started_at <= completed_at <= updated_at`, omitting
the optional timestamps only where the lifecycle permits it.

The Phase 4 result projection preserves `COMPLETED`, `FAILED_TECHNICAL`, and
`INTERRUPTED` as a separate vocabulary. Summary contracts include inferred
frames, raw detection-observation totals, heuristic temporal-event totals,
confirmed/tentative totals, per-class counts, and processing/inference timing.
Overall raw-observation, event, confirmed-event, and tentative-event totals
must each reconcile with their per-class counts; inconsistent records fail
validation rather than being repaired.

Event responses preserve canonical machine-readable `class_id`, `class_name`,
and `event_id`, while also exposing `friendly_display_name`. Event totals are
described as heuristic temporal damage events, not ground-truth counts of
unique physical-world defects.

## Error contract

Validation, missing-resource, known application, method, and internal errors
use one envelope:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Request validation failed.",
    "details": {}
  }
}
```

Validation details contain only field locations, safe messages, and error
types. Unexpected exceptions return a generic message; tracebacks, absolute
paths, environment variables, and exception representations are not exposed.

## Backend configuration

Only these runtime environment variables are accepted:

| Variable | Default | Meaning |
|---|---|---|
| `ROAD_DAMAGE_API_HOST` | `127.0.0.1` | Server bind host used by launch tooling |
| `ROAD_DAMAGE_API_PORT` | `8000` | Server port used by launch tooling |
| `ROAD_DAMAGE_API_DEBUG` | `false` | Reserved for safe logging integration; error bodies remain non-debug |
| `ROAD_DAMAGE_API_DOCS_ENABLED` | `true` | Enables development docs/OpenAPI routes |
| `ROAD_DAMAGE_API_MAX_UPLOAD_BYTES` | `536870912` | Future upload boundary limit |
| `ROAD_DAMAGE_API_OUTPUT_ROOT` | `outputs/api` | Controlled output root, required to remain below `outputs/` |
| `ROAD_DAMAGE_API_CORS_ORIGINS` | empty | Comma-separated explicit origins |
| `ROAD_DAMAGE_API_CORS_ALLOW_CREDENTIALS` | `false` | Credentialed CORS for an explicit allowlist only |

The loader uses a strict allowlist. Model paths, checkpoint selection,
confidence, NMS IoU, image size, class mapping, device, aggregation settings,
and all other scientific controls cannot be overridden through API environment
variables.

## Path and CORS safety

Future client input may provide a base filename for display, but never an
arbitrary server path. Absolute Windows/POSIX paths, drive names, separators,
`..`, alternate data-stream syntax, and NUL characters are rejected. Job
directories use canonical UUIDs and are confined below the configured output
root. The output root itself must be a dedicated child of project `outputs/`,
which prevents API writes into `src/`, `configs/`, `models/`, or `data/`.

CORS middleware is absent by default. When enabled it requires explicit
`http://` or `https://` origins; wildcard origins are rejected. Allowed methods
in Phase 5A are read-only `GET` requests.

## Phase 5B integration boundary

`AnalysisService` defines future create/get/summary/events operations.
`AnalysisExecutionDisabledService` is the only Phase 5A implementation. Phase
5B may implement asynchronous job persistence and invoke the existing shared
application service, but it must not duplicate inference logic or introduce
API-level scientific overrides.

## Current limitations

- No video upload or submission endpoint.
- No analysis execution, queue, worker, or database.
- No result-download endpoint or authentication layer.
- No cloud/deployment configuration.
- Backend dependencies are not part of the frozen training environment and
  must be installed explicitly from `requirements-backend.txt` before runtime
  API tests or server startup.
