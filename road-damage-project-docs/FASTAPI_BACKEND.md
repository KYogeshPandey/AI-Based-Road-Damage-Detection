# Phase 5B Secure Video Upload and Phase 4 Integration

## Scope and scientific boundary

Phase 5B accepts a user video, stages it in API-owned storage, performs a
lightweight OpenCV decodability check, and runs the existing Phase 4
`analyze_video(...)` application service as a FastAPI background task. It does
not duplicate inference, tracking, temporal aggregation, annotation, or report
generation.

The API exposes no checkpoint, model, confidence, NMS, class, device, tracking,
aggregation, or output-path controls. The frozen Phase 4 configuration and
checkpoint identity remain authoritative. Importing the ASGI application does
not import Ultralytics, PyTorch, or the Phase 4 analysis module; that module is
loaded only after an accepted job begins execution.

## Install and start

Backend dependencies are separate from the frozen detector requirements:

```powershell
python -m pip install -r requirements-backend.txt
python -m uvicorn road_damage.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

OpenAPI documentation is at `/docs` and the schema at `/openapi.json` when
`ROAD_DAMAGE_API_DOCS_ENABLED` is true. Starting Uvicorn does not load the
model; the first accepted background analysis does.

## Endpoints

All routes use `/api/v1`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Cheap process liveness with no model or GPU access |
| `GET` | `/system` | Frozen class and application capabilities |
| `GET` | `/analyses/capabilities` | Upload/execution state, extensions, and size limit |
| `POST` | `/analyses` | One multipart upload under the field `file`; returns `202` |
| `GET` | `/analyses/{job_id}` | Current immutable job status and artifact availability |

Only `.mp4`, `.avi`, `.mov`, and `.mkv` uploads are accepted. The client may
provide only a display filename and bytes—not a server path or scientific
setting.

### Manual PowerShell submission and polling

These commands are examples only. Choose a permission-safe local road video;
do not use protected test data.

```powershell
$response = curl.exe -sS -X POST -F "file=@C:\path\to\road-video.mp4" http://127.0.0.1:8000/api/v1/analyses
$job = $response | ConvertFrom-Json
$job
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/analyses/$($job.job_id)"
```

## Upload and storage safety

Uploads are copied in bounded chunks and the maximum byte count is enforced
while streaming. Before Starlette parses or spools multipart data, a targeted
ASGI receive guard also counts the raw request-body bytes. The raw-body ceiling
is the configured video-file limit plus a fixed 1,048,576-byte allowance for
multipart boundaries and headers. This keeps ingress bounded while still
allowing a file whose size is exactly the configured file maximum. Both a
declared `Content-Length` and the actual streamed bytes are checked, so absent
or understated headers do not bypass the guard.

Empty content, unsupported extensions, unsafe filenames, extra multipart
controls, oversized content, and undecodable video are rejected with stable
errors. Rejected or incomplete uploads remove their newly-created UUID job
directory.

An accepted job uses this controlled layout:

```text
outputs/api/<job-uuid>/
|-- input/
|   `-- video.<validated-extension>
`-- analysis/
    |-- annotated_video.mp4
    |-- run_manifest.json
    |-- summary.json
    |-- events.json
    |-- frame_detections.jsonl
    `-- completion.json
```

The original safe filename is metadata only. It never becomes a server path.
The `analysis/` directory is supplied directly to Phase 4 and is not created in
advance. Accepted inputs and any partial analysis artifacts are retained when
execution fails so the job remains auditable.

The UUID parent directory is reserved exclusively with fail-if-present
semantics before either child directory is created. A UUID collision or other
pre-existing parent is never reused or removed. Recursive cleanup is permitted
only when the current request successfully created that exact parent.

Before scheduling, OpenCV opens the staged file, checks positive dimensions,
frame rate and frame count, and decodes one frame. This validation does not
load the detector, read a checkpoint, inspect a GPU, or run inference.

## Lifecycle and truthful completion

The process-local registry is thread-safe and stores immutable response
objects. Allowed transitions are:

```text
QUEUED -> RUNNING -> COMPLETED
       `-> FAILED  `-> FAILED
                    `-> INTERRUPTED
```

Timestamps are timezone-aware and monotonic within a job. Running progress may
remain unknown rather than being fabricated. A job can become `COMPLETED` only
after an independent API-owned validator reconciles the actual frozen Phase 4
contract. It validates the completion, manifest, summary, and events schemas;
`COMPLETED` statuses; run IDs; manifest SHA-256; the exact required artifact
hash mapping; model and staged-input identities; frame/event totals; safety
flags; and the absence of Phase 4's `.run.lock`. Malformed, missing,
inconsistent, or tampered evidence fails closed. Physical artifact availability
may still be reported for a failed job, but file presence alone never proves
completion.

Failure messages returned to clients are stable and do not contain exception
text or local paths. Artifact availability flags are derived from the actual
files present.

The registry is intentionally in-memory for Phase 5B: jobs survive request
boundaries within one process, but not a server restart. A durable database,
result download endpoints, authentication, and a distributed queue remain
future work.

## Error contract

Errors use one envelope:

```json
{
  "error": {
    "code": "INVALID_UPLOAD",
    "message": "The uploaded video is invalid.",
    "details": null
  }
}
```

Stable upload/job codes include `INVALID_UPLOAD`, `UPLOAD_TOO_LARGE`,
`UNSUPPORTED_VIDEO_TYPE`, `VIDEO_VALIDATION_FAILED`, `UPLOAD_FAILED`,
`ANALYSIS_NOT_FOUND`, and `ANALYSIS_EXECUTION_FAILED` (in failed job state).

## Backend configuration

Only these runtime environment variables are accepted:

| Variable | Default | Meaning |
|---|---|---|
| `ROAD_DAMAGE_API_HOST` | `127.0.0.1` | Bind host used by launch tooling |
| `ROAD_DAMAGE_API_PORT` | `8000` | Server port used by launch tooling |
| `ROAD_DAMAGE_API_DEBUG` | `false` | Logging integration flag; responses remain non-debug |
| `ROAD_DAMAGE_API_DOCS_ENABLED` | `true` | Enable local docs and OpenAPI routes |
| `ROAD_DAMAGE_API_MAX_UPLOAD_BYTES` | `536870912` | Maximum streamed upload size |
| `ROAD_DAMAGE_API_UPLOAD_CHUNK_BYTES` | `1048576` | Bounded streaming chunk size |
| `ROAD_DAMAGE_API_OUTPUT_ROOT` | `outputs/api` | Dedicated output child directory |
| `ROAD_DAMAGE_API_CORS_ORIGINS` | empty | Explicit comma-separated origins |
| `ROAD_DAMAGE_API_CORS_ALLOW_CREDENTIALS` | `false` | Credentials only with an allowlist |

The output root must remain a dedicated child of project `outputs/`. CORS is
off by default; when enabled, only explicit HTTP(S) origins and `GET`/`POST`
methods are allowed.
