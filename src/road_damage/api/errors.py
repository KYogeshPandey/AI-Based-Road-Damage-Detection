"""Stable, non-leaking API error contracts."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiApplicationError(RuntimeError):
    """Known application error safe to expose through the public API."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class ResourceNotFoundError(ApiApplicationError):
    def __init__(self, resource: str) -> None:
        super().__init__(
            code="NOT_FOUND",
            message=f"{resource} was not found.",
            status_code=404,
        )


class AnalysisResultRetrievalUnavailableError(ApiApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="ANALYSIS_RESULT_RETRIEVAL_UNAVAILABLE",
            message="Full analysis result retrieval is not enabled in Phase 5B.",
            status_code=503,
        )


class InvalidUploadError(ApiApplicationError):
    def __init__(self, message: str = "The uploaded video is invalid.") -> None:
        super().__init__(code="INVALID_UPLOAD", message=message, status_code=400)


class UnsupportedVideoTypeError(ApiApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="UNSUPPORTED_VIDEO_TYPE",
            message="The uploaded file extension is not supported.",
            status_code=415,
        )


class UploadTooLargeError(ApiApplicationError):
    def __init__(self, maximum_bytes: int) -> None:
        super().__init__(
            code="UPLOAD_TOO_LARGE",
            message="The uploaded video exceeds the configured size limit.",
            status_code=413,
            details={"maximum_bytes": maximum_bytes},
        )


class VideoValidationFailedError(ApiApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="VIDEO_VALIDATION_FAILED",
            message="The uploaded file is not a supported decodable video.",
            status_code=422,
        )


class UploadStorageError(ApiApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="UPLOAD_FAILED",
            message="The video upload could not be stored.",
            status_code=500,
        )


class AnalysisNotFoundError(ApiApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="ANALYSIS_NOT_FOUND",
            message="Analysis job was not found.",
            status_code=404,
        )


def error_payload(code: str, message: str, details: Any | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


async def _application_error_handler(
    _request: Request, exc: ApiApplicationError
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(exc.code, exc.message, exc.details),
    )


async def _request_validation_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    violations = [
        {
            "location": [str(part) for part in error.get("loc", ())],
            "message": str(error.get("msg", "Invalid value.")),
            "type": str(error.get("type", "value_error")),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_payload(
            "INVALID_REQUEST",
            "Request validation failed.",
            {"violations": violations},
        ),
    )


async def _http_error_handler(
    _request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content=error_payload("NOT_FOUND", "Resource not found."),
        )
    if exc.status_code == 405:
        return JSONResponse(
            status_code=405,
            content=error_payload("METHOD_NOT_ALLOWED", "Method not allowed."),
        )
    if exc.status_code == 413:
        return JSONResponse(
            status_code=413,
            content=error_payload(
                "UPLOAD_TOO_LARGE",
                "The HTTP request body exceeds the configured upload limit.",
            ),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload("HTTP_ERROR", "The request could not be completed."),
    )


async def _unexpected_error_handler(
    _request: Request, _exc: Exception
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_payload(
            "INTERNAL_SERVER_ERROR", "An unexpected server error occurred."
        ),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register the stable error envelope for expected and unexpected errors."""
    app.add_exception_handler(ApiApplicationError, _application_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(Exception, _unexpected_error_handler)
