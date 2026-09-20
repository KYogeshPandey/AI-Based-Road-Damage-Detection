"""ASGI-level upload ingress limits applied before multipart parsing."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from road_damage.api.errors import error_payload


MULTIPART_OVERHEAD_ALLOWANCE_BYTES = 1_048_576

AsgiApplication = Callable[
    [dict[str, Any], Callable[[], Awaitable[dict[str, Any]]], Callable[[dict[str, Any]], Awaitable[None]]],
    Awaitable[None],
]


class _RequestBodyTooLarge(StarletteHTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=413, detail="Upload request body is too large.")


def upload_request_body_limit(max_upload_bytes: int) -> int:
    """Bound the file plus conservative multipart framing/header overhead."""
    return max_upload_bytes + MULTIPART_OVERHEAD_ALLOWANCE_BYTES


class UploadRequestBodyLimitMiddleware:
    """Count upload request bytes before Starlette can spool multipart content."""

    def __init__(
        self,
        app: AsgiApplication,
        *,
        maximum_body_bytes: int,
        upload_path: str,
    ) -> None:
        if maximum_body_bytes < 1:
            raise ValueError("maximum_body_bytes must be positive.")
        self._app = app
        self._maximum_body_bytes = maximum_body_bytes
        self._upload_path = upload_path

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self._upload_path
        ):
            await self._app(scope, receive, send)
            return

        content_lengths = [
            value
            for name, value in scope.get("headers", ())
            if name.lower() == b"content-length"
        ]
        if content_lengths:
            try:
                parsed_lengths = {int(value.decode("ascii")) for value in content_lengths}
            except (UnicodeError, ValueError) as exc:
                await self._invalid_length(scope, receive, send)
                return
            if len(parsed_lengths) != 1 or next(iter(parsed_lengths)) < 0:
                await self._invalid_length(scope, receive, send)
                return
            if next(iter(parsed_lengths)) > self._maximum_body_bytes:
                await self._too_large(scope, receive, send)
                return

        received_bytes = 0
        response_started = False

        async def guarded_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                received_bytes += len(body)
                if received_bytes > self._maximum_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def guarded_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, guarded_receive, guarded_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await self._too_large(scope, receive, send)

    async def _too_large(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        response = JSONResponse(
            status_code=413,
            content=error_payload(
                "UPLOAD_TOO_LARGE",
                "The HTTP request body exceeds the configured upload limit.",
                {"maximum_request_bytes": self._maximum_body_bytes},
            ),
        )
        await response(scope, receive, send)

    @staticmethod
    async def _invalid_length(
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        response = JSONResponse(
            status_code=400,
            content=error_payload(
                "INVALID_UPLOAD", "The request Content-Length is invalid."
            ),
        )
        await response(scope, receive, send)
