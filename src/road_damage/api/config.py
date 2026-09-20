"""Backend-only runtime configuration.

Scientific inference settings remain owned by the frozen, versioned Phase 4
configuration. They cannot be supplied through this environment loader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "api"
ENV_PREFIX = "ROAD_DAMAGE_API_"

_ENV_FIELDS = MappingProxyType(
    {
        "HOST": "host",
        "PORT": "port",
        "DEBUG": "debug",
        "DOCS_ENABLED": "docs_enabled",
        "MAX_UPLOAD_BYTES": "max_upload_bytes",
        "UPLOAD_CHUNK_BYTES": "upload_chunk_bytes",
        "OUTPUT_ROOT": "output_root",
        "CORS_ORIGINS": "cors_origins",
        "CORS_ALLOW_CREDENTIALS": "cors_allow_credentials",
    }
)


class BackendConfigurationError(ValueError):
    """Raised when backend runtime configuration is unsafe or invalid."""


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise BackendConfigurationError(f"{name} must be a boolean value.")


def _parse_int(name: str, value: str, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BackendConfigurationError(f"{name} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise BackendConfigurationError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return parsed


def resolve_output_root(value: str | Path) -> Path:
    """Resolve an administrator-configured output root under ``outputs/``."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    allowed_root = (PROJECT_ROOT / "outputs").resolve(strict=False)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise BackendConfigurationError(
            "ROAD_DAMAGE_API_OUTPUT_ROOT must remain under the project outputs directory."
        ) from exc
    if resolved == allowed_root:
        raise BackendConfigurationError(
            "ROAD_DAMAGE_API_OUTPUT_ROOT must be a dedicated child of outputs."
        )
    return resolved


def _parse_cors_origins(value: str) -> tuple[str, ...]:
    origins = tuple(item.strip().rstrip("/") for item in value.split(",") if item.strip())
    if any(origin == "*" for origin in origins):
        raise BackendConfigurationError("Wildcard CORS origins are not permitted.")
    for origin in origins:
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BackendConfigurationError(
                "CORS origins must be explicit http:// or https:// origins."
            )
        if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
            raise BackendConfigurationError(
                "CORS origins must not contain a path, query, or fragment."
            )
    if len(origins) != len(set(origins)):
        raise BackendConfigurationError("CORS origins must not contain duplicates.")
    return origins


@dataclass(frozen=True, slots=True)
class BackendSettings:
    """Runtime-only API settings; no model or scientific settings belong here."""

    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    docs_enabled: bool = True
    max_upload_bytes: int = 536_870_912
    upload_chunk_bytes: int = 1_048_576
    output_root: Path = DEFAULT_OUTPUT_ROOT
    cors_origins: tuple[str, ...] = ()
    cors_allow_credentials: bool = False

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise BackendConfigurationError("API host must not be empty.")
        if not 1 <= self.port <= 65_535:
            raise BackendConfigurationError("API port must be between 1 and 65535.")
        if self.max_upload_bytes < 1:
            raise BackendConfigurationError("Maximum upload size must be positive.")
        if not 1 <= self.upload_chunk_bytes <= 16 * 1024 * 1024:
            raise BackendConfigurationError(
                "Upload chunk size must be between 1 and 16777216 bytes."
            )
        object.__setattr__(self, "output_root", resolve_output_root(self.output_root))
        parsed_origins = _parse_cors_origins(",".join(self.cors_origins))
        object.__setattr__(self, "cors_origins", parsed_origins)
        if self.cors_allow_credentials and not parsed_origins:
            raise BackendConfigurationError(
                "Credentialed CORS requires an explicit origin allowlist."
            )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "BackendSettings":
        """Load a strict allowlist of backend settings from environment variables."""
        source = os.environ if environ is None else environ
        supplied = {
            key: value for key, value in source.items() if key.startswith(ENV_PREFIX)
        }
        unknown = sorted(set(supplied).difference(f"{ENV_PREFIX}{key}" for key in _ENV_FIELDS))
        if unknown:
            raise BackendConfigurationError(
                "Unsupported API environment setting(s): " + ", ".join(unknown)
            )

        def get(name: str) -> str | None:
            return supplied.get(f"{ENV_PREFIX}{name}")

        kwargs: dict[str, object] = {}
        if (value := get("HOST")) is not None:
            kwargs["host"] = value
        if (value := get("PORT")) is not None:
            kwargs["port"] = _parse_int(
                "ROAD_DAMAGE_API_PORT", value, minimum=1, maximum=65_535
            )
        if (value := get("DEBUG")) is not None:
            kwargs["debug"] = _parse_bool("ROAD_DAMAGE_API_DEBUG", value)
        if (value := get("DOCS_ENABLED")) is not None:
            kwargs["docs_enabled"] = _parse_bool(
                "ROAD_DAMAGE_API_DOCS_ENABLED", value
            )
        if (value := get("MAX_UPLOAD_BYTES")) is not None:
            kwargs["max_upload_bytes"] = _parse_int(
                "ROAD_DAMAGE_API_MAX_UPLOAD_BYTES",
                value,
                minimum=1,
                maximum=100 * 1024 * 1024 * 1024,
            )
        if (value := get("UPLOAD_CHUNK_BYTES")) is not None:
            kwargs["upload_chunk_bytes"] = _parse_int(
                "ROAD_DAMAGE_API_UPLOAD_CHUNK_BYTES",
                value,
                minimum=1,
                maximum=16 * 1024 * 1024,
            )
        if (value := get("OUTPUT_ROOT")) is not None:
            kwargs["output_root"] = Path(value)
        if (value := get("CORS_ORIGINS")) is not None:
            kwargs["cors_origins"] = _parse_cors_origins(value)
        if (value := get("CORS_ALLOW_CREDENTIALS")) is not None:
            kwargs["cors_allow_credentials"] = _parse_bool(
                "ROAD_DAMAGE_API_CORS_ALLOW_CREDENTIALS", value
            )
        return cls(**kwargs)
