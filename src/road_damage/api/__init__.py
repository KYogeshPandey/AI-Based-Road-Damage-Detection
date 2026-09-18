"""FastAPI boundary for the road-damage application.

Importing this package is intentionally side-effect free. Import
``road_damage.api.app`` explicitly when an ASGI application is required.
"""

API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

__all__ = ["API_PREFIX", "API_VERSION"]
