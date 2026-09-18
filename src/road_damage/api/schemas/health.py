"""Health-check response schema."""

from typing import Literal

from pydantic import Field

from road_damage.api.schemas.common import ApiSchema


class HealthResponse(ApiSchema):
    status: Literal["ok"]
    service: Literal["AI-Based Road Damage Detection API"]
    api_version: str = Field(min_length=1)
