from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from .config import Settings, load_settings
from .errors import register_error_handlers
from .protocol import SchedulerProfile


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    version: str
    host: str
    port: int
    default_profile: SchedulerProfile
    mode: str


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()
    app = FastAPI(
        title=resolved_settings.app.name,
        version=resolved_settings.app.version,
    )
    app.state.settings = resolved_settings
    register_error_handlers(app)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        current_settings: Settings = app.state.settings
        return HealthResponse(
            status="ok",
            version=current_settings.app.version,
            host=current_settings.server.host,
            port=current_settings.server.port,
            default_profile=current_settings.default_profile,
            mode=current_settings.mode,
        )

    return app


app = create_app()
