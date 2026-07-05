from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from .config import Settings, load_settings
from .errors import AppError, ErrorCode, register_error_handlers
from .protocol import (
    GlossaryUpdateRequest,
    GlossaryUpdateResponse,
    SchedulerProfile,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatusResponse,
)
from .sessions import (
    InvalidSessionTransitionError,
    SessionManager,
    SessionNotFoundError,
    SessionRecord,
)


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
    app.state.session_manager = SessionManager(
        default_profile=resolved_settings.default_profile,
        stale_after_seconds=resolved_settings.sessions.stale_after_seconds,
    )
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

    @app.post("/v1/sessions", response_model=SessionCreateResponse)
    async def create_session(
        request: SessionCreateRequest | None = None,
    ) -> SessionCreateResponse:
        current_settings: Settings = app.state.settings
        manager: SessionManager = app.state.session_manager
        profile = (
            request.profile
            if request is not None and request.profile is not None
            else current_settings.default_profile
        )
        session = manager.create_session(profile=profile)
        return _session_create_response(session, current_settings)

    @app.get("/v1/sessions/{session_id}", response_model=SessionStatusResponse)
    async def get_session(session_id: str) -> SessionStatusResponse:
        manager: SessionManager = app.state.session_manager
        session = manager.get_session(session_id, touch=True)
        if session is None:
            _raise_session_not_found(session_id)
        return _session_status_response(session)

    @app.post("/v1/sessions/{session_id}/stop", response_model=SessionStatusResponse)
    async def stop_session(session_id: str) -> SessionStatusResponse:
        manager: SessionManager = app.state.session_manager
        try:
            session = manager.request_stop(session_id)
        except SessionNotFoundError as exc:
            _raise_session_not_found(exc.session_id)
        except InvalidSessionTransitionError as exc:
            _raise_invalid_transition(exc)
        return _session_status_response(session)

    @app.put(
        "/v1/sessions/{session_id}/glossary",
        response_model=GlossaryUpdateResponse,
    )
    async def update_glossary(
        session_id: str,
        request: GlossaryUpdateRequest | None = None,
    ) -> GlossaryUpdateResponse:
        manager: SessionManager = app.state.session_manager
        glossary_version = request.glossary_version if request is not None else None
        try:
            session = manager.update_glossary(session_id, glossary_version)
        except SessionNotFoundError as exc:
            _raise_session_not_found(exc.session_id)
        return GlossaryUpdateResponse(
            session_id=session.session_id,
            glossary_version=(
                session.glossary_version or session.last_activity_at.isoformat()
            ),
            updated_at=session.last_activity_at,
        )

    return app


def _session_create_response(
    session: SessionRecord, settings: Settings
) -> SessionCreateResponse:
    return SessionCreateResponse(
        session_id=session.session_id,
        token=session.token,
        state=session.state,
        profile=session.profile,
        audio_ws_url=_websocket_url(settings, "audio", session.session_id),
        events_ws_url=_websocket_url(settings, "events", session.session_id),
        control_ws_url=_websocket_url(settings, "control", session.session_id),
        created_at=session.created_at,
    )


def _session_status_response(session: SessionRecord) -> SessionStatusResponse:
    return SessionStatusResponse(
        session_id=session.session_id,
        state=session.state,
        profile=session.profile,
        created_at=session.created_at,
        last_activity_at=session.last_activity_at,
        glossary_version=session.glossary_version,
    )


def _websocket_url(settings: Settings, channel: str, session_id: str) -> str:
    return f"ws://{settings.server.host}:{settings.server.port}/v1/{channel}/{session_id}"


def _raise_session_not_found(session_id: str) -> None:
    raise AppError(
        f"Unknown session: {session_id}",
        status_code=404,
        code=ErrorCode.not_found,
    )


def _raise_invalid_transition(exc: InvalidSessionTransitionError) -> None:
    raise AppError(
        str(exc),
        status_code=409,
        code="invalid_session_transition",
        details={
            "session_id": exc.session_id,
            "current_state": exc.current_state.value,
            "requested_state": exc.requested_state.value,
        },
    )


app = create_app()
