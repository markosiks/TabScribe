from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict

from backend.audio.ingest import (
    AudioFrameValidationError,
    AudioIngestResult,
    AudioIngestService,
    AudioValidationLimits,
    jitter_observation_payload,
)
from backend.audio.ring_buffer import AudioBufferFullError
from backend.telemetry.events import EventBroker, EventPublishResult

from .config import Settings, load_settings
from .errors import AppError, ErrorCode, register_error_handlers
from .protocol import (
    ControlCommandName,
    EventEnvelope,
    GlossaryUpdateRequest,
    GlossaryUpdateResponse,
    SchedulerProfile,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionState,
    SessionStatusResponse,
    WEBSOCKET_CLOSE_INVALID_PAYLOAD,
    WEBSOCKET_CLOSE_POLICY_VIOLATION,
    WEBSOCKET_CLOSE_TRY_AGAIN_LATER,
    WEBSOCKET_CLOSE_UNSUPPORTED_DATA,
)
from .sessions import (
    InvalidSessionTransitionError,
    SessionManager,
    SessionNotFoundError,
    SessionRecord,
)


_TERMINAL_AUDIO_STATES = frozenset(
    {
        SessionState.stopping,
        SessionState.finalizing,
        SessionState.complete,
        SessionState.error,
    }
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
    app.state.event_broker = EventBroker(
        subscriber_queue_size=resolved_settings.websockets.event_queue_size,
    )
    app.state.audio_ingest = AudioIngestService(
        max_buffer_ms=resolved_settings.audio.max_buffer_ms,
        max_buffer_frames=resolved_settings.audio.max_buffer_frames,
        validation_limits=AudioValidationLimits(
            min_sample_rate_hz=resolved_settings.audio.min_sample_rate_hz,
            max_sample_rate_hz=resolved_settings.audio.max_sample_rate_hz,
            max_channels=resolved_settings.audio.max_channels,
            max_duration_ms=resolved_settings.audio.max_frame_duration_ms,
        ),
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

    @app.websocket("/v1/audio/{session_id}")
    async def audio_websocket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept(subprotocol=_selected_response_subprotocol(websocket))
        current_settings: Settings = app.state.settings
        manager: SessionManager = app.state.session_manager
        broker: EventBroker = app.state.event_broker
        ingest: AudioIngestService = app.state.audio_ingest

        session = await _authenticate_websocket(
            websocket,
            session_id=session_id,
            manager=manager,
            auth_timeout_seconds=current_settings.websockets.auth_timeout_seconds,
        )
        if session is None:
            return

        await _publish_event(
            broker,
            session_id=session_id,
            event_type="transport.connected",
            payload={"channel": "audio"},
            ingest=ingest,
        )

        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                return

            if message["type"] == "websocket.disconnect":
                return

            data = message.get("bytes")
            if data is None:
                await _publish_transport_error(
                    broker,
                    ingest=ingest,
                    session_id=session_id,
                    code="unsupported_audio_message",
                    message="Audio WebSocket only accepts binary frames",
                )
                await _close_websocket(
                    websocket,
                    code=WEBSOCKET_CLOSE_UNSUPPORTED_DATA,
                    reason="binary audio frames required",
                )
                return

            try:
                result = await ingest.accept_frame(session_id, data)
                session = _record_audio_activity(manager, session_id)
            except AudioFrameValidationError as exc:
                await _publish_transport_error(
                    broker,
                    ingest=ingest,
                    session_id=session_id,
                    code=exc.code,
                    message=exc.message,
                    details=exc.details,
                )
                await _close_websocket(
                    websocket,
                    code=WEBSOCKET_CLOSE_INVALID_PAYLOAD,
                    reason=exc.code,
                )
                return
            except AudioBufferFullError as exc:
                await _publish_transport_error(
                    broker,
                    ingest=ingest,
                    session_id=session_id,
                    code="audio_buffer_full",
                    message="Audio queue is full",
                    details={
                        "current_depth_ms": exc.current_depth_ms,
                        "incoming_frame_duration_ms": exc.frame_duration_ms,
                        "max_depth_ms": exc.max_depth_ms,
                    },
                )
                await _close_websocket(
                    websocket,
                    code=WEBSOCKET_CLOSE_TRY_AGAIN_LATER,
                    reason="audio queue full",
                )
                return
            except (SessionNotFoundError, InvalidSessionTransitionError):
                await _publish_transport_error(
                    broker,
                    ingest=ingest,
                    session_id=session_id,
                    code="session_not_accepting_audio",
                    message="Session is not accepting audio frames",
                )
                await _close_websocket(
                    websocket,
                    code=WEBSOCKET_CLOSE_POLICY_VIOLATION,
                    reason="session not accepting audio",
                )
                return

            await _publish_audio_health(
                broker,
                ingest=ingest,
                session=session,
                result=result,
            )

    @app.websocket("/v1/events/{session_id}")
    async def events_websocket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept(subprotocol=_selected_response_subprotocol(websocket))
        current_settings: Settings = app.state.settings
        manager: SessionManager = app.state.session_manager
        broker: EventBroker = app.state.event_broker
        ingest: AudioIngestService = app.state.audio_ingest

        session = await _authenticate_websocket(
            websocket,
            session_id=session_id,
            manager=manager,
            auth_timeout_seconds=current_settings.websockets.auth_timeout_seconds,
        )
        if session is None:
            return

        subscription = await broker.subscribe(session_id)
        try:
            snapshot = await _session_snapshot_event(
                broker,
                ingest=ingest,
                session=session,
            )
            await websocket.send_json(snapshot.model_dump(mode="json"))
            await _events_send_loop(websocket, subscription)
        except WebSocketDisconnect:
            return
        finally:
            await subscription.close()

    @app.websocket("/v1/control/{session_id}")
    async def control_websocket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept(subprotocol=_selected_response_subprotocol(websocket))
        current_settings: Settings = app.state.settings
        manager: SessionManager = app.state.session_manager
        broker: EventBroker = app.state.event_broker
        ingest: AudioIngestService = app.state.audio_ingest

        session = await _authenticate_websocket(
            websocket,
            session_id=session_id,
            manager=manager,
            auth_timeout_seconds=current_settings.websockets.auth_timeout_seconds,
        )
        if session is None:
            return

        await _publish_event(
            broker,
            session_id=session_id,
            event_type="transport.connected",
            payload={"channel": "control"},
            ingest=ingest,
        )

        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                return

            if message["type"] == "websocket.disconnect":
                return

            text = message.get("text")
            if text is None:
                await _send_control_error_and_close(
                    websocket,
                    broker=broker,
                    ingest=ingest,
                    session_id=session_id,
                    command=None,
                    code="unsupported_control_message",
                    message="Control WebSocket only accepts JSON text messages",
                    close_code=WEBSOCKET_CLOSE_UNSUPPORTED_DATA,
                )
                return

            try:
                command = _parse_control_command(json.loads(text))
            except (JSONDecodeError, TypeError):
                await _send_control_error_and_close(
                    websocket,
                    broker=broker,
                    ingest=ingest,
                    session_id=session_id,
                    command=None,
                    code="invalid_control_json",
                    message="Control message must be valid JSON",
                    close_code=WEBSOCKET_CLOSE_INVALID_PAYLOAD,
                )
                return
            except ControlMessageError as exc:
                await _send_control_error_and_close(
                    websocket,
                    broker=broker,
                    ingest=ingest,
                    session_id=session_id,
                    command=exc.command,
                    code=exc.code,
                    message=exc.message,
                    close_code=WEBSOCKET_CLOSE_POLICY_VIOLATION,
                )
                return

            try:
                session = _apply_control_command(manager, session_id, command)
            except InvalidSessionTransitionError as exc:
                await _publish_control_ack(
                    websocket,
                    broker=broker,
                    ingest=ingest,
                    session_id=session_id,
                    command=command.name.value,
                    status="error",
                    code="invalid_session_transition",
                    message=str(exc),
                    details={
                        "current_state": exc.current_state.value,
                        "requested_state": exc.requested_state.value,
                    },
                )
                continue
            except SessionNotFoundError:
                await _send_control_error_and_close(
                    websocket,
                    broker=broker,
                    ingest=ingest,
                    session_id=session_id,
                    command=command.name.value,
                    code="unknown_session",
                    message="Session no longer exists",
                    close_code=WEBSOCKET_CLOSE_POLICY_VIOLATION,
                )
                return

            payload: dict[str, Any] = {
                "command": command.name.value,
                "status": "ok",
                "state": session.state.value,
                "profile": session.profile.value,
            }
            if command.name is ControlCommandName.ping:
                payload["pong"] = True
            await _publish_control_ack_event(
                websocket,
                broker=broker,
                ingest=ingest,
                session_id=session_id,
                payload=payload,
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


@dataclass(frozen=True, slots=True)
class ControlCommand:
    name: ControlCommandName
    profile: SchedulerProfile | None = None


class ControlMessageError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        command: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.command = command


async def _authenticate_websocket(
    websocket: WebSocket,
    *,
    session_id: str,
    manager: SessionManager,
    auth_timeout_seconds: float,
) -> SessionRecord | None:
    for token in _subprotocol_token_candidates(websocket):
        session = manager.authenticate_session(session_id, token)
        if session is not None:
            return session

    try:
        message = await asyncio.wait_for(
            websocket.receive(),
            timeout=auth_timeout_seconds,
        )
    except TimeoutError:
        await _close_websocket(
            websocket,
            code=WEBSOCKET_CLOSE_POLICY_VIOLATION,
            reason="authentication required",
        )
        return None
    except WebSocketDisconnect:
        return None

    if message["type"] == "websocket.disconnect":
        return None

    text = message.get("text")
    if text is None:
        await _close_websocket(
            websocket,
            code=WEBSOCKET_CLOSE_POLICY_VIOLATION,
            reason="authentication required",
        )
        return None

    try:
        payload = json.loads(text)
    except (JSONDecodeError, TypeError):
        await _close_websocket(
            websocket,
            code=WEBSOCKET_CLOSE_INVALID_PAYLOAD,
            reason="invalid auth json",
        )
        return None

    token = _token_from_auth_payload(payload)
    session = manager.authenticate_session(session_id, token)
    if session is None:
        await _close_websocket(
            websocket,
            code=WEBSOCKET_CLOSE_POLICY_VIOLATION,
            reason="unauthorized",
        )
        return None
    return session


def _subprotocol_token_candidates(websocket: WebSocket) -> tuple[str, ...]:
    raw_header = websocket.headers.get("sec-websocket-protocol")
    if not raw_header:
        return ()

    candidates: list[str] = []
    for value in raw_header.split(","):
        protocol = value.strip()
        if not protocol or protocol == "ctts.v1":
            continue
        if protocol.startswith("ctts-token."):
            candidates.append(protocol.removeprefix("ctts-token."))
        elif protocol.startswith("token."):
            candidates.append(protocol.removeprefix("token."))
        elif protocol.startswith("bearer."):
            candidates.append(protocol.removeprefix("bearer."))
        else:
            candidates.append(protocol)
    return tuple(candidates)


def _token_from_auth_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    token = payload.get("token")
    if isinstance(token, str):
        return token

    nested_payload = payload.get("payload")
    if isinstance(nested_payload, dict):
        nested_token = nested_payload.get("token")
        if isinstance(nested_token, str):
            return nested_token

    return None


def _selected_response_subprotocol(websocket: WebSocket) -> str | None:
    raw_header = websocket.headers.get("sec-websocket-protocol")
    if raw_header is None:
        return None
    offered = {value.strip() for value in raw_header.split(",")}
    return "ctts.v1" if "ctts.v1" in offered else None


def _record_audio_activity(
    manager: SessionManager,
    session_id: str,
) -> SessionRecord:
    session = manager.get_session(session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    if session.state in _TERMINAL_AUDIO_STATES:
        raise InvalidSessionTransitionError(
            session_id=session_id,
            current_state=session.state,
            requested_state=SessionState.recording,
        )
    if session.state in {SessionState.idle, SessionState.starting}:
        return manager.transition_state(session_id, SessionState.recording)
    return manager.touch_session(session_id)


async def _publish_audio_health(
    broker: EventBroker,
    *,
    ingest: AudioIngestService,
    session: SessionRecord,
    result: AudioIngestResult,
) -> None:
    await _publish_event(
        broker,
        session_id=session.session_id,
        event_type="transport.health",
        payload={
            "channel": "audio",
            "state": session.state.value,
            "profile": session.profile.value,
            "audio_queue_ms": result.buffer.depth_ms,
            "audio_queue_frames": result.buffer.frame_count,
            "audio_queue_bytes": result.buffer.byte_count,
            "frame": {
                "sequence_number": result.frame.sequence_number,
                "capture_time_ms": result.frame.capture_time_ms,
                "sample_rate": result.frame.sample_rate,
                "channels": result.frame.channels,
                "duration_ms": result.frame.duration_ms,
                "format": result.frame.format_code.name,
                "flags": result.frame.flags,
            },
            "jitter": jitter_observation_payload(result.jitter),
        },
        ingest=ingest,
    )


async def _publish_transport_error(
    broker: EventBroker,
    *,
    ingest: AudioIngestService,
    session_id: str,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    await _publish_event(
        broker,
        session_id=session_id,
        event_type="transport.error",
        payload={
            "code": code,
            "message": message,
            "details": details or {},
            "recoverable": code == "audio_buffer_full",
        },
        ingest=ingest,
    )


async def _publish_event(
    broker: EventBroker,
    *,
    session_id: str,
    event_type: str,
    payload: dict[str, Any],
    ingest: AudioIngestService,
) -> EventPublishResult:
    result = await broker.publish_event(
        session_id=session_id,
        event_type=event_type,
        payload=payload,
    )
    if result.dropped:
        await ingest.record_dropped_diagnostic_frames(session_id, result.dropped)
    return result


async def _session_snapshot_event(
    broker: EventBroker,
    *,
    ingest: AudioIngestService,
    session: SessionRecord,
) -> EventEnvelope:
    audio_snapshot = await ingest.session_snapshot(session.session_id)
    return await broker.make_event(
        session_id=session.session_id,
        event_type="session.snapshot",
        payload={
            "state": session.state.value,
            "profile": session.profile.value,
            "created_at": session.created_at.isoformat(),
            "last_activity_at": session.last_activity_at.isoformat(),
            "glossary_version": session.glossary_version,
            **audio_snapshot,
        },
    )


async def _events_send_loop(websocket: WebSocket, subscription: Any) -> None:
    while True:
        receive_task = asyncio.create_task(websocket.receive())
        event_task = asyncio.create_task(subscription.receive())
        done, pending = await asyncio.wait(
            {receive_task, event_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        for task in pending:
            with suppress(asyncio.CancelledError):
                await task

        if receive_task in done:
            message = receive_task.result()
            if message["type"] == "websocket.disconnect":
                return
            await _close_websocket(
                websocket,
                code=WEBSOCKET_CLOSE_UNSUPPORTED_DATA,
                reason="events endpoint is send-only",
            )
            return

        if event_task in done:
            event = event_task.result()
            await websocket.send_json(event.model_dump(mode="json"))


def _parse_control_command(payload: Any) -> ControlCommand:
    if not isinstance(payload, dict):
        raise ControlMessageError(
            "invalid_control_message",
            "Control message must be a JSON object",
        )

    nested_payload = payload.get("payload")
    if not isinstance(nested_payload, dict):
        nested_payload = {}

    command_value = payload.get("command")
    if command_value is None:
        payload_type = payload.get("type")
        if payload_type in {command.value for command in ControlCommandName}:
            command_value = payload_type
        else:
            command_value = nested_payload.get("command")

    if not isinstance(command_value, str):
        raise ControlMessageError(
            "missing_control_command",
            "Control command is required",
        )

    try:
        command_name = ControlCommandName(command_value)
    except ValueError as exc:
        raise ControlMessageError(
            "unsupported_control_command",
            "Control command is unsupported",
            command=command_value,
        ) from exc

    profile = None
    if command_name is ControlCommandName.set_profile:
        profile_value = payload.get("profile", nested_payload.get("profile"))
        if not isinstance(profile_value, str):
            raise ControlMessageError(
                "missing_control_profile",
                "set_profile requires a profile",
                command=command_name.value,
            )
        try:
            profile = SchedulerProfile(profile_value)
        except ValueError as exc:
            raise ControlMessageError(
                "invalid_control_profile",
                "set_profile profile is invalid",
                command=command_name.value,
            ) from exc

    return ControlCommand(name=command_name, profile=profile)


def _apply_control_command(
    manager: SessionManager,
    session_id: str,
    command: ControlCommand,
) -> SessionRecord:
    if command.name is ControlCommandName.pause:
        return manager.transition_state(session_id, SessionState.paused)
    if command.name is ControlCommandName.resume:
        return manager.transition_state(session_id, SessionState.recording)
    if command.name is ControlCommandName.stop:
        return manager.request_stop(session_id)
    if command.name is ControlCommandName.set_profile:
        if command.profile is None:
            raise ControlMessageError(
                "missing_control_profile",
                "set_profile requires a profile",
                command=command.name.value,
            )
        return manager.update_profile(session_id, command.profile)
    if command.name is ControlCommandName.ping:
        return manager.touch_session(session_id)

    raise ControlMessageError(
        "unsupported_control_command",
        "Control command is unsupported",
        command=command.name.value,
    )


async def _send_control_error_and_close(
    websocket: WebSocket,
    *,
    broker: EventBroker,
    ingest: AudioIngestService,
    session_id: str,
    command: str | None,
    code: str,
    message: str,
    close_code: int,
) -> None:
    await _publish_control_ack(
        websocket,
        broker=broker,
        ingest=ingest,
        session_id=session_id,
        command=command,
        status="error",
        code=code,
        message=message,
    )
    await _close_websocket(websocket, code=close_code, reason=code)


async def _publish_control_ack(
    websocket: WebSocket,
    *,
    broker: EventBroker,
    ingest: AudioIngestService,
    session_id: str,
    command: str | None,
    status: Literal["ok", "error"],
    code: str | None = None,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "command": command,
        "status": status,
    }
    if code is not None:
        payload["code"] = code
    if message is not None:
        payload["message"] = message
    if details is not None:
        payload["details"] = details
    await _publish_control_ack_event(
        websocket,
        broker=broker,
        ingest=ingest,
        session_id=session_id,
        payload=payload,
    )


async def _publish_control_ack_event(
    websocket: WebSocket,
    *,
    broker: EventBroker,
    ingest: AudioIngestService,
    session_id: str,
    payload: dict[str, Any],
) -> None:
    event = await broker.make_event(
        session_id=session_id,
        event_type="control.ack",
        payload=payload,
    )
    result = await broker.publish(event)
    if result.dropped:
        await ingest.record_dropped_diagnostic_frames(session_id, result.dropped)
    await websocket.send_json(event.model_dump(mode="json"))


async def _close_websocket(websocket: WebSocket, *, code: int, reason: str) -> None:
    with suppress(RuntimeError):
        await websocket.close(code=code, reason=reason[:120])


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
