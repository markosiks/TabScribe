import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from backend.app.config import ServerConfig, SessionsConfig, Settings
from backend.app.main import create_app
from backend.app.protocol import SchedulerProfile, SessionState
from backend.app.sessions import InvalidSessionTransitionError, SessionManager


def test_create_session_returns_token_and_configured_websocket_urls() -> None:
    app = _build_app(port=9876)

    response = _request(app, "POST", "/v1/sessions", json={"profile": "accuracy"})

    assert response.status_code == 200
    body = response.json()
    UUID(body["session_id"])
    assert len(body["token"]) >= 32
    assert body["state"] == "idle"
    assert body["profile"] == "accuracy"
    assert body["audio_ws_url"] == (
        f"ws://127.0.0.1:9876/v1/audio/{body['session_id']}"
    )
    assert body["events_ws_url"] == (
        f"ws://127.0.0.1:9876/v1/events/{body['session_id']}"
    )
    assert body["control_ws_url"] == (
        f"ws://127.0.0.1:9876/v1/control/{body['session_id']}"
    )
    assert _parse_datetime(body["created_at"]).tzinfo is not None


def test_create_session_uses_default_profile_and_validates_profile() -> None:
    app = _build_app(default_profile=SchedulerProfile.realtime)

    default_response = _request(app, "POST", "/v1/sessions")
    invalid_response = _request(
        app, "POST", "/v1/sessions", json={"profile": "fast"}
    )

    assert default_response.status_code == 200
    assert default_response.json()["profile"] == "realtime"
    assert invalid_response.status_code == 422


def test_get_session_returns_status_without_token() -> None:
    app = _build_app()
    created = _request(app, "POST", "/v1/sessions").json()

    response = _request(app, "GET", f"/v1/sessions/{created['session_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == created["session_id"]
    assert body["state"] == "idle"
    assert body["profile"] == "balanced"
    assert body["glossary_version"] is None
    assert "token" not in body


def test_unknown_session_returns_404() -> None:
    app = _build_app()

    response = _request(app, "GET", "/v1/sessions/not-a-session")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_stop_session_transitions_and_is_idempotent() -> None:
    app = _build_app()
    created = _request(app, "POST", "/v1/sessions").json()
    session_id = created["session_id"]

    first_stop = _request(app, "POST", f"/v1/sessions/{session_id}/stop")
    second_stop = _request(app, "POST", f"/v1/sessions/{session_id}/stop")

    assert first_stop.status_code == 200
    assert first_stop.json()["state"] == "stopping"
    assert second_stop.status_code == 200
    assert second_stop.json()["state"] == "stopping"

    manager: SessionManager = app.state.session_manager
    manager.transition_state(session_id, SessionState.finalizing)
    manager.transition_state(session_id, SessionState.complete)
    completed_stop = _request(app, "POST", f"/v1/sessions/{session_id}/stop")

    assert completed_stop.status_code == 200
    assert completed_stop.json()["state"] == "complete"


def test_invalid_stop_transition_returns_409() -> None:
    app = _build_app()
    created = _request(app, "POST", "/v1/sessions").json()
    session_id = created["session_id"]
    manager: SessionManager = app.state.session_manager
    manager.transition_state(session_id, SessionState.error)

    response = _request(app, "POST", f"/v1/sessions/{session_id}/stop")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_session_transition"


def test_glossary_update_stores_version_and_timestamp_fallback() -> None:
    app = _build_app()
    created = _request(app, "POST", "/v1/sessions").json()
    session_id = created["session_id"]

    explicit = _request(
        app,
        "PUT",
        f"/v1/sessions/{session_id}/glossary",
        json={"glossary_version": "glossary-v1"},
    )
    generated = _request(
        app, "PUT", f"/v1/sessions/{session_id}/glossary", json={}
    )
    status = _request(app, "GET", f"/v1/sessions/{session_id}")

    assert explicit.status_code == 200
    assert explicit.json()["glossary_version"] == "glossary-v1"
    assert generated.status_code == 200
    assert generated.json()["glossary_version"] != "glossary-v1"
    assert _parse_datetime(generated.json()["glossary_version"]).tzinfo is not None
    assert status.json()["glossary_version"] == generated.json()["glossary_version"]


def test_session_manager_validates_tokens_and_state_transitions() -> None:
    manager = SessionManager(
        default_profile=SchedulerProfile.balanced,
        stale_after_seconds=60,
    )
    session = manager.create_session(profile=SchedulerProfile.realtime)

    assert manager.validate_token(session.session_id, session.token)
    assert not manager.validate_token(session.session_id, "wrong-token")
    with pytest.raises(InvalidSessionTransitionError):
        manager.transition_state(session.session_id, SessionState.complete)

    recording = manager.transition_state(session.session_id, SessionState.recording)
    stopping = manager.request_stop(session.session_id)
    finalizing = manager.request_finalize(session.session_id)

    assert recording.state == "recording"
    assert stopping.state == "stopping"
    assert finalizing.state == "finalizing"


def test_session_manager_expires_stale_sessions_and_tokens() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def clock() -> datetime:
        return now

    manager = SessionManager(
        default_profile=SchedulerProfile.balanced,
        stale_after_seconds=10,
        clock=clock,
    )
    session = manager.create_session()

    assert manager.get_session(session.session_id) is not None
    assert manager.validate_token(session.session_id, session.token)

    now = now + timedelta(seconds=11)

    assert manager.expire_stale_sessions() == 1
    assert manager.get_session(session.session_id) is None
    assert not manager.validate_token(session.session_id, session.token)


def _build_app(
    *,
    default_profile: SchedulerProfile = SchedulerProfile.balanced,
    host: str = "127.0.0.1",
    port: int = 8765,
    stale_after_seconds: int = 3600,
) -> FastAPI:
    return create_app(
        Settings(
            default_profile=default_profile,
            server=ServerConfig(host=host, port=port),
            sessions=SessionsConfig(stale_after_seconds=stale_after_seconds),
        )
    )


def _request(
    app: FastAPI, method: str, path: str, **kwargs: object
) -> httpx.Response:
    return asyncio.run(_request_async(app, method, path, **kwargs))


async def _request_async(
    app: FastAPI, method: str, path: str, **kwargs: object
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
