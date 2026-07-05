import asyncio
import json
import struct
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from backend.app.config import AudioConfig, Settings, WebSocketConfig
from backend.app.main import create_app
from backend.app.protocol import (
    AUDIO_ENVELOPE_HEADER_LENGTH_BYTES,
    AUDIO_ENVELOPE_MAGIC,
    AUDIO_ENVELOPE_STRUCT_FORMAT,
    AUDIO_ENVELOPE_VERSION,
    AUDIO_FORMAT_PCM_S16LE,
    WEBSOCKET_CLOSE_POLICY_VIOLATION,
    WEBSOCKET_CLOSE_TRY_AGAIN_LATER,
)


def test_event_subscribers_receive_snapshot_and_audio_transport_health() -> None:
    asyncio.run(_event_subscribers_receive_snapshot_and_audio_transport_health())


def test_control_websocket_accepts_supported_commands_with_subprotocol_token() -> None:
    asyncio.run(_control_websocket_accepts_supported_commands_with_subprotocol_token())


def test_event_fanout_delivers_control_ack_to_multiple_subscribers() -> None:
    asyncio.run(_event_fanout_delivers_control_ack_to_multiple_subscribers())


def test_unauthorized_and_missing_tokens_are_rejected() -> None:
    asyncio.run(_unauthorized_and_missing_tokens_are_rejected())


def test_audio_queue_backpressure_closes_audio_socket_with_recoverable_event() -> None:
    asyncio.run(_audio_queue_backpressure_closes_audio_socket_with_recoverable_event())


async def _event_subscribers_receive_snapshot_and_audio_transport_health() -> None:
    app = _build_app()
    session = await _create_session(app)
    session_id = session["session_id"]
    token = session["token"]

    async with ASGIWebSocket(app, f"/v1/events/{session_id}") as events_a:
        await events_a.send_json({"token": token})
        snapshot_a = await events_a.receive_json()

        async with ASGIWebSocket(app, f"/v1/events/{session_id}") as events_b:
            await events_b.send_json({"token": token})
            snapshot_b = await events_b.receive_json()

            assert snapshot_a["type"] == "session.snapshot"
            assert snapshot_a["payload"]["state"] == "idle"
            assert snapshot_b["type"] == "session.snapshot"

            async with ASGIWebSocket(app, f"/v1/audio/{session_id}") as audio:
                await audio.send_json({"token": token})
                await audio.send_bytes(_audio_frame(sequence_number=0))

                health_a = await _receive_until(events_a, "transport.health")
                health_b = await _receive_until(events_b, "transport.health")

            assert health_a["payload"]["frame"]["sequence_number"] == 0
            assert health_a["payload"]["audio_queue_ms"] == 20
            assert health_b["payload"]["frame"]["sequence_number"] == 0


async def _control_websocket_accepts_supported_commands_with_subprotocol_token() -> None:
    app = _build_app()
    session = await _create_session(app)
    session_id = session["session_id"]
    token = session["token"]

    async with ASGIWebSocket(
        app,
        f"/v1/control/{session_id}",
        subprotocols=["ctts.v1", f"ctts-token.{token}"],
    ) as control:
        assert control.accepted_subprotocol == "ctts.v1"

        await control.send_json({"command": "ping"})
        ping = await control.receive_json()
        assert ping["type"] == "control.ack"
        assert ping["payload"]["pong"] is True

        await control.send_json({"command": "set_profile", "profile": "accuracy"})
        set_profile = await control.receive_json()
        assert set_profile["payload"]["status"] == "ok"
        assert set_profile["payload"]["profile"] == "accuracy"

        await control.send_json({"command": "resume"})
        resume = await control.receive_json()
        assert resume["payload"]["state"] == "recording"

        await control.send_json({"command": "pause"})
        pause = await control.receive_json()
        assert pause["payload"]["state"] == "paused"

        await control.send_json({"command": "stop"})
        stop = await control.receive_json()
        assert stop["payload"]["state"] == "stopping"


async def _event_fanout_delivers_control_ack_to_multiple_subscribers() -> None:
    app = _build_app()
    session = await _create_session(app)
    session_id = session["session_id"]
    token = session["token"]

    async with ASGIWebSocket(app, f"/v1/events/{session_id}") as events_a:
        await events_a.send_json({"token": token})
        assert (await events_a.receive_json())["type"] == "session.snapshot"

        async with ASGIWebSocket(app, f"/v1/events/{session_id}") as events_b:
            await events_b.send_json({"token": token})
            assert (await events_b.receive_json())["type"] == "session.snapshot"

            async with ASGIWebSocket(app, f"/v1/control/{session_id}") as control:
                await control.send_json({"token": token})
                await control.send_json({"command": "ping"})
                direct_ack = await control.receive_json()

                fanout_a = await _receive_until(events_a, "control.ack")
                fanout_b = await _receive_until(events_b, "control.ack")

            assert direct_ack["payload"]["command"] == "ping"
            assert fanout_a["payload"]["status"] == "ok"
            assert fanout_b["payload"]["status"] == "ok"


async def _unauthorized_and_missing_tokens_are_rejected() -> None:
    app = _build_app()
    session = await _create_session(app)
    session_id = session["session_id"]

    async with ASGIWebSocket(app, f"/v1/events/{session_id}") as events:
        await events.send_json({"token": "wrong-token"})
        with pytest.raises(WebSocketClosed) as exc:
            await events.receive_json()
        assert exc.value.code == WEBSOCKET_CLOSE_POLICY_VIOLATION

    async with ASGIWebSocket(app, f"/v1/audio/{session_id}") as audio:
        await audio.send_bytes(_audio_frame(sequence_number=0))
        with pytest.raises(WebSocketClosed) as exc:
            await audio.receive()
        assert exc.value.code == WEBSOCKET_CLOSE_POLICY_VIOLATION


async def _audio_queue_backpressure_closes_audio_socket_with_recoverable_event() -> None:
    app = _build_app(max_buffer_ms=20)
    session = await _create_session(app)
    session_id = session["session_id"]
    token = session["token"]

    async with ASGIWebSocket(app, f"/v1/events/{session_id}") as events:
        await events.send_json({"token": token})
        assert (await events.receive_json())["type"] == "session.snapshot"

        async with ASGIWebSocket(app, f"/v1/audio/{session_id}") as audio:
            await audio.send_json({"token": token})
            await audio.send_bytes(_audio_frame(sequence_number=0))
            assert (await _receive_until(events, "transport.health"))["type"] == (
                "transport.health"
            )

            await audio.send_bytes(_audio_frame(sequence_number=1))
            with pytest.raises(WebSocketClosed) as exc:
                await audio.receive()

            error = await _receive_until(events, "transport.error")

        assert exc.value.code == WEBSOCKET_CLOSE_TRY_AGAIN_LATER
        assert error["payload"]["code"] == "audio_buffer_full"
        assert error["payload"]["recoverable"] is True


class WebSocketClosed(Exception):
    def __init__(self, code: int, reason: str = "") -> None:
        super().__init__(f"WebSocket closed with code {code}: {reason}")
        self.code = code
        self.reason = reason


class ASGIWebSocket:
    def __init__(
        self,
        app: FastAPI,
        path: str,
        *,
        subprotocols: Sequence[str] = (),
    ) -> None:
        self._app = app
        self._path = path
        self._subprotocols = tuple(subprotocols)
        self._client_to_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._app_to_client: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self.accepted_subprotocol: str | None = None

    async def __aenter__(self) -> ASGIWebSocket:
        self._task = asyncio.create_task(self._run_app())
        await self._client_to_app.put({"type": "websocket.connect"})
        message = await self._receive_from_app()
        if message["type"] == "websocket.close":
            self._closed = True
            raise WebSocketClosed(message.get("code", 1000), message.get("reason", ""))
        assert message["type"] == "websocket.accept"
        self.accepted_subprotocol = message.get("subprotocol")
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        if not self._closed:
            await self._client_to_app.put(
                {"type": "websocket.disconnect", "code": 1000}
            )
            self._closed = True
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=1)
        except TimeoutError:
            self._task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await self._task

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._client_to_app.put(
            {"type": "websocket.receive", "text": json.dumps(payload)}
        )

    async def send_bytes(self, payload: bytes) -> None:
        await self._client_to_app.put({"type": "websocket.receive", "bytes": payload})

    async def receive_json(self) -> dict[str, Any]:
        message = await self.receive()
        text = message.get("text")
        assert isinstance(text, str)
        return json.loads(text)

    async def receive(self) -> dict[str, Any]:
        message = await self._receive_from_app()
        if message["type"] == "websocket.close":
            self._closed = True
            raise WebSocketClosed(message.get("code", 1000), message.get("reason", ""))
        assert message["type"] == "websocket.send"
        return message

    async def _run_app(self) -> None:
        await self._app(self._scope(), self._receive, self._send)

    async def _receive(self) -> dict[str, Any]:
        return await self._client_to_app.get()

    async def _send(self, message: dict[str, Any]) -> None:
        await self._app_to_client.put(message)

    async def _receive_from_app(self) -> dict[str, Any]:
        if self._task is not None and self._task.done():
            self._task.result()
        try:
            return await asyncio.wait_for(self._app_to_client.get(), timeout=2)
        except TimeoutError:
            if self._task is not None and self._task.done():
                self._task.result()
            raise AssertionError(f"Timed out waiting for WebSocket message: {self._path}")

    def _scope(self) -> dict[str, Any]:
        headers = [(b"host", b"testserver")]
        if self._subprotocols:
            headers.append(
                (
                    b"sec-websocket-protocol",
                    ", ".join(self._subprotocols).encode("ascii"),
                )
            )
        return {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "scheme": "ws",
            "path": self._path,
            "raw_path": self._path.encode("ascii"),
            "query_string": b"",
            "headers": headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "root_path": "",
            "subprotocols": list(self._subprotocols),
        }


def _build_app(*, max_buffer_ms: int = 100) -> FastAPI:
    return create_app(
        Settings(
            audio=AudioConfig(max_buffer_ms=max_buffer_ms, max_buffer_frames=10),
            websockets=WebSocketConfig(event_queue_size=8, auth_timeout_seconds=0.5),
        )
    )


async def _create_session(app: FastAPI) -> dict[str, str]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post("/v1/sessions")
    assert response.status_code == 200
    return response.json()


async def _receive_until(
    websocket: ASGIWebSocket,
    event_type: str,
) -> dict[str, Any]:
    for _ in range(8):
        event = await websocket.receive_json()
        if event["type"] == event_type:
            return event
    raise AssertionError(f"Did not receive event type {event_type!r}")


def _audio_frame(
    *,
    sequence_number: int = 0,
    capture_time_ms: int = 1234,
    sample_rate: int = 16000,
    channels: int = 1,
    duration_ms: int = 20,
) -> bytes:
    payload_length = sample_rate * channels * duration_ms * 2 // 1000
    header = struct.pack(
        AUDIO_ENVELOPE_STRUCT_FORMAT,
        AUDIO_ENVELOPE_MAGIC,
        AUDIO_ENVELOPE_VERSION,
        AUDIO_ENVELOPE_HEADER_LENGTH_BYTES,
        sequence_number,
        capture_time_ms,
        sample_rate,
        channels,
        duration_ms,
        int(AUDIO_FORMAT_PCM_S16LE),
        0,
    )
    return header + (b"\0" * payload_length)
