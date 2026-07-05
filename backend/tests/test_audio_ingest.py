import asyncio
import struct

import pytest

from backend.app.protocol import (
    AUDIO_ENVELOPE_HEADER_LENGTH_BYTES,
    AUDIO_ENVELOPE_MAGIC,
    AUDIO_ENVELOPE_STRUCT_FORMAT,
    AUDIO_ENVELOPE_VERSION,
    AUDIO_FORMAT_PCM_S16LE,
)
from backend.audio.ingest import (
    AudioFrameValidationError,
    AudioIngestService,
    parse_audio_frame,
)
from backend.audio.ring_buffer import AudioBufferFullError


def test_parse_valid_audio_frame_returns_typed_frame() -> None:
    frame = parse_audio_frame(_audio_frame(sequence_number=7))

    assert frame.sequence_number == 7
    assert frame.capture_time_ms == 1234
    assert frame.sample_rate == 16000
    assert frame.channels == 1
    assert frame.duration_ms == 20
    assert frame.format_code == AUDIO_FORMAT_PCM_S16LE
    assert len(frame.payload) == 640


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"magic": b"NOPE"}, "invalid_audio_magic"),
        ({"version": 2}, "invalid_audio_version"),
        ({"header_length": 16}, "invalid_audio_header_length"),
        ({"format_code": 99}, "unsupported_audio_format"),
        ({"payload_length": 10}, "invalid_audio_payload_length"),
    ],
)
def test_invalid_audio_headers_return_explicit_errors(
    kwargs: dict[str, object], code: str
) -> None:
    with pytest.raises(AudioFrameValidationError) as exc:
        parse_audio_frame(_audio_frame(**kwargs))

    assert exc.value.code == code


def test_ingest_tracks_sequence_gaps_duplicates_and_out_of_order_frames() -> None:
    service = AudioIngestService(max_buffer_ms=100, max_buffer_frames=10)

    first = asyncio.run(service.accept_frame("session-1", _audio_frame(sequence_number=1)))
    gap = asyncio.run(service.accept_frame("session-1", _audio_frame(sequence_number=3)))
    duplicate = asyncio.run(
        service.accept_frame("session-1", _audio_frame(sequence_number=3))
    )
    out_of_order = asyncio.run(
        service.accept_frame("session-1", _audio_frame(sequence_number=2))
    )

    assert first.jitter.missing_sequence_numbers == 0
    assert gap.jitter.missing_sequence_numbers == 1
    assert gap.jitter.stats.missing_sequence_numbers == 1
    assert duplicate.jitter.duplicate_sequence_number is True
    assert duplicate.jitter.stats.duplicate_sequence_numbers == 1
    assert out_of_order.jitter.out_of_order is True
    assert out_of_order.jitter.stats.out_of_order_frames == 1
    assert out_of_order.buffer.depth_ms == 80


def test_ingest_rejects_full_audio_buffer_without_dropping_audio() -> None:
    service = AudioIngestService(max_buffer_ms=20, max_buffer_frames=10)

    accepted = asyncio.run(
        service.accept_frame("session-1", _audio_frame(sequence_number=0))
    )
    assert accepted.buffer.depth_ms == 20

    with pytest.raises(AudioBufferFullError) as exc:
        asyncio.run(service.accept_frame("session-1", _audio_frame(sequence_number=1)))

    assert exc.value.current_depth_ms == 20
    snapshot = asyncio.run(service.session_snapshot("session-1"))
    assert snapshot["audio_queue_frames"] == 1
    assert snapshot["audio_queue_ms"] == 20


def test_short_audio_frame_is_rejected_before_unpacking_header() -> None:
    with pytest.raises(AudioFrameValidationError) as exc:
        parse_audio_frame(b"CTTS")

    assert exc.value.code == "audio_frame_too_short"


def _audio_frame(
    *,
    sequence_number: int = 0,
    magic: bytes = AUDIO_ENVELOPE_MAGIC,
    version: int = AUDIO_ENVELOPE_VERSION,
    header_length: int = AUDIO_ENVELOPE_HEADER_LENGTH_BYTES,
    capture_time_ms: int = 1234,
    sample_rate: int = 16000,
    channels: int = 1,
    duration_ms: int = 20,
    format_code: int = int(AUDIO_FORMAT_PCM_S16LE),
    flags: int = 0,
    payload_length: int | None = None,
) -> bytes:
    if payload_length is None:
        payload_length = sample_rate * channels * duration_ms * 2 // 1000
    header = struct.pack(
        AUDIO_ENVELOPE_STRUCT_FORMAT,
        magic,
        version,
        header_length,
        sequence_number,
        capture_time_ms,
        sample_rate,
        channels,
        duration_ms,
        format_code,
        flags,
    )
    return header + (b"\0" * payload_length)
