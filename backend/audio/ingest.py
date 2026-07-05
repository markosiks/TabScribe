from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from backend.app.protocol import (
    AUDIO_ENVELOPE_HEADER_LENGTH_BYTES,
    AUDIO_ENVELOPE_MAGIC,
    AUDIO_ENVELOPE_STRUCT_FORMAT,
    AUDIO_ENVELOPE_VERSION,
    AUDIO_FORMAT_PCM_S16LE,
    AudioEnvelopeHeader,
    AudioFormatCode,
)

from .jitter import JitterObservation, JitterStats, JitterTracker
from .ring_buffer import AudioBufferFullError, AudioBufferSnapshot, AudioRingBuffer


_HEADER_STRUCT = struct.Struct(AUDIO_ENVELOPE_STRUCT_FORMAT)
_BYTES_PER_SAMPLE = MappingProxyType({AudioFormatCode.pcm_s16le: 2})


class AudioFrameValidationError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class AudioValidationLimits:
    min_sample_rate_hz: int = 8000
    max_sample_rate_hz: int = 192000
    min_channels: int = 1
    max_channels: int = 2
    min_duration_ms: int = 1
    max_duration_ms: int = 1000


@dataclass(frozen=True, slots=True)
class AudioFrame:
    header: AudioEnvelopeHeader
    payload: bytes

    @property
    def sequence_number(self) -> int:
        return self.header.sequence_number

    @property
    def capture_time_ms(self) -> int:
        return self.header.capture_time_ms

    @property
    def sample_rate(self) -> int:
        return self.header.sample_rate

    @property
    def channels(self) -> int:
        return self.header.channels

    @property
    def duration_ms(self) -> int:
        return self.header.duration_ms

    @property
    def format_code(self) -> AudioFormatCode:
        return self.header.format_code

    @property
    def flags(self) -> int:
        return self.header.flags


@dataclass(frozen=True, slots=True)
class AudioIngestResult:
    frame: AudioFrame
    jitter: JitterObservation
    buffer: AudioBufferSnapshot


class AudioIngestService:
    def __init__(
        self,
        *,
        max_buffer_ms: int,
        max_buffer_frames: int,
        validation_limits: AudioValidationLimits | None = None,
    ) -> None:
        self._max_buffer_ms = max_buffer_ms
        self._max_buffer_frames = max_buffer_frames
        self._validation_limits = validation_limits or AudioValidationLimits()
        self._buffers: dict[str, AudioRingBuffer] = {}
        self._jitter: dict[str, JitterTracker] = {}
        self._lock = asyncio.Lock()

    @property
    def validation_limits(self) -> AudioValidationLimits:
        return self._validation_limits

    async def accept_frame(self, session_id: str, data: bytes) -> AudioIngestResult:
        frame = parse_audio_frame(data, limits=self._validation_limits)

        async with self._lock:
            buffer = self._buffer_for_session(session_id)
            buffer.append(frame)
            snapshot = buffer.snapshot()
            jitter = self._jitter_for_session(session_id)
            observation = jitter.observe(
                frame.sequence_number,
                queue_depth_ms=snapshot.depth_ms,
            )

        return AudioIngestResult(frame=frame, jitter=observation, buffer=snapshot)

    async def session_snapshot(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            buffer = self._buffers.get(session_id)
            jitter = self._jitter.get(session_id)
            if buffer is None:
                buffer_snapshot = AudioBufferSnapshot(
                    frame_count=0,
                    depth_ms=0,
                    byte_count=0,
                    max_depth_ms=self._max_buffer_ms,
                    max_frames=self._max_buffer_frames,
                )
            else:
                buffer_snapshot = buffer.snapshot()

            jitter_stats = jitter.stats if jitter is not None else JitterStats()

        return {
            "audio_queue_frames": buffer_snapshot.frame_count,
            "audio_queue_ms": buffer_snapshot.depth_ms,
            "audio_queue_bytes": buffer_snapshot.byte_count,
            "audio_queue_max_ms": buffer_snapshot.max_depth_ms,
            "audio_queue_max_frames": buffer_snapshot.max_frames,
            "jitter": _jitter_stats_payload(jitter_stats),
        }

    async def record_dropped_diagnostic_frames(
        self, session_id: str, count: int = 1
    ) -> JitterStats:
        async with self._lock:
            return self._jitter_for_session(session_id).record_dropped_diagnostic_frames(
                count
            )

    async def clear_session(self, session_id: str) -> None:
        async with self._lock:
            self._buffers.pop(session_id, None)
            self._jitter.pop(session_id, None)

    def _buffer_for_session(self, session_id: str) -> AudioRingBuffer:
        buffer = self._buffers.get(session_id)
        if buffer is None:
            buffer = AudioRingBuffer(
                max_depth_ms=self._max_buffer_ms,
                max_frames=self._max_buffer_frames,
            )
            self._buffers[session_id] = buffer
        return buffer

    def _jitter_for_session(self, session_id: str) -> JitterTracker:
        jitter = self._jitter.get(session_id)
        if jitter is None:
            jitter = JitterTracker()
            self._jitter[session_id] = jitter
        return jitter


def parse_audio_frame(
    data: bytes,
    *,
    limits: AudioValidationLimits | None = None,
) -> AudioFrame:
    validation_limits = limits or AudioValidationLimits()
    if len(data) < AUDIO_ENVELOPE_HEADER_LENGTH_BYTES:
        raise AudioFrameValidationError(
            "audio_frame_too_short",
            "Audio frame is shorter than the envelope header",
            details={
                "expected_header_length": AUDIO_ENVELOPE_HEADER_LENGTH_BYTES,
                "actual_length": len(data),
            },
        )

    if _HEADER_STRUCT.size != AUDIO_ENVELOPE_HEADER_LENGTH_BYTES:
        raise RuntimeError("Audio envelope struct size does not match header length")

    (
        magic,
        version,
        header_length,
        sequence_number,
        capture_time_ms,
        sample_rate,
        channels,
        duration_ms,
        format_code_value,
        flags,
    ) = _HEADER_STRUCT.unpack_from(data)

    if magic != AUDIO_ENVELOPE_MAGIC:
        raise AudioFrameValidationError(
            "invalid_audio_magic",
            "Audio frame magic bytes are invalid",
        )
    if version != AUDIO_ENVELOPE_VERSION:
        raise AudioFrameValidationError(
            "invalid_audio_version",
            "Audio frame envelope version is unsupported",
            details={
                "expected_version": AUDIO_ENVELOPE_VERSION,
                "actual_version": version,
            },
        )
    if header_length != AUDIO_ENVELOPE_HEADER_LENGTH_BYTES:
        raise AudioFrameValidationError(
            "invalid_audio_header_length",
            "Audio frame header length is invalid",
            details={
                "expected_header_length": AUDIO_ENVELOPE_HEADER_LENGTH_BYTES,
                "actual_header_length": header_length,
            },
        )

    try:
        format_code = AudioFormatCode(format_code_value)
    except ValueError as exc:
        raise AudioFrameValidationError(
            "unsupported_audio_format",
            "Audio frame format is unsupported",
            details={"format_code": format_code_value},
        ) from exc

    if format_code is not AUDIO_FORMAT_PCM_S16LE:
        raise AudioFrameValidationError(
            "unsupported_audio_format",
            "Audio frame format is unsupported",
            details={"format_code": int(format_code)},
        )

    _validate_range(
        "sample_rate",
        sample_rate,
        minimum=validation_limits.min_sample_rate_hz,
        maximum=validation_limits.max_sample_rate_hz,
        code="invalid_audio_sample_rate",
    )
    _validate_range(
        "channels",
        channels,
        minimum=validation_limits.min_channels,
        maximum=validation_limits.max_channels,
        code="invalid_audio_channels",
    )
    _validate_range(
        "duration_ms",
        duration_ms,
        minimum=validation_limits.min_duration_ms,
        maximum=validation_limits.max_duration_ms,
        code="invalid_audio_duration",
    )

    header = AudioEnvelopeHeader(
        sequence_number=sequence_number,
        capture_time_ms=capture_time_ms,
        sample_rate=sample_rate,
        channels=channels,
        duration_ms=duration_ms,
        format_code=format_code,
        flags=flags,
    )
    payload = data[AUDIO_ENVELOPE_HEADER_LENGTH_BYTES:]
    expected_payload_length = expected_pcm_payload_length(header)
    if len(payload) != expected_payload_length:
        raise AudioFrameValidationError(
            "invalid_audio_payload_length",
            "Audio frame payload length does not match the envelope metadata",
            details={
                "expected_payload_length": expected_payload_length,
                "actual_payload_length": len(payload),
            },
        )

    return AudioFrame(header=header, payload=bytes(payload))


def expected_pcm_payload_length(header: AudioEnvelopeHeader) -> int:
    bytes_per_sample = _BYTES_PER_SAMPLE[header.format_code]
    sample_milliseconds = header.sample_rate * header.duration_ms
    if sample_milliseconds % 1000 != 0:
        raise AudioFrameValidationError(
            "invalid_audio_duration",
            "Audio frame duration does not resolve to a whole PCM sample count",
            details={
                "sample_rate": header.sample_rate,
                "duration_ms": header.duration_ms,
            },
        )
    samples_per_channel = sample_milliseconds // 1000
    return samples_per_channel * header.channels * bytes_per_sample


def jitter_observation_payload(observation: JitterObservation) -> dict[str, Any]:
    return {
        "sequence_number": observation.sequence_number,
        "missing_sequence_numbers": observation.missing_sequence_numbers,
        "duplicate_sequence_number": observation.duplicate_sequence_number,
        "out_of_order": observation.out_of_order,
        "queue_depth_ms": observation.queue_depth_ms,
        "stats": _jitter_stats_payload(observation.stats),
    }


def _jitter_stats_payload(stats: JitterStats) -> dict[str, int]:
    return {
        "missing_sequence_numbers": stats.missing_sequence_numbers,
        "duplicate_sequence_numbers": stats.duplicate_sequence_numbers,
        "out_of_order_frames": stats.out_of_order_frames,
        "queue_depth_ms": stats.queue_depth_ms,
        "dropped_diagnostic_frames": stats.dropped_diagnostic_frames,
    }


def _validate_range(
    name: str,
    value: int,
    *,
    minimum: int,
    maximum: int,
    code: str,
) -> None:
    if not minimum <= value <= maximum:
        raise AudioFrameValidationError(
            code,
            f"Audio frame {name} is outside the supported range",
            details={
                name: value,
                "minimum": minimum,
                "maximum": maximum,
            },
        )
