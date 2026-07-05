"""Shared protocol contracts for backend and extension transport.

The binary audio envelope is defined here as constants only. Actual packing,
validation, and ingest behavior are added by later transport prompts.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionState(StrEnum):
    idle = "idle"
    starting = "starting"
    recording = "recording"
    paused = "paused"
    stopping = "stopping"
    finalizing = "finalizing"
    complete = "complete"
    error = "error"


class SchedulerProfile(StrEnum):
    realtime = "realtime"
    balanced = "balanced"
    accuracy = "accuracy"


class ASRMode(StrEnum):
    off = "off"
    probe = "probe"
    normal = "normal"
    boundary = "boundary"
    repair = "repair"
    final = "final"


class DiarizationMode(StrEnum):
    off = "off"
    speaker_change_only = "speaker_change_only"
    embedding = "embedding"
    final = "final"


class LLMMode(StrEnum):
    off = "off"
    boundary_classifier_only = "boundary_classifier_only"
    live_editor = "live_editor"
    final = "final"


class FilterMode(StrEnum):
    bypass = "bypass"
    light = "light"
    enhance_candidate = "enhance_candidate"
    enhanced = "enhanced"


class AudioFormatCode(IntEnum):
    pcm_s16le = 1


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


class AudioEnvelopeHeader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    magic: Literal["CTTS"] = "CTTS"
    version: Literal[1] = 1
    header_length: Literal[32] = 32
    sequence_number: int = Field(ge=0)
    capture_time_ms: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    duration_ms: int = Field(ge=0)
    format_code: AudioFormatCode = AudioFormatCode.pcm_s16le
    flags: int = Field(default=0, ge=0)


AUDIO_ENVELOPE_ENDIANNESS = "little"
AUDIO_ENVELOPE_MAGIC = b"CTTS"
AUDIO_ENVELOPE_VERSION = 1
AUDIO_ENVELOPE_HEADER_LENGTH_BYTES = 32
AUDIO_FORMAT_PCM_S16LE = AudioFormatCode.pcm_s16le

AUDIO_ENVELOPE_FIELDS: tuple[str, ...] = (
    "magic",
    "version",
    "header_length",
    "sequence_number",
    "capture_time_ms",
    "sample_rate",
    "channels",
    "duration_ms",
    "format_code",
    "flags",
)

# Little-endian, fixed-width header:
# magic(4s), version(u8), header_length(u8), sequence(u64),
# capture_time_ms(u64), sample_rate(u32), channels(u16),
# duration_ms(u16), format_code(u8), flags(u8) = 32 bytes.
AUDIO_ENVELOPE_STRUCT_FORMAT = "<4sBBQQIHHBB"
