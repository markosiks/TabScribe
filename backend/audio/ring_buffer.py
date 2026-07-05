from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol


class AudioFrameLike(Protocol):
    duration_ms: int
    payload: bytes


class AudioBufferFullError(Exception):
    def __init__(
        self,
        *,
        max_depth_ms: int,
        current_depth_ms: int,
        frame_duration_ms: int,
    ) -> None:
        super().__init__(
            "Audio buffer is full: "
            f"{current_depth_ms}ms queued, "
            f"{frame_duration_ms}ms incoming, "
            f"{max_depth_ms}ms maximum"
        )
        self.max_depth_ms = max_depth_ms
        self.current_depth_ms = current_depth_ms
        self.frame_duration_ms = frame_duration_ms


@dataclass(frozen=True, slots=True)
class AudioBufferSnapshot:
    frame_count: int
    depth_ms: int
    byte_count: int
    max_depth_ms: int
    max_frames: int


class AudioRingBuffer:
    def __init__(self, *, max_depth_ms: int, max_frames: int) -> None:
        if max_depth_ms < 1:
            raise ValueError("max_depth_ms must be greater than zero")
        if max_frames < 1:
            raise ValueError("max_frames must be greater than zero")

        self._max_depth_ms = max_depth_ms
        self._max_frames = max_frames
        self._frames: deque[AudioFrameLike] = deque()
        self._depth_ms = 0
        self._byte_count = 0

    @property
    def depth_ms(self) -> int:
        return self._depth_ms

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def append(self, frame: AudioFrameLike) -> None:
        if len(self._frames) >= self._max_frames:
            raise AudioBufferFullError(
                max_depth_ms=self._max_depth_ms,
                current_depth_ms=self._depth_ms,
                frame_duration_ms=frame.duration_ms,
            )
        if self._depth_ms + frame.duration_ms > self._max_depth_ms:
            raise AudioBufferFullError(
                max_depth_ms=self._max_depth_ms,
                current_depth_ms=self._depth_ms,
                frame_duration_ms=frame.duration_ms,
            )

        self._frames.append(frame)
        self._depth_ms += frame.duration_ms
        self._byte_count += len(frame.payload)

    def pop_left(self) -> AudioFrameLike | None:
        if not self._frames:
            return None

        frame = self._frames.popleft()
        self._depth_ms -= frame.duration_ms
        self._byte_count -= len(frame.payload)
        return frame

    def clear(self) -> None:
        self._frames.clear()
        self._depth_ms = 0
        self._byte_count = 0

    def snapshot(self) -> AudioBufferSnapshot:
        return AudioBufferSnapshot(
            frame_count=len(self._frames),
            depth_ms=self._depth_ms,
            byte_count=self._byte_count,
            max_depth_ms=self._max_depth_ms,
            max_frames=self._max_frames,
        )

