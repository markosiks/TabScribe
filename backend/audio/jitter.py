from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JitterStats:
    missing_sequence_numbers: int = 0
    duplicate_sequence_numbers: int = 0
    out_of_order_frames: int = 0
    queue_depth_ms: int = 0
    dropped_diagnostic_frames: int = 0


@dataclass(frozen=True, slots=True)
class JitterObservation:
    sequence_number: int
    missing_sequence_numbers: int
    duplicate_sequence_number: bool
    out_of_order: bool
    queue_depth_ms: int
    stats: JitterStats


class JitterTracker:
    def __init__(self, *, recent_window_size: int = 2048) -> None:
        if recent_window_size < 1:
            raise ValueError("recent_window_size must be greater than zero")

        self._expected_sequence_number: int | None = None
        self._recent_sequence_numbers: deque[int] = deque(maxlen=recent_window_size)
        self._recent_lookup: set[int] = set()
        self._missing_sequence_numbers = 0
        self._duplicate_sequence_numbers = 0
        self._out_of_order_frames = 0
        self._queue_depth_ms = 0
        self._dropped_diagnostic_frames = 0

    @property
    def stats(self) -> JitterStats:
        return JitterStats(
            missing_sequence_numbers=self._missing_sequence_numbers,
            duplicate_sequence_numbers=self._duplicate_sequence_numbers,
            out_of_order_frames=self._out_of_order_frames,
            queue_depth_ms=self._queue_depth_ms,
            dropped_diagnostic_frames=self._dropped_diagnostic_frames,
        )

    def observe(self, sequence_number: int, *, queue_depth_ms: int) -> JitterObservation:
        if sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        if queue_depth_ms < 0:
            raise ValueError("queue_depth_ms must be non-negative")

        self._queue_depth_ms = queue_depth_ms
        missing = 0
        duplicate = sequence_number in self._recent_lookup
        out_of_order = False

        if duplicate:
            self._duplicate_sequence_numbers += 1
        elif self._expected_sequence_number is None:
            self._expected_sequence_number = sequence_number + 1
            self._remember(sequence_number)
        elif sequence_number == self._expected_sequence_number:
            self._expected_sequence_number += 1
            self._remember(sequence_number)
        elif sequence_number > self._expected_sequence_number:
            missing = sequence_number - self._expected_sequence_number
            self._missing_sequence_numbers += missing
            self._expected_sequence_number = sequence_number + 1
            self._remember(sequence_number)
        else:
            out_of_order = True
            self._out_of_order_frames += 1
            self._remember(sequence_number)

        return JitterObservation(
            sequence_number=sequence_number,
            missing_sequence_numbers=missing,
            duplicate_sequence_number=duplicate,
            out_of_order=out_of_order,
            queue_depth_ms=self._queue_depth_ms,
            stats=self.stats,
        )

    def record_dropped_diagnostic_frames(self, count: int = 1) -> JitterStats:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._dropped_diagnostic_frames += count
        return self.stats

    def update_queue_depth(self, queue_depth_ms: int) -> JitterStats:
        if queue_depth_ms < 0:
            raise ValueError("queue_depth_ms must be non-negative")
        self._queue_depth_ms = queue_depth_ms
        return self.stats

    def _remember(self, sequence_number: int) -> None:
        if len(self._recent_sequence_numbers) == self._recent_sequence_numbers.maxlen:
            expired = self._recent_sequence_numbers.popleft()
            self._recent_lookup.discard(expired)
        self._recent_sequence_numbers.append(sequence_number)
        self._recent_lookup.add(sequence_number)

