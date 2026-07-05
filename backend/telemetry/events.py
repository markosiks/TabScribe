from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from backend.app.protocol import EventEnvelope


ClockMs = Callable[[], int]


@dataclass(frozen=True, slots=True)
class EventPublishResult:
    delivered: int
    dropped: int


class EventSubscription:
    def __init__(
        self,
        *,
        session_id: str,
        queue: asyncio.Queue[EventEnvelope],
        broker: EventBroker,
    ) -> None:
        self.session_id = session_id
        self._queue = queue
        self._broker = broker
        self._closed = False

    async def receive(self) -> EventEnvelope:
        return await self._queue.get()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._broker.unsubscribe(self)


class EventBroker:
    def __init__(
        self,
        *,
        subscriber_queue_size: int = 128,
        clock_ms: ClockMs | None = None,
    ) -> None:
        if subscriber_queue_size < 1:
            raise ValueError("subscriber_queue_size must be greater than zero")

        self._subscriber_queue_size = subscriber_queue_size
        self._clock_ms = clock_ms or _utc_now_ms
        self._subscribers: dict[str, set[asyncio.Queue[EventEnvelope]]] = defaultdict(
            set
        )
        self._lock = asyncio.Lock()
        self._dropped_diagnostic_frames: dict[str, int] = defaultdict(int)

    @property
    def subscriber_queue_size(self) -> int:
        return self._subscriber_queue_size

    async def subscribe(self, session_id: str) -> EventSubscription:
        queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(
            maxsize=self._subscriber_queue_size
        )
        async with self._lock:
            self._subscribers[session_id].add(queue)
        return EventSubscription(session_id=session_id, queue=queue, broker=self)

    async def unsubscribe(self, subscription: EventSubscription) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(subscription.session_id)
            if subscribers is None:
                return
            subscribers.discard(subscription._queue)
            if not subscribers:
                self._subscribers.pop(subscription.session_id, None)

    async def publish(self, event: EventEnvelope) -> EventPublishResult:
        async with self._lock:
            subscribers = tuple(self._subscribers.get(event.session_id, ()))

        delivered = 0
        dropped = 0
        for queue in subscribers:
            if queue.full():
                self._drop_oldest(queue, event.session_id)
                dropped += 1
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dropped += 1
            else:
                delivered += 1

        return EventPublishResult(delivered=delivered, dropped=dropped)

    async def make_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> EventEnvelope:
        return EventEnvelope(
            type=event_type,
            session_id=session_id,
            timestamp_ms=self._clock_ms(),
            payload=payload or {},
        )

    async def publish_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> EventPublishResult:
        return await self.publish(
            await self.make_event(
                session_id=session_id,
                event_type=event_type,
                payload=payload,
            )
        )

    async def dropped_diagnostic_frames(self, session_id: str) -> int:
        async with self._lock:
            return self._dropped_diagnostic_frames.get(session_id, 0)

    def _drop_oldest(
        self, queue: asyncio.Queue[EventEnvelope], session_id: str
    ) -> None:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        self._dropped_diagnostic_frames[session_id] += 1


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
