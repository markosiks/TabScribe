from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from threading import RLock
from types import MappingProxyType
from uuid import uuid4

from .protocol import SchedulerProfile, SessionState


Clock = Callable[[], datetime]


_ALLOWED_TRANSITIONS = MappingProxyType(
    {
        SessionState.idle: frozenset(
            {
                SessionState.starting,
                SessionState.recording,
                SessionState.stopping,
                SessionState.error,
            }
        ),
        SessionState.starting: frozenset(
            {
                SessionState.recording,
                SessionState.stopping,
                SessionState.error,
            }
        ),
        SessionState.recording: frozenset(
            {
                SessionState.paused,
                SessionState.stopping,
                SessionState.error,
            }
        ),
        SessionState.paused: frozenset(
            {
                SessionState.recording,
                SessionState.stopping,
                SessionState.error,
            }
        ),
        SessionState.stopping: frozenset(
            {
                SessionState.finalizing,
                SessionState.complete,
                SessionState.error,
            }
        ),
        SessionState.finalizing: frozenset(
            {SessionState.complete, SessionState.error}
        ),
        SessionState.complete: frozenset(),
        SessionState.error: frozenset(),
    }
)
_STOPPABLE_STATES = frozenset(
    {
        SessionState.idle,
        SessionState.starting,
        SessionState.recording,
        SessionState.paused,
    }
)
_STOP_IDEMPOTENT_STATES = frozenset(
    {SessionState.stopping, SessionState.finalizing, SessionState.complete}
)


class SessionError(Exception):
    pass


class SessionNotFoundError(SessionError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = session_id


class InvalidSessionTransitionError(SessionError):
    def __init__(
        self,
        *,
        session_id: str,
        current_state: SessionState,
        requested_state: SessionState,
    ) -> None:
        super().__init__(
            "Invalid session state transition: "
            f"{current_state.value} -> {requested_state.value}"
        )
        self.session_id = session_id
        self.current_state = current_state
        self.requested_state = requested_state


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    token: str
    profile: SchedulerProfile
    state: SessionState
    created_at: datetime
    last_activity_at: datetime
    glossary_version: str | None = None


class SessionManager:
    def __init__(
        self,
        *,
        default_profile: SchedulerProfile,
        stale_after_seconds: int,
        clock: Clock | None = None,
    ) -> None:
        if stale_after_seconds < 1:
            raise ValueError("stale_after_seconds must be greater than zero")

        self._default_profile = default_profile
        self._stale_after = timedelta(seconds=stale_after_seconds)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = RLock()

    @property
    def stale_after_seconds(self) -> int:
        return int(self._stale_after.total_seconds())

    def create_session(
        self, profile: SchedulerProfile | None = None
    ) -> SessionRecord:
        now = self._now()
        session_id = str(uuid4())
        session = SessionRecord(
            session_id=session_id,
            token=secrets.token_urlsafe(32),
            profile=profile or self._default_profile,
            state=SessionState.idle,
            created_at=now,
            last_activity_at=now,
        )

        with self._lock:
            self._expire_stale_locked(now)
            self._sessions[session_id] = session

        return session

    def get_session(
        self, session_id: str, *, touch: bool = False
    ) -> SessionRecord | None:
        now = self._now()
        with self._lock:
            session = self._get_live_session_locked(session_id, now)
            if session is None:
                return None
            if not touch:
                return session
            return self._touch_locked(session, now)

    def authenticate_session(
        self, session_id: str, token: str | None
    ) -> SessionRecord | None:
        if not token:
            return None

        now = self._now()
        with self._lock:
            session = self._get_live_session_locked(session_id, now)
            if session is None:
                return None
            if not secrets.compare_digest(session.token, token):
                return None
            return self._touch_locked(session, now)

    def validate_token(self, session_id: str, token: str | None) -> bool:
        return self.authenticate_session(session_id, token) is not None

    def touch_session(self, session_id: str) -> SessionRecord:
        now = self._now()
        with self._lock:
            session = self._require_live_session_locked(session_id, now)
            return self._touch_locked(session, now)

    def transition_state(
        self, session_id: str, requested_state: SessionState
    ) -> SessionRecord:
        now = self._now()
        with self._lock:
            session = self._require_live_session_locked(session_id, now)
            return self._transition_locked(session, requested_state, now)

    def update_profile(
        self, session_id: str, profile: SchedulerProfile
    ) -> SessionRecord:
        now = self._now()
        with self._lock:
            session = self._require_live_session_locked(session_id, now)
            updated = replace(
                session,
                profile=profile,
                last_activity_at=now,
            )
            self._sessions[session_id] = updated
            return updated

    def request_stop(self, session_id: str) -> SessionRecord:
        now = self._now()
        with self._lock:
            session = self._require_live_session_locked(session_id, now)
            if session.state in _STOP_IDEMPOTENT_STATES:
                return self._touch_locked(session, now)
            if session.state not in _STOPPABLE_STATES:
                raise InvalidSessionTransitionError(
                    session_id=session.session_id,
                    current_state=session.state,
                    requested_state=SessionState.stopping,
                )
            return self._transition_locked(session, SessionState.stopping, now)

    def request_finalize(self, session_id: str) -> SessionRecord:
        now = self._now()
        with self._lock:
            session = self._require_live_session_locked(session_id, now)
            if session.state in {SessionState.finalizing, SessionState.complete}:
                return self._touch_locked(session, now)
            if session.state is not SessionState.stopping:
                raise InvalidSessionTransitionError(
                    session_id=session.session_id,
                    current_state=session.state,
                    requested_state=SessionState.finalizing,
                )
            return self._transition_locked(session, SessionState.finalizing, now)

    def update_glossary(
        self, session_id: str, glossary_version: str | None = None
    ) -> SessionRecord:
        now = self._now()
        version = glossary_version or now.isoformat()
        with self._lock:
            session = self._require_live_session_locked(session_id, now)
            updated = replace(
                session,
                glossary_version=version,
                last_activity_at=now,
            )
            self._sessions[session_id] = updated
            return updated

    def expire_stale_sessions(self) -> int:
        now = self._now()
        with self._lock:
            return self._expire_stale_locked(now)

    def _transition_locked(
        self,
        session: SessionRecord,
        requested_state: SessionState,
        now: datetime,
    ) -> SessionRecord:
        if session.state is requested_state:
            return self._touch_locked(session, now)

        allowed_states = _ALLOWED_TRANSITIONS[session.state]
        if requested_state not in allowed_states:
            raise InvalidSessionTransitionError(
                session_id=session.session_id,
                current_state=session.state,
                requested_state=requested_state,
            )

        updated = replace(
            session,
            state=requested_state,
            last_activity_at=now,
        )
        self._sessions[session.session_id] = updated
        return updated

    def _touch_locked(
        self, session: SessionRecord, now: datetime
    ) -> SessionRecord:
        updated = replace(session, last_activity_at=now)
        self._sessions[session.session_id] = updated
        return updated

    def _require_live_session_locked(
        self, session_id: str, now: datetime
    ) -> SessionRecord:
        session = self._get_live_session_locked(session_id, now)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def _get_live_session_locked(
        self, session_id: str, now: datetime
    ) -> SessionRecord | None:
        self._expire_stale_locked(now)
        return self._sessions.get(session_id)

    def _expire_stale_locked(self, now: datetime) -> int:
        expired_session_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_activity_at > self._stale_after
        ]
        for session_id in expired_session_ids:
            del self._sessions[session_id]
        return len(expired_session_ids)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Session clock must return timezone-aware datetimes")
        return value.astimezone(timezone.utc)
