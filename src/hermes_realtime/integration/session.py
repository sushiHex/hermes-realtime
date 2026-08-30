"""Stable participant-to-Hermes session bindings."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True, slots=True)
class SessionBinding:
    """One authenticated participant bound to one Hermes session."""

    participant_id: str
    session_id: str
    generation: int


class SessionBindings:
    """Maintain one-to-one stable bindings across media reconnects."""

    def __init__(self) -> None:
        self._by_participant: dict[str, SessionBinding] = {}
        self._by_session: dict[str, SessionBinding] = {}
        self._next_generation = 1
        self._lock = RLock()

    @staticmethod
    def _normalize_identifier(value: str, name: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{name} must not be blank")
        return value

    def bind(self, participant_id: str, session_id: str) -> SessionBinding:
        participant_id = self._normalize_identifier(participant_id, "participant_id")
        session_id = self._normalize_identifier(session_id, "session_id")

        with self._lock:
            participant_binding = self._by_participant.get(participant_id)
            if participant_binding is not None:
                if participant_binding.session_id != session_id:
                    raise ValueError(f"participant already bound: {participant_id}")
                return participant_binding

            session_binding = self._by_session.get(session_id)
            if session_binding is not None:
                raise ValueError(f"session already bound: {session_id}")

            binding = SessionBinding(participant_id, session_id, self._next_generation)
            self._next_generation += 1
            self._by_participant[participant_id] = binding
            self._by_session[session_id] = binding
            return binding

    def session_for(self, participant_id: str) -> str | None:
        binding = self.binding_for(participant_id)
        return None if binding is None else binding.session_id

    def binding_for(self, participant_id: str) -> SessionBinding | None:
        """Return the current immutable binding generation for a participant."""

        participant_id = self._normalize_identifier(participant_id, "participant_id")
        with self._lock:
            return self._by_participant.get(participant_id)

    def is_active(self, binding: SessionBinding) -> bool:
        """Check whether a previously observed binding is still current."""

        with self._lock:
            return self._by_participant.get(binding.participant_id) is binding

    @contextmanager
    def admission(self, binding: SessionBinding) -> Iterator[bool]:
        """Hold the lifecycle lock while work is admitted for a binding."""

        with self._lock:
            yield self._by_participant.get(binding.participant_id) is binding

    def release(self, binding: SessionBinding) -> SessionBinding | None:
        """Release exactly one immutable binding generation from both indexes."""

        with self._lock:
            current = self._by_participant.get(binding.participant_id)
            if current is not binding:
                return None
            del self._by_participant[binding.participant_id]
            if self._by_session.get(binding.session_id) is binding:
                del self._by_session[binding.session_id]
            return binding
