"""Deterministic foreground turn lifecycle."""

from collections import deque
from enum import StrEnum


class TurnState(StrEnum):
    """Observable foreground conversation states."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    RESPONDING = "responding"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    RECOVERING = "recovering"


_ALLOWED_TRANSITIONS: dict[TurnState, frozenset[TurnState]] = {
    TurnState.IDLE: frozenset({TurnState.LISTENING, TurnState.RESPONDING}),
    TurnState.LISTENING: frozenset({TurnState.TRANSCRIBING, TurnState.IDLE}),
    TurnState.TRANSCRIBING: frozenset(
        {TurnState.LISTENING, TurnState.RESPONDING, TurnState.IDLE}
    ),
    TurnState.RESPONDING: frozenset(
        {
            TurnState.SPEAKING,
            TurnState.INTERRUPTED,
            TurnState.RECOVERING,
            TurnState.IDLE,
        }
    ),
    TurnState.SPEAKING: frozenset(
        {TurnState.INTERRUPTED, TurnState.RECOVERING, TurnState.IDLE}
    ),
    TurnState.INTERRUPTED: frozenset(
        {TurnState.RESPONDING, TurnState.RECOVERING, TurnState.IDLE}
    ),
    TurnState.RECOVERING: frozenset({TurnState.INTERRUPTED, TurnState.IDLE}),
}


class TurnStateMachine:
    """Fail closed on invalid foreground turn transitions."""

    def __init__(self, *, history_limit: int = 256) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self._current = TurnState.IDLE
        self._history: deque[TurnState] = deque(
            (self._current,), maxlen=history_limit
        )

    @property
    def current(self) -> TurnState:
        return self._current

    @property
    def history(self) -> tuple[TurnState, ...]:
        return tuple(self._history)

    def transition(self, next_state: TurnState) -> None:
        if next_state not in _ALLOWED_TRANSITIONS[self._current]:
            raise RuntimeError(
                f"invalid turn transition: {self._current.value} -> {next_state.value}"
            )
        self._current = next_state
        self._history.append(next_state)
