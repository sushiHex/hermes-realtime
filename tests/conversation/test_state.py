from hermes_realtime.conversation import TurnState, TurnStateMachine


def test_state_history_is_bounded() -> None:
    machine = TurnStateMachine(history_limit=3)

    for _ in range(2):
        machine.transition(TurnState.RESPONDING)
        machine.transition(TurnState.INTERRUPTED)
        machine.transition(TurnState.IDLE)

    assert machine.history == (
        TurnState.RESPONDING,
        TurnState.INTERRUPTED,
        TurnState.IDLE,
    )
