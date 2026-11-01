from scripts.lookup_backend_task import _known_execution_guard


def test_execution_guard_diagnostics_preserve_known_labels_and_redact_private_text() -> None:
    assert _known_execution_guard(ValueError("accepted_ack")) == "accepted_ack"
    assert (
        _known_execution_guard(
            ValueError("SYNTHETIC_PRIVATE_ERROR https://private.invalid/token")
        )
        == "other"
    )
    assert _known_execution_guard(RuntimeError("accepted_ack")) == "other"
