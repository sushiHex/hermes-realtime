from typing import cast

import pytest

from hermes_realtime.speech import (
    AudioFrame,
    DeliveredSpeechLedger,
    PlaybackReceipt,
    SpeechChunk,
    SpeechDeliveryAdmission,
    SpeechDeliveryStage,
    SpeechLedgerCapacityError,
)


def chunk(chunk_id: str, text: str, *, turn_id: str = "turn_001") -> SpeechChunk:
    return SpeechChunk(
        turn_id=turn_id,
        chunk_id=chunk_id,
        text=text,
        audio=AudioFrame(pcm=b"\x00\x00", sample_rate_hz=16_000, channels=1),
    )


@pytest.mark.parametrize("invalid", [True, False, 1.0, float("inf"), float("nan"), "1"])
def test_ledger_requires_exact_integer_retention_bound(invalid: object) -> None:
    with pytest.raises(TypeError, match="exact integer"):
        DeliveredSpeechLedger(max_closed_turns=cast(int, invalid))


def test_ledger_rejects_negative_retention_bound() -> None:
    with pytest.raises(ValueError, match="at least 0"):
        DeliveredSpeechLedger(max_closed_turns=-1)


def test_ledger_rejects_extreme_capacity_configuration() -> None:
    with pytest.raises(ValueError, match="supported maximum"):
        DeliveredSpeechLedger(max_closed_turns=10**100)
    with pytest.raises(ValueError, match="supported maximum"):
        DeliveredSpeechLedger(max_live_chunks=10**100)
    with pytest.raises(ValueError, match="supported maximum"):
        DeliveredSpeechLedger(max_live_pcm_bytes=10**100)
    with pytest.raises(ValueError, match="supported maximum"):
        DeliveredSpeechLedger(max_live_text_chars=10**100)
    with pytest.raises(ValueError, match="supported maximum"):
        DeliveredSpeechLedger(max_closed_chunks=10**100)
    with pytest.raises(ValueError, match="supported maximum"):
        DeliveredSpeechLedger(max_closed_text_chars=10**100)


def test_ledger_live_chunk_capacity_fails_before_mutation() -> None:
    ledger = DeliveredSpeechLedger(
        max_live_chunks=2,
        max_live_pcm_bytes=4,
        max_live_text_chars=32,
    )
    ledger.queue(chunk("chunk_001", "a"))
    ledger.queue(chunk("chunk_002", "b"))

    with pytest.raises(SpeechLedgerCapacityError, match="chunk"):
        ledger.queue(chunk("chunk_003", "c"))

    assert ledger.retained_chunk_count == 2
    assert ledger.stage_history("turn_001", "chunk_003") == ()


def test_ledger_live_byte_and_text_capacity_fail_before_mutation() -> None:
    byte_limited = DeliveredSpeechLedger(
        max_live_chunks=4,
        max_live_pcm_bytes=2,
        max_live_text_chars=32,
    )
    byte_limited.queue(chunk("chunk_001", "a"))
    with pytest.raises(SpeechLedgerCapacityError, match="PCM"):
        byte_limited.queue(chunk("chunk_002", "b"))

    text_limited = DeliveredSpeechLedger(
        max_live_chunks=4,
        max_live_pcm_bytes=8,
        max_live_text_chars=2,
    )
    text_limited.queue(chunk("chunk_001", "ab"))
    with pytest.raises(SpeechLedgerCapacityError, match="text"):
        text_limited.queue(chunk("chunk_002", "c"))

    assert byte_limited.stage_history("turn_001", "chunk_002") == ()
    assert text_limited.stage_history("turn_001", "chunk_002") == ()


def test_ledger_rejects_oversized_identifiers_before_mutation() -> None:
    ledger = DeliveredSpeechLedger()
    item = chunk("x" * 129, "bounded")

    with pytest.raises(ValueError, match="identifier length"):
        ledger.queue(item)

    assert ledger.retained_chunk_count == 0


def test_ledger_retains_only_chunks_confirmed_as_delivered() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "Initial answer.")
    second = chunk("chunk_002", " I am checking that.")
    ledger.queue(first)
    ledger.queue(second)

    receipt = ledger.mark_started("turn_001", first.chunk_id)
    ledger.mark_delivered(receipt)
    cancelled = ledger.cancel_pending("turn_001")

    assert cancelled == (second,)
    assert ledger.delivered_text("turn_001") == "Initial answer."
    assert ledger.pending("turn_001") == ()


def test_ledger_tracks_synthesized_queued_started_and_delivered_stages() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "Initial answer.")

    ledger.queue(first)
    receipt = ledger.mark_started("turn_001", first.chunk_id)
    ledger.mark_delivered(receipt)

    assert ledger.stage_history("turn_001", first.chunk_id) == (
        SpeechDeliveryStage.SYNTHESIZED,
        SpeechDeliveryStage.QUEUED,
        SpeechDeliveryStage.STARTED,
        SpeechDeliveryStage.DELIVERED,
    )


def test_snapshot_turn_counts_is_frozen_text_free_and_precedes_close() -> None:
    from dataclasses import FrozenInstanceError, fields

    from hermes_realtime.speech import SpeechTurnCounts

    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "private first")
    second = chunk("chunk_002", "private second")
    first_admission = SpeechDeliveryAdmission()
    second_admission = SpeechDeliveryAdmission()
    ledger.queue(first, admission=first_admission)
    ledger.queue(second, admission=second_admission)
    first_receipt = ledger.mark_started(first.turn_id, first.chunk_id)
    ledger.mark_started(second.turn_id, second.chunk_id)
    confirmation = ledger.mark_delivered_confirmed(first_receipt)
    assert ledger.consume_delivery_confirmation(confirmation, first_admission) == first.text

    snapshot = ledger.snapshot_turn_counts(first.turn_id)

    assert type(snapshot) is SpeechTurnCounts
    assert tuple(field.name for field in fields(snapshot)) == (
        "queued_chunk_count",
        "started_chunk_count",
        "transport_confirmed_full_count",
        "assistant_delivery_context_recorded",
    )
    assert snapshot == SpeechTurnCounts(
        queued_chunk_count=2,
        started_chunk_count=2,
        transport_confirmed_full_count=1,
        assistant_delivery_context_recorded=True,
    )
    assert "private" not in repr(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.queued_chunk_count = 99  # type: ignore[misc]

    ledger.cancel_pending(first.turn_id)
    ledger.close_turn(first.turn_id)
    with pytest.raises(RuntimeError, match="closed"):
        ledger.snapshot_turn_counts(first.turn_id)


def test_ledger_rejects_delivery_of_unknown_chunk() -> None:
    ledger = DeliveredSpeechLedger()
    receipt = PlaybackReceipt(1, chunk("missing", "Missing."))

    with pytest.raises(KeyError, match="unknown playback receipt"):
        ledger.mark_delivered(receipt)


def test_ledger_rejects_forged_playback_receipt() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "Initial answer.")
    ledger.queue(first)
    receipt = PlaybackReceipt(1, first)

    with pytest.raises(KeyError, match="unknown playback receipt"):
        ledger.mark_delivered(receipt)


def test_chunk_ids_are_scoped_to_a_turn() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "First turn.")
    second = chunk("chunk_001", "Second turn.", turn_id="turn_002")

    ledger.queue(first)
    receipt = ledger.mark_started("turn_001", "chunk_001")
    ledger.mark_delivered(receipt)
    ledger.queue(second)

    assert ledger.pending("turn_002") == (second,)


def test_closing_turn_releases_chunks_but_preserves_delivered_text() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "Delivered.")
    second = chunk("chunk_002", " Cancelled.")
    ledger.queue(first)
    ledger.queue(second)
    receipt = ledger.mark_started("turn_001", "chunk_001")
    ledger.mark_delivered(receipt)
    ledger.cancel_pending("turn_001")

    ledger.close_turn("turn_001")

    assert ledger.retained_chunk_count == 0
    assert ledger.delivered_text("turn_001") == "Delivered."
    ledger.queue(first)
    assert ledger.delivered_text("turn_001") == ""
    assert ledger.stage_history("turn_001", "chunk_001") == (
        SpeechDeliveryStage.SYNTHESIZED,
        SpeechDeliveryStage.QUEUED,
    )


def test_closed_turn_history_is_bounded() -> None:
    ledger = DeliveredSpeechLedger(max_closed_turns=1)
    for turn_id in ("turn_001", "turn_002"):
        item = chunk("chunk_001", turn_id, turn_id=turn_id)
        ledger.queue(item)
        receipt = ledger.mark_started(turn_id, item.chunk_id)
        ledger.mark_delivered(receipt)
        ledger.close_turn(turn_id)

    assert ledger.delivered_text("turn_001") == ""
    assert ledger.stage_history("turn_001", "chunk_001") == ()
    assert ledger.delivered_text("turn_002") == "turn_002"


def test_closed_history_uses_aggregate_chunk_and_text_budgets() -> None:
    ledger = DeliveredSpeechLedger(
        max_closed_turns=10,
        max_closed_chunks=2,
        max_closed_text_chars=3,
    )
    for turn_id in ("turn_001", "turn_002"):
        item = chunk("chunk_001", "aa", turn_id=turn_id)
        ledger.queue(item)
        receipt = ledger.mark_started(turn_id, item.chunk_id)
        ledger.mark_delivered(receipt)
        ledger.close_turn(turn_id)

    assert ledger.delivered_text("turn_001") == ""
    assert ledger.stage_history("turn_001", "chunk_001") == ()
    assert ledger.delivered_text("turn_002") == "aa"


def test_delivery_admission_is_globally_single_use() -> None:
    admission = SpeechDeliveryAdmission()
    first_ledger = DeliveredSpeechLedger()
    first_ledger.queue(chunk("chunk_001", "first"), admission=admission)

    with pytest.raises(ValueError, match="already bound"):
        first_ledger.queue(chunk("chunk_002", "second"), admission=admission)
    with pytest.raises(ValueError, match="already bound"):
        DeliveredSpeechLedger().queue(
            chunk("chunk_003", "third"),
            admission=admission,
        )

    assert tuple(item.chunk_id for item in first_ledger.pending("turn_001")) == (
        "chunk_001",
    )


def test_failed_queue_capacity_does_not_consume_delivery_admission() -> None:
    full_ledger = DeliveredSpeechLedger(max_live_chunks=1)
    full_ledger.queue(chunk("chunk_001", "first"))
    admission = SpeechDeliveryAdmission()

    with pytest.raises(SpeechLedgerCapacityError):
        full_ledger.queue(chunk("chunk_002", "second"), admission=admission)

    replacement = DeliveredSpeechLedger()
    replacement.queue(chunk("chunk_003", "third"), admission=admission)
    assert replacement.pending("turn_001")[0].chunk_id == "chunk_003"


def test_delivery_confirmation_requires_pre_playback_admission() -> None:
    ledger = DeliveredSpeechLedger()
    item = chunk("chunk_001", "delivered")
    ledger.queue(item)
    receipt = ledger.mark_started(item.turn_id, item.chunk_id)

    with pytest.raises(RuntimeError, match="pre-playback"):
        ledger.mark_delivered_confirmed(receipt)

    assert ledger.mark_delivered(receipt) == item


def test_confirmation_retains_no_pcm_outside_live_budget() -> None:
    ledger = DeliveredSpeechLedger(max_live_chunks=2, max_live_pcm_bytes=2)
    first_admission = SpeechDeliveryAdmission()
    first = chunk("chunk_001", "first")
    ledger.queue(first, admission=first_admission)
    confirmation = ledger.mark_delivered_confirmed(
        ledger.mark_started(first.turn_id, first.chunk_id)
    )

    second = chunk("chunk_002", "second")
    ledger.queue(second, admission=SpeechDeliveryAdmission())

    assert ledger.confirmed_text(confirmation, first_admission) == "first"
    assert ledger.pending(first.turn_id) == (second,)


def test_close_turn_releases_unconsumed_delivery_confirmations() -> None:
    ledger = DeliveredSpeechLedger()
    item = chunk("chunk_001", "delivered")
    admission = SpeechDeliveryAdmission()
    ledger.queue(item, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(
        ledger.mark_started(item.turn_id, item.chunk_id)
    )

    ledger.close_turn(item.turn_id)

    with pytest.raises(KeyError, match="delivery confirmation"):
        ledger.confirmed_text(confirmation, admission)


def test_begin_turn_clears_closed_incarnation_delivery_and_stage_history() -> None:
    ledger = DeliveredSpeechLedger()
    item = chunk("chunk_001", "Old incarnation.")
    ledger.queue(item)
    receipt = ledger.mark_started(item.turn_id, item.chunk_id)
    ledger.mark_delivered(receipt)
    ledger.close_turn(item.turn_id)
    assert ledger.delivered_text(item.turn_id) == item.text
    assert ledger.stage_history(item.turn_id, item.chunk_id)

    ledger.begin_turn(item.turn_id)

    assert ledger.delivered_text(item.turn_id) == ""
    assert ledger.stage_history(item.turn_id, item.chunk_id) == ()


def test_delivered_text_uses_playback_start_order_not_confirmation_order() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "first")
    second = chunk("chunk_002", "second")
    ledger.queue(first)
    ledger.queue(second)
    first_receipt = ledger.mark_started("turn_001", first.chunk_id)
    second_receipt = ledger.mark_started("turn_001", second.chunk_id)

    ledger.mark_delivered(second_receipt)
    assert ledger.delivered_text("turn_001") == "second"
    ledger.mark_delivered(first_receipt)
    assert ledger.delivered_text("turn_001") == "firstsecond"


def test_reconstructed_playback_receipt_is_rejected() -> None:
    ledger = DeliveredSpeechLedger()
    item = chunk("chunk_001", "Delivered once.")
    ledger.queue(item)
    issued = ledger.mark_started("turn_001", "chunk_001")
    reconstructed = PlaybackReceipt(issued.receipt_id, issued.chunk)

    with pytest.raises(KeyError, match="unknown playback receipt"):
        ledger.mark_delivered(reconstructed)
    assert ledger.mark_delivered(issued) == item


def test_stale_playback_receipt_cannot_deliver_reused_ids() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "Old incarnation.")
    ledger.queue(first)
    stale_receipt = ledger.mark_started("turn_001", "chunk_001")
    ledger.cancel_pending("turn_001")
    ledger.close_turn("turn_001")

    replacement = chunk("chunk_001", "New incarnation.")
    ledger.queue(replacement)
    ledger.mark_started("turn_001", "chunk_001")

    with pytest.raises(KeyError, match="unknown playback receipt"):
        ledger.mark_delivered(stale_receipt)
    assert ledger.delivered_text("turn_001") == ""
    assert ledger.started("turn_001") == (replacement,)


def test_ledger_rejects_duplicate_chunk_ids_within_one_turn() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "Initial answer.")
    ledger.queue(first)

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        ledger.queue(first)


def test_ledger_rejects_forged_chunk_fields_before_hashing() -> None:
    touched = False

    class SideEffectString(str):
        def __hash__(self) -> int:
            nonlocal touched
            touched = True
            raise AssertionError("untrusted hash reached")

    forged = object.__new__(SpeechChunk)
    object.__setattr__(forged, "turn_id", SideEffectString("turn_001"))
    object.__setattr__(forged, "chunk_id", "chunk_001")
    object.__setattr__(forged, "text", "untrusted")
    object.__setattr__(
        forged,
        "audio",
        AudioFrame(pcm=b"\x00\x00", sample_rate_hz=16_000, channels=1),
    )
    ledger = DeliveredSpeechLedger()

    with pytest.raises(TypeError):
        ledger.queue(forged)

    assert touched is False
    assert ledger.retained_chunk_count == 0


def test_ledger_lookup_rejects_string_subclasses_before_hashing() -> None:
    touched = False

    class SideEffectString(str):
        def __hash__(self) -> int:
            nonlocal touched
            touched = True
            raise AssertionError("untrusted hash reached")

    ledger = DeliveredSpeechLedger()
    ledger.queue(chunk("chunk_001", "trusted"))

    with pytest.raises(TypeError, match="exact built-in string"):
        ledger.mark_started("turn_001", SideEffectString("chunk_001"))

    assert touched is False
    assert ledger.started("turn_001") == ()


def test_ledger_rejects_forged_audio_frame_fields() -> None:
    forged_audio = object.__new__(AudioFrame)
    object.__setattr__(forged_audio, "pcm", bytearray(b"\x00\x00"))
    object.__setattr__(forged_audio, "sample_rate_hz", 16_000)
    object.__setattr__(forged_audio, "channels", 1)
    forged_chunk = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_001",
        text="untrusted",
        audio=forged_audio,
    )
    ledger = DeliveredSpeechLedger()

    with pytest.raises(TypeError):
        ledger.queue(forged_chunk)

    assert ledger.retained_chunk_count == 0


def test_mutated_issued_receipt_cannot_retarget_delivery() -> None:
    ledger = DeliveredSpeechLedger()
    first = chunk("chunk_001", "alpha")
    second = chunk("chunk_002", "bravo")
    ledger.queue(first)
    ledger.queue(second)
    exposed_pending = ledger.pending("turn_001")[0]
    object.__setattr__(exposed_pending, "text", "pending view tampered")
    first_receipt = ledger.mark_started(first.turn_id, first.chunk_id)
    second_receipt = ledger.mark_started(second.turn_id, second.chunk_id)

    object.__setattr__(first, "text", "caller tampered")
    object.__setattr__(first_receipt.chunk, "text", "receipt chunk tampered")
    object.__setattr__(first_receipt, "chunk", second)
    object.__setattr__(first_receipt, "receipt_id", 1001)

    assert ledger.mark_delivered(first_receipt).text == "alpha"
    assert ledger.mark_delivered(second_receipt) == second
    assert ledger.delivered_text("turn_001") == "alphabravo"
    assert ledger.pending("turn_001") == ()


def test_ledger_pins_started_receipts_until_they_settle() -> None:
    # Regression: the ledger held started receipts only weakly while indexing
    # them by id(receipt). A receipt dropped without being delivered could be
    # collected and its address recycled onto a later receipt, cross-linking two
    # receipt IDs to one object id and stranding the live chunk.
    import gc
    import weakref

    ledger = DeliveredSpeechLedger()
    queued = chunk("chunk_001", "alpha")
    ledger.queue(queued)
    receipt = ledger.mark_started(queued.turn_id, queued.chunk_id)
    observer = weakref.ref(receipt)

    del receipt
    gc.collect()
    assert observer() is not None

    ledger.cancel_pending("turn_001")
    gc.collect()
    assert observer() is None


def test_ledger_releases_started_receipts_after_delivery() -> None:
    import gc
    import weakref

    ledger = DeliveredSpeechLedger()
    queued = chunk("chunk_001", "alpha")
    ledger.queue(queued)
    receipt = ledger.mark_started(queued.turn_id, queued.chunk_id)
    observer = weakref.ref(receipt)

    assert ledger.mark_delivered(receipt).text == "alpha"
    del receipt
    gc.collect()
    assert observer() is None
