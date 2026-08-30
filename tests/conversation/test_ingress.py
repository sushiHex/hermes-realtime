import pytest

from hermes_realtime.conversation.ingress import (
    BoundedPcmIngress,
    IngressOverloadError,
    IngressRecord,
)
from hermes_realtime.speech import AudioFrame


def ingress_record(sequence: int, *, observed_at: float) -> IngressRecord:
    return IngressRecord(
        sequence=sequence,
        participant_identity="browser_user",
        session_generation=7,
        track_name="microphone-3",
        frame=AudioFrame(
            pcm=sequence.to_bytes(2, "little") * 160,
            sample_rate_hz=16_000,
            channels=1,
        ),
        observed_at=observed_at,
    )


@pytest.mark.asyncio
async def test_queue_age_faults_before_remaining_capacity_can_hide_consumer_lag() -> None:
    now = 10.0
    ingress = BoundedPcmIngress(
        max_frames=4,
        max_bytes=4096,
        max_age_seconds=0.1,
        clock=lambda: now,
    )
    first = ingress_record(1, observed_at=now)
    ingress.admit(first)
    assert await ingress.receive() is first
    ingress.admit(ingress_record(2, observed_at=now))

    now += 0.101
    with pytest.raises(IngressOverloadError, match="age"):
        ingress.admit(ingress_record(3, observed_at=now))

    ingress.complete(first)


@pytest.mark.asyncio
async def test_default_ingress_recovers_after_repeated_half_second_consumer_stalls() -> None:
    now = 100.0
    ingress = BoundedPcmIngress(clock=lambda: now)
    next_sequence = 1

    for _turn in range(5):
        active = ingress_record(next_sequence, observed_at=now)
        next_sequence += 1
        ingress.admit(active)
        assert await ingress.receive() is active

        queued: list[IngressRecord] = []
        for _frame in range(50):
            record = ingress_record(next_sequence, observed_at=now)
            next_sequence += 1
            ingress.admit(record)
            queued.append(record)

        now += 0.5
        ingress.complete(active)
        for expected in queued:
            claimed = await ingress.receive()
            assert claimed is expected
            ingress.complete(claimed)

    assert ingress.outstanding_frames == 0
    assert ingress.outstanding_bytes == 0
    ingress.check_health()


@pytest.mark.asyncio
async def test_duplicate_sequence_faults_before_the_record_can_be_consumed() -> None:
    ingress = BoundedPcmIngress(
        max_frames=4,
        max_bytes=4096,
        max_age_seconds=0.1,
        clock=lambda: 20.0,
    )
    first = ingress_record(1, observed_at=20.0)
    ingress.admit(first)
    assert await ingress.receive() is first
    ingress.complete(first)

    with pytest.raises(RuntimeError, match="sequence"):
        ingress.admit(ingress_record(1, observed_at=20.0))


@pytest.mark.asyncio
async def test_health_check_faults_stale_queue_without_another_pcm_admission() -> None:
    now = 30.0
    ingress = BoundedPcmIngress(
        max_frames=4,
        max_bytes=4096,
        max_age_seconds=0.1,
        clock=lambda: now,
    )
    first = ingress_record(1, observed_at=now)
    ingress.admit(first)
    assert await ingress.receive() is first
    ingress.admit(ingress_record(2, observed_at=now))

    now += 0.101
    with pytest.raises(IngressOverloadError, match="age"):
        ingress.check_health()

    ingress.complete(first)
