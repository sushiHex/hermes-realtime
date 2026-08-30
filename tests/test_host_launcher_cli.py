from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.client.session import BrowserBindingSnapshot
from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationWorkControlSurface,
)
from hermes_realtime.evidence import models as evidence_models
from hermes_realtime.host_launcher import (
    _bind_natural_work_tools,
    _build_argument_parser,
    _build_browser_model_configuration,
    _build_browser_selectable_model_catalog,
    _conversation_profile_duration_limit,
    _FullHostWorkCloseOwner,
    _load_tailnet_authorizer,
    _operator_evidence_capture_enabled,
    _project_conversation_observation,
    _qualification_checkpoint_contract,
    _readiness_cue_chunk,
    _ReadinessCueAuthority,
    _ReadinessCueTasks,
    _reserve_host_evidence_revoke,
    _resolve_evidence_database_path,
    _run_full_host_preflight,
    _run_host_cli,
    _SerializedSpeechPlayback,
    _SessionTokenUsageAccumulator,
    _voice_input_ready_data,
    main,
)
from hermes_realtime.providers.codex_app_server import (
    CodexAppServerStreamingInference,
    CodexModelConfiguration,
    CodexModelOption,
    CodexTokenUsage,
)
from hermes_realtime.speech import AudioFrame, SpeechChunk


def test_host_launcher_imports_first_from_candidate_source_root() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [sys.executable, "-c", "import hermes_realtime.host_launcher"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


class _BlockingPlayback:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def play(self, chunk: SpeechChunk, *, is_valid: Callable[[], bool]) -> None:
        del is_valid
        self.calls.append(chunk.chunk_id)
        if len(self.calls) == 1:
            self.first_started.set()
            await self.release_first.wait()

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class _UnusedTaskController:
    async def dispatch(self, *, objective: str, utterance_id: str) -> object:
        del objective, utterance_id
        raise AssertionError("preflight must not dispatch work")

    async def request_cancel(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> object:
        del task_id, reason
        raise AssertionError("preflight must not cancel work")


def _work_surface() -> ConversationWorkControlSurface:
    return ConversationWorkControlSurface(
        controller=_UnusedTaskController(),
        context=ConversationContextStore(),
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
    )


def _usage(total: int, *, context: int = 272_000) -> CodexTokenUsage:
    return CodexTokenUsage(
        input_tokens=total - 30,
        cached_input_tokens=10,
        output_tokens=20,
        reasoning_output_tokens=10,
        total_tokens=total,
        context_window_tokens=context,
    )


def test_session_usage_replaces_turn_updates_and_excludes_preflight() -> None:
    accumulator = _SessionTokenUsageAccumulator()

    assert accumulator.observe("turn_preflight", _usage(100)) is None
    assert accumulator.observe("turn_1", _usage(100))["totalTokens"] == 100
    assert accumulator.observe("turn_1", _usage(140))["totalTokens"] == 140
    total = accumulator.observe("turn_2", _usage(60))

    assert total == {
        "cachedInputTokens": 20,
        "contextWindowTokens": 272_000,
        "inputTokens": 140,
        "outputTokens": 40,
        "reasoningOutputTokens": 20,
        "totalTokens": 200,
    }


def test_session_usage_resets_for_a_new_browser_generation_and_saturates_public_counters() -> None:
    accumulator = _SessionTokenUsageAccumulator()
    accumulator.reset("browser_first", 1)
    assert accumulator.observe("turn_1", _usage(100))["totalTokens"] == 100

    accumulator.reset("browser_second", 2)
    second = accumulator.observe("turn_1", _usage((1 << 53) + 100))

    assert second == {
        "cachedInputTokens": 10,
        "contextWindowTokens": 272_000,
        "inputTokens": (1 << 53) - 1,
        "outputTokens": 20,
        "reasoningOutputTokens": 10,
        "totalTokens": (1 << 53) - 1,
    }


def test_readiness_cue_authority_admits_once_per_current_media_incarnation() -> None:
    authority = _ReadinessCueAuthority()

    assert authority.admit(3, 7) is True
    assert authority.admit(3, 7) is False
    assert authority.admit(3, 6) is False
    assert authority.admit(2, 8) is False
    assert authority.admit(3, 8) is True
    assert authority.admit(4, 1) is True


@pytest.mark.asyncio
async def test_serialized_playback_never_overlaps_readiness_and_foreground_audio() -> None:
    delegate = _BlockingPlayback()
    playback = _SerializedSpeechPlayback(delegate, asyncio.Lock())
    audio = AudioFrame(pcm=b"\x00\x00", sample_rate_hz=48_000, channels=1)
    first = SpeechChunk(turn_id="ready", chunk_id="ready", text="Ready.", audio=audio)
    second = SpeechChunk(turn_id="turn", chunk_id="answer", text="Answer.", audio=audio)

    first_task = asyncio.create_task(playback.play(first, is_valid=lambda: True))
    await delegate.first_started.wait()
    second_task = asyncio.create_task(playback.play(second, is_valid=lambda: True))
    await asyncio.sleep(0)

    assert delegate.calls == ["ready"]
    delegate.release_first.set()
    await asyncio.gather(first_task, second_task)
    assert delegate.calls == ["ready", "answer"]


@pytest.mark.asyncio
async def test_readiness_cue_tasks_are_cancelled_and_joined_on_close() -> None:
    tasks = _ReadinessCueTasks()
    started = asyncio.Event()
    settled = asyncio.Event()

    async def operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    tasks.create(operation(), name="readiness-test")
    await started.wait()
    await tasks.close()

    assert settled.is_set()


def test_readiness_cue_chunk_has_unique_non_conversational_playback_identity() -> None:
    audio = AudioFrame(pcm=b"\x00\x00", sample_rate_hz=48_000, channels=1)
    template = SpeechChunk(
        turn_id="turn_tts_preflight",
        chunk_id="chunk_tts_preflight",
        text="Ready.",
        audio=audio,
    )

    first = _readiness_cue_chunk(template, generation=3, media_incarnation=7)
    second = _readiness_cue_chunk(template, generation=3, media_incarnation=8)

    assert first.turn_id == "readiness_3_7"
    assert first.chunk_id == "readiness_3_7"
    assert first.text == "Ready."
    assert first.audio is audio
    assert second.chunk_id != first.chunk_id


@pytest.mark.parametrize("generation,incarnation", [(0, 1), (1, 0), (True, 1), (1, False)])
def test_readiness_cue_authority_rejects_invalid_owners(
    generation: int,
    incarnation: int,
) -> None:
    authority = _ReadinessCueAuthority()

    with pytest.raises((TypeError, ValueError)):
        authority.admit(generation, incarnation)


def test_voice_input_ready_data_binds_generation_and_media_incarnation() -> None:
    assert _voice_input_ready_data(7, 4) == {
        "generation": 7,
        "mediaIncarnation": 4,
    }


def test_host_drops_echo_diagnostics_when_projection_is_full() -> None:
    projection = BrowserEventProjection(capacity=1)
    projection.publish("notification_queued", {})

    _project_conversation_observation(projection, "echo_suppressed", {})
    _project_conversation_observation(projection, "echo_barge_in_confirmed", {})

    assert [event.kind for event in projection.events_after(0)] == ["notification_queued"]


@pytest.mark.parametrize(
    "kind,data",
    (
        ("barge_in_non_speech_suppressed", {}),
        ("barge_in_verifier_unavailable", {"reason": "classification_error"}),
    ),
)
def test_host_projects_speech_verifier_advisories(
    kind: str,
    data: dict[str, str | int | bool | None],
) -> None:
    projection = BrowserEventProjection()

    _project_conversation_observation(projection, kind, data)

    assert [event.kind for event in projection.events_after(0)] == [kind]


def test_host_cli_evidence_flags_are_default_off_bounded_and_no_service_exclusive() -> None:
    parser = _build_argument_parser()

    defaults = parser.parse_args([])
    assert defaults.evidence_capture is False
    assert defaults.evidence_retention_hours == 24
    assert defaults.evidence_db is None
    assert defaults.evidence_status is False
    assert defaults.purge_evidence is False

    enabled = parser.parse_args(
        [
            "--evidence-capture",
            "--evidence-retention-hours",
            "168",
            "--evidence-db",
            r"C:\capture-gate\capture-v1.sqlite3",
        ]
    )
    assert enabled.evidence_capture is True
    assert enabled.evidence_retention_hours == 168
    assert enabled.evidence_db == r"C:\capture-gate\capture-v1.sqlite3"

    for retention in ("0", "169"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--evidence-retention-hours", retention])
    with pytest.raises(SystemExit):
        parser.parse_args(["--evidence-status", "--purge-evidence"])


def test_qualification_no_task_seam_is_hidden_and_requires_the_complete_owned_child_contract(
) -> None:
    parser = _build_argument_parser()
    public_help = parser.format_help()

    assert "--qualification-no-hermes-tasks" not in public_help
    assert "--qualification-checkpoint-write-handle" not in public_help
    full = [
        "--qualification-no-hermes-tasks",
        "--qualification-checkpoint-write-handle",
        "42",
        "--qualification-checkpoint-resume-handle",
        "43",
        "--qualification-checkpoint-nonce",
        "a" * 64,
    ]
    for partial in (full[:1], full[:3], full[:5]):
        with pytest.raises(ValueError, match="qualification child"):
            _qualification_checkpoint_contract(parser.parse_args(partial), {})
    with pytest.raises(ValueError, match="qualification child"):
        _qualification_checkpoint_contract(
            parser.parse_args(full),
            {"HERMES_REALTIME_QUALIFICATION_CHILD": "0"},
        )
    aliased = [*full]
    aliased[4] = "42"
    with pytest.raises(ValueError, match="distinct"):
        _qualification_checkpoint_contract(
            parser.parse_args(aliased),
            {"HERMES_REALTIME_QUALIFICATION_CHILD": "1"},
        )

    assert _qualification_checkpoint_contract(
        parser.parse_args(full),
        {"HERMES_REALTIME_QUALIFICATION_CHILD": "1"},
    ) == (42, 43, "a" * 64)


def test_qualification_no_task_close_owner_retains_the_task10_baseline_surface() -> None:
    """10A observation/composition work cannot alter the frozen child close seam."""

    import inspect

    from hermes_realtime.host_launcher import _QualificationNoTaskCloseOwner

    assert str(inspect.signature(_QualificationNoTaskCloseOwner.__init__)) == (
        "(self, *owners: '_AsyncClose') -> 'None'"
    )
    source = inspect.getsource(_QualificationNoTaskCloseOwner)
    assert "production_observation" not in source
    assert "qualification work close failed" in source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("disposition", "capture_state", "expected"),
    (
        (
            evidence_models.RevokeDisposition.PURGE_COMPLETED,
            evidence_models.CaptureState.IDLE,
            {
                "captureState": "idle",
                "result": "purge_completed",
                "sequence": 2,
            },
        ),
        (
            evidence_models.RevokeDisposition.PURGE_FAILED,
            evidence_models.CaptureState.PURGE_FAILED,
            {
                "captureState": "purge_failed",
                "error": "purge_failed",
                "sequence": 2,
            },
        ),
    ),
)
async def test_host_revoke_route_maps_a_finalize_that_wins_before_response(
    disposition: evidence_models.RevokeDisposition,
    capture_state: evidence_models.CaptureState,
    expected: dict[str, object],
) -> None:
    """A request/finalize race must never be reported as a control timeout."""

    projection = BrowserEventProjection(capacity=3)
    binding = BrowserBindingSnapshot(
        participant_identity="browser_0123456789abcdef",
        binding_generation=7,
    )
    request = evidence_models.EvidenceRevokeRequestV1(sequence=2)

    class ImmediateTerminalRuntime:
        def take_lifecycle_status_reservation(self) -> object:
            return projection.reserve_capture_status()

        def reserve_browser_revoke(self, **kwargs: object) -> object:
            assert kwargs["binding_generation"] == binding.binding_generation
            assert kwargs["request"] is request
            return object()

        async def activate_revoke(self, _authority: object, *, timeout_seconds: float) -> object:
            assert timeout_seconds == 2.0
            return disposition

        def capture_status(self, *, disclosure_digest: str) -> object:
            assert len(disclosure_digest) == 64
            return evidence_models.CaptureStatusV1(
                available=True,
                capture_state=capture_state,
                retention_hours=24,
                consent_version=evidence_models.CONSENT_VERSION,
                disclosure_digest=disclosure_digest,
            )

    operation = _reserve_host_evidence_revoke(
        runtime=ImmediateTerminalRuntime(),  # type: ignore[arg-type]
        projection=projection,
        binding=binding,
        request=request,
    )

    response = await operation.complete()

    assert response.payload == expected
    assert response.status == (
        200 if disposition is evidence_models.RevokeDisposition.PURGE_COMPLETED else 503
    )
    [status] = projection.events_after(0)
    assert status.kind == "capture_status"
    assert status.data["captureState"] == capture_state.value


@pytest.mark.asyncio
async def test_qualification_child_never_reads_hermes_credentials_before_no_task_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt

    import hermes_realtime.host_launcher as host_launcher

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    args = _build_argument_parser().parse_args(
        [
            "--qualification-no-hermes-tasks",
            "--qualification-checkpoint-write-handle",
            str(msvcrt.get_osfhandle(checkpoint_write_fd)),
            "--qualification-checkpoint-resume-handle",
            str(msvcrt.get_osfhandle(resume_read_fd)),
            "--qualification-checkpoint-nonce",
            "a" * 64,
        ]
    )
    captured: dict[str, object] = {}

    def forbidden_credential_read(_path: object) -> str:
        raise AssertionError("qualification child must not read API_SERVER_KEY")

    class Launcher:
        async def start(self) -> str:
            raise asyncio.CancelledError

        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(host_launcher, "_load_api_bearer", forbidden_credential_read)
    monkeypatch.setattr(
        host_launcher,
        "_load_livekit_credentials",
        lambda **_kwargs: ("devkey", "local-" + "x" * 32),
    )
    def capture_builder(**kwargs: object) -> Launcher:
        captured.update(kwargs)
        captured["private_no_tasks"] = (
            host_launcher._QUALIFICATION_NO_TASK_COMPOSITION.get()
        )
        return Launcher()

    monkeypatch.setattr(
        host_launcher,
        "build_local_host_launcher",
        capture_builder,
    )
    monkeypatch.setenv("HERMES_REALTIME_QUALIFICATION_CHILD", "1")

    try:
        with pytest.raises(asyncio.CancelledError):
            await _run_host_cli(args)
        checkpoint_write_fd = -1
        resume_read_fd = -1
    finally:
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)

    assert captured["hermes_api_bearer"] is None
    assert "qualification_no_hermes_tasks" not in captured
    assert captured["private_no_tasks"] is True
    assert captured["closed"] is True


def test_evidence_database_path_uses_only_custom_or_localappdata_root() -> None:
    assert _resolve_evidence_database_path(
        r"D:\private-capture\capture-v1.sqlite3",
        {},
    ) == Path(r"D:\private-capture\capture-v1.sqlite3")
    assert _resolve_evidence_database_path(
        None,
        {"LOCALAPPDATA": r"C:\Users\owner\AppData\Local"},
    ) == Path(
        r"C:\Users\owner\AppData\Local\HermesRealtime\evidence\capture-v1.sqlite3"
    )
    for environment in ({}, {"LOCALAPPDATA": ""}):
        with pytest.raises(ValueError, match="LOCALAPPDATA"):
            _resolve_evidence_database_path(None, environment)


def test_emergency_disable_wins_over_operator_capture_enablement() -> None:
    assert _operator_evidence_capture_enabled(False, {}) is False
    assert _operator_evidence_capture_enabled(True, {}) is True
    assert (
        _operator_evidence_capture_enabled(
            True,
            {"HERMES_REALTIME_DISABLE_EVIDENCE": "1"},
        )
        is False
    )
    assert (
        _operator_evidence_capture_enabled(
            True,
            {"HERMES_REALTIME_DISABLE_EVIDENCE": "true"},
        )
        is True
    )
    with pytest.raises(TypeError, match="exact boolean"):
        _operator_evidence_capture_enabled(1, {})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "flag",
    ("--evidence-capture", "--evidence-status", "--purge-evidence"),
)
def test_unsupported_evidence_cli_exits_before_host_composition(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hermes_realtime.host_launcher as host_launcher

    composed = False

    async def reject_composition(args: object) -> None:
        nonlocal composed
        del args
        composed = True
        raise AssertionError("unsupported evidence command composed the host")

    monkeypatch.setattr(host_launcher.sys, "platform", "linux")
    monkeypatch.setattr(host_launcher, "_run_host_cli", reject_composition)
    monkeypatch.setattr(host_launcher.sys, "argv", ["hermes-realtime-host", flag])

    assert main() == 2
    captured = capsys.readouterr()
    assert captured.out == '{"error":"unsupported_platform","version":1}\n'
    assert captured.err == ""
    assert composed is False


@pytest.mark.parametrize(
    ("flag", "exit_code", "document"),
    (
        (
            "--evidence-status",
            0,
            '{"ownerState":"absent","result":"absent","version":1}',
        ),
        ("--purge-evidence", 3, '{"error":"ownership_unavailable","version":1}'),
    ),
)
def test_no_service_evidence_commands_dispatch_before_host_composition(
    flag: str,
    exit_code: int,
    document: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hermes_realtime.host_launcher as host_launcher

    composed = False
    dispatched: list[object] = []

    def no_service(args: object) -> tuple[int, str]:
        dispatched.append(args)
        return exit_code, document

    async def reject_composition(args: object) -> None:
        nonlocal composed
        del args
        composed = True
        raise AssertionError("no-service command composed the host")

    monkeypatch.setattr(host_launcher.sys, "platform", "win32")
    monkeypatch.setattr(host_launcher, "_run_evidence_no_service", no_service)
    monkeypatch.setattr(host_launcher, "_run_host_cli", reject_composition)
    monkeypatch.setattr(
        host_launcher,
        "_parse_kokoro_pronunciations",
        lambda _values: (_ for _ in ()).throw(
            AssertionError("no-service command ran host-only validation")
        ),
    )
    monkeypatch.setattr(
        host_launcher.sys,
        "argv",
        ["hermes-realtime-host", flag],
    )

    assert main() == exit_code
    captured = capsys.readouterr()
    assert captured.out == f"{document}\n"
    assert captured.err == ""
    assert len(dispatched) == 1
    assert composed is False


@pytest.mark.parametrize(
    ("disposition", "expected"),
    (
        (
            "absent",
            (
                0,
                '{"ownerState":"absent","result":"absent","version":1}',
            ),
        ),
        (
            "recovered",
            (
                0,
                '{"ownerState":"recovery_only","result":"recovered","version":1}',
            ),
        ),
        (
            "purge_completed",
            (
                0,
                '{"ownerState":"recovery_only","result":"purge_completed","version":1}',
            ),
        ),
        (
            "faulted",
            (
                0,
                '{"ownerState":"faulted","result":"faulted","version":1}',
            ),
        ),
        (
            "ownership_unavailable",
            (3, '{"error":"ownership_unavailable","version":1}'),
        ),
    ),
)
def test_no_service_status_maps_closed_dispositions_without_minting_ids(
    disposition: str,
    expected: tuple[int, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_realtime.host_launcher as host_launcher
    from hermes_realtime.evidence import RecoveryDisposition

    closed = False

    class StatusDaemon:
        def __init__(self, spool_factory: object) -> None:
            assert callable(spool_factory)

        def recover_existing_and_close(self) -> RecoveryDisposition:
            nonlocal closed
            closed = True
            return RecoveryDisposition(disposition)

    monkeypatch.setattr(host_launcher, "SQLiteEvidenceWriterDaemonV1", StatusDaemon)
    monkeypatch.setattr(
        host_launcher,
        "SQLiteEvidenceSpool",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CLI thread constructed an evidence spool")
        ),
    )
    monkeypatch.setattr(
        host_launcher.uuid,
        "uuid4",
        lambda: (_ for _ in ()).throw(AssertionError("status minted an identifier")),
    )
    args = _build_argument_parser().parse_args(
        ["--evidence-status", "--evidence-db", r"C:\status\capture-v1.sqlite3"]
    )

    assert host_launcher._run_evidence_no_service(args) == expected
    assert closed is True


@pytest.mark.parametrize(
    ("disposition", "expected"),
    (
        ("purge_completed", (0, '{"result":"purge_completed","version":1}')),
        ("already_absent", (0, '{"result":"already_absent","version":1}')),
        (
            "ownership_unavailable",
            (3, '{"error":"ownership_unavailable","version":1}'),
        ),
        ("purge_failed", (4, '{"error":"purge_failed","version":1}')),
    ),
)
def test_no_service_purge_mints_one_closed_authority_and_maps_disposition(
    disposition: str,
    expected: tuple[int, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_realtime.host_launcher as host_launcher
    from hermes_realtime.evidence import FullPurgeV1, PurgeDisposition, SentinelState

    commands: list[FullPurgeV1] = []
    closed = False
    minted = 0

    class PurgeDaemon:
        def __init__(self, spool_factory: object) -> None:
            assert callable(spool_factory)

        def purge_full_store_and_close(self, command: FullPurgeV1) -> PurgeDisposition:
            nonlocal closed
            commands.append(command)
            closed = True
            return PurgeDisposition(disposition)

    def uuid4() -> UUID:
        nonlocal minted
        minted += 1
        return UUID("00000000-0000-4000-8000-000000000901")

    monkeypatch.setattr(host_launcher, "SQLiteEvidenceWriterDaemonV1", PurgeDaemon)
    monkeypatch.setattr(
        host_launcher,
        "SQLiteEvidenceSpool",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CLI thread constructed an evidence spool")
        ),
    )
    monkeypatch.setattr(host_launcher.uuid, "uuid4", uuid4)
    args = _build_argument_parser().parse_args(
        ["--purge-evidence", "--evidence-db", r"C:\purge\capture-v1.sqlite3"]
    )

    assert host_launcher._run_evidence_no_service(args) == expected
    assert minted == 1
    assert closed is True
    assert len(commands) == 1
    assert commands[0] == FullPurgeV1(
        protocol_version=1,
        full_purge_generation_id="00000000-0000-4000-8000-000000000901",
        sentinel_state=SentinelState.FULL_PURGE_PENDING,
        artifact_manifest_version=1,
    )


def test_host_cli_preserves_default_and_accepts_diagnostic_bootstrap_ttl() -> None:
    parser = _build_argument_parser()

    assert parser.parse_args([]).bootstrap_ttl_seconds == 120
    assert parser.parse_args(["--bootstrap-ttl-seconds", "3600"]).bootstrap_ttl_seconds == 3_600
    assert parser.parse_args([]).tailnet_launch is False
    assert parser.parse_args(["--remote", "--tailnet-launch"]).tailnet_launch is True
    assert parser.parse_args([]).persistent_loopback_launch is False
    assert parser.parse_args(["--persistent-loopback-launch"]).persistent_loopback_launch is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--tailnet-launch", "--persistent-loopback-launch"])


@pytest.mark.asyncio
async def test_host_cli_rejects_persistent_loopback_remote_mode_before_side_effects() -> None:
    args = _build_argument_parser().parse_args(["--remote", "--persistent-loopback-launch"])

    with pytest.raises(ValueError, match="rejects --remote"):
        await _run_host_cli(args)


def test_host_cli_natural_work_tools_are_explicit_and_reversibly_disabled() -> None:
    parser = _build_argument_parser()

    assert parser.parse_args([]).natural_work_tools is False
    assert parser.parse_args(["--natural-work-tools"]).natural_work_tools is True
    assert parser.parse_args(["--no-natural-work-tools"]).natural_work_tools is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--natural-work-tools", "--no-natural-work-tools"])


def test_host_cli_natural_duplex_profile_preserves_provider_sentence_units() -> None:
    parser = _build_argument_parser()

    assert parser.parse_args([]).conversation_profile == "legacy"
    assert _conversation_profile_duration_limit("legacy") is None
    assert parser.parse_args(["--conversation-profile", "natural_v1"]).conversation_profile == (
        "natural_v1"
    )
    assert _conversation_profile_duration_limit("natural_v1") is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--conversation-profile", "natural_v2"])


def test_natural_work_binding_requires_exact_codex_inference_type() -> None:
    surface = _work_surface()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )

    _bind_natural_work_tools(
        inference=inference,
        surface=surface,
        enabled=False,
    )
    assert inference._work_tool_handler is None

    class CodexSubclass(CodexAppServerStreamingInference):
        pass

    accidental = CodexSubclass(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(TypeError, match="exact Codex"):
        _bind_natural_work_tools(
            inference=accidental,
            surface=surface,
            enabled=True,
        )
    assert accidental._work_tool_handler is None


@pytest.mark.asyncio
async def test_full_host_preflight_binds_once_only_after_tool_less_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    surface = _work_surface()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )
    original_bind = CodexAppServerStreamingInference.bind_work_tools

    class SessionProbe:
        async def start(self) -> None:
            events.append("session:start")

    class SynthesizerProbe:
        async def synthesize(self, text: str, turn_id: str):  # type: ignore[no-untyped-def]
            assert text == "Ready."
            assert turn_id == "turn_tts_preflight"
            events.append("tts:probe")
            yield SpeechChunk(
                turn_id=turn_id,
                chunk_id="chunk_preflight",
                text=text,
                audio=AudioFrame(
                    pcm=b"\x00\x00",
                    sample_rate_hz=24_000,
                    channels=1,
                ),
            )

    async def stream_without_tools(  # type: ignore[no-untyped-def]
        self: CodexAppServerStreamingInference,
        request: object,
        *,
        turn_id: str,
    ):
        del request
        assert self is inference
        assert turn_id == "turn_preflight"
        assert self._work_tool_handler is None
        events.append("inference:preflight")
        yield "Ready."

    def record_bind(
        self: CodexAppServerStreamingInference,
        handler: object,
    ) -> None:
        events.append("inference:bind")
        original_bind(self, handler)  # type: ignore[arg-type]

    monkeypatch.setattr(
        CodexAppServerStreamingInference,
        "stream",
        stream_without_tools,
    )
    monkeypatch.setattr(
        CodexAppServerStreamingInference,
        "bind_work_tools",
        record_bind,
    )

    cue = await _run_full_host_preflight(
        task_session=SessionProbe(),
        inference=inference,
        synthesizer=SynthesizerProbe(),
        surface=surface,
        natural_work_tools=True,
    )

    assert cue.chunk_id == "chunk_preflight"
    assert inference._work_tool_handler is surface
    assert events == [
        "session:start",
        "inference:preflight",
        "tts:probe",
        "inference:bind",
    ]


@pytest.mark.asyncio
async def test_full_host_preflight_failure_never_binds_work_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _work_surface()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )

    class SessionProbe:
        async def start(self) -> None:
            return None

    class SynthesizerProbe:
        async def synthesize(self, text: str, turn_id: str):  # type: ignore[no-untyped-def]
            del text, turn_id
            if False:
                yield

    async def ready_stream(  # type: ignore[no-untyped-def]
        self: CodexAppServerStreamingInference,
        request: object,
        *,
        turn_id: str,
    ):
        del self, request, turn_id
        yield "Ready."

    monkeypatch.setattr(CodexAppServerStreamingInference, "stream", ready_stream)

    with pytest.raises(RuntimeError, match="TTS preflight"):
        await _run_full_host_preflight(
            task_session=SessionProbe(),
            inference=inference,
            synthesizer=SynthesizerProbe(),
            surface=surface,
            natural_work_tools=True,
        )

    assert inference._work_tool_handler is None


@pytest.mark.asyncio
async def test_full_host_work_close_order_is_retry_safe() -> None:
    events: list[str] = []

    class CloseProbe:
        def __init__(self, name: str, *, fail_once: bool = False) -> None:
            self.name = name
            self.fail_once = fail_once
            self.calls = 0

        async def close(self) -> None:
            self.calls += 1
            events.append(f"{self.name}:{self.calls}")
            if self.fail_once and self.calls == 1:
                raise RuntimeError(f"transient {self.name} close failure")

    inference = CloseProbe("inference", fail_once=True)
    surface = CloseProbe("surface")
    controller = CloseProbe("controller")
    session = CloseProbe("session", fail_once=True)
    owner = _FullHostWorkCloseOwner(
        inference=inference,
        surface=surface,
        task_controller=controller,
        task_session=session,
    )

    with pytest.raises(BaseExceptionGroup, match="full-host work close") as first_error:
        await owner.close()
    assert {str(error) for error in first_error.value.exceptions} == {
        "transient inference close failure",
        "transient session close failure",
    }
    assert events == ["inference:1", "surface:1", "controller:1", "session:1"]

    await owner.close()
    await owner.close()
    assert events == [
        "inference:1",
        "surface:1",
        "controller:1",
        "session:1",
        "inference:2",
        "session:2",
    ]


def test_tailnet_identity_env_is_strict_and_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HERMES_REALTIME_TAILNET_NODE_STABLE_ID", raising=False)
    assert _load_tailnet_authorizer(enabled=False) is None
    with pytest.raises(RuntimeError, match="configured node identity"):
        _load_tailnet_authorizer(enabled=True)

    for invalid in ("", " ", "x" * 129, "not allowed"):
        monkeypatch.setenv("HERMES_REALTIME_TAILNET_NODE_STABLE_ID", invalid)
        with pytest.raises(RuntimeError, match="identity is invalid"):
            _load_tailnet_authorizer(enabled=True)

    monkeypatch.setenv("HERMES_REALTIME_TAILNET_NODE_STABLE_ID", "synthetic-node-id")

    async def resolve(_peer: object) -> bytes:
        return b'{"Node":{"StableID":"synthetic-node-id"}}'

    monkeypatch.setattr(
        "hermes_realtime.host_launcher.TailscaleCliWhoIsResolver",
        lambda: resolve,
    )
    assert _load_tailnet_authorizer(enabled=True) is not None


def test_host_projects_truthful_inference_configuration() -> None:
    codex = _build_browser_model_configuration(
        inference_provider="codex",
        ollama_model="unused",
        codex_model="gpt-5.6-terra",
        codex_effort="medium",
    )
    assert codex.provider == "openai-codex"
    assert codex.authentication == "subscription"
    assert codex.transport == "subscription-app-server"
    assert codex.model == "gpt-5.6-terra"
    assert codex.effort == "medium"
    assert codex.context_window_tokens is None
    assert codex.reports_token_usage is True

    ollama = _build_browser_model_configuration(
        inference_provider="ollama",
        ollama_model="hermes-local:latest",
        codex_model="unused",
        codex_effort="high",
    )
    assert ollama.provider == "ollama"
    assert ollama.authentication == "local"
    assert ollama.transport == "local-http"
    assert ollama.model == "hermes-local:latest"
    assert ollama.effort is None


def test_host_adapts_authoritative_codex_catalog_for_browser() -> None:
    catalog = _build_browser_selectable_model_catalog(
        CodexModelConfiguration(
            models=(
                CodexModelOption(
                    model="gpt-5.6-sol",
                    display_name="GPT-5.6 Sol",
                    description="Deep coding model",
                    supported_efforts=("medium", "high", "xhigh"),
                    default_effort="high",
                ),
            ),
            selected_model="gpt-5.6-sol",
            selected_effort="xhigh",
        )
    )

    assert catalog.selected_model == "gpt-5.6-sol"
    assert catalog.selected_effort == "xhigh"
    assert catalog.models[0].display_name == "GPT-5.6 Sol"
    assert catalog.models[0].supported_efforts == ("medium", "high", "xhigh")
