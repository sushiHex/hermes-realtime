"""The latches of the ``hermes_realtime`` package, as a ratchet.

A latch is state that, once set, makes its holder refuse for the rest of the holder's
life. Proposed ADR 0008 ("a latch ends its holder") starts from an inventory of them: this
is that inventory, bound to the source by equality. The test fails on a latch that is not
listed and on a listing that no longer matches the source, so a latch cannot be added,
reset, re-tripped from a new place or removed without this table changing in the same diff.
It changes no behaviour and states only what holds today.

**How a latch is recognised.** Structurally, never by name. Inside one class, an attribute
``self.x`` is a latch when all four hold:

1. ``__init__`` (or a class-level field) starts it clear: the literal ``None`` or ``False``;
2. a method other than ``__init__`` assigns it something that is not that literal;
3. a method raises because it is set: ``raise self.x``, or an ``if`` whose test is true
   when ``x`` is set (``self.x``, ``self.x is not None``, alone or inside ``and``/``or``)
   and whose body raises;
4. no method other than ``__init__`` ever assigns it ``None`` or ``False`` again.

Rule 4 is what makes it a latch and not a busy flag. An attribute that meets rules 1 to 3
and is cleared somewhere is measured too, but need not be listed. It may be listed with its
clearing methods, and is then bound by equality as well; the update executor's error, which
is cleared only as ``close`` reports it, is listed that way.

``KIND`` is the reviewed part of each row. The scan cannot tell a failure latch from a
close flag or a start-once guard, so a new row must be classified by a person, and that
classification is the review point.

**What the scan cannot see.** These are listed by hand in ``UNSEEN`` where they are known,
and are checked only to exist:

- state that is not ``None``/``False`` when clear: an enum, a string, a set of tasks;
- a guard that returns, sleeps or delegates to a helper call instead of raising in an ``if``;
- an attribute started clear in a base class, by a non-literal, or outside ``__init__``;
- a write through any receiver other than ``self`` (``binding._terminal_error = ...``), and
  ``setattr``;
- a trip inside a function nested in ``__init__``, such as a transport callback;
- state held in a closure, a module global or a local of a composition function;
- a latch with no attribute at all: a task that simply returns and is never restarted (the
  memory receiver, a dead Codex reader);
- a holder that is reset by being replaced. Whether tripping a latch ends its holder is a
  claim in the table, read from source, not something this test measures.

A method name stands for every function defined inside that method. Two methods of the same
name in one class collapse into one.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "hermes_realtime"
SOURCE = Path(__file__).resolve().parents[1] / "src" / PACKAGE

# What trips the latch. Only FAILURE rows are failure-domain latches in ADR 0008's sense.
FAILURE = "failure"  # an error or refusal the holder did not ask for
LIFECYCLE = "lifecycle"  # only the holder's own close or stop
ONE_SHOT = "one-shot"  # the first ordinary use: start once, consume once, bind once
KINDS = frozenset({FAILURE, LIFECYCLE, ONE_SHOT})

# How long the holder lives. UNTRACED means the constructor was not followed to its owner.
BINDING, HOST, HERMES, QUALIFICATION, UNTRACED = (
    "binding", "host", "hermes process", "qualification run", "untraced",
)

Key = tuple[str, str, str]  # (file, class, attribute)


@dataclass(frozen=True)
class Latch:
    kind: str
    set_in: tuple[str, ...]
    cleared_in: tuple[str, ...] = ()
    lifetime: str = ""
    trips_on: str = ""
    # Whether tripping it ends its holder today. None: not traced.
    ends_holder: bool | None = None


def _failure(
    lifetime: str, trips_on: str, ends_holder: bool | None, *set_in: str,
    cleared_in: tuple[str, ...] = (),
) -> Latch:
    return Latch(FAILURE, set_in, cleared_in, lifetime, trips_on, ends_holder)


def _lifecycle(*set_in: str) -> Latch:
    return Latch(LIFECYCLE, set_in)


def _one_shot(*set_in: str) -> Latch:
    return Latch(ONE_SHOT, set_in)


LATCHES: dict[Key, Latch] = {
    # --- failure latches whose holder lives as long as the host ---------------------------
    ("conversation/tasks.py", "ConversationTaskController", "_terminal_error"): _failure(
        HOST, "any dispatch exception; a contradictory acknowledgment; a failed terminal loop",
        False, "_fail_locked",
    ),
    ("conversation/updates.py", "ConversationUpdateDirector", "_error"): _failure(
        HOST, "the controller's latched error; a policy or publication failure",
        False, "_consume",
    ),
    ("conversation/actions.py", "ConversationUpdateExecutor", "_terminal_error"): _failure(
        HOST, "the director's latched error; a replaced announcement that did not settle",
        False, "_consume_decisions", cleared_in=("_close_owned",),
    ),
    ("providers/codex_app_server.py", "CodexAppServerStreamingInference", "_closed"): _failure(
        HOST, "a failed or late send, including to a dead child; also its own close",
        False, "_begin_close",
    ),
    ("providers/ollama_session.py", "OllamaSessionSelection", "_closed"): _failure(
        HOST, "a failed re-selection or adapter cleanup; also its own close",
        False, "_close_adapter", "_prepare", "close",
    ),
    ("providers/kokoro.py", "KokoroSynthesizer", "_managed_close_error"): _failure(
        HOST, "a managed speech worker whose close failed", False, "close",
    ),
    ("client/session.py", "BrowserSessionDirector", "_rebind_finalization_failed"): _failure(
        HOST, "a rebind rollback that timed out and then failed to settle",
        False, "_finalize_rebind_rollback",
    ),
    ("evidence/lifecycle.py", "EvidenceLifecycleOwner", "_faulted"): _failure(
        HOST, "a repeated generated identifier", None, "_allocate_uuid_locked",
    ),
    # --- failure latches with a shorter or different holder --------------------------------
    ("conversation/worker.py", "ConversationSessionWorker", "_terminal_error"): _failure(
        BINDING, "any audio-path error; any response error but the exact model refusal; "
        "the third consecutive refusal", True, "_latch_terminal_error", "_response_done",
    ),
    ("conversation/ingress.py", "BoundedPcmIngress", "_fault"): _failure(
        BINDING, "an out-of-order frame, exhausted capacity, or a queued frame too old",
        True, "_fault_if_oldest_expired", "admit",
    ),
    ("livekit/adapter.py", "LiveKitRoomPeer", "_closed"): _failure(
        "one peer: a binding's, or the host's verifier",
        "a failed connect or a transport disconnect; also its own disconnect",
        None, "connect", "disconnect",
    ),
    ("companion/archive.py", "VoiceArchive", "_fenced"): _failure(
        HERMES, "a quarantine that could not be saved", False, "_quarantine",
    ),
    ("companion/hermes_compat.py", "HermesArchivePort", "_parent_creation_failed"): _failure(
        HERMES, "a review parent that failed after it was created",
        False, "_make_review_parent_scoped",
    ),
    ("integration/plugin.py", "HermesPluginDispatcher", "_uncertain"): _failure(
        HERMES, "a dispatch that raised or returned an unusable result", None, "dispatch",
    ),
    ("integration/worker.py", "RealtimeHermesSession", "_terminal_error"): _failure(
        UNTRACED, "a failed bridge receiver", None, "_set_terminal_error",
    ),
    ("evidence/admission.py", "ConversationOperationScheduler", "_closed"): _failure(
        UNTRACED, "an exhausted reservation serial; also its own close",
        None, "_mint_locked", "close",
    ),
    ("_qualification.py", "_QualificationCheckpointChannelV1", "_failed"): _failure(
        QUALIFICATION, "a checkpoint out of order, or a failed write or acknowledgment",
        True, "_fail",
    ),
    ("_qualification.py", "_QualificationCapacityProbeV1", "_incomplete"): _failure(
        QUALIFICATION, "a contended or full observation record", None, "record",
    ),
    # --- one-shot guards --------------------------------------------------------------------
    ("_qualification.py", "_QualificationCapacityProbeV1",
     "_rollover_failure_checkpoint_consumed"): _one_shot("pause_after_rollover_rejection"),
    ("_qualification.py", "_QualificationCapacityProbeV1",
     "_rollover_publication_fault_consumed"): _one_shot("consume_rollover_publication_fault"),
    ("_qualification.py", "_QualificationFullHostDependenciesV1", "_consumed"): _one_shot("take"),
    ("client/bootstrap.py", "OneTimeBootstrapCapability", "_consumed"): _one_shot("_claim"),
    ("companion/host.py", "VoiceCompanionHost", "_thread"): _one_shot("start"),
    ("companion/review.py", "VoiceReviewCoordinator", "_interval"): _one_shot("start"),
    ("evidence/admission.py", "_OrdinarySchedulerBindingV1", "_claimed"): _one_shot("claim"),
    ("evidence/admission.py", "EvidenceAdmissionControllerV1", "_drain_authority"): _one_shot(
        "request_drain"
    ),
    ("evidence/admission.py", "EvidenceAdmissionControllerV1", "_rollover_claimed"): _one_shot(
        "_install_rollover_runtime_owner"
    ),
    ("evidence/runtime.py", "HostEvidenceRuntimeV1", "_retention_cancel"): _one_shot(
        "configure_retention_owner"
    ),
    ("integration/voice_tail.py", "VoiceTailWriter", "_opened"): _one_shot("open"),
    # A failed start leaves it set too; the process then exits.
    ("launcher.py", "LocalBrowserLauncher", "_start_attempted"): _one_shot("start"),
    ("livekit/adapter.py", "LiveKitRoomPeer", "_bound_remote_identity"): _one_shot(
        "bind_remote_identity"
    ),
    ("livekit/adapter.py", "LiveKitRoomPeer", "_connection_attempted"): _one_shot("connect"),
    ("providers/codex_app_server.py", "CodexAppServerStreamingInference",
     "_user_turn_admitted"): _one_shot("stream"),
    ("providers/codex_app_server.py", "CodexAppServerStreamingInference",
     "_work_tool_handler"): _one_shot("bind_work_tools"),
    ("providers/ollama.py", "OllamaStreamingInference", "_context_configuration"): _one_shot(
        "_configure_context"
    ),
    ("providers/ollama.py", "OllamaStreamingInference", "_inference_started"): _one_shot("stream"),
    # --- lifecycle flags: set only by the holder's own close or stop ------------------------
    ("_qualification.py", "_QualificationCheckpointChannelV1", "_closed"): _lifecycle("close"),
    ("companion/host.py", "VoiceCompanionService", "_memory_closed"): _lifecycle("close"),
    ("companion/review.py", "VoiceReviewCoordinator", "_closed"): _lifecycle("close"),
    ("conversation/actions.py", "ConversationUpdateExecutor", "_closed"): _lifecycle(
        "_close_owned"
    ),
    ("conversation/foreground.py", "ForegroundTurnCoordinator", "_closed"): _lifecycle("close"),
    ("conversation/ingress.py", "BoundedPcmIngress", "_closed"): _lifecycle("close"),
    ("conversation/knowledge.py", "KnowledgePrefetchCoordinator", "_closed"): _lifecycle("close"),
    ("conversation/streaming.py", "StreamingSpeechLoop", "_closed"): _lifecycle("_close_owned"),
    ("conversation/tasks.py", "ConversationTaskController", "_closed"): _lifecycle(
        "_close_owned"
    ),
    ("conversation/updates.py", "ConversationUpdateDirector", "_closed"): _lifecycle(
        "_close_owned"
    ),
    ("conversation/worker.py", "ConversationSessionWorker", "_closed"): _lifecycle(
        "_close_binding_owned"
    ),
    ("conversation/worker.py", "ReconnectSafeConversationWorker", "_closed"): _lifecycle(
        "_close_owned"
    ),
    ("evidence/admission.py", "BoundedEvidenceWriterQueueV1", "_consumer_stopped"): _lifecycle(
        "stop_consumer"
    ),
    ("evidence/runtime.py", "HostEvidenceRuntimeV1", "_closed"): _lifecycle(
        "_close_owned_stages"
    ),
    ("evidence/runtime.py", "HostEvidenceRuntimeV1", "_closing"): _lifecycle("close"),
    ("evidence/runtime.py", "HostEvidenceRuntimeV1", "_rollover_handoffs_closed"): _lifecycle(
        "close"
    ),
    ("integration/api.py", "HermesApiTaskSession", "_closed"): _lifecycle("_close_owned"),
    ("integration/bridge.py", "LocalHermesBridgeServer", "_closed_permanently"): _lifecycle(
        "_close_locked"
    ),
    ("integration/native_web_evidence.py", "NativeWebEvidence", "_closed"): _lifecycle("close"),
    ("integration/worker.py", "RealtimeHermesSession", "_closed"): _lifecycle("_close_owned"),
    ("launcher.py", "LocalBrowserLauncher", "_closed"): _lifecycle("_close_owned"),
    ("livekit/worker.py", "LiveKitConversationWorker", "_closed"): _lifecycle("_close_owned"),
    ("providers/codex_app_server.py", "SubprocessCodexJsonLineTransport", "_closed"): _lifecycle(
        "close"
    ),
    ("providers/current_facts.py", "DdgsCurrentFactLookup", "_closed"): _lifecycle("close"),
    ("providers/edge_tts.py", "EdgeTtsSynthesizer", "_closed"): _lifecycle("close"),
    ("providers/kokoro.py", "KokoroSynthesizer", "_closed"): _lifecycle("close"),
    ("providers/kokoro_worker.py", "KokoroWorkerClient", "_closed"): _lifecycle("close"),
    ("providers/moonshine.py", "MoonshineStreamingTranscriber", "_closed"): _lifecycle("close"),
    ("providers/ollama.py", "OllamaStreamingInference", "_closed"): _lifecycle("_close_owned"),
}

# Latches the scan cannot recognise, with the reason. Checked only to exist as an attribute
# of the named class, and not to be measured: a row the scan starts to see moves to LATCHES.
UNSEEN: dict[Key, tuple[str, Latch]] = {
    ("conversation/work_tools.py", "ConversationWorkControlSurface", "_health"): (
        "enum-valued; its guards compare against the open member",
        _failure(HOST, "any dispatch exception or unsettled work-control operation", False),
    ),
    ("providers/kokoro_mcp.py", "KokoroMcpWorkerFactory", "_launch_attempted"): (
        "started clear by a chained assignment; guarded through a helper call",
        Latch(ONE_SHOT, (), (), HOST,
              "the first launch, so a worker that dies cannot be launched again", False),
    ),
    ("integration/voice_archive.py", "VoiceArchiveSender", "_fence"): (
        "its guard returns from the sender loop instead of raising",
        _failure(HOST, "an archive refusal outside the transient set", False),
    ),
    ("integration/voice_review.py", "VoiceReviewSender", "_blocked"): (
        "its guard sleeps instead of raising",
        _failure(HOST, "a review that was not acknowledged, or a non-transient refusal", False),
    ),
    ("client/projection.py", "BrowserEventProjection", "_overflowed"): (
        "its guard returns a boolean; cleared by reset() when a browser session starts",
        _failure("one browser session", "a publication with no room left", False),
    ),
    ("evidence/admission.py", "EvidenceAdmissionControllerV1", "_sticky_fault"): (
        "no guard raises on it directly; it is read through the owner state",
        _failure(HOST, "a writer fault", None),
    ),
    ("evidence/sqlite_spool.py", "SQLiteEvidenceSpool", "_sticky_fault"): (
        "cleared by a full purge; read through the owner state",
        _failure(HOST, "a store fault", None),
    ),
    ("evidence/sqlite_spool.py", "SQLiteEvidenceWriterDaemonV1", "_sticky_fault"): (
        "no guard raises on it directly",
        _failure(HOST, "a corrupt store", None),
    ),
    ("integration/voice_deletes.py", "VoiceDeletes", "lost"): (
        "a field of a saved record, not instance state; it outlives the process",
        _failure("the saved conversation", "a deletion record that could not be read", False),
    ),
}


@dataclass(frozen=True)
class _Measured:
    set_in: tuple[str, ...]
    cleared_in: tuple[str, ...]


def _self_attribute(node: ast.AST) -> str | None:
    if type(node) is ast.Attribute and type(node.value) is ast.Name and node.value.id == "self":
        return node.attr
    return None


def _is_clear(node: ast.AST | None) -> bool:
    return type(node) is ast.Constant and (node.value is None or node.value is False)


def _writes(function: ast.AST) -> Iterator[tuple[str, ast.AST | None]]:
    """Yield (attribute, assigned value) for each write to ``self``; None when not literal."""
    for node in ast.walk(function):
        if type(node) is ast.Assign:
            for target in node.targets:
                if type(target) in (ast.Tuple, ast.List):
                    paired = (
                        type(node.value) in (ast.Tuple, ast.List)
                        and len(node.value.elts) == len(target.elts)
                    )
                    for index, element in enumerate(target.elts):
                        name = _self_attribute(element)
                        if name is not None:
                            yield name, node.value.elts[index] if paired else None
                else:
                    name = _self_attribute(target)
                    if name is not None:
                        yield name, node.value
        elif type(node) is ast.AnnAssign and node.value is not None:
            name = _self_attribute(node.target)
            if name is not None:
                yield name, node.value
        elif type(node) in (ast.AugAssign, ast.NamedExpr):
            name = _self_attribute(node.target)
            if name is not None:
                yield name, None


def _true_when_set(test: ast.expr) -> set[str]:
    """The attributes whose being set can make ``test`` true."""
    name = _self_attribute(test)
    if name is not None:
        return {name}
    if (
        type(test) is ast.Compare
        and len(test.ops) == 1
        and type(test.ops[0]) is ast.IsNot
        and _is_clear(test.comparators[0])
    ):
        name = _self_attribute(test.left)
        return set() if name is None else {name}
    if type(test) is ast.BoolOp:
        return set().union(*(_true_when_set(value) for value in test.values))
    return set()


def _guarded(function: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(function):
        if type(node) is ast.If and any(type(item) is ast.Raise for item in node.body):
            found |= _true_when_set(node.test)
        elif type(node) is ast.Raise and node.exc is not None:
            name = _self_attribute(node.exc)
            if name is not None:
                found.add(name)
    return found


def _scan(text: str) -> dict[tuple[str, str], _Measured]:
    """Measure rules 1 to 3 for every class in ``text``: (class, attribute) -> writes."""
    measured: dict[tuple[str, str], _Measured] = {}
    for cls in ast.walk(ast.parse(text)):
        if type(cls) is not ast.ClassDef:
            continue
        clear_at_start: set[str] = set()
        set_in: dict[str, set[str]] = {}
        cleared_in: dict[str, set[str]] = {}
        guarded: set[str] = set()
        for item in cls.body:
            if (
                type(item) is ast.AnnAssign
                and type(item.target) is ast.Name
                and _is_clear(item.value)
            ):
                clear_at_start.add(item.target.id)
            if type(item) not in (ast.FunctionDef, ast.AsyncFunctionDef):
                continue
            if item.name == "__init__":
                clear_at_start |= {name for name, value in _writes(item) if _is_clear(value)}
                continue
            for name, value in _writes(item):
                (cleared_in if _is_clear(value) else set_in).setdefault(name, set()).add(item.name)
            guarded |= _guarded(item)
        for name in clear_at_start & set_in.keys() & guarded:
            measured[(cls.name, name)] = _Measured(
                tuple(sorted(set_in[name])), tuple(sorted(cleared_in.get(name, ())))
            )
    return measured


def _package() -> dict[Key, _Measured]:
    measured: dict[Key, _Measured] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        file = path.relative_to(SOURCE).as_posix()
        for (cls, name), row in _scan(path.read_text(encoding="utf-8")).items():
            assert (file, cls, name) not in measured, f"{file} defines class {cls} twice"
            measured[(file, cls, name)] = row
    return measured


def _assigned(file: str, cls_name: str, attribute: str) -> bool:
    """Whether the class assigns ``self.attribute`` anywhere or declares it as a field."""
    for cls in ast.walk(ast.parse((SOURCE / file).read_text(encoding="utf-8"))):
        if type(cls) is not ast.ClassDef or cls.name != cls_name:
            continue
        for node in ast.walk(cls):
            targets: list[ast.expr] = []
            if type(node) is ast.Assign:
                targets = list(node.targets)
            elif type(node) is ast.AnnAssign:
                targets = [node.target]
            for target in targets:
                for element in getattr(target, "elts", [target]):
                    if _self_attribute(element) == attribute:
                        return True
                    if type(element) is ast.Name and element.id == attribute and node in cls.body:
                        return True
    return False


def test_every_latch_is_listed_and_every_listing_is_a_latch() -> None:
    measured = _package()
    sticky = {key for key, row in measured.items() if not row.cleared_in}
    listed_sticky = {key for key, row in LATCHES.items() if not row.cleared_in}

    new = sorted(sticky - listed_sticky)
    stale = sorted(listed_sticky - sticky)
    assert not new, f"latches that are not listed; classify and list them: {new}"
    assert not stale, f"listed latches the source no longer has, or now clears: {stale}"

    for key, row in LATCHES.items():
        assert key in measured, f"{key} is listed with a clearing method but is not measured"
        assert (row.set_in, row.cleared_in) == (measured[key].set_in, measured[key].cleared_in), (
            f"{key} is set in {measured[key].set_in} and cleared in "
            f"{measured[key].cleared_in}; the listing says {row.set_in} and {row.cleared_in}"
        )


def test_every_listing_is_classified() -> None:
    for key, row in [*LATCHES.items(), *((key, row) for key, (_, row) in UNSEEN.items())]:
        assert row.kind in KINDS, key
        assert row.ends_holder is None or type(row.ends_holder) is bool, key
        if row.kind == FAILURE:
            assert row.lifetime and row.trips_on, f"{key} needs its holder's lifetime and trip"
    assert all(type(reason) is str and reason for reason, _ in UNSEEN.values())


def test_hand_listed_latches_exist_and_are_not_ones_the_scan_sees() -> None:
    measured = _package()
    for key in UNSEEN:
        assert key not in LATCHES, f"{key} is listed twice"
        assert _assigned(*key), f"{key} no longer exists; delete its hand-written row"
        seen = key in measured and not measured[key].cleared_in
        assert not seen, f"the scan now sees {key}; move it to LATCHES"


_LATCH = """
class Holder:
    def __init__(self):
        self._error = None
        self._busy = False
        self._handle = None
        self._count = 0

    def fail(self, error):
        self._error = error

    def use(self):
        if self._error is not None:
            raise self._error
        if self._handle is None:
            raise RuntimeError("not started")
        if self._busy:
            raise RuntimeError("busy")
        self._busy = True
        self._handle = object()
        self._count = 1
        self._busy = False
"""


def test_the_scan_separates_a_latch_from_a_busy_flag_and_an_unset_handle() -> None:
    measured = _scan(_LATCH)
    # The error is sticky. The busy flag is measured with its clearing method. The handle
    # raises when unset, which is the opposite polarity; the count never starts clear.
    assert measured == {
        ("Holder", "_error"): _Measured(("fail",), ()),
        ("Holder", "_busy"): _Measured(("use",), ("use",)),
    }


def test_the_scan_sees_a_reset_a_second_trip_site_and_a_tuple_assignment() -> None:
    reset = _LATCH + "\n    def recover(self):\n        self._error = None\n"
    assert _scan(reset)[("Holder", "_error")] == _Measured(("fail",), ("recover",))

    swapped = _LATCH + "\n    def take(self):\n        error, self._error = self._error, None\n"
    assert _scan(swapped)[("Holder", "_error")] == _Measured(("fail",), ("take",))

    second = _LATCH + "\n    def also_fail(self, error):\n        self._error = error\n"
    assert _scan(second)[("Holder", "_error")] == _Measured(("also_fail", "fail"), ())


def test_the_scan_needs_a_raising_guard_and_a_clear_start() -> None:
    unguarded = _LATCH.replace("            raise self._error\n", "            return None\n")
    assert ("Holder", "_error") not in _scan(unguarded)

    truthy = _LATCH.replace("if self._error is not None:", "if self._ready or self._error:")
    assert ("Holder", "_error") in _scan(truthy)

    started_set = _LATCH.replace("self._error = None", "self._error = RuntimeError()")
    assert ("Holder", "_error") not in _scan(started_set)

    field = (
        "class Record:\n    lost: bool = False\n\n    def mark(self):\n        self.lost = True\n"
    )
    assert _scan(field) == {}
    guarded = (
        field + "\n    def read(self):\n        if self.lost:\n            raise ValueError\n"
    )
    assert _scan(guarded) == {("Record", "lost"): _Measured(("mark",), ())}
