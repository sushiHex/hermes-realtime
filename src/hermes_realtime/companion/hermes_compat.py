"""Every Hermes name the voice companion relies on, enumerated in one module.

Qualified against Hermes v0.21.0 (29112bef). ``SURFACE`` lists each name with the exact
parameters the pin gives it, and every lookup goes through ``resolve`` or ``_method``, which
refuse any name the tuple does not list: the surface is enforced, not documented. The
qualification (``scripts/qualify_hermes_voice_archive.py``) checks each name and signature
against the pin, and proves the private archive operation behaves exactly like Hermes's own
``append_messages_batch``.

The archive operation is private on purpose: Hermes's public append cannot verify the whole
archive and insert in the same write transaction, and nesting it inside ``_execute_write``
would open a second one.

Hermes is imported lazily, inside functions, so importing this module needs no Hermes.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import inspect
import json
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

from hermes_realtime.companion.integrity import (
    HEADER_COLUMNS,
    MESSAGE_COLUMNS,
    VOICE_SOURCE,
    ArchivePlan,
    ArchiveRefusal,
    Fingerprint,
    Projection,
    VoiceBatch,
    check_partition,
    plan_archive,
    platform_message_id,
    project,
    voice_metadata,
)
from hermes_realtime.companion.review import (
    MAX_REVIEW_BYTES,
    MAX_REVIEW_ROWS,
    MAX_REVIEW_TOKENS,
    ReviewRequest,
)
from hermes_realtime.companion.store import ConversationRecord

HERMES_MODULE = "hermes_state"


class SurfaceName(NamedTuple):
    """One relied-on name, and its exact parameter names at the pin (None: not callable)."""

    name: str
    parameters: tuple[str, ...] | None


ARCHIVE_SURFACE: tuple[SurfaceName, ...] = (
    SurfaceName("SessionDB", None),
    SurfaceName("SessionTurnLeaseLostError", None),
    SurfaceName("CompressionSessionClosedError", None),
    SurfaceName("SessionDB._TRANSCRIPT_WRITE_PATIENCE_S", None),
    SurfaceName("SessionDB._execute_write", ("self", "fn", "patience_s")),
    SurfaceName(
        "SessionDB._check_transcript_write_guards",
        (
            "self",
            "conn",
            "session_id",
            "compression_lock_holder",
            "turn_lease_holder",
            "turn_lease_ttl_seconds",
            "reject_active_turn_lease",
            "reject_active_compression_lock",
        ),
    ),
    SurfaceName("SessionDB._insert_message_rows", ("self", "conn", "session_id", "messages")),
    SurfaceName("SessionDB.create_session", ("self", "session_id", "source", "kwargs")),
    SurfaceName(
        "SessionDB.try_acquire_session_turn_lease",
        ("self", "session_id", "holder", "ttl_seconds", "patience_s"),
    ),
    SurfaceName(
        "SessionDB.refresh_session_turn_lease", ("self", "session_id", "holder", "ttl_seconds")
    ),
    SurfaceName("SessionDB.release_session_turn_lease", ("self", "session_id", "holder")),
    SurfaceName("SessionDB.close", ("self",)),
)
REVIEW_SURFACE: tuple[SurfaceName, ...] = (
    SurfaceName("SessionDB.db_path", None),
    SurfaceName("agent.background_review.prepare_background_review_run", ("agent",)),
    SurfaceName("agent.background_review.finish_background_review_run", ("agent", "run")),
    SurfaceName("agent.background_review.cancel_background_review_for_live_turn", ("agent",)),
    SurfaceName(
        "agent.background_review.spawn_background_review_thread",
        (
            "agent", "messages_snapshot", "review_memory", "review_skills",
            "focus", "task_cfg", "review_run",
        ),
    ),
    SurfaceName("hermes_cli.config.load_config_readonly", ()),
    SurfaceName("hermes_cli.config.get_config_path", ()),
    SurfaceName("hermes_cli.managed_scope.get_managed_dir", ()),
    SurfaceName(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        ("requested", "explicit_api_key", "explicit_base_url", "target_model"),
    ),
    SurfaceName("run_agent.AIAgent", None),
    SurfaceName("run_agent.AIAgent.close", ("self",)),
    SurfaceName("run_agent.AIAgent._safe_print", ("self", "args", "kwargs")),
    SurfaceName("run_agent.AIAgent._emit_auxiliary_failure", ("self", "task", "exc")),
    SurfaceName(
        "model_tools.get_tool_definitions",
        (
            "enabled_toolsets", "disabled_toolsets", "quiet_mode",
            "skip_tool_search_assembly",
        ),
    ),
    SurfaceName("gateway.run._profile_runtime_scope", ("profile_home",)),
    SurfaceName("hermes_constants.get_hermes_home", ()),
    SurfaceName("tools.memory_tool.MemoryStore", None),
    SurfaceName("tools.memory_tool.MemoryStore.load_from_disk", ("self",)),
    SurfaceName("agent.credential_pool.CredentialPool", None),
    SurfaceName("agent.credential_pool.CredentialPool.entries", ("self",)),
    SurfaceName("agent.credential_pool.PooledCredential", None),
    SurfaceName("agent.credential_pool.PooledCredential.to_dict", ("self",)),
    SurfaceName("run_agent.AIAgent.session_id", None),
    SurfaceName("run_agent.AIAgent.enabled_toolsets", None),
    SurfaceName("run_agent.AIAgent.disabled_toolsets", None),
    SurfaceName("run_agent.AIAgent.tools", None),
    SurfaceName("run_agent.AIAgent._memory_enabled", None),
    SurfaceName("run_agent.AIAgent._memory_store", None),
    SurfaceName("run_agent.AIAgent._memory_manager", None),
    SurfaceName("run_agent.AIAgent.background_review_callback", None),
    SurfaceName("run_agent.AIAgent._session_db", None),
    SurfaceName("run_agent.AIAgent._owns_session_db", None),
    SurfaceName("run_agent.AIAgent._end_session_on_close", None),
    SurfaceName("run_agent.AIAgent._persist_disabled", None),
    SurfaceName("run_agent.AIAgent._session_json_enabled", None),
    SurfaceName("agent.credential_pool.CredentialPool.provider", None),
)
REVIEW_INSTANCE_FIELDS = frozenset({
    "run_agent.AIAgent.session_id", "run_agent.AIAgent.enabled_toolsets",
    "run_agent.AIAgent.disabled_toolsets", "run_agent.AIAgent.tools",
    "run_agent.AIAgent._memory_enabled", "run_agent.AIAgent._memory_store",
    "run_agent.AIAgent._memory_manager", "run_agent.AIAgent.background_review_callback",
    "run_agent.AIAgent._session_db", "run_agent.AIAgent._owns_session_db",
    "run_agent.AIAgent._end_session_on_close", "run_agent.AIAgent._persist_disabled",
    "run_agent.AIAgent._session_json_enabled",
    "agent.credential_pool.CredentialPool.provider",
})
_REVIEW_MODULES = frozenset({
    "agent.background_review", "hermes_cli.config", "hermes_cli.runtime_provider",
    "hermes_cli.managed_scope", "run_agent", "model_tools", "gateway.run", "hermes_constants",
    "tools.memory_tool", "agent.credential_pool",
})
SURFACE = ARCHIVE_SURFACE + REVIEW_SURFACE
# The table shapes the projection relies on, bound by equality to the pin: every messages
# column (the fingerprint covers them all), and every sessions column (a new one could carry
# lineage or state the header does not bind).
MESSAGE_TABLE_COLUMNS = frozenset({"id", "session_id", *MESSAGE_COLUMNS})
SESSION_TABLE_COLUMNS = frozenset(
    {
        "id", "source", "user_id", "session_key", "chat_id", "chat_type", "thread_id",
        "display_name", "origin_json", "expiry_finalized", "model", "model_config",
        "system_prompt", "system_prompt_hash", "parent_session_id", "started_at", "ended_at",
        "end_reason", "message_count", "tool_call_count", "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "cwd", "git_branch",
        "git_repo_root", "git_metadata_generation", "billing_provider", "billing_base_url",
        "billing_mode", "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source",
        "pricing_version", "title", "title_source", "last_activity_at",
        "last_activity_description", "last_activity_provenance", "api_call_count",
        "handoff_state", "handoff_platform", "handoff_error",
        "compression_failure_cooldown_until", "compression_failure_error",
        "compression_fallback_streak", "compression_ineffective_count", "profile_name",
        "rewind_count", "archived", "pinned", "hidden", "last_read_at",
    }
)
if not {*HEADER_COLUMNS, "message_count"} <= SESSION_TABLE_COLUMNS:
    raise RuntimeError("the pinned sessions shape must hold every column the header reads")

_LISTED = frozenset(entry.name for entry in SURFACE)
_REVIEW_LISTED = frozenset(entry.name for entry in REVIEW_SURFACE)
_ROW_SQL = (
    f"SELECT {', '.join(MESSAGE_COLUMNS)} FROM messages "
    "WHERE session_id = ? ORDER BY id ASC LIMIT ?"
)
_HEADER_SQL = f"SELECT {', '.join(HEADER_COLUMNS)} FROM sessions WHERE id = ?"
# Lineage: any session, of any source, naming the archive as its parent.
_CHILD_SQL = "SELECT 1 FROM sessions WHERE parent_session_id = ? LIMIT 1"
# The counter Hermes keeps equal to the active rows; checked against the rows just read.
_COUNT_SQL = "SELECT message_count FROM sessions WHERE id = ?"


class CompatError(LookupError):
    """A Hermes name outside the enumerated surface was requested."""


def _lookup(name: str) -> Any:
    if name in _REVIEW_LISTED:
        for module_name in sorted(_REVIEW_MODULES, key=len, reverse=True):
            if name.startswith(module_name + "."):
                review_target = importlib.import_module(module_name)
                for part in name[len(module_name) + 1:].split("."):
                    review_target = getattr(review_target, part)
                return review_target
        raise CompatError("Hermes review module is not enumerated")
    target: Any = importlib.import_module(HERMES_MODULE)
    for part in name.split("."):
        target = getattr(target, part)
    return target


def resolve(name: str) -> Any:
    """Look up one listed Hermes name; an unlisted one is refused before any import."""

    if name not in _LISTED and name not in _REVIEW_LISTED:
        raise CompatError("Hermes name is not on the enumerated surface")
    return _lookup(name)


def _method(db: Any, name: str) -> Any:
    if name not in _LISTED or not name.startswith("SessionDB."):
        raise CompatError("Hermes name is not on the enumerated surface")
    return getattr(db, name.removeprefix("SessionDB."))


def _session_db(db: Any) -> Any:
    if type(db) is not resolve("SessionDB"):
        raise TypeError("the companion binds only an exact Hermes SessionDB")
    return db


def check_surface() -> tuple[str, ...]:
    """Names that are missing, or whose parameters differ from the pin; empty when current."""

    failures: list[str] = []
    for entry in ARCHIVE_SURFACE:
        try:
            target = _lookup(entry.name)
        except (ImportError, AttributeError):
            failures.append(f"missing:{entry.name}")
            continue
        if entry.parameters is not None:
            try:
                parameters = tuple(inspect.signature(target).parameters)
            except (TypeError, ValueError):
                parameters = ()
            if parameters != entry.parameters:
                failures.append(f"signature:{entry.name}")
    return tuple(failures)


def check_review_surface() -> tuple[str, ...]:
    """Refuse a changed pinned review function before building a review parent."""

    failures: list[str] = []
    for entry in REVIEW_SURFACE:
        if entry.name == "SessionDB.db_path" or entry.name in REVIEW_INSTANCE_FIELDS:
            continue  # Instance attribute, verified against the exact SessionDB below.
        try:
            target = _lookup(entry.name)
            if entry.parameters is not None:
                actual = tuple(inspect.signature(target).parameters)
                if actual != entry.parameters:
                    failures.append(f"signature:{entry.name}")
        except (ImportError, AttributeError, TypeError, ValueError):
            failures.append(f"missing:{entry.name}")
    return tuple(failures)


def check_shapes(db: Any) -> tuple[str, ...]:
    """Tables whose columns differ from what the projection relies on; empty when current."""

    def read(conn: Any) -> tuple[frozenset[str], frozenset[str]]:
        messages = frozenset(row[1] for row in conn.execute("PRAGMA table_info(messages)"))
        sessions = frozenset(row[1] for row in conn.execute("PRAGMA table_info(sessions)"))
        return messages, sessions

    messages, sessions = _method(_session_db(db), "SessionDB._execute_write")(read)
    failures: list[str] = []
    if messages != MESSAGE_TABLE_COLUMNS:
        failures.append("shape:messages")
    if sessions != SESSION_TABLE_COLUMNS:
        failures.append("shape:sessions")
    return tuple(failures)


def durability_level(db: Any) -> int:
    """``PRAGMA synchronous`` on the connection Hermes writes the archive through."""

    level = _method(_session_db(db), "SessionDB._execute_write")(
        lambda conn: conn.execute("PRAGMA synchronous").fetchone()[0]
    )
    if type(level) is not int:
        raise TypeError("PRAGMA synchronous did not answer an integer")
    return level


def _read_projection(conn: Any, session_id: str, cap: int) -> Projection | None:
    header = conn.execute(_HEADER_SQL, (session_id,)).fetchone()
    if header is None:
        return None
    # At most cap + 1 rows, inactive and compacted included, in Hermes's own read order.
    rows = conn.execute(_ROW_SQL, (session_id, cap + 1)).fetchall()
    child = conn.execute(_CHILD_SQL, (session_id,)).fetchone()
    (message_count,) = conn.execute(_COUNT_SQL, (session_id,)).fetchone()
    return project(
        dict(zip(HEADER_COLUMNS, tuple(header), strict=True)),
        [dict(zip(MESSAGE_COLUMNS, tuple(row), strict=True)) for row in rows],
        cap,
        has_children=child is not None,
        message_count=message_count,
    )


def read_projection(db: Any, session_id: str, cap: int) -> Projection | None:
    """One consistent read of a session's projection, serialized with every writer."""

    return _method(_session_db(db), "SessionDB._execute_write")(  # type: ignore[no-any-return]
        lambda conn: _read_projection(conn, session_id, cap)
    )


def open_session_db() -> Any:
    """The active profile's ``state.db``, exactly as Hermes itself resolves it.

    Called once per owned start, so the companion binds one profile's database for its life.
    """

    return resolve("SessionDB")()


def close_session_db(db: Any) -> None:
    """Close the database the companion opened; called once, at unload."""

    _method(_session_db(db), "SessionDB.close")()


def create_voice_session(db: Any, session_id: str) -> None:
    """Create the voice session.

    Hermes's creation is an upsert that keeps an existing row's fields. The caller verifies
    the created session against the expected genesis fingerprint, so a session that already
    held this id, with any other header or any row, is quarantined rather than adopted.
    """

    _method(_session_db(db), "SessionDB.create_session")(session_id, source=VOICE_SOURCE)


def acquire_lease(db: Any, session_id: str, holder: str, ttl_seconds: float) -> bool:
    acquire = _method(_session_db(db), "SessionDB.try_acquire_session_turn_lease")
    return acquire(session_id, holder, ttl_seconds=ttl_seconds) is True


def refresh_lease(db: Any, session_id: str, holder: str, ttl_seconds: float) -> bool:
    refresh = _method(_session_db(db), "SessionDB.refresh_session_turn_lease")
    return refresh(session_id, holder, ttl_seconds=ttl_seconds) is True


def release_lease(db: Any, session_id: str, holder: str) -> None:
    _method(_session_db(db), "SessionDB.release_session_turn_lease")(session_id, holder)


def archive_voice_rows(
    db: Any,
    session_id: str,
    holder: str,
    batch: VoiceBatch,
    expected_committed: Fingerprint,
    expected_pending: Fingerprint,
    *,
    conversation_id: str,
    cap: int,
    lease_ttl_seconds: float,
) -> ArchivePlan:
    """Verify the whole archive and append only the missing rows, in one write transaction.

    A batch whose rows and gaps do not partition its range is refused whole, before any
    transaction. Inside Hermes's ``BEGIN IMMEDIATE``: the lease guard with this holder; a
    projection of every row, which must equal ``expected_committed`` (or ``expected_pending``:
    already applied); an identity and full-content check of the batch; the insert of only
    the missing rows; the ``message_count`` update Hermes's own append makes; and a re-read
    that must equal ``expected_pending``. Any refusal raises inside the transaction, which
    rolls it back, so a refused batch never mutates the archive.
    """

    rows = check_partition(batch).rows
    db = _session_db(db)
    execute_write = _method(db, "SessionDB._execute_write")
    guard = _method(db, "SessionDB._check_transcript_write_guards")
    insert = _method(db, "SessionDB._insert_message_rows")
    patience = _method(db, "SessionDB._TRANSCRIPT_WRITE_PATIENCE_S")
    lease_lost = resolve("SessionTurnLeaseLostError")
    rotated = resolve("CompressionSessionClosedError")

    def archive(conn: Any) -> ArchivePlan:
        try:
            guard(
                conn,
                session_id,
                None,
                turn_lease_holder=holder,
                turn_lease_ttl_seconds=lease_ttl_seconds,
            )
        except lease_lost:
            raise ArchiveRefusal("lease_lost") from None
        except rotated:
            raise ArchiveRefusal("rotated") from None
        plan = plan_archive(
            _read_projection(conn, session_id, cap),
            conversation_id,
            rows,
            expected_committed,
            expected_pending,
            cap,
        )
        if plan.inserts:
            messages = [
                {
                    "role": row.role,
                    "content": row.text,
                    "timestamp": row.timestamp,
                    "platform_message_id": platform_message_id(conversation_id, row.identity),
                    "display_metadata": voice_metadata(row),
                }
                for row in plan.inserts
            ]
            insert(conn, session_id, messages)
            conn.execute(
                "UPDATE sessions SET message_count = message_count + ? WHERE id = ?",
                (len(messages), session_id),
            )
        after = _read_projection(conn, session_id, cap)
        if after is None or after.fingerprint() != expected_pending:
            raise ArchiveRefusal("drift")
        return plan

    return execute_write(archive, patience_s=patience)  # type: ignore[no-any-return]


class HermesArchivePort:
    """The archive port over one exact Hermes ``SessionDB``."""

    def __init__(self, db: Any) -> None:
        self._db = _session_db(db)
        self._review_bindings: dict[int, tuple[Any, str]] = {}
        self._failed_parent_closes: list[tuple[Any, threading.Thread, threading.Event]] = []
        self._rollback_finishes: dict[
            int, tuple[Any, Any, threading.Thread, threading.Event]
        ] = {}
        self._rollback_lock = threading.Lock()
        self._parent_creation_failed = False

    def drain_failed_parents(self, timeout: float) -> bool:
        if type(timeout) not in {float, int} or not 0 <= timeout <= 300:
            raise ValueError("invalid parent drain timeout")
        if not self._drain_rollback_finishes(None, timeout):
            return False
        for _, worker, _ in self._failed_parent_closes:
            if worker.ident is not None:
                worker.join(timeout)
        if any(worker.is_alive() for _, worker, _ in self._failed_parent_closes):
            return False
        failed = [parent for parent, _, succeeded in self._failed_parent_closes
                  if not succeeded.is_set()]
        self._failed_parent_closes.clear()
        for parent in failed:
            self._schedule_failed_parent_close(parent)
        return not failed

    def _schedule_failed_parent_close(self, parent: Any) -> None:
        succeeded = threading.Event()

        def cleanup() -> None:
            try:
                self.close_parent(parent)
            except BaseException:
                return
            succeeded.set()

        worker = threading.Thread(
            target=cleanup,
            name="voice-review-parent-cleanup", daemon=True,
        )
        self._failed_parent_closes.append((parent, worker, succeeded))
        try:
            worker.start()
        except Exception:
            # The unstarted worker still records ownership for the next drain.
            return

    def _rollback_worker(
        self, parent: Any, token: Any
    ) -> tuple[Any, Any, threading.Thread, threading.Event]:
        succeeded = threading.Event()

        def cleanup() -> None:
            try:
                self.finish(parent, token)
            except BaseException:
                return
            succeeded.set()

        worker = threading.Thread(
            target=cleanup, name="voice-review-rollback-finish", daemon=True,
        )
        return parent, token, worker, succeeded

    def _remember_rollback_finish(self, parent: Any, token: Any) -> None:
        pending = self._rollback_finishes
        with self._rollback_lock:
            operation = self._rollback_worker(parent, token)
            pending[id(token)] = operation
            # An unstarted worker still retains the token for the next drain.
            with contextlib.suppress(Exception):
                operation[2].start()

    def _drain_rollback_finishes(self, parent: Any | None, timeout: float) -> bool:
        pending = self._rollback_finishes
        if not pending:
            return True
        deadline = time.monotonic() + timeout
        with self._rollback_lock:
            keys = tuple(pending)
        for key in keys:
            retries = 0
            while True:
                with self._rollback_lock:
                    operation = pending.get(key)
                if operation is None or (parent is not None and operation[0] is not parent):
                    break
                _, token, worker, succeeded = operation
                if worker.ident is not None:
                    worker.join(max(0.0, deadline - time.monotonic()))
                with self._rollback_lock:
                    if pending.get(key) is not operation:
                        continue
                    if worker.is_alive():
                        return False
                    if succeeded.is_set():
                        del pending[key]
                        break
                    if retries:
                        return False
                    replacement = self._rollback_worker(operation[0], token)
                    pending[key] = replacement
                    with contextlib.suppress(Exception):
                        replacement[2].start()
                    retries += 1
        return True

    def _runtime_identity(self, runtime: dict[str, Any]) -> list[Any]:
        pool = runtime.get("credential_pool")
        if pool is None:
            credential: object = ["single", runtime.get("api_key")]
        else:
            if type(pool) is not resolve("agent.credential_pool.CredentialPool"):
                raise ArchiveRefusal("configuration")
            if "provider" not in pool.__dict__ or type(pool.provider) is not str:
                raise ArchiveRefusal("configuration")
            entries = resolve("agent.credential_pool.CredentialPool.entries")(pool)
            if type(entries) is not list:
                raise ArchiveRefusal("configuration")
            identities = []
            for entry in entries:
                if type(entry) is not resolve("agent.credential_pool.PooledCredential"):
                    raise ArchiveRefusal("configuration")
                value = resolve("agent.credential_pool.PooledCredential.to_dict")(entry)
                if type(value) is not dict:
                    raise ArchiveRefusal("configuration")
                identities.append([value.get("id"), value.get("source"), value.get("auth_type")])
            credential = ["pool", pool.provider, identities]
        return [
            runtime.get("provider"), runtime.get("model"), runtime.get("base_url"),
            runtime.get("api_mode"), runtime.get("request_overrides"), credential,
        ]

    def _binding(self, config: dict[str, Any], model: str, runtime: dict[str, Any]) -> str:
        """Bind main and routed credentials; pool token refresh keeps its stable identity."""

        auxiliary = config.get("auxiliary", {})
        if type(auxiliary) is not dict:
            raise ArchiveRefusal("configuration")
        review = auxiliary.get("background_review", {})
        if type(review) is not dict:
            raise ArchiveRefusal("configuration")
        routed: list[Any] | None = None
        provider = review.get("provider")
        route_model = review.get("model")
        if provider is not None or route_model is not None:
            if type(provider) is not str or type(route_model) is not str:
                raise ArchiveRefusal("configuration")
            if (
                provider.strip() and provider.strip() != "auto" and route_model.strip()
                and (provider.strip() != runtime.get("provider") or route_model.strip() != model)
            ):
                key, base = review.get("api_key"), review.get("base_url")
                if key is not None and type(key) is not str:
                    raise ArchiveRefusal("configuration")
                if base is not None and type(base) is not str:
                    raise ArchiveRefusal("configuration")
                route_runtime = resolve(
                    "hermes_cli.runtime_provider.resolve_runtime_provider"
                )(
                    requested=provider.strip(), target_model=route_model.strip(),
                    explicit_api_key=key.strip() or None if key is not None else None,
                    explicit_base_url=base.strip() or None if base is not None else None,
                )
                if type(route_runtime) is not dict:
                    raise ArchiveRefusal("configuration")
                routed = self._runtime_identity(route_runtime)
        payload = [config, model, self._runtime_identity(runtime), routed]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def verify_parent_binding(self, parent: Any) -> None:
        """Refuse a cached parent if its profile or credential authority has drifted."""

        registered = self._review_bindings.get(id(parent))
        if registered is None or registered[0] is not parent:
            raise ArchiveRefusal("configuration")
        try:
            with self._review_scope():
                self._verify_review_parent_shape(parent)
                signature = self._config_signature()
                config = resolve("hermes_cli.config.load_config_readonly")()
                if type(config) is not dict or type(config.get("model")) is not dict:
                    raise ArchiveRefusal("configuration")
                model_cfg = config["model"]
                model = model_cfg.get("default") or model_cfg.get("model")
                if type(model) is not str or not model.strip():
                    raise ArchiveRefusal("configuration")
                runtime = resolve("hermes_cli.runtime_provider.resolve_runtime_provider")(
                    target_model=model
                )
                if type(runtime) is not dict or self._config_signature() != signature:
                    raise ArchiveRefusal("configuration")
                if self._binding(config, model, runtime) != registered[1]:
                    raise ArchiveRefusal("configuration")
        except ArchiveRefusal:
            raise
        except Exception:
            raise ArchiveRefusal("configuration") from None

    def _verify_review_parent_shape(self, parent: Any) -> None:
        if type(parent) is not resolve("run_agent.AIAgent"):
            raise ArchiveRefusal("configuration")
        instance_fields = {
            name.removeprefix("run_agent.AIAgent.")
            for name in REVIEW_INSTANCE_FIELDS
            if name.startswith("run_agent.AIAgent.")
        }
        if not instance_fields <= parent.__dict__.keys():
            raise ArchiveRefusal("configuration")
        expected_tools = resolve("model_tools.get_tool_definitions")(
            enabled_toolsets=["memory", "skills"], quiet_mode=True
        )
        if (
            parent.enabled_toolsets != ["memory", "skills"]
            or parent.disabled_toolsets is not None
            or parent.tools != expected_tools
            or parent._memory_enabled is not True
            or type(parent._memory_store) is not resolve("tools.memory_tool.MemoryStore")
            or parent._memory_manager is not None
            or parent.background_review_callback is not None
            or parent._session_db is not None
            or parent._owns_session_db is not False
            or parent._end_session_on_close is not False
            or parent._persist_disabled is not True
            or parent._session_json_enabled is not False
        ):
            raise ArchiveRefusal("configuration")

    def bind_parent_callbacks(self, parent: Any, failed: Any) -> None:
        self.verify_parent_binding(parent)
        if not callable(failed):
            raise ArchiveRefusal("configuration")
        parent._safe_print = lambda *args, **kwargs: None
        parent.background_review_callback = None
        parent._emit_auxiliary_failure = lambda *args, **kwargs: failed()

    def close_parent(self, parent: Any) -> None:
        if not self._drain_rollback_finishes(parent, 5.0):
            raise ArchiveRefusal("busy")
        if type(parent) is not resolve("run_agent.AIAgent"):
            raise ArchiveRefusal("configuration")
        with self._review_scope():
            resolve("run_agent.AIAgent.close")(parent)
        self._review_bindings.pop(id(parent), None)

    def _review_home(self) -> Path:
        path = _method(self._db, "SessionDB.db_path")
        if type(path) is not type(Path()) or path.name != "state.db":
            raise ArchiveRefusal("configuration")
        return path.resolve().parent

    def _review_scope(self) -> Any:
        home = self._review_home()
        return resolve("gateway.run._profile_runtime_scope")(home)

    def _config_signature(self) -> tuple[
        tuple[int, int, str] | None, tuple[int, int, str] | None
    ]:
        """Parse bounded raw sources before Hermes can return a cached fallback config."""

        import yaml  # type: ignore[import-untyped]

        user_path = resolve("hermes_cli.config.get_config_path")()
        if (
            type(user_path) is not type(Path())
            or user_path.resolve() != self._review_home() / "config.yaml"
        ):
            raise ArchiveRefusal("configuration")
        managed_dir = resolve("hermes_cli.managed_scope.get_managed_dir")()
        if managed_dir is not None and type(managed_dir) is not type(Path()):
            raise ArchiveRefusal("configuration")

        def inspect_file(path: Path) -> tuple[int, int, str] | None:
            try:
                before = path.stat()
            except FileNotFoundError:
                return None
            if before.st_size > 1_048_576 or before.st_size < 0:
                raise ArchiveRefusal("configuration")
            with path.open("rb") as source:
                raw = source.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ArchiveRefusal("configuration")
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise ArchiveRefusal("configuration")
            value = yaml.safe_load(raw.decode("utf-8"))
            if value is not None and type(value) is not dict:
                raise ArchiveRefusal("configuration")
            return after.st_mtime_ns, after.st_size, hashlib.sha256(raw).hexdigest()

        try:
            return (
                inspect_file(user_path),
                None if managed_dir is None else inspect_file(managed_dir / "config.yaml"),
            )
        except ArchiveRefusal:
            raise
        except Exception:
            raise ArchiveRefusal("configuration") from None

    def check_compatibility(self) -> tuple[str, ...]:
        """Surface names, signatures and table shapes that differ from the pin."""
        return check_surface() + check_shapes(self._db)

    def durability_level(self) -> int:
        return durability_level(self._db)

    def create_session(self, session_id: str) -> None:
        create_voice_session(self._db, session_id)

    def acquire_lease(self, session_id: str, holder: str, ttl_seconds: float) -> bool:
        return acquire_lease(self._db, session_id, holder, ttl_seconds)

    def refresh_lease(self, session_id: str, holder: str, ttl_seconds: float) -> bool:
        return refresh_lease(self._db, session_id, holder, ttl_seconds)

    def release_lease(self, session_id: str, holder: str) -> None:
        release_lease(self._db, session_id, holder)

    def close(self) -> None:
        if not self._drain_rollback_finishes(None, 5.0):
            raise ArchiveRefusal("busy")
        close_session_db(self._db)

    def read_projection(self, session_id: str, cap: int) -> Projection | None:
        return read_projection(self._db, session_id, cap)

    def archive_rows(
        self,
        session_id: str,
        holder: str,
        conversation_id: str,
        batch: VoiceBatch,
        expected_committed: Fingerprint,
        expected_pending: Fingerprint,
        cap: int,
        lease_ttl_seconds: float,
    ) -> ArchivePlan:
        return archive_voice_rows(
            self._db,
            session_id,
            holder,
            batch,
            expected_committed,
            expected_pending,
            conversation_id=conversation_id,
            cap=cap,
            lease_ttl_seconds=lease_ttl_seconds,
        )

    def settings(self) -> tuple[int, dict[str, object]]:
        """Read the active profile's review policy afresh for every spawn."""

        if check_review_surface():
            raise ArchiveRefusal("incompatible")
        try:
            with self._review_scope():
                return self._settings_scoped()
        except ArchiveRefusal:
            raise
        except Exception:
            raise ArchiveRefusal("configuration") from None

    def _settings_scoped(self) -> tuple[int, dict[str, object]]:
        try:
            if resolve("hermes_constants.get_hermes_home")().resolve() != self._review_home():
                raise ArchiveRefusal("configuration")
            signature = self._config_signature()
            config = resolve("hermes_cli.config.load_config_readonly")()
            if self._config_signature() != signature:
                raise ArchiveRefusal("configuration")
            if type(config) is not dict:
                raise TypeError("profile config")
            memory = config.get("memory", {})
            auxiliary = config.get("auxiliary", {})
            if type(memory) is not dict or type(auxiliary) is not dict:
                raise TypeError("profile config")
            task_cfg = auxiliary.get("background_review", {})
            if type(task_cfg) is not dict:
                raise TypeError("review config")
            interval = memory.get("nudge_interval", 10)
            if type(interval) is not int or not 1 <= interval <= 1000:
                raise ArchiveRefusal("configuration")
            allowed = {
                definition["function"]["name"]
                for definition in resolve("model_tools.get_tool_definitions")(
                    enabled_toolsets=["memory", "skills"], quiet_mode=True
                )
            } | {"read_file", "search_files"}
            if allowed != {
                "memory", "skills_list", "skill_view", "skill_manage", "read_file", "search_files"
            }:
                raise ArchiveRefusal("incompatible")
            return interval, copy.deepcopy({
                **task_cfg,
                "enabled": memory.get("enabled", True) is True
                and task_cfg.get("enabled", True) is True,
            })
        except ArchiveRefusal:
            raise
        except Exception:
            raise ArchiveRefusal("configuration") from None

    def make_review_parent(self, session_id: str) -> Any:
        """Construct one confined parent using this profile's model and credentials."""

        if type(session_id) is not str or not session_id:
            raise TypeError("session ID is invalid")
        if self._parent_creation_failed:
            raise ArchiveRefusal("configuration")
        try:
            with self._review_scope():
                return self._make_review_parent_scoped(session_id)
        except ArchiveRefusal:
            raise
        except Exception:
            raise ArchiveRefusal("configuration") from None

    def _make_review_parent_scoped(self, session_id: str) -> Any:
        parent: Any = None
        try:
            if resolve("hermes_constants.get_hermes_home")().resolve() != self._review_home():
                raise ArchiveRefusal("configuration")
            signature = self._config_signature()
            config = resolve("hermes_cli.config.load_config_readonly")()
            model_cfg = config["model"]
            if type(model_cfg) is not dict:
                raise TypeError("model config")
            model = model_cfg.get("default") or model_cfg.get("model")
            if type(model) is not str or not model.strip():
                raise TypeError("model config")
            runtime = resolve("hermes_cli.runtime_provider.resolve_runtime_provider")(
                target_model=model
            )
            if self._config_signature() != signature:
                raise ArchiveRefusal("configuration")
            if type(runtime) is not dict:
                raise TypeError("runtime")
            binding = self._binding(config, model, runtime)
            parent = resolve("run_agent.AIAgent")(
                model=model,
                provider=runtime.get("provider"),
                base_url=runtime.get("base_url"),
                api_key=runtime.get("api_key"),
                api_mode=runtime.get("api_mode"),
                credential_pool=runtime.get("credential_pool"),
                request_overrides=runtime.get("request_overrides") or {},
                session_id=session_id,
                session_db=None,
                enabled_toolsets=["memory", "skills"],
                skip_memory=True,
                skip_background_review=True,
                quiet_mode=True,
                platform="api_server",
            )
            if type(parent) is not resolve("run_agent.AIAgent"):
                raise TypeError("review parent is not exact")
            if parent.session_id != session_id or parent._session_db is not None:
                raise TypeError("review parent session differs")
            parent._end_session_on_close = False
            parent._persist_disabled = True
            parent._session_json_enabled = False
            self._verify_review_parent_shape(parent)
            resolve("tools.memory_tool.MemoryStore.load_from_disk")(
                parent._memory_store
            )
            if self._config_signature() != signature:
                raise ArchiveRefusal("configuration")
            self._review_bindings[id(parent)] = parent, binding
            return parent
        except Exception:
            if parent is not None:
                self._parent_creation_failed = True
                # Constructor validation may fail after native resources were
                # allocated. Keep ownership until its cleanup thread exits.
                parent._session_db = None
                parent._owns_session_db = False
                parent._end_session_on_close = False
                parent._persist_disabled = True
                parent._session_json_enabled = False
                self._schedule_failed_parent_close(parent)
            raise ArchiveRefusal("configuration") from None

    def admit(
        self, parent: Any, record: ConversationRecord, request: ReviewRequest,
        cap: int, lease_ttl_seconds: float,
    ) -> tuple[list[dict[str, str]], Any] | None:
        """Verify the chain, snapshot the exact range, and take the run token in one write lock."""

        if type(record) is not ConversationRecord or type(request) is not ReviewRequest:
            raise TypeError("review admission needs exact records")
        committed = record.committed
        if committed is None or record.pending is not None:
            raise ArchiveRefusal("busy")
        if record.holder is None:
            raise ArchiveRefusal("lease_lost")
        if not self._drain_rollback_finishes(parent, 5.0):
            raise ArchiveRefusal("busy")
        prepare = resolve("agent.background_review.prepare_background_review_run")
        guard = _method(self._db, "SessionDB._check_transcript_write_guards")
        lease_lost = resolve("SessionTurnLeaseLostError")
        rotated = resolve("CompressionSessionClosedError")

        def take(conn: Any) -> tuple[list[dict[str, str]], Any] | None:
            try:
                guard(
                    conn, record.session_id, None,
                    turn_lease_holder=record.holder,
                    turn_lease_ttl_seconds=lease_ttl_seconds,
                )
            except lease_lost:
                raise ArchiveRefusal("lease_lost") from None
            except rotated:
                raise ArchiveRefusal("rotated") from None
            projection = _read_projection(conn, record.session_id, cap)
            if projection is None or projection.fingerprint() != committed.fingerprint:
                raise ArchiveRefusal("missing" if projection is None else "mismatch")
            rows = conn.execute(
                "SELECT role, content, platform_message_id FROM messages "
                "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (record.session_id, cap + 1),
            ).fetchall()
            prefix = f"voice:{request.conversation_id}:{request.generation}:"
            snapshot: list[dict[str, str]] = []
            for role, content, identity in rows:
                if type(identity) is not str or not identity.startswith(prefix):
                    continue
                tail = identity[len(prefix):]
                if not tail.isascii() or not tail.isdecimal() or str(int(tail)) != tail:
                    raise ArchiveRefusal("mismatch")
                seq = int(tail)
                if request.seq_from <= seq <= request.seq_through:
                    if type(role) is not str or role not in {"user", "assistant"}:
                        raise ArchiveRefusal("mismatch")
                    if type(content) is not str or not content.strip():
                        raise ArchiveRefusal("mismatch")
                    snapshot.append({"role": role, "content": content})
            if not 1 <= len(snapshot) <= MAX_REVIEW_ROWS:
                raise ArchiveRefusal("window")
            payload_bytes = len(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))
            # The snapshot itself has a conservative token bound (one token per
            # UTF-8 byte). Hermes separately accounts for its prompt and schema.
            if payload_bytes > MAX_REVIEW_BYTES or payload_bytes > MAX_REVIEW_TOKENS:
                raise ArchiveRefusal("window")
            token = prepare(parent)
            if token is None:
                return None
            return snapshot, token

        prepared_token: Any = None

        def owned_take(conn: Any) -> tuple[list[dict[str, str]], Any] | None:
            nonlocal prepared_token
            admitted = take(conn)
            if admitted is not None:
                prepared_token = admitted[1]
            return admitted

        try:
            return _method(self._db, "SessionDB._execute_write")(owned_take)  # type: ignore[no-any-return]
        except BaseException:
            if prepared_token is not None:
                try:
                    self.finish(parent, prepared_token)
                except BaseException:
                    self._remember_rollback_finish(parent, prepared_token)
            raise

    def spawn(
        self, parent: Any, snapshot: list[dict[str, str]], token: Any,
        task_cfg: dict[str, object],
    ) -> Any:
        _, current_cfg = self.settings()
        if current_cfg != task_cfg:
            raise ArchiveRefusal("configuration")
        target, _ = resolve("agent.background_review.spawn_background_review_thread")(
            parent, snapshot, review_memory=True, review_skills=True, focus=None,
            task_cfg=task_cfg, review_run=token,
        )
        if not callable(target):
            raise ArchiveRefusal("incompatible")
        def scoped_target() -> None:
            with self._review_scope():
                _, at_dispatch = self.settings()
                if at_dispatch != task_cfg or at_dispatch.get("enabled") is not True:
                    raise ArchiveRefusal("configuration")
                target()
        return scoped_target

    def finish(self, parent: Any, token: Any) -> None:
        resolve("agent.background_review.finish_background_review_run")(parent, token)

    def cancel(self, parent: Any, token: Any) -> None:
        resolve("agent.background_review.cancel_background_review_for_live_turn")(parent)
