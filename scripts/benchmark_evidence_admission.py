"""Strict artifact contracts and threaded production-scheduler benchmark runner."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
import tracemalloc
import uuid
from contextlib import suppress
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any
from uuid import UUID

SCHEMA_VERSION = "benchmark-report-v1"
_MACHINE_VERSION = 1
_PROFILES = ((1, 100_000), (4, 25_000), (16, 6_250))
_LATE_YIELD_INTERVAL_CALLS = 8
_MAX_INTEGER = 9_223_372_036_854_775_807
_DIGEST = "0" * 64
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_EDITION_RE = re.compile(r"^[A-Za-z0-9 ._()&+-]+$")
_BUILD_RE = re.compile(r"^[0-9]{4,6}(?:\.[0-9]{1,6})?$")
_PYTHON_RE = re.compile(
    r"^3\.11\.[0-9]{1,3}(?:[abrc][0-9]{1,4})?(?: \([ -~]{1,96}\)(?: \[[ -~]{1,96}\])?)?$"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("canonical JSON cannot encode this value") from error
    return encoded.encode("utf-8") + b"\n"


def parse_canonical_json_bytes(raw: bytes) -> object:
    """Accept only the exact canonical UTF-8 serialization used by this module."""
    if type(raw) is not bytes or raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("canonical JSON bytes are invalid")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise ValueError("canonical JSON bytes are invalid") from error

    def duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("canonical JSON bytes are invalid")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=duplicates,
            parse_constant=lambda _: _raise_canonical(),
            parse_float=lambda _: _raise_canonical(),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("canonical JSON bytes are invalid") from error
    if canonical_json_bytes(value) != raw:
        raise ValueError("canonical JSON bytes are invalid")
    return value


def _raise_canonical() -> object:
    raise ValueError("canonical JSON bytes are invalid")


def _exact_int(value: object, *, minimum: int = 0, maximum: int = _MAX_INTEGER) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _is_clean_machine_text(value: object, *, allow_slash: bool = False) -> bool:
    if type(value) is not str or any(
        ord(char) < 32 or 127 <= ord(char) <= 159 or char == "\ufeff" for char in value
    ):
        return False
    if not allow_slash and (
        "/" in value or "\\" in value or re.search(r"[A-Za-z]:", value) or value.startswith("\\\\")
    ):
        return False
    needles = {os.environ.get("USERNAME", "").lower(), os.environ.get("COMPUTERNAME", "").lower()}
    return not any(needle and needle in value.lower() for needle in needles)


def validate_machine_manifest(value: object) -> dict[str, object]:
    expected = {
        "schemaVersion",
        "cpuModel",
        "logicalCpuCount",
        "installedRamBytes",
        "windowsEdition",
        "windowsBuild",
        "windowsArchitecture",
        "pythonFullVersion",
        "pythonArchitecture",
        "acPower",
        "powerScheme",
        "benchmarkScriptSha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("machine manifest has an invalid shape")
    if value["schemaVersion"] != 1 or type(value["schemaVersion"]) is not int:
        raise ValueError("machine manifest schema version is invalid")
    if not (
        type(value["cpuModel"]) is str
        and 1 <= len(value["cpuModel"]) <= 512
        and _is_clean_machine_text(value["cpuModel"])
    ):
        raise ValueError("machine manifest cpu model is invalid")
    if not _exact_int(value["logicalCpuCount"], minimum=1, maximum=65535) or not _exact_int(
        value["installedRamBytes"], minimum=1
    ):
        raise ValueError("machine manifest integer is invalid")
    if not (
        type(value["windowsEdition"]) is str
        and 1 <= len(value["windowsEdition"]) <= 128
        and _EDITION_RE.fullmatch(value["windowsEdition"])
        and _is_clean_machine_text(value["windowsEdition"])
    ):
        raise ValueError("machine manifest edition is invalid")
    if not (
        type(value["windowsBuild"]) is str
        and _BUILD_RE.fullmatch(value["windowsBuild"])
        and _is_clean_machine_text(value["windowsBuild"])
    ):
        raise ValueError("machine manifest build is invalid")
    if (
        value["windowsArchitecture"] not in {"x86", "AMD64", "ARM64"}
        or type(value["windowsArchitecture"]) is not str
        or not _is_clean_machine_text(value["windowsArchitecture"])
    ):
        raise ValueError("machine manifest windows architecture is invalid")
    if not (
        type(value["pythonFullVersion"]) is str
        and 1 <= len(value["pythonFullVersion"]) <= 256
        and _PYTHON_RE.fullmatch(value["pythonFullVersion"])
    ):
        raise ValueError("machine manifest Python version is invalid")
    if (
        value["pythonArchitecture"] not in {"32bit", "64bit"}
        or type(value["pythonArchitecture"]) is not str
        or not _is_clean_machine_text(value["pythonArchitecture"])
    ):
        raise ValueError("machine manifest Python architecture is invalid")
    if (
        type(value["acPower"]) is not bool
        or value["powerScheme"] != "high_performance"
        or type(value["powerScheme"]) is not str
    ):
        raise ValueError("machine manifest power value is invalid")
    if (
        type(value["benchmarkScriptSha256"]) is not str
        or not _DIGEST_RE.fullmatch(value["benchmarkScriptSha256"])
        or not _is_clean_machine_text(value["benchmarkScriptSha256"])
    ):
        raise ValueError("machine manifest script digest is invalid")
    return value


def _native_windows_values() -> dict[str, object]:
    if sys.platform != "win32":
        raise ValueError("live benchmark manifest requires native Windows")
    import winreg

    key = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as current:
        edition = winreg.QueryValueEx(current, "ProductName")[0]
        build = str(winreg.QueryValueEx(current, "CurrentBuildNumber")[0])
        with suppress(FileNotFoundError):
            build += "." + str(winreg.QueryValueEx(current, "UBR")[0])
    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    ) as cpu_key:
        cpu = winreg.QueryValueEx(cpu_key, "ProcessorNameString")[0]

    class Memory(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("avail_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended", ctypes.c_ulonglong),
        ]

    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("GlobalMemoryStatusEx failed")

    class Power(ctypes.Structure):
        _fields_ = [
            ("ac", ctypes.c_byte),
            ("battery", ctypes.c_byte),
            ("percent", ctypes.c_byte),
            ("reserved", ctypes.c_byte),
            ("life", ctypes.c_ulong),
            ("full", ctypes.c_ulong),
        ]

    power = Power()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)) or power.ac not in (
        0,
        1,
    ):
        raise ValueError("AC power status is unavailable")

    class SystemInfo(ctypes.Structure):
        _fields_ = [
            ("arch", ctypes.c_ushort),
            ("reserved", ctypes.c_ushort),
            ("page", ctypes.c_uint32),
            ("minaddr", ctypes.c_void_p),
            ("maxaddr", ctypes.c_void_p),
            ("mask", ctypes.c_void_p),
            ("count", ctypes.c_uint32),
            ("ptype", ctypes.c_uint32),
            ("level", ctypes.c_uint16),
            ("revision", ctypes.c_uint16),
        ]

    info = SystemInfo()
    ctypes.windll.kernel32.GetNativeSystemInfo(ctypes.byref(info))
    architecture = {0: "x86", 9: "AMD64", 12: "ARM64"}.get(info.arch)
    if architecture is None:
        raise ValueError("native Windows architecture is unavailable")
    # Personality must be GUID_MIN_POWER_SAVINGS; unavailable/native API failure is closed.
    active = ctypes.c_void_p()
    power_dll = ctypes.WinDLL("powrprof")
    if power_dll.PowerGetActiveScheme(None, ctypes.byref(active)) != 0:
        raise ValueError("active power scheme is unavailable")
    try:
        personality = (ctypes.c_byte * 16).from_buffer_copy(
            UUID("245d8541-3943-4422-b025-13a784f679b7").bytes_le
        )
        high_performance = UUID("8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")
        index = ctypes.c_ulong()
        if (
            power_dll.PowerReadACValueIndex(
                None, active, None, ctypes.byref(personality), ctypes.byref(index)
            )
            != 0
        ):
            raise ValueError("power personality index is unavailable")
        value_type = ctypes.c_ulong()
        size = ctypes.c_ulong(0)
        if (
            power_dll.PowerReadPossibleValue(
                None,
                None,
                ctypes.byref(personality),
                ctypes.byref(value_type),
                index.value,
                None,
                ctypes.byref(size),
            )
            != 0
            or size.value != 16
        ):
            raise ValueError("power personality value is unavailable")
        raw_personality = (ctypes.c_ubyte * size.value)()
        if (
            power_dll.PowerReadPossibleValue(
                None,
                None,
                ctypes.byref(personality),
                ctypes.byref(value_type),
                index.value,
                ctypes.byref(raw_personality),
                ctypes.byref(size),
            )
            != 0
            or UUID(bytes_le=bytes(raw_personality)) != high_performance
        ):
            raise ValueError("high-performance power personality is unavailable")
    finally:
        ctypes.windll.kernel32.LocalFree(active)
    return {
        "cpuModel": cpu,
        "logicalCpuCount": info.count,
        "installedRamBytes": memory.total_phys,
        "windowsEdition": edition,
        "windowsBuild": build,
        "windowsArchitecture": architecture,
        "acPower": bool(power.ac),
    }


def collect_live_machine_manifest() -> dict[str, object]:
    values = _native_windows_values()
    script = Path(__file__).resolve()
    result = {
        "schemaVersion": 1,
        **values,
        "pythonFullVersion": platform.python_version(),
        "pythonArchitecture": "64bit" if sys.maxsize > 2**32 else "32bit",
        "powerScheme": "high_performance",
        "benchmarkScriptSha256": sha256_bytes(script.read_bytes()),
    }
    return validate_machine_manifest(result)


def _write_atomic_bytes(path: Path, payload: bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (path.name + ".tmp-" + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    reopened = path.read_bytes()
    if reopened != payload:
        raise ValueError("atomic output reopen mismatch")
    return reopened


def write_machine_manifest_atomic(path: Path, manifest: object) -> str:
    validated = validate_machine_manifest(manifest)
    script_digest = sha256_bytes(Path(__file__).resolve().read_bytes())
    if validated["benchmarkScriptSha256"] != script_digest:
        raise ValueError("machine manifest script binding is invalid")
    reopened = _write_atomic_bytes(path, canonical_json_bytes(validated))
    if validate_machine_manifest(parse_canonical_json_bytes(reopened)) != validated:
        raise ValueError("reopened machine manifest differs")
    return sha256_bytes(reopened)


def load_machine_manifest(path: Path, *, require_live: bool) -> dict[str, object]:
    result = validate_machine_manifest(parse_canonical_json_bytes(path.read_bytes()))
    script_digest = sha256_bytes(Path(__file__).resolve().read_bytes())
    if result["benchmarkScriptSha256"] != script_digest:
        raise ValueError("machine manifest script binding is invalid")
    if require_live and result != collect_live_machine_manifest():
        raise ValueError("machine manifest does not match this live machine")
    return result


def nearest_rank(values: tuple[int, ...], fraction: float) -> int:
    if type(values) is not tuple or not values or any(not _exact_int(item) for item in values):
        raise ValueError("samples must be nonempty exact integers")
    numerators = {0.50: 50, 0.95: 95, 0.99: 99}
    if type(fraction) is not float or fraction not in numerators:
        raise ValueError("unsupported percentile")
    ordered = sorted(values)
    return ordered[max(0, (numerators[fraction] * len(ordered) + 99) // 100 - 1)]


def partition_ranges(*, calls: int, producers: int) -> tuple[tuple[int, int], ...]:
    profile = dict(_PROFILES)
    if (
        not _exact_int(calls, minimum=1)
        or not _exact_int(producers, minimum=1)
        or profile.get(producers) != calls // producers
        or calls != profile.get(producers, 0) * producers
    ):
        raise ValueError("calls must use a fixed benchmark profile")
    return tuple(
        (number * (calls // producers), (number + 1) * (calls // producers))
        for number in range(producers)
    )


def _snapshot() -> Any:
    from hermes_realtime.evidence import models as models

    def identifier(number: int) -> str:
        return str(UUID(int=number, version=4))

    return models.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=identifier(1),
        producer_instance_id=identifier(2),
        logical_session_id=identifier(3),
        event_id=identifier(4),
        event_sequence=1,
        event_kind=models.EventKind.SESSION_OPENED,
        payload=models.SessionOpenedPayloadV1(
            consent_epoch_id=identifier(5),
            binding_id=identifier(6),
            consent_version=models.CONSENT_VERSION,
            disclosure_digest="a" * 64,
            retention_hours=24,
            microphone_accepted=True,
            typed_accepted=True,
            predecessor_session_id=None,
        ),
    )


def _partition_any(calls: int, producers: int) -> tuple[tuple[int, int], ...]:
    if not _exact_int(calls) or not _exact_int(producers, minimum=1):
        raise ValueError("workload partition is invalid")
    quotient, remainder = divmod(calls, producers)
    start = 0
    result: list[tuple[int, int]] = []
    for index in range(producers):
        width = quotient + (1 if index < remainder else 0)
        result.append((start, start + width))
        start += width
    return tuple(result)


def _timing_metrics(samples: tuple[int, ...]) -> dict[str, int]:
    return {
        "p50Ns": nearest_rank(samples, 0.50),
        "p95Ns": nearest_rank(samples, 0.95),
        "p99Ns": nearest_rank(samples, 0.99),
        "maxNs": max(samples),
    }


def _ticker(
    stop: threading.Event,
    samples: list[int],
    *,
    minimum: int,
    interval_ns: int,
    start: threading.Event | None = None,
    ready: threading.Event | None = None,
    epoch_ns: list[int] | None = None,
) -> None:
    if minimum < 1 or interval_ns < 1:
        raise ValueError("ticker configuration is invalid")
    if ready is not None:
        ready.set()
    if start is not None:
        while not stop.is_set() and not start.wait(0.05):
            pass
        if stop.is_set():
            return
    epoch = time.perf_counter_ns() if epoch_ns is None else epoch_ns[0]
    if epoch < 1:
        raise ValueError("ticker epoch is invalid")
    deadline = epoch + interval_ns
    while not stop.is_set():
        remaining = deadline - time.perf_counter_ns()
        if remaining > 0:
            time.sleep(remaining / 1_000_000_000)
        if stop.is_set():
            return
        now = time.perf_counter_ns()
        samples.append(max(0, now - deadline))
        deadline += interval_ns


def _pace_call(
    *, local_index: int, local_count: int, window_start_ns: int, window_ns: int
) -> None:
    target_ns = window_start_ns + ((local_index + 1) * window_ns // local_count)
    remaining = target_ns - time.perf_counter_ns()
    if remaining > 0:
        time.sleep(remaining / 1_000_000_000)
    elif (local_index + 1) % _LATE_YIELD_INTERVAL_CALLS == 0:
        time.sleep(0)


def _run_null_lag_once(
    *,
    producers: int,
    calls: int,
    warmups: int,
    ticker_minimum: int,
    ticker_interval_ns: int,
) -> int:
    barrier = threading.Barrier(producers + 1)
    start = threading.Event()
    ticker_ready = threading.Event()
    window_closed = threading.Event()
    producers_done = threading.Event()
    permit = threading.Semaphore(32)
    handoff: SimpleQueue[object] = SimpleQueue()
    completed = [0]
    completed_lock = threading.Lock()
    lag_samples: list[int] = []
    window_start_ns = [0]
    window_ns = ticker_minimum * ticker_interval_ns + max(
        20_000_000, 2 * ticker_interval_ns
    )
    ranges = _partition_any(calls, producers)
    warmup_ranges = _partition_any(warmups, producers)
    total_items = calls + warmups

    def consumer() -> None:
        while True:
            with completed_lock:
                current = completed[0]
            if producers_done.is_set() and current >= total_items:
                return
            try:
                handoff.get_nowait()
            except Empty:
                time.sleep(0)
                continue
            with completed_lock:
                completed[0] += 1
            permit.release()
            time.sleep(0)

    def producer(index: int, work: tuple[int, int]) -> None:
        # Same producer/consumer/permit choreography as healthy, without scheduler work.
        for _ in range(*warmup_ranges[index]):
            permit.acquire()
            time.perf_counter_ns()
            handoff.put(None)
        barrier.wait()
        start.wait()
        local_count = work[1] - work[0]
        for local_index in range(local_count):
            _pace_call(
                local_index=local_index,
                local_count=local_count,
                window_start_ns=window_start_ns[0],
                window_ns=window_ns,
            )
            permit.acquire()
            before = time.perf_counter_ns()
            time.perf_counter_ns() - before
            handoff.put(None)

    ticker = threading.Thread(
        target=_ticker,
        args=(window_closed, lag_samples),
        kwargs={
            "minimum": ticker_minimum,
            "interval_ns": ticker_interval_ns,
            "start": start,
            "ready": ticker_ready,
            "epoch_ns": window_start_ns,
        },
    )
    threads = [
        threading.Thread(target=producer, args=(index, work))
        for index, work in enumerate(ranges)
    ]
    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()
    ticker.start()
    if not ticker_ready.wait(1.0):
        window_closed.set()
        ticker.join(1.0)
        raise RuntimeError("null ticker did not become ready")
    for thread in threads:
        thread.start()
    barrier.wait()
    gc.collect()
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    window_start_ns[0] = time.perf_counter_ns()
    start.set()
    try:
        for thread in threads:
            thread.join()
    finally:
        if was_enabled:
            gc.enable()
    window_closed.set()
    ticker.join()
    producers_done.set()
    consumer_thread.join(5.0)
    if consumer_thread.is_alive() or completed[0] != total_items:
        raise RuntimeError("null consumer did not complete exact handoff count")
    if len(lag_samples) < ticker_minimum:
        raise RuntimeError("null ticker produced too few samples")
    return nearest_rank(tuple(lag_samples), 0.99)


def _run_healthy_once(
    *,
    producers: int,
    calls: int,
    warmups: int,
    ticker_minimum: int,
    ticker_interval_ns: int,
) -> tuple[dict[str, int], tuple[int, ...], tuple[int, str]]:
    from hermes_realtime.evidence import admission

    scheduler = admission._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())
    first_ordinal = scheduler.next_admission_ordinal
    permit = threading.Semaphore(32)
    barrier = threading.Barrier(producers + 1)
    timed_start = threading.Event()
    ticker_ready = threading.Event()
    producers_done = threading.Event()
    cancelled = threading.Event()
    window_closed = threading.Event()
    lag_samples: list[int] = []
    window_start_ns = [0]
    window_ns = ticker_minimum * ticker_interval_ns + max(
        20_000_000, 2 * ticker_interval_ns
    )
    samples: list[int] = []
    lock = threading.Lock()
    errors: list[BaseException] = []
    counts = {
        "attempted": 0,
        "admitted": 0,
        "dequeued": 0,
        "completed": 0,
        "droppedCapacity": 0,
        "otherDisposition": 0,
    }
    total_items = calls + warmups
    call_ranges = _partition_any(calls, producers)
    warmup_ranges = _partition_any(warmups, producers)

    def fail_worker(error: BaseException) -> None:
        with lock:
            if not errors:
                errors.append(error)
        cancelled.set()
        producers_done.set()
        timed_start.set()
        with suppress(Exception):
            barrier.abort()
        for _ in range(32 * producers):
            permit.release()

    def acquire_permit() -> None:
        while not permit.acquire(timeout=0.05):
            if cancelled.is_set():
                raise RuntimeError("healthy benchmark cancelled")
        if cancelled.is_set():
            permit.release()
            raise RuntimeError("healthy benchmark cancelled")

    def consumer() -> None:
        try:
            while True:
                with lock:
                    completed = counts["completed"]
                if cancelled.is_set() or (
                    producers_done.is_set() and completed >= total_items
                ):
                    return
                try:
                    item = scheduler.dequeue_nowait()
                except Empty:
                    time.sleep(0)
                    continue
                scheduler.complete(item)
                with lock:
                    counts["dequeued"] += 1
                    counts["completed"] += 1
                permit.release()
                time.sleep(0)
        except BaseException as error:
            fail_worker(error)

    def producer(index: int) -> None:
        try:
            for _ in range(*warmup_ranges[index]):
                acquire_permit()
                item = scheduler.try_admit(prepared)
                if item is None:
                    permit.release()
                    raise RuntimeError("healthy warmup dropped capacity")
            barrier.wait()
            timed_start.wait()
            local_samples: list[int] = []
            local_admitted = local_dropped = local_other = 0
            work = call_ranges[index]
            local_count = work[1] - work[0]
            for local_index in range(local_count):
                _pace_call(
                    local_index=local_index,
                    local_count=local_count,
                    window_start_ns=window_start_ns[0],
                    window_ns=window_ns,
                )
                acquire_permit()
                before = time.perf_counter_ns()
                try:
                    item = scheduler.try_admit(prepared)
                except Exception:
                    permit.release()
                    raise
                elapsed = time.perf_counter_ns() - before
                if item is None:
                    permit.release()
                    local_dropped += 1
                else:
                    local_admitted += 1
                local_samples.append(elapsed)
            with lock:
                samples.extend(local_samples)
                counts["attempted"] += work[1] - work[0]
                counts["admitted"] += local_admitted
                counts["droppedCapacity"] += local_dropped
                counts["otherDisposition"] += local_other
        except BaseException as error:
            fail_worker(error)

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    ticker_thread = threading.Thread(
        target=_ticker,
        args=(window_closed, lag_samples),
        kwargs={
            "minimum": ticker_minimum,
            "interval_ns": ticker_interval_ns,
            "start": timed_start,
            "ready": ticker_ready,
            "epoch_ns": window_start_ns,
        },
    )
    producer_threads = [
        threading.Thread(target=producer, args=(index,), daemon=True)
        for index in range(producers)
    ]
    consumer_thread.start()
    ticker_thread.start()
    if not ticker_ready.wait(1.0):
        window_closed.set()
        ticker_thread.join(1.0)
        raise RuntimeError("healthy ticker did not become ready")
    for thread in producer_threads:
        thread.start()
    try:
        barrier.wait()
    except threading.BrokenBarrierError as error:
        fail_worker(error)
        for thread in producer_threads:
            thread.join(1.0)
        consumer_thread.join(1.0)
        window_closed.set()
        ticker_thread.join(1.0)
        raise RuntimeError("healthy benchmark worker failed before timing") from errors[0]
    gc.collect()
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    window_start_ns[0] = time.perf_counter_ns()
    timed_start.set()
    try:
        deadline = time.monotonic() + (window_ns / 1_000_000_000) + 5.0
        for thread in producer_threads:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                fail_worker(RuntimeError("healthy producer did not terminate"))
                break
    finally:
        if was_enabled:
            gc.enable()
    window_closed.set()
    ticker_thread.join(1.0)
    if ticker_thread.is_alive():
        fail_worker(RuntimeError("healthy ticker did not terminate"))
    producers_done.set()
    consumer_thread.join(1.0 if errors else 5.0)
    if consumer_thread.is_alive():
        fail_worker(RuntimeError("healthy consumer did not terminate"))
    if errors:
        raise RuntimeError("healthy benchmark worker failed") from errors[0]
    if len(samples) != calls or len(lag_samples) < ticker_minimum:
        raise RuntimeError("healthy benchmark sample cardinality is invalid")
    expected = {
        "attempted": calls,
        "admitted": calls,
        "dequeued": total_items,
        "completed": total_items,
        "droppedCapacity": 0,
        "otherDisposition": 0,
    }
    if counts != expected:
        raise RuntimeError("healthy benchmark count invariant failed")
    if scheduler.credits != (0, 0, 0):
        raise RuntimeError("healthy benchmark did not restore scheduler credits")
    if scheduler.next_admission_ordinal != first_ordinal + total_items:
        raise RuntimeError("healthy benchmark ordinal invariant failed")
    result = dict(counts)
    result["dequeued"] = calls
    result["completed"] = calls
    result.update(_timing_metrics(tuple(samples)))
    result["healthyEventLoopLagP99Ns"] = nearest_rank(tuple(lag_samples), 0.99)
    return result, tuple(samples), (prepared.canonical_byte_charge, prepared.canonical_sha256)


def _allocation_bytes_per_call(calls: int) -> int:
    from hermes_realtime.evidence import admission

    scheduler = admission._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(calls):
        item = scheduler.try_admit(prepared)
        if item is None:
            raise RuntimeError("allocation characterization dropped capacity")
        scheduler.complete(scheduler.dequeue_nowait())
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    if scheduler.credits != (0, 0, 0):
        raise RuntimeError("allocation characterization leaked credits")
    allocated = sum(max(0, item.size_diff) for item in after.compare_to(before, "lineno"))
    return allocated // calls


def _run_saturation_once(
    *, calls: int
) -> tuple[dict[str, int], tuple[int, ...], int, str]:
    from hermes_realtime.evidence import admission

    scheduler = admission._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())
    for _ in range(64):
        if scheduler.try_admit(prepared) is None:
            raise RuntimeError("saturation prefill dropped early")
    expected_bytes = 64 * prepared.canonical_byte_charge
    if scheduler.credits != (64, expected_bytes, 64):
        raise RuntimeError("saturation prefill credits are invalid")
    ordinal = scheduler.next_admission_ordinal
    samples: list[int] = []
    admitted = other = 0
    gc.collect()
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for _ in range(calls):
            before = time.perf_counter_ns()
            try:
                item = scheduler.try_admit(prepared)
            except Exception:
                other += 1
            else:
                admitted += item is not None
            samples.append(time.perf_counter_ns() - before)
    finally:
        if was_enabled:
            gc.enable()
    if admitted or other or scheduler.next_admission_ordinal != ordinal:
        raise RuntimeError("saturation attempts mutated admission state")
    if scheduler.credits != (64, expected_bytes, 64):
        raise RuntimeError("saturation attempts mutated scheduler credits")
    for _ in range(64):
        scheduler.complete(scheduler.dequeue_nowait())
    if scheduler.credits != (0, 0, 0):
        raise RuntimeError("saturation drain did not restore scheduler credits")
    result = {
        "attempted": calls,
        "admitted": 0,
        "dequeued": 64,
        "completed": 64,
        "droppedCapacity": calls,
        "otherDisposition": 0,
        **_timing_metrics(tuple(samples)),
    }
    return result, tuple(samples), prepared.canonical_byte_charge, prepared.canonical_sha256


def _maxima(repetitions: list[dict[str, int]], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: max(item[key] for item in repetitions) for key in keys}


def run_benchmark(
    *,
    repetitions: int = 5,
    warmups: int = 10_000,
    healthy_calls: int = 100_000,
    saturation_calls: int = 10_000,
    ticker_minimum: int = 10_000,
    ticker_interval_ns: int = 1_000_000,
    allocation_calls: int = 10_000,
) -> tuple[list[dict[str, object]], dict[str, object], int, str]:
    return _run_benchmark(
        repetitions=repetitions,
        warmups=warmups,
        healthy_calls=healthy_calls,
        saturation_calls=saturation_calls,
        ticker_minimum=ticker_minimum,
        ticker_interval_ns=ticker_interval_ns,
        allocation_calls=allocation_calls,
    )


def _run_benchmark(
    *,
    repetitions: int = 5,
    warmups: int = 10_000,
    healthy_calls: int = 100_000,
    saturation_calls: int = 10_000,
    ticker_minimum: int = 10_000,
    ticker_interval_ns: int = 1_000_000,
    allocation_calls: int = 10_000,
) -> tuple[list[dict[str, object]], dict[str, object], int, str]:
    if any(not _exact_int(value, minimum=1) for value in (
        repetitions, healthy_calls, saturation_calls, ticker_minimum,
        ticker_interval_ns, allocation_calls,
    )) or not _exact_int(warmups):
        raise ValueError("benchmark workload is invalid")
    profiles: list[dict[str, object]] = []
    payload_identity: tuple[int, str] | None = None
    for producers, designated_per_producer in _PROFILES:
        calls = healthy_calls if healthy_calls != 100_000 else producers * designated_per_producer
        healthy_repetitions: list[dict[str, int]] = []
        for _ in range(repetitions):
            null_lag = _run_null_lag_once(
                producers=producers,
                calls=calls,
                warmups=warmups,
                ticker_minimum=ticker_minimum,
                ticker_interval_ns=ticker_interval_ns,
            )
            healthy, samples, identity = _run_healthy_once(
                producers=producers,
                calls=calls,
                warmups=warmups,
                ticker_minimum=ticker_minimum,
                ticker_interval_ns=ticker_interval_ns,
            )
            if len(samples) != calls:
                raise RuntimeError("healthy timing sample cardinality changed")
            if payload_identity is not None and identity != payload_identity:
                raise RuntimeError("prepared payload identity changed across schedulers")
            payload_identity = identity
            healthy["nullEventLoopLagP99Ns"] = null_lag
            healthy["eventLoopLagRegressionP99Ns"] = max(
                0, healthy["healthyEventLoopLagP99Ns"] - null_lag
            )
            healthy_repetitions.append(healthy)
        aggregate_keys = (
            "p50Ns", "p95Ns", "p99Ns", "maxNs", "healthyEventLoopLagP99Ns",
            "nullEventLoopLagP99Ns", "eventLoopLagRegressionP99Ns",
        )
        profiles.append(
            {
                "producerCount": producers,
                "callsPerProducer": calls // producers,
                "repetitions": healthy_repetitions,
                **_maxima(healthy_repetitions, aggregate_keys),
                "allocationBytesPerCall": _allocation_bytes_per_call(allocation_calls),
            }
        )
    saturation_repetitions: list[dict[str, int]] = []
    for _ in range(repetitions):
        saturation, _, payload_bytes, payload_sha = _run_saturation_once(calls=saturation_calls)
        identity = (payload_bytes, payload_sha)
        if payload_identity is not None and identity != payload_identity:
            raise RuntimeError("prepared payload identity changed across repetitions")
        payload_identity = identity
        saturation_repetitions.append(saturation)
    assert payload_identity is not None
    saturation_report: dict[str, object] = {
        "prefilledRecords": 64,
        "drainedRecords": 64,
        "repetitions": saturation_repetitions,
        **_maxima(saturation_repetitions, ("p50Ns", "p95Ns", "p99Ns", "maxNs")),
    }
    return profiles, saturation_report, payload_identity[0], payload_identity[1]


def _healthy_repetition(metric: int) -> dict[str, int]:
    return {
        "attempted": 100_000,
        "admitted": 100_000,
        "dequeued": 100_000,
        "completed": 100_000,
        "droppedCapacity": 0,
        "otherDisposition": 0,
        "p50Ns": metric,
        "p95Ns": metric,
        "p99Ns": metric,
        "maxNs": metric,
        "healthyEventLoopLagP99Ns": metric,
        "nullEventLoopLagP99Ns": metric,
        "eventLoopLagRegressionP99Ns": 0,
    }


def make_report_for_testing(
    *, required: bool = False, passed: bool = False, metric: int = 0
) -> dict[str, object]:
    repetitions = [_healthy_repetition(metric) for _ in range(5)]
    profiles: list[dict[str, object]] = []
    for producers, calls in _PROFILES:
        profiles.append(
            {
                "producerCount": producers,
                "callsPerProducer": calls,
                "repetitions": [dict(item) for item in repetitions],
                "p50Ns": metric,
                "p95Ns": metric,
                "p99Ns": metric,
                "maxNs": metric,
                "healthyEventLoopLagP99Ns": metric,
                "nullEventLoopLagP99Ns": metric,
                "eventLoopLagRegressionP99Ns": 0,
                "allocationBytesPerCall": 0,
            }
        )
    saturation_repetition = {
        "attempted": 10_000,
        "admitted": 0,
        "dequeued": 64,
        "completed": 64,
        "droppedCapacity": 10_000,
        "otherDisposition": 0,
        "p50Ns": metric,
        "p95Ns": metric,
        "p99Ns": metric,
        "maxNs": metric,
    }
    return {
        "schemaVersion": SCHEMA_VERSION,
        "benchmarkScriptSha256": _DIGEST,
        "machineManifestSha256": _DIGEST,
        "payloadSha256": _DIGEST,
        "method": {
            "payloadCanonicalBytes": 1,
            "queueRecordCapacity": 64,
            "harnessPermitCapacity": 32,
            "lateYieldIntervalCalls": _LATE_YIELD_INTERVAL_CALLS,
            "warmupsPerProfile": 10_000,
            "repetitionsPerProfile": 5,
            "tickerIntervalNs": 1_000_000,
            "tickerMinimumSamples": 10_000,
        },
        "profiles": profiles,
        "saturation": {
            "prefilledRecords": 64,
            "drainedRecords": 64,
            "repetitions": [dict(saturation_repetition) for _ in range(5)],
            "p50Ns": metric,
            "p95Ns": metric,
            "p99Ns": metric,
            "maxNs": metric,
        },
        "thresholds": {
            "required": required,
            "healthyP99Ns": 1_000_000,
            "healthyMaxNs": 5_000_000,
            "lagRegressionP99Ns": 2_000_000,
        },
        "passed": passed,
    }


def _closed(value: object, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("report has an invalid shape")
    return value


def _metrics(value: dict[str, Any], keys: tuple[str, ...]) -> None:
    if set(value) != set(keys) or any(not _exact_int(value[key]) for key in keys):
        raise ValueError("report metric is invalid")
    if not (value["p50Ns"] <= value["p95Ns"] <= value["p99Ns"] <= value["maxNs"]):
        raise ValueError("report percentiles are invalid")


def validate_report(value: object) -> dict[str, object]:
    root = _closed(
        value,
        {
            "schemaVersion",
            "benchmarkScriptSha256",
            "machineManifestSha256",
            "payloadSha256",
            "method",
            "profiles",
            "saturation",
            "thresholds",
            "passed",
        },
    )
    if (
        root["schemaVersion"] != SCHEMA_VERSION
        or type(root["passed"]) is not bool
        or any(
            type(root[key]) is not str or not _DIGEST_RE.fullmatch(root[key])
            for key in ("benchmarkScriptSha256", "machineManifestSha256", "payloadSha256")
        )
    ):
        raise ValueError("report root is invalid")
    method = _closed(
        root["method"],
        {
            "payloadCanonicalBytes",
            "queueRecordCapacity",
            "harnessPermitCapacity",
            "lateYieldIntervalCalls",
            "warmupsPerProfile",
            "repetitionsPerProfile",
            "tickerIntervalNs",
            "tickerMinimumSamples",
        },
    )
    if not _exact_int(method["payloadCanonicalBytes"], minimum=1) or {
        key: method[key] for key in method if key != "payloadCanonicalBytes"
    } != {
        "queueRecordCapacity": 64,
        "harnessPermitCapacity": 32,
        "lateYieldIntervalCalls": _LATE_YIELD_INTERVAL_CALLS,
        "warmupsPerProfile": 10_000,
        "repetitionsPerProfile": 5,
        "tickerIntervalNs": 1_000_000,
        "tickerMinimumSamples": 10_000,
    }:
        raise ValueError("report method is invalid")
    if type(root["profiles"]) is not list or len(root["profiles"]) != 3:
        raise ValueError("report profile cardinality is invalid")
    healthy_keys = (
        "attempted",
        "admitted",
        "dequeued",
        "completed",
        "droppedCapacity",
        "otherDisposition",
        "p50Ns",
        "p95Ns",
        "p99Ns",
        "maxNs",
        "healthyEventLoopLagP99Ns",
        "nullEventLoopLagP99Ns",
        "eventLoopLagRegressionP99Ns",
    )
    profile_keys = {
        "producerCount",
        "callsPerProducer",
        "repetitions",
        "p50Ns",
        "p95Ns",
        "p99Ns",
        "maxNs",
        "healthyEventLoopLagP99Ns",
        "nullEventLoopLagP99Ns",
        "eventLoopLagRegressionP99Ns",
        "allocationBytesPerCall",
    }
    for profile, expected in zip(root["profiles"], _PROFILES, strict=True):
        profile = _closed(profile, profile_keys)
        if (
            (profile["producerCount"], profile["callsPerProducer"]) != expected
            or type(profile["repetitions"]) is not list
            or len(profile["repetitions"]) != 5
        ):
            raise ValueError("report profile order is invalid")
        for repetition in profile["repetitions"]:
            repetition = _closed(repetition, set(healthy_keys))
            _metrics(repetition, healthy_keys)
            if tuple(repetition[key] for key in healthy_keys[:6]) != (
                100_000,
                100_000,
                100_000,
                100_000,
                0,
                0,
            ) or repetition["eventLoopLagRegressionP99Ns"] != max(
                0, repetition["healthyEventLoopLagP99Ns"] - repetition["nullEventLoopLagP99Ns"]
            ):
                raise ValueError("healthy repetition semantics are invalid")
        aggregate_keys = healthy_keys[6:] + ("allocationBytesPerCall",)
        if any(not _exact_int(profile[key]) for key in aggregate_keys):
            raise ValueError("profile aggregate is invalid")
        for key in healthy_keys[6:]:
            if profile[key] != max(item[key] for item in profile["repetitions"]):
                raise ValueError("profile aggregate is invalid")
    saturation = _closed(
        root["saturation"],
        {"prefilledRecords", "drainedRecords", "repetitions", "p50Ns", "p95Ns", "p99Ns", "maxNs"},
    )
    saturation_keys = (
        "attempted",
        "admitted",
        "dequeued",
        "completed",
        "droppedCapacity",
        "otherDisposition",
        "p50Ns",
        "p95Ns",
        "p99Ns",
        "maxNs",
    )
    if (
        saturation["prefilledRecords"] != 64
        or saturation["drainedRecords"] != 64
        or type(saturation["repetitions"]) is not list
        or len(saturation["repetitions"]) != 5
    ):
        raise ValueError("saturation cardinality is invalid")
    for repetition in saturation["repetitions"]:
        repetition = _closed(repetition, set(saturation_keys))
        _metrics(repetition, saturation_keys)
        if tuple(repetition[key] for key in saturation_keys[:6]) != (10_000, 0, 64, 64, 10_000, 0):
            raise ValueError("saturation repetition is invalid")
    if any(not _exact_int(saturation[key]) for key in saturation_keys[6:]) or any(
        saturation[key] != max(item[key] for item in saturation["repetitions"])
        for key in saturation_keys[6:]
    ):
        raise ValueError("saturation aggregate is invalid")
    thresholds = _closed(
        root["thresholds"], {"required", "healthyP99Ns", "healthyMaxNs", "lagRegressionP99Ns"}
    )
    if type(thresholds["required"]) is not bool or (
        thresholds["healthyP99Ns"],
        thresholds["healthyMaxNs"],
        thresholds["lagRegressionP99Ns"],
    ) != (1_000_000, 5_000_000, 2_000_000):
        raise ValueError("thresholds are invalid")
    qualifies = all(
        profile["p99Ns"] < 1_000_000
        and profile["maxNs"] < 5_000_000
        and profile["eventLoopLagRegressionP99Ns"] < 2_000_000
        for profile in root["profiles"]
    )
    if (not thresholds["required"] and root["passed"]) or (
        thresholds["required"] and root["passed"] != qualifies
    ):
        raise ValueError("report threshold result is invalid")
    return root


def write_report_atomic(
    path: Path,
    report: object,
    *,
    machine_manifest_sha256: str,
    payload_canonical_bytes: int,
    payload_sha256: str,
) -> str:
    payload = canonical_json_bytes(validate_report(report))
    reopened = _write_atomic_bytes(path, payload)
    reopened_report = validate_report(parse_canonical_json_bytes(reopened))
    reopened_method = reopened_report["method"]
    if not isinstance(reopened_method, dict):
        raise ValueError("reopened report method is invalid")
    if (
        reopened_report["benchmarkScriptSha256"],
        reopened_report["machineManifestSha256"],
        reopened_method["payloadCanonicalBytes"],
        reopened_report["payloadSha256"],
    ) != (
        sha256_bytes(Path(__file__).resolve().read_bytes()),
        machine_manifest_sha256,
        payload_canonical_bytes,
        payload_sha256,
    ):
        raise ValueError("reopened benchmark bindings do not match live inputs")
    return sha256_bytes(reopened)


def _uuid_only(count: int) -> None:
    if not _exact_int(count, minimum=1, maximum=1_000_000):
        raise ValueError("uuid-count must be between 1 and 1000000")
    seen: set[str] = set()
    for _ in range(count):
        value = str(uuid.uuid4())
        if value in seen:
            raise RuntimeError("uuid collision observed")
        seen.add(value)


def make_report(
    *,
    machine_bytes: bytes,
    profiles: list[dict[str, object]],
    saturation: dict[str, object],
    payload_bytes: int,
    payload_sha: str,
    thresholds_required: bool,
) -> dict[str, object]:
    def profile_metric(profile: dict[str, object], name: str) -> int:
        value = profile[name]
        if type(value) is not int:
            raise ValueError("profile aggregate must be an exact integer")
        return value

    qualifies = all(
        profile_metric(profile, "p99Ns") < 1_000_000
        and profile_metric(profile, "maxNs") < 5_000_000
        and profile_metric(profile, "eventLoopLagRegressionP99Ns") < 2_000_000
        for profile in profiles
    )
    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "benchmarkScriptSha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "machineManifestSha256": sha256_bytes(machine_bytes),
        "payloadSha256": payload_sha,
        "method": {
            "payloadCanonicalBytes": payload_bytes,
            "queueRecordCapacity": 64,
            "harnessPermitCapacity": 32,
            "lateYieldIntervalCalls": _LATE_YIELD_INTERVAL_CALLS,
            "warmupsPerProfile": 10_000,
            "repetitionsPerProfile": 5,
            "tickerIntervalNs": 1_000_000,
            "tickerMinimumSamples": 10_000,
        },
        "profiles": profiles,
        "saturation": saturation,
        "thresholds": {
            "required": thresholds_required,
            "healthyP99Ns": 1_000_000,
            "healthyMaxNs": 5_000_000,
            "lagRegressionP99Ns": 2_000_000,
        },
        "passed": thresholds_required and qualifies,
    }
    return validate_report(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=100_000)
    parser.add_argument("--producer-counts", default="1,4,16")
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--machine-manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-thresholds", action="store_true")
    parser.add_argument("--uuid-only", action="store_true")
    parser.add_argument("--uuid-count", type=int, default=1_000_000)
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)
    if args.uuid_only:
        benchmark_options = {
            "--calls",
            "--producer-counts",
            "--warmup",
            "--repetitions",
            "--machine-manifest",
            "--report",
            "--require-thresholds",
        }
        if any(token.split("=", 1)[0] in benchmark_options for token in raw_args):
            parser.error("--uuid-only cannot be combined with benchmark arguments")
        _uuid_only(args.uuid_count)
        print("no collision observed")
        return 0
    if not (
        args.calls == 100_000
        and args.producer_counts == "1,4,16"
        and args.warmup == 10_000
        and args.repetitions == 5
        and args.machine_manifest
        and args.report
        and args.uuid_count == 1_000_000
    ):
        parser.error("only designated fixed benchmark arguments are accepted")
    machine_bytes = args.machine_manifest.read_bytes()
    load_machine_manifest(args.machine_manifest, require_live=args.require_thresholds)
    profiles, saturation, payload_bytes, payload_sha = run_benchmark()
    report = make_report(
        machine_bytes=machine_bytes,
        profiles=profiles,
        saturation=saturation,
        payload_bytes=payload_bytes,
        payload_sha=payload_sha,
        thresholds_required=args.require_thresholds,
    )
    write_report_atomic(
        args.report,
        report,
        machine_manifest_sha256=sha256_bytes(machine_bytes),
        payload_canonical_bytes=payload_bytes,
        payload_sha256=payload_sha,
    )
    return 0 if not args.require_thresholds or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
