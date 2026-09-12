"""Task 11 contract tests for the strict evidence-admission benchmark."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import platform
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_evidence_admission.py"
MACHINE_SCHEMA = ROOT / "scripts" / "schemas" / "benchmark-machine-v1.schema.json"
REPORT_SCHEMA = ROOT / "scripts" / "schemas" / "benchmark-report-v1.schema.json"
DIGEST = "0" * 64


def benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_evidence_admission", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_schema_definitions_are_literal_closed_objects() -> None:
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    definitions = schema["$defs"]
    assert definitions
    for name, definition in definitions.items():
        assert definition.get("type") == "object", name
        assert definition.get("additionalProperties") is False, name
        assert set(definition.get("required", ())) == set(definition.get("properties", {})), name
        assert "$ref" not in definition, name


def machine_value() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "cpuModel": "Synthetic Test CPU",
        "logicalCpuCount": 8,
        "installedRamBytes": 17_179_869_184,
        "windowsEdition": "Windows Test Edition",
        "windowsBuild": "26000.1",
        "windowsArchitecture": "AMD64",
        "pythonFullVersion": "3.11.15 (main, Jul  1 2026, 00:00:00) [MSC v.1944 64 bit (AMD64)]",
        "pythonArchitecture": "64bit",
        "acPower": True,
        "powerScheme": "high_performance",
        "benchmarkScriptSha256": DIGEST,
    }


def healthy_repetition() -> dict[str, int]:
    return {
        "attempted": 100_000,
        "admitted": 100_000,
        "dequeued": 100_000,
        "completed": 100_000,
        "droppedCapacity": 0,
        "otherDisposition": 0,
        "p50Ns": 10,
        "p95Ns": 20,
        "p99Ns": 30,
        "maxNs": 40,
        "healthyEventLoopLagP99Ns": 50,
        "nullEventLoopLagP99Ns": 20,
        "eventLoopLagRegressionP99Ns": 30,
    }


def saturation_repetition() -> dict[str, int]:
    return {
        "attempted": 10_000,
        "admitted": 0,
        "dequeued": 64,
        "completed": 64,
        "droppedCapacity": 10_000,
        "otherDisposition": 0,
        "p50Ns": 10,
        "p95Ns": 20,
        "p99Ns": 30,
        "maxNs": 40,
    }


def report_value(*, required: bool = False, passed: bool = False) -> dict[str, object]:
    repetitions = [healthy_repetition() for _ in range(5)]
    profiles = []
    for producers, calls in ((1, 100_000), (4, 25_000), (16, 6_250)):
        profiles.append(
            {
                "producerCount": producers,
                "callsPerProducer": calls,
                "repetitions": copy.deepcopy(repetitions),
                "p50Ns": 10,
                "p95Ns": 20,
                "p99Ns": 30,
                "maxNs": 40,
                "healthyEventLoopLagP99Ns": 50,
                "nullEventLoopLagP99Ns": 20,
                "eventLoopLagRegressionP99Ns": 30,
                "allocationBytesPerCall": 0,
            }
        )
    saturation_repetitions = [saturation_repetition() for _ in range(5)]
    return {
        "schemaVersion": "benchmark-report-v1",
        "benchmarkScriptSha256": DIGEST,
        "machineManifestSha256": DIGEST,
        "payloadSha256": DIGEST,
        "method": {
            "payloadCanonicalBytes": 1_024,
            "queueRecordCapacity": 64,
            "harnessPermitCapacity": 32,
            "lateYieldIntervalCalls": 8,
            "warmupsPerProfile": 10_000,
            "repetitionsPerProfile": 5,
            "tickerIntervalNs": 1_000_000,
            "tickerMinimumSamples": 10_000,
        },
        "profiles": profiles,
        "saturation": {
            "prefilledRecords": 64,
            "drainedRecords": 64,
            "repetitions": saturation_repetitions,
            "p50Ns": 10,
            "p95Ns": 20,
            "p99Ns": 30,
            "maxNs": 40,
        },
        "thresholds": {
            "required": required,
            "healthyP99Ns": 1_000_000,
            "healthyMaxNs": 5_000_000,
            "lagRegressionP99Ns": 2_000_000,
        },
        "passed": passed,
    }


def test_strict_canonical_parser_rejects_duplicate_bom_nonfinite_and_noncanonical_bytes() -> None:
    module = benchmark()
    good = b'{"a":1}\n'
    assert module.parse_canonical_json_bytes(good) == {"a": 1}
    for invalid in (
        b'\xef\xbb\xbf{"a":1}\n',
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{ "a":1}\n',
        b'{"a":1}\r\n',
        b'{"a":1}',
        b"\xff\n",
    ):
        with pytest.raises(ValueError, match="canonical"):
            module.parse_canonical_json_bytes(invalid)


def test_nearest_rank_and_fixed_partition_are_exact() -> None:
    module = benchmark()
    assert module.nearest_rank((1, 2, 3, 4), 0.50) == 2
    assert module.nearest_rank((1, 2, 3, 4), 0.99) == 4
    assert module.partition_ranges(calls=100_000, producers=4) == (
        (0, 25_000),
        (25_000, 50_000),
        (50_000, 75_000),
        (75_000, 100_000),
    )
    with pytest.raises(ValueError):
        module.nearest_rank((), 0.50)
    with pytest.raises(ValueError):
        module.partition_ranges(calls=10, producers=3)


def test_checked_in_schemas_have_exact_identity_and_closed_object_contracts() -> None:
    machine = json.loads(MACHINE_SCHEMA.read_text(encoding="utf-8"))
    report = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    assert machine["$schema"] == report["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert machine["$id"] == "https://hermes.local/schemas/benchmark-machine-v1.schema.json"
    assert report["$id"] == "https://hermes.local/schemas/benchmark-report-v1.schema.json"
    assert machine["additionalProperties"] is False
    assert report["additionalProperties"] is False
    assert machine["required"] == [
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
    ]
    assert report["properties"]["profiles"]["minItems"] == 3
    assert report["properties"]["profiles"]["maxItems"] == 3
    assert report["properties"]["saturation"]["additionalProperties"] is False


def test_machine_manifest_semantics_reject_shape_identity_paths_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark()
    monkeypatch.setenv("USERNAME", "privateuser")
    value = machine_value()
    assert module.validate_machine_manifest(value) is value
    mutations = (
        ("extra", 1),
        ("schemaVersion", True),
        ("logicalCpuCount", 0),
        ("installedRamBytes", 0),
        ("windowsArchitecture", "x64"),
        ("pythonArchitecture", "AMD64"),
        ("powerScheme", "balanced"),
        ("cpuModel", "C:/private/model"),
        ("windowsEdition", "privateuser edition"),
        ("benchmarkScriptSha256", "A" * 64),
    )
    for key, replacement in mutations:
        broken = dict(value)
        if key == "extra":
            broken[key] = replacement
        else:
            broken[key] = replacement
        with pytest.raises(
            ValueError, match="machine|cpuModel|Windows|architecture|power|SHA|identity"
        ):
            module.validate_machine_manifest(broken)


def test_live_machine_manifest_uses_canonical_patch_python_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark()
    monkeypatch.setattr(module, "_native_windows_values", lambda: {
        "cpuModel": "Synthetic Test CPU",
        "logicalCpuCount": 8,
        "installedRamBytes": 17_179_869_184,
        "windowsEdition": "Windows Test Edition",
        "windowsBuild": "26000.1",
        "windowsArchitecture": "AMD64",
        "acPower": True,
    })
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.16")

    observed = module.collect_live_machine_manifest()

    assert observed["pythonFullVersion"] == "3.11.16"


def test_atomic_machine_writer_fsyncs_reopens_and_revalidates(tmp_path: Path) -> None:
    module = benchmark()
    output = tmp_path / "machine.json"
    manifest = machine_value()
    manifest["benchmarkScriptSha256"] = module.sha256_bytes(SCRIPT.read_bytes())

    digest = module.write_machine_manifest_atomic(output, manifest)

    assert digest == module.sha256_bytes(output.read_bytes())
    assert module.parse_canonical_json_bytes(output.read_bytes()) == manifest
    assert not tuple(tmp_path.glob("machine.json.tmp-*"))


def test_report_semantics_reject_every_material_cross_field_drift() -> None:
    module = benchmark()
    value = report_value()
    assert module.validate_report(value) is value
    broken_values = []
    extra = copy.deepcopy(value)
    extra["extra"] = 1
    broken_values.append(extra)
    count = copy.deepcopy(value)
    count["profiles"][0]["repetitions"][0]["attempted"] = 99_999
    broken_values.append(count)
    percentile = copy.deepcopy(value)
    percentile["profiles"][0]["repetitions"][0]["p50Ns"] = 31
    broken_values.append(percentile)
    aggregate = copy.deepcopy(value)
    aggregate["profiles"][1]["p99Ns"] = 31
    broken_values.append(aggregate)
    lag = copy.deepcopy(value)
    lag["profiles"][2]["repetitions"][0]["eventLoopLagRegressionP99Ns"] = 31
    broken_values.append(lag)
    characterization = copy.deepcopy(value)
    characterization["passed"] = True
    broken_values.append(characterization)
    saturation = copy.deepcopy(value)
    saturation["saturation"]["repetitions"][0]["droppedCapacity"] = 9_999
    broken_values.append(saturation)
    for broken in broken_values:
        with pytest.raises(ValueError):
            module.validate_report(broken)
    assert module.validate_report(report_value(required=True, passed=True))["passed"] is True


def test_profile_lag_aggregate_uses_per_metric_maxima_across_repetitions() -> None:
    module = benchmark()
    report = report_value()
    profile = cast(dict[str, Any], cast(list[object], report["profiles"])[0])
    first, second = profile["repetitions"][:2]
    first["healthyEventLoopLagP99Ns"] = 100
    first["nullEventLoopLagP99Ns"] = 90
    first["eventLoopLagRegressionP99Ns"] = 10
    second["healthyEventLoopLagP99Ns"] = 80
    second["nullEventLoopLagP99Ns"] = 0
    second["eventLoopLagRegressionP99Ns"] = 80
    profile["healthyEventLoopLagP99Ns"] = 100
    profile["nullEventLoopLagP99Ns"] = 90
    profile["eventLoopLagRegressionP99Ns"] = 80
    assert module.validate_report(report) is report


def test_atomic_report_writer_fsyncs_reopens_and_revalidates(tmp_path: Path) -> None:
    module = benchmark()
    output = tmp_path / "report.json"
    report = report_value()
    report["benchmarkScriptSha256"] = module.sha256_bytes(SCRIPT.read_bytes())
    digest = module.write_report_atomic(
        output,
        report,
        machine_manifest_sha256=DIGEST,
        payload_canonical_bytes=1_024,
        payload_sha256=DIGEST,
    )
    assert digest == module.sha256_bytes(output.read_bytes())
    assert module.parse_canonical_json_bytes(output.read_bytes()) == report
    assert not tuple(tmp_path.glob(".report.json.*.tmp"))


def test_atomic_report_writer_rejects_reopened_binding_mismatch(tmp_path: Path) -> None:
    module = benchmark()
    report = report_value()
    with pytest.raises(ValueError, match="reopened benchmark bindings"):
        module.write_report_atomic(
            tmp_path / "report.json",
            report,
            machine_manifest_sha256=DIGEST,
            payload_canonical_bytes=1_024,
            payload_sha256=DIGEST,
        )


def test_argument_rejection_and_uuid_only_output(capsys: pytest.CaptureFixture[str]) -> None:
    module = benchmark()
    with pytest.raises(SystemExit):
        module.main(["--calls", "99", "--producer-counts", "1"])
    with pytest.raises(SystemExit):
        module.main(["--uuid-only", "--uuid-count", "20", "--require-thresholds"])
    assert module.main(["--uuid-only", "--uuid-count", "20"]) == 0
    assert capsys.readouterr().out.strip().endswith("no collision observed")


def test_uuid_only_rejects_even_default_valued_benchmark_options() -> None:
    module = benchmark()
    with pytest.raises(SystemExit):
        module.main(["--uuid-only", "--calls", "100000"])


def test_reduced_runner_uses_real_scheduler_and_restores_credits() -> None:
    """The test-only small workload still exercises the shipped scheduler, never a fake."""
    module = benchmark()
    profiles, saturation, payload_bytes, payload_sha = module.run_benchmark(
        repetitions=1,
        warmups=2,
        healthy_calls=4,
        saturation_calls=3,
        ticker_minimum=2,
        ticker_interval_ns=100_000,
        allocation_calls=3,
    )
    assert [(item["producerCount"], item["callsPerProducer"]) for item in profiles] == [
        (1, 4),
        (4, 1),
        (16, 0),
    ]
    assert all(item["repetitions"][0]["attempted"] == 4 for item in profiles)
    assert all(item["repetitions"][0]["admitted"] == 4 for item in profiles)
    assert saturation["repetitions"][0]["droppedCapacity"] == 3
    assert payload_bytes > 0 and len(payload_sha) == 64


def test_healthy_runner_places_permit_before_timed_admission_and_releases_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark()
    events: list[str] = []

    class Permit:
        def acquire(self, timeout: float | None = None) -> bool:
            del timeout
            events.append("acquire")
            return True

        def release(self) -> None:
            events.append("release")

    monkeypatch.setattr(module.threading, "Semaphore", lambda _: Permit())
    original_admit = None
    from hermes_realtime.evidence import admission

    original_admit = admission._ProductionEvidenceSchedulerV1.try_admit

    def observed_admit(
        self: admission._ProductionEvidenceSchedulerV1,
        prepared: admission._PreparedEvidenceRecordV1,
    ) -> object:
        events.append("admit")
        return original_admit(self, prepared)

    monkeypatch.setattr(admission._ProductionEvidenceSchedulerV1, "try_admit", observed_admit)
    module._run_healthy_once(
        producers=1, calls=2, warmups=1, ticker_minimum=1, ticker_interval_ns=100_000
    )
    assert events.index("acquire") < events.index("admit")
    assert events.index("release") > events.index("admit")


def test_healthy_runner_prepares_once_and_reuses_the_exact_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark()
    from hermes_realtime.evidence import admission
    from hermes_realtime.evidence.models import EvidenceSnapshotV1

    prepare_calls: list[object] = []
    admitted_handles: list[object] = []
    original_prepare = admission._ProductionEvidenceSchedulerV1.prepare
    original_admit = admission._ProductionEvidenceSchedulerV1.try_admit

    def observed_prepare(
        self: admission._ProductionEvidenceSchedulerV1,
        snapshot: EvidenceSnapshotV1,
    ) -> admission._PreparedEvidenceRecordV1:
        result = original_prepare(self, snapshot)
        prepare_calls.append(result)
        return result

    def observed_admit(
        self: admission._ProductionEvidenceSchedulerV1,
        prepared: admission._PreparedEvidenceRecordV1,
    ) -> object:
        admitted_handles.append(prepared)
        return original_admit(self, prepared)

    monkeypatch.setattr(admission._ProductionEvidenceSchedulerV1, "prepare", observed_prepare)
    monkeypatch.setattr(admission._ProductionEvidenceSchedulerV1, "try_admit", observed_admit)
    module._run_healthy_once(
        producers=2, calls=4, warmups=2, ticker_minimum=1, ticker_interval_ns=100_000
    )
    assert len(prepare_calls) == 1
    assert len(admitted_handles) == 6
    assert all(handle is prepare_calls[0] for handle in admitted_handles)


def test_saturation_drop_does_not_allocate_ordinal_or_credits() -> None:
    module = benchmark()
    result, samples, payload_bytes, _ = module._run_saturation_once(calls=4)
    assert len(samples) == 4
    assert payload_bytes > 0
    assert result["droppedCapacity"] == 4
    assert result["admitted"] == result["otherDisposition"] == 0


def test_benchmark_does_not_change_process_or_system_timer_resolution() -> None:
    module = benchmark()
    assert not hasattr(module, "_windows_timer_resolution_1ms")
    source = inspect.getsource(module.run_benchmark)
    assert "timeBeginPeriod" not in source


def test_ticker_and_producers_share_one_measurement_start_event() -> None:
    module = benchmark()
    null_source = inspect.getsource(module._run_null_lag_once)
    healthy_source = inspect.getsource(module._run_healthy_once)
    assert "ticker_start" not in null_source
    assert '"start": start' in null_source
    assert "ticker_start" not in healthy_source
    assert '"start": timed_start' in healthy_source


def test_late_paced_call_proceeds_without_an_artificial_zero_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark()
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: 2_000)

    def forbidden_sleep(_seconds: float) -> None:
        raise AssertionError("late paced call yielded artificially")

    monkeypatch.setattr(module.time, "sleep", forbidden_sleep)
    module._pace_call(local_index=0, local_count=32, window_start_ns=1_000, window_ns=500)


def test_late_paced_calls_yield_only_at_the_fixed_guard_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark()
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: 2_000)
    sleeps: list[float] = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    module._pace_call(local_index=7, local_count=32, window_start_ns=1_000, window_ns=500)

    assert sleeps == [0]


def test_ticker_signals_ready_before_waiting_for_measurement_start() -> None:
    module = benchmark()
    stop = threading.Event()
    start = threading.Event()
    ready = threading.Event()
    samples: list[int] = []
    ticker = threading.Thread(
        target=module._ticker,
        args=(stop, samples),
        kwargs={
            "minimum": 1,
            "interval_ns": 100_000_000,
            "start": start,
            "ready": ready,
            "epoch_ns": [1],
        },
    )
    ticker.start()
    assert ready.wait(0.5)
    assert ticker.is_alive()
    stop.set()
    ticker.join(0.5)
    assert not ticker.is_alive()
    assert samples == []


def test_ticker_deadline_is_anchored_to_measurement_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark()
    stop = threading.Event()
    start = threading.Event()
    start.set()

    class Samples(list[int]):
        def append(self, value: int) -> None:
            super().append(value)
            stop.set()

    samples = Samples()
    clock = iter((1_100, 1_150, 1_150))
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    module._ticker(
        stop,
        samples,
        minimum=1,
        interval_ns=100,
        start=start,
        epoch_ns=[1_000],
    )

    assert samples == [50]


def test_ticker_never_appends_after_measurement_window_closes() -> None:
    module = benchmark()
    stop = threading.Event()
    stop.set()
    samples: list[int] = []
    module._ticker(stop, samples, minimum=3, interval_ns=1)
    assert samples == []


def test_ticker_closure_interrupts_an_in_progress_interval_without_a_sample() -> None:
    module = benchmark()
    stop = threading.Event()
    samples: list[int] = []
    ticker = threading.Thread(
        target=module._ticker,
        args=(stop, samples),
        kwargs={"minimum": 3, "interval_ns": 100_000_000},
    )
    ticker.start()
    assert not stop.wait(0.01)
    stop.set()
    ticker.join(0.5)
    assert not ticker.is_alive()
    assert samples == []


@pytest.mark.parametrize("fault", ["admit", "complete"])
def test_healthy_scheduler_fault_fails_closed_without_hanging(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    module = benchmark()
    from hermes_realtime.evidence import admission

    def broken(*_: object) -> object:
        raise RuntimeError(f"forced {fault} fault")

    if fault == "admit":
        monkeypatch.setattr(admission._ProductionEvidenceSchedulerV1, "try_admit", broken)
    else:
        monkeypatch.setattr(admission._ProductionEvidenceSchedulerV1, "complete", broken)
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            module._run_healthy_once(
                producers=1,
                calls=100,
                warmups=0,
                ticker_minimum=5,
                ticker_interval_ns=100_000,
            )
        except BaseException as error:
            failures.append(error)

    runner = threading.Thread(target=invoke, daemon=True)
    runner.start()
    runner.join(2.0)
    assert not runner.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)


def test_runner_aggregates_repetition_maxima_and_lag_subtraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark()
    counter = iter(range(1, 10))

    def null(**_: object) -> int:
        return 3

    def healthy(**_: object) -> tuple[dict[str, int], tuple[int, ...], tuple[int, str]]:
        value = next(counter)
        return (
            {
                **healthy_repetition(),
                "p99Ns": value,
                "maxNs": value + 1,
                "healthyEventLoopLagP99Ns": value + 5,
            },
            (value,) * 4,
            (1, "a" * 64),
        )

    monkeypatch.setattr(module, "_run_null_lag_once", null)
    monkeypatch.setattr(module, "_run_healthy_once", healthy)
    monkeypatch.setattr(module, "_allocation_bytes_per_call", lambda _: 0)
    monkeypatch.setattr(module, "_run_saturation_once", lambda *, calls: (
        saturation_repetition(), (1,) * calls, 1, "a" * 64
    ))
    profiles, _, _, _ = module.run_benchmark(
        repetitions=1, warmups=0, healthy_calls=4, saturation_calls=1,
        ticker_minimum=1, ticker_interval_ns=1, allocation_calls=1,
    )
    assert [profile["p99Ns"] for profile in profiles] == [1, 2, 3]
    assert [profile["eventLoopLagRegressionP99Ns"] for profile in profiles] == [3, 4, 5]
