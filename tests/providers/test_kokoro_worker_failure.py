"""Fixed worker diagnostics with fake dependencies only; never load a model."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from hermes_realtime.providers import kokoro_worker as worker

CATEGORIES = {
    80: "authentication_failed", 81: "parent_watchdog_failed",
    82: "asset_attestation_failed", 83: "dependency_import_failed",
    84: "dll_preload_failed", 85: "cuda_session_failed", 86: "cuda_provider_rejected",
    87: "kokoro_engine_failed", 88: "attestation_synthesis_failed",
    89: "cuda_profile_unavailable_or_invalid", 90: "cuda_profile_missing_core_ops",
    91: "cuda_profile_forbidden_cpu_ops", 92: "loopback_connection_failed",
    93: "worker_service_failed",
}
PRIVATE_DETAIL = "SYNTHETIC_PRIVATE_TOKEN_PATH_AND_STDERR"


def install_fake_dependencies(monkeypatch, tmp_path, fault=None, error_type=RuntimeError):
    def fail():
        raise error_type(PRIVATE_DETAIL)

    model, voices = tmp_path / "model", tmp_path / "voices"
    model.write_bytes(b"synthetic model")
    voices.write_bytes(b"synthetic voices")
    args = SimpleNamespace(
        model=str(model), voices=str(voices), parent_pid=123, connect_port=12345,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        voices_sha256=hashlib.sha256(voices.read_bytes()).hexdigest(),
        attestation_voice="bf_isabella", language="en-gb", max_pcm_bytes=1024,
    )
    profile = tmp_path / "profile.json"
    cuda_ops = ["Conv", "Gemm", "LSTM"] if fault != "missing_core" else ["Conv", "Gemm"]
    events = [{"args": {"provider": "CUDAExecutionProvider", "op_name": name}}
              for name in cuda_ops]
    if fault == "cpu_core":
        events.append({"args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}})
    profile.write_text(json.dumps(events), encoding="utf-8")
    calls = []

    def watchdog(parent):
        calls.append("watchdog")
        if fault == "watchdog":
            fail()

    def preload(**kwargs):
        calls.append("preload")
        assert kwargs == {"directory": ""}
        if fault == "preload":
            fail()

    class Session:
        def __init__(self, *args, **kwargs):
            calls.append("session")
            assert kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if fault == "session":
                fail()

        def get_providers(self):
            return (["CPUExecutionProvider"] if fault == "provider"
                    else ["CUDAExecutionProvider", "CPUExecutionProvider"])

        def end_profiling(self):
            if fault == "profile":
                fail()
            return str(profile)

    class Engine:
        def create(self, *args, **kwargs):
            calls.append("attestation")
            if fault == "attestation":
                fail()
            return [0.1], 24000

    def from_session(*args):
        calls.append("engine")
        if fault == "engine":
            fail()
        return Engine()

    def dependency(name):
        calls.append(name)
        if fault == name:
            fail()
        return {
            "onnxruntime": SimpleNamespace(
                preload_dlls=preload, SessionOptions=SimpleNamespace,
                ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
                InferenceSession=Session,
            ),
            "kokoro_onnx": SimpleNamespace(Kokoro=SimpleNamespace(from_session=from_session)),
            "numpy": SimpleNamespace(asarray=lambda data: SimpleNamespace(size=1)),
        }[name]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, timeout):
            assert timeout is None

    def connect(*args, **kwargs):
        calls.append("connect")
        if fault == "connect":
            fail()
        return Connection()

    def serve(**kwargs):
        calls.append("serve")
        assert kwargs["actual_providers"] == ("CUDAExecutionProvider", "CPUExecutionProvider")
        assert kwargs["profiled_cuda_events"] == 3
        if fault == "serve":
            fail()

    monkeypatch.setattr(worker, "_worker_argument_parser", lambda: SimpleNamespace(
        parse_args=lambda: args,
    ))
    monkeypatch.setattr(worker, "_start_parent_watchdog", watchdog)
    monkeypatch.setattr(worker, "importlib", SimpleNamespace(import_module=dependency))
    monkeypatch.setattr(worker, "socket", SimpleNamespace(create_connection=connect))
    monkeypatch.setattr(worker, "serve_kokoro_worker_connection", serve)
    monkeypatch.setenv(worker._TOKEN_ENV, "synthetic-one-use-token")
    if fault == "authentication":
        monkeypatch.delenv(worker._TOKEN_ENV)
    if fault == "asset":
        args.model_sha256 = "0" * 64
    return calls


@pytest.mark.parametrize("fault,code", (
    ("authentication", 80), ("watchdog", 81), ("asset", 82), ("onnxruntime", 83),
    ("kokoro_onnx", 83), ("numpy", 83), ("preload", 84), ("session", 85),
    ("provider", 86), ("engine", 87), ("attestation", 88), ("profile", 89),
    ("missing_core", 90), ("cpu_core", 91), ("connect", 92), ("serve", 93),
))
def test_worker_failure_returns_fixed_stage_without_dependency_details(
    fault, code, monkeypatch, tmp_path, capsys,
):
    install_fake_dependencies(monkeypatch, tmp_path, fault)
    assert worker.main() == code
    output = capsys.readouterr()
    assert output.out == ""
    assert len(output.err) <= 256 and output.err.count("\n") == 1
    assert output.err.startswith("[kokoro-worker-failure] ")
    payload = json.loads(output.err.removeprefix("[kokoro-worker-failure] "))
    assert payload == {"version": 1, "category": CATEGORIES[code], "kind": "RuntimeError"}
    assert PRIVATE_DETAIL not in output.err


def test_worker_success_retains_original_operation_order_and_emits_no_failure(
    monkeypatch, tmp_path, capsys,
):
    calls = install_fake_dependencies(monkeypatch, tmp_path)
    assert worker.main() == 0
    assert calls == ["watchdog", "onnxruntime", "preload", "kokoro_onnx", "session",
                     "engine", "numpy", "attestation", "connect", "serve"]
    assert capsys.readouterr() == ("", "")


def test_unknown_exception_kind_and_args_are_never_serialized(monkeypatch, tmp_path, capsys):
    class SyntheticPrivateClass(Exception):
        def __str__(self):
            raise AssertionError("exception text must never be inspected")

    install_fake_dependencies(monkeypatch, tmp_path, "engine", SyntheticPrivateClass)
    assert worker.main() == 87
    output = capsys.readouterr()
    payload = json.loads(output.err.removeprefix("[kokoro-worker-failure] "))
    assert payload == {"version": 1, "category": CATEGORIES[87], "kind": "unknown_exception"}
    assert "SyntheticPrivate" not in output.err and PRIVATE_DETAIL not in output.err


def test_broken_failure_marker_stream_cannot_replace_worker_exit_code(monkeypatch, tmp_path):
    class BrokenStream:
        def write(self, text):
            raise OSError(PRIVATE_DETAIL)

        def flush(self):
            raise OSError(PRIVATE_DETAIL)

    install_fake_dependencies(monkeypatch, tmp_path, "session")
    monkeypatch.setattr(worker.sys, "stderr", BrokenStream())
    assert worker.main() == 85


@pytest.mark.parametrize("phase_code", (0, True, 94, "85"))
def test_invalid_internal_phase_cannot_report_success(phase_code, monkeypatch, capsys):
    monkeypatch.setattr(worker, "_worker_argument_parser", lambda: SimpleNamespace(
        parse_args=lambda: None,
    ))

    def fail(args, *, failure_phase):
        failure_phase.exit_code = phase_code
        raise RuntimeError(PRIVATE_DETAIL)

    monkeypatch.setattr(worker, "_run_cuda_worker", fail)
    assert worker.main() == 1
    assert json.loads(capsys.readouterr().err.removeprefix("[kokoro-worker-failure] ")) == {
        "version": 1, "category": "unknown_worker_failure", "kind": "RuntimeError",
    }


def test_system_exit_is_not_reclassified_as_a_worker_failure(monkeypatch, capsys):
    monkeypatch.setattr(worker, "_worker_argument_parser", lambda: SimpleNamespace(
        parse_args=lambda: None,
    ))

    def exit_directly(args, *, failure_phase):
        raise SystemExit(70)

    monkeypatch.setattr(worker, "_run_cuda_worker", exit_directly)
    with pytest.raises(SystemExit) as caught:
        worker.main()
    assert caught.value.code == 70
    assert capsys.readouterr() == ("", "")
