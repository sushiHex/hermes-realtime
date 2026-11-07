"""Model-free guard91 witness contract and startup race controls."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_realtime.providers import kokoro_worker as worker

MODEL = hashlib.sha256(b"model").hexdigest()
VOICES = hashlib.sha256(b"voices").hexdigest()
TOKEN = "synthetic-one-use-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    model, voices, script = (tmp_path / name for name in ("model", "voices", "worker.py"))
    model.write_bytes(b"model")
    voices.write_bytes(b"voices")
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(worker, "_PROFILE_OPERATION_CATALOGS", {
        MODEL: frozenset({"Conv", "Gemm", "LSTM", "Mul", "Add"}),
    })
    value = worker.KokoroWorkerClient(
        python_executable=Path(sys.executable), worker_script=script,
        model_path=model, voices_path=voices, expected_model_sha256=MODEL,
        expected_voices_sha256=VOICES, language="en-gb", max_pcm_bytes=1024,
    )
    yield value
    value.close()


def header(**changes):
    result = {"version": 1, "type": "startup_failure", "token": TOKEN,
              "modelSha256": MODEL, "voicesSha256": VOICES,
              "category": "cuda_profile_forbidden_cpu_ops", "cpuOperationTypes": ["Conv", "Mul"]}
    result.update(changes)
    return result


def test_valid_witness_is_complete_immutable_and_has_fixed_error_text(client):
    operations = client._validate_startup_failure(header(), TOKEN)
    error = worker.KokoroCpuPlacementRejected(operations, model_sha256=MODEL)
    assert operations == ("Conv", "Mul")
    assert error.cpu_operation_types == ("Conv", "Mul")
    assert error.args == ("Kokoro CUDA profile rejected CPU operation placement",)
    with pytest.raises(AttributeError):
        error.cpu_operation_types = ("Add",)
    assert client.actual_providers == () and client._socket is None


@pytest.mark.parametrize("changes", [
    {"token": "wrong"}, {"modelSha256": "0" * 64}, {"voicesSha256": "0" * 64},
    {"version": True}, {"version": 2}, {"type": "ready"}, {"category": "other"},
    {"extra": "SYNTHETIC_PRIVATE_DETAIL"}, {"cpuOperationTypes": []},
    {"cpuOperationTypes": ["Mul", "Conv"]}, {"cpuOperationTypes": ["Conv", "Conv"]},
    {"cpuOperationTypes": ["PRIVATE_NODE_NAME"]}, {"cpuOperationTypes": ["Cast"]},
    {"cpuOperationTypes": ["X" * 65]}, {"cpuOperationTypes": ["Conv"] * 65},
    {"cpuOperationTypes": "Conv"}, {"cpuOperationTypes": [True]},
])
def test_invalid_witness_is_unavailable_without_echo_or_readiness(client, changes):
    with pytest.raises(RuntimeError) as caught:
        client._validate_startup_failure(header(**changes), TOKEN)
    assert str(caught.value) == "Kokoro worker startup failure witness is unavailable"
    assert client.actual_providers == () and client._socket is None


def test_catalog_and_full_set_are_required(client, monkeypatch):
    validate = worker._validated_cpu_operation_types
    assert validate(["Conv", "Mul"], MODEL) == ("Conv", "Mul")
    assert validate(["Conv", "PRIVATE_NODE_NAME"], MODEL) is None
    assert validate(["Conv"], "0" * 64) is None
    monkeypatch.setattr(worker, "_PROFILE_OPERATION_CATALOGS", {
        MODEL: frozenset({"X" + str(i) for i in range(65)}),
    })
    assert validate(sorted("X" + str(i) for i in range(65)), MODEL) is None


def test_profile_rejection_retains_complete_cpu_type_set_without_allowlisted_ops():
    events = [{"args": {"provider": "CUDAExecutionProvider", "op_name": op}}
              for op in ("Conv", "Gemm", "LSTM")]
    events += [{"args": {"provider": "CPUExecutionProvider", "op_name": op}}
               for op in ("Mul", "Conv", "Cast", "Mul")]
    with pytest.raises(worker._CudaProfileFailure) as caught:
        worker._validate_cuda_profile_events(events)
    assert caught.value.exit_code == 91
    assert caught.value.cpu_operation_types == ("Conv", "Mul")


def test_startup_receive_keeps_original_deadline_across_fragments(monkeypatch):
    times = iter([1.0, 1.4, 1.8, 2.1])
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(times))
    class Trickle:
        def __init__(self):
            self.timeouts = []
            self.reads = 0
        def settimeout(self, value):
            self.timeouts.append(value)
        def recv(self, size):
            self.reads += 1
            return b"x"
    connection = Trickle()
    with pytest.raises(TimeoutError):
        worker.KokoroWorkerClient._receive_exact(connection, 4, deadline=2.0)
    assert connection.reads == 3
    assert connection.timeouts == pytest.approx([1.0, 0.6, 0.2])


def test_startup_failure_header_uses_unchanged_cleanup_and_never_publishes_ready(
    client, monkeypatch,
):
    run_start(client, monkeypatch, header(), exit_code=None)


def test_fast_exit91_drains_queued_witness(client, monkeypatch):
    run_start(client, monkeypatch, header(), exit_code=91)


def test_dead_child_ready_header_cannot_publish_readiness(client, monkeypatch):
    with pytest.raises(RuntimeError, match="exited before startup attestation"):
        run_start(client, monkeypatch, {
            "version": 1, "type": "ready", "token": TOKEN,
            "modelSha256": MODEL, "voicesSha256": VOICES,
            "provider": "CUDAExecutionProvider",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "executionPolicy": "cuda-primary-profiled", "profiledCudaEvents": 3,
            "profiledCpuEvents": 1, "sampleRateHz": 24000,
        }, exit_code=91, expect_witness=False)
    assert client.actual_providers == () and client._socket is None


def run_start(client, monkeypatch, message, *, exit_code, expect_witness=True):
    encoded = json.dumps(message).encode()
    class Connection:
        def __init__(self):
            self.data = struct.pack("!I", len(encoded)) + encoded
            self.closed = False
            self.timeouts = []
        def settimeout(self, value):
            self.timeouts.append(value)
        def recv(self, size):
            block, self.data = self.data[:size], self.data[size:]
            return block
        def close(self):
            self.closed = True
    connection = Connection()
    class Listener:
        def getsockname(self):
            return ("127.0.0.1", 12345)
        def settimeout(self, value):
            self.timeout = value
        def accept(self):
            return connection, ("127.0.0.1", 45678)
        def close(self):
            pass
    listener = Listener()
    process = SimpleNamespace(poll=lambda: exit_code, returncode=exit_code)
    settled = []
    monkeypatch.setattr(worker, "_create_exclusive_listener", lambda: listener)
    monkeypatch.setattr(worker.secrets, "token_urlsafe", lambda size: TOKEN)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(client, "_settle_process", lambda value: settled.append(value))
    if expect_witness:
        with pytest.raises(worker.KokoroCpuPlacementRejected) as caught:
            client.warm()
        assert caught.value.cpu_operation_types == ("Conv", "Mul")
    else:
        client.warm()
    assert connection.closed and settled == [process]
    assert client.actual_providers == () and client._socket is None
    if exit_code == 91:
        assert listener.timeout == 0.0


@pytest.mark.parametrize("value,digest", [(("PRIVATE_NODE_NAME",), MODEL), (("Conv",), "0" * 64),
                                        (("Mul", "Conv"), MODEL), (["Conv"], MODEL)])
def test_typed_error_cannot_be_constructed_with_unvalidated_data(client, value, digest):
    with pytest.raises(ValueError, match="witness is unavailable"):
        worker.KokoroCpuPlacementRejected(value, model_sha256=digest)


@pytest.mark.parametrize("invalid", [None, True, 123, ["Conv"]])
def test_missing_cpu_operation_metadata_makes_complete_witness_unavailable(invalid):
    events = [{"args": {"provider": "CUDAExecutionProvider", "op_name": op}}
              for op in ("Conv", "Gemm", "LSTM")]
    events += [{"args": {"provider": "CPUExecutionProvider", "op_name": op}}
               for op in ("Conv", invalid)]
    with pytest.raises(worker._CudaProfileFailure) as caught:
        worker._validate_cuda_profile_events(events)
    assert caught.value.exit_code == 91
    assert caught.value.cpu_operation_types == ()



def test_non_ascii_wire_token_has_fixed_unavailable_refusal(client):
    with pytest.raises(RuntimeError) as caught:
        client._validate_startup_failure(header(token="synthetic-\N{SNOWMAN}"), TOKEN)
    assert str(caught.value) == "Kokoro worker startup failure witness is unavailable"
