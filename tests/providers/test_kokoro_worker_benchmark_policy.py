"""Off-by-default model/runtime-bound policy; model-free contract proof."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from hermes_realtime.providers import kokoro_worker as worker

STRICT = "cuda-primary-profiled"
BENCHMARK = "cuda-primary-profiled-kokoro-v1-benchmark"
FOUR = frozenset({"ReduceProd", "Reshape", "Shape", "SplitToSequence"})
MODEL = hashlib.sha256(b"model").hexdigest()
VOICES = hashlib.sha256(b"voices").hexdigest()


def events(cpu=("ReduceProd", "Reshape", "Shape", "SplitToSequence")):
    return ([{"args": {"provider": "CUDAExecutionProvider", "op_name": op}}
             for op in ("Conv", "Gemm", "LSTM")]
            + [{"args": {"provider": "CPUExecutionProvider", "op_name": op}} for op in cpu])


def test_benchmark_cpu_union_is_exact_and_immutable_without_changing_strict():
    assert worker._BENCHMARK_ALLOWED_CPU_PROFILE_OPS == worker._ALLOWED_CPU_PROFILE_OPS | FOUR
    assert type(worker._BENCHMARK_ALLOWED_CPU_PROFILE_OPS) is frozenset
    assert not worker._ALLOWED_CPU_PROFILE_OPS & FOUR
    assert worker._validate_cuda_profile_events(events(), execution_policy=BENCHMARK) == (3, 4)
    with pytest.raises(worker._CudaProfileFailure) as caught:
        worker._validate_cuda_profile_events(events())
    assert caught.value.exit_code == 91


@pytest.mark.parametrize("operation", [None, True, 123, ""])
def test_benchmark_rejects_incomplete_cpu_metadata(operation):
    with pytest.raises(RuntimeError, match="CPU operation metadata is invalid"):
        worker._validate_cuda_profile_events(events((operation,)), execution_policy=BENCHMARK)


def test_strict_legacy_missing_metadata_behavior_is_preserved():
    assert worker._validate_cuda_profile_events(events((None,))) == (3, 0)


def test_benchmark_preserves_required_cuda_core_and_rejects_other_cpu_ops():
    with pytest.raises(worker._CudaProfileFailure) as caught:
        worker._validate_cuda_profile_events(events(("Conv",)), execution_policy=BENCHMARK)
    assert caught.value.exit_code == 91
    with pytest.raises(worker._CudaProfileFailure) as caught:
        worker._validate_cuda_profile_events(events()[1:], execution_policy=BENCHMARK)
    assert caught.value.exit_code == 90


@pytest.fixture
def client(tmp_path, monkeypatch):
    files = [tmp_path / name for name in ("model", "voices", "worker.py")]
    for path, data in zip(files, (b"model", b"voices", b""), strict=True):
        path.write_bytes(data)
    monkeypatch.setattr(worker, "_BENCHMARK_MODEL_SHA256", MODEL)
    def make(policy=STRICT, *, expected_model_sha256=MODEL):
        return worker.KokoroWorkerClient(
            python_executable=Path(sys.executable), worker_script=files[2],
            model_path=files[0], voices_path=files[1], expected_model_sha256=expected_model_sha256,
            expected_voices_sha256=VOICES, language="en-gb", max_pcm_bytes=1024,
            execution_policy=policy,
        )
    return make


def ready(policy):
    return {"version": 1, "type": "ready", "token": "synthetic-token",
            "modelSha256": MODEL, "voicesSha256": VOICES,
            "provider": "CUDAExecutionProvider",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "executionPolicy": policy, "profiledCudaEvents": 3, "profiledCpuEvents": 4,
            "sampleRateHz": 24000}


@pytest.mark.parametrize("expected,reported", [(STRICT, BENCHMARK), (BENCHMARK, STRICT)])
def test_ready_policy_requires_exact_expected_identity(client, expected, reported):
    instance = client(expected)
    with pytest.raises(RuntimeError, match="startup attestation"):
        instance._validate_ready(ready(reported), "synthetic-token")
    assert instance.actual_providers == ()


@pytest.mark.parametrize("policy", [STRICT, BENCHMARK])
def test_ready_accepts_selected_policy_identity_only(client, policy):
    instance = client(policy)
    assert instance._validate_ready(ready(policy), "synthetic-token") == (
        "CUDAExecutionProvider", "CPUExecutionProvider",
    )


@pytest.mark.parametrize("policy", ["other", "", True, None])
def test_client_cannot_select_unknown_policy(client, policy):
    with pytest.raises(ValueError, match="execution policy"):
        client(policy)


def test_parser_policy_default_strict_and_only_two_fixed_choices():
    parser = worker._worker_argument_parser()
    action = next(action for action in parser._actions if action.dest == "execution_policy")
    assert action.default == STRICT and set(action.choices) == {STRICT, BENCHMARK}


@pytest.mark.parametrize("policy,argc", [(STRICT, 21), (BENCHMARK, 23)])
def test_worker_argv_preserves_legacy_default_and_explicit_benchmark_only(
    client, monkeypatch, policy, argc,
):
    instance = client(policy)
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        raise OSError("synthetic stop before process launch")
    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="synthetic stop"):
        instance.warm()
    assert len(commands[0]) == argc
    if policy == STRICT:
        assert "--execution-policy" not in commands[0]
    else:
        assert commands[0][-2:] == ["--execution-policy", BENCHMARK]
    instance.close()


@pytest.mark.parametrize("policy", [STRICT, BENCHMARK])
def test_worker_ready_reports_actual_selected_policy(monkeypatch, policy):
    sent = []
    monkeypatch.setattr(worker.KokoroWorkerClient, "_send_json", staticmethod(
        lambda connection, value: sent.append(value),
    ))
    monkeypatch.setattr(worker.KokoroWorkerClient, "_receive_json", staticmethod(
        lambda connection: {"version": 1, "type": "close"},
    ))
    worker.serve_kokoro_worker_connection(
        connection=object(), token="synthetic-token", model_sha256=MODEL,
        voices_sha256=VOICES, actual_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        profiled_cuda_events=3, profiled_cpu_events=4, max_pcm_bytes=1024,
        synthesize_pcm=lambda *args: b"", execution_policy=policy,
    )
    assert sent[0]["executionPolicy"] == policy


def test_unknown_profile_policy_cannot_apply_a_placement_rule():
    with pytest.raises(RuntimeError, match="execution policy"):
        worker._validate_cuda_profile_events(events(), execution_policy="unknown")


@pytest.mark.parametrize("policy,model", [("unknown", MODEL), (BENCHMARK, "0" * 64)])
def test_new_constructor_policy_refusal_retains_fixed_marker(client, capsys, policy, model):
    with pytest.raises(ValueError, match="execution policy"):
        client(policy, expected_model_sha256=model)
    output = capsys.readouterr()
    assert output.out == "" and output.err.count("\n") == 1
    assert json.loads(output.err.removeprefix("[kokoro-startup-witness-refused] ")) == {
        "version": 1, "category": "policy_rejected",
    }
    assert MODEL not in output.err and "0" * 64 not in output.err
    assert "unknown" not in output.err


def test_ready_policy_mismatch_records_fixed_marker(client, capsys):
    instance = client(BENCHMARK)
    with pytest.raises(RuntimeError, match="startup attestation"):
        instance._validate_ready(ready(STRICT), "synthetic-token")
    output = capsys.readouterr()
    assert output.out == "" and output.err.count("\n") == 1
    assert json.loads(output.err.removeprefix("[kokoro-startup-witness-refused] ")) == {
        "version": 1, "category": "policy_rejected",
    }
    assert MODEL not in output.err and STRICT not in output.err


def test_valid_policy_paths_remain_silent(client, capsys):
    for policy in (STRICT, BENCHMARK):
        instance = client(policy)
        instance._validate_ready(ready(policy), "synthetic-token")
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("boundary", ["constructor", "ready"])
def test_policy_refusal_diagnostic_failure_preserves_original_error(client, monkeypatch, boundary):
    instance = client(BENCHMARK)
    class BrokenStream:
        def write(self, value):
            raise OSError("SYNTHETIC_PRIVATE_FAILURE")
        def flush(self):
            raise OSError("SYNTHETIC_PRIVATE_FAILURE")
    monkeypatch.setattr(worker.sys, "stderr", BrokenStream())
    if boundary == "constructor":
        with pytest.raises(ValueError, match="execution policy binding is unsupported"):
            client("unknown")
    else:
        with pytest.raises(RuntimeError, match="failed startup attestation"):
            instance._validate_ready(ready(STRICT), "synthetic-token")
    assert worker.sys.stderr is None
