from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import qualification_launch as launch


def _probe_launch_capability() -> str:
    # Independent of production guardian code: a successful explicit breakaway
    # can still leave a process in a more restrictive ancestor Job.
    program = (
        "import ctypes,json\n"
        "k=ctypes.WinDLL('kernel32',use_last_error=True)\n"
        "k.GetCurrentProcess.restype=ctypes.c_void_p\n"
        "k.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]\n"
        "k.IsProcessInJob.restype=ctypes.c_int\nf=ctypes.c_int()\n"
        "if not k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(f)):"
        "raise OSError('probe_membership_failed')\n"
        "print(json.dumps(not bool(f.value)))\n"
    )
    try:
        result = subprocess.run(
            (sys._base_executable, "-B", "-c", program),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
            creationflags=0x1000000 | 0x200 | 0x8000000,
        )
    except OSError as error:
        if getattr(error, "winerror", None) == 5:
            return "breakaway_not_permitted"
        raise
    assert result.returncode == 0, "independent_breakaway_probe_failed"
    assert len(result.stdout) <= 16, "independent_breakaway_probe_output"
    outside = json.loads(result.stdout)
    assert type(outside) is bool, "independent_breakaway_probe_shape"
    return "outside" if outside else "guardian_inherited_job"


def _assert_host_refusal(receipt: Path, capability: str) -> None:
    rows = [json.loads(row) for row in receipt.read_bytes().splitlines()]
    final = rows[-1]
    assert [row["stage"] for row in rows] == [
        "reserved",
        "refused" if capability == "breakaway_not_permitted" else "incomplete",
    ]
    assert final["category"] == capability
    assert final["controller_completed"] is False
    assert final["cleanup_verified"] is (None if capability == "breakaway_not_permitted" else False)


@pytest.fixture(scope="module")
def launch_capability(tmp_path_factory: pytest.TempPathFactory) -> str:
    capability = _probe_launch_capability()
    print("[qualification-launch-capability] " + json.dumps({"capability": capability}))
    if capability != "outside":
        folder = tmp_path_factory.mktemp("verified-host-refusal")
        marker = folder / "controller-started"
        script = folder / "controller.py"
        script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
        binding, digest = _binding(folder)
        receipt = folder / "receipt.jsonl"
        with pytest.raises(RuntimeError):
            launch.run_owned_controller(
                script,
                (),
                binding_path=binding,
                binding_sha256=digest,
                receipt=receipt,
                timeout_seconds=10,
            )
        _assert_host_refusal(receipt, capability)
        assert not marker.exists(), "constrained_host_executed_controller"
    return capability


@pytest.fixture
def native_completion_host(launch_capability: str) -> None:
    if launch_capability != "outside":
        pytest.skip("Verified host refusal; controller-dependent native fault not exercised")


@pytest.mark.parametrize(
    "observation,expected",
    [(b"true", "outside"), (b"false", "guardian_inherited_job"), (5, "breakaway_not_permitted")],
)
def test_independent_capability_probe_has_only_closed_host_outcomes(
    monkeypatch: pytest.MonkeyPatch, observation: bytes | int, expected: str
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert args[0][0] == sys._base_executable
        assert kwargs["creationflags"] == 0x1000000 | 0x200 | 0x8000000
        assert kwargs["timeout"] == 10 and kwargs["stdin"] is subprocess.DEVNULL
        if type(observation) is int:
            error = OSError("synthetic")
            error.winerror = observation
            raise error
        return subprocess.CompletedProcess(args[0], 0, observation, b"")

    monkeypatch.setattr(subprocess, "run", run)
    assert _probe_launch_capability() == expected


@pytest.mark.parametrize(
    "observation", [b'"false"', b"{}", b"null", b"x", b"nonzero", b" " * 16 + b"true", 87]
)
def test_capability_probe_unknown_failure_never_becomes_constrained(
    monkeypatch: pytest.MonkeyPatch, observation: bytes | int
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if type(observation) is int:
            error = OSError("synthetic")
            error.winerror = observation
            raise error
        return subprocess.CompletedProcess(
            args[0],
            1 if observation == b"nonzero" else 0,
            b"true" if observation == b"nonzero" else observation,
            b"",
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises((AssertionError, ValueError, OSError)):
        _probe_launch_capability()


def _receipt_fields(path: Path) -> list[dict[str, object]]:
    stages = {
        "reserved",
        "started",
        "complete",
        "incomplete",
        "refused",
        "observer_timeout",
        "observer_interrupted",
    }
    categories = {
        "complete",
        "refused",
        "guardian_timeout",
        "interrupted",
        "breakaway_not_permitted",
        "guardian_spawn_failed",
        "guardian_inherited_job",
        "timed_out",
        "invalid_input",
        "execution_failed",
    }
    try:
        with path.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            return [{"stage": "unreadable"}]
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) > 8 or any(type(row) is not dict for row in rows):
            return [{"stage": "unreadable"}]
        result = []
        for row in rows:
            safe: dict[str, object] = {}
            for key, allowed in (("stage", stages), ("category", categories)):
                if key in row:
                    value = row[key]
                    safe[key] = value if type(value) is str and value in allowed else "unrecognized"
            for key in ("controller_completed", "cleanup_verified"):
                if key in row:
                    safe[key] = row[key] if type(row[key]) is bool else None
            result.append(safe)
        return result
    except (OSError, ValueError):
        return [{"stage": "unreadable"}]


@pytest.fixture(autouse=True)
def _show_closed_receipt_fields(tmp_path: Path) -> Iterator[None]:
    yield
    # Pytest shows captured teardown output on failure; paths and controller
    # output remain local. No receipt is interpreted as qualification success.
    for path in sorted(tmp_path.glob("*.jsonl"))[:4]:
        print("[qualification-launch-test] " + json.dumps(_receipt_fields(path)))


@pytest.mark.parametrize(
    "fault", ["none", "private_values", "private_flags", "oversize", "rows", "shape", "malformed"]
)
def test_receipt_diagnostics_keep_only_bounded_closed_fields(tmp_path: Path, fault: str) -> None:
    receipt = tmp_path / "receipt.jsonl"
    row = {
        "stage": "incomplete",
        "category": "execution_failed",
        "controller_completed": False,
        "cleanup_verified": False,
        "private_output": "PRIVATE_SENTINEL",
        "pid": 12345,
    }
    if fault == "private_values":
        row.update(stage="PRIVATE_SENTINEL", category="PRIVATE_SENTINEL")
    if fault == "private_flags":
        row.update(controller_completed="PRIVATE_SENTINEL", cleanup_verified="PRIVATE_SENTINEL")
    receipt.write_bytes(
        b" " * 65535 + b"{}"
        if fault == "oversize"
        else (json.dumps(row) + "\n").encode() * 9
        if fault == "rows"
        else b"[]\n"
        if fault == "shape"
        else b"{"
        if fault == "malformed"
        else (json.dumps(row) + "\n").encode()
    )
    actual = _receipt_fields(receipt)
    if fault in {"oversize", "rows", "shape", "malformed"}:
        assert actual == [{"stage": "unreadable"}]
    else:
        assert actual == [
            {
                "stage": "unrecognized" if fault == "private_values" else "incomplete",
                "category": "unrecognized" if fault == "private_values" else "execution_failed",
                "controller_completed": None if fault == "private_flags" else False,
                "cleanup_verified": None if fault == "private_flags" else False,
            }
        ]
    assert "PRIVATE_SENTINEL" not in json.dumps(actual)


def test_receipt_diagnostics_read_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = []

    class Stream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: Stream(b"{}"))
    assert _receipt_fields(tmp_path / "receipt.jsonl") == [{}]
    assert requested == [65537]


def _binding(tmp_path: Path) -> tuple[Path, str]:
    def image(path: str) -> dict[str, str]:
        return {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

    raw = json.dumps(
        {
            "schema_version": 1,
            "guardian": image(sys._base_executable),
            "controller": image(sys.executable),
        }
    ).encode()
    path = tmp_path / "interpreters.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_interpreter_roles_are_explicit_and_default_selection_is_shared(tmp_path: Path) -> None:
    path, digest = _binding(tmp_path)
    binding = launch.load_binding(path, digest)
    assert launch.guardian_command(binding)[0] == sys._base_executable
    assert binding.controller == sys.executable


@pytest.mark.parametrize(
    "change", ["digest", "image_digest", "null_sha", "unknown", "base", "controller", "version"]
)
def test_interpreter_binding_refuses_drift(tmp_path: Path, change: str) -> None:
    path, digest = _binding(tmp_path)
    if change != "digest":
        value = json.loads(path.read_bytes())
        if change == "null_sha":
            value["guardian"]["sha256"] = None
        elif change == "image_digest":
            value["guardian"]["sha256"] = "0" * 64
        elif change == "version":
            value["schema_version"] = True
        elif change == "unknown":
            value["unknown"] = True
        else:
            copy = tmp_path / "another.exe"
            copy.write_bytes(Path(sys._base_executable).read_bytes())
            value["guardian" if change == "base" else "controller"] = {
                "path": str(copy),
                "sha256": hashlib.sha256(copy.read_bytes()).hexdigest(),
            }
        path.write_text(json.dumps(value))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        digest = "0" * 64
    with pytest.raises(ValueError):
        launch.load_binding(path, digest)


def test_missing_completion_is_not_success() -> None:
    assert launch.completion_status(None, cleanup_verified=True) == "incomplete"
    assert (
        launch.completion_status({"normal_exit": True, "exit_code": 0}, cleanup_verified=False)
        == "incomplete"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "output",
        "stderr",
        "late_stderr",
        "cleanup_stderr",
        "late_stdout",
        "cleanup_completion",
        "exit",
        "wait",
        "deadline",
        "late_deadline",
        "late_cleanup_zero",
        "query_type",
        "active",
        "terminate",
        "cleanup_wait",
        "close",
        "kernel_binding",
    ],
)
def test_native_owner_retains_failure_and_requires_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from types import SimpleNamespace

    from scripts import qualify_evidence_slice_zero as core

    events = []
    clock = [0]

    class Kernel:
        def __init__(self) -> None:
            def exit_code(handle: object, code: object) -> int:
                code._obj.value = 1 if fault == "exit" else 0
                return 1

            def waited(*args: object) -> int:
                if fault == "late_deadline":
                    clock[0] = 20
                if fault == "late_stderr":
                    (tmp_path / "stderr").write_bytes(b"x" * 65537)
                if fault == "late_stdout":
                    (tmp_path / "stdout").write_bytes(b"xxxx")
                return 1 if fault == "wait" else 0

            self.api = SimpleNamespace(
                WaitForSingleObject=waited,
                GetExitCodeProcess=exit_code,
            )

        def _api(self) -> object:
            return self.api

        def create_job(self, limits: object) -> int:
            events.append("job")
            return 10

        def create_process_suspended(
            self,
            command: object,
            environment: object,
            cwd: object,
            handles: object,
            flags: int,
            *,
            job: int,
        ) -> object:
            assert job == 10 and flags & 0x8000000 and flags & 0x4
            events.append("atomic_create")
            if fault == "stderr":
                (tmp_path / "stderr").write_bytes(b"x" * 65537)
            return SimpleNamespace(process_handle=20, thread_handle=21)

        def resume_thread(self, thread: int) -> None:
            events.append("resume")

        def terminate_job(self, job: int) -> None:
            events.append("terminate")
            if fault == "terminate":
                raise RuntimeError("synthetic")

        def wait(self, process: int, timeout: int) -> None:
            events.append("wait")
            if fault == "cleanup_wait":
                raise RuntimeError("synthetic")

        def query_job_active_process_count(self, job: int) -> int:
            events.append("query")
            if fault == "late_cleanup_zero":
                clock[0] = 40
            if fault == "cleanup_stderr":
                (tmp_path / "stderr").write_bytes(b"x" * 65537)
            if fault == "cleanup_completion":
                payload = b'{"normal_exit":true,"exit_code":0}'
                (tmp_path / "completion").write_bytes(b" " * (65537 - len(payload)) + payload)
            return False if fault == "query_type" else 1 if fault == "active" else 0

        def close_handle(self, handle: int) -> None:
            events.append("close")
            if fault == "close":
                raise RuntimeError("synthetic")

    monkeypatch.setattr(core, "_CtypesWindowsKernelV1", Kernel)
    if os.name != "nt":
        monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: fd))
    kernel_path = Path(core.__file__).resolve()
    plan = {
        "timeout_seconds": -1 if fault == "deadline" else 10,
        "work": str(tmp_path),
        "plan_path": str(tmp_path / "plan"),
        "cwd": str(tmp_path),
        "completion": str(tmp_path / "completion"),
        "kernel_path": str(kernel_path),
        "kernel_sha256": "0" * 64
        if fault == "kernel_binding"
        else hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "completion").write_text('{"normal_exit":true,"exit_code":0}')
    if fault == "output":
        monkeypatch.setattr(launch, "_MAX_OUTPUT", -1)
    if fault == "late_stdout":
        monkeypatch.setattr(launch, "_MAX_OUTPUT", 3)
    evidence = {}
    if fault in {"late_deadline", "late_cleanup_zero"}:
        monkeypatch.setattr(
            launch, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None)
        )
    if fault == "active":
        times = iter([0, 0, 0, 0, 100])
        monkeypatch.setattr(
            launch, "time", SimpleNamespace(monotonic=lambda: next(times), sleep=lambda _: None)
        )
    if fault == "none":
        stdout, completion, clean = launch._owned_controller(
            plan, launch.InterpreterBinding("base", "venv", "x", "y"), evidence
        )
        assert stdout == b"" and completion["normal_exit"] is True and clean is True
        assert events == [
            "job",
            "atomic_create",
            "resume",
            "terminate",
            "wait",
            "query",
            "close",
            "close",
            "close",
        ]
    else:
        with pytest.raises((BaseExceptionGroup, ValueError)) as caught:
            launch._owned_controller(
                plan, launch.InterpreterBinding("base", "venv", "x", "y"), evidence
            )
        assert evidence.get("cleanup_verified", False) is (
            fault
            in {
                "output",
                "stderr",
                "late_stderr",
                "cleanup_stderr",
                "late_stdout",
                "cleanup_completion",
                "exit",
                "wait",
                "deadline",
                "late_deadline",
            }
        )
        primary = (
            caught.value.exceptions[0]
            if isinstance(caught.value, BaseExceptionGroup)
            else caught.value
        )
        expected = {
            "output": "launch_output_bound",
            "stderr": "launch_output_bound",
            "late_stderr": "launch_input_bound",
            "cleanup_stderr": "launch_input_bound",
            "late_stdout": "launch_input_bound",
            "cleanup_completion": "launch_input_bound",
            "exit": "launch_controller_exit",
            "wait": "launch_controller_wait",
            "deadline": "launch_controller_deadline",
            "late_deadline": "launch_controller_deadline",
            "late_cleanup_zero": "launch_cleanup_deadline",
            "query_type": "launch_cleanup_count",
            "active": "launch_cleanup_deadline",
            "kernel_binding": "launch_kernel_binding",
            "terminate": "synthetic",
            "cleanup_wait": "synthetic",
            "close": "synthetic",
        }
        assert str(primary) == expected[fault]
        if fault in {
            "deadline",
            "late_deadline",
            "late_cleanup_zero",
            "active",
            "exit",
            "cleanup_wait",
        }:
            # Feed the actual error group raised above by the real owner with
            # a fake kernel into the guardian's durable classifier.
            original_error = caught.value
            if fault == "active":
                # A cleanup deadline can be nested beside another failure;
                # timeout classification must not inspect only the first leaf.
                original_error = BaseExceptionGroup(
                    "nested cleanup deadline", [RuntimeError("synthetic primary"), original_error]
                )

            def owned(plan: object, binding: object, final: dict[str, object]) -> object:
                final.update(evidence)
                raise original_error

            monkeypatch.setattr(launch, "_outside_jobs", lambda: True)
            monkeypatch.setattr(
                launch,
                "load_binding",
                lambda *a, **k: launch.InterpreterBinding("b", "v", "x", "y"),
            )
            monkeypatch.setattr(launch, "_owned_controller", owned)
            raw = json.dumps(
                {
                    "receipt": str(tmp_path / "guardian.jsonl"),
                    "binding": "unused",
                    "binding_sha256": "unused",
                    "launcher_sha256": hashlib.sha256(
                        Path(launch.__file__).read_bytes()
                    ).hexdigest(),
                }
            ).encode()
            guardian_plan = tmp_path / "guardian-plan.json"
            guardian_plan.write_bytes(raw)
            assert launch._guardian(guardian_plan, hashlib.sha256(raw).hexdigest()) == 2
            final = json.loads((tmp_path / "guardian.jsonl").read_bytes().splitlines()[-1])
            assert final["category"] == (
                "timed_out"
                if fault in {"deadline", "late_deadline", "late_cleanup_zero", "active"}
                else "execution_failed"
            )
            assert final["stage"] == "incomplete" and final["controller_completed"] is False
            assert final["cleanup_verified"] is evidence["cleanup_verified"]
            if fault == "deadline":
                original_error = primary
                assert launch._guardian(guardian_plan, hashlib.sha256(raw).hexdigest()) == 2
                final = json.loads((tmp_path / "guardian.jsonl").read_bytes().splitlines()[-1])
                assert final["category"] == "timed_out" and final["cleanup_verified"] is True
        if fault in {"terminate", "cleanup_wait", "query_type", "active", "late_cleanup_zero"}:
            assert "close" not in events


def test_completion_requires_normal_exact_exit_code() -> None:
    assert (
        launch.completion_status({"normal_exit": True, "exit_code": False}, cleanup_verified=True)
        == "incomplete"
    )
    assert (
        launch.completion_status({"normal_exit": True, "exit_code": 1}, cleanup_verified=True)
        == "complete"
    )
    assert (
        launch.completion_status({"normal_exit": False, "exit_code": 0}, cleanup_verified=True)
        == "incomplete"
    )


@pytest.mark.parametrize("kind", ["duplicate", "oversize"])
def test_bounded_strict_binding_input(tmp_path: Path, kind: str) -> None:
    valid, _ = _binding(tmp_path)
    raw = valid.read_bytes()
    if kind == "duplicate":
        raw = raw.replace(b'"schema_version": 1', b'"schema_version": 1,"schema_version": 1')
    else:
        raw += b" " * (65537 - len(raw))
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        launch.load_binding(path, hashlib.sha256(raw).hexdigest())


def test_guardian_refuses_inherited_job_before_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt"
    plan = tmp_path / "plan"
    raw = json.dumps({"receipt": str(receipt)}).encode()
    plan.write_bytes(raw)
    monkeypatch.setattr(launch, "_outside_jobs", lambda: False)
    monkeypatch.setattr(
        launch, "load_binding", lambda *a, **k: pytest.fail("controller admission reached")
    )
    assert launch._guardian(plan, hashlib.sha256(raw).hexdigest()) == 2
    final = json.loads(receipt.read_bytes())
    assert final["category"] == "guardian_inherited_job" and final["controller_completed"] is False


@pytest.mark.parametrize("fault", ["none", "existing", "bound", "short"])
def test_lifecycle_record_is_exclusive_bounded_and_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    calls = []
    monkeypatch.setattr(launch.os, "fsync", lambda fd: calls.append(fd))
    path = tmp_path / "receipt"
    if fault == "short":

        class Sink:
            def __enter__(self) -> object:
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def write(self, raw: bytes) -> int:
                return 0

            def fileno(self) -> int:
                return 1

        class Destination:
            def open(self, *args: object, **kwargs: object) -> Sink:
                return Sink()

        with pytest.raises(OSError):
            launch._record(Destination(), {"stage": "test"}, reserve=True)
        assert calls == []
    elif fault == "bound":
        with pytest.raises(ValueError):
            launch._record(path, {"data": "x" * 65537}, reserve=True)
        assert not path.exists()
    else:
        launch._record(path, {"stage": "start"}, reserve=True)
        if fault == "existing":
            with pytest.raises(FileExistsError):
                launch._record(path, {"stage": "again"}, reserve=True)
        else:
            launch._record(path, {"stage": "end"})
        assert len(calls) == (1 if fault == "existing" else 2)


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job ownership")
def test_default_launch_uses_base_guardian_and_explicit_controller(
    tmp_path: Path, launch_capability: str
) -> None:
    binding, digest = _binding(tmp_path)
    script = tmp_path / "controller.py"
    marker = tmp_path / "controller-started"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "print(json.dumps({'pass':False,'venv':sys.prefix!=sys.base_prefix}))\n"
        "raise SystemExit(1)\n"
    )
    receipt = tmp_path / "receipt.jsonl"
    if launch_capability != "outside":
        with pytest.raises(RuntimeError):
            launch.run_owned_controller(
                script,
                (),
                binding_path=binding,
                binding_sha256=digest,
                receipt=receipt,
                timeout_seconds=30,
            )
        _assert_host_refusal(receipt, launch_capability)
        assert not marker.exists()
        return
    result = launch.run_owned_controller(
        script, (), binding_path=binding, binding_sha256=digest, receipt=receipt, timeout_seconds=30
    )
    assert result.exit_code == 1
    assert marker.exists()
    assert json.loads(result.stdout)["pass"] is False
    rows = [json.loads(row) for row in receipt.read_bytes().splitlines()]
    assert [row["stage"] for row in rows] == ["reserved", "started", "complete"]
    assert rows[-1]["cleanup_verified"] is True
    assert rows[-1]["controller_completed"] is True
    assert (
        rows[-1]["guardian_sha256"]
        == hashlib.sha256(Path(sys._base_executable).read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        launch.run_owned_controller(
            script,
            (),
            binding_path=binding,
            binding_sha256=digest,
            receipt=receipt,
            timeout_seconds=30,
        )


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job ownership")
def test_controller_death_retains_incomplete_receipt(
    tmp_path: Path, native_completion_host: None
) -> None:
    binding, digest = _binding(tmp_path)
    script = tmp_path / "controller.py"
    script.write_text("import os\nos._exit(0)\n")
    receipt = tmp_path / "receipt.jsonl"
    with pytest.raises(RuntimeError):
        launch.run_owned_controller(
            script,
            (),
            binding_path=binding,
            binding_sha256=digest,
            receipt=receipt,
            timeout_seconds=30,
        )
    final = json.loads(receipt.read_bytes().splitlines()[-1])
    assert final["stage"] == "incomplete"
    assert final["controller_completed"] is False
    assert final["cleanup_verified"] is True


def test_evaluator_consumes_owned_launch_without_changing_model_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import evaluate_natural_work_routing as evaluator

    seen = []

    def run(script: Path, args: tuple[str, ...], **kwargs: object) -> launch.LaunchResult:
        seen.append((script, args, kwargs))
        return launch.LaunchResult(1, b'{"pass":false}\n')

    monkeypatch.setattr(launch, "run_owned_controller", run)
    monkeypatch.setattr(evaluator, "_verify_controller_source", lambda: None)
    binding, digest = _binding(tmp_path)
    code = evaluator.main(
        [
            "--model",
            "synthetic",
            "--launch-binding",
            str(binding),
            "--launch-binding-sha256",
            digest,
            "--launch-receipt",
            str(tmp_path / "receipt.jsonl"),
        ]
    )
    assert code == 1 and json.loads(capsys.readouterr().out) == {"pass": False}
    assert seen[0][0].resolve() == Path(evaluator.__file__).resolve()
    assert seen[0][1] == (
        "--corpus",
        str(evaluator._DEFAULT_CORPUS),
        "--model",
        "synthetic",
        "--effort",
        "low",
    )


def test_legacy_cli_does_not_claim_owned_launch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import evaluate_natural_work_routing as evaluator

    async def evaluate(**kwargs: object) -> dict[str, object]:
        return {"pass": True}

    monkeypatch.setattr(evaluator, "run_evaluation", evaluate)
    monkeypatch.setattr(evaluator, "_verify_controller_source", lambda: None)
    monkeypatch.setattr(
        launch, "run_owned_controller", lambda *a, **k: pytest.fail("legacy invoked owner")
    )
    assert evaluator.main([]) == 0
    assert "owned" not in capsys.readouterr().out


@pytest.mark.skipif(os.name != "nt", reason="native Windows default CLI launch")
def test_actual_evaluator_opted_in_cli_uses_default_owner_without_model(
    tmp_path: Path, launch_capability: str
) -> None:
    binding, digest = _binding(tmp_path)
    receipt = tmp_path / "cli-receipt.jsonl"
    root = Path(__file__).resolve().parents[1]
    wrong = tmp_path / "wrong"
    (wrong / "hermes_realtime").mkdir(parents=True)
    (wrong / "hermes_realtime/__init__.py").write_text(
        "raise RuntimeError('wrong source imported')"
    )
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(wrong), str(root))))
    # A deliberately absent executable refuses before any provider subprocess.
    # This checks the real opted-in CLI and dependency refusal.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_natural_work_routing",
            "--launch-binding",
            str(binding),
            "--launch-binding-sha256",
            digest,
            "--launch-receipt",
            str(receipt),
            "--codex-executable",
            str(tmp_path / "absent.exe"),
        ],
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=30,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["pass"] is False
    if launch_capability != "outside":
        assert json.loads(completed.stdout)["error"]["code"] == "evaluation_failed"
        _assert_host_refusal(receipt, launch_capability)
        return
    assert json.loads(completed.stdout)["error"]["code"] == "codex_setup_failed"
    rows = [json.loads(row) for row in receipt.read_bytes().splitlines()]
    assert rows[-1]["stage"] == "complete" and rows[-1]["exit_code"] == 2
    assert rows[-1]["cleanup_verified"] is True


def test_evaluator_refuses_already_loaded_wrong_source_before_evaluation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    reached = tmp_path / "evaluation-reached"
    program = (
        f"import sys;sys.path[:0]=[{str(root / 'src')!r},{str(root)!r}]\n"
        "from scripts import evaluate_natural_work_routing as e\n"
        "from hermes_realtime.providers import codex_app_server as c\n"
        "from pathlib import Path\nc.__file__='wrong-source.py'\n"
        "async def reached(**kwargs):\n"
        f" Path({str(reached)!r}).write_text('unexpected')\n raise AssertionError()\n"
        "e.run_evaluation=reached\nraise SystemExit(e.main([]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert result.returncode == 2 and not reached.exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows boundary faults")
@pytest.mark.parametrize(
    "fault", ["plan", "launcher", "script", "result", "result_size", "breakaway"]
)
def test_launch_rejects_bound_artifact_or_spawn_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    capsys: pytest.CaptureFixture[str],
    request: pytest.FixtureRequest,
) -> None:
    if fault in {"launcher", "script", "result", "result_size"}:
        request.getfixturevalue("native_completion_host")
    binding, digest = _binding(tmp_path)
    script = tmp_path / "controller.py"
    script.write_text("print('harmless')\n")
    receipt = tmp_path / "receipt.jsonl"
    original = subprocess.Popen

    def popen(command: tuple[str, ...], **kwargs: object) -> subprocess.Popen[bytes]:
        if fault == "breakaway":
            error = OSError("synthetic")
            error.winerror = 5
            raise error
        args = list(command)
        path = Path(args[-2])
        raw = path.read_bytes()
        plan = json.loads(raw)
        if fault == "plan":
            path.write_bytes(raw + b" ")
        if fault == "launcher":
            plan["launcher_sha256"] = "0" * 64
            path.write_text(json.dumps(plan))
            args[-1] = hashlib.sha256(path.read_bytes()).hexdigest()
        if fault == "script":
            script.write_text("print('changed')\n")
        process = original(args, **kwargs)
        if fault in {"result", "result_size"}:
            wait = process.wait

            def changed(*args: object, **kwargs: object) -> int:
                code = wait(*args, **kwargs)
                if fault == "result":
                    target = Path(plan["result"])
                    before = target.read_bytes()
                    target.write_bytes(bytes([before[0] ^ 1]) + before[1:])
                else:
                    rows = [json.loads(row) for row in receipt.read_bytes().splitlines()]
                    rows[-1]["stdout_bytes"] = 0
                    receipt.write_text("".join(json.dumps(row) + "\n" for row in rows))
                return code

            process.wait = changed
        return process

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    with pytest.raises((RuntimeError, ValueError)):
        launch.run_owned_controller(
            script,
            (),
            binding_path=binding,
            binding_sha256=digest,
            receipt=receipt,
            timeout_seconds=20,
        )
    if fault == "breakaway":
        final = json.loads(receipt.read_bytes().splitlines()[-1])
        assert final["stage"] == "refused" and final["category"] == "breakaway_not_permitted"
        assert "breakaway_not_permitted" in capsys.readouterr().err
    elif fault == "plan":
        # The digest check precedes guardian capability admission. A refusal
        # from a later guard must not masquerade as this check being reached.
        assert [json.loads(row)["stage"] for row in receipt.read_bytes().splitlines()] == [
            "reserved"
        ]


@pytest.mark.parametrize(
    "timeout,args", [(False, ()), (0, ()), (14401, ()), (1, ["not_tuple"]), (1, ("\0",))]
)
def test_launch_bounds_fail_before_creation(tmp_path: Path, timeout: object, args: object) -> None:
    with pytest.raises(ValueError):
        launch.run_owned_controller(
            tmp_path / "absent",
            args,
            binding_path=tmp_path / "absent",
            binding_sha256="0" * 64,
            receipt=tmp_path / "receipt",
            timeout_seconds=timeout,
        )
    assert not (tmp_path / "receipt").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher watchdog contract")
@pytest.mark.parametrize("fault", ["none", "kill", "wait", "both", "record"])
def test_guardian_watchdog_persists_timeout_even_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fault: str
) -> None:
    binding, digest = _binding(tmp_path)
    script = tmp_path / "controller.py"
    script.write_text("print('never executed')\n")
    receipt = tmp_path / "receipt.jsonl"
    expired = subprocess.TimeoutExpired("synthetic", 65)
    events = []

    class Process:
        def wait(self, *, timeout: int) -> int:
            events.append(("wait", timeout))
            if len(events) == 1:
                raise expired
            if fault in {"wait", "both"}:
                raise OSError("synthetic wait failure")
            return 0

        def kill(self) -> None:
            events.append(("kill", None))
            if fault in {"kill", "both"}:
                raise OSError("synthetic kill failure")

    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: Process())
    record = launch._record

    def recorded(path: Path, value: dict[str, object], **kwargs: object) -> None:
        if fault == "record" and value.get("stage") == "observer_timeout":
            raise OSError("synthetic record failure")
        record(path, value, **kwargs)

    monkeypatch.setattr(launch, "_record", recorded)
    with pytest.raises((TimeoutError, BaseExceptionGroup)) as caught:
        launch.run_owned_controller(
            script,
            (),
            binding_path=binding,
            binding_sha256=digest,
            receipt=receipt,
            timeout_seconds=5,
        )
    assert events == [("wait", 65), ("kill", None), ("wait", 30)]
    if fault == "none":
        assert isinstance(caught.value, TimeoutError) and caught.value.__cause__ is expired
    else:
        assert isinstance(caught.value, BaseExceptionGroup)
        assert caught.value.exceptions[0] is expired
        assert len(caught.value.exceptions) == (3 if fault == "both" else 2)
    marker = capsys.readouterr().err.split("[qualification-launch] ")[-1]
    assert json.loads(marker)["category"] == "guardian_timeout"
    rows = [json.loads(row) for row in receipt.read_bytes().splitlines()]
    if fault == "record":
        assert rows[-1]["stage"] == "reserved"
    else:
        assert rows[-1]["stage"] == "observer_timeout"
        assert rows[-1]["category"] == "guardian_timeout"
        assert rows[-1]["cleanup_verified"] is None and rows[-1]["controller_completed"] is False


@pytest.mark.skipif(os.name != "nt", reason="native Windows caller-interruption proof")
def test_caller_interruption_keeps_guardian_inputs_until_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_completion_host: None
) -> None:
    binding, digest = _binding(tmp_path)
    script = tmp_path / "short.py"
    script.write_text("import time\ntime.sleep(.3)\nprint('harmless')\n")
    receipt = tmp_path / "interrupted.jsonl"
    observed = []
    original = subprocess.Popen

    def popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = original(*args, **kwargs)
        observed.append(process)
        process.wait = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
        return process

    try:
        with monkeypatch.context() as patch:
            patch.setattr(launch.subprocess, "Popen", popen)
            with pytest.raises(KeyboardInterrupt):
                launch.run_owned_controller(
                    script,
                    (),
                    binding_path=binding,
                    binding_sha256=digest,
                    receipt=receipt,
                    timeout_seconds=20,
                )
        assert (receipt.with_name(receipt.name + ".work") / "plan.json").is_file()
    finally:
        for process in observed:
            original.wait(process, timeout=25)
    final = json.loads(receipt.read_bytes().splitlines()[-1])
    assert final["stage"] == "complete" and final["cleanup_verified"] is True
    assert (
        receipt.with_name(receipt.name + ".work") / "result"
    ).read_bytes().strip() == b"harmless"


@pytest.mark.skipif(os.name != "nt", reason="native Windows ownership fault injection")
def test_launcher_loss_preserves_guardian_then_guardian_loss_exits_descendants(
    tmp_path: Path,
    native_completion_host: None,
) -> None:
    import ctypes

    from scripts import qualify_evidence_slice_zero as core

    binding, digest = _binding(tmp_path)
    root = Path(__file__).resolve().parents[1]
    ready, guardian_ready = tmp_path / "children.json", tmp_path / "guardian.json"
    script = tmp_path / "controller.py"
    script.write_text(
        "import json,os,subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys._base_executable,'-c','import time;time.sleep(60)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,"
        "creationflags=subprocess.CREATE_NO_WINDOW)\n"
        f"p=Path({str(ready)!r}); t=p.with_suffix('.tmp')\n"
        "t.write_text(json.dumps([os.getpid(),child.pid])); t.replace(p)\ntime.sleep(60)\n"
    )
    helper = tmp_path / "release.py"
    helper.write_text(
        "import json,subprocess,sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0,{str(root)!r})\nfrom scripts import qualification_launch as q\n"
        "original=subprocess.Popen\ndef observed(*args,**kwargs):\n"
        " p=original(*args,**kwargs)\n"
        f" f=Path({str(guardian_ready)!r}); t=f.with_suffix('.tmp')\n"
        " t.write_text(json.dumps(p.pid)); t.replace(f); return p\n"
        "subprocess.Popen=observed\n"
        f"q.run_owned_controller(Path({str(script)!r}),(),binding_path=Path({str(binding)!r}),"
        f"binding_sha256={digest!r},"
        f"receipt=Path({str(tmp_path / 'fault.jsonl')!r}),timeout_seconds=45)\n"
    )
    kernel = core._CtypesWindowsKernelV1()
    job = kernel.create_job(core._WindowsJobLimitsV1(8, 1024**3, 2 * 1024**3))
    api = kernel._api()
    info = core._JOBOBJECT_EXTENDED_LIMIT_INFORMATION_V1()
    assert api.QueryInformationJobObject(
        ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info), None
    )
    info.BasicLimitInformation.LimitFlags |= 0x800  # explicit, not silent, breakaway
    assert api.SetInformationJobObject(
        ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info)
    )
    handles = []
    release = None
    try:
        import msvcrt

        with open(os.devnull, "r+b") as null:
            os.set_inheritable(null.fileno(), True)
            environment = tuple(
                sorted(os.environ.items(), key=lambda item: (item[0].casefold(), item[0]))
            )
            release = kernel.create_process_suspended(
                (sys.executable, "-B", str(helper)),
                environment,
                str(root),
                (msvcrt.get_osfhandle(null.fileno()),),
                0x4 | 0x400 | 0x80000 | 0x8000000,
                job=job,
            )
            kernel.resume_thread(release.thread_handle)
            until = time.monotonic() + 15
            while not ready.exists() or not guardian_ready.exists():
                assert time.monotonic() < until, "harmless controller did not start"
                time.sleep(0.02)
            pids = [json.loads(guardian_ready.read_bytes()), *json.loads(ready.read_bytes())]
            handles = [kernel.open_process(pid, 0x101001) for pid in pids]
            kernel.terminate_job(job)
            kernel.wait(release.process_handle, 5000)
            assert kernel.query_job_active_process_count(job) == 0
            assert all(
                api.WaitForSingleObject(ctypes.c_void_p(handle), 100) == 258 for handle in handles
            )
            kernel.terminate_process(handles[0])
            for handle in handles:
                kernel.wait(handle, 5000)
            rows = [json.loads(row) for row in (tmp_path / "fault.jsonl").read_bytes().splitlines()]
            assert rows[-1]["stage"] == "started"  # no invented completion/cleanup
    finally:
        if handles and api.WaitForSingleObject(ctypes.c_void_p(handles[0]), 0) == 258:
            kernel.terminate_process(handles[0])
        kernel.terminate_job(job)
        for handle in handles:
            kernel.close_handle(handle)
        if release is not None:
            kernel.close_handle(release.thread_handle)
            kernel.close_handle(release.process_handle)
        kernel.close_handle(job)
