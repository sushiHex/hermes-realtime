import os
from pathlib import Path
from runpy import run_path

import pytest


def _release_workflow() -> str:
    return (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-gates.yml"
    ).read_text(encoding="utf-8")


def _logical_requirements(path: Path) -> list[str]:
    logical: list[str] = []
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current += line.removesuffix("\\").strip()
        if not line.endswith("\\"):
            logical.append(current)
            current = ""
    assert not current
    return logical


def test_cuda_worker_installation_is_hash_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "setup-kokoro-cuda-worker.sh").read_text(encoding="utf-8")
    requirements = (
        root / "requirements" / "kokoro-cuda-worker-win-py311.txt",
        root / "requirements" / "kokoro-onnx-package-win-py311.txt",
    )

    assert script.count("--require-hashes") == 2
    for path in requirements:
        assert path.is_file()
        entries = _logical_requirements(path)
        assert entries
        assert all("--hash=sha256:" in entry for entry in entries)


def test_native_release_gate_rejects_short_hmac_key_warnings() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "release_gate.py").read_text(
        encoding="utf-8"
    )

    assert '"error::jwt.warnings.InsecureKeyLengthWarning"' in script


def test_browser_bundle_legal_notices_are_release_gated() -> None:
    root = Path(__file__).resolve().parents[1]
    build_script = (root / "web" / "build.mjs").read_text(encoding="utf-8")
    release_gate = (root / "scripts" / "release_gate.py").read_text(encoding="utf-8")

    assert 'legalComments: "external"' in build_script
    assert "metafile: true" in build_script
    assert '"package-lock.json"' in build_script
    assert '"assets/app.js.LEGAL.txt"' in build_script
    assert '"hermes_realtime/client/static/assets/app.js.LEGAL.txt"' in release_gate


def test_native_release_gate_runs_the_synthetic_full_host_audio_tracer() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "release_gate.py").read_text(
        encoding="utf-8"
    )
    livekit_block = script.split("def check_livekit(", maxsplit=1)[1].split(
        "def run_script_mypy(", maxsplit=1
    )[0]

    assert livekit_block.count(
        '"tests/integration/test_qualification_full_host_synthetic_audio.py"'
    ) == 1


def test_browser_owned_livekit_verifies_the_listener_process_id() -> None:
    source = (
        Path(__file__).resolve().parent / "integration" / "test_browser_self_acceptance.py"
    ).read_text(encoding="utf-8")

    assert "def _livekit_listener_pid() -> int:" in source
    assert "_livekit_listener_pid() != process.pid" in source


def test_release_gate_delegates_to_one_materialized_candidate_path() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "release_gate.py").read_text(
        encoding="utf-8"
    )

    assert "def gate_materialized_candidate(" in script
    legacy_gate = script.split("def gate(\n", maxsplit=1)[1].split(
        "\ndef expect_failure", maxsplit=1
    )[0]
    assert legacy_gate.count("gate_materialized_candidate(") == 1


def test_release_gate_runs_real_speech_presence_calibration_extra() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "release_gate.py").read_text(
        encoding="utf-8"
    )

    assert '"speech-verification"' in script
    assert '"tests/providers/test_speech_presence.py"' in script
    assert 'speech_verification_env["HERMES_RELEASE_SPEECH_VERIFICATION"] = "1"' in script


def test_every_gate_suite_records_per_test_durations() -> None:
    # Issue #13: a five-second timeout only proves the bound was exceeded. The
    # durations of the runs that *passed* are the baseline a recurrence is
    # measured against, so they must be captured on every run rather than
    # switched on after a third failure.
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "release_gate.py").read_text(encoding="utf-8")
    release_gate = run_path(str(root / "scripts" / "release_gate.py"))

    assert release_gate["PYTEST_DURATIONS"] == ("--durations=25",)
    assert release_gate["VITEST_DURATIONS"] == ("--", "--reporter=verbose")

    # Every pytest invocation the gate owns, and the browser suite, splat the
    # shared constants; a new suite added without them fails here.
    assert script.count('"pytest",\n        "-q",\n        *PYTEST_DURATIONS,') == 3
    assert script.count("*VITEST_DURATIONS") == 1
    assert script.count('run(NPM, "test", *VITEST_DURATIONS,') == 1


def test_release_workflow_uses_reviewed_node24_action_pins() -> None:
    workflow = _release_workflow()
    node24_pins = (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",  # v7.0.1
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",  # v7.0.0
        "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",  # v7.0.0
        "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",  # v10.0.1
    )

    expected_uses = (4, 4, 2, 3)
    for pin, expected_count in zip(node24_pins, expected_uses, strict=True):
        assert workflow.count(f"uses: {pin}") == expected_count
    assert workflow.count("prune-cache: true") == 3
    assert workflow.count('version: "0.11.28"') == 3

    setup_uv = "      - uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d\n"
    hermetic_setup_uv = setup_uv + (
        "        with:\n"
        "          version: \"0.11.28\"\n"
        "          enable-cache: true\n"
        "          prune-cache: true\n"
    )
    native_setup_uv = hermetic_setup_uv + "          save-cache: false\n"
    hermetic_job = workflow.split("  release-candidate:", maxsplit=1)[1].split(
        "  native-livekit:", maxsplit=1
    )[0]
    native_job = workflow.split("  native-livekit:", maxsplit=1)[1]
    assert hermetic_job.count(hermetic_setup_uv) == 1
    assert "save-cache: false" not in hermetic_job
    assert native_job.count(native_setup_uv) == 1
    assert native_job.count("save-cache: false") == 1


def test_windows_release_jobs_provision_owner_only_temp_roots() -> None:
    workflow = _release_workflow()
    windows_jobs = (
        workflow.split("  release-candidate:", maxsplit=1)[1].split(
            "  native-livekit:", maxsplit=1
        )[0],
        workflow.split("  native-livekit:", maxsplit=1)[1],
    )

    for job in windows_jobs:
        provision_marker = "      - name: Provision owner-only Windows test temp"
        cleanup_marker = "      - name: Remove owner-only Windows test temp"
        assert job.count(provision_marker) == 1
        assert job.count(cleanup_marker) == 1

        provision = job.split(provision_marker, maxsplit=1)[1].split(
            "\n      - name:", maxsplit=1
        )[0]
        assert "$env:RUNNER_TEMP" in provision
        assert "[Security.Principal.WindowsPrincipal]" in provision
        assert "[Security.Principal.WindowsBuiltInRole]::Administrator" in provision
        assert "$authoritySid = 'S-1-5-32-544'" in provision
        assert "icacls.exe" in provision
        assert "/inheritance:r" in provision
        assert "/setowner" in provision
        assert "/grant:r" in provision
        assert '"*${authoritySid}:(OI)(CI)F"' in provision
        assert '"*${authoritySid}"' in provision
        assert '$probe = Join-Path $testTemp "owner-inheritance-probe"' in provision
        assert "_path_grants_only_owner" in provision
        assert "$testTemp $probe" in provision
        assert "Remove-Item -LiteralPath $probe" in provision
        assert provision.index("/inheritance:r") < provision.index("/grant:r")
        assert provision.index("/grant:r") < provision.index("/setowner")
        assert provision.index("/setowner") < provision.index(
            "New-Item -ItemType Directory -Path $probe"
        )
        assert provision.index("New-Item -ItemType Directory -Path $probe") < provision.index(
            "_path_grants_only_owner"
        )
        assert '"TEMP=$testTemp"' in provision
        assert '"TMP=$testTemp"' in provision
        assert "HERMES_OWNER_ONLY_TEST_TEMP" not in provision

        cleanup = job.split(cleanup_marker, maxsplit=1)[1].split(
            "\n      - name:", maxsplit=1
        )[0]
        assert "if: always()" in cleanup
        assert (
            '$testTemp = Join-Path $env:RUNNER_TEMP '
            '"hermes-realtime-owner-only-$env:GITHUB_JOB"'
        ) in cleanup
        assert "Remove-Item -LiteralPath $testTemp" in cleanup
        assert "HERMES_OWNER_ONLY_TEST_TEMP" not in cleanup


def test_windows_release_jobs_run_the_candidate_e2e_fast_track() -> None:
    workflow = _release_workflow()
    hermetic_job = workflow.split("  release-candidate:", maxsplit=1)[1].split(
        "  native-livekit:", maxsplit=1
    )[0]
    native_job = workflow.split("  native-livekit:", maxsplit=1)[1]

    assert hermetic_job.count("python scripts/candidate_e2e_fast_track.py") == 1
    assert native_job.count("python scripts/candidate_e2e_fast_track.py") == 1
    assert "--candidate $env:GITHUB_WORKSPACE" in hermetic_job
    assert "--candidate $env:GITHUB_WORKSPACE" in native_job
    assert "--require-livekit" in native_job
    assert "--livekit-executable $env:LIVEKIT_SERVER" in native_job
    assert "--livekit-executable-sha256 $env:LIVEKIT_SERVER_SHA256" in native_job
    assert "--livekit-pid $process.Id" in native_job
    assert "python scripts/release_gate.py" not in hermetic_job
    assert "python scripts/release_gate.py" not in native_job


def test_native_livekit_process_lifetime_covers_release_gate() -> None:
    workflow = _release_workflow()
    native_job = workflow.split("  native-livekit:", maxsplit=1)[1]
    marker = "      - name: Run native LiveKit integration gate"
    assert native_job.count(marker) == 1
    gate_step = native_job.split(marker, maxsplit=1)[1].split("\n      - name:", maxsplit=1)[0]

    process_assignment = "$process = Start-Process"
    cleanup = "Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue"
    assert gate_step.count(process_assignment) == 1
    assert "-PassThru" in gate_step.split(process_assignment, maxsplit=1)[1].splitlines()[0]
    assert "$env:LIVEKIT_API_KEY = 'dev' + 'key'" in gate_step
    assert "$env:LIVEKIT_API_SECRET = 'local' + '-' + ('x' * 32)" in gate_step
    assert '$env:LIVEKIT_KEYS = "${env:LIVEKIT_API_KEY}: ${env:LIVEKIT_API_SECRET}`n"' in gate_step

    start_index = gate_step.index("Start-Process")
    try_index = gate_step.index("try {")
    readiness_index = gate_step.index("http://127.0.0.1:7880/")
    gate_index = gate_step.index("scripts/candidate_e2e_fast_track.py")
    finally_index = gate_step.index("} finally {")
    stop_index = gate_step.index("Stop-Process")

    assert start_index < try_index < readiness_index < gate_index < finally_index < stop_index
    try_body, finally_body = gate_step.split("} finally {", maxsplit=1)
    assert "scripts/candidate_e2e_fast_track.py" in try_body
    assert try_body.count("try {") >= 2
    assert "catch {}" in try_body
    assert "Start-Sleep -Seconds 1" in try_body
    assert try_body.count("if ($LASTEXITCODE -ne 0)") == 1
    assert 'throw "Native release gate failed with exit code $LASTEXITCODE"' in try_body
    assert "Stop-Process" not in try_body
    assert finally_body.count("Stop-Process") == 1
    assert cleanup in finally_body


def test_native_job_runs_browser_self_acceptance_with_fresh_owned_livekit() -> None:
    workflow = _release_workflow()
    native_job = workflow.split("  native-livekit:", maxsplit=1)[1]
    native_marker = "      - name: Run native LiveKit integration gate"
    browser_marker = "      - name: Run real-browser self-acceptance gate"
    cleanup_marker = "      - name: Remove owner-only Windows test temp"

    assert native_job.count(browser_marker) == 1
    assert native_job.index(native_marker) < native_job.index(browser_marker) < native_job.index(
        cleanup_marker
    )
    browser_step = native_job.split(browser_marker, maxsplit=1)[1].split(
        "\n      - name:", maxsplit=1
    )[0]
    assert "$env:HERMES_REALTIME_BROWSER_SELF_ACCEPTANCE = '1'" in browser_step
    assert "$env:HERMES_REALTIME_BROWSER_LIVEKIT_SERVER = $env:LIVEKIT_SERVER" in browser_step
    assert (
        "$env:HERMES_REALTIME_BROWSER_LIVEKIT_SERVER_SHA256 = "
        "$env:LIVEKIT_SERVER_SHA256"
    ) in browser_step
    assert "uv run --frozen --group dev --extra browser-acceptance pytest -q" in browser_step
    assert "tests/integration/test_browser_self_acceptance.py" in browser_step


def test_browser_self_acceptance_accepts_only_explicit_livekit_executable_override() -> None:
    source = (
        Path(__file__).resolve().parent / "integration" / "test_browser_self_acceptance.py"
    ).read_text(encoding="utf-8")

    assert 'os.environ.get("HERMES_REALTIME_BROWSER_LIVEKIT_SERVER")' in source
    assert 'os.environ.get("HERMES_REALTIME_BROWSER_LIVEKIT_SERVER_SHA256", "")' in source
    assert "Path(__file__).parents[2] / \".tools/livekit/livekit-server.exe\"" in source


def test_native_livekit_gate_binds_the_verified_executable_and_owned_listener() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = _release_workflow()
    native_job = workflow.split("  native-livekit:", maxsplit=1)[1]
    script = (root / "scripts" / "release_gate.py").read_text(encoding="utf-8")

    assert "--livekit-executable" in script
    assert "--livekit-executable-sha256" in script
    assert "--livekit-pid" in script
    assert "LiveKit executable hash does not match the verified value" in script
    assert "LiveKit signaling listener is not owned by the supplied process" in script
    assert "Get-NetTCPConnection -LocalPort 7880 -State Listen" in native_job
    assert "$env:LIVEKIT_SERVER_SHA256" in native_job
    assert "--livekit-executable $env:LIVEKIT_SERVER" in native_job
    assert "--livekit-executable-sha256 $env:LIVEKIT_SERVER_SHA256" in native_job
    assert "--livekit-pid $process.Id" in native_job


def test_linux_null_capture_consumes_the_single_hash_identified_candidate_wheel() -> None:
    workflow = _release_workflow()
    candidate = workflow.split("  candidate-wheel:", maxsplit=1)[1].split(
        "  linux-null-capture:", maxsplit=1
    )[0]
    linux = workflow.split("  linux-null-capture:", maxsplit=1)[1].split(
        "  release-candidate:", maxsplit=1
    )[0]

    assert "runs-on: ubuntu-24.04" in candidate
    assert "uv build --wheel --out-dir candidate" in candidate
    assert "candidate/candidate-wheel.sha256" in candidate
    assert "requirements/linux-null-capture.txt" in candidate
    assert "--group dev" in candidate
    assert "wheelhouse/" in candidate
    assert "needs: candidate-wheel" in linux
    assert "sha256sum --check candidate/candidate-wheel.sha256" in linux
    assert "--no-index" in linux
    assert "--require-hashes" in linux
    assert 'PATH="$PWD/candidate-input/runtime/bin:$PATH"' in linux
    assert "tests/integration/test_linux_null_capture.py" in linux


def test_committed_test_source_contains_no_credential_shaped_literals() -> None:
    root = Path(__file__).resolve().parents[1]
    release_gate = run_path(str(root / "scripts" / "release_gate.py"))
    test_path = root / "tests" / "evidence" / "test_sqlite_spool.py"

    findings = release_gate["secret_findings"](
        test_path.relative_to(root).as_posix(),
        test_path.read_bytes(),
    )

    assert findings == []


def test_release_gate_registers_crash_worker_as_required_sdist_resource() -> None:
    root = Path(__file__).resolve().parents[1]
    release_gate = run_path(str(root / "scripts" / "release_gate.py"))

    assert "tests/evidence/spool_crash_worker.py" in release_gate["required_sdist_paths"]()


def test_release_gate_registers_task11_benchmark_and_native_full_host_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "release_gate.py").read_text(encoding="utf-8")
    release_gate = run_path(str(root / "scripts" / "release_gate.py"))

    required = release_gate["required_sdist_paths"]()
    assert "scripts/benchmark_evidence_admission.py" in required
    assert "scripts/schemas/benchmark-machine-v1.schema.json" in required
    assert "scripts/schemas/benchmark-report-v1.schema.json" in required
    assert '"scripts/benchmark_evidence_admission.py"' in script
    assert '"tests/integration/test_qualification_full_host_ingress.py"' in script


def test_release_gate_registers_all_task12_schemas_and_runners_exactly() -> None:
    root = Path(__file__).resolve().parents[1]
    release_gate_path = root / "scripts" / "release_gate.py"
    source = release_gate_path.read_text(encoding="utf-8")
    release_gate = run_path(str(release_gate_path))
    required = release_gate["required_sdist_paths"]()
    task12_paths = {
        "scripts/qualify_evidence_slice_zero.py",
        "scripts/qualify_hermes_v020_pluginmanager.py",
        "scripts/schemas/benchmark-machine-v1.schema.json",
        "scripts/schemas/benchmark-report-v1.schema.json",
        "scripts/schemas/wheelhouse-manifest-v1.schema.json",
        "scripts/schemas/qualification-input-v1.schema.json",
        "scripts/schemas/qualification-report-v1.schema.json",
        "scripts/schemas/release-manifest-v1.schema.json",
    }

    assert task12_paths <= required
    assert all((root / relative).is_file() for relative in task12_paths)
    typecheck_block = source.split("script_type_env =", 1)[1].split("cwd=root", 1)[0]
    for runner in (
        '"scripts/qualify_evidence_slice_zero.py"',
        '"scripts/qualify_hermes_v020_pluginmanager.py"',
    ):
        assert typecheck_block.count(runner) == 1


def test_release_gate_registers_shared_source_archive_authority_once_for_sdist_and_mypy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    release_gate = run_path(str(root / "scripts" / "release_gate.py"))
    registered = "scripts/source_archive_authority.py"
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def observe_run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> None:
        assert env is not None
        calls.append((command, cwd, env))

    monkeypatch.setitem(release_gate["run_script_mypy"].__globals__, "run", observe_run)
    release_gate["run_script_mypy"](root, {"BASE": "retained"})

    assert tuple(release_gate["required_sdist_paths"]()).count(registered) == 1
    assert len(calls) == 1
    command, cwd, environment = calls[0]
    assert cwd == root
    assert command[:6] == ("uv", "run", "--frozen", "--group", "dev", "mypy")
    assert command.count(registered) == 1
    assert environment["BASE"] == "retained"
    assert environment["MYPYPATH"] == os.pathsep.join((str(root / "src"), str(root / "scripts")))


def test_release_gate_registers_task13_orchestrator_for_sdist_and_script_mypy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    release_gate = run_path(str(root / "scripts" / "release_gate.py"))
    registered = "scripts/task13_artifact_orchestrator.py"
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def observe_run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> None:
        assert env is not None
        calls.append((command, cwd, env))

    monkeypatch.setitem(release_gate["run_script_mypy"].__globals__, "run", observe_run)

    release_gate["run_script_mypy"](root, {"BASE": "retained"})

    assert tuple(release_gate["required_sdist_paths"]()).count(registered) == 1
    assert len(calls) == 1
    command, cwd, environment = calls[0]
    assert cwd == root
    assert command[:6] == ("uv", "run", "--frozen", "--group", "dev", "mypy")
    assert "--follow-imports=skip" in command
    assert command.count(registered) == 1
    assert environment["BASE"] == "retained"
    assert environment["MYPYPATH"] == os.pathsep.join((str(root / "src"), str(root / "scripts")))


def test_release_gate_registers_fast_track_for_sdist_and_script_mypy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    release_gate = run_path(str(root / "scripts" / "release_gate.py"))
    registered = "scripts/candidate_e2e_fast_track.py"
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def observe_run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> None:
        assert env is not None
        calls.append((command, cwd, env))

    monkeypatch.setitem(release_gate["run_script_mypy"].__globals__, "run", observe_run)
    release_gate["run_script_mypy"](root, {"BASE": "retained"})

    assert tuple(release_gate["required_sdist_paths"]()).count(registered) == 1
    assert len(calls) == 1
    command, cwd, environment = calls[0]
    assert cwd == root
    assert command.count(registered) == 1
    assert environment["BASE"] == "retained"


def test_canonical_baseline_diff_uses_exact_no_ext_diff_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    release_gate = run_path(str(root / "scripts" / "release_gate.py"))
    head = "a" * 40
    baseline = "b" * 40
    calls: list[tuple[str, ...]] = []

    class Result:
        def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    def observe(command: tuple[str, ...], **_: object) -> Result:
        calls.append(command)
        if command[1:3] == ("rev-parse", "--verify"):
            return Result((head if command[3] == "HEAD^{commit}" else baseline).encode() + b"\n")
        if command[1] == "merge-base":
            return Result()
        if command[1] == "diff":
            return Result(b"canonical patch")
        raise AssertionError(command)

    monkeypatch.setattr(release_gate["subprocess"], "run", observe)
    release_gate["canonical_baseline_diff_sha256"](root, baseline)

    assert calls[-1] == (
        "git",
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        f"{baseline}..{head}",
    )
