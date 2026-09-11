#!/usr/bin/env python3
# ruff: noqa: E501
"""Hermetic release-candidate validation for hermes-realtime.

The gate deliberately archives the selected Git revision into a new temporary
checkout.  This prevents ignored build output, virtual environments, and other
ambient worktree directories from affecting a release artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

MAX_ARTIFACT_BYTES = 15 * 1024 * 1024
NPM = "npm.cmd" if os.name == "nt" else "npm"
# Per-test durations on every gate run, passing or failing. Timeout diagnostics
# are not a consistent source of comparable healthy-run timing: some restate
# only the configured authority, while others also report the elapsed failure
# duration. Capturing passing-run timings before a recurrence is the only way
# to have a baseline to compare against. These reach the retained job log,
# which is where a recurrence is investigated. See issue #13.
#
# The tail is uncapped, with the noise floor pinned explicitly. A fixed
# slowest-N ranks raw phase duration, and can therefore omit a lower-duration
# phase that sits behind a narrower internal timeout or other authority
# boundary. Retaining every phase above the floor keeps a later incident
# comparable with healthy runs.
PYTEST_DURATIONS = ("--durations=0", "--durations-min=0.005")
VITEST_DURATIONS = ("--", "--reporter=verbose", "--slowTestThreshold=100")
REQUIRED_STATIC = {
    "hermes_realtime/client/static/index.html": "web/index.html",
    "hermes_realtime/client/static/assets/app.js": "src/hermes_realtime/client/static/assets/app.js",
    "hermes_realtime/client/static/assets/app.js.LEGAL.txt": (
        "src/hermes_realtime/client/static/assets/app.js.LEGAL.txt"
    ),
    "hermes_realtime/client/static/assets/styles.css": "web/src/styles.css",
}


def validate_disclosure_manifest(package_root: Path) -> None:
    """Fail closed if canonical disclosure or packaged browser bytes drift."""

    disclosure = package_root / "hermes_realtime" / "evidence" / "disclosure_v1.txt"
    manifest = package_root / "hermes_realtime" / "evidence" / "disclosure_manifest_v1.json"
    static = package_root / "hermes_realtime" / "client" / "static"
    if not disclosure.is_file() or not manifest.is_file():
        fail("canonical evidence disclosure resource is missing")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("evidence disclosure manifest is not strict UTF-8 JSON") from error
    disclosure_bytes = disclosure.read_bytes()
    assets = {
        "app.js": static / "assets" / "app.js",
        "index.html": static / "index.html",
        "styles.css": static / "assets" / "styles.css",
    }
    expected = {
        "assets": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in assets.items()
        },
        "consentVersion": "realtime-evidence-consent-v1",
        "disclosureDigest": hashlib.sha256(
            b"realtime-evidence-consent-v1\0" + disclosure_bytes
        ).hexdigest(),
        "disclosureSha256": hashlib.sha256(disclosure_bytes).hexdigest(),
        "version": 1,
    }
    if document != expected:
        fail("evidence disclosure manifest does not match packaged resources")


REQUIRED_MODEL_RESOURCES = {
    "hermes_realtime/providers/models/SILERO_LICENSE": None,
    "hermes_realtime/providers/models/SILERO_NOTICE.md": None,
    "hermes_realtime/providers/models/silero_vad.onnx": (
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    ),
}
ALLOWED_SDIST_PATHS = (
    ".gitignore",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "uv.lock",
    "docs/",
    "requirements/",
    "scripts/",
    "src/",
    "tests/",
    "web/",
)
FORBIDDEN_ARCHIVE_PARTS = {
    ".git",
    ".hermes",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "coverage",
    "artifacts",
    "research",
    "spikes",
    "web/tmp",
}
SECRET_RULES = {
    "private-key": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "github-token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "openai-key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}
PUBLIC_PATH_PLACEHOLDERS = frozenset(
    {b"gate", b"gate-user", b"me", b"name", b"owner", b"private", b"user"}
)
WINDOWS_USER_PATH = re.compile(rb"(?i)(?:[A-Z]:)?[\\/]+Users[\\/]+([^\\/\s\"']+)")
PUBLIC_HARDWARE_RULES = {
    "private hardware marker": (
        re.compile(rb"\b[A-Z][A-Z0-9-]{2,}[ \t]+(?:USB[ \t]+)?(?:DAC|ADC)\b"),
        re.compile(rb"\b[A-Z][0-9]{2,3}[ \t]+(?:MQA|DAC)\b"),
    ),
    "private reference-host marker": (
        re.compile(rb"(?i)\bAMD[ \t]+Ryzen[ \t]+[3579][ \t]+[0-9]{4}[A-Z]{0,3}\b"),
        re.compile(rb"(?i)\bNVIDIA[ \t]+GeForce[ \t]+RTX[ \t]+[0-9]{3,4}[A-Z]{0,3}\b"),
    ),
}


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> None:
    printable = " ".join(command)
    print(f"+ {printable}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # Project selection, environment placement, and ambient uv configuration must
    # not redirect commands away from the materialized candidate. Cache and
    # interpreter installation locations supplied by CI remain operational inputs.
    for name in (
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "UV_PROJECT",
        "UV_WORKING_DIR",
        "UV_CONFIG_FILE",
        "UV_PROJECT_ENVIRONMENT",
    ):
        environment.pop(name, None)
    environment["UV_NO_CONFIG"] = "1"
    return environment


def clean_git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.upper().startswith("GIT_"):
            environment.pop(name)
    environment["LC_ALL"] = "C"
    return environment


def _resolved_commit_oid(source: Path, revision: str, *, label: str) -> str:
    resolved = subprocess.run(
        ("git", "rev-parse", "--verify", f"{revision}^{{commit}}"),
        cwd=source,
        env=clean_git_environment(),
        capture_output=True,
    )
    if resolved.returncode != 0 or re.fullmatch(rb"[0-9a-f]{40}\n", resolved.stdout) is None:
        fail(f"{label} must resolve to a full lowercase Git commit OID")
    return resolved.stdout[:-1].decode("ascii")


def canonical_candidate_diff_sha256(source: Path, baseline_commit: str | None) -> str:
    """Return the SHA-256 of the candidate's canonical raw binary diff."""

    head_oid = _resolved_commit_oid(source, "HEAD", label="candidate HEAD")
    if baseline_commit is None:
        parents = subprocess.run(
            ("git", "rev-list", "--parents", "-n", "1", head_oid),
            cwd=source,
            env=clean_git_environment(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().split()
        if parents == [head_oid]:
            diff = subprocess.run(
                (
                    "git",
                    "diff-tree",
                    "--root",
                    "--binary",
                    "--full-index",
                    "--no-renames",
                    "--no-ext-diff",
                    "--no-commit-id",
                    "-r",
                    head_oid,
                ),
                cwd=source,
                env=clean_git_environment(),
                capture_output=True,
                check=True,
            ).stdout
            return hashlib.sha256(diff).hexdigest()
        if len(parents) != 2 or parents[0] != head_oid:
            fail("a merge candidate requires an explicit baseline commit")
        baseline_commit = parents[1]

    baseline_oid = _resolved_commit_oid(source, baseline_commit, label="baseline commit")
    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", baseline_oid, head_oid),
        cwd=source,
        env=clean_git_environment(),
        capture_output=True,
    )
    if ancestry.returncode != 0:
        fail("baseline commit must be an ancestor of candidate HEAD")
    diff = subprocess.run(
        (
            "git",
            "diff",
            "--binary",
            "--full-index",
            "--no-renames",
            "--no-ext-diff",
            f"{baseline_oid}..{head_oid}",
        ),
        cwd=source,
        env=clean_git_environment(),
        capture_output=True,
        check=True,
    ).stdout
    return hashlib.sha256(diff).hexdigest()


def git_archive(source: Path, destination: Path) -> None:
    probe = subprocess.run(
        ("git", "rev-parse", "--is-inside-work-tree"),
        cwd=source,
        env=clean_git_environment(),
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        fail(f"candidate must be a Git checkout: {source}")
    archive = subprocess.run(
        ("git", "archive", "--format=tar", "HEAD"),
        cwd=source,
        env=clean_git_environment(),
        capture_output=True,
        check=True,
    )
    with tarfile.open(fileobj=__import__("io").BytesIO(archive.stdout), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def reject_ambient_paths(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        parts = set(relative.split("/"))
        forbidden_prefix = any(
            relative == forbidden or relative.startswith(f"{forbidden}/")
            for forbidden in FORBIDDEN_ARCHIVE_PARTS
            if "/" in forbidden
        )
        forbidden_part = bool(parts & {part for part in FORBIDDEN_ARCHIVE_PARTS if "/" not in part})
        if forbidden_prefix or forbidden_part:
            fail(f"fresh candidate unexpectedly contains ambient path: {relative}")


def secret_findings(relative: str, data: bytes) -> list[str]:
    findings: list[str] = []
    name = Path(relative).name
    suffix = Path(relative).suffix.lower()
    if name.startswith(".env") or suffix in {".pem", ".key", ".p12", ".pfx"}:
        findings.append(f"forbidden credential file: {relative}")
    else:
        for rule, pattern in SECRET_RULES.items():
            if pattern.search(data):
                findings.append(f"{rule}: {relative}")
    return findings


def public_disclosure_findings(relative: str, data: bytes) -> list[str]:
    findings: list[str] = []
    if relative == ".hermes" or relative.startswith(".hermes/"):
        findings.append(f"private governance path: {relative}")
    for match in WINDOWS_USER_PATH.finditer(data):
        if match.group(1).lower() not in PUBLIC_PATH_PLACEHOLDERS:
            findings.append(f"private machine path: {relative}")
            break
    for label, patterns in PUBLIC_HARDWARE_RULES.items():
        if any(pattern.search(data) for pattern in patterns):
            findings.append(f"{label}: {relative}")
    return findings


def fail_for_secret_findings(findings: list[str]) -> None:
    if findings:
        fail("secret scan failed (values redacted):\n" + "\n".join(sorted(findings)))


def canonical_baseline_diff_sha256(source: Path, baseline_commit: str) -> str:
    """Return the canonical candidate diff for an explicit baseline commit."""

    return canonical_candidate_diff_sha256(source, baseline_commit)


def scan_git_blobs(source: Path) -> None:
    """Scan every committed blob in HEAD, including paths hidden by export-ignore."""
    tree = subprocess.run(
        ("git", "ls-tree", "-r", "-z", "--full-tree", "HEAD"),
        cwd=source,
        env=clean_git_environment(),
        capture_output=True,
        check=True,
    ).stdout
    findings: list[str] = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, raw_path = entry.split(b"\t", 1)
            _mode, object_type, object_id = metadata.split()
        except ValueError as error:
            raise RuntimeError("unable to parse Git tree entry during secret scan") from error
        if object_type != b"blob":
            continue
        data = subprocess.run(
            ("git", "cat-file", "blob", object_id.decode("ascii")),
            cwd=source,
            env=clean_git_environment(),
            capture_output=True,
            check=True,
        ).stdout
        relative = raw_path.decode("utf-8", "surrogateescape")
        findings.extend(secret_findings(relative, data))
        findings.extend(public_disclosure_findings(relative, data))
    fail_for_secret_findings(findings)


def scan_for_secrets(root: Path) -> None:
    findings: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            findings.extend(secret_findings(relative, data))
            findings.extend(public_disclosure_findings(relative, data))
    fail_for_secret_findings(findings)


def snapshot_packaged_static(root: Path, snapshot: Path) -> None:
    for packaged in REQUIRED_STATIC:
        source = root / "src" / packaged
        destination = snapshot / packaged
        if not source.is_file():
            fail(f"missing packaged static asset before web build: {source.relative_to(root)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def check_static_parity(root: Path, snapshot: Path) -> None:
    for packaged in REQUIRED_STATIC:
        # `left` is copied before build.mjs can overwrite package static files;
        # `right` is build.mjs's newly generated output.
        left = snapshot / packaged
        right = root / "src" / packaged
        if not left.is_file() or not right.is_file():
            fail(f"missing static parity input: {left if not left.is_file() else right}")
        left_bytes = left.read_bytes().replace(b"\r\n", b"\n")
        right_bytes = right.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(left_bytes).digest() != hashlib.sha256(right_bytes).digest():
            fail(f"generated static asset is stale: {packaged} != fresh web build output")


def required_sdist_paths() -> frozenset[str]:
    """Return checked-in runtime/qualification files that sdists must retain."""

    return frozenset(
        {
            "THIRD_PARTY_NOTICES.md",
            "requirements/README.md",
            "requirements/kokoro-cuda-worker.in",
            "requirements/kokoro-cuda-worker-win-py311.txt",
            "requirements/kokoro-onnx-package-win-py311.txt",
            "scripts/benchmark_evidence_admission.py",
            "scripts/qualify_evidence_slice_zero.py",
            "scripts/deterministic_equivalence.py",
            "scripts/equivalence_process.py",
            "scripts/equivalence_worker.py",
            "scripts/qualify_deterministic_equivalence.py",
            "scripts/candidate_wheel.py",
            "scripts/revoke_race.py",
            "scripts/revoke_race_worker.py",
            "scripts/qualify_revoke_race.py",
            "scripts/packaged_scenario.py",
            "scripts/capacity_rollover.py",
            "scripts/capacity_rollover_worker.py",
            "scripts/qualify_capacity_rollover.py",
            "scripts/evidence_observation.py",
            "scripts/over_budget_turn.py",
            "scripts/over_budget_turn_worker.py",
            "scripts/qualify_over_budget_turn.py",
            "scripts/owned_close_faults.py",
            "scripts/owned_close_faults_worker.py",
            "scripts/qualify_owned_close_faults.py",
            "scripts/spool_crash_matrix.py",
            "scripts/storage_process.py",
            "scripts/storage_worker.py",
            "scripts/storage_observation.py",
            "scripts/windows_storage_oracle.py",
            "scripts/evidence_protocol_oracle.py",
            "scripts/spool_crash_oracle.py",
            "scripts/qualify_spool_crash_matrix.py",
            "scripts/full_purge_cleanup.py",
            "scripts/full_purge_observation.py",
            "scripts/full_purge_worker.py",
            "scripts/qualify_full_purge_cleanup.py",
            "tests/support/qualification.py",
            "tests/integration/test_qualification_full_host_ingress.py",
            "scripts/qualify_hermes_v020_pluginmanager.py",
            "scripts/source_archive_authority.py",
            "scripts/real_gate_support.py",
            "scripts/real_natural_work_gate.py",
            "scripts/task13_artifact_orchestrator.py",
            "scripts/candidate_source_archive_oracle.py",
            "scripts/candidate_e2e_fast_track.py",
            "scripts/schemas/benchmark-machine-v1.schema.json",
            "scripts/schemas/benchmark-report-v1.schema.json",
            "scripts/schemas/qualification-input-v1.schema.json",
            "scripts/schemas/qualification-report-v1.schema.json",
            "scripts/schemas/release-manifest-v1.schema.json",
            "scripts/schemas/wheelhouse-manifest-v1.schema.json",
            "src/hermes_realtime/evidence/disclosure_manifest_v1.json",
            "src/hermes_realtime/evidence/disclosure_v1.txt",
            "tests/evidence/spool_crash_worker.py",
        }
    )


def check_artifacts(root: Path, artifacts: Path) -> Path:
    wheels = list(artifacts.glob("*.whl"))
    sdists = list(artifacts.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(f"expected exactly one wheel and one sdist, found wheels={wheels}, sdists={sdists}")
    for artifact in [*wheels, *sdists]:
        if artifact.stat().st_size > MAX_ARTIFACT_BYTES:
            fail(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes: {artifact.name}")
    validate_disclosure_manifest(root / "src")
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_names = set(wheel.namelist())
        for packaged in REQUIRED_STATIC:
            if packaged not in wheel_names:
                fail(f"wheel is missing required static asset: {packaged}")
        for packaged, expected_sha256 in REQUIRED_MODEL_RESOURCES.items():
            if packaged not in wheel_names:
                fail(f"wheel is missing required model resource: {packaged}")
            wheel_bytes = wheel.read(packaged)
            source_bytes = (root / "src" / packaged).read_bytes()
            if wheel_bytes != source_bytes:
                fail(f"wheel model resource differs from source: {packaged}")
            if (
                expected_sha256 is not None
                and hashlib.sha256(wheel_bytes).hexdigest() != expected_sha256
            ):
                fail(f"wheel model resource checksum mismatch: {packaged}")
        forbidden = [
            name
            for name in wheel_names
            if any(part in FORBIDDEN_ARCHIVE_PARTS for part in name.split("/"))
        ]
        if forbidden:
            fail("wheel contains forbidden path(s): " + ", ".join(sorted(forbidden)))
        for packaged, generated in REQUIRED_STATIC.items():
            if (
                hashlib.sha256(wheel.read(packaged)).digest()
                != hashlib.sha256((root / generated).read_bytes()).digest()
            ):
                fail(f"wheel static asset differs from fresh web build: {packaged}")
        for packaged in (
            "hermes_realtime/evidence/disclosure_v1.txt",
            "hermes_realtime/evidence/disclosure_manifest_v1.json",
        ):
            source_bytes = (root / "src" / packaged).read_bytes()
            if wheel.read(packaged) != source_bytes:
                fail(f"wheel disclosure resource differs from source: {packaged}")
    with tarfile.open(sdists[0], "r:gz") as sdist:
        sdist_names = [
            member.name.split("/", 1)[1] for member in sdist.getmembers() if "/" in member.name
        ]
    missing = sorted(required_sdist_paths() - set(sdist_names))
    if missing:
        fail("sdist is missing required path(s): " + ", ".join(missing))
    unexpected = [
        name
        for name in sdist_names
        if not name.endswith("/") and not name.startswith(ALLOWED_SDIST_PATHS)
    ]
    forbidden = [
        name
        for name in sdist_names
        if any(part in FORBIDDEN_ARCHIVE_PARTS for part in name.split("/"))
    ]
    if unexpected or forbidden:
        fail("sdist manifest violation: " + ", ".join(sorted(set(unexpected + forbidden))))
    return wheels[0]


def build_sdist_wheel(root: Path, sdist: Path, environment: dict[str, str]) -> Path:
    """Build one wheel from the produced sdist, never from the checkout."""

    output = root / "sdist-built-artifacts"
    output.mkdir()
    run(
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(output),
        str(sdist),
        cwd=root,
        env=environment,
    )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        fail(f"expected exactly one sdist-built wheel, found {wheels}")
    return wheels[0]


def check_sdist_built_wheel_disclosure_parity(root: Path, wheel: Path) -> None:
    """Verify the sdist's rebuilt wheel carries the canonical browser disclosure."""

    expected_paths = {
        *REQUIRED_STATIC,
        "hermes_realtime/evidence/disclosure_v1.txt",
        "hermes_realtime/evidence/disclosure_manifest_v1.json",
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = sorted(expected_paths - names)
        if missing:
            fail("sdist-built wheel is missing disclosure resource(s): " + ", ".join(missing))
        for packaged in sorted(expected_paths):
            source = root / "src" / packaged
            if archive.read(packaged) != source.read_bytes():
                fail(f"sdist-built wheel disclosure resource differs from source: {packaged}")


def check_installed_wheel(root: Path, wheel: Path, environment: dict[str, str]) -> None:
    installation = root.parent / ".release-import-env"
    run("uv", "venv", "--clear", "--python", "3.11", str(installation), cwd=root, env=environment)
    python = installation / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run("uv", "pip", "install", "--python", str(python), str(wheel), cwd=root, env=environment)
    outside = root.parent / "wheel-import"
    outside.mkdir(exist_ok=True)
    script = (
        "import hermes_realtime, pathlib; "
        "import hashlib; "
        "from importlib.resources import files; "
        "from importlib.metadata import distribution; "
        "p=pathlib.Path(hermes_realtime.__file__).resolve(); "
        "print(p); "
        "assert 'site-packages' in p.as_posix(), p; "
        f"assert not p.is_relative_to(pathlib.Path({str(root)!r})), p; "
        "eps={e.name:e for e in distribution('hermes-realtime').entry_points}; "
        "expected={'hermes-realtime','hermes-realtime-host','hermes-realtime-local'}; "
        "assert set(eps)==expected, eps; "
        "assert eps['hermes-realtime'].group=='hermes_agent.plugins'; "
        "assert eps['hermes-realtime-host'].group=='console_scripts'; "
        "assert eps['hermes-realtime-local'].group=='console_scripts'; "
        "loaded={name:ep.load() for name,ep in eps.items()}; "
        "assert loaded['hermes-realtime'].__name__=='hermes_realtime.hermes_plugin'; "
        "assert callable(loaded['hermes-realtime-host']); "
        "assert callable(loaded['hermes-realtime-local']); "
        "models=files('hermes_realtime.providers').joinpath('models'); "
        "assert models.joinpath('SILERO_LICENSE').is_file(); "
        "assert models.joinpath('SILERO_NOTICE.md').is_file(); "
        "model=models.joinpath('silero_vad.onnx').read_bytes(); "
        "assert hashlib.sha256(model).hexdigest()=='1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'"
    )
    run(str(python), "-I", "-c", script, cwd=outside, env=environment)


def _verified_livekit_listener_pid() -> int:
    """Return the sole loopback signaling listener, rejecting ambient listeners."""

    if os.name != "nt":
        fail("native LiveKit ownership verification is only supported on Windows")
    completed = subprocess.run(
        ("netstat", "-ano", "-p", "tcp"),
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        fail("could not inspect the owned LiveKit signaling listener")
    listener_pids: list[int] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != "TCP":
            continue
        local_address, state, pid = fields[1], fields[3], fields[4]
        if not local_address.endswith(":7880") or state != "LISTENING":
            continue
        if local_address != "127.0.0.1:7880" or not pid.isdecimal():
            fail("LiveKit signaling listener is not an exact IPv4 loopback listener")
        listener_pids.append(int(pid))
    if len(listener_pids) != 1:
        fail("expected exactly one owned LiveKit signaling listener")
    return listener_pids[0]


def check_livekit(
    root: Path,
    environment: dict[str, str],
    *,
    executable: Path,
    expected_sha256: str,
    process_id: int,
) -> None:
    if not executable.is_file():
        fail("verified LiveKit executable does not exist")
    if hashlib.sha256(executable.read_bytes()).hexdigest() != expected_sha256:
        fail("LiveKit executable hash does not match the verified value")
    if _verified_livekit_listener_pid() != process_id:
        fail("LiveKit signaling listener is not owned by the supplied process")
    expected_environment = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "local-" + "x" * 32,
        "LIVEKIT_KEYS": "devkey: local-" + "x" * 32 + "\n",
    }
    if {key: environment.get(key) for key in expected_environment} != expected_environment:
        fail("native LiveKit gate requires the exact owned local credential environment")
    try:
        with urllib.request.urlopen("http://127.0.0.1:7880/", timeout=5) as response:
            if response.read().strip() != b"OK":
                fail("local LiveKit readiness endpoint did not return OK")
    except OSError as error:
        raise RuntimeError(
            "native LiveKit gate requires a running loopback server at 127.0.0.1:7880"
        ) from error
    local_env = environment | {"HERMES_REALTIME_LIVEKIT_LOCAL": "1"}
    run(
        "uv",
        "run",
        "--frozen",
        "--extra",
        "local",
        "--group",
        "dev",
        "pytest",
        "-q",
        *PYTEST_DURATIONS,
        "-W",
        "error::jwt.warnings.InsecureKeyLengthWarning",
        "tests/integration/test_local_livekit.py",
        "tests/integration/test_browser_livekit.py",
        "tests/integration/test_local_launcher.py",
        "tests/integration/test_qualification_full_host_ingress.py",
        "tests/integration/test_qualification_full_host_synthetic_audio.py",
        cwd=root,
        env=local_env,
    )


def run_script_mypy(root: Path, environment: dict[str, str]) -> None:
    script_type_env = environment | {
        "MYPYPATH": os.pathsep.join((str(root / "src"), str(root / "scripts")))
    }
    run(
        "uv",
        "run",
        "--frozen",
        "--group",
        "dev",
        "mypy",
        "--follow-imports=skip",
        "scripts/benchmark_evidence_admission.py",
        "scripts/qualify_evidence_slice_zero.py",
        "scripts/deterministic_equivalence.py",
        "scripts/equivalence_process.py",
        "scripts/equivalence_worker.py",
        "scripts/qualify_deterministic_equivalence.py",
        "scripts/candidate_wheel.py",
        "scripts/revoke_race.py",
        "scripts/revoke_race_worker.py",
        "scripts/qualify_revoke_race.py",
        "scripts/packaged_scenario.py",
        "scripts/capacity_rollover.py",
        "scripts/capacity_rollover_worker.py",
        "scripts/qualify_capacity_rollover.py",
        "scripts/evidence_observation.py",
        "scripts/over_budget_turn.py",
        "scripts/over_budget_turn_worker.py",
        "scripts/qualify_over_budget_turn.py",
        "scripts/owned_close_faults.py",
        "scripts/owned_close_faults_worker.py",
        "scripts/qualify_owned_close_faults.py",
        "scripts/spool_crash_matrix.py",
        "scripts/storage_process.py",
        "scripts/storage_worker.py",
        "scripts/storage_observation.py",
        "scripts/windows_storage_oracle.py",
        "scripts/evidence_protocol_oracle.py",
        "scripts/spool_crash_oracle.py",
        "scripts/qualify_spool_crash_matrix.py",
        "scripts/full_purge_cleanup.py",
        "scripts/full_purge_observation.py",
        "scripts/full_purge_worker.py",
        "scripts/qualify_full_purge_cleanup.py",
        "scripts/qualify_hermes_v020_pluginmanager.py",
        "scripts/source_archive_authority.py",
        "scripts/real_natural_work_gate.py",
        "scripts/real_gate_support.py",
        "scripts/real_hermes_api_gate.py",
        "scripts/task13_artifact_orchestrator.py",
        "scripts/candidate_source_archive_oracle.py",
        "scripts/candidate_e2e_fast_track.py",
        cwd=root,
        env=script_type_env,
    )


def gate_materialized_candidate(
    root: Path,
    *,
    livekit: bool,
    livekit_executable: Path | None = None,
    livekit_executable_sha256: str | None = None,
    livekit_pid: int | None = None,
) -> None:
    """Qualify one already-materialized disposable candidate source tree."""

    environment = clean_environment()
    workspace = root.parent
    reject_ambient_paths(root)
    scan_for_secrets(root)
    run(
        "uv", "lock", "--check", "--project", str(root), "--no-config", "--python", "3.11",
        cwd=root, env=environment,
    )
    # Preserve the committed package bytes before any test or build command can
    # rewrite generated static output.
    packaged_static = workspace / "packaged-static-before-web-build"
    snapshot_packaged_static(root, packaged_static)
    run("uv", "sync", "--frozen", "--python", "3.11", "--dev", cwd=root, env=environment)
    source_env = environment | {"PYTHONPATH": str(root / "src")}
    run(
        "uv",
        "run",
        "--frozen",
        "--group",
        "dev",
        "python",
        "-c",
        "import hermes_realtime, pathlib; p=pathlib.Path(hermes_realtime.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path('src').resolve()), p",
        cwd=root,
        env=source_env,
    )
    run(
        "uv",
        "run",
        "--frozen",
        "--group",
        "dev",
        "pytest",
        "-q",
        *PYTEST_DURATIONS,
        cwd=root,
        env=environment,
    )
    speech_verification_env = dict(environment)
    speech_verification_env["HERMES_RELEASE_SPEECH_VERIFICATION"] = "1"
    run(
        "uv",
        "run",
        "--frozen",
        "--extra",
        "speech-verification",
        "--group",
        "dev",
        "pytest",
        "-q",
        *PYTEST_DURATIONS,
        "tests/providers/test_speech_presence.py",
        cwd=root,
        env=speech_verification_env,
    )
    run(
        "uv",
        "run",
        "--frozen",
        "--group",
        "dev",
        "ruff",
        "check",
        ".",
        cwd=root,
        env=environment,
    )
    run("uv", "run", "--frozen", "--group", "dev", "mypy", "src", cwd=root, env=environment)
    run_script_mypy(root, environment)
    run(NPM, "ci", "--ignore-scripts", cwd=root / "web", env=environment)
    run(NPM, "test", *VITEST_DURATIONS, cwd=root / "web", env=environment)
    run(NPM, "run", "build", cwd=root / "web", env=environment)
    check_static_parity(root, packaged_static)
    validate_disclosure_manifest(root / "src")
    artifacts = root / "release-artifacts"
    run("uv", "build", "--out-dir", str(artifacts), cwd=root, env=environment)
    wheel = check_artifacts(root, artifacts)
    sdist = next(artifacts.glob("*.tar.gz"), None)
    if sdist is None:
        fail("release artifact build did not produce an sdist")
    sdist_built_wheel = build_sdist_wheel(root, sdist, environment)
    check_sdist_built_wheel_disclosure_parity(root, sdist_built_wheel)
    check_installed_wheel(root, wheel, environment)
    if livekit:
        if (
            livekit_executable is None
            or livekit_executable_sha256 is None
            or livekit_pid is None
        ):
            fail("native LiveKit gate requires verified executable ownership arguments")
        check_livekit(
            root,
            environment,
            executable=livekit_executable,
            expected_sha256=livekit_executable_sha256,
            process_id=livekit_pid,
        )


def gate(
    source: Path,
    livekit: bool,
    *,
    livekit_executable: Path | None = None,
    livekit_executable_sha256: str | None = None,
    livekit_pid: int | None = None,
) -> None:
    diff_sha256 = canonical_candidate_diff_sha256(source, None)
    print(f"canonical candidate diff SHA-256: {diff_sha256}", flush=True)
    with tempfile.TemporaryDirectory(prefix="hermes-realtime-release-") as temporary:
        root = Path(temporary) / "candidate"
        root.mkdir()
        scan_git_blobs(source)
        git_archive(source, root)
        gate_materialized_candidate(
            root,
            livekit=livekit,
            livekit_executable=livekit_executable,
            livekit_executable_sha256=livekit_executable_sha256,
            livekit_pid=livekit_pid,
        )
    print("release gate passed")


def expect_failure(action: Callable[[], None], expected: str) -> None:
    try:
        action()
    except RuntimeError as error:
        if expected not in str(error):
            raise RuntimeError(f"self-test failed with unexpected error: {error}") from error
    else:
        fail(f"self-test did not reject {expected}")


def self_test() -> None:
    if MAX_ARTIFACT_BYTES <= 0 or not REQUIRED_STATIC or not SECRET_RULES:
        fail("release gate constants are invalid")
    with tempfile.TemporaryDirectory(prefix="hermes-realtime-release-self-test-") as temporary:
        root = Path(temporary)
        repository = root / "repository"
        repository.mkdir()
        for command in (
            ("git", "init", "-q"),
            ("git", "config", "user.email", "release-gate-test@example.invalid"),
            ("git", "config", "user.name", "release-gate-test"),
        ):
            subprocess.run(command, cwd=repository, check=True)
        (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
        subprocess.run(("git", "add", "tracked.txt"), cwd=repository, check=True)
        subprocess.run(("git", "commit", "-qm", "clean fixture"), cwd=repository, check=True)
        scan_git_blobs(repository)
        (repository / ".gitattributes").write_text("secret.txt export-ignore\n", encoding="utf-8")
        token = "gh" + "p_abcdefghijklmnopqrstuvwxyz1234567890"
        (repository / "secret.txt").write_text(f"{token}\n", encoding="utf-8")
        subprocess.run(("git", "add", ".gitattributes", "secret.txt"), cwd=repository, check=True)
        subprocess.run(
            ("git", "commit", "-qm", "export-ignore secret fixture"), cwd=repository, check=True
        )
        archived = root / "archived"
        archived.mkdir()
        git_archive(repository, archived)
        if (archived / "secret.txt").exists():
            fail("self-test fixture did not exercise export-ignore")
        expect_failure(lambda: scan_git_blobs(repository), "github-token: secret.txt")

        candidate = root / "candidate"
        snapshot = root / "packaged-static"
        for packaged in REQUIRED_STATIC:
            path = candidate / "src" / packaged
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(packaged.encode("utf-8"))
        snapshot_packaged_static(candidate, snapshot)
        check_static_parity(candidate, snapshot)
        stale = candidate / "src" / "hermes_realtime/client/static/assets/app.js"
        snapshotted = snapshot / "hermes_realtime/client/static/assets/app.js"
        snapshotted.write_bytes(b"const first = 1;\r\nconst second = 2;\r\n")
        stale.write_bytes(b"const first = 1;\nconst second = 2;\n")
        check_static_parity(candidate, snapshot)
        stale.write_bytes(b"fresh build output")
        expect_failure(
            lambda: check_static_parity(candidate, snapshot), "generated static asset is stale"
        )

        ambient = root / "ambient-candidate" / "web" / "tmp"
        ambient.mkdir(parents=True)
        (ambient / "trace.json").write_text("{}", encoding="utf-8")
        expect_failure(
            lambda: reject_ambient_paths(root / "ambient-candidate"),
            "fresh candidate unexpectedly contains ambient path: web/tmp",
        )
    print("release gate self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path.cwd(),
        help="Git checkout whose HEAD is the candidate",
    )
    parser.add_argument(
        "--require-livekit", action="store_true", help="run native local LiveKit integration tests"
    )
    parser.add_argument("--livekit-executable", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--livekit-executable-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--livekit-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run adversarial release-gate checks without building",
    )
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    ownership_arguments = (
        arguments.livekit_executable,
        arguments.livekit_executable_sha256,
        arguments.livekit_pid,
    )
    if arguments.require_livekit:
        if any(value is None for value in ownership_arguments):
            parser.error("--require-livekit requires verified executable, SHA-256, and PID")
        assert arguments.livekit_executable_sha256 is not None
        assert arguments.livekit_pid is not None
        if re.fullmatch(r"[0-9a-f]{64}", arguments.livekit_executable_sha256) is None:
            parser.error("--livekit-executable-sha256 must be lowercase hexadecimal SHA-256")
        if arguments.livekit_pid < 1:
            parser.error("--livekit-pid must be positive")
    elif any(value is not None for value in ownership_arguments):
        parser.error("LiveKit ownership arguments require --require-livekit")
    gate(
        arguments.candidate.resolve(),
        arguments.require_livekit,
        livekit_executable=arguments.livekit_executable,
        livekit_executable_sha256=arguments.livekit_executable_sha256,
        livekit_pid=arguments.livekit_pid,
    )


if __name__ == "__main__":
    main()
