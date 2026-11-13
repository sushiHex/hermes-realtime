"""The pinned LiveKit archive may come from a cache, but only the pin admits it.

GitHub documents that cache contents "are not signed or verified", so the cache is
transport only. These tests execute the workflow's own install step under PowerShell with
the download replaced by a recording stub and the pinned digest replaced by a fixture's, and
check that restored bytes meet the same comparison as downloaded bytes.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE_ACTION = "55cc8345863c7cc4c66a329aec7e433d2d1c52a9"  # actions/cache v6.1.0
CACHE = "      - name: Cache pinned local LiveKit archive"
INSTALL = "      - name: Install pinned local LiveKit server"
STUB = """\
function Invoke-WebRequest {
  [CmdletBinding()]
  param(
    [Parameter(Position = 0)][string]$Uri,
    [string]$OutFile,
    [int]$MaximumRetryCount,
    [int]$RetryIntervalSec
  )
  Add-Content -LiteralPath $env:FAKE_DOWNLOAD_LOG -Value "$Uri|$MaximumRetryCount|$RetryIntervalSec"
  Copy-Item -LiteralPath $env:FAKE_DOWNLOAD_SOURCE -Destination $OutFile
}
"""


def _pins() -> tuple[str, str, str]:
    pins = run_path(str(ROOT / "scripts" / "local_livekit.py"))
    return pins["VERSION"], pins["ARCHIVE_SHA256"], pins["ARCHIVE_URL"]


def _jobs() -> dict[str, str]:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(encoding="utf-8")
    return {
        "packaged-capture": workflow.split("  packaged-capture:", maxsplit=1)[1].split(
            "  candidate-wheel:", maxsplit=1
        )[0],
        "native-livekit": workflow.split("  native-livekit:", maxsplit=1)[1],
    }


def _step(job: str, marker: str) -> str:
    assert job.count(marker) == 1
    return job.split(marker, maxsplit=1)[1].split("\n      - ", maxsplit=1)[0] + "\n"


def _install_script() -> str:
    bodies = {
        "\n".join(
            line.removeprefix("          ")
            for line in _step(job, INSTALL).split("        run: |\n", maxsplit=1)[1].splitlines()
            if line.startswith("          ")
        )
        for job in _jobs().values()
    }
    assert len(bodies) == 1
    return bodies.pop()


def _archive(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("livekit-server.exe", payload)
    return buffer.getvalue()


class _Run:
    def __init__(self, tmp_path: Path, *, restored: bytes | None, served: bytes) -> None:
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            # Hosted runners execute this very step with pwsh; only a local host may lack it.
            assert os.environ.get("GITHUB_ACTIONS") != "true"
            pytest.skip("PowerShell 7 (pwsh) is required to execute the workflow step")
        version, pinned, _ = _pins()
        self.pinned_archive = _archive(b"pinned server")
        script = _install_script()
        assert script.count(f"'{pinned}'") == 2
        script = script.replace(pinned, hashlib.sha256(self.pinned_archive).hexdigest())
        self.root = tmp_path
        self.archive = tmp_path / f"livekit_{version}_windows_amd64.zip"
        if restored is not None:
            self.archive.write_bytes(restored)
        source = tmp_path / "served.zip"
        source.write_bytes(served)
        self.log = tmp_path / "downloads.log"
        self.environment_file = tmp_path / "github-env"
        self.environment_file.write_bytes(b"")
        harness = tmp_path / "step.ps1"
        harness.write_text(STUB + script + "\n", encoding="utf-8")
        self.completed = subprocess.run(
            (pwsh, "-NoProfile", "-NonInteractive", "-File", str(harness)),
            env=os.environ
            | {
                "RUNNER_TEMP": str(tmp_path),
                "GITHUB_ENV": str(self.environment_file),
                "FAKE_DOWNLOAD_LOG": str(self.log),
                "FAKE_DOWNLOAD_SOURCE": str(source),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    @property
    def downloads(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").split() if self.log.exists() else []

    @property
    def exported(self) -> dict[str, str]:
        lines = self.environment_file.read_text(encoding="utf-8-sig").splitlines()
        return dict(line.split("=", maxsplit=1) for line in lines if line)

    def assert_installed(self) -> None:
        assert self.completed.returncode == 0, self.completed.stderr
        server = self.root / "livekit" / "livekit-server.exe"
        assert server.read_bytes() == b"pinned server"
        assert self.exported == {
            "LIVEKIT_SERVER": str(server),
            "LIVEKIT_SERVER_SHA256": hashlib.sha256(b"pinned server").hexdigest(),
        }

    def assert_refused(self) -> None:
        assert self.completed.returncode != 0
        assert "LiveKit checksum mismatch" in self.completed.stderr
        assert self.exported == {}
        assert not (self.root / "livekit").exists()


def test_cache_miss_downloads_the_pinned_archive_within_the_existing_bounds(
    tmp_path: Path,
) -> None:
    _, _, url = _pins()
    run = _Run(tmp_path, restored=None, served=_archive(b"pinned server"))

    run.assert_installed()
    assert run.downloads == [f"{url}|8|30"]
    # What the save step then stores is exactly the verified archive.
    assert run.archive.read_bytes() == run.pinned_archive


def test_restored_pinned_archive_installs_without_any_download(tmp_path: Path) -> None:
    run = _Run(tmp_path, restored=_archive(b"pinned server"), served=_archive(b"unused"))

    run.assert_installed()
    assert run.downloads == []


def test_restored_archive_with_another_digest_fails_closed_without_a_download(
    tmp_path: Path,
) -> None:
    # A well-formed archive holding a different server: with the comparison removed this
    # would install and export it, so the refusal below is the comparison's alone.
    run = _Run(tmp_path, restored=_archive(b"other server"), served=_archive(b"pinned server"))

    run.assert_refused()
    assert "(restored)" in run.completed.stderr
    # No silent repair: a wrong cached archive is a finding, not a retry.
    assert run.downloads == []


def test_downloaded_archive_with_another_digest_fails_closed(tmp_path: Path) -> None:
    run = _Run(tmp_path, restored=None, served=_archive(b"other server"))

    run.assert_refused()
    assert "(downloaded)" in run.completed.stderr
    assert len(run.downloads) == 1


def test_install_step_uses_the_one_repository_pin_and_one_comparison() -> None:
    version, pinned, url = _pins()
    script = _install_script()

    assert script.count(f"Invoke-WebRequest '{url}' -OutFile $archive") == 1
    assert script.count("Invoke-WebRequest ") == 1
    # Each source sets the same pinned digest, and one statement compares it.
    assert script.count(f"$expected = '{pinned}'") == 2
    assert set(re.findall(r"[0-9a-f]{64}", script)) == {pinned}
    assert script.count("$actual = (Get-FileHash $archive -Algorithm SHA256)") == 2
    assert script.count("$actual -ne $expected") == 1
    assert script.count("throw ") == 1
    assert script.count(f"'livekit_{version}_windows_amd64.zip'") == 2
    comparison = script.index("if ($actual -ne $expected) { throw ")
    assert script.index("Invoke-WebRequest '") < comparison < script.index("Expand-Archive")
    assert script.index("Expand-Archive") < script.index("LIVEKIT_SERVER=")


def test_cache_step_is_keyed_on_version_and_digest_and_precedes_the_verification() -> None:
    version, pinned, _ = _pins()
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(encoding="utf-8")

    assert workflow.count(f"uses: actions/cache@{CACHE_ACTION}") == 2
    assert workflow.count("actions/cache") == 2
    for job in _jobs().values():
        # The step right after the cache step is the one that verifies what it restored.
        assert job.split(CACHE, maxsplit=1)[1].split("\n      - ", maxsplit=2)[1].startswith(
            INSTALL.removeprefix("      - ")
        )
        # Exactly these inputs: no prefix restore key, and the cache step's own failure
        # falls back to the download.
        assert _step(job, CACHE) == (
            "\n"
            f"        uses: actions/cache@{CACHE_ACTION}\n"
            "        continue-on-error: true\n"
            "        with:\n"
            f"          path: ${{{{ runner.temp }}}}/livekit_{version}_windows_amd64.zip\n"
            f"          key: livekit-windows-amd64-{version}-sha256-{pinned}\n"
        )


def test_scheduled_cold_fetch_checks_upstream_without_the_cache_and_gates_nothing() -> None:
    workflow = (ROOT / ".github" / "workflows" / "pinned-inputs.yml").read_text(encoding="utf-8")
    triggers = workflow.split("\non:\n", maxsplit=1)[1].split("\npermissions:\n", maxsplit=1)[0]

    # Scheduled and manual only: an upstream outage must never fail a pull request.
    assert triggers == '  schedule:\n    - cron: "23 6 * * 1"\n  workflow_dispatch:\n'
    assert "\npermissions:\n  contents: read\n" in workflow
    assert "actions/cache" not in workflow
    assert "enable-cache: false" in workflow
    assert [line.strip() for line in workflow.splitlines() if "uses:" in line] == [
        "- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "- uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "- uses: astral-sh/setup-uv@bec219d24cd3e171d82865faccec33120bb574f4",
    ]
    assert workflow.count("run: ") == 1
    assert workflow.rstrip().endswith(
        "run: uv run --frozen --group dev python -m scripts.local_livekit verify-upstream"
    )
