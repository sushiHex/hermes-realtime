"""The one shared LiveKit server: its path, its pins, its verification and its command."""

import hashlib
import io
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pytest

from scripts import local_livekit
from scripts import qualification_livekit_files as livekit_files

_ROOT = Path(__file__).resolve().parents[1]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _archive(executable: bytes = b"synthetic livekit\n", sidecar: bytes = b"notice\n") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("livekit-server.exe", executable)
        bundle.writestr("LICENSE", sidecar)
    return stream.getvalue()


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, bytes]:
    """Pin the helper to a synthetic release whose archive and executable are known."""

    executable = b"synthetic livekit\n"
    archive = _archive(executable)
    monkeypatch.setattr(local_livekit, "ARCHIVE_SHA256", _sha256(archive))
    monkeypatch.setattr(local_livekit, "EXECUTABLE_SHA256", _sha256(executable))
    return archive, executable


def _never(url: str) -> bytes:
    raise AssertionError("a verified server must not be downloaded again")


def test_the_server_lives_once_per_user_outside_every_checkout(tmp_path: Path) -> None:
    path = local_livekit.shared_path({"LOCALAPPDATA": str(tmp_path)})

    assert path == tmp_path / "hermes-realtime" / "tools" / "livekit-1.13.4" / "livekit-server.exe"
    assert not path.is_relative_to(_ROOT)


@pytest.mark.parametrize("environ", [{}, {"LOCALAPPDATA": ""}, {"LOCALAPPDATA": "relative"}])
def test_the_shared_path_needs_an_absolute_per_user_directory(environ: dict[str, str]) -> None:
    with pytest.raises(local_livekit.LiveKitUnavailable):
        local_livekit.shared_path(environ)


def test_the_pins_equal_every_other_record_of_them() -> None:
    # The qualification reads its LiveKit policy from these two sources with these patterns.
    workflow = (_ROOT / ".github/workflows/release-gates.yml").read_bytes()
    browser = (_ROOT / "tests/integration/test_browser_self_acceptance.py").read_bytes()
    docs = (_ROOT / "docs/local-livekit.md").read_text(encoding="utf-8")
    asset = f"livekit_{local_livekit.VERSION}_windows_amd64.zip"

    assert (
        livekit_files._WORKFLOW_PIN.findall(workflow)
        == [(asset.encode(), local_livekit.ARCHIVE_SHA256.encode())] * 2
    )
    release = f"https://github.com/livekit/livekit/releases/download/v{local_livekit.VERSION}"
    assert f"{release}/{asset}" == local_livekit.ARCHIVE_URL
    assert (
        re.findall(rb"Invoke-WebRequest '([^']+)' -OutFile \$archive", workflow)
        == [local_livekit.ARCHIVE_URL.encode()] * 2
    )
    assert livekit_files._BROWSER_PIN.findall(browser) == [local_livekit.EXECUTABLE_SHA256.encode()]
    assert re.findall(r"- LiveKit Server: `v([0-9.]+)`", docs) == [local_livekit.VERSION]
    assert re.findall(r"- SHA-256: `([0-9a-f]{64})`", docs) == [local_livekit.ARCHIVE_SHA256]


def test_install_places_the_verified_server_once(
    tmp_path: Path, release: tuple[bytes, bytes]
) -> None:
    archive, executable = release
    environ = {"LOCALAPPDATA": str(tmp_path)}
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return archive

    path = local_livekit.install(environ, fetch=fetch)

    assert path == local_livekit.shared_path(environ)
    assert path.read_bytes() == executable
    assert fetched == [local_livekit.ARCHIVE_URL]
    assert list(path.parent.iterdir()) == [path]
    assert local_livekit.install(environ, fetch=_never) == path


def test_install_refuses_an_archive_that_differs_from_its_pin(
    tmp_path: Path, release: tuple[bytes, bytes]
) -> None:
    _, executable = release
    environ = {"LOCALAPPDATA": str(tmp_path)}
    # The executable inside still matches its own pin; only the archive differs.
    altered = _archive(executable, sidecar=b"altered notice\n")

    with pytest.raises(local_livekit.LiveKitUnavailable, match="archive differs"):
        local_livekit.install(environ, fetch=lambda url: altered)
    assert not local_livekit.shared_path(environ).parent.exists()


def test_install_refuses_an_executable_that_differs_from_its_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = {"LOCALAPPDATA": str(tmp_path)}
    archive = _archive(b"an unexpected build\n")
    # The archive matches its pin; the executable it carries does not.
    monkeypatch.setattr(local_livekit, "ARCHIVE_SHA256", _sha256(archive))
    monkeypatch.setattr(local_livekit, "EXECUTABLE_SHA256", _sha256(b"synthetic livekit\n"))

    with pytest.raises(local_livekit.LiveKitUnavailable, match="executable differs"):
        local_livekit.install(environ, fetch=lambda url: archive)
    assert not local_livekit.shared_path(environ).parent.exists()


def test_install_refuses_an_archive_the_qualification_inspection_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("LICENSE", b"notice\n")
    archive = stream.getvalue()
    environ = {"LOCALAPPDATA": str(tmp_path)}
    monkeypatch.setattr(local_livekit, "ARCHIVE_SHA256", _sha256(archive))

    with pytest.raises(local_livekit.LiveKitUnavailable, match="member differs"):
        local_livekit.install(environ, fetch=lambda url: archive)
    assert not local_livekit.shared_path(environ).parent.exists()


def test_the_download_reads_one_byte_past_the_qualification_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

    requested: list[str] = []

    def urlopen(url: str, timeout: float) -> Response:
        requested.append(url)
        return Response(b"x" * 16)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(livekit_files, "_MAX_ARCHIVE", 8)

    assert local_livekit._download(local_livekit.ARCHIVE_URL) == b"x" * 9
    assert requested == [local_livekit.ARCHIVE_URL]


def test_install_replaces_a_server_that_differs_from_its_pin(
    tmp_path: Path, release: tuple[bytes, bytes]
) -> None:
    archive, executable = release
    environ = {"LOCALAPPDATA": str(tmp_path)}
    path = local_livekit.shared_path(environ)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"a stale build\n")

    assert local_livekit.install(environ, fetch=lambda url: archive) == path
    assert path.read_bytes() == executable


def test_the_verified_server_refuses_a_missing_or_altered_copy(
    tmp_path: Path, release: tuple[bytes, bytes]
) -> None:
    _, executable = release
    environ = {"LOCALAPPDATA": str(tmp_path)}
    path = local_livekit.shared_path(environ)

    with pytest.raises(local_livekit.LiveKitUnavailable, match="not installed"):
        local_livekit.verified_server(environ)
    path.parent.mkdir(parents=True)
    path.write_bytes(executable + b"tampered")
    with pytest.raises(local_livekit.LiveKitUnavailable, match="SHA-256"):
        local_livekit.verified_server(environ)
    path.write_bytes(executable)
    assert local_livekit.verified_server(environ) == path


def test_every_launcher_starts_development_mode_with_signaling_on_loopback() -> None:
    assert local_livekit.server_command(Path("livekit-server.exe")) == [
        "livekit-server.exe",
        "--dev",
        "--bind",
        "127.0.0.1",
    ]


def test_every_local_launcher_resolves_the_server_through_the_helper() -> None:
    rehearsal = (_ROOT / "scripts/rehearse_desktop_mvp.py").read_text(encoding="utf-8")
    browser = (_ROOT / "tests/integration/test_browser_self_acceptance.py").read_text(
        encoding="utf-8"
    )

    assert "local_livekit.verified_server()" in rehearsal
    assert "local_livekit.server_command(" in rehearsal
    assert "local_livekit.DEVELOPMENT_KEYS" in rehearsal
    assert local_livekit.EXECUTABLE_SHA256 not in rehearsal
    assert "local_livekit.verified_server()" in browser
    assert "local_livekit.server_command(executable)" in browser
    for source in (rehearsal, browser):
        assert ".tools" not in source
        assert '"--dev", "--bind"' not in source


def test_install_removes_its_staged_copy_when_placing_it_fails(
    tmp_path: Path, release: tuple[bytes, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _ = release
    environ = {"LOCALAPPDATA": str(tmp_path)}
    path = local_livekit.shared_path(environ)

    def refuse(source: str, target: Path) -> None:
        raise PermissionError("the server is in use")

    monkeypatch.setattr(local_livekit.os, "replace", refuse)

    with pytest.raises(PermissionError):
        local_livekit.install(environ, fetch=lambda url: archive)
    assert list(path.parent.iterdir()) == []


def test_the_path_command_prints_only_a_verified_server(
    tmp_path: Path,
    release: tuple[bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, executable = release
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    path = local_livekit.shared_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(executable + b"tampered")

    assert local_livekit.main(["path"]) == 1
    assert capsys.readouterr().out == ""
    path.write_bytes(executable)
    assert local_livekit.main(["path"]) == 0
    assert capsys.readouterr().out == f"{path}\n"


class _Launched:
    calls: list[tuple[list[str], object, dict[str, str]]] = []

    def __init__(self, command: list[str], *, stdin: object, env: dict[str, str]) -> None:
        self.calls.append((command, stdin, env))

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_serve_launches_the_verified_server_with_the_development_keys(
    tmp_path: Path, release: tuple[bytes, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, executable = release
    environ = {"LOCALAPPDATA": str(tmp_path)}
    path = local_livekit.shared_path(environ)
    path.parent.mkdir(parents=True)
    path.write_bytes(executable)
    monkeypatch.setattr(_Launched, "calls", [])
    monkeypatch.setattr(local_livekit.subprocess, "Popen", _Launched)

    assert local_livekit.serve(environ) == 0
    [(command, stdin, env)] = _Launched.calls
    assert command == local_livekit.server_command(path)
    assert stdin is subprocess.DEVNULL
    assert env["LIVEKIT_KEYS"] == local_livekit.DEVELOPMENT_KEYS


def test_serve_refuses_an_altered_server_before_launching_it(
    tmp_path: Path, release: tuple[bytes, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, executable = release
    environ = {"LOCALAPPDATA": str(tmp_path)}
    path = local_livekit.shared_path(environ)
    path.parent.mkdir(parents=True)
    path.write_bytes(executable + b"tampered")
    monkeypatch.setattr(_Launched, "calls", [])
    monkeypatch.setattr(local_livekit.subprocess, "Popen", _Launched)

    with pytest.raises(local_livekit.LiveKitUnavailable, match="SHA-256"):
        local_livekit.serve(environ)
    assert _Launched.calls == []
