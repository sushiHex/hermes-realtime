"""The one shared, pinned, verified LiveKit development server for every local launcher.

Windows Firewall keys its decisions to a program's path. A copy of the server in each
checkout asks again from every worktree and clone, so every local launcher runs the one copy
installed per user at ``%LOCALAPPDATA%\\hermes-realtime\\tools\\livekit-<version>\\``, outside
every checkout, and verifies its SHA-256 just before each launch. One path is one firewall
decision.

``--bind 127.0.0.1`` holds only signaling to loopback; LiveKit 1.13.4 opens its RTC sockets on
every interface. A configuration can hold those to loopback too (``rtc.tcp_port: 0`` with
``rtc.ips`` admitting only 127.0.0.1), but then no client can connect: libwebrtc gathers ICE
candidates only on non-loopback adapters, and Windows refuses to send from such an address to
127.0.0.1.

    uv run python -m scripts.local_livekit install   # download, verify and place it once
    uv run python -m scripts.local_livekit path      # print the verified executable
    uv run python -m scripts.local_livekit serve     # run it in development mode until Ctrl-C

Resolving and verifying the server needs only the standard library, so a script run out of
``scripts/`` can import this as ``local_livekit``. Installing reuses the qualification's
bounded archive inspection and so runs as ``scripts.local_livekit``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path

VERSION = "1.13.4"
ARCHIVE_URL = (
    f"https://github.com/livekit/livekit/releases/download/v{VERSION}/"
    f"livekit_{VERSION}_windows_amd64.zip"
)
ARCHIVE_SHA256 = "a326e025de516e93dfb3719bcd28e5a4ac16f21bcf1ef562499403ca98cc65fe"
EXECUTABLE_SHA256 = "4d60c4043c8c6ff34845727587c7a7f86946d92c390b879ea35ad3793fcbd916"
# The public loopback development credentials every local gate already uses.
DEVELOPMENT_KEYS = "devkey: local-" + "x" * 32 + "\n"
_EXECUTABLE = "livekit-server.exe"
_INSTALL = "run: uv run python -m scripts.local_livekit install"
_REMOTE_INSTALL = "run: uv run python -m scripts.local_livekit install-remote"


class LiveKitUnavailable(RuntimeError):
    """The shared server is missing or differs from its pin; the message names the fix."""


def shared_path(environ: Mapping[str, str] | None = None) -> Path:
    """Where the one shared server lives: once per user, outside every checkout."""

    return _profile_path(environ, f"livekit-{VERSION}")


def remote_path(environ: Mapping[str, str] | None = None) -> Path:
    """The separate per-user executable for the explicit tailnet profile."""

    return _profile_path(environ, f"livekit-{VERSION}-tailnet")


def _profile_path(environ: Mapping[str, str] | None, directory: str) -> Path:
    base = (os.environ if environ is None else environ).get("LOCALAPPDATA", "")
    if not base or not Path(base).is_absolute():
        raise LiveKitUnavailable("LOCALAPPDATA must name an absolute per-user directory")
    return Path(base) / "hermes-realtime" / "tools" / directory / _EXECUTABLE


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verified_server(environ: Mapping[str, str] | None = None) -> Path:
    """The shared server, only once its bytes match the pin."""

    return _verified_at_path(shared_path(environ), _INSTALL, "the shared LiveKit server")


def verified_remote_server(environ: Mapping[str, str] | None = None) -> Path:
    """The separate tailnet server, only once its bytes match the same pin."""

    return _verified_at_path(remote_path(environ), _REMOTE_INSTALL, "the remote LiveKit server")


def _verified_at_path(path: Path, install_hint: str, label: str) -> Path:
    if not path.is_file():
        raise LiveKitUnavailable(f"LiveKit {VERSION} is not installed; {install_hint}")
    if _file_sha256(path) != EXECUTABLE_SHA256:
        raise LiveKitUnavailable(f"{label} differs from its SHA-256; {install_hint}")
    return path


def _download(url: str) -> bytes:
    from scripts import qualification_livekit_files as livekit_files

    with urllib.request.urlopen(url, timeout=120) as response:
        # One byte past the bound, so the inspection below refuses an oversized archive.
        payload: bytes = response.read(livekit_files._MAX_ARCHIVE + 1)
    return payload


def _pinned_executable(archive: bytes) -> bytes:
    from scripts import qualification_livekit_files as livekit_files

    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise LiveKitUnavailable("the downloaded LiveKit archive differs from its SHA-256")
    try:
        files, _ = livekit_files._inspect_archive(archive)
    except ValueError as error:
        raise LiveKitUnavailable(f"the LiveKit archive is refused: {error}") from error
    executable: bytes = files[livekit_files._ARCHIVE_MEMBER]
    if hashlib.sha256(executable).hexdigest() != EXECUTABLE_SHA256:
        raise LiveKitUnavailable("the LiveKit executable differs from its SHA-256")
    return executable


def install(
    environ: Mapping[str, str] | None = None,
    *,
    fetch: Callable[[str], bytes] = _download,
) -> Path:
    """Place the pinned server at the shared path; a verified copy is kept as it is."""

    return _install_at_path(shared_path(environ), _INSTALL, "the shared LiveKit server", fetch)


def install_remote(
    environ: Mapping[str, str] | None = None,
    *,
    fetch: Callable[[str], bytes] = _download,
) -> Path:
    """Place the same pinned bytes at the distinct tailnet path without launching them."""

    return _install_at_path(
        remote_path(environ), _REMOTE_INSTALL, "the remote LiveKit server", fetch
    )


def _install_at_path(
    path: Path, install_hint: str, label: str, fetch: Callable[[str], bytes]
) -> Path:
    with suppress(LiveKitUnavailable):
        return _verified_at_path(path, install_hint, label)
    executable = _pinned_executable(fetch(ARCHIVE_URL))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(dir=path.parent, prefix=".livekit-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(executable)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(staged)
        raise
    return _verified_at_path(path, install_hint, label)


def server_command(executable: Path) -> list[str]:
    """The command every local launcher starts: development mode, signaling on loopback."""

    return [str(executable), "--dev", "--bind", "127.0.0.1"]


def serve(environ: Mapping[str, str] | None = None) -> int:
    """Run the verified server in the foreground until it exits or Ctrl-C."""

    command = server_command(verified_server(environ))
    environment = dict(os.environ) | {"LIVEKIT_KEYS": DEVELOPMENT_KEYS}
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=environment)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return process.wait(timeout=10)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="download, verify and place the shared server")
    commands.add_parser("path", help="print the verified shared server")
    commands.add_parser("serve", help="run the shared server until Ctrl-C")
    commands.add_parser("install-remote", help="install the separate verified tailnet server")
    commands.add_parser("path-remote", help="print the verified tailnet server path")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "install":
            print(install())
        elif arguments.command == "path":
            print(verified_server())
        elif arguments.command == "install-remote":
            print(install_remote())
        elif arguments.command == "path-remote":
            print(verified_remote_server())
        else:
            return serve()
    except LiveKitUnavailable as error:
        print(f"local-livekit: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
