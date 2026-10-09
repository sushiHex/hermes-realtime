"""Shared credential, identity and loopback helpers for installed-boundary gates."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

# The owner-selected qualification baseline (#67, #159): a reference, not a version ceiling.
HERMES_BASELINE = {"version": "0.21.0", "commit": "29112bef099274229cadff79cdff7bf7b99c4b77"}
_UPSTREAM = "https://github.com/NousResearch/hermes-agent.git"
PINNED_HERMES = (
    Path(__file__).resolve().parents[1]
    / ".hermes"
    / "bench"
    / f"hermes-{HERMES_BASELINE['commit'][:12]}"
)


class SetupRunner(Protocol):
    """Run one named setup command to completion within ``timeout``, or raise."""

    def __call__(
        self,
        category: str,
        argv: list[str],
        *,
        cwd: Path,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


def _checked_run(
    category: str,
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    del category
    return subprocess.run(
        argv,
        cwd=cwd,
        env=None if env is None else dict(env),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def provision_pinned_hermes(run: SetupRunner = _checked_run) -> Path:
    """Install the baseline Hermes the way its installer does, reusing the cache when current.

    Returns the interpreter of its environment; the checkout is ``PINNED_HERMES / "source"``.
    Every command is bounded and goes through ``run``, so a caller can name its failures.
    """

    source = PINNED_HERMES / "source"
    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").exists():
        run("hermes_cache", ["git", "init", "-q"], cwd=source, timeout=60)
        run("hermes_cache", ["git", "remote", "add", "origin", _UPSTREAM], cwd=source, timeout=60)
        # The documentation site is not runtime code and exceeds Windows path limits.
        run(
            "hermes_cache",
            ["git", "sparse-checkout", "set", "--no-cone", "/*", "!/website/"],
            cwd=source,
            timeout=60,
        )
    if _cached_head(run, source) != HERMES_BASELINE["commit"]:
        run(
            "hermes_cache",
            ["git", "fetch", "-q", "--depth", "1", "origin", HERMES_BASELINE["commit"]],
            cwd=source,
            timeout=600,
        )
        run(
            "hermes_cache", ["git", "checkout", "-q", "--detach", "FETCH_HEAD"],
            cwd=source, timeout=600,
        )  # fmt: skip
    if _cached_head(run, source) != HERMES_BASELINE["commit"]:
        raise RuntimeError("the Hermes cache is not at the pinned commit")
    venv = PINNED_HERMES / "venv"
    run(
        "hermes_cache_sync",
        ["uv", "sync", "--extra", "all", "--locked", "--python", "3.11", "--quiet"],
        cwd=source,
        env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(venv)},
        timeout=1800,
    )
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _cached_head(run: SetupRunner, source: Path) -> str:
    """The cache's HEAD commit, or "(initial)" before its first checkout; never an error."""

    status = run(
        "hermes_cache",
        ["git", "status", "--porcelain=v2", "--branch", "--untracked-files=no"],
        cwd=source,
        timeout=60,
    ).stdout
    for line in status.splitlines():
        if line.startswith("# branch.oid "):
            return line.removeprefix("# branch.oid ").strip()
    return ""


def load_api_key(env_file: Path) -> str:
    """Load exactly one strong API server bearer without logging it."""

    if not isinstance(env_file, Path):
        raise TypeError("Hermes env file must be a Path")
    matches: list[str] = []
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("API_SERVER_KEY="):
            matches.append(line.split("=", 1)[1].strip().strip('"').strip("'"))
    if len(matches) != 1 or len(matches[0]) < 32:
        raise RuntimeError("Hermes env file must contain one explicit strong API_SERVER_KEY")
    return matches[0]


def installed_hermes_identity(version: str, checkout: Path) -> dict[str, object]:
    """Name the exact Hermes a gate ran against, and whether it is the qualified baseline.

    Evidence binds to a commit, so a checkout whose tracked files differ from every commit is
    refused. Untracked files, such as the installer's virtual environment, do not change it.
    """

    refusal: str | None = "unidentifiable"
    try:
        commit = _git(checkout, "rev-parse", "HEAD").strip()
        refusal = "modified"
        status = _git(checkout, "status", "--porcelain", "-z", "--untracked-files=no")
        changed = {entry[3:] for entry in status.split("\0") if entry}
        if changed - _unrepresentable_paths(checkout):
            raise RuntimeError("installed Hermes has local changes that no commit describes")
        refusal = None
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "installed Hermes is not a git checkout, so no commit describes it"
        ) from None
    finally:
        if refusal is not None:
            evidence = json.dumps({"refusal": refusal, "version": 1}, separators=(",", ":"))
            print(f"[hermes-identity] {evidence}", flush=True)
    identity = {"version": version, "commit": commit}
    return identity | {"baseline": identity == HERMES_BASELINE}


def _unrepresentable_paths(checkout: Path) -> set[str]:
    """Committed paths a case-insensitive checkout cannot hold apart, left exactly as committed.

    Such spellings share one file on disk, so git reports all but one as changed. They are
    unchanged when that one file is byte-identical to one of their committed versions.
    """

    ignorecase = _git(checkout, "config", "--type=bool", "--default=false", "core.ignorecase")
    if ignorecase.strip() != "true":
        return set()
    spellings: dict[str, dict[str, str]] = defaultdict(dict)
    for entry in _git(checkout, "ls-tree", "-r", "-z", "--full-tree", "HEAD").split("\0"):
        if entry:
            header, path = entry.split("\t", 1)
            spellings[path.casefold()][path] = header.split()[2]
    exempt: set[str] = set()
    for committed in spellings.values():
        if len(committed) > 1:
            on_disk = _git(checkout, "hash-object", "--", next(iter(committed))).strip()
            if on_disk in committed.values():
                exempt.update(committed)
    return exempt


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=checkout, check=True, capture_output=True, text=True
    ).stdout


def available_port() -> int:
    """Reserve and release one currently available loopback TCP port."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
