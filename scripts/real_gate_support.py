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

try:
    import release_gate
except ModuleNotFoundError as error:
    if error.name != "release_gate":
        raise
    from scripts import release_gate

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
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
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
        if release_gate.has_hidden_index_entries(
            checkout, _git(checkout, "ls-files", "-v", "-z"), allow_sparse_website=True,
        ):
            raise RuntimeError("installed Hermes has hidden changes that no commit describes")
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

    Exempt only one physical directory entry: core.ignorecase can be copied to
    a case-sensitive filesystem, and separate hardlinks can share file identity.
    Single-link regular aliases must equal a committed variant too.
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
            paths = [checkout / name for name in committed]
            try:
                if any(
                    path.is_symlink() or not path.is_file()
                    or path.stat().st_nlink != 1 or not path.samefile(paths[0])
                    for path in paths
                ):
                    continue
            except OSError:
                continue
            on_disk = _git(checkout, "hash-object", "--", next(iter(committed))).strip()
            if on_disk in committed.values():
                exempt.update(committed)
    return exempt


def _git(checkout: Path, *arguments: str) -> str:
    return _checked_run(
        "hermes_identity", ["git", "-c", "core.fsmonitor=false", *arguments],
        cwd=checkout, timeout=60, env=release_gate.worktree_git_environment(checkout),
    ).stdout


def available_port() -> int:
    """Reserve and release one currently available loopback TCP port."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def hermes_namespace_tools(value: object) -> list[dict[str, object]] | None:
    """Return the function tools of the one exact Codex ``hermes`` namespace, or None.

    An empty ``dynamicTools`` list is valid and yields no tools. Any other shape is None.
    """

    if type(value) is not list:
        return None
    if not value:
        return []
    if len(value) != 1:
        return None
    namespace = value[0]
    if (
        type(namespace) is not dict
        or set(namespace) != {"type", "name", "description", "tools"}
        or namespace.get("type") != "namespace"
        or namespace.get("name") != "hermes"
    ):
        return None
    tools = namespace.get("tools")
    if type(tools) is not list or len(tools) > 16:
        return None
    for tool in tools:
        if (
            type(tool) is not dict
            or tool.get("type") != "function"
            or type(tool.get("name")) is not str
            or not 1 <= len(tool["name"]) <= 64
        ):
            return None
    return tools
