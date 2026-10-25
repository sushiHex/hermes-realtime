"""Explicit disposable backend setup. Preparation never starts a process or task.

Prepared Hermes/dependencies are trusted prerequisites. Equality here covers the
supplied realtime wheel/uv target, not the complete interpreter environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import secrets
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIN = "29112bef099274229cadff79cdff7bf7b99c4b77"
FIELDS = frozenset(
    [
        "version",
        "candidate",
        "tree",
        "hermes_source",
        "hermes_python",
        "runtime_target",
        "wheel",
        "wheel_sha256",
        "ollama_runtime",
        "ollama_archive",
        "ollama_archive_sha256",
        "model_store",
        "model",
        "model_manifest_sha256",
        "model_port",
        "api_port",
        "companion_port",
        "objective",
        "presentation_key_env",
    ]
)


def require(value: Any, category: Any) -> Any:
    if not value:
        raise ValueError(category)


def validate_manifest(value: Any) -> Any:
    require(type(value) is dict and set(value) == FIELDS, "manifest_fields")
    require(type(value["version"]) is int and value["version"] == 1, "manifest_version")
    for name in (
        "candidate",
        "tree",
        "wheel_sha256",
        "ollama_archive_sha256",
        "model_manifest_sha256",
    ):
        size = 40 if name in {"candidate", "tree"} else 64
        require(
            type(value[name]) is str and re.fullmatch(rf"[0-9a-f]{{{size}}}", value[name]),
            "manifest_digest",
        )
    for name in (
        "hermes_source",
        "hermes_python",
        "runtime_target",
        "wheel",
        "ollama_runtime",
        "ollama_archive",
        "model_store",
    ):
        require(
            type(value[name]) is str
            and 0 < len(value[name]) <= 1024
            and Path(value[name]).is_absolute(),
            "manifest_path",
        )
    ports = [value[name] for name in ("model_port", "api_port", "companion_port")]
    require(
        all(type(p) is int and 1024 <= p <= 65535 for p in ports)
        and len(set(ports)) == 3
        and 11434 not in ports,
        "manifest_ports",
    )
    require(
        type(value["model"]) is str
        and re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}:[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", value["model"]
        ),
        "manifest_model",
    )
    require(
        type(value["objective"]) is str
        and value["objective"].strip()
        and len(value["objective"]) <= 1024,
        "manifest_objective",
    )
    require(
        type(value["presentation_key_env"]) is str
        and re.fullmatch(r"(?:HERMES|LOOKUP)_[A-Z0-9_]{1,100}_KEY", value["presentation_key_env"]),
        "manifest_key_name",
    )
    return dict(value)


def read_json(path: Any, bound: Any = 65536) -> Any:
    with Path(path).open("rb") as stream:
        raw = stream.read(bound + 1)
    require(len(raw) <= bound, "json_bound")

    def pairs(items: Any) -> Any:
        require(len(dict(items)) == len(items), "json_duplicate")
        return dict(items)

    return json.loads(raw, object_pairs_hook=pairs)


def digest(path: Any) -> Any:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(path: Any, *args: Any) -> Any:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ).stdout.strip()


def safe_member(name: Any) -> Any:
    require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_./-]{1,512}", name), "member_name")
    require(
        all(part not in {"", ".", ".."} and not part.endswith(".") for part in name.split("/")),
        "member_path",
    )


def reject_links(path: Any) -> Any:
    """Inspect lexical components before resolve can erase a junction/symlink."""
    path = Path(path)
    for component in (path, *path.parents):
        try:
            observed = component.lstat()
        except FileNotFoundError:
            continue
        require(
            not component.is_symlink() and not (getattr(observed, "st_file_attributes", 0) & 0x400),
            "path_link",
        )


def protect_directory(path: Any) -> Any:
    """Reuse the existing Windows DACL mechanism, with child inheritance enabled."""
    from scripts.qualification_execution_files import (
        _open_security_control,
        _security_sddl,
        _set_security,
    )
    from scripts.windows_storage_oracle import _api

    kernel, _ = _api()
    handle, _ = _open_security_control(path, directory=True)
    try:
        # Same principals and rights as the existing writable workspace; only
        # inheritance differs because this gateway creates its own descendants.
        sddl = _security_sddl(directory=True, frozen=False).replace("(A;;", "(A;OICI;")
        _set_security(handle, sddl)
    finally:
        require(bool(kernel.CloseHandle(handle)), "private_directory_handle")


def verify_record(raw: Any, record: Any, members: Any) -> Any:
    require(len(raw) <= 1024**2, "record_bound")
    rows = list(csv.reader(io.StringIO(raw.decode()), strict=True))
    require(
        all(len(row) == 3 for row in rows)
        and len(rows) == len(members)
        and len({row[0] for row in rows}) == len(rows)
        and {row[0] for row in rows} == set(members),
        "record_set",
    )
    for name, encoded, size in rows:
        safe_member(name)
        if name == record:
            require(encoded == size == "", "record_self")
        else:
            expected = base64.urlsafe_b64encode(hashlib.sha256(members[name]).digest())
            require(
                (encoded, size)
                == ("sha256=" + expected.rstrip(b"=").decode(), str(len(members[name]))),
                "record_bytes",
            )


def verify_runtime(m: Any) -> Any:
    """Migrated real20 scoped wheel/source/uv-target comparator; no install action."""
    require(
        git(ROOT, "rev-parse", "HEAD") == m["candidate"]
        and git(ROOT, "rev-parse", "HEAD^{tree}") == m["tree"],
        "candidate_binding",
    )
    require(not git(ROOT, "status", "--porcelain=v1"), "candidate_dirty")
    require(
        git(m["hermes_source"], "rev-parse", "HEAD") == PIN
        and not git(m["hermes_source"], "diff", "--name-only", "HEAD", "--", "*.py"),
        "hermes_binding",
    )
    # This remains a trusted prepared interpreter/dependency environment, not a
    # complete environment attestation. Refuse ordinary untracked import sources
    # as well as tracked edits; ignored dependency/cache roots are prerequisites.
    require(
        not git(m["hermes_source"], "ls-files", "--others", "--exclude-standard", "--", "*.py"),
        "hermes_untracked_python",
    )
    require(digest(m["wheel"]) == m["wheel_sha256"], "wheel_digest")
    with zipfile.ZipFile(m["wheel"]) as archive:
        names = archive.namelist()
        require(
            0 < len(names) <= 4096
            and len(set(names)) == len(names)
            and len({n.casefold() for n in names}) == len(names),
            "wheel_names",
        )
        for name in names:
            safe_member(name)
        require(sum(item.file_size for item in archive.infolist()) <= 64 * 1024**2, "wheel_bound")
        members = {name: archive.read(name) for name in names}
    infos = {name.split("/")[0] for name in members if ".dist-info/" in name}
    require(len(infos) == 1, "wheel_distribution")
    info = infos.pop()
    require(
        info.startswith("hermes_realtime-")
        and all(name.startswith(("hermes_realtime/", info + "/")) for name in members),
        "wheel_namespace",
    )
    record = info + "/RECORD"
    verify_record(members[record], record, members)
    source_names = git(
        ROOT, "ls-tree", "-r", "--name-only", m["candidate"], "--", "src/hermes_realtime"
    ).splitlines()
    source = {name.removeprefix("src/"): (ROOT / name).read_bytes() for name in source_names}
    require(
        source
        and source == {n: raw for n, raw in members.items() if n.startswith("hermes_realtime/")},
        "source_wheel",
    )
    reject_links(m["runtime_target"])
    target = Path(m["runtime_target"]).resolve(strict=True)
    installed = {}
    for path in target.rglob("*"):
        require(
            not path.is_symlink() and not (getattr(path.lstat(), "st_file_attributes", 0) & 0x400),
            "target_link",
        )
        if path.is_file():
            name = path.relative_to(target).as_posix()
            safe_member(name)
            require(
                path.stat().st_nlink == 1 and path.stat().st_size <= 64 * 1024**2, "target_file"
            )
            installed[name] = path.read_bytes()
            require(
                len(installed) <= 4096 and sum(map(len, installed.values())) <= 64 * 1024**2,
                "target_bound",
            )
    generated = {
        info + "/" + name
        for name in ("INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json", "RECORD")
    }
    payload = {name: raw for name, raw in members.items() if name != record}
    require(
        set(installed)
        == set(payload)
        | generated
        | {".lock", "bin/hermes-realtime-host.exe", "bin/hermes-realtime-local.exe"},
        "target_set",
    )
    require(
        installed[".lock"] == b""
        and installed[info + "/INSTALLER"] == b"uv"
        and installed[info + "/REQUESTED"] == b"",
        "target_origin",
    )
    require(all(installed[n] == raw for n, raw in payload.items()), "target_bytes")
    verify_record(
        installed[record], record, {n: raw for n, raw in installed.items() if n != ".lock"}
    )
    return {
        "packageRoot": str(target / "hermes_realtime"),
        "recordPath": str(target / record),
        "recordSha256": digest(target / record),
        "packageFiles": len(source),
        "wheelSha256": m["wheel_sha256"],
        "scope": "realtime-wheel-target-only",
    }


def profile_config(m: Any) -> Any:
    route = f"http://127.0.0.1:{m['model_port']}/v1"
    return {
        "approvals": {"mode": "manual", "unattended_mode": "deny"},
        "auxiliary": {"title_generation": {"enabled": False}},
        "model": {
            "provider": "custom",
            "default": m["model"],
            "base_url": route,
            "api_key": "local",
            "context_length": 64000,
        },
        "platform_toolsets": {"api_server": ["web"]},
        "plugins": {
            "enabled": ["hermes-realtime", "ordinary-lookup-experimental"],
            "entries": {
                "hermes-realtime": {"allow_tool_override": False},
                "ordinary-lookup-experimental": {
                    "allow_tool_override": True,
                    "settings": {
                        "presentation": {
                            "provider": "custom",
                            "base_url": route,
                            "model": m["model"],
                            "api_key_env": m["presentation_key_env"],
                        }
                    },
                },
            },
        },
        "web": {
            "cache_enabled": False,
            "search_backend": "exa",
            "extract_backend": "exa",
            "extract_char_limit": 12000,
            "keyless_fallback": True,
            "keyless_rescue": False,
            "provider_tier": {
                "exa": "free",
                "firecrawl": "paid",
                "keenable": "paid",
                "parallel": "paid",
            },
        },
    }


def prepare(m: Any, output: Any) -> Any:
    """Verify before writes, then stage first into a fresh caller-owned profile."""
    require(os.name == "nt", "windows_required")
    runtime = verify_runtime(m)
    from scripts.lookup_backend_model import verify_artifacts

    verified = verify_artifacts(m, __import__("time").monotonic() + 600)
    require(os.environ.get(m["presentation_key_env"]) == "local", "local_key_required")
    reject_links(output)
    output = Path(output).resolve()
    require(
        not output.exists()
        and all(
            not output.is_relative_to(Path(m[name]).resolve())
            and not Path(m[name]).resolve().is_relative_to(output)
            for name in ("hermes_source", "runtime_target", "ollama_runtime", "model_store")
        ),
        "output_scope",
    )
    require(not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output), "output_scope")
    output.mkdir(parents=False)
    protect_directory(output)  # Before a credential, objective or raw log is written.
    profile = output / "home"
    profile.mkdir()
    from scripts.stage_lookup_policy import stage_lookup_policy

    artifact = stage_lookup_policy(profile, ROOT / "tests/fixtures/plugin_generation_candidate")
    (profile / "config.yaml").write_text(json.dumps(profile_config(m)), encoding="utf-8")
    token = secrets.token_urlsafe(32)
    (profile / ".env").write_text(
        f"API_SERVER_KEY={token}\nHERMES_REALTIME_COMPANION_TOKEN={secrets.token_urlsafe(32)}\n"
        f"HERMES_REALTIME_COMPANION_PORT={m['companion_port']}\n"
        f"{m['presentation_key_env']}=local\n",
        encoding="utf-8",
    )
    prepared = {
        **m,
        **runtime,
        "profile": str(profile),
        "output": str(output),
        "baseUrl": f"http://127.0.0.1:{m['api_port']}",
        "artifact": artifact,
        "artifactVerification": dict(verified.facts),
    }
    (output / "prepared.json").write_text(json.dumps(prepared), encoding="utf-8")
    prepared["_verified_artifacts"] = verified
    return prepared


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result: dict[str, Any] = {"category": "refused", "version": 1}
    try:
        m = validate_manifest(read_json(args.manifest))
        prepared = prepare(m, args.output)
        if args.execute:
            from scripts.lookup_backend_task import run_owned

            result = run_owned(prepared)
        else:
            result = {"category": "prepared", "version": 1, "processes_started": 0}
    except Exception as error:
        result["failure_kind"] = type(error).__name__
        result["failure_category"] = "validation_refused" if type(error) is ValueError else "other"
        # Preserve only the model helper's existing fixed, source-owned enum.
        # Arbitrary native ValueError text must never become public output.
        model_module = sys.modules.get("scripts.lookup_backend_model")
        allowed: frozenset[str] = getattr(model_module, "_REFUSALS", frozenset())
        if (
            type(error) in {ValueError, RuntimeError, TimeoutError}
            and len(error.args) == 1
            and type(error.args[0]) is str
            and error.args[0] in allowed
        ):
            result["failure_category"] = error.args[0]
    print("[lookup-backend-setup] " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["category"] in {"prepared", "completed"} else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
