"""Private normal-exit worker for the real full-purge fixture."""

from __future__ import annotations

import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from scripts.equivalence_process import _read_frame, _require, _require_ack, _write_frame
from scripts.full_purge_observation import observe_full_purge
from scripts.storage_worker import _make_spool


def _prepare(case: Path) -> None:
    from tests.evidence import spool_crash_worker as driver

    owned = _make_spool(case)
    try:
        _require(
            owned.create_epoch(driver._make_create_epoch()).value == "committed",
            "full-purge fixture did not commit",
        )
    finally:
        owned.close()
    for name, raw in {
        "purge-decoy.bin": b"not-owned",
        "capture-v1.sqlite3.backup": b"database-prefix-decoy",
        "capture-v1.sqlite3-wal.backup": b"sidecar-prefix-decoy",
        **{
            "capture-v1.sqlite3" + suffix: b"owned-sidecar"
            for suffix in ("-journal", "-wal", "-shm", "-vacuum", "-tmp")
        },
    }.items():
        with (case / "evidence" / name).open("xb") as file:
            file.write(raw)
    adjacent = case / "adjacent"
    adjacent.mkdir()
    (adjacent / "capture-v1.sqlite3").write_bytes(b"adjacent-database-decoy")
    (adjacent / "keep.bin").write_bytes(b"adjacent-decoy")


def _run(case: Path, phase: str) -> dict[str, Any]:
    from hermes_realtime.evidence.models import FullPurgeV1, SentinelState

    if phase == "purge":
        _prepare(case)
    before = observe_full_purge(case)
    owned = _make_spool(case)
    try:
        result = owned.purge_full_store(
            FullPurgeV1(
                protocol_version=1,
                full_purge_generation_id="40000000-0000-4000-8000-000000000096",
                sentinel_state=SentinelState.FULL_PURGE_PENDING,
                artifact_manifest_version=1,
            )
        ).value
    finally:
        owned.close()
    return {"phase": phase, "before": before, "result": result, "after": observe_full_purge(case)}


def main() -> None:
    _require(
        os.name == "nt"
        and bool(sys.flags.isolated)
        and not sys.flags.optimize
        and len(sys.argv) == 3,
        "full-purge worker launch differs",
    )
    import msvcrt

    import hermes_realtime

    _require(
        Path(hermes_realtime.__file__)
        .resolve()
        .is_relative_to(Path(sys.path[0]).resolve(strict=True)),
        "full purge imported source instead of wheel",
    )
    handles = tuple(int(value) for value in sys.argv[1:])
    for handle in handles:
        os.set_handle_inheritable(handle, False)
    request = msvcrt.open_osfhandle(handles[0], os.O_RDONLY | os.O_BINARY)
    response = msvcrt.open_osfhandle(handles[1], os.O_WRONLY | os.O_BINARY)
    try:
        config = _read_frame(request, time.monotonic() + 10)
        _require(
            type(config) is dict
            and set(config)
            == {
                "version",
                "nonce",
                "point",
                "mode",
                "action",
                "clock",
                "workspace",
            },
            "full-purge configuration differs",
        )
        _require(
            type(config["version"]) is int
            and config["version"] == 1
            and type(config["mode"]) is int
            and config["mode"] == 0
            and config["point"] == "full_purge_cleanup"
            and config["action"] in {"purge", "repeat_purge"}
            and config["clock"] == "caught-up"
            and type(config["nonce"]) is str
            and re.fullmatch("[0-9a-f]{64}", config["nonce"]) is not None,
            "full-purge operation differs",
        )
        workspace = Path(config["workspace"]).resolve(strict=True)
        _require(workspace == Path.cwd().resolve(strict=True), "full-purge workspace differs")
        try:
            observation = _run(workspace / "full_purge_cleanup-exit0-caught-up", config["action"])
        except Exception as error:
            frames = traceback.extract_tb(error.__traceback__)
            observation = {
                "failure": "value"
                if isinstance(error, ValueError)
                else "os"
                if isinstance(error, OSError)
                else "other",
                "source_line": frames[-1].lineno,
            }
            failed = True
        else:
            failed = False
        _write_frame(
            response,
            {key: config[key] for key in ("version", "nonce", "point", "mode", "action")}
            | {
                "pid": os.getpid(),
                "observation": observation,
            },
        )
        _require_ack(_read_frame(request, time.monotonic() + 10), config["nonce"], 0)
        _require(not failed, "full-purge worker failed")
    finally:
        os.close(request)
        os.close(response)


if __name__ == "__main__":
    main()
