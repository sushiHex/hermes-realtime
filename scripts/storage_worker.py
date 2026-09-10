"""Private packaged-spool driver; checkpoints carry no source text or paths."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from scripts.equivalence_process import _read_frame, _require, _require_ack, _write_frame


def _make_spool(case_root: Path, *, clock: Any = None) -> Any:
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool
    from hermes_realtime.evidence.storage_security import WindowsStorageProbeV1
    from tests.evidence import spool_crash_worker as driver

    root = case_root / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    return SQLiteEvidenceSpool(
        root / "capture-v1.sqlite3",
        clock=clock or driver._Clock(),
        uuid_factory=driver._Uuids(),
        probe=WindowsStorageProbeV1(),
        connection_factory=driver._CrashConnection,
    )


def main() -> None:
    _require(
        os.name == "nt"
        and bool(sys.flags.isolated)
        and not sys.flags.optimize
        and len(sys.argv) == 3,
        "storage worker launch differs",
    )
    import msvcrt

    import hermes_realtime
    from scripts.storage_observation import observe_storage
    from tests.evidence import spool_crash_worker as driver

    package = Path(sys.path[0]).resolve(strict=True)
    _require(
        Path(hermes_realtime.__file__).resolve().is_relative_to(package),
        "storage imported candidate source instead of wheel",
    )
    request_handle, response_handle = (int(value) for value in sys.argv[1:])
    for handle in (request_handle, response_handle):
        os.set_handle_inheritable(handle, False)
    request = msvcrt.open_osfhandle(request_handle, os.O_RDONLY | os.O_BINARY)
    response = msvcrt.open_osfhandle(response_handle, os.O_WRONLY | os.O_BINARY)
    config: Any = None
    try:
        config = _read_frame(request, time.monotonic() + 10)
        _require(
            type(config) is dict
            and set(config)
            == {"version", "nonce", "point", "mode", "action", "clock", "workspace"},
            "storage configuration fields differ",
        )
        point, mode = config["point"], config["mode"]
        _require(
            type(config["version"]) is int
            and config["version"] == 1
            and point in driver.FAILPOINTS
            and type(mode) is int
            and mode in driver.EXIT_MODES,
            "storage checkpoint differs",
        )
        _require(
            config["clock"] in {"caught-up", "regressed"}
            and config["action"] in {"crash", "recover"},
            "storage operation differs",
        )
        workspace = Path(config["workspace"]).resolve(strict=True)
        _require(workspace == Path.cwd().resolve(strict=True), "storage workspace differs")
        case_root = workspace / f"{point}-exit{mode}-{config['clock']}"

        def emit(observation: dict[str, Any]) -> None:
            _write_frame(
                response,
                {
                    "version": 1,
                    "nonce": config["nonce"],
                    "point": point,
                    "mode": mode,
                    "action": config["action"],
                    "pid": os.getpid(),
                    "observation": observation,
                },
            )
            _require_ack(_read_frame(request, time.monotonic() + 10), config["nonce"], 0)

        if config["action"] == "crash":
            driver._make_spool = _make_spool

            def crashpoint(failpoint: str, exit_mode: int) -> None:
                _require(
                    (failpoint, exit_mode) == (point, mode), "storage reached another checkpoint"
                )
                _write_frame(
                    response,
                    {
                        "version": 1,
                        "nonce": config["nonce"],
                        "point": point,
                        "mode": mode,
                        "action": "crash",
                        "pid": os.getpid(),
                        "observation": {
                            "checkpoint": failpoint,
                            "vacuum_returned": driver._VACUUM_RETURNED,
                        },
                    },
                )
                if mode == 197:
                    _require_ack(_read_frame(request, time.monotonic() + 10), config["nonce"], 0)
                    os._exit(197)
                threading.Event().wait()

            driver._crashpoint = crashpoint
            driver.main(
                ["--case-root", str(case_root), "--failpoint", point, "--exit-mode", str(mode)]
            )
            raise ValueError("storage crash driver returned")
        before = observe_storage(case_root / "evidence")
        baseline = {}
        if point in driver.FAILPOINTS[28:32]:
            baseline = json.loads(
                (case_root / "rollback-baseline.json").read_text(encoding="utf-8")
            )
        elif point == "after_full_purge_marker_fsync":
            baseline = json.loads(
                (case_root / "full-purge-baseline.json").read_text(encoding="utf-8")
            )
        recovered = _make_spool(
            case_root,
            clock=driver._Clock(
                driver.START + timedelta(hours=2 if config["clock"] == "caught-up" else -2)
            ),
        )
        try:
            disposition = recovered.recover_existing().value
        finally:
            recovered.close()
        after = observe_storage(case_root / "evidence")
        emit({"before": before, "disposition": disposition, "after": after, "baseline": baseline})
    except Exception as error:
        import traceback

        if type(config) is dict and set(config) == {
            "version",
            "nonce",
            "point",
            "mode",
            "action",
            "clock",
            "workspace",
        }:
            kind = (
                "timeout"
                if isinstance(error, TimeoutError)
                else "value"
                if isinstance(error, ValueError)
                else "os"
                if isinstance(error, OSError)
                else "other"
            )
            frames = traceback.extract_tb(error.__traceback__)
            owned = [
                frame
                for frame in frames
                if Path(frame.filename).name in {"storage_worker.py", "storage_observation.py"}
            ]
            line = (owned or frames)[-1].lineno
            _write_frame(
                response,
                {
                    "version": 1,
                    "nonce": config["nonce"],
                    "point": config["point"],
                    "mode": config["mode"],
                    "action": config["action"],
                    "pid": os.getpid(),
                    "observation": {"failure": kind, "source_line": line},
                },
            )
            _read_frame(request, time.monotonic() + 10)
        raise
    finally:
        os.close(request)
        os.close(response)


if __name__ == "__main__":
    main()
