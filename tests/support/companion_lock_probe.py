"""One plugin registration in its own process, for the cross-process ownership test.

    python companion_lock_probe.py hold <data_dir> <control_dir>
    python companion_lock_probe.py contend <data_dir> <control_dir>

Both register this plugin, as Hermes does in every process, with the companion endpoint in
the environment and an in-memory Hermes. ``hold`` owns the companion until
``<control_dir>/stop`` appears. ``contend`` reports, as one JSON line, whether it owns a
companion and how many times it touched the Hermes port.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TESTS / "companion"))

from test_archive import FakeHermes  # noqa: E402

from hermes_realtime import hermes_plugin  # noqa: E402


class _State:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir


class _Context:
    def __init__(self, data_dir: Path) -> None:
        self.state = _State(data_dir)
        self.unload: list[Callable[[], None]] = []

    @property
    def subagent_lifecycle(self) -> object:
        return object()

    def on_unload(self, callback: Callable[[], None]) -> object:
        self.unload.append(callback)
        return object()


def main() -> None:
    mode, data_dir, control = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    hermes = FakeHermes()
    touched = {"open_port": 0}

    def open_port() -> FakeHermes:
        touched["open_port"] += 1
        return hermes

    hermes_plugin._open_archive_port = open_port  # type: ignore[assignment]
    context = _Context(data_dir)
    hermes_plugin.register(context)
    companion = hermes_plugin._companion
    if mode == "hold":
        if companion is None or not companion.wait_ready(30.0):
            raise SystemExit(2)
        (control / "ready").write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 60
        while not (control / "stop").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        for callback in context.unload:
            callback()
        return
    if companion is not None:
        companion.wait_ready(10.0)
    result = {
        "owned": companion is not None,
        "open_port": touched["open_port"],
        "hermes_calls": len(hermes.calls),
        "unload_callbacks": len(context.unload),
    }
    print("[probe] " + json.dumps(result, sort_keys=True), flush=True)
    for callback in context.unload:
        callback()


if __name__ == "__main__":
    main()
