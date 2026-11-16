"""A failed test's remaining phases stay inside the per-test timeout (#554).

pytest-timeout cancels its timer in ``pytest_exception_interact``, which pytest calls for every
failed phase. ``tests.support.failed_phase_timeout`` re-arms it. Each case runs a child pytest
with a short bound so a regression shows up as a bounded failure here, not as a hung suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_timeout import Settings

from tests.support import failed_phase_timeout

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "tests.support.failed_phase_timeout"
# Far above the child's own bound and startup, far below the suite's 300 s per-test bound.
CHILD_KILL_SECONDS = 60.0

HANGING_TEARDOWN = """
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
from pytest_timeout import Settings

STARTED = Path(__file__).with_name("started.txt")


@pytest.fixture
def hangs_in_teardown():
    yield
    threading.Event().wait()


@pytest.fixture
def fails_in_setup():
    raise RuntimeError("setup failed")


def test_call_fails(hangs_in_teardown):
    raise AssertionError("call failed")


def test_setup_fails(hangs_in_teardown, fails_in_setup):
    pass


def test_call_fails_late(hangs_in_teardown):
    STARTED.write_text(repr(time.time()), encoding="utf-8")
    time.sleep(SLEEP_SECONDS)
    raise AssertionError("call failed late")


def test_call_fails_cleanly():
    raise AssertionError("call failed")
"""


def _run_child(
    tmp_path: Path, test_name: str, timeout: float, sleep: float = 0.0, *extra: str
) -> tuple[int, str, float]:
    source = HANGING_TEARDOWN.replace("SLEEP_SECONDS", repr(sleep))
    (tmp_path / "test_child.py").write_text(source, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text(
        f"[pytest]\ntimeout = {timeout}\ntimeout_method = thread\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    log = tmp_path / "child.log"
    with log.open("wb") as output:
        child = subprocess.Popen(
            (
                sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", PLUGIN,
                "--rootdir", str(tmp_path), "-c", str(tmp_path / "pytest.ini"),
                *extra, f"{tmp_path / 'test_child.py'}::{test_name}",
            ),
            cwd=tmp_path,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            returncode = child.wait(timeout=CHILD_KILL_SECONDS)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            pytest.fail(
                f"{test_name}: the child pytest was still running after {CHILD_KILL_SECONDS} s; "
                "teardown after the failure was not bounded"
            )
        exited = time.time()
    return returncode, log.read_text(encoding="utf-8", errors="replace"), exited


@pytest.mark.parametrize("test_name", ["test_call_fails", "test_setup_fails"])
def test_teardown_after_a_failed_phase_ends_at_the_bound_with_a_stack_dump(
    tmp_path: Path, test_name: str
) -> None:
    returncode, output, _ = _run_child(tmp_path, test_name, timeout=2.0)

    assert returncode == 1, output
    assert "+ Timeout +" in output, output
    assert "hangs_in_teardown" in output, output


def test_rearmed_bound_keeps_the_original_deadline(tmp_path: Path) -> None:
    # The child fails 2 s into a 3 s bound. Restoring the deadline ends it about 3 s after the call
    # started; a fresh 3 s timer from the failure would end it about 5 s after.
    returncode, output, exited = _run_child(
        tmp_path, "test_call_fails_late", timeout=3.0, sleep=2.0
    )

    assert returncode == 1, output
    assert "+ Timeout +" in output, output
    started = float((tmp_path / "started.txt").read_text(encoding="utf-8"))
    assert exited - started < 4.0, output


def test_clean_teardown_after_a_failure_is_not_cut_short(tmp_path: Path) -> None:
    returncode, output, _ = _run_child(tmp_path, "test_call_fails_cleanly", timeout=30.0)

    assert returncode == 1, output
    assert "+ Timeout +" not in output, output
    assert "1 failed" in output, output


def test_plugin_loads_when_pytest_timeout_is_disabled(tmp_path: Path) -> None:
    returncode, output, _ = _run_child(
        tmp_path, "test_call_fails_cleanly", 30.0, 0.0, "-p", "no:timeout"
    )

    assert returncode == 1, output
    assert "PluginValidationError" not in output, output
    assert "1 failed" in output, output


def test_rearm_after_the_deadline_passes_a_positive_timeout() -> None:
    # pytest-timeout's signal method arms with setitimer(ITIMER_REAL, timeout), and a zero there
    # disarms the alarm, so a spent deadline must still re-arm with a positive value.
    armed: list[float] = []

    def set_timer(*, item: object, settings: Settings) -> bool:
        armed.append(settings.timeout)
        return True

    node = SimpleNamespace(
        stash=pytest.Stash(), ihook=SimpleNamespace(pytest_timeout_set_timer=set_timer)
    )
    settings = Settings(
        timeout=300.0, method="signal", func_only=False, disable_debugger_detection=False
    )
    node.stash[failed_phase_timeout._DEADLINE] = (settings, time.monotonic() - 1.0)

    hook = failed_phase_timeout.pytest_exception_interact(node)  # type: ignore[arg-type]
    next(hook)
    with pytest.raises(StopIteration):
        next(hook)

    assert len(armed) == 1
    assert armed[0] > 0.0


def test_suite_registers_the_failed_phase_timeout_plugin(request: pytest.FixtureRequest) -> None:
    assert request.config.pluginmanager.is_registered(failed_phase_timeout)
