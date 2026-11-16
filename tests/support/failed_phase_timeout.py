"""Keep pytest-timeout's per-test bound armed after a failed phase.

pytest calls ``pytest_exception_interact`` for every failed phase, not only under ``--pdb``.
pytest-timeout implements that hook as "stop the timeout when pytest enters pdb" and cancels the
item's timer unconditionally, so whatever runs after a failure (the teardown of every fixture,
including an event loop waiting on a task that ignores cancellation) has no bound at all.

This plugin wraps that hook. Once every implementation has run, including pdb's post-mortem when
``--pdb`` is in use, it re-arms the same timer through pytest-timeout's public
``pytest_timeout_set_timer`` hook with the time left until the item's original deadline. The test
keeps the single wall-clock bound that ``pyproject.toml`` configures, and a hang after a failure
ends with pytest-timeout's own stack dump and exit instead of the CI job limit. After pdb has been
entered, pytest-timeout suppresses its timers itself, so an interactive session is not cut short.
"""

from __future__ import annotations

from collections.abc import Generator
from time import monotonic as _real_monotonic
from typing import Any

import pytest

# The clock is bound at import: the re-arm runs before fixture teardown, so a test's monkeypatched
# time.monotonic (frozen, synthetic or exhausted) is still in place when it does.
_DEADLINE = pytest.StashKey[tuple[Any, float]]()
# A spent deadline still re-arms with a positive timeout: the signal method arms with
# setitimer(ITIMER_REAL, timeout), where zero disarms the alarm instead of firing it.
_MIN_REMAINING_SECONDS = 0.001


@pytest.hookimpl(wrapper=True, optionalhook=True)
def pytest_timeout_set_timer(item: pytest.Item, settings: Any) -> Generator[None, object, object]:
    # With func_only, pytest-timeout bounds only the call phase and cancels the timer when the call
    # returns, so there is nothing to restore after a failure.
    if settings.func_only is False:
        item.stash[_DEADLINE] = (settings, _real_monotonic() + settings.timeout)
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_exception_interact(node: pytest.Item | pytest.Collector) -> Generator[None, None, None]:
    yield
    armed = node.stash.get(_DEADLINE, None)
    if armed is None:
        return
    settings, deadline = armed
    remaining = max(deadline - _real_monotonic(), _MIN_REMAINING_SECONDS)
    node.ihook.pytest_timeout_set_timer(item=node, settings=settings._replace(timeout=remaining))
