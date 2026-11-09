"""Keep automatic garbage collection out of a test's tight timed window."""

import gc
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def no_automatic_gc() -> Iterator[None]:
    """Collect now, then disable automatic collection until the block exits.

    A stop-the-world collection on a large full-suite heap can outlast a sub-second request
    budget, and a fake whose sends never suspend then fails its after-deadline check.
    """

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
