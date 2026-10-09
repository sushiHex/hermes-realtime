"""Read a record file that a live writer may be replacing at that moment.

``write_run_record`` replaces the file with ``os.replace``. On Windows an open that lands
inside the replace is refused with a sharing violation (``PermissionError``), and the
refusal clears as soon as the replace returns. The product never reads a record while its
own writer runs, so only a test that observes a live writer meets the refusal.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from hermes_realtime.integration.run_record import read_run_record

# One replace takes milliseconds, and the tests poll every few milliseconds already.
DENIED_ATTEMPTS = 200
DENIED_PAUSE_SECONDS = 0.005
REFUSED_MARKER = "[live-record-refused] "


def read_live_record(path: Path, max_bytes: int = 1 << 20) -> bytes | None:
    """Read ``path``, waiting out a refusal caused by a replace in flight.

    Only ``PermissionError`` is retried, and only ``DENIED_ATTEMPTS`` times: a record that
    stays unreadable raises the last refusal instead of reading as absent, and leaves one
    ``[live-record-refused]`` marker with the count. A missing record is ``None``, exactly
    as ``read_run_record`` returns it.
    """
    refusals = 0
    try:
        while True:
            try:
                return read_run_record(path, max_bytes)
            except PermissionError:
                refusals += 1
                if refusals == DENIED_ATTEMPTS:
                    raise
                time.sleep(DENIED_PAUSE_SECONDS)
    finally:
        if refusals == DENIED_ATTEMPTS:
            evidence = json.dumps({"refusals": refusals, "kind": "PermissionError"})
            print(REFUSED_MARKER + evidence, flush=True)
