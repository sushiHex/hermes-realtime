"""What a process running the plugin loaded, for the companion's acceptance to attest.

The plugin captures it once, when Hermes loads it, so a gateway that kept running across an
update or reinstall still attests what it actually imported. Anything that cannot be named
is ``unknown`` (or ``elsewhere``), which a qualification refuses.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sysconfig
from pathlib import Path

import hermes_realtime
from hermes_realtime.protocol import RuntimeAttestation

_MAX_HEAD_BYTES = 256
_MAX_RECORD_BYTES = 1024 * 1024


def attest_runtime() -> RuntimeAttestation:
    """This process, the Hermes it imported, and the hermes-realtime wheel it imported.

    It never raises: a process whose install cannot be read attests nothing but itself.
    """

    try:
        return _attest()
    except Exception:
        return RuntimeAttestation(
            pid=os.getpid(),
            hermes_version="unknown",
            hermes_commit="unknown",
            realtime_version="unknown",
            realtime_install="elsewhere",
            realtime_record="unknown",
        )


def _attest() -> RuntimeAttestation:
    hermes_cli = importlib.import_module("hermes_cli")
    origin = hermes_cli.__file__
    if origin is None:
        raise ImportError("hermes_cli has no file")
    # hermes_cli sits at the root of the installer's checkout.
    checkout = Path(origin).resolve().parents[1]
    with (checkout / ".git" / "HEAD").open("rb") as head_file:
        head = head_file.read(_MAX_HEAD_BYTES)
    venv = str(checkout / "venv")
    site = Path(sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv})).resolve()
    # Only the installed package's own file is the wheel; anything else is not what it built.
    wheel = Path(hermes_realtime.__file__).resolve() == site / "hermes_realtime" / "__init__.py"
    return RuntimeAttestation(
        pid=os.getpid(),
        hermes_version=hermes_cli.__version__,
        # The strict model admits only a detached commit: a branch HEAD names nothing.
        hermes_commit=head.decode("ascii").strip(),
        realtime_version=hermes_realtime.__version__,
        realtime_install="wheel" if wheel else "elsewhere",
        realtime_record=_record(site) if wheel else "unknown",
    )


def _record(site: Path) -> str:
    """The SHA-256 of the one installed wheel's ``RECORD``, which hashes every file it put down."""

    records = list(site.glob("hermes_realtime-*.dist-info/RECORD"))
    if len(records) != 1:
        return "unknown"
    with records[0].open("rb") as record_file:
        content = record_file.read(_MAX_RECORD_BYTES + 1)
    return hashlib.sha256(content).hexdigest() if len(content) <= _MAX_RECORD_BYTES else "unknown"
