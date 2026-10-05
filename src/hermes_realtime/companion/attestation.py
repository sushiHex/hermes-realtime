"""What a process running the plugin loaded, for the companion's welcome to attest.

The plugin captures it once, when Hermes loads it, so a gateway that kept running across an
update or reinstall still attests what it actually imported. Anything that cannot be named
is ``unknown`` (or ``elsewhere``), which a qualification refuses.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import sysconfig
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from urllib.request import url2pathname

from pydantic import TypeAdapter, ValidationError

import hermes_realtime
from hermes_realtime.protocol import RuntimeAttestation
from hermes_realtime.protocol.events import AttestedVersion

_COMMIT = re.compile(rb"[0-9a-f]{40}\n?")
_MAX_HEAD_BYTES = 256
_VERSION: TypeAdapter[str] = TypeAdapter(AttestedVersion)


def attest_runtime() -> RuntimeAttestation:
    """This process, the Hermes it imported, and where its ``hermes_realtime`` came from."""

    try:
        hermes_cli = importlib.import_module("hermes_cli")
    except ImportError:
        hermes_cli = None
    origin = getattr(hermes_cli, "__file__", None)
    # hermes_cli sits at the root of the installer's checkout.
    checkout = Path(origin).resolve().parents[1] if type(origin) is str else None
    return RuntimeAttestation(
        pid=os.getpid(),
        hermes_version=_version(getattr(hermes_cli, "__version__", None)),
        hermes_commit=_detached_commit(checkout),
        realtime_version=_version(hermes_realtime.__version__),
        realtime_install=_realtime_install(checkout),
    )


def _version(value: object) -> str:
    try:
        return _VERSION.validate_python(value, strict=True)
    except ValidationError:
        return "unknown"


def _detached_commit(checkout: Path | None) -> str:
    """The commit a detached checkout's HEAD names, read without running git."""

    if checkout is None:
        return "unknown"
    try:
        with (checkout / ".git" / "HEAD").open("rb") as head:
            content = head.read(_MAX_HEAD_BYTES)
    except OSError:
        return "unknown"
    return content.decode("ascii").strip() if _COMMIT.fullmatch(content) else "unknown"


def _realtime_install(checkout: Path | None) -> Literal["wheel", "editable", "elsewhere"]:
    """Where the imported module's file is: the install's environment, its editable source,
    or anywhere else."""

    if checkout is None:
        return "elsewhere"
    module = Path(hermes_realtime.__file__).resolve()
    venv = str(checkout / "venv")
    site = Path(sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv})).resolve()
    if module == site / "hermes_realtime" / "__init__.py":
        return "wheel"
    try:
        distribution = importlib.metadata.distribution("hermes-realtime")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return "elsewhere"
    if Path(str(distribution.locate_file(""))).resolve() != site or type(direct_url) is not dict:
        return "elsewhere"
    url, dir_info = direct_url.get("url"), direct_url.get("dir_info")
    if type(url) is not str or type(dir_info) is not dict or dir_info.get("editable") is not True:
        return "elsewhere"
    parts = urlsplit(url)
    if parts.scheme != "file" or parts.netloc not in ("", "localhost"):
        return "elsewhere"
    source = Path(url2pathname(parts.path)).resolve()
    return "editable" if module.is_relative_to(source) else "elsewhere"
