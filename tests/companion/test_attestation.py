from __future__ import annotations

import hashlib
import os
import sys
import sysconfig
import types
from pathlib import Path

import pytest

import hermes_realtime
from hermes_realtime.companion.attestation import attest_runtime

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_RECORD = b"hermes_realtime/__init__.py,sha256=abc,10\n"
_NOTHING_NAMED = {
    "pid": os.getpid(),
    "hermes_version": "unknown",
    "hermes_commit": "unknown",
    "realtime_version": "unknown",
    "realtime_install": "elsewhere",
    "realtime_record": "unknown",
}


def _hermes(
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    *,
    head: str = _COMMIT + "\n",
    version: object = "0.21.0",
) -> Path:
    """A Hermes checkout whose ``hermes_cli`` this process imported; returns its site dir."""

    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "HEAD").write_bytes(head.encode("ascii"))  # git writes LF
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__file__ = str(checkout / "hermes_cli" / "__init__.py")
    hermes_cli.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    venv = str(checkout / "venv")
    return Path(sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv}))


def _wheel(monkeypatch: pytest.MonkeyPatch, site: Path, record: bytes | None = _RECORD) -> None:
    """hermes-realtime installed in ``site``, and imported from there."""

    metadata = site / f"hermes_realtime-{hermes_realtime.__version__}.dist-info"
    metadata.mkdir(parents=True)
    if record is not None:
        (metadata / "RECORD").write_bytes(record)
    monkeypatch.setattr(hermes_realtime, "__file__", str(site / "hermes_realtime" / "__init__.py"))


def test_a_wheel_in_the_install_is_attested_with_its_record_and_this_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hermes(monkeypatch, tmp_path / "hermes-agent")
    _wheel(monkeypatch, site)

    assert attest_runtime().model_dump() == {
        "pid": os.getpid(),
        "hermes_version": "0.21.0",
        "hermes_commit": _COMMIT,
        "realtime_version": hermes_realtime.__version__,
        "realtime_install": "wheel",
        "realtime_record": hashlib.sha256(_RECORD).hexdigest(),
    }


def test_a_module_imported_from_anywhere_else_is_elsewhere_and_its_record_unnamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hermes(monkeypatch, tmp_path / "hermes-agent")
    _wheel(monkeypatch, site)
    # Inside the install's environment, but not the installed package's own file.
    monkeypatch.setattr(hermes_realtime, "__file__", str(site / "vendor" / "hermes_realtime.py"))

    attestation = attest_runtime()

    assert (attestation.realtime_install, attestation.realtime_record) == ("elsewhere", "unknown")


@pytest.mark.parametrize("layout", ["missing", "two-installs", "oversized"])
def test_a_record_that_does_not_name_one_install_is_unknown(
    layout: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hermes(monkeypatch, tmp_path / "hermes-agent")
    _wheel(
        monkeypatch,
        site,
        record=None if layout == "missing" else b"x" * (2 * 1024 * 1024) if layout == "oversized"
        else _RECORD,
    )
    if layout == "two-installs":
        other = site / "hermes_realtime-0.0.1.dist-info"
        other.mkdir()
        (other / "RECORD").write_bytes(_RECORD)

    attestation = attest_runtime()

    assert (attestation.realtime_install, attestation.realtime_record) == ("wheel", "unknown")


@pytest.mark.parametrize(
    "head",
    [
        pytest.param("ref: refs/heads/main\n", id="branch"),
        pytest.param(_COMMIT.upper() + "\n", id="uppercase"),
        pytest.param(_COMMIT[:39] + "\n", id="short"),
        pytest.param("", id="empty"),
    ],
)
def test_a_head_that_is_not_a_detached_commit_is_unknown(
    head: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hermes(monkeypatch, tmp_path / "hermes-agent", head=head)

    assert attest_runtime().hermes_commit == "unknown"


def test_a_broken_install_never_raises_and_names_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hermes(monkeypatch, tmp_path / "hermes-agent")
    _wheel(monkeypatch, site, record=None)
    (site / f"hermes_realtime-{hermes_realtime.__version__}.dist-info" / "RECORD").mkdir()

    assert attest_runtime().model_dump() == _NOTHING_NAMED


def test_a_checkout_without_git_metadata_names_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hermes(monkeypatch, tmp_path / "hermes-agent")
    (tmp_path / "hermes-agent" / ".git" / "HEAD").unlink()

    assert attest_runtime().model_dump() == _NOTHING_NAMED


def test_a_hermes_at_a_filesystem_root_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__file__ = str(Path(tmp_path.anchor) / "__init__.py")
    hermes_cli.__version__ = "0.21.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)

    assert attest_runtime().model_dump() == _NOTHING_NAMED


@pytest.mark.parametrize("version", [21, "0.21.0 beta", ""])
def test_an_unattestable_hermes_version_names_nothing(
    version: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hermes(monkeypatch, tmp_path / "hermes-agent", version=version)

    assert attest_runtime().model_dump() == _NOTHING_NAMED


def test_a_process_without_hermes_names_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hermes_cli", None)

    assert attest_runtime().model_dump() == _NOTHING_NAMED
