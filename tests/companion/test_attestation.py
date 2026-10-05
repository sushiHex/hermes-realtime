from __future__ import annotations

import json
import os
import sys
import sysconfig
import types
from pathlib import Path

import pytest

import hermes_realtime
from hermes_realtime.companion.attestation import attest_runtime

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_REPOSITORY = Path(hermes_realtime.__file__).resolve().parents[2]


def _hermes(
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    *,
    head: str = _COMMIT + "\n",
    version: object = "0.21.0",
) -> Path:
    """A Hermes checkout whose ``hermes_cli`` this process has imported."""

    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "HEAD").write_bytes(head.encode("ascii"))  # git writes LF
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__file__ = str(checkout / "hermes_cli" / "__init__.py")
    hermes_cli.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    venv = str(checkout / "venv")
    return Path(sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv}))


def _distribution(site: Path, direct_url: object) -> None:
    metadata = site / f"hermes_realtime-{hermes_realtime.__version__}.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: hermes-realtime\nVersion: {hermes_realtime.__version__}\n",
        encoding="utf-8",
    )
    (metadata / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")


def test_a_wheel_in_the_install_is_attested_with_this_process_and_its_hermes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hermes(monkeypatch, tmp_path / "hermes-agent")
    monkeypatch.setattr(hermes_realtime, "__file__", str(site / "hermes_realtime" / "__init__.py"))

    attestation = attest_runtime()

    assert attestation.model_dump() == {
        "pid": os.getpid(),
        "hermes_version": "0.21.0",
        "hermes_commit": _COMMIT,
        "realtime_version": hermes_realtime.__version__,
        "realtime_install": "wheel",
    }


def test_an_editable_install_in_the_install_names_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hermes(monkeypatch, tmp_path / "hermes-agent")
    _distribution(site, {"url": _REPOSITORY.as_uri(), "dir_info": {"editable": True}})
    monkeypatch.syspath_prepend(str(site))

    assert attest_runtime().realtime_install == "editable"


@pytest.mark.parametrize(
    "direct_url",
    [
        pytest.param(None, id="no-distribution-in-the-install"),
        pytest.param({"url": "file:///elsewhere", "dir_info": {"editable": True}}, id="other"),
        pytest.param({"url": "REPOSITORY", "dir_info": {}}, id="not-editable"),
        pytest.param({"url": "REPOSITORY", "dir_info": {"editable": "yes"}}, id="truthy"),
        pytest.param(["REPOSITORY"], id="malformed"),
    ],
)
def test_a_module_the_install_did_not_provide_is_elsewhere(
    direct_url: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hermes(monkeypatch, tmp_path / "hermes-agent")
    if direct_url is not None:
        text = json.dumps(direct_url).replace("REPOSITORY", _REPOSITORY.as_uri())
        _distribution(site, json.loads(text))
        monkeypatch.syspath_prepend(str(site))

    assert attest_runtime().realtime_install == "elsewhere"


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


def test_a_checkout_without_git_metadata_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hermes(monkeypatch, tmp_path / "hermes-agent")
    (tmp_path / "hermes-agent" / ".git" / "HEAD").unlink()

    assert attest_runtime().hermes_commit == "unknown"


@pytest.mark.parametrize("version", [21, "0.21.0 beta", ""])
def test_an_unattestable_hermes_version_is_unknown(
    version: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hermes(monkeypatch, tmp_path / "hermes-agent", version=version)

    assert attest_runtime().hermes_version == "unknown"


def test_a_process_without_hermes_attests_nothing_it_cannot_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "hermes_cli", None)

    attestation = attest_runtime()

    assert (attestation.hermes_version, attestation.hermes_commit) == ("unknown", "unknown")
    assert attestation.realtime_install == "elsewhere"
