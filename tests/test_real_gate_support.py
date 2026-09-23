from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

import pytest

_SUPPORT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "real_gate_support.py"
_SUPPORT_SPEC = importlib.util.spec_from_file_location("real_gate_support", _SUPPORT_PATH)
assert _SUPPORT_SPEC is not None and _SUPPORT_SPEC.loader is not None
_SUPPORT = importlib.util.module_from_spec(_SUPPORT_SPEC)
sys.modules[_SUPPORT_SPEC.name] = _SUPPORT
_SUPPORT_SPEC.loader.exec_module(_SUPPORT)

available_port = _SUPPORT.available_port
installed_hermes_identity = _SUPPORT.installed_hermes_identity
load_api_key = _SUPPORT.load_api_key


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=checkout, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "hermes.py").write_text("VERSION = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "hermes.py")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "hermes")
    return tmp_path


def test_the_gate_records_the_exact_installed_hermes(
    checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (checkout / "venv-output.log").write_text("untracked install output\n", encoding="utf-8")

    identity = installed_hermes_identity("0.22.0", checkout)

    assert identity == {
        "version": "0.22.0",
        "commit": _git(checkout, "rev-parse", "HEAD"),
        "baseline": False,
    }
    assert capsys.readouterr().out == ""


@pytest.fixture
def case_colliding_checkout(checkout: Path) -> Path:
    # Upstream 29112bef tracks such paths; a case-insensitive filesystem holds only one.
    (checkout / "spare.txt").write_text("other spelling\n", encoding="utf-8")
    blob = _git(checkout, "hash-object", "-w", "spare.txt")
    (checkout / "spare.txt").unlink()
    _git(checkout, "update-index", "--add", "--cacheinfo", f"100644,{blob},HERMES.py")
    _git(checkout, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "case")
    _git(checkout, "checkout", "-q", "-f", "HEAD")
    return checkout


def test_the_gate_identifies_a_checkout_whose_paths_differ_only_in_case(
    case_colliding_checkout: Path,
) -> None:
    identity = installed_hermes_identity("0.21.0", case_colliding_checkout)

    assert identity["commit"] == _git(case_colliding_checkout, "rev-parse", "HEAD")


def test_the_gate_refuses_an_edit_to_a_path_that_differs_only_in_case(
    case_colliding_checkout: Path,
) -> None:
    (case_colliding_checkout / "hermes.py").write_text("VERSION = 3\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no commit describes"):
        installed_hermes_identity("0.21.0", case_colliding_checkout)


@pytest.mark.parametrize(
    ("version", "same_commit", "baseline"),
    [("0.21.0", True, True), ("0.21.1", True, False), ("0.21.0", False, False)],
    ids=["baseline", "other-version", "other-commit"],
)
def test_the_gate_names_only_the_exact_baseline(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    same_commit: bool,
    baseline: bool,
) -> None:
    commit = _git(checkout, "rev-parse", "HEAD") if same_commit else "0" * 40
    monkeypatch.setattr(_SUPPORT, "HERMES_BASELINE", {"version": "0.21.0", "commit": commit})

    assert installed_hermes_identity(version, checkout)["baseline"] is baseline


@pytest.mark.parametrize("refusal", ["modified", "unidentifiable"])
def test_the_gate_refuses_hermes_that_no_commit_describes(
    checkout: Path,
    tmp_path_factory: pytest.TempPathFactory,
    capsys: pytest.CaptureFixture[str],
    refusal: str,
) -> None:
    if refusal == "modified":
        (checkout / "hermes.py").write_text("VERSION = 2\n", encoding="utf-8")
    else:
        checkout = tmp_path_factory.mktemp("not-a-checkout")

    with pytest.raises(RuntimeError, match="no commit describes"):
        installed_hermes_identity("0.21.0", checkout)

    assert capsys.readouterr().out.splitlines() == [
        '[hermes-identity] {"refusal":"' + refusal + '","version":1}'
    ]


def test_load_api_key_accepts_one_explicit_strong_value(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OTHER=value\nAPI_SERVER_KEY='abcdefghijklmnopqrstuvwxyz123456'\n",
        encoding="utf-8",
    )

    assert load_api_key(env_file) == "abcdefghijklmnopqrstuvwxyz123456"


@pytest.mark.parametrize(
    "content",
    [
        "",
        "API_SERVER_KEY=too-short\n",
        "API_SERVER_KEY=" + ("a" * 32) + "\nAPI_SERVER_KEY=" + ("b" * 32) + "\n",
    ],
)
def test_load_api_key_rejects_missing_weak_or_duplicate_values(
    tmp_path: Path,
    content: str,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeError, match="one explicit strong"):
        load_api_key(env_file)


def test_available_port_returns_a_bindable_loopback_port() -> None:
    port = available_port()

    assert 1 <= port <= 65535
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
