from __future__ import annotations

import importlib.util
import socket
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
load_api_key = _SUPPORT.load_api_key


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
