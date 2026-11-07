from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_realtime import host_launcher as host


def _arguments(**changes):
    return {"tts_provider": "kokoro", "edge_voice": "synthetic", "kokoro_voice": "bf_isabella",
            "kokoro_speed": 1.0, "kokoro_intra_op_threads": 16, **changes}


def _factory(monkeypatch):
    calls = []
    factory = SimpleNamespace(close=lambda: calls.append("close"))
    module = SimpleNamespace(KokoroMcpWorkerFactory=SimpleNamespace(
        from_config=lambda path: (calls.append(path), factory)[1]))
    monkeypatch.setitem(sys.modules, "hermes_realtime.providers.kokoro_mcp", module)
    return factory, calls


def test_explicit_gpu_host_uses_mcp_factory_without_direct_worker(monkeypatch, tmp_path):
    factory, calls = _factory(monkeypatch)
    config = tmp_path / "operator.json"
    selected = []
    monkeypatch.setattr(host, "KokoroSynthesizer", lambda **kw: selected.append(kw) or kw)
    host._build_synthesizer(**_arguments(kokoro_mcp_config=config))
    assert calls == [config]
    assert selected[0]["worker_client_factory"] is factory
    assert selected[0]["worker_python"] is None


@pytest.mark.parametrize("changes", [
    {"kokoro_worker_python": Path(sys.executable)}, {"tts_provider": "edge"},
])
def test_mcp_host_refuses_conflicting_backend_before_factory(monkeypatch, tmp_path, changes):
    _, calls = _factory(monkeypatch)
    with pytest.raises(ValueError, match="MCP"):
        host._build_synthesizer(**_arguments(kokoro_mcp_config=tmp_path / "operator", **changes))
    assert calls == []


def test_host_construction_failure_closes_owned_factory(monkeypatch, tmp_path):
    _, calls = _factory(monkeypatch)
    def fail(**kwargs):
        raise ValueError("synthetic")
    monkeypatch.setattr(host, "KokoroSynthesizer", fail)
    config = tmp_path / "operator"
    with pytest.raises(ValueError, match="synthetic"):
        host._build_synthesizer(**_arguments(kokoro_mcp_config=config))
    assert calls == [config, "close"]


def test_host_parser_accepts_mcp_config_and_cpu_remains_explicit_rollback(tmp_path):
    parser = host._build_argument_parser()
    config = tmp_path / "operator.json"
    config.write_text("{}")
    assert parser.parse_args(["--kokoro-mcp-config", str(config)]).kokoro_mcp_config == config
    assert parser.parse_args([]).kokoro_mcp_config is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--kokoro-mcp-config", "relative.json"])
