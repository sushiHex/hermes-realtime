import pytest

from hermes_realtime import host_launcher as host


def test_native_host_discovers_tools_without_foreground_model_or_lookup_overrides(monkeypatch):
    captured = []

    def capture_session(**kwargs):
        captured.append(kwargs["config"])
        raise RuntimeError("captured configuration")

    monkeypatch.setattr(host, "HermesApiTaskSession", capture_session)
    with pytest.raises(RuntimeError, match="captured configuration"):
        host.build_local_host_launcher(
            hermes_api_bearer="synthetic-host-bearer-32-characters",
            inference_provider="codex", hermes_model_source="gateway",
            hermes_discover_tools=True,
            allow_unsandboxed_tasks=True,
        )
    config, = captured
    assert config.discover_tools is True
    assert config.lookup_policy is None
    assert config.run_provider is config.run_model is None
