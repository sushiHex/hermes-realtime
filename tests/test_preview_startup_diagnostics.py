"""Host failure evidence is observational and never changes readiness or cleanup."""
import io
import json
from types import SimpleNamespace

import pytest

from scripts import preview_cutover
from scripts.lookup_backend_logs import CappedCapture, CappedProcesses


def _host_fixture(tmp_path, monkeypatch, fault):
    ports = dict(api=1, companion=2, livekit=3, host=4)
    plan = SimpleNamespace(output=tmp_path, ports=ports, policy_mode='legacy',
        state_root=tmp_path.parent / (tmp_path.name + '-state'),
        host_options=dict(allow_unsandboxed_tasks=False, natural_work_tools=True,
                          task_start_policy='ordinary', conversation_profile='default'),
        models=dict(foreground='synthetic', foreground_effort='low'),
        executables=dict(codex='synthetic'),
        speech=dict(stt_provider='synthetic', stt_tier='medium', tts_provider='synthetic',
                    tts_voice='synthetic', tts_threads=1, tts_chunk_chars=128))
    calls = []
    error = RuntimeError('SYNTHETIC_PRIVATE_VALUE /private/path token=not-real')

    class Launcher:
        async def start(self):
            calls.append('start')
            if fault in {'host_start', 'start_and_close'}:
                raise error
        async def close(self):
            calls.append('close')
            if fault in {'host_close', 'start_and_close'}:
                raise error

    def build(**kwargs):
        calls.append('build')
        if fault == 'host_construction':
            raise error
        return Launcher()

    def active(_):
        if fault == 'configuration':
            raise error
        return plan, None, tmp_path

    async def sleep(_):
        raise error

    host = SimpleNamespace(_load_api_bearer=lambda _: 'synthetic',
        _load_voice_companion=lambda *_: None, LoopbackPeerAuthorizer=object,
        build_local_host_launcher=build)
    monkeypatch.setattr(preview_cutover, '_active_plan', active)
    monkeypatch.setattr(preview_cutover, '_runtime_modules', lambda _: (None, host))
    monkeypatch.setattr(preview_cutover, '_bound_kokoro_mcp_config', lambda _: None)
    monkeypatch.setattr(preview_cutover.asyncio, 'sleep', sleep)
    if fault != 'serving':
        (tmp_path / 'host.stop').touch()
    return calls, error


@pytest.mark.asyncio
@pytest.mark.parametrize('fault,stage,calls', [
    ('configuration', 'configuration', []),
    ('host_construction', 'host_construction', ['build']),
    ('host_start', 'host_start', ['build', 'start', 'close']),
    ('serving', 'serving', ['build', 'start', 'close']),
    ('host_close', 'host_close', ['build', 'start', 'close']),
    ('start_and_close', 'host_close', ['build', 'start', 'close']),
])
async def test_host_child_failure_preserves_exception_cleanup_and_fixed_stage(
    tmp_path, monkeypatch, capsys, fault, stage, calls,
):
    observed, error = _host_fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(RuntimeError) as caught:
        await preview_cutover._host_child(tmp_path / 'active')
    assert caught.value is error
    assert observed == calls
    output = capsys.readouterr().err
    assert output.startswith('[voice-preview-host-failure] ')
    assert json.loads(output.split(' ', 1)[1]) == {
        'version': 1, 'stage': stage, 'failure_kind': 'RuntimeError',
    }
    assert 'SYNTHETIC_PRIVATE_VALUE' not in output
    assert '/private/path' not in output


@pytest.mark.asyncio
async def test_successful_host_has_no_failure_marker(tmp_path, monkeypatch, capsys):
    calls, _ = _host_fixture(tmp_path, monkeypatch, None)
    assert await preview_cutover._host_child(tmp_path / 'active') == 0
    assert calls == ['build', 'start', 'close']
    assert capsys.readouterr().err == ''


def test_host_marker_print_failure_cannot_replace_original_exception(monkeypatch):
    class Broken(io.StringIO):
        def write(self, value):
            raise OSError('SYNTHETIC_PRIVATE_VALUE')
    monkeypatch.setattr(preview_cutover.sys, 'stderr', Broken())
    preview_cutover._emit_host_failure({'stage': 'host_start'}, RuntimeError('private'))
    assert preview_cutover.sys.stderr is None


@pytest.mark.parametrize('prefix,payload,expected', [
    ('voice-preview-host-failure', {'version': 1, 'stage': 'host_start',
                                   'failure_kind': 'RuntimeError'},
     'host_start_RuntimeError'),
    ('kokoro-mcp-refused', {'version': 1, 'category': 'client_binding'},
     'kokoro_mcp_client_binding'),
])
def test_startup_markers_survive_split_capture_and_raw_deletion(
    tmp_path, prefix, payload, expected,
):
    owner = SimpleNamespace(roots={'host': SimpleNamespace(poll=lambda: 1)})
    collector = CappedProcesses(owner, tmp_path)
    path = tmp_path / 'host.private.log'
    capture = CappedCapture(path, limit=1)
    collector.captures['host'] = capture
    raw = ('[' + prefix + '] ' + json.dumps(payload) + '\n').encode()
    capture.feed(raw[:13])
    capture.feed(raw[13:])
    facts = collector.finish()['host']
    path.unlink()
    assert facts['categories'][expected]['count'] == 1
    assert facts['exit_code'] == 1
    assert facts['overflow'] is True
    assert not path.exists()


@pytest.mark.parametrize('prefix,payload', [
    ('voice-preview-host-failure', {'version': 1, 'stage': 'SYNTHETIC_PRIVATE_VALUE',
                                   'failure_kind': 'RuntimeError'}),
    ('voice-preview-host-failure', {'version': 1, 'stage': 'host_start',
                                   'failure_kind': 'SYNTHETIC_PRIVATE_VALUE'}),
    ('voice-preview-host-failure', {'version': True, 'stage': 'host_start',
                                   'failure_kind': 'RuntimeError'}),
    ('kokoro-mcp-refused', {'version': 1, 'category': 'SYNTHETIC_PRIVATE_VALUE'}),
    ('kokoro-mcp-refused', {'version': 1, 'category': 'client_binding',
                          'private': 'SYNTHETIC_PRIVATE_VALUE'}),
])
def test_startup_marker_metadata_rejects_private_or_extra_fields(tmp_path, prefix, payload):
    capture = CappedCapture(tmp_path / 'raw')
    capture.feed(('[' + prefix + '] ' + json.dumps(payload) + '\n').encode())
    facts = capture.finish()
    assert set(facts['categories']) == {
        'host_failure_invalid' if prefix == 'voice-preview-host-failure'
        else 'kokoro_mcp_failure_invalid'
    }
    assert 'SYNTHETIC_PRIVATE_VALUE' not in json.dumps(facts)


def test_startup_marker_duplicate_fields_are_not_reinterpreted(tmp_path):
    capture = CappedCapture(tmp_path / 'raw')
    capture.feed(b'[kokoro-mcp-refused] {"version":1,"version":1,"category":"client_binding"}\n')
    assert set(capture.finish()['categories']) == {'kokoro_mcp_failure_invalid'}


@pytest.mark.parametrize('code', [None, True, '1', -(2**31)-1, 2**32])
def test_owned_exit_code_rejects_noninteger_or_out_of_range(tmp_path, code):
    owner = SimpleNamespace(roots={'host': SimpleNamespace(poll=lambda: code)})
    collector = CappedProcesses(owner, tmp_path)
    collector.captures['host'] = CappedCapture(tmp_path / 'raw')
    assert collector.finish()['host']['exit_code'] is None


@pytest.mark.parametrize('code', [-(2**31), 0, 1, 2**32-1])
def test_owned_exit_code_retains_bounded_scalar_only(tmp_path, code):
    owner = SimpleNamespace(roots={'host': SimpleNamespace(poll=lambda: code)})
    collector = CappedProcesses(owner, tmp_path)
    collector.captures['host'] = CappedCapture(tmp_path / 'raw')
    assert collector.finish()['host']['exit_code'] == code


def test_exit_observation_failure_is_null_and_does_not_fail_finish(tmp_path):
    def poll():
        raise OSError("SYNTHETIC_PRIVATE_VALUE")
    collector = CappedProcesses(
        SimpleNamespace(roots={"host": SimpleNamespace(poll=poll)}), tmp_path,
    )
    collector.captures["host"] = CappedCapture(tmp_path / "raw")
    assert collector.finish()["host"]["exit_code"] is None


@pytest.mark.asyncio
async def test_host_cancellation_is_reraised_after_existing_close(tmp_path, monkeypatch, capsys):
    import asyncio
    calls, _ = _host_fixture(tmp_path, monkeypatch, None)
    cancellation = asyncio.CancelledError()
    async def run(active, facts):
        facts["stage"] = "host_start"
        try:
            raise cancellation
        finally:
            calls.append("close")
    monkeypatch.setattr(preview_cutover, "_run_host_child", run)
    with pytest.raises(asyncio.CancelledError) as caught:
        await preview_cutover._host_child(tmp_path / "active")
    assert caught.value is cancellation and calls == ["close"]
    assert json.loads(capsys.readouterr().err.split(" ", 1)[1])["failure_kind"] == "other"


def test_native_model_free_child_retains_managed_cause_and_host_failure(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(preview_cutover.__file__).resolve().parents[1]
    body = ('async def fail(active,facts):\n'
            ' facts["stage"]="host_start"\n'
            ' raise KokoroMcpError("client_binding")')
    script = (
        "import sys;sys.path[:0]=" + repr([str(root), str(root / "src")]) + ";"
        "from scripts import preview_cutover as p;"
        "from hermes_realtime.providers.kokoro_mcp import KokoroMcpError;"
        "exec(" + repr(body) + ");"
        "p._run_host_child=fail;raise SystemExit(p.main(['host-child','--active','synthetic']))"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                            capture_output=True, timeout=10)
    assert result.returncode == 1
    assert b"Traceback" not in result.stdout + result.stderr
    collector = CappedProcesses(SimpleNamespace(roots={
        "host": SimpleNamespace(poll=lambda: result.returncode),
    }), tmp_path)
    collector.captures["host"] = CappedCapture(tmp_path / "host.private.log")
    collector.captures["host"].feed(result.stdout + result.stderr)
    facts = collector.finish()["host"]
    (tmp_path / "host.private.log").unlink()
    assert facts["exit_code"] == 1
    assert "kokoro_mcp_client_binding" in facts["categories"], facts["categories"]
    assert facts["categories"]["kokoro_mcp_client_binding"]["count"] == 1
    assert facts["categories"]["host_start_other"]["count"] == 1
