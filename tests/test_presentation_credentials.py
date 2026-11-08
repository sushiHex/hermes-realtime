"""Model-free authority and ownership controls for live credential helpers."""
import asyncio
import json
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from hermes_realtime.integration.presentation_transport import Presentation

BASE = {"provider": "custom", "base_url": "https://example.invalid/v1", "model": "test"}

def helper(tmp_path, body):
    path = tmp_path / "helper.py"
    path.write_text(body)
    return [sys.executable, "-I", "-B", str(path)]

def owner():
    row = {"deadline": time.monotonic() + 5, "cancelled": False}
    return SimpleNamespace(lock=nullcontext(), scope=lambda: ("synthetic", row)), row

@pytest.mark.parametrize("change", [
    {"api_key_env": "KEY", "key_cmd_argv": [sys.executable, "helper"]},
    {"key_cmd_argv": "helper"}, {"key_cmd_argv": []},
    {"key_cmd_argv": ["relative", "helper"]},
    {"key_cmd_argv": [sys.executable, "bad\narg"]},
    {"key_cmd_argv": [sys.executable, 1]},
])
def test_helper_configuration_refuses_ambiguous_or_malformed_authority(change):
    with pytest.raises(ValueError, match="presentation-config"):
        Presentation(BASE | change)

@pytest.mark.asyncio
async def test_helper_reads_new_access_only_payload_per_call(tmp_path):
    argv = helper(tmp_path, 'import json; print(json.dumps('
                  '{"access_token":"synthetic-token", "expires_in":1}))')
    presentation = Presentation(BASE | {"key_cmd_argv": argv})
    argv[-1] = "mutated"
    generation, row = owner()
    evidence = {}
    assert await presentation._credential(generation, row, evidence) == "synthetic-token"
    assert evidence["credential_helper_settled"] is True
    assert row["presentation_active"] is None
    assert await presentation._credential(generation, row, {}) == "synthetic-token"

@pytest.mark.parametrize("payload", [
    {"access_token":"secret", "expires_in":True},
    {"access_token":"secret", "expires_in":0},
    {"access_token":"secret", "expires_in":1, "refresh_token":"secret"},
    {"access_token":"secret\nline", "expires_in":1},
    {"access_token":1, "expires_in":1},
])
@pytest.mark.asyncio
async def test_helper_rejects_whole_invalid_payload_without_echo(tmp_path, payload):
    argv = helper(tmp_path, "print(" + repr(json.dumps(payload)) + ")")
    generation, row = owner()
    with pytest.raises(ValueError, match="^presentation-credential$") as error:
        await Presentation(BASE | {"key_cmd_argv": argv})._credential(generation, row, {})
    assert "secret" not in str(error.value)
    assert row["presentation_active"] is None

@pytest.mark.asyncio
async def test_deadline_stops_and_settles_real_owned_helper(tmp_path):
    argv = helper(tmp_path, "import time; time.sleep(30)")
    generation, row = owner()
    row["deadline"] = time.monotonic() + .08
    evidence = {}
    with pytest.raises(ValueError, match="^presentation-credential$"):
        await Presentation(BASE | {"key_cmd_argv": argv})._credential(generation, row, evidence)
    assert evidence["credential_helper_settled"] is True
    assert row["presentation_active"] is None

@pytest.mark.asyncio
async def test_cancel_during_spawn_still_settles_owned_child(tmp_path, monkeypatch):
    argv = helper(tmp_path, "import time; time.sleep(30)")
    original = asyncio.create_subprocess_exec
    entered, release = asyncio.Event(), asyncio.Event()
    children = []
    async def delayed(*args, **kwargs):
        child = await original(*args, **kwargs)
        children.append(child)
        entered.set()
        await release.wait()
        return child
    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
    generation, row = owner()
    evidence = {}
    task = asyncio.create_task(Presentation(BASE | {"key_cmd_argv": argv})._credential(
        generation, row, evidence))
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert children[0].returncode is not None
    assert evidence["credential_helper_settled"] is True
    assert row["presentation_active"] is None

@pytest.mark.asyncio
async def test_output_bound_refuses_and_reaps_helper(tmp_path):
    argv = helper(tmp_path, 'print(\'{"access_token":"synthetic", "expires_in":1}\' + " " * 40000)')
    generation, row = owner()
    evidence = {}
    with pytest.raises(ValueError, match="^presentation-credential$"):
        await Presentation(BASE | {"key_cmd_argv": argv})._credential(generation, row, evidence)
    assert evidence["credential_helper_settled"] is True


@pytest.mark.asyncio
async def test_public_cancel_and_repeat_cancel_cannot_abandon_cleanup(tmp_path, monkeypatch):
    from hermes_realtime.integration.presentation_transport import cancel_presentation
    argv = helper(tmp_path, "import time; time.sleep(30)")
    original = asyncio.create_subprocess_exec
    entered, release = asyncio.Event(), asyncio.Event()
    children = []
    async def delayed(*args, **kwargs):
        child = await original(*args, **kwargs)
        children.append(child)
        entered.set()
        await release.wait()
        return child
    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
    generation, row = owner()
    evidence = {}
    task = asyncio.create_task(Presentation(BASE | {"key_cmd_argv": argv})._credential(
        generation, row, evidence))
    await entered.wait()
    cancel_presentation(generation, row)
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert children[0].returncode is not None
    assert evidence["credential_helper_settled"] is True
    assert row["presentation_active"] is None


@pytest.mark.asyncio
async def test_expired_or_wrong_owner_never_spawns(tmp_path, monkeypatch):
    argv = helper(tmp_path, "raise AssertionError('must not run')")
    async def forbidden(*args, **kwargs):
        raise AssertionError("spawn before authority")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    generation, row = owner()
    presentation = Presentation(BASE | {"key_cmd_argv": argv})
    row["deadline"] = time.monotonic() - 1
    with pytest.raises(ValueError, match="^presentation-deadline$"):
        await presentation._credential(generation, row, {})
    row["cancelled"] = True
    with pytest.raises(ValueError, match="^presentation-ownership$"):
        await presentation._credential(generation, row, {})


@pytest.mark.asyncio
async def test_complete_uses_fresh_helper_key_on_codex_route(tmp_path, monkeypatch):
    import types
    argv = helper(tmp_path, 'print(\'{"access_token":"synthetic-token","expires_in":1}\')')
    presentation = Presentation(BASE | {"api_mode": "codex_responses", "key_cmd_argv": argv})
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=object))
    observed = []
    async def complete(*args):
        observed.append(args[4])
        return "synthetic-result"
    monkeypatch.setattr(presentation, "_complete_codex", complete)
    generation, row = owner()
    assert await presentation.complete(generation, "synthetic", row, "", {}) == "synthetic-result"
    assert observed == ["synthetic-token"]


@pytest.mark.parametrize("body", [
    'import sys; print(\'{"access_token":"SYNTHETIC_SECRET", "expires_in":1}\'); sys.exit(1)',
    'print(\'{"access_token":"SYNTHETIC_SECRET","expires_in":1,"expires_in":2}\')',
])
@pytest.mark.asyncio
async def test_nonzero_and_duplicate_payloads_are_fixed_refusals(tmp_path, body, capsys):
    generation, row = owner()
    evidence = {}
    with pytest.raises(ValueError, match="^presentation-credential$"):
        await Presentation(BASE | {"key_cmd_argv": helper(tmp_path, body)})._credential(
            generation, row, evidence)
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err == ('[presentation-credential-refused] '
                            '{"version":1,"category":"credential"}\n')
    assert evidence["credential_helper_settled"] is True


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_correlated_owner_unknown(monkeypatch):
    class Child:
        returncode = None
        stdout = None
        def terminate(self):
            raise OSError("SYNTHETIC_PRIVATE_DETAIL")
    async def spawn(*args, **kwargs):
        return Child()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    generation, row = owner()
    evidence = {}
    with pytest.raises(ValueError, match="^presentation-credential-cleanup$"):
        await Presentation(BASE | {"key_cmd_argv": [sys.executable, "synthetic"]})._credential(
            generation, row, evidence)
    assert evidence["credential_helper_settled"] is False
    assert row["presentation_active"]["closing"] is True
