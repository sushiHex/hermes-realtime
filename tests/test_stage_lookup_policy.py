"""Packaging and inert load checks; no native Hermes or provider is started."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from aiohttp import web

_SPEC = importlib.util.spec_from_file_location(
    "stage_lookup_policy", Path(__file__).parents[1] / "scripts/stage_lookup_policy.py"
)
stage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage)
FIXTURE = Path(__file__).parent / "fixtures/plugin_generation_candidate"


def profile(tmp_path):
    owned = tmp_path / "owned-profile"
    owned.mkdir()
    return owned


def test_stage_copies_only_reviewed_policy_and_new_native_bootstrap(tmp_path):
    owned = profile(tmp_path)
    result = stage.stage_lookup_policy(owned, FIXTURE)
    package = owned / "plugins/ordinary-lookup-experimental"
    assert {p.relative_to(owned).as_posix() for p in owned.rglob("*") if p.is_file()} == {
        "plugins/ordinary-lookup-experimental/" + name
        for name in ("__init__.py", "entrypoint.py", "policy.py", "plugin.yaml")
    }
    for name in ("entrypoint.py", "policy.py"):
        assert (package / name).read_bytes() == (FIXTURE / name).read_bytes()
    assert (package / "__init__.py").read_bytes() == b"from .entrypoint import register\n"
    manifest = (package / "plugin.yaml").read_text(encoding="utf-8")
    assert "ordinary-lookup-experimental" in manifest
    assert "offline synthetic fixture" not in manifest.lower()
    assert "dedicated disposable profile" in manifest
    assert set(result) == {"plugin", "policy", "reviewed_revision", "files_sha256"}
    assert set(result["files_sha256"]) == {
        "__init__.py",
        "entrypoint.py",
        "policy.py",
        "plugin.yaml",
    }
    assert str(owned) not in json.dumps(result)


@pytest.mark.parametrize("name", ["entrypoint.py", "policy.py"])
def test_stage_refuses_changed_reviewed_bytes_before_writing(tmp_path, name):
    owned = profile(tmp_path)
    source = tmp_path / "reviewed"
    source.mkdir()
    for filename in ("entrypoint.py", "policy.py"):
        (source / filename).write_bytes((FIXTURE / filename).read_bytes())
    with (source / name).open("ab") as stream:
        stream.write(b"\nCHANGED = True\n")
    with pytest.raises(stage.StagingRefusal, match="reviewed-source-mismatch"):
        stage.stage_lookup_policy(owned, source)
    assert list(owned.iterdir()) == []


def test_stage_refuses_existing_profile_without_touching_it(tmp_path):
    owned = profile(tmp_path)
    sentinel = owned / "config.yaml"
    sentinel.write_bytes(b"existing-profile-must-not-change")
    with pytest.raises(stage.StagingRefusal, match="profile-not-fresh"):
        stage.stage_lookup_policy(owned, FIXTURE)
    assert {p.name for p in owned.iterdir()} == {"config.yaml"}
    assert sentinel.read_bytes() == b"existing-profile-must-not-change"


def test_stage_never_creates_a_caller_profile(tmp_path):
    owned = tmp_path / "missing"
    with pytest.raises(stage.StagingRefusal, match="profile-not-fresh"):
        stage.stage_lookup_policy(owned, FIXTURE)
    assert not owned.exists()


@pytest.mark.asyncio
async def test_staged_native_register_wires_only_real_policy_with_inert_dependencies(
    tmp_path,
    monkeypatch,
):
    owned = profile(tmp_path)
    stage.stage_lookup_policy(owned, FIXTURE)
    package = owned / "plugins/ordinary-lookup-experimental"
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("packaging-must-not-call-a-provider-or-tool")

    entries = {
        name: types.SimpleNamespace(
            handler=forbidden,
            toolset="web",
            schema={},
            check_fn=None,
            requires_env=[],
            is_async=asynchronous,
            description="",
            emoji="",
        )
        for name, asynchronous in (("web_search", False), ("web_extract", True))
    }
    for name in ("tools", "openai", "openai.types", "openai.types.chat"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    registry = types.SimpleNamespace(get_entry=entries.get)
    for name, attributes in {
        "tools.registry": {"registry": registry},
        "tools.approval": {"get_current_session_key": forbidden},
        "tools.web_tools": {},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["openai.types.chat"].ChatCompletion = types.SimpleNamespace(
        model_validate=lambda value: value
    )
    registrations, middleware, platforms = [], [], []

    class Context:
        def register_tool(self, name, toolset, schema, handler, **options):
            assert options["override"] is True
            registrations.append(name)
            entries[name] = types.SimpleNamespace(handler=handler, is_async=options["is_async"])
            return types.SimpleNamespace(active=True)

        def register_middleware(self, name, handler):
            middleware.append(name)
            return types.SimpleNamespace(active=True)

        def register_platform_handler(self, name, handler):
            platforms.append((name, handler))

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    name = "isolated_staged_policy"
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    try:
        spec.loader.exec_module(module)
        assert module.register is sys.modules[name + ".entrypoint"].register
        assert "completion" not in vars(module)
        module.register(Context())
        assert registrations == ["web_search", "web_extract"]
        assert middleware == ["llm_execution", "tool_execution"]
        assert [platform for platform, _ in platforms] == ["api_server"]
        app = web.Application()
        platforms[0][1](app, None)
        assert {route.resource.canonical for route in app.router.routes()} == {
            "/lookup-policy/ready"
        }
        ready = next(route for route in app.router.routes() if route.method == "GET")
        response = await ready.handler(None)
        body = json.loads(response.body)
        assert body["ready"] is True and body["policy"] == "ordinary-lookup/v1"
        assert len(body["generation"]) == 32
        assert called == []
    finally:
        for imported in list(sys.modules):
            if imported.startswith(name + "."):
                del sys.modules[imported]


@pytest.mark.parametrize("corruption", ["bytes", "extra-file"])
def test_stage_refuses_corrupted_output_and_leaves_owned_partial_profile(
    tmp_path,
    monkeypatch,
    corruption,
):
    import contextlib

    owned = profile(tmp_path)
    original = Path.open

    @contextlib.contextmanager
    def changed(path, mode="r", *args, **kwargs):
        with original(path, mode, *args, **kwargs) as stream:
            yield stream
        if mode == "xb" and path.name == "policy.py":
            target = path if corruption == "bytes" else path.with_name("unexpected.py")
            with original(target, "ab") as stream:
                stream.write(b"\nUNREVIEWED = True\n")

    monkeypatch.setattr(Path, "open", changed)
    with pytest.raises(stage.StagingRefusal, match="artifact-integrity-refused"):
        stage.stage_lookup_policy(owned, FIXTURE)
    assert (owned / "plugins/ordinary-lookup-experimental").is_dir()


def test_stage_refusal_reporting_preserves_primary_refusal(tmp_path, monkeypatch):
    owned = profile(tmp_path)
    (owned / "existing").write_bytes(b"preserved")

    def unavailable(*args, **kwargs):
        raise OSError("synthetic unavailable output")

    monkeypatch.setattr("builtins.print", unavailable)
    with pytest.raises(stage.StagingRefusal, match="profile-not-fresh"):
        stage.stage_lookup_policy(owned, FIXTURE)


def test_load_only_receipt_requires_exact_binding_and_pending_parent_acceptance():
    spec = importlib.util.spec_from_file_location(
        "deployment_driver", Path(__file__).with_name("offline_lookup_deployment_proof.py")
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    expected = {"reviewed.py": "exact"}
    good = {
        "ok": False,
        "semantic_ok": True,
        "nonce": "current",
        "consent": True,
        "bindings": expected,
    }
    driver.validate_worker(good, expected, "current", True)
    for change in (
        {"nonce": "old"},
        {"ok": True},
        {"semantic_ok": False},
        {"consent": False},
        {"bindings": expected | {"unreviewed.py": "extra"}},
    ):
        with pytest.raises(RuntimeError, match="worker-evidence-refused"):
            driver.validate_worker(good | change, expected, "current", True)


def test_stage_source_ingress_and_readback_are_bounded(tmp_path, monkeypatch):
    import contextlib

    owned = profile(tmp_path)
    original = Path.open
    lengths = stage.REVIEWED_SIZES | {
        "__init__.py": len(stage.BOOTSTRAP),
        "plugin.yaml": len(stage.MANIFEST),
    }

    @contextlib.contextmanager
    def observed(path, mode="r", *args, **kwargs):
        with original(path, mode, *args, **kwargs) as stream:
            if mode != "rb":
                yield stream
                return

            def read(size=-1):
                assert size == lengths[path.name] + 1, "unbounded-policy-artifact-read"
                return stream.read(size)

            yield types.SimpleNamespace(read=read)

    monkeypatch.setattr(Path, "open", observed)
    stage.stage_lookup_policy(owned, FIXTURE)
