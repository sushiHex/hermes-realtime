"""Standard isolated bootstrap controls; only synthetic modules are imported."""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generation_imports", Path(__file__).with_name("generation_imports.py")
)
bootstrap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap)
STANDARD = (
    importlib.machinery.BuiltinImporter,
    importlib.machinery.FrozenImporter,
    importlib.machinery.PathFinder,
)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    export = tmp_path / "export"
    original = tmp_path / "original"
    reviewed = tmp_path / "reviewed-tests"
    base = tmp_path / "base"
    venv = original / ".venv"
    executable = venv / "Scripts" / "python.exe"
    site = venv / "Lib" / "site-packages"
    for directory in (export, original, reviewed, base / "Lib", site, executable.parent):
        directory.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"synthetic executable path only")
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    package = export / "upstream"
    package.mkdir()
    (package / "__init__.py").write_text("PINNED = True\n")
    (original / "upstream.py").write_text("PINNED = False\n")
    flags = {name: getattr(sys.flags, name) for name in sys.flags.__match_args__}
    flags.update(isolated=1, no_site=1)
    monkeypatch.setattr(sys, "flags", SimpleNamespace(**flags))
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setattr(sys, "base_exec_prefix", str(base))
    monkeypatch.setattr(sys, "prefix", str(base))
    monkeypatch.setattr(sys, "meta_path", list(STANDARD))
    monkeypatch.setattr(sys, "path", ["", str(original), str(base / "Lib")])
    monkeypatch.setattr(sys, "path_importer_cache", {})
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.chdir(tmp_path)
    yield SimpleNamespace(
        export=export,
        original=original,
        reviewed=reviewed,
        base=base,
        venv=venv,
        executable=executable,
        site=site,
    )
    sys.modules.pop("upstream", None)
    sys.modules.pop("prepared_dependency", None)


def configure(layout):
    return bootstrap.configure_imports(layout.export, layout.original, layout.reviewed)


def test_healthy_bootstrap_uses_export_and_explicit_prepared_site_without_pth(layout):
    sentinel = layout.original / "pth-executed"
    (layout.site / "overlay.pth").write_text(
        f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('bad')\n"
    )
    (layout.site / "prepared_dependency.py").write_text("TRUSTED_PREPARATION = True\n")
    result = configure(layout)
    assert sys.path == [
        str(layout.export),
        str(layout.site),
        str(layout.base / "Lib"),
    ]
    assert Path.cwd() == layout.export
    assert sys.dont_write_bytecode is True
    assert not sentinel.exists()
    import prepared_dependency
    import upstream

    assert upstream.PINNED is True and prepared_dependency.TRUSTED_PREPARATION is True
    assert bootstrap.verify_import_origins(layout.export)["source_import_modules_checked"] == 1
    assert result["source_import_roots"] == 1
    assert result["source_import_prepared_site_explicit"] is True
    assert str(layout.original) not in str(result)



def test_mutable_reviewed_directory_cannot_supply_imports(layout):
    (layout.reviewed / ".gitignore").write_text("*.py\n")
    for name, trusted in (
        ("prepared_dependency", layout.site),
        ("stdlib_probe", layout.base / "Lib"),
    ):
        (trusted / (name + ".py")).write_text("TRUSTED_SOURCE = True\n")
        (layout.reviewed / (name + ".py")).write_text("TRUSTED_SOURCE = False\n")
    configure(layout)
    try:
        import prepared_dependency
        import stdlib_probe

        assert prepared_dependency.TRUSTED_SOURCE is True
        assert stdlib_probe.TRUSTED_SOURCE is True
        assert Path(prepared_dependency.__file__).parent == layout.site
        assert Path(stdlib_probe.__file__).parent == layout.base / "Lib"
    finally:
        sys.modules.pop("prepared_dependency", None)
        sys.modules.pop("stdlib_probe", None)


@pytest.mark.parametrize("flag", ["isolated", "no_site"])
def test_nonisolated_or_site_enabled_startup_refuses(layout, monkeypatch, flag):
    monkeypatch.setattr(sys.flags, flag, 0)
    with pytest.raises(bootstrap.ImportBootstrapError, match="startup"):
        configure(layout)


def test_custom_or_editable_meta_finder_refuses_by_identity(layout, monkeypatch):
    class Impostor:
        def __eq__(self, other):
            return True

    monkeypatch.setattr(sys, "meta_path", [*STANDARD[:2], Impostor()])
    with pytest.raises(bootstrap.ImportBootstrapError, match="finder"):
        configure(layout)


def test_preloaded_upstream_module_refuses_even_before_path_changes(layout, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "upstream", SimpleNamespace(__file__=str(layout.original / "upstream.py"))
    )
    before = list(sys.path)
    with pytest.raises(bootstrap.ImportBootstrapError, match="preloaded"):
        configure(layout)
    assert sys.path == before


@pytest.mark.parametrize("missing", ["config", "site"])
def test_missing_prepared_venv_layout_refuses(layout, missing):
    if missing == "config":
        (layout.venv / "pyvenv.cfg").unlink()
    else:
        layout.site.rmdir()
    with pytest.raises(bootstrap.ImportBootstrapError, match="environment"):
        configure(layout)


def test_executable_location_not_a_venv_refuses(layout, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(layout.base / "python.exe"))
    with pytest.raises(bootstrap.ImportBootstrapError, match="environment"):
        configure(layout)


@pytest.mark.parametrize("field", ["file", "package_path", "spec_origin", "spec_path"])
def test_loaded_source_origin_outside_export_refuses(layout, monkeypatch, field):
    configure(layout)
    inside = str(layout.export / "upstream" / "__init__.py")
    outside = str(layout.original / "upstream.py")
    module = SimpleNamespace(
        __file__=inside,
        __path__=[str(layout.export / "upstream")],
        __spec__=SimpleNamespace(origin=inside, submodule_search_locations=None),
    )
    if field == "file":
        module.__file__ = outside
    elif field == "package_path":
        module.__path__ = [str(layout.original)]
    elif field == "spec_origin":
        module.__spec__.origin = outside
    else:
        module.__spec__.submodule_search_locations = [str(layout.original)]
    monkeypatch.setitem(sys.modules, "upstream", module)
    with pytest.raises(bootstrap.ImportBootstrapError, match="origin"):
        bootstrap.verify_import_origins(layout.export)


def test_venv_is_derived_before_executable_symlink_resolution(layout, monkeypatch):
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == layout.executable:
            return layout.base / "python.exe"
        return original_resolve(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "resolve", resolve)
        result = configure(layout)
    assert result["source_import_prepared_site_explicit"] is True
    assert str(layout.site) in sys.path


def test_namespace_root_origins_are_checked(layout, monkeypatch):
    namespace = layout.export / "namespace_root"
    namespace.mkdir()
    configure(layout)
    monkeypatch.setitem(
        sys.modules,
        "namespace_root",
        SimpleNamespace(
            __path__=[str(namespace)],
            __spec__=SimpleNamespace(origin=None, submodule_search_locations=[str(namespace)]),
        ),
    )
    assert bootstrap.verify_import_origins(layout.export)["source_import_modules_checked"] == 1
    sys.modules.pop("namespace_root")


def test_late_custom_finder_refuses_origin_verification(layout, monkeypatch):
    configure(layout)
    monkeypatch.setattr(sys, "meta_path", [*STANDARD, object()])
    with pytest.raises(bootstrap.ImportBootstrapError, match="finder"):
        bootstrap.verify_import_origins(layout.export)


def test_no_supported_stdlib_paths_refuses(layout, monkeypatch):
    monkeypatch.setattr(sys, "path", ["", str(layout.original), str(layout.site)])
    with pytest.raises(bootstrap.ImportBootstrapError, match="environment"):
        configure(layout)


def test_original_checkout_cannot_be_reviewed_helper_path(layout):
    with pytest.raises(bootstrap.ImportBootstrapError, match="environment"):
        bootstrap.configure_imports(layout.export, layout.original, layout.original)


def test_loaded_root_without_any_origin_refuses(layout, monkeypatch):
    configure(layout)
    monkeypatch.setitem(sys.modules, "upstream", SimpleNamespace())
    with pytest.raises(bootstrap.ImportBootstrapError, match="origin"):
        bootstrap.verify_import_origins(layout.export)


def test_top_level_module_origin_is_checked(layout, monkeypatch):
    (layout.export / "upstream_module.py").write_text("PINNED = True\n")
    configure(layout)
    monkeypatch.setitem(
        sys.modules,
        "upstream_module",
        SimpleNamespace(__file__=str(layout.original / "upstream_module.py")),
    )
    with pytest.raises(bootstrap.ImportBootstrapError, match="origin"):
        bootstrap.verify_import_origins(layout.export)


def test_stale_path_importer_cache_is_cleared(layout):
    sys.path_importer_cache[str(layout.export)] = object()
    configure(layout)
    assert str(layout.export) not in sys.path_importer_cache
