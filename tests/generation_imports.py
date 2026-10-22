"""Minimal standard -I -S bootstrap for the opt-in pinned-source proof.

Python/interpreter and the prepared venv dependency site are trusted prerequisites;
this does not attest that environment. No site/.pth processing or custom finder.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
from pathlib import Path


class ImportBootstrapError(RuntimeError):
    """Content-free refusal of an unsupported proof import environment."""


def _standard_finders() -> None:
    expected = (
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    )
    if len(sys.meta_path) != len(expected) or any(
        actual is not required for actual, required in zip(sys.meta_path, expected, strict=True)
    ):
        raise ImportBootstrapError("source_import_finder_unsupported")


def _roots(export: Path) -> set[str]:
    roots = set()
    for entry in export.iterdir():
        if entry.is_dir() and entry.name.isidentifier():
            roots.add(entry.name)  # Namespace packages need no __init__.py.
        elif entry.is_file():
            for suffix in importlib.machinery.all_suffixes():
                if entry.name.endswith(suffix):
                    name = entry.name[: -len(suffix)]
                    if name.isidentifier():
                        roots.add(name)
    return roots


def _within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def configure_imports(export: Path, original: Path, reviewed_tests: Path) -> dict[str, int | bool]:
    """Add only exact exported source, reviewed proof helpers and prepared deps.

    On Python3.11 -S, prefix may be the base prefix. Derive the venv BEFORE
    resolving the executable: POSIX venv interpreters are commonly symlinks.
    """
    if not sys.flags.isolated or not sys.flags.no_site:
        raise ImportBootstrapError("source_import_startup_requires_isolated_no_site")
    _standard_finders()
    export, original, reviewed_tests = (
        path.resolve() for path in (export, original, reviewed_tests)
    )
    if not all(path.is_dir() for path in (export, original, reviewed_tests)) or (
        export == original or _within(reviewed_tests, original)
    ):
        raise ImportBootstrapError("source_import_environment_unsupported")
    executable = Path(sys.executable).absolute()
    directory = executable.parent
    if directory.name.casefold() not in {"scripts", "bin"}:
        raise ImportBootstrapError("source_import_environment_unsupported")
    venv = directory.parent.resolve()
    site = venv / (
        "Lib/site-packages"
        if directory.name.casefold() == "scripts"
        else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    if not (venv / "pyvenv.cfg").is_file() or not site.is_dir():
        raise ImportBootstrapError("source_import_environment_unsupported")
    roots = _roots(export)
    if any(name.split(".", 1)[0] in roots for name in sys.modules):
        raise ImportBootstrapError("source_import_preloaded_upstream")
    bases = {Path(sys.base_prefix).resolve(), Path(sys.base_exec_prefix).resolve()}
    standard = []
    for value in sys.path:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).resolve()
        if (
            any(_within(path, base) for base in bases)
            and not _within(path, original)
            and not {"site-packages", "dist-packages"}.intersection(path.parts)
            and str(path) not in standard
        ):
            standard.append(str(path))
    if not standard:
        raise ImportBootstrapError("source_import_environment_unsupported")
    sys.path[:] = [str(export), str(reviewed_tests), str(site), *standard]
    sys.path_importer_cache.clear()
    sys.dont_write_bytecode = True
    os.chdir(export)
    return {
        "source_import_roots": len(roots),
        "source_import_prepared_site_explicit": True,
        "source_import_isolated_no_site": True,
    }


def verify_import_origins(export: Path) -> dict[str, int]:
    """Require every loaded upstream root's file/package/spec origins in export."""
    _standard_finders()
    export = export.resolve()
    roots = _roots(export)
    checked = 0
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in roots:
            continue
        locations = []
        file = getattr(module, "__file__", None)
        if file is not None:
            locations.append(file)
        locations.extend(getattr(module, "__path__", ()) or ())
        spec = getattr(module, "__spec__", None)
        if spec is not None:
            if spec.origin is not None:
                locations.append(spec.origin)
            locations.extend(spec.submodule_search_locations or ())
        if not locations or any(
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or not _within(Path(value).resolve(), export)
            for value in locations
        ):
            raise ImportBootstrapError("source_import_origin_outside_export")
        checked += 1
    return {"source_import_modules_checked": checked}
