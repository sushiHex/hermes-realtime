import json
import tomllib
from pathlib import Path

import hermes_realtime

_RELEASE_VERSION = "0.0.3"
_PYTHON_RANGE = ">=3.11,<3.12"
_UV_LOCK_PYTHON_RANGE = "==3.11.*"
_HATCHLING_REQUIREMENT = "hatchling==1.27.0"
_KNOWLEDGE_USER_AGENT = f"Hermes-Realtime-Knowledge/{_RELEASE_VERSION}"


def test_release_metadata_versions_are_coherent() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    uv_lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    package = json.loads((root / "web" / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads((root / "web" / "package-lock.json").read_text(encoding="utf-8"))

    locked_project = next(
        entry for entry in uv_lock["package"] if entry["name"] == project["project"]["name"]
    )

    assert project["build-system"]["requires"] == [_HATCHLING_REQUIREMENT]
    assert project["project"]["version"] == _RELEASE_VERSION
    assert project["project"]["requires-python"] == _PYTHON_RANGE
    assert uv_lock["requires-python"] == _UV_LOCK_PYTHON_RANGE
    assert hermes_realtime.__version__ == _RELEASE_VERSION
    assert locked_project["version"] == _RELEASE_VERSION
    assert package["version"] == _RELEASE_VERSION
    assert package_lock["version"] == _RELEASE_VERSION
    assert package_lock["packages"][""]["version"] == _RELEASE_VERSION

    production_version_literals: dict[str, tuple[str, ...]] = {}
    for path in sorted((root / "src" / "hermes_realtime").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        versions = tuple(version for version in ("0.0.2", "0.0.3") if version in source)
        if versions:
            production_version_literals[path.relative_to(root).as_posix()] = versions
    assert production_version_literals == {
        "src/hermes_realtime/__init__.py": (_RELEASE_VERSION,),
    }

    for relative_path in (
        "src/hermes_realtime/providers/current_facts.py",
        "src/hermes_realtime/providers/knowledge_backends.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert source.count("from hermes_realtime import __version__") == 1
        assert source.count('f"Hermes-Realtime-Knowledge/{__version__}"') == 1
        assert _KNOWLEDGE_USER_AGENT not in source

    release_workflow = (root / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )
    assert release_workflow.count(f"hermes-realtime=={_RELEASE_VERSION}") == 1
    assert "hermes-realtime==0.0.2" not in release_workflow
