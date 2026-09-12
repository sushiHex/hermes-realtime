import base64
import csv
import hashlib
import io
import json
import platform
import tarfile
import zipfile
from pathlib import Path
from runpy import run_path

import pytest


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reference(path: str, raw: bytes) -> dict[str, object]:
    return {
        "role": "wheel",
        "relativePath": path,
        "basename": Path(path).name,
        "sha256": _sha(raw),
        "bytes": len(raw),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    helper = run_path(str(Path.cwd() / "tests/test_qualification_wheelhouse.py"))["wheel"]
    dependency_name, dependency = helper("synthetic_base")
    direct_name, direct = helper("hermes_realtime", "0.0.3", requires=("synthetic-base==1.0",))
    root = tmp_path / "input"
    wheel_path = "linux-runtime/wheels/" + dependency_name
    requirements = f"synthetic-base==1.0 --hash=sha256:{_sha(dependency)}\n".encode()
    constraints = b"# No additional constraints\n"
    paths = {
        "requirements": "linux-runtime/requirements.txt",
        "constraints": "linux-runtime/constraints.txt",
    }
    values = {
        wheel_path: dependency,
        paths["requirements"]: requirements,
        paths["constraints"]: constraints,
        "candidate/" + direct_name: direct,
    }
    for relative, raw in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = {
        "schemaVersion": 1,
        "purpose": "realtime_linux_runtime",
        "pythonVersion": platform.python_version(),
        "platform": "linux_x86_64",
        "requirements": {**_reference(paths["requirements"], requirements), "role": "requirements"},
        "constraints": {**_reference(paths["constraints"], constraints), "role": "constraints"},
        "wheels": [_reference(wheel_path, dependency)],
    }
    manifest_path = root / "linux-runtime/wheelhouse-manifest-v1.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    installation = tmp_path / "installation"
    site = installation / "site"
    site.mkdir(parents=True)
    installed: dict[str, bytes] = {}
    for raw in (dependency, direct):
        owned: set[str] = set()
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            info = next(
                name.split("/", 1)[0] for name in bundle.namelist() if ".dist-info/" in name
            )
            for name in bundle.namelist():
                if name.endswith(".dist-info/RECORD"):
                    continue
                installed[name] = bundle.read(name)
                owned.add(name)
            installed[info + "/INSTALLER"] = b"pip\n"
            installed[info + "/REQUESTED"] = b""
            owned.update({info + "/INSTALLER", info + "/REQUESTED", info + "/RECORD"})
            record = io.StringIO(newline="")
            writer = csv.writer(record, lineterminator="\n")
            for name in sorted(owned):
                if name.endswith("/RECORD"):
                    writer.writerow((name, "", ""))
                    continue
                payload = installed[name]
                digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
                writer.writerow((name, "sha256=" + digest.decode("ascii"), len(payload)))
            installed[info + "/RECORD"] = record.getvalue().encode()
    for relative, raw in installed.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root, installation, installed


def test_installed_inventory_requires_exact_wheel_bytes_and_closed_namespace(tmp_path):
    from scripts.qualification_linux_worker import installed_inventory

    root, installation, installed = _fixture(tmp_path)
    digest, count = installed_inventory(root, installation)
    assert len(digest) == 64 and count == len(installed)

    target = installation / "site/synthetic_base/__init__.py"
    target.write_bytes(b"drift")
    with pytest.raises(ValueError, match="source bytes differ"):
        installed_inventory(root, installation)


def test_installed_inventory_refuses_unowned_file(tmp_path):
    from scripts.qualification_linux_worker import installed_inventory

    root, installation, _ = _fixture(tmp_path)
    (installation / "site/foreign.py").write_text("# foreign\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unowned"):
        installed_inventory(root, installation)


def test_installed_inventory_refuses_forged_installer_metadata(tmp_path):
    from scripts.qualification_linux_worker import installed_inventory

    root, installation, _ = _fixture(tmp_path)
    target = next((installation / "site").glob("*.dist-info/INSTALLER"))
    target.write_bytes(b"foreign\n")
    with pytest.raises(ValueError, match="installer-generated"):
        installed_inventory(root, installation)


def test_wheel_owner_ignores_nested_vendored_dist_info():
    from scripts import qualification_linux_worker as worker

    helper = run_path(str(Path.cwd() / "tests/test_qualification_wheelhouse.py"))["wheel"]
    name, raw = helper("synthetic_base")
    source = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as original, zipfile.ZipFile(source, "w") as rewritten:
        for entry in original.infolist():
            rewritten.writestr(entry, original.read(entry))
        rewritten.writestr("synthetic_base/vendor.dist-info/METADATA", b"vendored")
    files, _, info = worker._wheel_files(name, source.getvalue())
    assert info == "synthetic_base-1.0.dist-info"
    assert "synthetic_base/vendor.dist-info/METADATA" in files


def test_installed_entry_script_is_derived_from_owned_entry_point():
    from scripts.qualification_linux_worker import _entry_script

    raw = _entry_script("hermes_realtime.cli:main")
    # Extracted after verifying pinned OCI layer
    # sha256:882a85d461488e9f738169100e18dabf5b52710cf9de01203645106010390643.
    assert raw == (
        b"#!/usr/local/bin/python\n"
        b"# -*- coding: utf-8 -*-\n"
        b"import re\n"
        b"import sys\n"
        b"from hermes_realtime.cli import main\n"
        b"if __name__ == '__main__':\n"
        b"    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
        b"    sys.exit(main())\n"
    )


def test_archived_wrapper_executes_only_the_owned_installed_cli(tmp_path):
    from scripts.qualification_linux_worker import _run_installed_entrypoint

    site = tmp_path / "site"
    script = site / "bin/hermes-realtime-host"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(7)\n", encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        _run_installed_entrypoint(site, script, ("--help",))
    assert stopped.value.code == 7
    with pytest.raises(ValueError, match="entry point"):
        _run_installed_entrypoint(site, script, ("--foreign",))


def test_observe_records_only_its_closed_failure_stage(tmp_path, monkeypatch):
    from scripts import qualification_linux_worker as worker

    installation = tmp_path / "installation"
    (installation / "site").mkdir(parents=True)
    output = tmp_path / "output/observation.json"
    output.parent.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker, "_source_binding", lambda root: ("1" * 64,) * 3)

    def fail_inventory(input_root, installed):
        raise RuntimeError("private module and path must not escape")

    monkeypatch.setattr(worker, "installed_inventory", fail_inventory)
    with pytest.raises(RuntimeError, match="private module"):
        worker.observe(tmp_path, installation, output, scratch)

    failure = output.with_name(worker._FAILURE)
    assert failure.read_bytes() == b'{"stage":"installed_inventory","version":1}\n'
    assert b"private module" not in failure.read_bytes()
    assert not output.exists()


def test_observe_propagates_dependency_import_stage_without_global_state(tmp_path, monkeypatch):
    from scripts import qualification_linux_worker as worker

    installation = tmp_path / "installation"
    (installation / "site").mkdir(parents=True)
    output = tmp_path / "output/observation.json"
    output.parent.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker, "_source_binding", lambda root: ("1" * 64,) * 3)
    monkeypatch.setattr(worker, "installed_inventory", lambda input_root, installed: ("2" * 64, 1))
    monkeypatch.setattr(worker, "_import_roots", lambda root: ("private_dependency",))

    def fail_import(name):
        raise ImportError("private loader detail")

    monkeypatch.setattr(worker.importlib, "import_module", fail_import)
    with pytest.raises(ImportError, match="private loader"):
        worker.observe(tmp_path, installation, output, scratch)

    failure = output.with_name(worker._FAILURE)
    assert failure.read_bytes() == b'{"stage":"dependency_imports","version":1}\n'


def test_worker_is_bound_to_its_candidate_archive_member(tmp_path):
    from scripts import qualification_linux_worker as worker

    root = tmp_path / "input"
    archive_path = root / "source/candidate-source.tar"
    archive_path.parent.mkdir(parents=True)
    stream = io.BytesIO()
    values = {
        "scripts/qualification_linux_worker.py": Path(worker.__file__).read_bytes(),
        "scripts/qualification_linux_producer.py": Path(worker.__file__)
        .with_name("qualification_linux_producer.py")
        .read_bytes(),
        ".github/workflows/release-gates.yml": b"name: release-gates\n",
        "uv.lock": b"version = 1\n",
    }
    with tarfile.open(fileobj=stream, mode="w:") as archive:
        for relative, raw in values.items():
            info = tarfile.TarInfo("hermes-realtime-0.0.3/" + relative)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    archive_path.write_bytes(stream.getvalue())
    source, workflow, lock = worker._source_binding(root)
    assert source == _sha(stream.getvalue())
    assert workflow == _sha(values[".github/workflows/release-gates.yml"])
    assert lock == _sha(values["uv.lock"])

    values["scripts/qualification_linux_worker.py"] = b"foreign"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as archive:
        for relative, raw in values.items():
            info = tarfile.TarInfo("hermes-realtime-0.0.3/" + relative)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    archive_path.write_bytes(stream.getvalue())
    with pytest.raises(ValueError, match="worker differs"):
        worker._source_binding(root)
