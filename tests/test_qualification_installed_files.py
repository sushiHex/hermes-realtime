"""Installed RECORD rewrites cannot hide source changes or additional imports."""

import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from runpy import run_path

import pytest

_wheels = run_path(str(Path(__file__).with_name("test_qualification_wheelhouse.py")))


def installation(wheels):
    installed = {".lock": b""}
    for raw in wheels.values():
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            installed.update({name: archive.read(name) for name in archive.namelist()})
    for path in tuple(installed):
        if path.count("/") == 1 and path.endswith(".dist-info/RECORD"):
            prefix = path.rsplit("/", 1)[0]
            installed[prefix + "/INSTALLER"] = b"uv"
            installed[prefix + "/REQUESTED"] = b""
            installed[prefix + "/uv_cache.json"] = json.dumps(
                {
                    "timestamp": {"secs_since_epoch": 1, "nanos_since_epoch": 0},
                    "commit": None,
                    "tags": None,
                    "env": {},
                    "directories": {},
                }
            ).encode()
            names = [row[0] for row in csv.reader(io.StringIO(installed[path].decode()))]
            names += [prefix + "/" + name for name in ("INSTALLER", "REQUESTED", "uv_cache.json")]
            rewrite(installed, path, names)
    return installed


def rewrite(installed, record, names=None):
    if names is None:
        names = [row[0] for row in csv.reader(io.StringIO(installed[record].decode()))]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for name in names:
        raw = installed[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        writer.writerow(
            (
                name,
                "" if name == record else "sha256=" + digest,
                "" if name == record else str(len(raw)),
            )
        )
    installed[record] = output.getvalue().encode()


def inspect(wheels, installed):
    from scripts.qualification_installed_files import inspect_installed_wheels

    return inspect_installed_wheels(
        requirements=_wheels["pins"](wheels),
        constraints=b"",
        wheels=wheels,
        python_version="3.11.16",
        platform="windows_amd64",
        installed=installed,
    )


def test_installed_inventory_includes_source_files_and_installer_generated_records():
    wheels = dict([_wheels["wheel"]("synthetic_base")])
    installed = installation(wheels)
    result = inspect(wheels, installed)
    assert result.file_count == len(installed)
    assert result.distribution_count == 1
    assert len(result.inventory_sha256) == 64


@pytest.mark.parametrize("changed", [False, True])
def test_isolated_installed_hooks_must_match_original_wheel_bytes(changed):
    artifact = _wheels["_with_member"](
        _wheels["wheel"]("synthetic_base"), "retained.pth", b"import synthetic_hook\n"
    )
    wheels = dict([artifact])
    installed = installation(wheels)
    if changed:
        installed["retained.pth"] = b"import changed_hook\n"
        rewrite(installed, "synthetic_base-1.0.dist-info/RECORD")
        with pytest.raises(ValueError):
            inspect(wheels, installed)
    else:
        assert inspect(wheels, installed).file_count == len(installed)


def test_installed_owner_is_not_taken_from_nested_vendored_metadata():
    artifact = _wheels["_with_member"](
        _wheels["wheel"]("synthetic_base"),
        "aaa_vendor/peer-1.0.dist-info/METADATA",
        b"synthetic vendored metadata\n",
    )
    wheels = dict([artifact])
    installed = installation(wheels)
    assert inspect(wheels, installed).file_count == len(installed)


@pytest.mark.parametrize(
    "fault",
    [
        "changed_source",
        "missing_source",
        "unlisted",
        "recorded_extra",
        "hook",
        "installer",
        "requested",
        "cache_path",
        "record_digest",
        "record_duplicate",
        "cross_distribution",
        "lock",
    ],
)
def test_installed_file_changes_and_record_laundering_are_refused(fault):
    wheels = dict([_wheels["wheel"]("synthetic_base"), _wheels["wheel"]("synthetic_peer")])
    installed = installation(wheels)
    record = "synthetic_base-1.0.dist-info/RECORD"
    metadata = "synthetic_base-1.0.dist-info/"
    if fault == "changed_source":
        installed["synthetic_base/__init__.py"] = b"# altered source\n"
        rewrite(installed, record)
    elif fault == "missing_source":
        del installed["synthetic_base/__init__.py"]
    elif fault in {"unlisted", "recorded_extra", "hook"}:
        name = "injected.pth" if fault == "hook" else "injected.py"
        installed[name] = b"# additional import\n"
        if fault != "unlisted":
            rows = [row[0] for row in csv.reader(io.StringIO(installed[record].decode()))]
            rewrite(installed, record, [*rows, name])
    elif fault == "installer":
        installed[metadata + "INSTALLER"] = b"unbound"
        rewrite(installed, record)
    elif fault == "requested":
        installed[metadata + "REQUESTED"] = b"unexpected"
        rewrite(installed, record)
    elif fault == "cache_path":
        installed[metadata + "uv_cache.json"] = b'{"env":{"PYTHONPATH":"unbound"}}'
        rewrite(installed, record)
    elif fault == "record_digest":
        installed[record] = installed[record].replace(b"sha256=", b"sha256=x", 1)
    elif fault == "record_duplicate":
        installed[record] += installed[record].splitlines()[0] + b"\n"
    elif fault == "cross_distribution":
        rows = [row[0] for row in csv.reader(io.StringIO(installed[record].decode()))]
        rewrite(installed, record, [*rows, "synthetic_peer/__init__.py"])
    else:
        installed[".lock"] = b"unexpected"
    with pytest.raises(ValueError):
        inspect(wheels, installed)


@pytest.mark.parametrize("fault", [None, "missing", "unlisted", "file_directory"])
def test_generated_entry_points_follow_the_wheel_and_preserve_the_namespace(fault):
    info = "synthetic_base-1.0.dist-info"
    artifact = _wheels["_with_member"](
        _wheels["wheel"]("synthetic_base"),
        info + "/entry_points.txt",
        b"[console_scripts]\nsynthetic-cli = synthetic_base:main\n",
    )
    if fault == "file_directory":
        artifact = _wheels["_with_member"](artifact, "bin", b"synthetic file")
    wheels = dict([artifact])
    installed = installation(wheels)
    name = "bin/synthetic-cli.exe"
    if fault != "missing":
        installed[name] = b"synthetic generated image; no execution claim"
        record = info + "/RECORD"
        names = [row[0] for row in csv.reader(io.StringIO(installed[record].decode()))]
        rewrite(installed, record, [*names, name])
    if fault == "unlisted":
        installed["bin/unlisted.exe"] = b"unlisted generated image"
    if fault is None:
        assert inspect(wheels, installed).file_count == len(installed)
    else:
        with pytest.raises(ValueError):
            inspect(wheels, installed)


@pytest.mark.parametrize("scheme", ["purelib", "platlib"])
def test_relocated_library_members_still_match_their_original_wheel_bytes(scheme):
    source = f"synthetic_base-1.0.data/{scheme}/synthetic_extra.py"
    artifact = _wheels["_with_member"](
        _wheels["wheel"]("synthetic_base"),
        source,
        b"# synthetic relocated source\n",
    )
    wheels = dict([artifact])
    installed = installation(wheels)
    installed["synthetic_extra.py"] = installed.pop(source)
    record = "synthetic_base-1.0.dist-info/RECORD"
    names = [row[0] for row in csv.reader(io.StringIO(installed[record].decode()))]
    rewrite(installed, record, ["synthetic_extra.py" if name == source else name for name in names])
    assert inspect(wheels, installed).file_count == len(installed)
    installed["synthetic_extra.py"] = b"# changed after relocation\n"
    rewrite(installed, record)
    with pytest.raises(ValueError, match="source bytes"):
        inspect(wheels, installed)
