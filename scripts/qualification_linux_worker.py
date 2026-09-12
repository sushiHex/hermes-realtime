"""Source-bound stdlib worker for the installed Linux null-capture proof."""

from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import importlib
import io
import json
import os
import pkgutil
import platform
import re
import runpy
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

_MANIFEST = "linux-runtime/wheelhouse-manifest-v1.json"
_ARCHIVE = "source/candidate-source.tar"
_PREFIX = "hermes-realtime-0.0.3/"
_SELF = "scripts/qualification_linux_worker.py"
_PRODUCER = "scripts/qualification_linux_producer.py"
_WORKFLOW = ".github/workflows/release-gates.yml"
_MAX_FILE = 512 * 1024**2
_MAX_TOTAL = 4 * 1024**3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    _require(0 < len(raw) <= 4 * 1024**2 and raw.endswith(b"\n"), label + " differs")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " differs") from error
    _require(type(value) is dict and _canonical(value) == raw, label + " differs")
    return cast(dict[str, Any], value)


def _member(path: str) -> None:
    value = PurePosixPath(path)
    _require(
        path == value.as_posix()
        and not value.is_absolute()
        and bool(value.parts)
        and all(part not in {"", ".", ".."} for part in value.parts),
        "Linux worker path differs",
    )


def _inputs(
    root: Path, *, python_version: str | None = None
) -> tuple[dict[str, Any], dict[str, bytes], bytes, bytes, Path]:
    root = root.resolve(strict=True)
    manifest_path = (root / _MANIFEST).resolve(strict=True)
    _require(manifest_path.is_file() and root in manifest_path.parents, "Linux manifest differs")
    document = _load_canonical(manifest_path, "Linux manifest")
    _require(
        document.get("schemaVersion") == 1
        and document.get("purpose") == "realtime_linux_runtime"
        and document.get("platform") == "linux_x86_64"
        and document.get("pythonVersion") == (python_version or platform.python_version()),
        "Linux manifest target differs",
    )
    references = [document.get("requirements"), document.get("constraints")]
    wheels = document.get("wheels")
    _require(
        all(type(item) is dict for item in references)
        and type(wheels) is list
        and 0 < len(wheels) <= 2048
        and all(type(item) is dict for item in wheels),
        "Linux manifest references differ",
    )
    assert isinstance(wheels, list)
    references.extend(wheels)
    contents: dict[str, bytes] = {}
    for reference in references:
        assert isinstance(reference, dict)
        relative = reference.get("relativePath")
        _require(type(relative) is str, "Linux input path differs")
        assert isinstance(relative, str)
        _member(relative)
        path = (root / relative).resolve(strict=True)
        _require(path.is_file() and root in path.parents, "Linux input escaped its root")
        raw = path.read_bytes()
        _require(
            0 < len(raw) <= _MAX_FILE
            and reference.get("basename") == path.name
            and reference.get("sha256") == _sha(raw)
            and reference.get("bytes") == len(raw)
            and relative not in contents,
            "Linux input bytes differ",
        )
        contents[relative] = raw
    _require(sum(map(len, contents.values())) <= _MAX_TOTAL, "Linux inputs exceed their bound")
    candidates = sorted((root / "candidate").glob("*.whl"))
    _require(len(candidates) == 1 and candidates[0].is_file(), "Linux candidate wheel differs")
    direct = candidates[0].read_bytes()
    _require(0 < len(direct) <= 16 * 1024**2, "Linux candidate wheel differs")
    requirements = contents[document["requirements"]["relativePath"]]
    constraints = contents[document["constraints"]["relativePath"]]
    wheel_bytes = {
        cast(str, item["basename"]): contents[cast(str, item["relativePath"])] for item in wheels
    }
    _require(candidates[0].name not in wheel_bytes, "Linux candidate wheel is duplicated")
    wheel_bytes[candidates[0].name] = direct
    return document, wheel_bytes, requirements, constraints, candidates[0]


def _source_binding(input_root: Path) -> tuple[str, str, str]:
    archive_path = (input_root / _ARCHIVE).resolve(strict=True)
    archive = archive_path.read_bytes()
    _require(0 < len(archive) <= 32 * 1024**2, "Linux source archive differs")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        names = bundle.getnames()
        _require(len(names) == len(set(names)), "Linux source archive is ambiguous")
        values = []
        for relative in (_SELF, _PRODUCER, _WORKFLOW, "uv.lock"):
            member = bundle.getmember(_PREFIX + relative)
            _require(
                member.isfile() and 0 < member.size <= 4 * 1024**2,
                "Linux source member differs",
            )
            stream = bundle.extractfile(member)
            _require(stream is not None, "Linux source member is unavailable")
            assert stream is not None
            with stream:
                values.append(stream.read(4 * 1024**2 + 1))
    own, producer, workflow, lock = values
    _require(
        own == Path(__file__).read_bytes()
        and producer == Path(__file__).with_name("qualification_linux_producer.py").read_bytes(),
        "Linux producer or worker differs from candidate archive",
    )
    return _sha(archive), _sha(workflow), _sha(lock)


def _pins(requirements: bytes) -> set[tuple[str, str]]:
    result = set()
    for line in requirements.decode("ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(" --hash=sha256:")
        _require(len(parts) == 2 and len(parts[1]) == 64, "Linux requirement differs")
        result.add((parts[0], parts[1]))
    return result


def install(input_root: Path, installation: Path) -> None:
    """Install the exact read-only wheel recipe into a new owned volume."""
    _require(sys.platform == "linux", "Linux install worker requires Linux")
    _source_binding(input_root)
    document, wheels, requirements, constraints, candidate = _inputs(input_root)
    installation = installation.resolve(strict=True)
    _require(not any(installation.iterdir()), "Linux installation volume is not empty")
    for name, raw in wheels.items():
        _require(
            (name.split("-", 1)[0].replace("_", "-").lower(), _sha(raw))
            in {
                (pin.split("==", 1)[0].replace("_", "-").lower(), digest)
                for pin, digest in _pins(requirements)
            }
            or name == candidate.name,
            "Linux wheel is not hash-pinned",
        )
    candidate_requirement = installation / "candidate-requirement.txt"
    candidate_requirement.write_text(
        f"hermes-realtime==0.0.3 --hash=sha256:{_sha(candidate.read_bytes())}\n",
        encoding="ascii",
        newline="\n",
    )
    wheel_directories = sorted(
        {
            str((input_root / item["relativePath"]).resolve(strict=True).parent)
            for item in document["wheels"]
        }
        | {str(candidate.parent.resolve(strict=True))}
    )
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--no-index",
        "--no-compile",
        "--require-hashes",
        "--target",
        str(installation / "site"),
    ]
    for directory in wheel_directories:
        command.extend(("--find-links", directory))
    command.extend(
        (
            "-r",
            str((input_root / document["requirements"]["relativePath"]).resolve(strict=True)),
            "-r",
            str(candidate_requirement),
            "-c",
            str((input_root / document["constraints"]["relativePath"]).resolve(strict=True)),
        )
    )
    home = Path("/tmp/hermes-linux-pip-home")
    home.mkdir(mode=0o700)
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=300,
        env={"HOME": str(home), "PATH": os.environ["PATH"], "PYTHONNOUSERSITE": "1"},
    )
    _require(completed.returncode == 0, "offline Linux installation failed")
    candidate_requirement.unlink()
    _require((installation / "site").is_dir(), "Linux installation is unavailable")


class _EntryPoints(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _wheel_files(name: str, raw: bytes) -> tuple[dict[str, tuple[str, int]], dict[str, str], str]:
    source: dict[str, tuple[str, int]] = {}
    scripts: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        entries = bundle.infolist()
        roots = {
            item.filename.split("/", 1)[0]
            for item in entries
            if "/" in item.filename and item.filename.split("/", 1)[0].endswith(".dist-info")
        }
        _require(len(roots) == 1, "Linux wheel metadata differs")
        info = next(iter(roots))
        data = info.removesuffix(".dist-info") + ".data/"
        for entry in entries:
            if entry.is_dir():
                continue
            target = entry.filename
            _member(target)
            if target.startswith(data):
                scheme, target = target[len(data) :].split("/", 1)
                _require(scheme in {"purelib", "platlib"}, "Linux wheel scheme differs")
            payload = bundle.read(entry)
            if not target.endswith(".dist-info/RECORD"):
                source[target] = (_sha(payload), len(payload))
        entry_points = info + "/entry_points.txt"
        if entry_points in bundle.namelist():
            config = _EntryPoints(interpolation=None, strict=True)
            config.read_string(bundle.read(entry_points).decode("utf-8"))
            for section in ("console_scripts", "gui_scripts"):
                if config.has_section(section):
                    for script, target in config[section].items():
                        _require(
                            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", script) is not None
                            and script not in scripts,
                            "Linux wheel entry point differs",
                        )
                        scripts["bin/" + script] = target
    _require(name.endswith(".whl"), "Linux wheel basename differs")
    return source, scripts, info


def _entry_script(target: str) -> bytes:
    match = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):"
        r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
        target,
    )
    _require(match is not None, "Linux wheel entry point target differs")
    assert match is not None
    module, function = match.groups()
    imported = function.split(".", 1)[0]
    return (
        "#!/usr/local/bin/python\n"
        "# -*- coding: utf-8 -*-\n"
        "import re\n"
        "import sys\n"
        f"from {module} import {imported}\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
        f"    sys.exit({function}())\n"
    ).encode()


def _record_path(value: str, owned: set[str]) -> str:
    if value.startswith("../"):
        match = re.fullmatch(
            r"(?:\.\./)+bin/([A-Za-z0-9][A-Za-z0-9_.-]*)",
            value,
        )
        _require(match is not None, "Linux installed RECORD escaped")
        assert match is not None
        value = "bin/" + match.group(1)
    _member(value)
    _require(value in owned, "Linux installed RECORD differs")
    return value


def _validate_record(
    info: str,
    installed: dict[str, bytes],
    owned: set[str],
) -> None:
    record_name = info + "/RECORD"
    try:
        rows = list(csv.reader(installed[record_name].decode("utf-8").splitlines()))
    except (UnicodeError, csv.Error) as error:
        raise ValueError("Linux installed RECORD differs") from error
    observed: set[str] = set()
    for row in rows:
        _require(len(row) == 3, "Linux installed RECORD differs")
        name = _record_path(row[0], owned)
        _require(name in owned and name not in observed, "Linux installed RECORD differs")
        observed.add(name)
        if name == record_name:
            _require(row[1:] == ["", ""], "Linux installed RECORD differs")
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(installed[name]).digest()).rstrip(b"=")
        _require(
            row[1] == "sha256=" + digest.decode("ascii")
            and row[2].isascii()
            and row[2].isdecimal()
            and int(row[2]) == len(installed[name]),
            "Linux installed RECORD differs",
        )
    _require(observed == owned, "Linux installed RECORD coverage differs")


def _files(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "Linux installed namespace is indirect")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        _member(relative)
        raw = path.read_bytes()
        _require(len(raw) <= _MAX_FILE, "Linux installed file exceeds its bound")
        result[relative] = raw
    _require(
        0 < len(result) <= 16384 and sum(map(len, result.values())) <= _MAX_TOTAL,
        "Linux installed inventory exceeds its bound",
    )
    return result


def installed_inventory(input_root: Path, installation: Path) -> tuple[str, int]:
    """Compare every wheel-owned source byte with the isolated installation."""
    _, wheels, _, _, _ = _inputs(input_root)
    installed = _files(installation / "site")
    expected: dict[str, tuple[str, int]] = {}
    generated: dict[str, bytes] = {}
    distributions: list[tuple[str, set[str]]] = []
    for name, raw in wheels.items():
        source, scripts, info = _wheel_files(name, raw)
        _require(
            not (set(expected) | set(generated)).intersection(source),
            "Linux wheel namespace is ambiguous",
        )
        expected.update(source)
        owned = set(source)
        for script, target in scripts.items():
            _require(
                script not in expected and script not in generated,
                "Linux installed script is ambiguous",
            )
            generated[script] = _entry_script(target)
            owned.add(script)
        generated[info + "/INSTALLER"] = b"pip\n"
        generated[info + "/REQUESTED"] = b""
        generated[info + "/RECORD"] = b""
        owned.update({info + "/INSTALLER", info + "/REQUESTED", info + "/RECORD"})
        distributions.append((info, owned))
    _require(set(expected) <= set(installed), "Linux installed source bytes are missing")
    for name, binding in expected.items():
        _require(
            (_sha(installed[name]), len(installed[name])) == binding,
            "Linux installed source bytes differ",
        )
    _require(
        set(installed) == set(expected) | set(generated),
        "Linux installed namespace has unowned files",
    )
    for name, raw in generated.items():
        if not name.endswith("/RECORD"):
            _require(installed[name] == raw, "Linux installer-generated bytes differ")
    for info, owned in distributions:
        _validate_record(info, installed, owned)
    rows = [(name, _sha(raw), len(raw)) for name, raw in sorted(installed.items())]
    return _sha(json.dumps(rows, separators=(",", ":")).encode()), len(rows)


def _import_roots(input_root: Path) -> tuple[str, ...]:
    _, wheels, _, _, _ = _inputs(input_root)
    roots = set()
    for raw in wheels.values():
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            for entry in bundle.infolist():
                first = entry.filename.split("/", 1)[0]
                if not first or first.endswith((".dist-info", ".data")):
                    continue
                candidate = first[:-3] if first.endswith(".py") else first.split(".", 1)[0]
                if candidate.isidentifier():
                    roots.add(candidate)
    _require("hermes_realtime" in roots, "Linux import roots differ")
    return tuple(sorted(roots))


def prove_linux_null_capture(
    *,
    site: Path | None = None,
    scratch_parent: Path | None = None,
    import_roots: tuple[str, ...] = (),
) -> int:
    """Run the Linux null-capture behavior without pytest or a dev dependency."""
    _require(sys.platform == "linux", "Linux null capture requires Linux")
    if site is not None:
        sys.path.insert(0, str(site))
        resolved_site = site.resolve(strict=True)
        for name in import_roots:
            module = importlib.import_module(name)
            locations = []
            if getattr(module, "__file__", None):
                locations.append(Path(cast(str, module.__file__)).resolve(strict=True))
            if getattr(module, "__path__", None):
                locations.extend(Path(value).resolve(strict=True) for value in module.__path__)
            _require(
                bool(locations)
                and all(
                    path == resolved_site or resolved_site in path.parents for path in locations
                ),
                "Linux dependency import escaped the installed namespace",
            )
    import hermes_realtime

    names = sorted(
        module.name
        for module in pkgutil.walk_packages(hermes_realtime.__path__, "hermes_realtime.")
    )
    for name in names:
        importlib.import_module(name)
    hermes_origins = {
        str(Path(cast(str, module.__file__)).resolve(strict=True))
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and module.__name__.split(".", 1)[0] == "hermes_realtime"
    }
    if site is not None:
        _require(
            bool(hermes_origins)
            and all(resolved_site in Path(path).parents for path in hermes_origins),
            "Linux import escaped the installed namespace",
        )
        installed_origins = {
            str(Path(cast(str, module.__file__)).resolve(strict=True))
            for module in tuple(sys.modules.values())
            if getattr(module, "__file__", None)
            and resolved_site in Path(cast(str, module.__file__)).resolve(strict=True).parents
        }
    else:
        installed_origins = hermes_origins
    host = shutil.which("hermes-realtime-host", path=str(site / "bin") if site else None)
    local = shutil.which("hermes-realtime-local", path=str(site / "bin") if site else None)
    _require(host is not None and local is not None, "Linux installed CLI is unavailable")
    assert host is not None and local is not None
    with tempfile.TemporaryDirectory(
        prefix="hermes-linux-null-capture-", dir=scratch_parent
    ) as temporary:
        root = Path(temporary)
        before = {path.relative_to(root) for path in root.rglob("*")}
        before_threads = {thread.ident for thread in threading.enumerate()}
        environment = {
            "HOME": str(root / "home"),
            "PATH": str(site / "bin") if site else os.environ["PATH"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(root),
        }

        def run(executable: str, *arguments: str) -> subprocess.CompletedProcess[str]:
            command = (executable, *arguments)
            if site is not None:
                command = (
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(Path(__file__).resolve(strict=True)),
                    "entrypoint",
                    str(site.resolve(strict=True)),
                    executable,
                    *arguments,
                )
            return subprocess.run(
                command, check=False, capture_output=True, env=environment, text=True, timeout=30
            )

        _require(run(host, "--help").returncode == 0, "Linux host help failed")
        _require(run(local, "--help").returncode == 0, "Linux local help failed")
        local_evidence = run(local, "--evidence-capture")
        _require(
            local_evidence.returncode == 2 and "unrecognized arguments" in local_evidence.stderr,
            "Linux local capture surface differs",
        )
        for flag in ("--evidence-capture", "--evidence-status", "--purge-evidence"):
            response = run(host, flag)
            _require(
                response.returncode == 2
                and response.stdout == '{"error":"unsupported_platform","version":1}\n',
                "Linux host null capture differs",
            )
        after = {path.relative_to(root) for path in root.rglob("*")}
        _require(after == before, "Linux null capture left local residue")
        _require(
            {thread.ident for thread in threading.enumerate()} == before_threads,
            "Linux null capture left a background thread",
        )
    return len(installed_origins)


def observe(input_root: Path, installation: Path, output: Path, scratch: Path) -> None:
    """Observe the read-only installed volume and emit one provisional fact set."""
    _require(sys.platform == "linux", "Linux observe worker requires Linux")
    source_archive_sha256, workflow_sha256, source_lock_sha256 = _source_binding(input_root)
    digest, count = installed_inventory(input_root, installation)
    site = (installation / "site").resolve(strict=True)
    imports = prove_linux_null_capture(
        site=site,
        scratch_parent=scratch.resolve(strict=True),
        import_roots=_import_roots(input_root),
    )
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    stdlib_rows = [
        (path.relative_to(stdlib).as_posix(), _sha(path.read_bytes()), path.stat().st_size)
        for path in sorted(stdlib.rglob("*"))
        if path.is_file() and "site-packages" not in path.parts and "__pycache__" not in path.parts
    ]
    libc_name, libc_version = platform.libc_ver()
    _require(libc_name == "glibc" and bool(libc_version), "Linux libc observation differs")
    value = {
        "version": 1,
        "sourceArchiveSha256": source_archive_sha256,
        "workflowSha256": workflow_sha256,
        "sourceLockSha256": source_lock_sha256,
        "runtime": {
            "pythonVersion": platform.python_version(),
            "soabi": sysconfig.get_config_var("SOABI"),
            "extSuffix": sysconfig.get_config_var("EXT_SUFFIX"),
            "multiarch": sysconfig.get_config_var("MULTIARCH"),
            "glibc": libc_version,
            "interpreterSha256": _sha(Path(sys.executable).resolve(strict=True).read_bytes()),
            "stdlibInventorySha256": _sha(json.dumps(stdlib_rows, separators=(",", ":")).encode()),
        },
        "installation": {
            "installedInventorySha256": digest,
            "installedFileCount": count,
            "importOriginCount": imports,
            "nullCapturePassed": True,
        },
    }
    raw = _canonical(value)
    _require(
        output.parent.resolve(strict=True) == output.resolve().parent and not output.exists(),
        "Linux observation output differs",
    )
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _run_installed_entrypoint(site: Path, script: Path, arguments: tuple[str, ...]) -> None:
    site = site.resolve(strict=True)
    supplied = script
    _require(
        not supplied.is_symlink()
        and arguments
        in {
            ("--help",),
            ("--evidence-capture",),
            ("--evidence-status",),
            ("--purge-evidence",),
        },
        "Linux installed entry point differs",
    )
    script = supplied.resolve(strict=True)
    _require(
        script.is_file()
        and script.parent == site / "bin"
        and script.name in {"hermes-realtime-host", "hermes-realtime-local"},
        "Linux installed entry point differs",
    )
    sys.path.insert(0, str(site))
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")


def main(arguments: list[str] | None = None) -> int:
    values = sys.argv[1:] if arguments is None else arguments
    if len(values) >= 4 and values[0] == "entrypoint":
        _run_installed_entrypoint(Path(values[1]), Path(values[2]), tuple(values[3:]))
        return 0
    _require(len(values) in {3, 5}, "Linux worker arguments differ")
    if values[0] == "install" and len(values) == 3:
        install(Path(values[1]), Path(values[2]))
        return 0
    if values[0] == "observe" and len(values) == 5:
        observe(Path(values[1]), Path(values[2]), Path(values[3]), Path(values[4]))
        return 0
    raise ValueError("Linux worker operation differs")


if __name__ == "__main__":
    raise SystemExit(main())
