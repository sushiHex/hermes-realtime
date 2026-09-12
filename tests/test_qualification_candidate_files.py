"""Input byte leases must bind to genuine source and source-derived wheel authority."""

import base64
import csv
import hashlib
import io
import json
import os
import tarfile
import zipfile
from pathlib import Path
from runpy import run_path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="retained Windows source and input files")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _make_source(tmp_path_factory)


def _make_source(tmp_path_factory, extra_package_members=None):
    if os.name != "nt":
        pytest.skip("genuine Windows source authority")
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive

    root = Path.cwd()
    helpers = run_path(str(root / "tests/test_candidate_source_archive_oracle.py"))
    wheel_helpers = run_path(str(root / "tests/test_revoke_race.py"))
    repository, baseline = helpers["_repository"](tmp_path_factory.mktemp("bound-source"))
    members = wheel_helpers["_source_members"]()
    members.update(extra_package_members or {})
    metadata_name = "hermes_realtime-0.0.3.dist-info/METADATA"
    members[metadata_name] = members[metadata_name].replace(
        b"Requires-Dist: numpy",
        b'Requires-Dist: moonshine-voice==0.1.0; extra == "local"\n'
        b'Requires-Dist: kokoro-onnx==0.6.1; extra == "local"\nRequires-Dist: numpy',
    )
    for name, raw in members.items():
        if name.startswith("hermes_realtime/"):
            path = repository / "src" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    (repository / "pyproject.toml").write_bytes(
        wheel_helpers["_SOURCE_PROJECT"].replace(
            b"local = [", b'local = ["moonshine-voice==0.1.0", "kokoro-onnx==0.6.1", '
        )
        + b'\n[tool.hatch.build.targets.sdist]\ninclude = ["/LICENSE", "/README.md", '
        b'"/pyproject.toml", "/src/**", "/docs/**", "/scripts/**"]\nexclude = []\n'
    )
    (repository / "README.md").write_bytes(b"# Synthetic description\n")
    (repository / "LICENSE").write_bytes(b"synthetic license\n")
    (repository / ".gitignore").write_bytes(b"__pycache__/\n.runtime/\n")
    dependency_helpers = run_path(str(Path(__file__).with_name("test_qualification_wheelhouse.py")))
    lock = 'version = 1\nrevision = 3\nrequires-python = "==3.11.*"\n'
    for name, version in (
        ("hatchling", "1.27.0"),
        ("aiohttp", "3.11.0"),
        ("pydantic", "2.11.0"),
        ("numpy", "1.26.0"),
        ("moonshine-voice", "0.1.0"),
        ("kokoro-onnx", "0.6.1"),
    ):
        basename, raw = dependency_helpers["wheel"](name.replace("-", "_"), version)
        lock += (
            f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
            'wheels = [{ url = "https://files.pythonhosted.org/packages/'
            + basename
            + '", hash = "sha256:'
            + hashlib.sha256(raw).hexdigest()
            + '", size = '
            + str(len(raw))
            + " }]\n"
        )
    (repository / "uv.lock").write_text(lock, encoding="utf-8", newline="\n")
    for relative in (
        "docs/evidence-capture.md",
        "docs/qualification-execution.md",
        "scripts/qualify_evidence_slice_zero.py",
        "scripts/qualify_hermes_v020_pluginmanager.py",
        "scripts/qualification_build_worker.py",
        *[
            path.relative_to(root).as_posix()
            for path in sorted((root / "scripts/schemas").glob("*.json"))
        ],
    ):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    helpers["_git"]("add", ".", cwd=repository)
    helpers["_git"]("commit", "-qm", "synthetic qualification source", cwd=repository)
    identity = helpers["_identity"](repository, baseline)
    archive = capture_candidate_source_archive(repository, identity, helpers["_pin"]())
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    record_name = "hermes_realtime-0.0.3.dist-info/RECORD"
    del members[record_name]
    for name, raw in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + digest, len(raw)))
    writer.writerow((record_name, "", ""))
    members[record_name] = record.getvalue().encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as wheel:
        for name, raw in members.items():
            wheel.writestr(name, raw)
    return repository, archive, identity, stream.getvalue()


@pytest.fixture
def bound_graph(tmp_path, source):
    from scripts.candidate_source_archive_oracle import _archive_bytes_for_consumer

    repository, archive, identity, wheel = source
    helpers = run_path(str(Path.cwd() / "tests/test_qualify_evidence_slice_zero.py"))
    root, manifest, _, _, _ = helpers["_make_input_root"](tmp_path)
    document = json.loads(manifest.read_bytes())
    document["candidate"] = {
        "baselineCommit": identity.canonical_baseline_oid,
        "candidateCommit": identity.candidate_head_oid,
        "tree": identity.candidate_tree_oid,
        "canonicalDiffSha256": identity.canonical_diff_sha256,
        "version": "0.0.3",
    }
    sources = {
        "governing_plan": "docs/evidence-capture.md",
        "qualification_runner": "scripts/qualify_evidence_slice_zero.py",
        "hermes_pluginmanager_runner": "scripts/qualify_hermes_v020_pluginmanager.py",
        **{
            role: f"scripts/schemas/{filename}"
            for role, filename in (
                ("benchmark_machine_schema", "benchmark-machine-v1.schema.json"),
                ("benchmark_report_schema", "benchmark-report-v1.schema.json"),
                ("wheelhouse_manifest_schema", "wheelhouse-manifest-v1.schema.json"),
                ("qualification_input_schema", "qualification-input-v1.schema.json"),
                ("qualification_report_schema", "qualification-report-v1.schema.json"),
                ("release_manifest_schema", "release-manifest-v1.schema.json"),
            )
        },
    }
    for reference in document["files"]:
        role = reference["role"]
        raw = (
            (repository / sources[role]).read_bytes()
            if role in sources
            else _archive_bytes_for_consumer(archive, identity)
            if role == "candidate_source_archive"
            else wheel
            if role
            in {
                "direct_wheel",
                "direct_wheel_repeat",
                "sdist_built_wheel",
                "sdist_built_wheel_repeat",
            }
            else _sdist(source)
            if role in {"sdist", "sdist_repeat"}
            else None
        )
        if raw is not None:
            (root / reference["relativePath"]).write_bytes(raw)
            reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    for key, role in helpers["EXPECTED_SCHEMA_ROLES"].items():
        document["expected"][key] = next(
            item["sha256"] for item in document["files"] if item["role"] == role
        )
    return root, manifest, document, archive, identity


def freeze(graph):
    from scripts.qualify_evidence_slice_zero import canonical_json_bytes
    from scripts.retained_qualification_inputs import retain_qualification_input_files

    root, manifest, document, _, _ = graph
    raw = canonical_json_bytes(document)
    manifest.write_bytes(raw)
    return retain_qualification_input_files(root, manifest.name, hashlib.sha256(raw).hexdigest())


def test_input_identity_source_blobs_and_all_four_wheels_bind_and_expire(bound_graph):
    from scripts.qualification_candidate_files import bind_candidate_files, candidate_file_metadata

    with freeze(bound_graph) as files:
        receipt = bind_candidate_files(files, bound_graph[3], bound_graph[4])
        metadata = candidate_file_metadata(receipt)
        assert metadata.source_commit == bound_graph[4].candidate_head_oid
        assert len(metadata.wheel_sha256s) == 4
        assert len(set(metadata.wheel_sha256s)) == 1
    with pytest.raises(ValueError, match="closed"):
        candidate_file_metadata(receipt)


@pytest.mark.parametrize(
    "field", ["tree", "candidateCommit", "baselineCommit", "canonicalDiffSha256"]
)
def test_manifest_claims_cannot_substitute_for_actual_source_identity(bound_graph, field):
    from scripts.qualification_candidate_files import bind_candidate_files

    value = bound_graph[2]["candidate"][field]
    bound_graph[2]["candidate"][field] = "0" * len(value)
    with freeze(bound_graph) as files, pytest.raises(ValueError, match="candidate identity"):
        bind_candidate_files(files, bound_graph[3], bound_graph[4])


@pytest.mark.parametrize(
    "role",
    [
        "governing_plan",
        "qualification_runner",
        "qualification_report_schema",
        "sdist_built_wheel_repeat",
        "sdist",
        "sdist_repeat",
    ],
)
def test_rehashed_foreign_role_bytes_do_not_become_candidate_authority(bound_graph, role):
    from scripts.qualification_candidate_files import bind_candidate_files

    root, _, document, archive, identity = bound_graph
    reference = next(item for item in document["files"] if item["role"] == role)
    raw = b"synthetic foreign artifact\n"
    (root / reference["relativePath"]).write_bytes(raw)
    reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    if role == "qualification_report_schema":
        document["expected"]["qualificationReportSchemaSha256"] = reference["sha256"]
    with freeze(bound_graph) as files, pytest.raises(ValueError):
        bind_candidate_files(files, archive, identity)


def _sdist(source, corruption=None):
    from scripts.candidate_source_archive_oracle import verified_candidate_source_archive_metadata

    repository, archive, _, wheel = source
    metadata = verified_candidate_source_archive_metadata(archive)
    members = {
        item.path: (repository / item.path).read_bytes()
        for item in metadata.manifest
        if item.kind == "file"
        and (
            item.path in {"LICENSE", "README.md", "pyproject.toml", ".gitignore"}
            or item.path.startswith(("src/", "docs/", "scripts/"))
        )
    }
    with zipfile.ZipFile(io.BytesIO(wheel)) as distribution:
        members["PKG-INFO"] = distribution.read("hermes_realtime-0.0.3.dist-info/METADATA")
    if corruption == "changed":
        members["README.md"] = b"changed source blob\n"
    elif corruption == "missing":
        del members["README.md"]
    elif corruption == "unlisted":
        members["scripts/unlisted.py"] = b"pass\n"
    elif corruption == "traversal":
        members["../escaped.py"] = b"pass\n"
    elif corruption == "metadata":
        members["PKG-INFO"] = members["PKG-INFO"].replace(b"Version: 0.0.3", b"Version: 0.0.4")
    elif corruption == "vcs_changed":
        members[".gitignore"] = b"foreign-fixture/\n"
    elif corruption == "vcs_missing":
        del members[".gitignore"]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as output:
        for name, raw in members.items():
            member = tarfile.TarInfo("hermes_realtime-0.0.3/" + name)
            member.size = len(raw)
            if corruption == "link" and name == "README.md":
                member.type, member.linkname, member.size = tarfile.SYMTYPE, "LICENSE", 0
            output.addfile(member, io.BytesIO(raw))
            if corruption == "duplicate" and name == "README.md":
                output.addfile(member, io.BytesIO(raw))
    return stream.getvalue()


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "changed",
        "missing",
        "unlisted",
        "traversal",
        "metadata",
        "link",
        "duplicate",
        "vcs_changed",
        "vcs_missing",
    ],
)
def test_sdist_requires_exact_selected_source_blobs_and_wheel_metadata(source, corruption):
    from scripts import candidate_wheel
    from scripts.qualification_sdist import inspect_candidate_sdist

    _, archive, identity, wheel = source
    bound_wheel = candidate_wheel._verify_candidate_wheel_bytes_v1(
        archive, identity, wheel, hashlib.sha256(wheel).hexdigest()
    )
    raw = _sdist(source, corruption)
    if corruption is None:
        metadata = inspect_candidate_sdist(archive, identity, bound_wheel, raw)
        assert metadata.sha256 == hashlib.sha256(raw).hexdigest()
        assert metadata.source_commit == identity.candidate_head_oid
    else:
        with pytest.raises(ValueError):
            inspect_candidate_sdist(archive, identity, bound_wheel, raw)


@pytest.fixture
def dependency_graph(bound_graph):
    from scripts import qualify_evidence_slice_zero as core

    helpers = run_path(str(Path(__file__).with_name("test_qualification_wheelhouse.py")))
    root, _, document, _, _ = bound_graph
    build = helpers["wheel"]("hatchling", "1.27.0")
    runtime = dict(
        [
            helpers["wheel"]("aiohttp", "3.11.0"),
            helpers["wheel"]("pydantic", "2.11.0"),
            helpers["wheel"]("numpy", "1.26.0"),
        ]
    )
    providers = dict(
        [
            helpers["wheel"]("moonshine_voice", "0.1.0"),
            helpers["wheel"]("kokoro_onnx", "0.6.1"),
        ]
    )
    for provider, package, version in (
        ("moonshine", "moonshine_voice", "0.1.0"),
        ("kokoro", "kokoro_onnx", "0.6.1"),
    ):
        raw = providers[f"{package}-{version}-py3-none-any.whl"]
        item = next(
            item for item in document["files"] if item["role"] == provider + "_distribution"
        )
        (root / item["relativePath"]).write_bytes(raw)
        item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        document["expected"][provider + "Version"] = version
    for purpose, role in core._WHEELHOUSE_ROLES.items():
        selected = dict([build]) if purpose == "build" else dict(runtime)
        if purpose in {"realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime"}:
            selected.update(providers)
        elif purpose != "build":
            # NumPy and providers are activated by the candidate's local extra.
            selected.pop("numpy-1.26.0-py3-none-any.whl")
        reference = next(item for item in document["files"] if item["role"] == role)
        path = root / reference["relativePath"]
        value = json.loads(path.read_bytes())
        for previous in value["wheels"]:
            old = root / previous["relativePath"]
            assert old.resolve().is_relative_to(root.resolve())
            old.unlink()
        value["platform"] = (
            "linux_x86_64" if purpose == "realtime_linux_runtime" else "windows_amd64"
        )
        value["wheels"] = []
        for name, raw in sorted(selected.items()):
            item = {
                "role": "wheel",
                "relativePath": (path.parent / "wheels" / name).relative_to(root).as_posix(),
                "basename": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
            (root / item["relativePath"]).write_bytes(raw)
            value["wheels"].append(item)
        for key, raw in (
            ("requirements", helpers["pins"](selected)),
            ("constraints", b"# No additional constraints\n"),
        ):
            item = value[key]
            (root / item["relativePath"]).write_bytes(raw)
            item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        raw = core.canonical_json_bytes(value)
        path.write_bytes(raw)
        reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    tool = next(item for item in document["toolIdentities"] if item["role"] == "hatchling")
    (root / tool["artifact"]["relativePath"]).write_bytes(build[1])
    tool["artifact"].update(sha256=hashlib.sha256(build[1]).hexdigest(), bytes=len(build[1]))
    next(item for item in document["toolIdentities"] if item["role"] == "build_python")[
        "version"
    ] = document["expected"]["pythonFullVersion"]
    return bound_graph


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "platform",
        "python",
        "installer_option",
        "hatchling",
        "candidate_dependency",
        "local_dependency",
    ],
)
def test_retained_dependency_binding_checks_purposes_tools_and_pins(dependency_graph, mutation):
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import (
        bind_dependency_files,
        dependency_file_metadata,
    )

    root, _, document, archive, identity = dependency_graph
    if mutation in {
        "platform",
        "python",
        "installer_option",
        "candidate_dependency",
        "local_dependency",
    }:
        target_role = (
            "windows_direct_runtime_wheelhouse_manifest"
            if mutation == "local_dependency"
            else "linux_runtime_wheelhouse_manifest"
        )
        reference = next(item for item in document["files"] if item["role"] == target_role)
        path = root / reference["relativePath"]
        value = json.loads(path.read_bytes())
        if mutation == "platform":
            value["platform"] = "windows_amd64"
        elif mutation == "python":
            value["pythonVersion"] = "3.12.0"
        elif mutation == "installer_option":
            item = value["requirements"]
            raw = (
                root / item["relativePath"]
            ).read_bytes() + b"--index-url https://example.invalid/simple\n"
            (root / item["relativePath"]).write_bytes(raw)
            item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        else:
            prefix = "numpy-" if mutation == "local_dependency" else "aiohttp-"
            removed = next(item for item in value["wheels"] if item["basename"].startswith(prefix))
            old = root / removed["relativePath"]
            assert old.resolve().is_relative_to(root.resolve())
            old.unlink()
            value["wheels"].remove(removed)
            item = value["requirements"]
            raw = b"".join(
                line + b"\n"
                for line in (root / item["relativePath"]).read_bytes().splitlines()
                if removed["sha256"].encode() not in line
            )
            (root / item["relativePath"]).write_bytes(raw)
            item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        raw = core.canonical_json_bytes(value)
        path.write_bytes(raw)
        reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    elif mutation == "hatchling":
        item = next(
            tool["artifact"] for tool in document["toolIdentities"] if tool["role"] == "hatchling"
        )
        raw = b"synthetic foreign build tool\n"
        (root / item["relativePath"]).write_bytes(raw)
        item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    with freeze(dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        if mutation is not None:
            with pytest.raises(ValueError):
                bind_dependency_files(files, candidate)
        else:
            receipt = bind_dependency_files(files, candidate)
            observation = dependency_file_metadata(receipt)
            assert len(observation.wheelhouses) == 5
            assert (
                observation.qualification_input_sha256
                == hashlib.sha256(core.canonical_json_bytes(document)).hexdigest()
            )
    if mutation is None:
        with pytest.raises(ValueError, match="closed"):
            dependency_file_metadata(receipt)


@pytest.mark.parametrize(
    "purpose",
    [
        "build",
        "hermes_v020_pluginmanager_runtime",
        "realtime_linux_runtime",
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
    ],
)
def test_rehashed_dependency_requires_the_candidate_source_lock(
    dependency_graph, purpose, monkeypatch
):
    from scripts import qualification_dependency_files as dependencies
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import bind_dependency_files

    helpers = run_path(str(Path(__file__).with_name("test_qualification_wheelhouse.py")))
    root, _, document, archive, identity = dependency_graph
    role = core._WHEELHOUSE_ROLES[purpose]
    reference = next(item for item in document["files"] if item["role"] == role)
    path = root / reference["relativePath"]
    value = json.loads(path.read_bytes())
    name, version = ("hatchling", "1.27.0") if purpose == "build" else ("aiohttp", "3.11.0")
    basename, replacement = helpers["wheel"](name, version, python=">=3.11.0")
    inspect = dependencies.inspect_wheelhouse_files

    def inspect_admitted_only(**arguments):
        assert replacement not in arguments["wheels"].values(), (
            "unadmitted wheel reached ZIP parser"
        )
        return inspect(**arguments)

    monkeypatch.setattr(dependencies, "inspect_wheelhouse_files", inspect_admitted_only)
    item = next(item for item in value["wheels"] if item["basename"] == basename)
    (root / item["relativePath"]).write_bytes(replacement)
    item.update(sha256=hashlib.sha256(replacement).hexdigest(), bytes=len(replacement))
    if purpose == "build":
        tool = next(
            tool["artifact"] for tool in document["toolIdentities"] if tool["role"] == "hatchling"
        )
        (root / tool["relativePath"]).write_bytes(replacement)
        tool.update(sha256=item["sha256"], bytes=len(replacement))
    wheels = {
        item["basename"]: (root / item["relativePath"]).read_bytes() for item in value["wheels"]
    }
    requirements = helpers["pins"](wheels)
    item = value["requirements"]
    (root / item["relativePath"]).write_bytes(requirements)
    item.update(sha256=hashlib.sha256(requirements).hexdigest(), bytes=len(requirements))
    raw = core.canonical_json_bytes(value)
    path.write_bytes(raw)
    reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    with freeze(dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        with pytest.raises(ValueError, match="source lock"):
            bind_dependency_files(files, candidate)


@pytest.mark.parametrize("provider", ["moonshine", "kokoro"])
@pytest.mark.parametrize("mutation", ["distribution", "version", "missing_direct", "missing_sdist"])
def test_provider_roles_require_the_same_real_windows_wheel_files(
    dependency_graph, provider, mutation
):
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import bind_dependency_files

    root, _, document, archive, identity = dependency_graph
    if mutation == "distribution":
        reference = next(v for v in document["files"] if v["role"] == provider + "_distribution")
        raw = b"synthetic substituted distribution\n"
        (root / reference["relativePath"]).write_bytes(raw)
        reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    elif mutation == "version":
        document["expected"][provider + "Version"] = "9.9.9"
    else:
        role = (
            "windows_direct_runtime_wheelhouse_manifest"
            if mutation == "missing_direct"
            else "windows_sdist_built_runtime_wheelhouse_manifest"
        )
        reference = next(v for v in document["files"] if v["role"] == role)
        path = root / reference["relativePath"]
        value = json.loads(path.read_bytes())
        removed = next(v for v in value["wheels"] if v["basename"].startswith(provider + "_"))
        value["wheels"].remove(removed)
        target = root / removed["relativePath"]
        assert target.resolve().is_relative_to(root.resolve())
        target.unlink()
        requirements = value["requirements"]
        raw = b"".join(
            line + b"\n"
            for line in (root / requirements["relativePath"]).read_bytes().splitlines()
            if removed["sha256"].encode() not in line
        )
        (root / requirements["relativePath"]).write_bytes(raw)
        requirements.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        raw = core.canonical_json_bytes(value)
        path.write_bytes(raw)
        reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    with freeze(dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        with pytest.raises(ValueError, match="provider distribution|dependency closure"):
            bind_dependency_files(files, candidate)


@pytest.mark.parametrize(
    "purpose",
    [
        "build",
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
        "realtime_linux_runtime",
        "hermes_v020_pluginmanager_runtime",
    ],
)
def test_source_locked_but_unrelated_package_cannot_join_a_purpose(dependency_graph, purpose):
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import bind_dependency_purpose

    root, _, document, archive, identity = dependency_graph
    references = {row["role"]: row for row in document["files"]}
    origin = (
        "windows_direct_runtime_wheelhouse_manifest"
        if purpose == "build"
        else "build_wheelhouse_manifest"
    )
    original = json.loads((root / references[origin]["relativePath"]).read_bytes())
    prefix = "kokoro_onnx-" if purpose == "build" else "hatchling-"
    extra = next(row for row in original["wheels"] if row["basename"].startswith(prefix))
    raw = (root / extra["relativePath"]).read_bytes()
    reference = references[core._WHEELHOUSE_ROLES[purpose]]
    path = root / reference["relativePath"]
    manifest = json.loads(path.read_bytes())
    target = path.parent / "wheels" / extra["basename"]
    target.write_bytes(raw)
    manifest["wheels"].append({**extra, "relativePath": target.relative_to(root).as_posix()})
    manifest["wheels"].sort(key=lambda row: row["basename"])
    requirements = manifest["requirements"]
    target = root / requirements["relativePath"]
    name, version = ("kokoro-onnx", "0.6.1") if purpose == "build" else ("hatchling", "1.27.0")
    raw = target.read_bytes() + f"{name}=={version} --hash=sha256:{extra['sha256']}\n".encode()
    target.write_bytes(raw)
    requirements.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    raw = core.canonical_json_bytes(manifest)
    path.write_bytes(raw)
    reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    with freeze(dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        with pytest.raises(ValueError, match="unrelated"):
            bind_dependency_purpose(files, candidate, purpose=purpose)


def test_authenticated_linux_target_cannot_authorize_a_different_recipe(dependency_graph):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import bind_dependency_purpose
    from scripts.qualification_wheelhouse import (
        _LinuxWheelRecipeBinding,
        _mint_authenticated_linux_target,
        _target,
    )

    environment, tags = _target("3.11.16", "linux_x86_64")
    foreign = _LinuxWheelRecipeBinding(
        "0" * 40,
        "1" * 40,
        "2" * 64,
        ("foreign.whl", "3" * 64, 1),
        "4" * 64,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        (("foreign.whl", "3" * 64, 1),),
    )
    target = _mint_authenticated_linux_target(
        environment,
        tags,
        foreign,
        ("foreign-image", "8" * 64, ("9" * 64,), ("a" * 64,)),
    )
    root, _, _, archive, identity = dependency_graph
    with freeze(dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        with pytest.raises(ValueError, match="target recipe differs"):
            bind_dependency_purpose(
                files,
                candidate,
                purpose="realtime_linux_runtime",
                linux_target=target,
            )


def test_complete_dependency_binding_accepts_its_authenticated_linux_target(dependency_graph):
    from scripts.qualification_candidate_files import (
        bind_candidate_files,
        candidate_file_metadata,
    )
    from scripts.qualification_dependency_files import (
        _source_locked_wheels,
        bind_dependency_files,
        dependency_file_metadata,
    )
    from scripts.qualification_wheelhouse import (
        _LinuxWheelRecipeBinding,
        _mint_authenticated_linux_target,
        _target,
    )

    root, _, document, archive, identity = dependency_graph
    linux_reference = next(
        item for item in document["files"] if item["role"] == "linux_runtime_wheelhouse_manifest"
    )
    manifest_raw = (root / linux_reference["relativePath"]).read_bytes()
    manifest = json.loads(manifest_raw)
    requirements = (root / manifest["requirements"]["relativePath"]).read_bytes()
    constraints = (root / manifest["constraints"]["relativePath"]).read_bytes()
    direct_reference = next(item for item in document["files"] if item["role"] == "direct_wheel")
    direct_raw = (root / direct_reference["relativePath"]).read_bytes()
    direct = (
        direct_reference["basename"],
        hashlib.sha256(direct_raw).hexdigest(),
        len(direct_raw),
    )
    requirements += (f"\nhermes-realtime==0.0.3 --hash=sha256:{direct[1]}\n").encode("ascii")
    wheels = [
        (
            item["basename"],
            hashlib.sha256((root / item["relativePath"]).read_bytes()).hexdigest(),
            len((root / item["relativePath"]).read_bytes()),
        )
        for item in manifest["wheels"]
    ]
    wheels.append(("hermes_realtime-0.0.3-py3-none-any.whl", direct[1], direct[2]))
    with freeze(dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        source = candidate_file_metadata(candidate)
        source_lock_sha256, _ = _source_locked_wheels(candidate)
        environment, tags = _target(manifest["pythonVersion"], "linux_x86_64")
        target = _mint_authenticated_linux_target(
            environment,
            tags,
            _LinuxWheelRecipeBinding(
                source.source_commit,
                source.source_tree,
                source.source_archive_sha256,
                direct,
                hashlib.sha256(manifest_raw).hexdigest(),
                source_lock_sha256,
                hashlib.sha256(requirements).hexdigest(),
                hashlib.sha256(constraints).hexdigest(),
                tuple(sorted(wheels)),
            ),
            ("image", "1" * 64, ("2" * 64,), ("3" * 64,)),
        )
        receipt = bind_dependency_files(files, candidate, linux_target=target)
        assert len(dependency_file_metadata(receipt).wheelhouses) == 5
