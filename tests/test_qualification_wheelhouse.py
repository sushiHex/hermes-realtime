"""Real wheel bytes must close pinned dependencies before any installer executes."""

import base64
import csv
import hashlib
import io
import zipfile

import pytest


def wheel(
    name, version="1.0", *, requires=(), extras=(), tag="py3-none-any", python=">=3.11", mutation=""
):
    info = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": b"# Synthetic distribution\n",
        f"{info}/METADATA": (
            f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\nRequires-Python: {python}\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
            + "".join(f"Provides-Extra: {extra}\n" for extra in extras)
            + "\n"
        ).encode(),
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n\n".encode(),
    }
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for path, raw in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        writer.writerow((path, "sha256=" + digest, len(raw)))
    writer.writerow((f"{info}/RECORD", "", ""))
    files[f"{info}/RECORD"] = record.getvalue().encode()
    if mutation == "payload":
        files[f"{name}/__init__.py"] = b"altered"
    elif mutation == "unlisted":
        files["unlisted.py"] = b"# synthetic\n"
    elif mutation == "traversal":
        files["../escape.py"] = b"# synthetic\n"
    elif mutation == "pth":
        files["inject.pth"] = b"import synthetic_hook\n"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for path, raw in files.items():
            archive.writestr(zipfile.ZipInfo(path, (2026, 1, 1, 0, 0, 0)), raw)
    return f"{name}-{version}-{tag}.whl", stream.getvalue()


def pins(wheels):
    return "".join(
        f"{name.split('-')[0]}=={name.split('-')[1]} "
        f"--hash=sha256:{hashlib.sha256(raw).hexdigest()}\n"
        for name, raw in sorted(wheels.items())
    ).encode()


def inspect(wheels, requirements=None, constraints=b"", *, platform="windows_amd64"):
    from scripts.qualification_wheelhouse import inspect_wheelhouse_files

    return inspect_wheelhouse_files(
        requirements=pins(wheels) if requirements is None else requirements,
        constraints=constraints,
        wheels=wheels,
        python_version="3.11.9",
        platform=platform,
    )


def test_wheel_bytes_close_pins_metadata_and_target_dependencies():
    wheels = dict(
        [wheel("synthetic_base"), wheel("synthetic_app", requires=("synthetic-base>=1",))]
    )
    result = inspect(wheels, constraints=b"synthetic-base==1.0\n")
    assert tuple(item.name for item in result) == ("synthetic-app", "synthetic-base")
    assert all(item.version == "1.0" for item in result)


@pytest.mark.parametrize("extra", [False, True])
def test_selected_roots_reach_transitive_extras_without_admitting_unrelated_packages(extra):
    from scripts.qualification_wheelhouse import inspect_wheelhouse_files

    wheels = dict(
        [
            wheel("synthetic_app", requires=("synthetic-base[accel]",)),
            wheel(
                "synthetic_base",
                extras=("accel",),
                requires=("synthetic-kernel; extra == 'accel'",),
            ),
            wheel("synthetic_kernel"),
        ]
    )
    if extra:
        wheels.update([wheel("unrelated")])
    arguments = dict(
        requirements=pins(wheels),
        constraints=b"",
        wheels=wheels,
        python_version="3.11.16",
        platform="windows_amd64",
        roots=("synthetic-app",),
    )
    if extra:
        with pytest.raises(ValueError, match="unrelated"):
            inspect_wheelhouse_files(**arguments)
    else:
        assert len(inspect_wheelhouse_files(**arguments)) == 3


@pytest.mark.parametrize("mutation", ["payload", "unlisted", "traversal", "pth"])
def test_altered_or_unlisted_executable_wheel_members_are_refused(mutation):
    with pytest.raises(ValueError):
        inspect(dict([wheel("synthetic_base", mutation=mutation)]))


@pytest.mark.parametrize(
    "text",
    [
        b"--index-url https://example.invalid/simple\n",
        b"-e .\n",
        b"-r nested.txt\n",
        b"synthetic-base @ https://example.invalid/foreign.whl\n",
        b"synthetic-base>=1\n",
        b"synthetic-base==1.0 ; sys_platform == 'win32'\n",
    ],
)
def test_supplied_installer_options_urls_and_unresolved_inputs_are_refused(text):
    with pytest.raises(ValueError):
        inspect(dict([wheel("synthetic_base")]), requirements=text)


@pytest.mark.parametrize("requirement", ["missing-runtime>=1", "synthetic-base>=2"])
def test_incomplete_or_incompatible_runtime_dependency_refuses(requirement):
    wheels = dict([wheel("synthetic_base"), wheel("synthetic_app", requires=(requirement,))])
    with pytest.raises(ValueError, match="dependency"):
        inspect(wheels)


def test_dependency_markers_use_the_requested_platform_not_the_observer():
    wheels = dict([wheel("synthetic_app", requires=('missing-runtime; sys_platform == "linux"',))])
    assert len(inspect(wheels)) == 1
    with pytest.raises(ValueError, match="dependency"):
        inspect(wheels, platform="linux_x86_64")


@pytest.mark.parametrize("option", ["python", "tag", "constraint", "digest"])
def test_target_and_artifact_binding_refuse_mismatch(option):
    wheels = dict(
        [
            wheel(
                "synthetic_base",
                python=">=3.12" if option == "python" else ">=3.11",
                tag="cp311-cp311-manylinux_2_17_x86_64" if option == "tag" else "py3-none-any",
            )
        ]
    )
    requirements = pins(wheels)
    if option == "digest":
        requirements = requirements.rsplit(b":", 1)[0] + b":" + b"0" * 64 + b"\n"
    with pytest.raises(ValueError):
        inspect(
            wheels,
            requirements=requirements,
            constraints=b"synthetic-base==2.0\n" if option == "constraint" else b"",
        )


@pytest.mark.parametrize(
    "case", ["healthy", "missing", "unknown", "transitive", "transitive_healthy"]
)
def test_explicit_and_transitive_extras_close_their_actual_dependencies(case):
    from scripts.qualification_wheelhouse import inspect_wheelhouse_files

    values = [
        wheel(
            "synthetic_app",
            extras=("local",),
            requires=(
                "synthetic-base[accel]; extra == 'local'"
                if case.startswith("transitive")
                else "synthetic-base; extra == 'local'",
            ),
        )
    ]
    if case != "missing":
        values.append(
            wheel(
                "synthetic_base",
                extras=("accel",),
                requires=("synthetic-kernel; extra == 'accel'",),
            )
        )
    if case == "transitive_healthy":
        values.append(wheel("synthetic_kernel"))
    wheels = dict(values)
    options = dict(
        requirements=pins(wheels),
        constraints=b"",
        wheels=wheels,
        python_version="3.11.9",
        platform="windows_amd64",
        root_extras={"synthetic-app": ("unknown" if case == "unknown" else "local",)},
    )
    if case in {"healthy", "transitive_healthy"}:
        assert len(inspect_wheelhouse_files(**options)) == len(values)
    else:
        with pytest.raises(ValueError, match="extra|dependency"):
            inspect_wheelhouse_files(**options)


def _with_member(artifact, path, raw):
    name, payload = artifact
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        members = {entry: source.read(entry) for entry in source.namelist()}
    record_name = next(entry for entry in members if entry.endswith(".dist-info/RECORD"))
    del members[record_name]
    members[path] = raw
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for member, value in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
        writer.writerow((member, "sha256=" + digest, len(value)))
    writer.writerow((record_name, "", ""))
    members[record_name] = record.getvalue().encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for member, value in members.items():
            archive.writestr(zipfile.ZipInfo(member, (2026, 1, 1, 0, 0, 0)), value)
    return name, output.getvalue()


@pytest.mark.parametrize(
    "target",
    [
        "shared/module.py",
        "Shared/MODULE.py",
        "synthetic_b-1.0.data/purelib/shared/module.py",
        "synthetic_b-1.0.data/platlib/shared/module.py",
        "shared/module.py/child.py",
    ],
)
def test_installation_namespace_collisions_are_refused_before_execution(target):
    first = _with_member(wheel("synthetic_a"), "shared/module.py", b"# synthetic shared\n")
    second = _with_member(wheel("synthetic_b"), target, b"# synthetic shared\n")
    with pytest.raises(ValueError, match="installation namespace"):
        inspect(dict([first, second]))


def test_purelib_data_maps_into_the_same_unambiguous_import_namespace():
    artifact = _with_member(
        wheel("synthetic_a"), "synthetic_a-1.0.data/purelib/shared/extra.py", b"# synthetic\n"
    )
    assert len(inspect(dict([artifact]))) == 1


@pytest.mark.parametrize(
    "member", ["synthetic_a.libs/libstdc++-6.dll", "synthetic_a/data/ordinary resource.bin"]
)
def test_ordinary_native_resource_names_keep_their_exact_record_bytes(member):
    artifact = _with_member(wheel("synthetic_a"), member, b"synthetic native resource")
    inventory = inspect(dict([artifact]))
    assert (member, hashlib.sha256(b"synthetic native resource").hexdigest(), 25) in (
        inventory[0].members
    )


@pytest.mark.parametrize(
    "member",
    [
        "synthetic_a/../escape.dll",
        "synthetic_a//empty.dll",
        "synthetic_a/./alias.dll",
        "synthetic_a/native.dll:stream",
        "synthetic_a/CON.dll",
        "synthetic_a/trailing.dll.",
        "synthetic_a/trailing.dll ",
        "synthetic_a/control\x01.dll",
        "synthetic_a\\redirect.dll",
        "/absolute.dll",
    ],
)
def test_native_resource_names_cannot_escape_or_alias_the_installation(member):
    artifact = _with_member(wheel("synthetic_a"), member, b"synthetic resource")
    with pytest.raises(ValueError):
        inspect(dict([artifact]))


def test_isolated_inventory_preserves_inert_path_hook_bytes_without_processing_them():
    from scripts.qualification_wheelhouse import inspect_wheelhouse_files

    artifact = wheel("synthetic_base", mutation="pth")
    # Rebuild RECORD so this is an original distribution member, not injection.
    artifact = _with_member(artifact, "inject.pth", b"import synthetic_hook\n")
    wheels = dict([artifact])
    with pytest.raises(ValueError, match="startup hooks"):
        inspect(wheels)
    result = inspect_wheelhouse_files(
        requirements=pins(wheels),
        constraints=b"",
        wheels=wheels,
        python_version="3.11.16",
        platform="windows_amd64",
        site_processing=False,
    )
    assert any(name == "inject.pth" for name, _, _ in result[0].members)


@pytest.mark.parametrize("filename", ["METADATA", "WHEEL", "RECORD"])
def test_vendored_metadata_is_payload_under_the_one_top_level_distribution(filename):
    member = "synthetic_a/_vendor/peer-1.0.dist-info/" + filename
    artifact = _with_member(wheel("synthetic_a"), member, b"synthetic vendored payload\n")
    inventory = inspect(dict([artifact]))
    assert len(inventory) == 1 and inventory[0].name == "synthetic-a"
    assert any(path == member for path, _, _ in inventory[0].members)


@pytest.mark.parametrize(
    "target",
    [
        "foreign-1.0.data/purelib/extra.py",
        "synthetic_a-1.0.data/scripts/tool",
        "synthetic_a-1.0.data/headers/header.h",
        "synthetic_a-1.0.data/data/share/resource.bin",
    ],
)
def test_unqualified_data_installation_schemes_are_explicitly_unavailable(target):
    artifact = _with_member(wheel("synthetic_a"), target, b"# synthetic\n")
    with pytest.raises(ValueError, match="installation scheme"):
        inspect(dict([artifact]))


@pytest.mark.parametrize("version", ["2.1", "2.2", "2.3", "2.4"])
@pytest.mark.parametrize("fault", [None, "missing", "traversal", "duplicate", "future_field"])
def test_license_file_compatibility_preserves_dependency_and_record_validation(version, fault):
    artifact = wheel("synthetic_legacy")
    info = "synthetic_legacy-1.0.dist-info"
    license_name = "../LICENSE.txt" if fault == "traversal" else "LICENSE.txt"
    extra = "License-Expression: MIT\n" if fault == "future_field" and version != "2.4" else ""
    if fault == "future_field" and version == "2.4":
        extra = "Unknown-Execution-Requirement: unbound\n"
    text = (
        f"Metadata-Version: {version}\nName: synthetic_legacy\nVersion: 1.0\n"
        "Requires-Python: >=3.11\n"
        + extra
        + f"License-File: {license_name}\n"
        + (f"License-File: {license_name}\n" if fault == "duplicate" else "")
        + "\n"
    )
    artifact = _with_member(artifact, info + "/METADATA", text.encode())
    if fault != "missing":
        artifact = _with_member(
            artifact,
            info + ("/licenses/" if version == "2.4" else "/") + "LICENSE.txt",
            b"synthetic license\n",
        )
    if fault is None:
        assert inspect(dict([artifact]))[0].name == "synthetic-legacy"
    else:
        with pytest.raises(ValueError):
            inspect(dict([artifact]))


@pytest.mark.parametrize("version", ["2.1", "2.2", "2.3"])
@pytest.mark.parametrize("both", [False, True])
def test_pre_24_hatch_license_layout_requires_one_unambiguous_member(version, both):
    info = "synthetic_legacy-1.0.dist-info"
    artifact = _with_member(
        wheel("synthetic_legacy"),
        info + "/METADATA",
        (
            f"Metadata-Version: {version}\nName: synthetic_legacy\n"
            "Version: 1.0\nLicense-File: LICENSE.txt\n\n"
        ).encode(),
    )
    artifact = _with_member(artifact, info + "/licenses/LICENSE.txt", b"synthetic license\n")
    if both:
        artifact = _with_member(artifact, info + "/LICENSE.txt", b"ambiguous legacy license\n")
        with pytest.raises(ValueError):
            inspect(dict([artifact]))
    else:
        assert inspect(dict([artifact]))[0].name == "synthetic-legacy"
