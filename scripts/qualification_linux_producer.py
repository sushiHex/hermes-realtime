"""Own the two-container Linux proof and mint its receipt only after cleanup."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, cast

from scripts import qualification_linux_image as images
from scripts import qualification_linux_receipt as payloads
from scripts import qualification_linux_worker as worker

_REGISTRY = "https://registry-1.docker.io"
_TOKEN = (
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull"
)
_OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
_WORKFLOW = ".github/workflows/release-gates.yml"
_FAILURE_MAX = 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _failure_stage(root: Path) -> str | None:
    path = root / worker._FAILURE
    descriptor: int | None = None
    try:
        observed = os.lstat(path)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size <= 0
            or observed.st_size > _FAILURE_MAX
        ):
            return None
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (observed.st_dev, observed.st_ino, observed.st_size)
        ):
            return None
        chunks = bytearray()
        while len(chunks) <= _FAILURE_MAX:
            chunk = os.read(descriptor, _FAILURE_MAX + 1 - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        raw = bytes(chunks)
        if len(raw) != opened.st_size or len(raw) > _FAILURE_MAX:
            return None
        value = json.loads(raw)
        if (
            type(value) is not dict
            or set(value) != {"version", "stage"}
            or type(value.get("version")) is not int
            or value["version"] != 1
            or type(value.get("stage")) is not str
            or value["stage"] not in worker._FAILURE_STAGES
            or _canonical(value) != raw
        ):
            return None
        return cast(str, value["stage"])
    except BaseException:
        return None
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _owned_output_identity(root: Path) -> tuple[int, int]:
    observed = os.lstat(root)
    get_uid = getattr(os, "geteuid", None)
    get_gid = getattr(os, "getegid", None)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o700
        and callable(get_uid)
        and callable(get_gid)
        and type(observed.st_uid) is int
        and type(observed.st_gid) is int
        and 0 <= observed.st_uid <= 2**31 - 1
        and 0 <= observed.st_gid <= 2**31 - 1,
        "Linux observation output owner differs",
    )
    assert callable(get_uid) and callable(get_gid)
    _require(
        observed.st_uid == get_uid() and observed.st_gid == get_gid(),
        "Linux observation output owner differs",
    )
    return observed.st_uid, observed.st_gid


def _repo_digest_matches(value: object, manifest_sha256: str) -> bool:
    admitted = {"python", "library/python", "docker.io/library/python"}
    return (
        type(value) is list
        and all(type(item) is str for item in value)
        and any(
            item.rsplit("@sha256:", 1) == [repository, manifest_sha256]
            for item in value
            for repository in admitted
        )
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _open(request: urllib.request.Request) -> Any:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect).open(
        request, timeout=30
    )


def _http_bytes(
    url: str,
    *,
    accept: str | None = None,
    bearer: str | None = None,
    allow_redirect: bool = False,
) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    _require(
        parsed.scheme == "https"
        and parsed.username is None
        and parsed.password is None
        and parsed.port is None
        and parsed.hostname in {"auth.docker.io", "registry-1.docker.io"},
        "Linux image publisher URL differs",
    )
    headers = {} if accept is None else {"Accept": accept}
    if bearer is not None:
        headers["Authorization"] = "Bearer " + bearer
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        response = _open(request)
    except urllib.error.HTTPError as error:
        location = error.headers.get("Location")
        if not allow_redirect or error.code not in {302, 307} or type(location) is not str:
            error.close()
            raise ValueError("Linux image publisher response differs") from error
        error.close()
        target = urllib.parse.urljoin(url, location)
        redirected = urllib.parse.urlsplit(target)
        _require(
            redirected.scheme == "https"
            and redirected.username is None
            and redirected.password is None
            and redirected.port is None,
            "Linux image publisher redirect differs",
        )
        redirect_headers = {} if accept is None else {"Accept": accept}
        redirected_request = urllib.request.Request(target, headers=redirect_headers, method="GET")
        try:
            response = _open(redirected_request)
        except urllib.error.HTTPError as redirected_error:
            redirected_error.close()
            raise ValueError("Linux image publisher redirect differs") from redirected_error
        _require(
            response.status == 200 and response.url == target,
            "Linux image publisher redirect differs",
        )
    else:
        _require(
            response.status == 200 and response.url == url, "Linux image publisher response differs"
        )
    with response:
        raw = response.read(64 * 1024 + 1)
    _require(len(raw) <= 64 * 1024, "Linux image publisher response exceeds its bound")
    return cast(bytes, raw)


def _publisher_image() -> tuple[images.AdmittedLinuxRuntimeImageV1, bytes, bytes]:
    token_raw = _http_bytes(_TOKEN)
    token_value = json.loads(token_raw)
    _require(
        type(token_value) is dict and type(token_value.get("token")) is str,
        "Linux image registry token differs",
    )
    token = cast(str, token_value["token"])
    _require(0 < len(token) <= 8192, "Linux image registry token differs")
    digest = images._POLICY.manifest_sha256
    manifest_url = _REGISTRY + "/v2/library/python/manifests/sha256:" + digest
    manifest = _http_bytes(manifest_url, accept=_OCI_MANIFEST, bearer=token)
    _require(_sha(manifest) == digest, "Linux image publisher manifest differs")
    document = json.loads(manifest)
    config_digest = document["config"]["digest"]
    _require(
        type(config_digest) is str and config_digest.startswith("sha256:"),
        "Linux image config reference differs",
    )
    config = _http_bytes(
        _REGISTRY + "/v2/library/python/blobs/" + config_digest,
        accept="application/octet-stream",
        bearer=token,
        allow_redirect=True,
    )
    return images.admit_linux_runtime_image(manifest, config), manifest, config


@dataclass(frozen=True, slots=True)
class _ContainerSpec:
    name: str
    image: str
    command: tuple[str, ...]
    input_root: Path
    source_root: Path
    volume: str
    volume_read_only: bool
    output_root: Path | None = None
    user: tuple[int, int] | None = None


class _Docker:
    def __init__(self) -> None:
        executable = shutil.which("docker")
        _require(executable is not None, "Docker executable is unavailable")
        assert executable is not None
        self.executable = str(Path(executable).resolve(strict=True))
        self._owned_containers: set[str] = set()
        self._owned_volumes: set[str] = set()

    def _run(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            (self.executable, *arguments),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=360,
            env={"PATH": os.environ["PATH"]},
        )
        _require(
            len(result.stdout) <= 16 * 1024**2 and len(result.stderr) <= 1024**2,
            "owned Docker output exceeds its bound",
        )
        if check:
            _require(result.returncode == 0, "owned Docker operation failed")
        return result

    def _container_names(self) -> set[str]:
        result = self._run("container", "ls", "--all", "--format", "{{.Names}}")
        names = result.stdout.splitlines()
        _require(
            all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names),
            "Docker container list differs",
        )
        return set(names)

    def _volume_names(self) -> set[str]:
        result = self._run("volume", "ls", "--format", "{{.Name}}")
        names = result.stdout.splitlines()
        _require(
            all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names),
            "Docker volume list differs",
        )
        return set(names)

    def admit_image(self, metadata: images.LinuxRuntimeImageMetadataV1) -> None:
        self._run("pull", "--platform", "linux/amd64", metadata.image_reference)
        result = self._run("image", "inspect", metadata.image_reference)
        value = json.loads(result.stdout)
        _require(
            type(value) is list and len(value) == 1 and type(value[0]) is dict,
            "Docker image observation differs",
        )
        observed = value[0]
        rootfs = observed.get("RootFS")
        repo_digests = observed.get("RepoDigests")
        manifest = metadata.image_reference.rsplit("@sha256:", 1)[1]
        _require(
            observed.get("Id") == "sha256:" + metadata.config_sha256
            and _repo_digest_matches(repo_digests, manifest)
            and observed.get("Architecture") == "amd64"
            and observed.get("Os") == "linux"
            and type(rootfs) is dict
            and rootfs.get("Type") == "layers"
            and rootfs.get("Layers")
            == ["sha256:" + value for value in metadata.layer_diff_sha256s],
            "Docker image graph differs from publisher admission",
        )

    def create_volume(self, name: str) -> None:
        _require(name not in self._volume_names(), "owned Docker volume name already exists")
        self._owned_volumes.add(name)
        result = self._run(
            "volume", "create", "--label", "hermes-realtime-qualification=linux-v1", name
        )
        _require(result.stdout.strip() == name, "owned Docker volume identity differs")
        observed = json.loads(self._run("volume", "inspect", name).stdout)
        _require(
            type(observed) is list
            and len(observed) == 1
            and observed[0].get("Name") == name
            and observed[0].get("Labels") == {"hermes-realtime-qualification": "linux-v1"},
            "owned Docker volume observation differs",
        )

    def _create_arguments(self, spec: _ContainerSpec) -> tuple[str, ...]:
        _require(
            (spec.output_root is None) == (spec.user is None),
            "owned container user differs",
        )
        if spec.user is not None:
            _require(
                type(spec.user) is tuple
                and len(spec.user) == 2
                and all(
                    type(value) is int and 0 <= value <= 2**31 - 1
                    for value in spec.user
                ),
                "owned container user differs",
            )
        arguments = [
            "create",
            "--name",
            spec.name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
        ]
        if spec.user is not None:
            arguments.extend(("--user", f"{spec.user[0]}:{spec.user[1]}"))
        arguments.extend(
            [
                "--mount",
                f"type=bind,source={spec.input_root},target=/input,readonly",
                "--mount",
                f"type=bind,source={spec.source_root},target=/source,readonly",
                "--mount",
                f"type=volume,source={spec.volume},target=/installation"
                + (",readonly" if spec.volume_read_only else ""),
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,size=268435456,mode=1777",
            ]
        )
        if spec.output_root is not None:
            assert spec.user is not None
            arguments.extend(
                (
                    "--mount",
                    f"type=bind,source={spec.output_root},target=/output",
                    "--tmpfs",
                    "/scratch:rw,nosuid,nodev,noexec,size=67108864,mode=700,"
                    f"uid={spec.user[0]},gid={spec.user[1]}",
                )
            )
        arguments.extend((spec.image, *spec.command))
        return tuple(arguments)

    def _validate_container(self, identifier: str, spec: _ContainerSpec) -> None:
        result = self._run("inspect", identifier)
        rows = json.loads(result.stdout)
        _require(
            type(rows) is list and len(rows) == 1 and type(rows[0]) is dict,
            "owned container observation differs",
        )
        row = rows[0]
        host = row.get("HostConfig")
        mounts = row.get("Mounts")
        config = row.get("Config")
        _require(
            type(host) is dict
            and host.get("ReadonlyRootfs") is True
            and host.get("NetworkMode") == "none"
            and host.get("CapDrop") == ["ALL"]
            and host.get("PidsLimit") == 256
            and host.get("SecurityOpt") == ["no-new-privileges"]
            and host.get("Tmpfs", {}).get("/tmp")
            == "rw,nosuid,nodev,noexec,size=268435456,mode=1777"
            and type(config) is dict
            and config.get("Image") == spec.image
            and config.get("Cmd") == list(spec.command)
            and config.get("User")
            == ("" if spec.user is None else f"{spec.user[0]}:{spec.user[1]}")
            and row.get("Id") == identifier
            and row.get("Name") == "/" + spec.name
            and type(mounts) is list,
            "owned container configuration differs",
        )
        expected: dict[str, tuple[str, str | None, str | None, bool]] = {
            "/input": ("bind", str(spec.input_root), None, False),
            "/source": ("bind", str(spec.source_root), None, False),
            "/installation": ("volume", None, spec.volume, not spec.volume_read_only),
        }
        if spec.output_root is not None:
            _require(spec.user is not None, "owned container user differs")
            assert spec.user is not None
            expected["/output"] = ("bind", str(spec.output_root), None, True)
            _require(
                host.get("Tmpfs", {}).get("/scratch")
                == "rw,nosuid,nodev,noexec,size=67108864,mode=700,"
                f"uid={spec.user[0]},gid={spec.user[1]}",
                "owned observation scratch mount differs",
            )
        _require(len(mounts) == len(expected), "owned container mounts differ")
        actual: dict[str, tuple[str, str | None, str | None, bool]] = {}
        for item in mounts:
            _require(type(item) is dict, "owned container mounts differ")
            destination = item.get("Destination")
            _require(
                type(destination) is str and destination not in actual,
                "owned container mounts differ",
            )
            mount_type = item.get("Type")
            source = item.get("Source") if mount_type == "bind" else None
            name = item.get("Name") if mount_type == "volume" else None
            actual[destination] = (mount_type, source, name, item.get("RW"))
        _require(actual == expected, "owned container mounts differ")

    def run_owned(self, spec: _ContainerSpec) -> None:
        _require(
            spec.name not in self._container_names(), "owned Docker container name already exists"
        )
        self._owned_containers.add(spec.name)
        try:
            created = self._run(*self._create_arguments(spec)).stdout.strip()
            _require(
                re.fullmatch(r"[0-9a-f]{64}", created) is not None, "owned container ID differs"
            )
            self._validate_container(created, spec)
            started = self._run("start", "--attach", created, check=False)
            if started.returncode != 0:
                stage = _failure_stage(spec.output_root) if spec.output_root is not None else None
                if stage is not None:
                    raise ValueError("owned Docker observation failed at " + stage)
                raise ValueError("owned Docker operation failed")
            state = json.loads(self._run("inspect", created).stdout)[0]["State"]
            _require(
                state.get("Status") == "exited" and state.get("ExitCode") == 0,
                "owned container failed",
            )
        finally:
            removed = self._run("rm", "--force", spec.name, check=False)
            if removed.returncode == 0:
                self._owned_containers.discard(spec.name)
            _require(removed.returncode == 0, "owned container cleanup failed")

    def remove_volume(self, name: str) -> None:
        self._run("volume", "rm", name)
        self._owned_volumes.discard(name)

    def require_absent(self, containers: tuple[str, ...], volume: str) -> None:
        _require(
            not set(containers).intersection(self._container_names())
            and volume not in self._volume_names(),
            "owned Linux container or volume remains",
        )

    def cleanup(self, containers: tuple[str, ...], volume: str) -> None:
        existing_containers = self._container_names()
        existing_volumes = self._volume_names()
        failed = False
        for identifier in tuple(self._owned_containers):
            if identifier in existing_containers:
                result = self._run("rm", "--force", identifier, check=False)
                failed = failed or result.returncode != 0
                if result.returncode == 0:
                    self._owned_containers.discard(identifier)
            else:
                self._owned_containers.discard(identifier)
        for name in tuple(self._owned_volumes):
            if name in existing_volumes:
                result = self._run("volume", "rm", "--force", name, check=False)
                failed = failed or result.returncode != 0
                if result.returncode == 0:
                    self._owned_volumes.discard(name)
            else:
                self._owned_volumes.discard(name)
        try:
            self.require_absent(containers, volume)
        except ValueError:
            failed = True
        _require(not failed, "owned Docker cleanup failed")


def _strict_observation(raw: bytes) -> dict[str, Any]:
    _require(0 < len(raw) <= 128 * 1024 and raw.endswith(b"\n"), "Linux observation differs")
    value = json.loads(raw)
    _require(type(value) is dict and _canonical(value) == raw, "Linux observation differs")
    _require(
        set(value)
        == {
            "version",
            "sourceArchiveSha256",
            "workflowSha256",
            "sourceLockSha256",
            "runtime",
            "installation",
        }
        and value["version"] == 1
        and type(value["runtime"]) is dict
        and type(value["installation"]) is dict
        and set(value["runtime"])
        == {
            "pythonVersion",
            "soabi",
            "extSuffix",
            "multiarch",
            "glibc",
            "interpreterSha256",
            "stdlibInventorySha256",
        }
        and set(value["installation"])
        == {
            "installedInventorySha256",
            "installedFileCount",
            "importOriginCount",
            "nullCapturePassed",
        }
        and value["installation"]["nullCapturePassed"] is True,
        "Linux observation fields differ",
    )
    return cast(dict[str, Any], value)


def _git(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(source_root), *arguments),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env={"PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1"},
    )
    _require(result.returncode == 0, "Linux producer source identity failed")
    return result.stdout.strip()


def _git_archive(source_root: Path) -> bytes:
    command = (
        "git",
        "-C",
        str(source_root),
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "archive",
        "--format=tar",
        "--prefix=hermes-realtime-0.0.3/",
        "HEAD",
    )
    values = []
    for _ in range(2):
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=60,
            env={"PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1"},
        )
        _require(
            result.returncode == 0 and 0 < len(result.stdout) <= 32 * 1024**2,
            "Linux source archive failed",
        )
        values.append(result.stdout)
    _require(values[0] == values[1], "Linux source archives differ")
    return values[0]


def _normalized_requirements(wheels: tuple[Path, ...]) -> bytes:
    rows: dict[str, tuple[str, str]] = {}
    for wheel_path in wheels:
        raw = wheel_path.read_bytes()
        _require(0 < len(raw) <= worker._MAX_FILE, "Linux wheel differs")
        with zipfile.ZipFile(wheel_path) as bundle:
            metadata_names = [
                name
                for name in bundle.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            _require(len(metadata_names) == 1, "Linux wheel metadata differs")
            with bundle.open(metadata_names[0]) as stream:
                metadata_raw = stream.read(4 * 1024**2 + 1)
        _require(len(metadata_raw) <= 4 * 1024**2, "Linux wheel metadata differs")
        metadata = BytesParser(policy=policy.compat32).parsebytes(metadata_raw)
        names = metadata.get_all("Name", [])
        versions = metadata.get_all("Version", [])
        _require(len(names) == 1 and len(versions) == 1, "Linux wheel identity differs")
        name, version = names[0], versions[0]
        _require(
            name.isascii()
            and version.isascii()
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is not None
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version) is not None,
            "Linux wheel identity differs",
        )
        canonical_name = re.sub(r"[-_.]+", "-", name).lower()
        _require(canonical_name not in rows, "Linux wheel distribution is duplicated")
        rows[canonical_name] = (version, _sha(raw))
    _require(bool(rows), "Linux wheel recipe is empty")
    return "".join(
        f"{name}=={version} --hash=sha256:{digest}\n"
        for name, (version, digest) in sorted(rows.items())
    ).encode("ascii")


def prepare_linux_inputs(input_root: Path, source_root: Path) -> None:
    """Create the canonical runtime manifest and exact-head archive for the CI handoff."""
    input_root, source_root = input_root.resolve(strict=True), source_root.resolve(strict=True)
    requirements_path = input_root / "linux-runtime/requirements.txt"
    constraints_path = input_root / "linux-runtime/constraints.txt"
    wheels_root = input_root / "linux-runtime/wheels"
    candidates = sorted((input_root / "candidate").glob("*.whl"))
    wheels = sorted(wheels_root.glob("*.whl"))
    _require(
        not requirements_path.exists()
        and constraints_path.is_file()
        and len(candidates) == 1
        and 0 < len(wheels) <= 2048
        and all(path.is_file() for path in wheels),
        "Linux preparation inputs differ",
    )
    manifest_path = input_root / worker._MANIFEST
    archive_path = input_root / worker._ARCHIVE
    _require(
        not manifest_path.exists() and not archive_path.exists(), "Linux preparation output exists"
    )
    requirements_path.write_bytes(_normalized_requirements(tuple(wheels)))

    def reference(role: str, path: Path) -> dict[str, object]:
        resolved = path.resolve(strict=True)
        _require(input_root in resolved.parents, "Linux preparation path escaped")
        raw = resolved.read_bytes()
        relative = resolved.relative_to(input_root).as_posix()
        return {
            "role": role,
            "relativePath": relative,
            "basename": resolved.name,
            "sha256": _sha(raw),
            "bytes": len(raw),
        }

    document = {
        "schemaVersion": 1,
        "purpose": "realtime_linux_runtime",
        "pythonVersion": images._POLICY.python_version,
        "platform": "linux_x86_64",
        "requirements": reference("requirements", requirements_path),
        "constraints": reference("constraints", constraints_path),
        "wheels": [reference("wheel", path) for path in wheels],
    }
    manifest_path.write_bytes(_canonical(document))
    archive_path.parent.mkdir(parents=True, exist_ok=False)
    archive_path.write_bytes(_git_archive(source_root))
    worker._source_binding(input_root)


def _facts(
    input_root: Path, source_root: Path, image: images.LinuxRuntimeImageMetadataV1
) -> dict[str, Any]:
    document, wheels, requirements, constraints, candidate = worker._inputs(
        input_root, python_version=image.python_version
    )
    archive_sha, workflow_sha, lock_sha = worker._source_binding(input_root)
    commit = _git(source_root, "rev-parse", "--verify", "HEAD")
    tree = _git(source_root, "rev-parse", "--verify", "HEAD^{tree}")
    expected_head = os.environ.get("HERMES_CANDIDATE_HEAD")
    _require(expected_head == commit, "Linux producer checkout differs from candidate head")
    direct = candidate.read_bytes()
    combined_requirements = requirements + (
        f"\nhermes-realtime==0.0.3 --hash=sha256:{_sha(direct)}\n"
    ).encode("ascii")
    return {
        "candidate": {
            "commit": commit,
            "tree": tree,
            "sourceArchiveSha256": archive_sha,
            "workflowSha256": workflow_sha,
        },
        "directWheel": {"basename": candidate.name, "sha256": _sha(direct), "bytes": len(direct)},
        "linuxWheelhouse": {
            "manifestSha256": _sha((input_root / worker._MANIFEST).read_bytes()),
            "sourceLockSha256": lock_sha,
            "requirementsSha256": _sha(combined_requirements),
            "constraintsSha256": _sha(constraints),
            "wheels": [
                {"basename": name, "sha256": _sha(raw), "bytes": len(raw)}
                for name, raw in sorted(wheels.items())
            ],
        },
        "image": {
            "reference": image.image_reference,
            "configSha256": image.config_sha256,
            "layerSha256s": list(image.layer_sha256s),
            "layerDiffSha256s": list(image.layer_diff_sha256s),
        },
    }


def _service() -> dict[str, int]:
    def number(name: str) -> int:
        value = os.environ.get(name)
        _require(
            value is not None and value.isascii() and value.isdecimal(),
            "Linux service environment differs",
        )
        assert value is not None
        return int(value)

    result = {
        "repositoryId": number("GITHUB_REPOSITORY_ID"),
        "workflowId": payloads._WORKFLOW_ID,
        "runId": number("GITHUB_RUN_ID"),
        "attempt": number("GITHUB_RUN_ATTEMPT"),
    }
    _require(
        result["repositoryId"] == payloads._REPOSITORY_ID
        and result["attempt"] == payloads._ATTEMPT,
        "Linux service identity differs",
    )
    return result


def _write_receipt(path: Path, facts: dict[str, Any], observation: dict[str, Any]) -> None:
    runtime = observation["runtime"]
    installation = {**observation["installation"], "cleanupRemoved": True}
    _require(
        observation["sourceArchiveSha256"] == facts["candidate"]["sourceArchiveSha256"]
        and observation["workflowSha256"] == facts["candidate"]["workflowSha256"]
        and observation["sourceLockSha256"] == facts["linuxWheelhouse"]["sourceLockSha256"],
        "Linux observation source binding differs",
    )
    document = {
        "version": 1,
        "service": _service(),
        **facts,
        "runtime": {"platform": "linux_x86_64", **runtime},
        "installation": installation,
    }
    raw = _canonical(document)
    payloads.parse_linux_receipt_payload(raw)
    _require(
        not path.exists() and path.parent.resolve(strict=True) == path.resolve().parent,
        "Linux receipt output differs",
    )
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(8))
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _produce(
    input_root: Path,
    source_root: Path,
    output: Path,
    docker: _Docker,
    admitted: images.AdmittedLinuxRuntimeImageV1,
) -> None:
    input_root, source_root = input_root.resolve(strict=True), source_root.resolve(strict=True)
    _require(
        output.parent.resolve(strict=True) == output.resolve().parent and not output.exists(),
        "Linux receipt output differs",
    )
    image = images.linux_image_metadata(admitted)
    facts = _facts(input_root, source_root, image)
    docker.admit_image(image)
    token = secrets.token_hex(12)
    volume = "hermes-linux-volume-" + token
    containers = ("hermes-linux-install-" + token, "hermes-linux-observe-" + token)
    observation_raw: bytes | None = None
    try:
        docker.create_volume(volume)
        with tempfile.TemporaryDirectory(prefix="hermes-linux-observation-") as temporary:
            observed = Path(temporary).resolve(strict=True)
            observer = _owned_output_identity(observed)
            install_spec = _ContainerSpec(
                containers[0],
                image.image_reference,
                (
                    "/usr/local/bin/python",
                    "-I",
                    "-B",
                    "/source/scripts/qualification_linux_worker.py",
                    "install",
                    "/input",
                    "/installation",
                ),
                input_root,
                source_root,
                volume,
                False,
            )
            observe_spec = _ContainerSpec(
                containers[1],
                image.image_reference,
                (
                    "/usr/local/bin/python",
                    "-I",
                    "-S",
                    "-B",
                    "/source/scripts/qualification_linux_worker.py",
                    "observe",
                    "/input",
                    "/installation",
                    "/output/observation.json",
                    "/scratch",
                ),
                input_root,
                source_root,
                volume,
                True,
                observed,
                observer,
            )
            docker.run_owned(install_spec)
            docker.run_owned(observe_spec)
            observation_raw = (observed / "observation.json").read_bytes()
        docker.remove_volume(volume)
        docker.require_absent(containers, volume)
    except BaseException:
        docker.cleanup(containers, volume)
        raise
    assert observation_raw is not None
    _write_receipt(output, facts, _strict_observation(observation_raw))


def produce_linux_receipt(input_root: Path, source_root: Path, output: Path) -> None:
    admitted, _, _ = _publisher_image()
    _produce(input_root, source_root, output, _Docker(), admitted)


def main(arguments: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    values = parser.parse_args(arguments)
    if values.prepare_inputs:
        _require(values.output is None, "Linux preparation output argument differs")
        prepare_linux_inputs(values.input_root, values.source_root)
        return 0
    _require(values.output is not None, "Linux receipt output is required")
    produce_linux_receipt(values.input_root, values.source_root, values.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
