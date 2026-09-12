import hashlib
import io
import json
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _input(tmp_path: Path) -> Path:
    from scripts import qualification_linux_worker as worker

    wheel = run_path(str(Path.cwd() / "tests/test_qualification_wheelhouse.py"))["wheel"]
    dependency_name, dependency = wheel("synthetic_base")
    direct_name, direct = wheel("hermes_realtime", "0.0.3", requires=("synthetic-base==1.0",))
    root = tmp_path / "input"
    values = {
        "linux-runtime/requirements.txt": (
            f"synthetic-base==1.0 --hash=sha256:{_sha(dependency)}\n"
        ).encode(),
        "linux-runtime/constraints.txt": b"# No additional constraints\n",
        "linux-runtime/wheels/" + dependency_name: dependency,
        "candidate/" + direct_name: direct,
    }
    for relative, raw in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def ref(role: str, relative: str) -> dict[str, object]:
        raw = values[relative]
        return {
            "role": role,
            "relativePath": relative,
            "basename": Path(relative).name,
            "sha256": _sha(raw),
            "bytes": len(raw),
        }

    manifest = {
        "schemaVersion": 1,
        "purpose": "realtime_linux_runtime",
        "pythonVersion": "3.11.16",
        "platform": "linux_x86_64",
        "requirements": ref("requirements", "linux-runtime/requirements.txt"),
        "constraints": ref("constraints", "linux-runtime/constraints.txt"),
        "wheels": [ref("wheel", "linux-runtime/wheels/" + dependency_name)],
    }
    manifest_path = root / worker._MANIFEST
    manifest_path.write_bytes(_canonical(manifest))
    archive_path = root / worker._ARCHIVE
    archive_path.parent.mkdir(parents=True)
    archive = io.BytesIO()
    sources = {
        "scripts/qualification_linux_worker.py": Path(worker.__file__).read_bytes(),
        "scripts/qualification_linux_producer.py": Path(worker.__file__)
        .with_name("qualification_linux_producer.py")
        .read_bytes(),
        ".github/workflows/release-gates.yml": b"name: release-gates\n",
        "uv.lock": b"version = 1\n",
    }
    with tarfile.open(fileobj=archive, mode="w:") as bundle:
        for relative, raw in sources.items():
            info = tarfile.TarInfo(worker._PREFIX + relative)
            info.size = len(raw)
            bundle.addfile(info, io.BytesIO(raw))
    archive_path.write_bytes(archive.getvalue())
    return root


class _FakeDocker:
    def __init__(self, *, cleanup_fails: bool = False) -> None:
        self.events: list[str] = []
        self.cleanup_fails = cleanup_fails

    def admit_image(self, metadata) -> None:
        self.events.append("image")

    def create_volume(self, name: str) -> None:
        self.events.append("volume")

    def run_owned(self, spec) -> None:
        self.events.append("observe" if spec.volume_read_only else "install")
        if spec.output_root is not None:
            from scripts import qualification_linux_worker as worker

            archive, workflow, lock = worker._source_binding(spec.input_root)
            value = {
                "version": 1,
                "sourceArchiveSha256": archive,
                "workflowSha256": workflow,
                "sourceLockSha256": lock,
                "runtime": {
                    "pythonVersion": "3.11.16",
                    "soabi": "cpython-311-x86_64-linux-gnu",
                    "extSuffix": ".cpython-311-x86_64-linux-gnu.so",
                    "multiarch": "x86_64-linux-gnu",
                    "glibc": "2.36",
                    "interpreterSha256": "8" * 64,
                    "stdlibInventorySha256": "9" * 64,
                },
                "installation": {
                    "installedInventorySha256": "a" * 64,
                    "installedFileCount": 101,
                    "importOriginCount": 12,
                    "nullCapturePassed": True,
                },
            }
            (spec.output_root / "observation.json").write_bytes(_canonical(value))

    def remove_volume(self, name: str) -> None:
        self.events.append("remove")

    def require_absent(self, containers, volume: str) -> None:
        self.events.append("absent")
        if self.cleanup_fails:
            raise ValueError("cleanup absent observation failed")

    def cleanup(self, containers, volume: str) -> None:
        self.events.append("cleanup")


def _image():
    return SimpleNamespace(
        image_reference="docker.io/library/python@sha256:" + "1" * 64,
        python_version="3.11.16",
        config_sha256="2" * 64,
        layer_sha256s=("3" * 64,),
        layer_diff_sha256s=("4" * 64,),
        compressed_bytes=123,
    )


def test_controller_mints_only_after_two_owned_invocations_and_cleanup(tmp_path, monkeypatch):
    from scripts import qualification_linux_producer as producer
    from scripts.qualification_linux_receipt import parse_linux_receipt_payload

    root = _input(tmp_path)
    output = tmp_path / "receipt.json"
    docker = _FakeDocker()
    monkeypatch.setattr(producer.images, "linux_image_metadata", lambda value: _image())
    monkeypatch.setattr(
        producer, "_git", lambda root, *args: "b" * 40 if args[-1] == "HEAD" else "c" * 40
    )
    monkeypatch.setenv("HERMES_CANDIDATE_HEAD", "b" * 40)
    monkeypatch.setenv("GITHUB_REPOSITORY_ID", "1351392603")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    producer._produce(root, Path.cwd(), output, docker, object())
    payload = parse_linux_receipt_payload(output.read_bytes())
    assert docker.events == ["image", "volume", "install", "observe", "remove", "absent"]
    assert payload.cleanup_removed is True and payload.run_id == 42


def test_cleanup_failure_cannot_mint_receipt(tmp_path, monkeypatch):
    from scripts import qualification_linux_producer as producer

    root = _input(tmp_path)
    output = tmp_path / "receipt.json"
    docker = _FakeDocker(cleanup_fails=True)
    monkeypatch.setattr(producer.images, "linux_image_metadata", lambda value: _image())
    monkeypatch.setattr(
        producer, "_git", lambda root, *args: "b" * 40 if args[-1] == "HEAD" else "c" * 40
    )
    monkeypatch.setenv("HERMES_CANDIDATE_HEAD", "b" * 40)
    with pytest.raises(ValueError, match="cleanup"):
        producer._produce(root, Path.cwd(), output, docker, object())
    assert not output.exists() and docker.events[-1] == "cleanup"


def test_docker_specs_make_inputs_and_observed_installation_read_only(tmp_path):
    from scripts.qualification_linux_producer import _ContainerSpec, _Docker

    spec = _ContainerSpec(
        "observe",
        "image@sha256:digest",
        ("python", "worker.py"),
        tmp_path,
        tmp_path,
        "volume",
        True,
        tmp_path,
    )
    arguments = _Docker._create_arguments(object.__new__(_Docker), spec)
    assert "--read-only" in arguments and "--network" in arguments
    assert f"type=bind,source={tmp_path},target=/input,readonly" in arguments
    assert "type=volume,source=volume,target=/installation,readonly" in arguments


def test_official_image_reference_accepts_only_docker_native_normalizations():
    from scripts.qualification_linux_producer import _repo_digest_matches

    digest = "1" * 64
    assert _repo_digest_matches(["python@sha256:" + digest], digest)
    assert _repo_digest_matches(["docker.io/library/python@sha256:" + digest], digest)
    assert not _repo_digest_matches(["publisher.invalid/python@sha256:" + digest], digest)
    assert not _repo_digest_matches(["python@sha256:" + "2" * 64], digest)


def test_container_observation_binds_complete_mount_identity(tmp_path):
    from scripts.qualification_linux_producer import _ContainerSpec, _Docker

    input_root = tmp_path / "input"
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    spec = _ContainerSpec(
        "observe",
        "image@sha256:digest",
        ("python", "worker.py"),
        input_root,
        source_root,
        "owned-volume",
        True,
        output_root,
    )
    row = {
        "Id": "identifier",
        "Name": "/observe",
        "Config": {"Image": spec.image, "Cmd": list(spec.command)},
        "HostConfig": {
            "ReadonlyRootfs": True,
            "NetworkMode": "none",
            "CapDrop": ["ALL"],
            "PidsLimit": 256,
            "SecurityOpt": ["no-new-privileges"],
            "Tmpfs": {
                "/tmp": "rw,nosuid,nodev,noexec,size=268435456",
                "/scratch": "rw,nosuid,nodev,noexec,size=67108864",
            },
        },
        "Mounts": [
            {"Type": "bind", "Source": str(input_root), "Destination": "/input", "RW": False},
            {"Type": "bind", "Source": str(source_root), "Destination": "/source", "RW": False},
            {"Type": "volume", "Name": "owned-volume", "Destination": "/installation", "RW": False},
            {"Type": "bind", "Source": str(output_root), "Destination": "/output", "RW": True},
        ],
    }
    docker = object.__new__(_Docker)
    docker._run = lambda *arguments: SimpleNamespace(stdout=json.dumps([row]))
    docker._validate_container("identifier", spec)
    row["Mounts"][0]["Source"] = str(tmp_path / "substitute")
    with pytest.raises(ValueError, match="mounts"):
        docker._validate_container("identifier", spec)


def test_cleanup_absence_requires_successful_daemon_inventory():
    from scripts.qualification_linux_producer import _Docker

    docker = object.__new__(_Docker)
    docker._run = lambda *arguments: SimpleNamespace(stdout="")
    docker.require_absent(("first", "second"), "volume")

    def unavailable(*arguments):
        raise ValueError("owned Docker operation failed")

    docker._run = unavailable
    with pytest.raises(ValueError, match="Docker operation"):
        docker.require_absent(("first", "second"), "volume")


def test_publisher_redirect_does_not_forward_registry_bearer(monkeypatch):
    from scripts import qualification_linux_producer as producer

    target = "https://signed.publisher.invalid/config?signature=opaque"
    requests: list[urllib.request.Request] = []

    class Response:
        status = 200
        url = target

        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            return None

        def read(self, size: int) -> bytes:
            return b"config"

    def open_request(request):
        requests.append(request)
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                307,
                "redirect",
                {"Location": target},
                None,
            )
        return Response()

    monkeypatch.setattr(producer, "_open", open_request)
    assert (
        producer._http_bytes(
            "https://registry-1.docker.io/v2/library/python/blobs/sha256:digest",
            bearer="secret",
            allow_redirect=True,
        )
        == b"config"
    )
    assert requests[0].get_header("Authorization") == "Bearer secret"
    assert requests[1].get_header("Authorization") is None


def test_prepare_inputs_writes_runtime_only_manifest_and_source_archive(tmp_path, monkeypatch):
    from scripts import qualification_linux_producer as producer

    wheel = run_path(str(Path.cwd() / "tests/test_qualification_wheelhouse.py"))["wheel"]
    dependency_name, dependency = wheel("synthetic_base")
    direct_name, direct = wheel("hermes_realtime", "0.0.3")
    values = {
        "candidate/" + direct_name: direct,
        "linux-runtime/wheels/" + dependency_name: dependency,
        "linux-runtime/constraints.txt": b"# No additional constraints\n",
    }
    for relative, raw in values.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def archive(root):
        stream = io.BytesIO()
        sources = {
            "scripts/qualification_linux_worker.py": Path(producer.worker.__file__).read_bytes(),
            "scripts/qualification_linux_producer.py": Path(producer.__file__).read_bytes(),
            ".github/workflows/release-gates.yml": b"name: release-gates\n",
            "uv.lock": b"version = 1\n",
        }
        with tarfile.open(fileobj=stream, mode="w:") as bundle:
            for relative, raw in sources.items():
                info = tarfile.TarInfo(producer.worker._PREFIX + relative)
                info.size = len(raw)
                bundle.addfile(info, io.BytesIO(raw))
        return stream.getvalue()

    monkeypatch.setattr(producer, "_git_archive", archive)
    producer.prepare_linux_inputs(tmp_path, Path.cwd())
    document = json.loads((tmp_path / "linux-runtime/wheelhouse-manifest-v1.json").read_bytes())
    assert [item["basename"] for item in document["wheels"]] == [dependency_name]
    assert direct_name not in {item["basename"] for item in document["wheels"]}
    assert (tmp_path / "linux-runtime/requirements.txt").read_bytes() == (
        f"synthetic-base==1.0 --hash=sha256:{_sha(dependency)}\n"
    ).encode()
    assert (tmp_path / "source/candidate-source.tar").read_bytes() == archive(Path.cwd())
