"""Test the worker's bounded platform-independent wheel container serializer."""

from __future__ import annotations

import io
import os
import struct
import warnings
import zipfile
from pathlib import Path

import pytest

from scripts import qualification_build_worker as worker

_FIXED_TIME = (2020, 2, 2, 0, 0, 0)
_FIXED_ATTR = (0o100644 & 0xFFFF) << 16


def _wheel(
    entries: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    create_system: int = 0,
    archive_comment: bytes = b"variant",
    metadata_variant: bool = True,
) -> bytes:
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w", allowZip64=True) as bundle:
            bundle.comment = archive_comment
            for index, (name, payload) in enumerate(entries):
                timestamp = (2024, 1, 2, 3, 4, 6) if metadata_variant else _FIXED_TIME
                info = zipfile.ZipInfo(name, timestamp)
                info.create_system = create_system
                info.create_version = 63 if metadata_variant else 20
                info.extract_version = 20
                info.internal_attr = index + 1 if metadata_variant else 0
                info.external_attr = ((0o100600 + index) & 0xFFFF) << 16
                info.comment = b"member-comment" if metadata_variant else b""
                info.extra = b"\x0a\x00\x00\x00" if metadata_variant else b""
                bundle.writestr(info, payload, compress_type=compression)
    return output.getvalue()


def _members(raw: bytes) -> tuple[zipfile.ZipInfo, ...]:
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        return tuple(bundle.infolist())


def _backslash_member() -> bytes:
    raw = _wheel([("back/slash.py", b"x")])
    assert raw.count(b"back/slash.py") == 2
    return raw.replace(b"back/slash.py", b"back\\slash.py")


def test_platform_order_compression_and_metadata_variants_converge() -> None:
    payloads = [
        ("hermes_realtime/__init__.py", b"__version__ = '0.0.3'\n"),
        ("hermes_realtime-0.0.3.dist-info/RECORD", b"exact,record,payload\n"),
    ]
    windows = _wheel(payloads, compression=zipfile.ZIP_DEFLATED, create_system=0)
    linux = _wheel(
        list(reversed(payloads)),
        compression=zipfile.ZIP_STORED,
        create_system=3,
        archive_comment=b"different",
    )

    canonical = worker.canonical_wheel_bytes(windows)
    assert canonical == worker.canonical_wheel_bytes(linux)
    assert canonical == worker.canonical_wheel_bytes(canonical)
    assert 0 < len(canonical) <= 15 * 1024**2

    with zipfile.ZipFile(io.BytesIO(canonical)) as bundle:
        assert bundle.comment == b""
        assert bundle.namelist() == sorted(name for name, _ in payloads)
        assert {name: bundle.read(name) for name, _ in payloads} == dict(payloads)
        for info in bundle.infolist():
            assert info.date_time == _FIXED_TIME
            assert info.create_system == 3
            assert info.create_version == 20
            assert info.extract_version == 20
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.flag_bits == 0
            assert info.internal_attr == 0
            assert info.external_attr == _FIXED_ATTR
            assert info.comment == info.extra == b""


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not a ZIP",
        b"x" * (15 * 1024**2 + 1),
        _wheel([("../escape.py", b"x")]),
        _wheel([("/absolute.py", b"x")]),
        _backslash_member(),
        _wheel([("alias.py", b"x"), ("ALIAS.py", b"y")]),
        _wheel([("parent", b"x"), ("parent/child.py", b"y")]),
        _wheel([("directory/", b"")]),
    ],
    ids=(
        "empty",
        "malformed",
        "input-size",
        "traversal",
        "absolute",
        "backslash",
        "case-alias",
        "file-directory-collision",
        "directory-member",
    ),
)
def test_canonical_wheel_refuses_malformed_unsafe_or_ambiguous_inputs(raw: bytes) -> None:
    with pytest.raises((TypeError, ValueError)):
        worker.canonical_wheel_bytes(raw)


def test_canonical_wheel_refuses_duplicate_members() -> None:
    raw = _wheel([("same.py", b"one"), ("same.py", b"two")])
    with pytest.raises(ValueError):
        worker.canonical_wheel_bytes(raw)


def test_canonical_wheel_checks_declared_count_before_zip_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = bytearray(_wheel([("one.py", b"one"), ("two.py", b"two")]))
    end = raw.rfind(b"PK\x05\x06")
    struct.pack_into("<HH", raw, end + 8, 1, 1)

    def refuse_parser(*arguments: object, **options: object) -> None:
        pytest.fail("ZipFile was constructed before the raw central-directory check")

    monkeypatch.setattr(worker.zipfile, "ZipFile", refuse_parser)
    with pytest.raises(ValueError, match="central directory is ambiguous"):
        worker.canonical_wheel_bytes(bytes(raw))


def test_canonical_wheel_refuses_links_encryption_and_unsupported_compression() -> None:
    linked = io.BytesIO()
    with zipfile.ZipFile(linked, "w") as bundle:
        info = zipfile.ZipInfo("linked.py")
        info.create_system = 3
        info.external_attr = (0o120777 & 0xFFFF) << 16
        bundle.writestr(info, b"target")

    unsupported = _wheel([("payload.py", b"payload")], compression=zipfile.ZIP_BZIP2)

    encrypted = bytearray(_wheel([("payload.py", b"payload")]))
    local = encrypted.index(b"PK\x03\x04")
    central = encrypted.index(b"PK\x01\x02")
    local_flags = struct.unpack_from("<H", encrypted, local + 6)[0]
    struct.pack_into("<H", encrypted, local + 6, local_flags | 1)
    struct.pack_into(
        "<H", encrypted, central + 8, struct.unpack_from("<H", encrypted, central + 8)[0] | 1
    )

    for raw in (linked.getvalue(), unsupported, bytes(encrypted)):
        with pytest.raises(ValueError):
            worker.canonical_wheel_bytes(raw)


def test_canonical_wheel_refuses_expansion_member_and_stored_output_bounds() -> None:
    expanded = _wheel([("large.bin", b"x" * (32 * 1024**2 + 1))])
    stored_too_large = _wheel([("large.bin", b"x" * (15 * 1024**2))])
    many = bytearray(_wheel([(f"files/{index:04d}.txt", b"") for index in range(4097)]))
    end = many.rfind(b"PK\x05\x06")
    struct.pack_into("<HH", many, end + 8, 1, 1)

    for raw in (expanded, stored_too_large, bytes(many)):
        with pytest.raises(ValueError):
            worker.canonical_wheel_bytes(raw)


def test_canonicalize_wheel_file_is_atomic_and_reads_only_bounded_files(tmp_path: Path) -> None:
    wheel = tmp_path / "candidate.whl"
    original = _wheel([("payload.py", b"payload")])
    wheel.write_bytes(original)

    worker.canonicalize_wheel_file(wheel)
    assert wheel.read_bytes() == worker.canonical_wheel_bytes(original)
    assert not (tmp_path / "candidate.whl.canonical").exists()

    oversized = tmp_path / "oversized.whl"
    oversized.write_bytes(b"x" * (15 * 1024**2 + 1))
    with pytest.raises(ValueError):
        worker.canonicalize_wheel_file(oversized)
    assert oversized.stat().st_size == 15 * 1024**2 + 1
    assert not (tmp_path / "oversized.whl.canonical").exists()


def test_canonicalize_wheel_file_leaves_original_on_replacement_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "candidate.whl"
    original = _wheel([("payload.py", b"payload")])
    wheel.write_bytes(original)

    def refuse_replace(source: str, destination: str) -> None:
        assert source.endswith(".canonical") and destination == str(wheel)
        raise OSError("synthetic replacement refusal")

    monkeypatch.setattr(worker.os, "replace", refuse_replace)
    with pytest.raises(OSError, match="synthetic replacement refusal"):
        worker.canonicalize_wheel_file(wheel)
    assert wheel.read_bytes() == original
    assert not (tmp_path / "candidate.whl.canonical").exists()


def test_canonicalize_wheel_file_owns_descriptor_if_stream_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "candidate.whl"
    original = _wheel([("payload.py", b"payload")])
    wheel.write_bytes(original)
    real_fdopen = worker.os.fdopen
    write_descriptors: list[int] = []

    def fail_write_stream(descriptor: int, mode: str, **options: object) -> io.BufferedReader:
        if mode == "wb":
            write_descriptors.append(descriptor)
            raise OSError("synthetic stream creation refusal")
        return real_fdopen(descriptor, mode, **options)

    monkeypatch.setattr(worker.os, "fdopen", fail_write_stream)
    with pytest.raises(OSError, match="synthetic stream creation refusal"):
        worker.canonicalize_wheel_file(wheel)
    assert len(write_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(write_descriptors[0])
    assert wheel.read_bytes() == original
    assert not (tmp_path / "candidate.whl.canonical").exists()
