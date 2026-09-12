"""Isolated build-tool worker; its output alone is never qualification evidence."""

from __future__ import annotations

import io
import os
import stat
import struct
import sys
import zipfile
from contextlib import suppress

_MAX_WHEEL_BYTES = 15 * 1024**2
_MAX_EXPANDED_BYTES = 32 * 1024**2
_MAX_MEMBERS = 4096
_MAX_MEMBER_NAME_BYTES = 4096
_FIXED_WHEEL_TIME = (2020, 2, 2, 0, 0, 0)
_FIXED_WHEEL_MODE = (stat.S_IFREG | 0o644) << 16
_EOCD = b"PK\x05\x06"

_RUNTIME_PURPOSES = {
    "realtime_windows_direct_runtime",
    "realtime_windows_sdist_built_runtime",
    "hermes_v020_pluginmanager_runtime",
}


def _wheel_member_name(name: str) -> tuple[str, ...]:
    try:
        encoded = name.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("wheel member name is unsafe") from None
    parts = name.split("/")
    if (
        not encoded
        or len(encoded) > _MAX_MEMBER_NAME_BYTES
        or name.startswith("/")
        or "\\" in name
        or any(
            not part
            or part in {".", ".."}
            or len(part.encode("ascii")) > 255
            or any(not (character.isalnum() or character in "._-") for character in part)
            for part in parts
        )
    ):
        raise ValueError("wheel member name is unsafe")
    return tuple(parts)


def _complete_zip(raw: bytes) -> tuple[int, int]:
    start = max(0, len(raw) - (65535 + 22))
    end = raw.rfind(_EOCD, start)
    if end < 0 or end + 22 > len(raw):
        raise ValueError("wheel ZIP structure is invalid")
    (
        signature,
        disk,
        central_disk,
        disk_members,
        members,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", raw, end)
    if (
        signature != _EOCD
        or disk != 0
        or central_disk != 0
        or disk_members != members
        or central_offset + central_size != end
        or end + 22 + comment_size != len(raw)
    ):
        raise ValueError("wheel ZIP structure is ambiguous")
    cursor = central_offset
    observed_members = 0
    while cursor < end:
        if cursor + 46 > end or raw[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError("wheel central directory is invalid")
        name_size, extra_size, member_comment_size = struct.unpack_from("<3H", raw, cursor + 28)
        cursor += 46 + name_size + extra_size + member_comment_size
        observed_members += 1
        if cursor > end or observed_members > _MAX_MEMBERS:
            raise ValueError("wheel central directory exceeds its bound")
    if cursor != end or observed_members != members:
        raise ValueError("wheel central directory is ambiguous")
    return central_offset, members


def canonical_wheel_bytes(raw: bytes) -> bytes:
    """Return one bounded, platform-independent container for a wheel payload map."""

    if type(raw) is not bytes:
        raise TypeError("wheel must be bytes")
    if not 0 < len(raw) <= _MAX_WHEEL_BYTES or not raw.startswith(b"PK\x03\x04"):
        raise ValueError("wheel size or structure is invalid")
    central_offset, expected_members = _complete_zip(raw)
    if not 0 < expected_members <= _MAX_MEMBERS:
        raise ValueError("wheel member count is outside its bound")
    members: list[tuple[str, bytes]] = []
    names: set[str] = set()
    folded: set[str] = set()
    parts_by_name: dict[str, tuple[str, ...]] = {}
    expanded = 0
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as source:
            entries = source.infolist()
            if len(entries) != expected_members:
                raise ValueError("wheel member count is outside its bound")
            previous_end = 0
            for entry in entries:
                name_parts = _wheel_member_name(entry.orig_filename)
                name = entry.filename
                folded_name = name.casefold()
                mode = (entry.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if (
                    entry.is_dir()
                    or entry.orig_filename != name
                    or entry.file_size < 0
                    or entry.compress_size < 0
                    or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or entry.flag_bits != 0
                    or kind not in {0, stat.S_IFREG}
                    or entry.external_attr & 0x10
                    or name in names
                    or folded_name in folded
                ):
                    raise ValueError("wheel member is unsupported or ambiguous")
                if entry.header_offset != previous_end or entry.header_offset + 30 > central_offset:
                    raise ValueError("wheel member layout is ambiguous")
                (
                    signature,
                    _version,
                    flags,
                    compression,
                    _time,
                    _date,
                    crc,
                    compressed_size,
                    file_size,
                    name_size,
                    extra_size,
                ) = struct.unpack_from("<4s5H3L2H", raw, entry.header_offset)
                data_start = entry.header_offset + 30 + name_size + extra_size
                previous_end = data_start + compressed_size
                local_name = raw[entry.header_offset + 30 : entry.header_offset + 30 + name_size]
                if (
                    signature != b"PK\x03\x04"
                    or flags != entry.flag_bits
                    or compression != entry.compress_type
                    or crc != entry.CRC
                    or compressed_size != entry.compress_size
                    or file_size != entry.file_size
                    or local_name != entry.orig_filename.encode("ascii")
                    or previous_end > central_offset
                ):
                    raise ValueError("wheel member layout differs")
                expanded += entry.file_size
                if expanded > _MAX_EXPANDED_BYTES:
                    raise ValueError("wheel expansion exceeds its bound")
                names.add(name)
                folded.add(folded_name)
                parts_by_name[folded_name] = tuple(part.casefold() for part in name_parts)
            if previous_end != central_offset:
                raise ValueError("wheel member layout is ambiguous")
            for name_parts in parts_by_name.values():
                for length in range(1, len(name_parts)):
                    if "/".join(name_parts[:length]) in folded:
                        raise ValueError("wheel member namespace is ambiguous")
            for entry in entries:
                with source.open(entry, "r") as stream:
                    payload = stream.read(entry.file_size + 1)
                    if len(payload) != entry.file_size or stream.read(1):
                        raise ValueError("wheel member size differs")
                members.append((entry.filename, payload))
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError):
        raise ValueError("wheel ZIP is unreadable") from None

    output = io.BytesIO()
    try:
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_STORED, allowZip64=False
        ) as target:
            target.comment = b""
            for name, payload in sorted(members):
                entry = zipfile.ZipInfo(name, _FIXED_WHEEL_TIME)
                entry.create_system = 3
                entry.create_version = 20
                entry.extract_version = 20
                entry.compress_type = zipfile.ZIP_STORED
                entry.flag_bits = 0
                entry.internal_attr = 0
                entry.external_attr = _FIXED_WHEEL_MODE
                entry.comment = b""
                entry.extra = b""
                target.writestr(entry, payload, compress_type=zipfile.ZIP_STORED)
    except (ValueError, RuntimeError, OSError, zipfile.LargeZipFile):
        raise ValueError("canonical wheel cannot be written") from None
    canonical = output.getvalue()
    if not 0 < len(canonical) <= _MAX_WHEEL_BYTES:
        raise ValueError("canonical wheel size exceeds its bound")
    return canonical


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


def canonicalize_wheel_file(path: str | os.PathLike[str]) -> None:
    """Replace one ordinary wheel file without reading beyond the wheel bound."""

    name = os.path.abspath(os.fsdecode(path))
    before = os.lstat(name)
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_WHEEL_BYTES:
        raise ValueError("wheel file is not an ordinary bounded file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags)
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(_MAX_WHEEL_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        _file_identity(before) != _file_identity(opened)
        or _file_identity(opened) != _file_identity(after)
        or len(raw) != before.st_size
    ):
        raise ValueError("wheel file changed while it was read")
    canonical = canonical_wheel_bytes(raw)
    temporary = name + ".canonical"
    created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOINHERIT", 0),
            0o600,
        )
        created = True
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                if stream.write(canonical) != len(canonical):
                    raise OSError("canonical wheel write was incomplete")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        if _file_identity(os.lstat(name)) != _file_identity(before):
            raise ValueError("wheel file changed before replacement")
        os.replace(temporary, name)
        created = False
    finally:
        if created:
            with suppress(FileNotFoundError):
                os.unlink(temporary)


def main(arguments: list[str]) -> int:
    if (
        len(arguments) != 5
        or arguments[0] not in {"imports", "wheel", "sdist"} | _RUNTIME_PURPOSES
        or any(not value or len(value) > 4096 or "\x00" in value for value in arguments)
        or not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode)
        or sys.prefix != sys.base_prefix
    ):
        return 2
    try:
        import importlib
        import json
        from pathlib import Path

        kind, package_name, source_name, output_name, report_name = arguments
        packages = Path(package_name).resolve(strict=True)
        source = Path(source_name).resolve(strict=True)
        output = Path(output_name).resolve(strict=True)
        runtime = Path(sys.base_prefix).resolve(strict=True)
        worker = Path(__file__).resolve(strict=True)
        if not all(path.is_dir() for path in (packages, source, output)) or not Path(
            sys.executable
        ).resolve(strict=True).is_relative_to(runtime):
            return 2
        sys.path.insert(0, str(packages))
        names = (
            (
                "hermes_realtime",
                "hermes_realtime.host_launcher",
                "hermes_realtime.launcher",
                "hermes_realtime.hermes_plugin",
            )
            if kind in _RUNTIME_PURPOSES
            else ("hatchling", "packaging", "pathspec", "pluggy", "trove_classifiers")
        )
        for name in names:
            module = importlib.import_module(name)
            origin = module.__file__
            if not isinstance(origin, str) or not Path(origin).resolve(strict=True).is_relative_to(
                packages
            ):
                return 1
        if (
            kind in _RUNTIME_PURPOSES
            and getattr(sys.modules["hermes_realtime"], "__version__", None) != "0.0.3"
        ):
            return 1
        artifact = None
        if kind in {"wheel", "sdist"}:
            build = importlib.import_module("hatchling.build")

            os.chdir(source)
            # The admitted Hatchling 1.27.0 default, made explicit for each build.
            os.environ["SOURCE_DATE_EPOCH"] = "1580601600"
            if kind == "wheel":
                artifact = build.build_wheel(str(output))
                expected = "hermes_realtime-0.0.3-py3-none-any.whl"
            else:
                artifact = build.build_sdist(str(output))
                expected = "hermes_realtime-0.0.3.tar.gz"
            if artifact != expected or not (output / expected).is_file():
                return 1
            if kind == "wheel":
                canonicalize_wheel_file(output / expected)
        origins = []
        for module in list(sys.modules.values()):
            origin = getattr(module, "__file__", None)
            if origin is None:
                continue
            if not isinstance(origin, str):
                return 1
            origins.append(Path(origin).resolve(strict=True))
        if not all(
            path == worker or path.is_relative_to(packages) or path.is_relative_to(runtime)
            for path in origins
        ):
            return 1
        if not all(
            Path(path).resolve().is_relative_to(packages)
            or Path(path).resolve().is_relative_to(runtime)
            for path in sys.path
        ):
            return 1
        observation = {
            "version": 1,
            "kind": kind,
            "pid": os.getpid(),
            "imports": len(names),
            "file_origins": len(origins),
            "source_fallback": False,
            "artifact": artifact,
        }
        with Path(report_name).open("x", encoding="utf-8", newline="\n") as report:
            json.dump(observation, report, sort_keys=True, separators=(",", ":"))
            report.write("\n")
        return 0
    except Exception:
        # Paths, installed module diagnostics and backend exceptions stay private.
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
