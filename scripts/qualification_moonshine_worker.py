"""Read the installed Moonshine native STT catalog without loading model resources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_MODEL_BASE = "https://download.moonshine.ai/model/medium-streaming-en/quantized"
_SPELLING_BASE = "https://download.moonshine.ai/model/spelling-en"


def _strict_json(raw: str) -> object:
    import json

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        if len(dict(items)) != len(items):
            raise ValueError("native catalog contains duplicate fields")
        return dict(items)

    if not raw or len(raw.encode("utf-8")) > 64 * 1024:
        raise ValueError("native catalog is absent or unbounded")
    return json.loads(raw, object_pairs_hook=pairs)


def _hash_file(path: Path, limit: int) -> tuple[str, int]:
    import hashlib

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError("installed Moonshine file exceeds its bound")
            digest.update(chunk)
    if size == 0:
        raise ValueError("installed Moonshine file is empty")
    return digest.hexdigest(), size


def _crc32c(raw: bytes, crc: int = 0) -> int:
    value = crc ^ 0xFFFFFFFF
    for byte in raw:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    return value ^ 0xFFFFFFFF


def _catalog_group(value: object, *, base_url: str, count: int) -> tuple[tuple[str, int, str], ...]:
    import base64
    import re

    if type(value) is not dict or set(value) != {"base_url", "files"}:
        raise ValueError("native catalog group fields differ")
    files = value["files"]
    if value["base_url"] != base_url or type(files) is not list or len(files) != count:
        raise ValueError("native catalog group profile differs")
    rows = []
    for item in files:
        if type(item) is not dict or set(item) != {
            "checksum",
            "checksum_type",
            "name",
            "size",
            "url",
        }:
            raise ValueError("native catalog resource fields differ")
        name, size, checksum = item["name"], item["size"], item["checksum"]
        if (
            type(name) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None
            or type(size) is not int
            or not 0 < size <= 512 * 1024**2
            or type(checksum) is not str
            or item["checksum_type"] != "crc32c"
            or item["url"] != f"{base_url}/{name}"
        ):
            raise ValueError("native catalog resource identity differs")
        try:
            decoded = base64.b64decode(checksum, validate=True)
        except ValueError:
            raise ValueError("native catalog checksum is invalid") from None
        if len(decoded) != 4 or base64.b64encode(decoded).decode("ascii") != checksum:
            raise ValueError("native catalog checksum is noncanonical")
        rows.append((item["url"], size, checksum))
    if tuple(row[0] for row in rows) != tuple(sorted(row[0] for row in rows)):
        raise ValueError("native catalog resource order differs")
    if len({row[0].casefold() for row in rows}) != len(rows):
        raise ValueError("native catalog resource aliases another")
    return tuple(rows)


def _selected_resources(
    without_spelling: object, with_spelling: object
) -> tuple[tuple[str, int, str], ...]:
    if (
        type(without_spelling) is not dict
        or set(without_spelling) != {"groups"}
        or type(without_spelling["groups"]) is not list
        or len(without_spelling["groups"]) != 1
        or type(with_spelling) is not dict
        or set(with_spelling) != {"groups"}
        or type(with_spelling["groups"]) is not list
        or len(with_spelling["groups"]) != 2
        or with_spelling["groups"][0] != without_spelling["groups"][0]
    ):
        raise ValueError("native catalog spelling profile differs")
    resources = _catalog_group(
        without_spelling["groups"][0], base_url=_MODEL_BASE, count=7
    ) + _catalog_group(with_spelling["groups"][1], base_url=_SPELLING_BASE, count=2)
    if sum(row[1] for row in resources) > 1024**3:
        raise ValueError("native catalog resources exceed their bound")
    return resources


def _download_resource(
    opener: object, url: str, target: Path, expected_size: int, expected_crc32c: str
) -> str:
    import base64
    import hashlib
    import urllib.request
    from typing import Any, cast

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "hermes-realtime-qualification/1",
        },
    )
    response = cast(Any, opener).open(request, timeout=60)
    digest = hashlib.sha256()
    crc = 0
    size = 0
    with response:
        length = response.headers.get("Content-Length")
        encoding = response.headers.get("Content-Encoding")
        if (
            response.status != 200
            or response.geturl() != url
            or encoding not in {None, "identity"}
            or (length is not None and (not length.isdecimal() or int(length) != expected_size))
        ):
            raise ValueError("publisher response identity differs")
        with target.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > expected_size:
                    raise ValueError("publisher resource exceeds its catalog size")
                output.write(chunk)
                digest.update(chunk)
                crc = _crc32c(chunk, crc)
    encoded = base64.b64encode(crc.to_bytes(4, "big")).decode("ascii")
    if size != expected_size or encoded != expected_crc32c:
        raise ValueError("publisher resource differs from its native catalog")
    return digest.hexdigest()


def _native_catalog(library: object, *, include_spelling: bool) -> object:
    import ctypes
    from typing import Any, cast

    class Option(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char_p), ("value", ctypes.c_char_p)]

    native = cast(Any, library)
    query = native.moonshine_get_stt_dependencies
    release = native.moonshine_free_buffer
    query.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(Option),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    query.restype = ctypes.c_int32
    release.argtypes = [ctypes.c_void_p]
    release.restype = None
    pairs = [(b"model_arch", b"5")]
    if include_spelling:
        pairs.append((b"include_spelling", b"true"))
    options = (Option * len(pairs))(*(Option(*pair) for pair in pairs))
    pointer = ctypes.c_void_p()
    if query(b"en", options, len(pairs), ctypes.byref(pointer)) != 0 or not pointer.value:
        raise ValueError("native catalog query failed")
    try:
        raw = ctypes.string_at(pointer.value)
        if not raw or len(raw) > 64 * 1024:
            raise ValueError("native catalog is absent or unbounded")
        return _strict_json(raw.decode("utf-8"))
    finally:
        release(pointer)


def main(arguments: list[str]) -> int:
    acquiring = len(arguments) == 4 and arguments[3] == "acquire"
    resources: Path | None = None
    stage = "arguments"
    if (
        (len(arguments) != 3 and not acquiring)
        or any(not value or len(value) > 4096 or "\x00" in value for value in arguments)
        or not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode)
        or sys.prefix != sys.base_prefix
    ):
        return 2
    try:
        import ctypes
        import importlib.metadata
        import os
        stage = "namespace"
        packages = Path(arguments[0]).resolve(strict=True)
        workspace = Path(arguments[1]).resolve(strict=True)
        output = Path(arguments[2])
        runtime = Path(sys.base_prefix).resolve(strict=True)
        worker = Path(__file__).resolve(strict=True)
        if acquiring:
            resources = output.resolve(strict=True)
            report = resources / "moonshine-preparation.json"
        else:
            resources = None
            report = output
        if (
            not packages.is_dir()
            or not workspace.is_dir()
            or report.exists()
            or not Path(sys.executable).resolve(strict=True).is_relative_to(runtime)
        ):
            return 2
        if acquiring:
            assert resources is not None
            if (
                not resources.is_dir()
                or any(resources.iterdir())
                or resources in (workspace, packages)
            ):
                return 2
        elif report != workspace / "moonshine-catalog.json":
            return 2
        initial_paths = tuple(
            Path(value).resolve()
            for value in sys.path
            if isinstance(value, str) and value
        )
        if not initial_paths or not all(path.is_relative_to(runtime) for path in initial_paths):
            return 1
        package = packages / "moonshine_voice"
        api_file = package / "moonshine_api.py"
        native_file = package / "moonshine.dll"
        distributions = tuple(packages.glob("moonshine_voice-*.dist-info"))
        if len(distributions) != 1:
            return 1
        distribution = importlib.metadata.Distribution.at(distributions[0])
        if (
            distribution.metadata["Name"] != "moonshine-voice"
            or distribution.version != "0.1.0"
            or not api_file.is_file()
            or not native_file.is_file()
        ):
            return 1

        native_sha256, native_bytes = _hash_file(native_file, 64 * 1024**2)
        api_sha256, api_bytes = _hash_file(api_file, 512 * 1024)
        stage = "native-library"
        with os.add_dll_directory(str(package)):
            library = ctypes.CDLL(str(native_file))
        library_name = getattr(library, "_name", None)
        if (
            not isinstance(library_name, str)
            or Path(library_name).resolve(strict=True) != native_file
        ):
            return 1
        stage = "native-catalog"
        without_spelling = _native_catalog(library, include_spelling=False)
        with_spelling = _native_catalog(library, include_spelling=True)

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
            Path(value).resolve().is_relative_to(packages)
            or Path(value).resolve().is_relative_to(runtime)
            for value in sys.path
            if isinstance(value, str) and value
        ):
            return 1

        observation: dict[str, object] = {
            "version": 1,
            "pid": os.getpid(),
            "language": "en",
            "model_arch": 5,
            "native_library_sha256": native_sha256,
            "native_library_bytes": native_bytes,
            "python_api_sha256": api_sha256,
            "python_api_bytes": api_bytes,
            "file_origins": len(origins),
            "source_fallback": False,
            "without_spelling": without_spelling,
            "with_spelling": with_spelling,
        }
        result: object = observation
        if acquiring:
            import ssl
            import urllib.request

            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *args: object, **kwargs: object) -> None:
                    raise ValueError("publisher redirect is unavailable")

            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler(),
                urllib.request.HTTPSHandler(context=ssl.create_default_context()),
                NoRedirect(),
            )
            rows = []
            assert resources is not None
            for url, size, crc32c in _selected_resources(without_spelling, with_spelling):
                stage = "download/" + url.rsplit("/", 1)[-1]
                cache_path = url.removeprefix("https://")
                target = resources.joinpath(*cache_path.split("/"))
                sha256 = _download_resource(opener, url, target, size, crc32c)
                rows.append(
                    {
                        "cache_path": cache_path,
                        "url": url,
                        "size": size,
                        "crc32c": crc32c,
                        "sha256": sha256,
                    }
                )
            result = {"catalog": observation, "resources": rows}
        stage = "report"
        with report.open("x", encoding="utf-8", newline="\n") as destination:
            json.dump(
                result,
                destination,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            destination.write("\n")
        return 0
    except Exception as error:
        # Keep only a bounded public stage/type diagnostic; paths and payloads remain private.
        if acquiring and resources is not None and resources.is_dir():
            try:
                with (resources / "moonshine-preparation-error.json").open(
                    "x", encoding="utf-8", newline="\n"
                ) as diagnostic:
                    json.dump(
                        {"error": type(error).__name__, "stage": stage},
                        diagnostic,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    diagnostic.write("\n")
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
