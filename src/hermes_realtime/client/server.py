"""Bounded HTTP/1.1 transport and allowlisted static serving for the browser client."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import re
import ssl
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from .http import BrowserBootstrapApplication, BrowserBootstrapResponse
from .loopback import LoopbackPeerAddress
from .tailnet import TailnetPeerAddress

BrowserPeerAddress = LoopbackPeerAddress | TailnetPeerAddress

_MAX_HEADER_BYTES = 16_384
_MAX_BODY_BYTES = 8_192
_MAX_STATIC_BYTES = 1_048_576
_HEADER_TIMEOUT_SECONDS = 5.0
_BODY_TIMEOUT_SECONDS = 5.0
_STATIC_PATHS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.js": ("assets/app.js", "text/javascript; charset=utf-8"),
    "/assets/styles.css": ("assets/styles.css", "text/css; charset=utf-8"),
}
_DNS_HOSTNAME = re.compile(
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\."
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\Z"
)
_ENDS_IN_NUMBER = re.compile(r"(?:[0-9]+|0[xX][0-9A-Fa-f]*)\Z")


def _livekit_http_origin(value: object) -> str:
    error = "LiveKit URL must be an exact WebSocket origin"
    if type(value) is not str:
        raise TypeError(error)
    try:
        if (
            any(ord(character) < 33 or ord(character) > 126 for character in value)
            or "\\" in value
            or "?" in value
            or "#" in value
        ):
            raise ValueError
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError(error) from None
    if (
        parsed.scheme not in {"ws", "wss"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.netloc.endswith(":")
        or "%" in hostname
    ):
        raise ValueError(error)
    try:
        canonical_address = ipaddress.ip_address(hostname).compressed
    except ValueError:
        if (
            _DNS_HOSTNAME.fullmatch(hostname) is None
            or _ENDS_IN_NUMBER.fullmatch(hostname.rsplit(".", 1)[-1]) is not None
        ):
            raise ValueError(error) from None
        canonical_host = hostname.lower()
    else:
        canonical_host = f"[{canonical_address}]" if ":" in canonical_address else canonical_address
    scheme = "http" if parsed.scheme == "ws" else "https"
    port_suffix = "" if port in {None, 80 if scheme == "http" else 443} else f":{port}"
    return f"{scheme}://{canonical_host}{port_suffix}"


def _content_security_policy(livekit_url: object) -> str:
    diagnostic_origin = _livekit_http_origin(livekit_url)
    return (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        f"connect-src 'self' ws: wss: {diagnostic_origin}; "
        "media-src 'self' blob:; img-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )


class BrowserHttpServer:
    """One-request-per-connection HTTP server with strict, bounded parsing."""

    def __init__(
        self,
        *,
        application: BrowserBootstrapApplication,
        static_root: Path,
        livekit_url: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        lan_mode: bool = False,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if type(application) is not BrowserBootstrapApplication:
            raise TypeError("application must be an exact BrowserBootstrapApplication")
        if not isinstance(static_root, Path):
            raise TypeError("static_root must be a pathlib Path")
        if type(host) is not str:
            raise TypeError("host must be an exact built-in string")
        if type(port) is not int:
            raise TypeError("port must be an exact integer")
        if type(lan_mode) is not bool:
            raise TypeError("lan_mode must be an exact boolean")
        if not 0 <= port <= 65_535:
            raise ValueError("port is outside the supported range")
        if not lan_mode and host not in {"127.0.0.1", "::1"}:
            raise ValueError("non-LAN mode binds only literal loopback")
        if lan_mode and ssl_context is None:
            raise ValueError("LAN mode requires a configured TLS context")
        if ssl_context is not None and type(ssl_context) is not ssl.SSLContext:
            raise TypeError("ssl_context must be an exact SSLContext")

        root = static_root.resolve(strict=True)
        csp = _content_security_policy(livekit_url)
        static: dict[str, tuple[bytes, str]] = {}
        for request_path, (relative_path, content_type) in _STATIC_PATHS.items():
            path = root / relative_path
            if not path.is_file():
                if request_path == "/":
                    raise ValueError("static_root must contain index.html")
                continue
            payload = path.read_bytes()
            if not payload or len(payload) > _MAX_STATIC_BYTES:
                raise ValueError(f"static asset is empty or oversized: {relative_path}")
            static[request_path] = (payload, content_type)

        self._application = application
        self._static = MappingProxyType(static)
        self._csp = csp
        self._host = host
        self._configured_port = port
        self._ssl_context = ssl_context
        self._server: asyncio.Server | None = None

    @property
    def port(self) -> int:
        server = self._server
        if server is None or not server.sockets:
            raise RuntimeError("browser HTTP server is not started")
        return int(server.sockets[0].getsockname()[1])

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("browser HTTP server is already started")
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._host,
            port=self._configured_port,
            ssl=self._ssl_context,
            limit=_MAX_HEADER_BYTES + 1,
        )

    async def close(self) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        server.close()
        await server.wait_closed()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peername = writer.get_extra_info("peername")
        try:
            peer: BrowserPeerAddress | None = TailnetPeerAddress.from_peername(peername)
        except Exception:
            try:
                peer = LoopbackPeerAddress.from_peername(peername)
            except Exception:
                peer = None
        try:
            response = await self._read_and_dispatch(reader, peer=peer)
        except PermissionError:
            response = self._error_response(403)
        except (TypeError, ValueError, UnicodeError, asyncio.IncompleteReadError):
            response = self._error_response(400)
        except (TimeoutError, asyncio.LimitOverrunError):
            response = self._error_response(408)
        except RuntimeError:
            response = self._error_response(409)
        except Exception:
            response = self._error_response(503)
        try:
            writer.write(self._encode_response(response))
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                async with asyncio.timeout(1.0):
                    await writer.wait_closed()

    async def _read_and_dispatch(
        self,
        reader: asyncio.StreamReader,
        *,
        peer: BrowserPeerAddress | None,
    ) -> BrowserBootstrapResponse:
        async with asyncio.timeout(_HEADER_TIMEOUT_SECONDS):
            raw_head = await reader.readuntil(b"\r\n\r\n")
        if len(raw_head) > _MAX_HEADER_BYTES:
            raise ValueError("request headers exceed the supported bound")
        try:
            lines = raw_head[:-4].decode("ascii").split("\r\n")
        except UnicodeDecodeError:
            raise ValueError("request headers must be ASCII") from None
        request_line = lines[0].split(" ")
        if len(request_line) != 3 or request_line[2] != "HTTP/1.1":
            raise ValueError("request line is invalid")
        method, target, _ = request_line
        if not target.startswith("/") or "?" in target or "#" in target:
            raise ValueError("request target is invalid")

        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line or line[0] in " \t" or ":" not in line:
                raise ValueError("request header is invalid")
            raw_name, raw_value = line.split(":", 1)
            name = raw_name.lower()
            value = raw_value.strip(" \t")
            if (
                not name
                or not name.replace("-", "a").isalnum()
                or name in headers
                or "\x00" in value
                or "\r" in value
                or "\n" in value
            ):
                raise ValueError("request header is invalid")
            headers[name] = value
        if "transfer-encoding" in headers:
            raise ValueError("transfer encoding is not supported")

        raw_length = headers.get("content-length")
        if raw_length is None:
            content_length = 0
            headers["content-length"] = "0"
        elif not raw_length.isdigit():
            raise ValueError("content-length is invalid")
        else:
            content_length = int(raw_length)
        if content_length > _MAX_BODY_BYTES:
            raise ValueError("request body exceeds the supported bound")
        async with asyncio.timeout(_BODY_TIMEOUT_SECONDS):
            body = await reader.readexactly(content_length)

        if method == "GET":
            if body or target not in self._static:
                raise PermissionError("static path is not available")
            payload, content_type = self._static[target]
            headers_out = {
                "cache-control": "no-store",
                "content-security-policy": self._csp,
                "content-type": content_type,
                "referrer-policy": "no-referrer",
                "x-content-type-options": "nosniff",
                "x-frame-options": "DENY",
            }
            return BrowserBootstrapResponse(200, headers_out, payload)
        return await self._application.handle(
            method=method,
            path=target,
            headers=headers,
            body=body,
            peer=peer,
        )

    @staticmethod
    def _error_response(status: int) -> BrowserBootstrapResponse:
        return BrowserBootstrapResponse(
            status=status,
            headers={
                "cache-control": "no-store",
                "content-type": "application/json; charset=utf-8",
                "referrer-policy": "no-referrer",
                "x-content-type-options": "nosniff",
            },
            body=b'{"error":"request_rejected","version":1}',
        )

    @staticmethod
    def _encode_response(response: BrowserBootstrapResponse) -> bytes:
        reasons = {
            200: "OK",
            202: "Accepted",
            400: "Bad Request",
            403: "Forbidden",
            408: "Request Timeout",
            409: "Conflict",
            503: "Service Unavailable",
        }
        reason = reasons.get(response.status, "Error")
        headers = dict(response.headers)
        headers["connection"] = "close"
        headers["content-length"] = str(len(response.body))
        head = [f"HTTP/1.1 {response.status} {reason}\r\n"]
        for name in sorted(headers):
            head.append(f"{name}: {headers[name]}\r\n")
        head.append("\r\n")
        return "".join(head).encode("ascii") + response.body
