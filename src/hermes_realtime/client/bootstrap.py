"""Server-side issuance of narrowly scoped browser LiveKit credentials."""

from __future__ import annotations

import hmac
import math
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

from livekit import api

from hermes_realtime.livekit import LiveKitConnection

_ROOM_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_IDENTITY_PATTERN = re.compile(r"browser_[a-f0-9]{16,64}\Z")


@dataclass(frozen=True, slots=True)
class BrowserJoinCredential:
    """Browser-visible short-lived room credential without server secrets."""

    url: str
    room_name: str
    participant_identity: str
    expires_in_seconds: int
    token: str = field(repr=False)


class OneTimeBootstrapCapability:
    """One expiring bearer capability for establishing a browser session."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 120,
        token_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(ttl_seconds) is not int:
            raise TypeError("ttl_seconds must be an exact integer")
        if ttl_seconds < 30 or ttl_seconds > 3_600:
            raise ValueError("ttl_seconds must be between 30 and 3600")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if token_factory is not None and not callable(token_factory):
            raise TypeError("token_factory must be callable")
        token = (token_factory or self._new_token)()
        if type(token) is not str:
            raise TypeError("generated capability must be an exact built-in string")
        if re.fullmatch(r"[A-Za-z0-9_-]{43,128}", token) is None:
            raise ValueError("generated capability must contain 43 to 128 URL-safe characters")
        now = clock()
        if type(now) not in (int, float):
            raise TypeError("clock must return an exact number")
        if not math.isfinite(now):
            raise ValueError("clock must return a finite number")

        self._token = token
        self._expires_at = float(now) + ttl_seconds
        self._clock = clock
        self._consumed = False
        self._claim_lock = threading.Lock()

    @staticmethod
    def _new_token() -> str:
        return secrets.token_urlsafe(32)

    @property
    def token(self) -> str:
        """Return the capability for placement in a URL fragment, never a query."""

        return self._token

    def consume(self, presented: str) -> None:
        """Consume an exact, unexpired capability once."""

        if type(presented) is not str:
            raise TypeError("presented capability must be an exact built-in string")
        now = self._clock()
        if type(now) not in (int, float):
            raise TypeError("clock must return an exact number")
        if not math.isfinite(now):
            raise ValueError("clock must return a finite number")
        with self._claim_lock:
            if self._consumed:
                raise RuntimeError("bootstrap capability has already been consumed")
            if now >= self._expires_at:
                raise RuntimeError("bootstrap capability has expired")
            if len(presented) > 128 or not hmac.compare_digest(self._token, presented):
                raise PermissionError("bootstrap capability is invalid")
            self._consumed = True


class BrowserTokenIssuer:
    """Issue microphone-only credentials for one configured room."""

    def __init__(
        self,
        *,
        connection: LiveKitConnection,
        room_name: str,
        ttl_seconds: int = 60,
        identity_factory: Callable[[], str] | None = None,
    ) -> None:
        if type(connection) is not LiveKitConnection:
            raise TypeError("connection must be an exact LiveKitConnection")
        if type(room_name) is not str:
            raise TypeError("room_name must be an exact built-in string")
        if _ROOM_PATTERN.fullmatch(room_name) is None:
            raise ValueError("room_name must contain 1 to 128 safe characters")
        if type(ttl_seconds) is not int:
            raise TypeError("ttl_seconds must be an exact integer")
        if ttl_seconds < 30 or ttl_seconds > 300:
            raise ValueError("ttl_seconds must be between 30 and 300")
        if identity_factory is not None and not callable(identity_factory):
            raise TypeError("identity_factory must be callable")

        self._connection = connection
        self._room_name = room_name
        self._ttl_seconds = ttl_seconds
        self._identity_factory = identity_factory or self._new_identity

    @staticmethod
    def _new_identity() -> str:
        return f"browser_{secrets.token_hex(16)}"

    def issue(self) -> BrowserJoinCredential:
        """Create one independently generated, short-lived room credential."""

        identity = self._identity_factory()
        return self.issue_for_identity(identity)

    def issue_for_identity(self, identity: str) -> BrowserJoinCredential:
        """Refresh a credential for one already-authorized browser identity."""

        if type(identity) is not str:
            raise TypeError("identity must be an exact built-in string")
        if _IDENTITY_PATTERN.fullmatch(identity) is None:
            raise ValueError("identity is not a valid browser identity")
        return BrowserJoinCredential(
            url=self._connection.url,
            room_name=self._room_name,
            participant_identity=identity,
            expires_in_seconds=self._ttl_seconds,
            token=self._connection.token(
                identity=identity,
                room_name=self._room_name,
                ttl_seconds=self._ttl_seconds,
            ),
        )


class BrowserTokenVerifier:
    """Verify browser bearer tokens against one exact room and grant set."""

    def __init__(self, *, connection: LiveKitConnection, room_name: str) -> None:
        if type(connection) is not LiveKitConnection:
            raise TypeError("connection must be an exact LiveKitConnection")
        if type(room_name) is not str:
            raise TypeError("room_name must be an exact built-in string")
        if _ROOM_PATTERN.fullmatch(room_name) is None:
            raise ValueError("room_name must contain 1 to 128 safe characters")
        self._room_name = room_name
        self._verifier = api.TokenVerifier(
            connection.api_key,
            connection.api_secret,
            leeway=timedelta(0),
        )

    def verify(self, token: str) -> str:
        """Return the exact browser identity after signature, expiry, and grant checks."""

        if type(token) is not str:
            raise TypeError("token must be an exact built-in string")
        if not 32 <= len(token) <= 8192 or token.count(".") != 2:
            raise PermissionError("browser token is invalid")
        try:
            claims = self._verifier.verify(token)
            identity = claims.identity
            grants = claims.video
            valid = (
                type(identity) is str
                and _IDENTITY_PATTERN.fullmatch(identity) is not None
                and grants is not None
                and type(grants.room_join) is bool
                and grants.room_join
                and type(grants.room) is str
                and grants.room == self._room_name
                and type(grants.can_publish) is bool
                and grants.can_publish
                and type(grants.can_subscribe) is bool
                and grants.can_subscribe
                and type(grants.can_publish_data) is bool
                and not grants.can_publish_data
                and type(grants.can_publish_sources) is list
                and grants.can_publish_sources == ["microphone"]
                and type(grants.can_update_own_metadata) is bool
                and not grants.can_update_own_metadata
            )
        except Exception:
            raise PermissionError("browser token is invalid") from None
        if not valid:
            raise PermissionError("browser token is invalid")
        return cast(str, identity)
