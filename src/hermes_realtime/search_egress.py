"""Binding-scoped authority for transcript-derived public search egress."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from threading import Lock

SEARCH_EGRESS_CONSENT_VERSION = "realtime-search-egress-consent-v1"
SEARCH_EGRESS_DISCLOSURE_TEXT = (
    "Public search is optional. It is unavailable unless the host operator enables it, and it "
    "remains off for each browser session until you consent here.\n\n"
    "If you consent, Hermes Realtime may use a search query derived from stable partial or final "
    "speech transcript text, or final typed input text, and send that query over HTTPS to Bing "
    "Search RSS and, for outcome-shaped queries, Google News RSS. Raw microphone audio "
    "is not sent to those services by this feature.\n\n"
    "Search results are untrusted external data used only to support source-sensitive answers. "
    "The destinations may receive the query, network metadata such as your host's public IP "
    "address, and ordinary HTTP metadata under their own terms and privacy policies.\n\n"
    "Consent applies only to the active browser binding. Rebinding, projection resynchronization, "
    "stopping, inactivity expiry, or revocation closes the server gate for new lookups. Revocation "
    "cannot cancel a lookup admitted while consent was active or recall a request that was already "
    "sent.\n"
)
SEARCH_EGRESS_DISCLOSURE_BYTES = SEARCH_EGRESS_DISCLOSURE_TEXT.encode("utf-8")
SEARCH_EGRESS_DISCLOSURE_DIGEST = hashlib.sha256(SEARCH_EGRESS_DISCLOSURE_BYTES).hexdigest()
_IDENTITY = re.compile(r"browser_[0-9a-f]{16,64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SAFE_INTEGER = (1 << 53) - 1


@dataclass(frozen=True, slots=True)
class SearchEgressBindingV1:
    """One browser identity and generation allowed to control search egress."""

    participant_identity: str
    binding_generation: int

    def __post_init__(self) -> None:
        if (
            type(self.participant_identity) is not str
            or _IDENTITY.fullmatch(self.participant_identity) is None
        ):
            raise ValueError("participant_identity must be a canonical browser identity")
        if (
            type(self.binding_generation) is not int
            or not 1 <= self.binding_generation <= _MAX_SAFE_INTEGER
        ):
            raise ValueError("binding_generation is outside the browser-safe range")


@dataclass(frozen=True, slots=True)
class SearchEgressConsentRequestV1:
    """One exact opt-in request for the active browser binding."""

    sequence: int
    accepted: bool
    consent_version: str
    disclosure_digest: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or not 1 <= self.sequence <= _MAX_SAFE_INTEGER:
            raise ValueError("sequence is outside the browser-safe range")
        if type(self.accepted) is not bool or self.accepted is not True:
            raise ValueError("accepted must be exactly true")
        if self.consent_version != SEARCH_EGRESS_CONSENT_VERSION:
            raise ValueError("search egress consent version is invalid")
        if (
            type(self.disclosure_digest) is not str
            or _DIGEST.fullmatch(self.disclosure_digest) is None
            or self.disclosure_digest != SEARCH_EGRESS_DISCLOSURE_DIGEST
        ):
            raise ValueError("search egress disclosure digest is invalid")


@dataclass(frozen=True, slots=True)
class SearchEgressRevokeRequestV1:
    """One exact revoke request for the active browser binding."""

    sequence: int

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or not 1 <= self.sequence <= _MAX_SAFE_INTEGER:
            raise ValueError("sequence is outside the browser-safe range")


@dataclass(frozen=True, slots=True)
class SearchEgressStatusV1:
    """Content-free public status for the current search egress authority."""

    available: bool
    state: str
    consent_version: str = SEARCH_EGRESS_CONSENT_VERSION
    disclosure_digest: str = SEARCH_EGRESS_DISCLOSURE_DIGEST

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("available must be an exact boolean")
        if self.state not in {"unavailable", "idle", "active"}:
            raise ValueError("search egress state is invalid")
        if self.available != (self.state != "unavailable"):
            raise ValueError("search egress availability contradicts its state")
        if self.consent_version != SEARCH_EGRESS_CONSENT_VERSION:
            raise ValueError("search egress status consent version is invalid")
        if self.disclosure_digest != SEARCH_EGRESS_DISCLOSURE_DIGEST:
            raise ValueError("search egress status disclosure digest is invalid")


class SearchEgressAdmission:
    """One admitted lookup that remains valid until explicitly released."""

    def __init__(self, *, authority: SearchEgressAuthority, token: int) -> None:
        self._authority = authority
        self._token = token
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._authority._release_admission(self._token)


class SearchEgressAuthority:
    """Own binding-scoped, default-closed public search permission."""

    def __init__(self, *, operator_enabled: bool = False) -> None:
        if type(operator_enabled) is not bool:
            raise TypeError("operator_enabled must be an exact boolean")
        self._operator_enabled = operator_enabled
        self._lock = Lock()
        self._binding: SearchEgressBindingV1 | None = None
        self._last_invalidated_binding: SearchEgressBindingV1 | None = None
        self._consented = False
        self._last_sequence = 0
        self._last_control: (
            tuple[str, SearchEgressConsentRequestV1 | SearchEgressRevokeRequestV1] | None
        ) = None
        self._next_admission_token = 0
        self._open_admissions: set[int] = set()

    def bind(self, binding: SearchEgressBindingV1) -> None:
        if type(binding) is not SearchEgressBindingV1:
            raise TypeError("binding must be an exact SearchEgressBindingV1")
        with self._lock:
            self._binding = binding if self._operator_enabled else None
            self._last_invalidated_binding = None
            self._consented = False
            self._last_sequence = 0
            self._last_control = None

    def consent(
        self,
        binding: SearchEgressBindingV1,
        request: SearchEgressConsentRequestV1,
    ) -> SearchEgressStatusV1:
        if type(binding) is not SearchEgressBindingV1:
            raise TypeError("binding must be an exact SearchEgressBindingV1")
        if type(request) is not SearchEgressConsentRequestV1:
            raise TypeError("request must be an exact SearchEgressConsentRequestV1")
        with self._lock:
            if not self._operator_enabled:
                raise RuntimeError("public search egress is disabled by the operator")
            if binding != self._binding:
                raise PermissionError("participant does not own the active search egress binding")
            control = ("consent", request)
            if request.sequence == self._last_sequence and control == self._last_control:
                return self._status_locked()
            if request.sequence != self._last_sequence + 1:
                raise RuntimeError("search egress control sequence conflicts with current binding")
            self._consented = True
            self._last_sequence = request.sequence
            self._last_control = control
            return self._status_locked()

    def revoke(
        self,
        binding: SearchEgressBindingV1,
        request: SearchEgressRevokeRequestV1,
    ) -> SearchEgressStatusV1:
        if type(binding) is not SearchEgressBindingV1:
            raise TypeError("binding must be an exact SearchEgressBindingV1")
        if type(request) is not SearchEgressRevokeRequestV1:
            raise TypeError("request must be an exact SearchEgressRevokeRequestV1")
        with self._lock:
            if not self._operator_enabled:
                raise RuntimeError("public search egress is disabled by the operator")
            if binding != self._binding:
                raise PermissionError("participant does not own the active search egress binding")
            control = ("revoke", request)
            if request.sequence == self._last_sequence and control == self._last_control:
                return self._status_locked()
            if request.sequence != self._last_sequence + 1:
                raise RuntimeError("search egress control sequence conflicts with current binding")
            self._consented = False
            self._last_sequence = request.sequence
            self._last_control = control
            return self._status_locked()

    def invalidate(self, binding: SearchEgressBindingV1) -> None:
        if type(binding) is not SearchEgressBindingV1:
            raise TypeError("binding must be an exact SearchEgressBindingV1")
        with self._lock:
            if binding == self._binding:
                self._binding = None
                self._last_invalidated_binding = binding
            elif self._binding is None and binding == self._last_invalidated_binding:
                return
            else:
                raise PermissionError("participant does not own the active search egress binding")
            self._consented = False
            self._last_sequence = 0
            self._last_control = None

    def status(self) -> SearchEgressStatusV1:
        with self._lock:
            return self._status_locked()

    def admit(self) -> SearchEgressAdmission | None:
        """Reserve one lookup under the currently active consented binding."""

        with self._lock:
            if not self._permits_egress_locked():
                return None
            self._next_admission_token += 1
            token = self._next_admission_token
            self._open_admissions.add(token)
            return SearchEgressAdmission(authority=self, token=token)

    def permits_egress(self) -> bool:
        """Return advisory state only; callers must use admit() for authority."""

        with self._lock:
            return self._permits_egress_locked()

    def _permits_egress_locked(self) -> bool:
        return bool(
            self._operator_enabled
            and self._binding is not None
            and self._consented
        )

    def _release_admission(self, token: int) -> None:
        with self._lock:
            if token not in self._open_admissions:
                raise RuntimeError("search egress admission is not open")
            self._open_admissions.remove(token)

    def _status_locked(self) -> SearchEgressStatusV1:
        available = self._operator_enabled and self._binding is not None
        if not available:  # noqa: SIM108 - explicit authority states are easier to audit
            state = "unavailable"
        else:
            state = "active" if self._consented else "idle"
        return SearchEgressStatusV1(
            available=available,
            state=state,
        )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_exact_object(raw: bytes, *, maximum_bytes: int) -> dict[str, object]:
    if type(raw) is not bytes:
        raise TypeError("request body must be exact bytes")
    if not 1 <= len(raw) <= maximum_bytes:
        raise ValueError("request body size is invalid")
    try:
        decoded = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_json_object)
    except UnicodeDecodeError:
        raise ValueError("request body must be strict UTF-8 JSON") from None
    except json.JSONDecodeError:
        raise ValueError("request body must be strict JSON") from None
    if type(decoded) is not dict:
        raise ValueError("request body must be a JSON object")
    return decoded


def parse_search_egress_consent_request(raw: bytes) -> SearchEgressConsentRequestV1:
    """Parse one exact versioned browser search-egress consent request."""

    value = _parse_exact_object(raw, maximum_bytes=384)
    expected = {"accepted", "consentVersion", "disclosureDigest", "sequence"}
    if set(value) != expected:
        raise ValueError("search egress consent must contain exact fields")
    return SearchEgressConsentRequestV1(
        sequence=value["sequence"],  # type: ignore[arg-type]
        accepted=value["accepted"],  # type: ignore[arg-type]
        consent_version=value["consentVersion"],  # type: ignore[arg-type]
        disclosure_digest=value["disclosureDigest"],  # type: ignore[arg-type]
    )


def parse_search_egress_revoke_request(raw: bytes) -> SearchEgressRevokeRequestV1:
    """Parse one exact browser search-egress revoke request."""

    value = _parse_exact_object(raw, maximum_bytes=64)
    if set(value) != {"sequence"}:
        raise ValueError("search egress revoke must contain exact fields")
    return SearchEgressRevokeRequestV1(sequence=value["sequence"])  # type: ignore[arg-type]


def search_egress_status_to_primitive(status: SearchEgressStatusV1) -> dict[str, object]:
    """Project content-free search-egress state for the browser."""

    if type(status) is not SearchEgressStatusV1:
        raise TypeError("status must be an exact SearchEgressStatusV1")
    return {
        "available": status.available,
        "consentVersion": status.consent_version,
        "disclosureDigest": status.disclosure_digest,
        "searchEgressState": status.state,
    }
