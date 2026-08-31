"""Strict transport-neutral browser bootstrap HTTP application."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from hermes_realtime.evidence.models import (
    parse_evidence_consent_request,
    parse_evidence_revoke_request,
)
from hermes_realtime.search_egress import (
    parse_search_egress_consent_request,
    parse_search_egress_revoke_request,
    search_egress_status_to_primitive,
)

from .bootstrap import BrowserJoinCredential, BrowserTokenVerifier, OneTimeBootstrapCapability
from .loopback import LoopbackPeerAddress, LoopbackPeerAuthorizer
from .session import BrowserAudioDiagnostic, BrowserSessionDirector
from .tailnet import TailnetPeerAddress, TailnetPeerAuthorizer

StablePeerAddress = LoopbackPeerAddress | TailnetPeerAddress
StablePeerAuthorizer = LoopbackPeerAuthorizer | TailnetPeerAuthorizer


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if type(key) is not str:
            raise ValueError("JSON object key must be an exact string")
        if key in value:
            raise ValueError("typed input contains a duplicate field")
        value[key] = item
    return value


@dataclass(frozen=True, slots=True)
class BrowserBootstrapResponse:
    """Transport-neutral HTTP response from the bootstrap boundary."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class BrowserBootstrapApplication:
    """Strict same-origin exchange after identity-bound worker provisioning."""

    _WORKER_IDENTITY = re.compile(r"worker_[A-Za-z0-9_-]{8,64}\Z")
    _RESPONSE_HEADERS = {
        "cache-control": "no-store",
        "content-type": "application/json; charset=utf-8",
        "referrer-policy": "no-referrer",
        "x-content-type-options": "nosniff",
    }

    def __init__(
        self,
        *,
        sessions: BrowserSessionDirector,
        verifier: BrowserTokenVerifier,
        capability: OneTimeBootstrapCapability | None,
        allowed_origin: str,
        worker_identity: str,
        tailnet_authorizer: TailnetPeerAuthorizer | None = None,
        loopback_authorizer: LoopbackPeerAuthorizer | None = None,
    ) -> None:
        if type(sessions) is not BrowserSessionDirector:
            raise TypeError("sessions must be an exact BrowserSessionDirector")
        if type(verifier) is not BrowserTokenVerifier:
            raise TypeError("verifier must be an exact BrowserTokenVerifier")
        if capability is not None and type(capability) is not OneTimeBootstrapCapability:
            raise TypeError("capability must be an exact OneTimeBootstrapCapability or None")
        if type(allowed_origin) is not str:
            raise TypeError("allowed_origin must be an exact built-in string")
        if type(worker_identity) is not str:
            raise TypeError("worker_identity must be an exact built-in string")
        if self._WORKER_IDENTITY.fullmatch(worker_identity) is None:
            raise ValueError("worker_identity must use the reserved worker namespace")
        parsed = urlsplit(allowed_origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("allowed_origin must be an exact HTTP origin")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("non-loopback browser origins require HTTPS")

        self._sessions = sessions
        self._verifier = verifier
        self._capability = capability
        self._allowed_origin = allowed_origin
        self._worker_identity = worker_identity
        if tailnet_authorizer is not None and type(tailnet_authorizer) is not TailnetPeerAuthorizer:
            raise TypeError("tailnet_authorizer must be an exact TailnetPeerAuthorizer")
        if (
            loopback_authorizer is not None
            and type(loopback_authorizer) is not LoopbackPeerAuthorizer
        ):
            raise TypeError("loopback_authorizer must be an exact LoopbackPeerAuthorizer")
        stable_authorizers = tuple(
            authorizer
            for authorizer in (tailnet_authorizer, loopback_authorizer)
            if authorizer is not None
        )
        if (1 if capability is not None else 0) + len(stable_authorizers) != 1:
            raise ValueError("exactly one bootstrap authority must be configured")
        self._tailnet_authorizer = tailnet_authorizer
        self._stable_authorizer: StablePeerAuthorizer | None = (
            stable_authorizers[0] if stable_authorizers else None
        )

    async def handle(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        peer: StablePeerAddress | None = None,
    ) -> BrowserBootstrapResponse:
        """Validate, claim, provision, then expose one browser credential."""

        if type(method) is not str or type(path) is not str:
            raise TypeError("method and path must be exact built-in strings")
        if type(headers) is not dict:
            raise TypeError("headers must be an exact built-in dictionary")
        if type(body) is not bytes:
            raise TypeError("body must be exact bytes")
        routes = {
            "/api/v1/approval",
            "/api/v1/audio-diagnostic",
            "/api/v1/events",
            "/api/v1/evidence-consent",
            "/api/v1/evidence-revoke",
            "/api/v1/input",
            "/api/v1/media",
            "/api/v1/model",
            "/api/v1/models",
            "/api/v1/projection-resync",
            "/api/v1/rebind",
            "/api/v1/refresh",
            "/api/v1/search-egress-consent",
            "/api/v1/search-egress-revoke",
            "/api/v1/stop",
            "/api/v1/voice",
            "/api/v1/voices",
            "/api/v1/yield",
        }
        if self._capability is not None:
            routes.add("/api/v1/bootstrap")
        if self._stable_authorizer is not None:
            routes.add("/api/v1/stable-bootstrap")
            routes.add("/api/v1/stable-rebind")
        if self._tailnet_authorizer is not None:
            routes.add("/api/v1/tailnet-bootstrap")
            routes.add("/api/v1/tailnet-rebind")
        if method != "POST" or path not in routes:
            raise PermissionError("client endpoint is not available")
        if len(headers) > 32:
            raise ValueError("too many client request headers")
        for name, value in headers.items():
            if type(name) is not str or type(value) is not str:
                raise TypeError("header names and values must be exact strings")
            if name != name.lower() or len(name) > 64 or len(value) > 8192:
                raise ValueError("client request header is invalid")
        if headers.get("origin") != self._allowed_origin:
            raise PermissionError("client origin is not allowed")
        raw_length = headers.get("content-length")
        if raw_length is None or not raw_length.isdigit() or raw_length != str(len(body)):
            raise ValueError("client content-length is invalid")
        bearer = ""
        stable_bootstrap_paths = {"/api/v1/stable-bootstrap", "/api/v1/tailnet-bootstrap"}
        stable_rebind_paths = {"/api/v1/stable-rebind", "/api/v1/tailnet-rebind"}
        if path in stable_bootstrap_paths | stable_rebind_paths:
            if "authorization" in headers or headers.get("sec-fetch-site") != "same-origin":
                raise PermissionError("stable session request is invalid")
            authorizer = self._stable_authorizer
            if type(authorizer) is TailnetPeerAuthorizer:
                if type(peer) is not TailnetPeerAddress:
                    raise PermissionError("stable session request is invalid")
                await authorizer.authorize(peer)
            elif type(authorizer) is LoopbackPeerAuthorizer:
                if type(peer) is not LoopbackPeerAddress:
                    raise PermissionError("stable session request is invalid")
                await authorizer.authorize(peer)
            else:
                raise PermissionError("stable session request is invalid")
            if path in stable_bootstrap_paths:
                if body:
                    raise ValueError("stable bootstrap request body must be empty")
                credential = await self._sessions.start()
            else:
                if not 1 <= len(body) <= 384:
                    raise ValueError("stable rebind body must contain 1 to 384 bytes")
                if headers.get("content-type") != "application/json":
                    raise ValueError("stable rebind content-type must be application/json")
                try:
                    decoded = json.loads(
                        body.decode("utf-8"),
                        object_pairs_hook=_strict_json_object,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ValueError("stable rebind body must be strict UTF-8 JSON") from None
                if type(decoded) is not dict or set(decoded) != {
                    "participantIdentity",
                    "requestId",
                }:
                    raise ValueError(
                        "stable rebind must contain exact participantIdentity and requestId"
                    )
                participant_identity = decoded["participantIdentity"]
                request_id = decoded["requestId"]
                if type(participant_identity) is not str or type(request_id) is not str:
                    raise TypeError("stable rebind fields must be exact strings")
                credential = await self._sessions.rebind(
                    participant_identity=participant_identity,
                    request_id=request_id,
                )
            payload: object = self._credential_payload(credential)
            status = 200
        else:
            authorization = headers.get("authorization")
            if authorization is None or not authorization.startswith("Bearer "):
                raise PermissionError("client authorization is invalid")
            bearer = authorization[7:]

        if path == "/api/v1/bootstrap":
            if body:
                raise ValueError("bootstrap request body must be empty")
            assert self._capability is not None
            self._capability.consume(bearer)
            credential = await self._sessions.start()
            payload = self._credential_payload(credential)
            status = 200
        elif path in stable_bootstrap_paths | stable_rebind_paths:
            pass
        elif path == "/api/v1/projection-resync":
            if body or "content-type" in headers:
                raise ValueError("projection resync request body must be empty")
            identity = self._verifier.verify(bearer)
            credential = await self._sessions.projection_resync(
                participant_identity=identity,
            )
            payload = self._credential_payload(credential)
            status = 200
        elif path == "/api/v1/rebind":
            if not 1 <= len(body) <= 192:
                raise ValueError("rebind request body must contain 1 to 192 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("rebind content-type must be application/json")
            try:
                decoded = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("rebind body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {"requestId"}:
                raise ValueError("rebind must contain exactly requestId")
            request_id = decoded["requestId"]
            if type(request_id) is not str:
                raise TypeError("requestId must be an exact string")
            identity = self._verifier.verify(bearer)
            credential = await self._sessions.rebind(
                participant_identity=identity,
                request_id=request_id,
            )
            payload = self._credential_payload(credential)
            status = 200
        elif path == "/api/v1/evidence-consent":
            if headers.get("content-type") != "application/json":
                raise ValueError("evidence consent content-type must be application/json")
            identity = self._verifier.verify(bearer)
            consent_request = parse_evidence_consent_request(body)
            result = await self._sessions.consent_to_evidence(
                participant_identity=identity,
                request=consent_request,
            )
            payload = result.payload
            status = result.status
        elif path == "/api/v1/evidence-revoke":
            if headers.get("content-type") != "application/json":
                raise ValueError("evidence revoke content-type must be application/json")
            identity = self._verifier.verify(bearer)
            revoke_request = parse_evidence_revoke_request(body)
            result = await self._sessions.revoke_evidence(
                participant_identity=identity,
                request=revoke_request,
            )
            payload = result.payload
            status = result.status
        elif path == "/api/v1/search-egress-consent":
            if headers.get("content-type") != "application/json":
                raise ValueError("search egress consent content-type must be application/json")
            identity = self._verifier.verify(bearer)
            search_consent_request = parse_search_egress_consent_request(body)
            search_status = await self._sessions.consent_to_search_egress(
                participant_identity=identity,
                request=search_consent_request,
            )
            payload = search_egress_status_to_primitive(search_status)
            status = 200
        elif path == "/api/v1/search-egress-revoke":
            if headers.get("content-type") != "application/json":
                raise ValueError("search egress revoke content-type must be application/json")
            identity = self._verifier.verify(bearer)
            search_revoke_request = parse_search_egress_revoke_request(body)
            search_status = await self._sessions.revoke_search_egress(
                participant_identity=identity,
                request=search_revoke_request,
            )
            payload = search_egress_status_to_primitive(search_status)
            status = 200
        elif path == "/api/v1/audio-diagnostic":
            if not 1 <= len(body) <= 1024:
                raise ValueError("audio diagnostic body must contain 1 to 1024 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("audio diagnostic content-type must be application/json")
            identity = self._verifier.verify(bearer)
            try:
                decoded = json.loads(body.decode("utf-8"), object_pairs_hook=_strict_json_object)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("audio diagnostic body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {
                "version",
                "streamId",
                "supported",
                "applied",
                "render",
            }:
                raise ValueError("audio diagnostic fields are invalid")
            supported = decoded["supported"]
            applied = decoded["applied"]
            render = decoded["render"]
            processor_fields = {
                "echoCancellation",
                "autoGainControl",
                "noiseSuppression",
                "voiceIsolation",
            }
            render_fields = {"subscribedToAttachMs", "attachToPlayingMs", "playingToAdvanceMs"}
            if (
                type(supported) is not dict
                or set(supported) != processor_fields
                or type(applied) is not dict
                or set(applied) != processor_fields
                or type(render) is not dict
                or set(render) != render_fields
                or type(decoded["version"]) is not int
                or decoded["version"] != 1
            ):
                raise ValueError("audio diagnostic nested schema is invalid")
            diagnostic = BrowserAudioDiagnostic(
                stream_id=decoded["streamId"],
                supported_echo_cancellation=supported["echoCancellation"],
                supported_auto_gain_control=supported["autoGainControl"],
                supported_noise_suppression=supported["noiseSuppression"],
                supported_voice_isolation=supported["voiceIsolation"],
                applied_echo_cancellation=applied["echoCancellation"],
                applied_auto_gain_control=applied["autoGainControl"],
                applied_noise_suppression=applied["noiseSuppression"],
                applied_voice_isolation=applied["voiceIsolation"],
                subscribed_to_attach_ms=render["subscribedToAttachMs"],
                attach_to_playing_ms=render["attachToPlayingMs"],
                playing_to_advance_ms=render["playingToAdvanceMs"],
            )
            await self._sessions.submit_audio_diagnostic(
                participant_identity=identity, diagnostic=diagnostic
            )
            payload = {"streamId": diagnostic.stream_id, "version": 1}
            status = 202
        elif path == "/api/v1/media":
            if not 1 <= len(body) <= 128:
                raise ValueError("media activation body must contain 1 to 128 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("media activation content-type must be application/json")
            identity = self._verifier.verify(bearer)
            try:
                decoded = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("media activation body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {"mediaIncarnation"}:
                raise ValueError("media activation must contain exactly mediaIncarnation")
            media_incarnation = decoded["mediaIncarnation"]
            if type(media_incarnation) is not int:
                raise TypeError("mediaIncarnation must be an exact integer")
            await self._sessions.activate_media(
                participant_identity=identity,
                media_incarnation=media_incarnation,
            )
            payload = {"mediaIncarnation": media_incarnation, "version": 1}
            status = 202
        elif path == "/api/v1/yield":
            if not 1 <= len(body) <= 512:
                raise ValueError("yield body must contain 1 to 512 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("yield content-type must be application/json")
            identity = self._verifier.verify(bearer)
            try:
                decoded = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("yield body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {
                "turnId",
                "turnGeneration",
                "chunkId",
                "streamId",
            }:
                raise ValueError(
                    "yield must contain exact turnId, turnGeneration, chunkId, and streamId"
                )
            turn_id = decoded["turnId"]
            turn_generation = decoded["turnGeneration"]
            chunk_id = decoded["chunkId"]
            stream_id = decoded["streamId"]
            if type(turn_id) is not str or type(chunk_id) is not str or type(stream_id) is not str:
                raise TypeError("yield fields must be exact strings")
            if type(turn_generation) is not int:
                raise TypeError("yield turnGeneration must be an exact integer")
            matched = await self._sessions.yield_speech(
                participant_identity=identity,
                turn_id=turn_id,
                turn_generation=turn_generation,
                chunk_id=chunk_id,
                stream_id=stream_id,
            )
            payload = {"matched": matched, "version": 1}
            status = 202
        elif path == "/api/v1/input":
            if not 1 <= len(body) <= 8192:
                raise ValueError("typed input body must contain 1 to 8192 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("typed input content-type must be application/json")
            identity = self._verifier.verify(bearer)
            try:
                decoded = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("typed input body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {"sequence", "text"}:
                raise ValueError("typed input must contain exact sequence and text fields")
            sequence = decoded["sequence"]
            text = decoded["text"]
            if type(sequence) is not int or type(text) is not str:
                raise TypeError("typed input fields have invalid primitive types")
            await self._sessions.submit_text(
                participant_identity=identity,
                sequence=sequence,
                text=text,
            )
            payload = {"sequence": sequence, "version": 1}
            status = 202
        elif path == "/api/v1/approval":
            if not 1 <= len(body) <= 1024:
                raise ValueError("approval body must contain 1 to 1024 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("approval content-type must be application/json")
            identity = self._verifier.verify(bearer)
            try:
                decoded = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("approval body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {
                "approvalId",
                "decision",
                "sequence",
            }:
                raise ValueError(
                    "approval request must contain exact request, decision, and sequence fields"
                )
            approval_id = decoded["approvalId"]
            decision = decoded["decision"]
            sequence = decoded["sequence"]
            if (
                type(approval_id) is not str
                or type(decision) is not str
                or type(sequence) is not int
            ):
                raise TypeError("approval fields have invalid primitive types")
            await self._sessions.decide_approval(
                participant_identity=identity,
                approval_id=approval_id,
                sequence=sequence,
                decision=decision,
            )
            payload = {
                "approvalId": approval_id,
                "decision": decision,
                "sequence": sequence,
                "version": 1,
            }
            status = 202
        elif path == "/api/v1/events":
            if not 1 <= len(body) <= 1024:
                raise ValueError("events request body must contain 1 to 1024 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("events content-type must be application/json")
            identity = self._verifier.verify(bearer)
            try:
                decoded = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("events body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {"after"}:
                raise ValueError("events request must contain the exact after field")
            after = decoded["after"]
            if type(after) is not int:
                raise TypeError("events after field must be an exact integer")
            events = await self._sessions.public_events_after(
                participant_identity=identity,
                sequence=after,
            )
            payload = {
                "events": [
                    {
                        "data": dict(event.data),
                        "kind": event.kind,
                        "monotonicMs": event.monotonic_ms,
                        "sequence": event.sequence,
                    }
                    for event in events
                ],
                "version": 1,
            }
            status = 200
        elif path == "/api/v1/voices":
            if body:
                raise ValueError("voices request body must be empty")
            identity = self._verifier.verify(bearer)
            voices, selected_voice = await self._sessions.voice_configuration(
                participant_identity=identity,
            )
            payload = {
                "selectedVoice": selected_voice,
                "version": 1,
                "voices": list(voices),
            }
            status = 200
        elif path == "/api/v1/models":
            if body:
                raise ValueError("models request body must be empty")
            identity = self._verifier.verify(bearer)
            catalog = await self._sessions.selectable_model_configuration(
                participant_identity=identity,
            )
            payload = catalog.public_data()
            status = 200
        elif path == "/api/v1/model":
            if not 1 <= len(body) <= 1024:
                raise ValueError("model body must contain 1 to 1024 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("model content-type must be application/json")
            identity = self._verifier.verify(bearer)
            try:
                decoded = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("model body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {"effort", "model"}:
                raise ValueError("model request must contain exact effort and model fields")
            model = decoded["model"]
            effort = decoded["effort"]
            if type(model) is not str or type(effort) is not str:
                raise TypeError("model request fields must be exact built-in strings")
            catalog = await self._sessions.change_model(
                participant_identity=identity,
                model=model,
                effort=effort,
            )
            payload = catalog.public_data()
            status = 200
        elif path == "/api/v1/voice":
            if not 1 <= len(body) <= 256:
                raise ValueError("voice body must contain 1 to 256 bytes")
            if headers.get("content-type") != "application/json":
                raise ValueError("voice content-type must be application/json")
            identity = self._verifier.verify(bearer)
            try:
                decoded = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("voice body must be strict UTF-8 JSON") from None
            if type(decoded) is not dict or set(decoded) != {"voice"}:
                raise ValueError("voice request must contain the exact voice field")
            voice = decoded["voice"]
            if type(voice) is not str:
                raise TypeError("voice field must be an exact built-in string")
            await self._sessions.change_voice(
                participant_identity=identity,
                voice=voice,
            )
            payload = {"selectedVoice": voice, "version": 1}
            status = 200
        elif path == "/api/v1/refresh":
            if body:
                raise ValueError("refresh request body must be empty")
            identity = self._verifier.verify(bearer)
            refreshed = await self._sessions.refresh_credential(
                participant_identity=identity,
            )
            payload = {
                "expiresInSeconds": refreshed.expires_in_seconds,
                "participantIdentity": refreshed.participant_identity,
                "roomName": refreshed.room_name,
                "token": refreshed.token,
                "url": refreshed.url,
                "version": 1,
                "workerIdentity": self._worker_identity,
            }
            status = 200
        else:
            request_id = None
            if body:
                if not len(body) <= 192:
                    raise ValueError("stop request body must contain at most 192 bytes")
                if headers.get("content-type") != "application/json":
                    raise ValueError("stop content-type must be application/json")
                try:
                    decoded = json.loads(
                        body.decode("utf-8"),
                        object_pairs_hook=_strict_json_object,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ValueError("stop body must be strict UTF-8 JSON") from None
                if type(decoded) is not dict or set(decoded) != {"requestId"}:
                    raise ValueError("stop must contain exactly requestId")
                request_id = decoded["requestId"]
                if type(request_id) is not str:
                    raise TypeError("requestId must be an exact string")
            identity = self._verifier.verify(bearer)
            await self._sessions.stop(
                participant_identity=identity,
                request_id=request_id,
            )
            payload = {"stopped": True, "version": 1}
            status = 200

        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return BrowserBootstrapResponse(
            status=status,
            headers=self._RESPONSE_HEADERS,
            body=encoded,
        )

    def _credential_payload(self, credential: object) -> dict[str, object]:
        if type(credential) is not BrowserJoinCredential:
            raise TypeError("credential must be an exact BrowserJoinCredential")
        return {
            "expiresInSeconds": credential.expires_in_seconds,
            "participantIdentity": credential.participant_identity,
            "roomName": credential.room_name,
            "token": credential.token,
            "url": credential.url,
            "version": 1,
            "workerIdentity": self._worker_identity,
        }
