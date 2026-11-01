"""Pure, bounded presentation of completed-task results."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_COMPLETION_REFERENCE = re.compile(
    r"(?:\[([^\]]+)\]\()?(https?://[^\s<>]+)", re.IGNORECASE,
)
_COMPLETION_MACHINE_TOKEN = re.compile(r"\b(?:task_|run_|deleg_)[A-Za-z0-9_.:-]+")
_PRIVATE_RESULT_HANDLE = re.compile(r"(?:deleg_[A-Za-z0-9_-]*|run_[A-Za-z0-9_-]{8,})")
_TAGGED_COMPLETION = re.compile(
    r"\A<voice>(?P<voice>[\s\S]*?)</voice>\s*"
    r"<detail>(?P<detail>[\s\S]*?)</detail>\Z"
)
_COMPLETION_PROTOCOL_TAG = re.compile(r"</?(?:voice|detail)>", re.IGNORECASE)
_MAX_SPOKEN_COMPLETION_CHARS = 240
_MAX_CONVERSATION_COMPLETION_DETAIL_CHARS = 3200
_NO_DISPLAYABLE_RESULT = "No displayable result was returned."


def _completion_projections(detail: str) -> tuple[str, str]:
    tagged = _TAGGED_COMPLETION.fullmatch(detail)
    if tagged is not None:
        voice = tagged.group("voice").strip()
        visual = tagged.group("detail").strip()
        visual = _COMPLETION_PROTOCOL_TAG.sub("", visual).strip()
        valid_voice = (
            1 <= len(voice) <= _MAX_SPOKEN_COMPLETION_CHARS
            and len(voice.splitlines()) == 1
            and "<" not in voice
            and ">" not in voice
        )
        if valid_voice:
            # Providers depend on the conversation types during package startup.
            from hermes_realtime.providers import _speech_text

            projected = _speech_text.project_speech_text(voice)
            # Formatting must not drop a valid result or widen its publication bound.
            # The TTS projection still localizes the original if expansion will not fit.
            if len(projected) <= _MAX_SPOKEN_COMPLETION_CHARS:
                return projected, visual or voice
            return voice, visual or voice
        protocol_free = _COMPLETION_PROTOCOL_TAG.sub("", detail).strip()
        return "", visual or protocol_free or _NO_DISPLAYABLE_RESULT
    if _COMPLETION_PROTOCOL_TAG.search(detail) is not None:
        visual = _COMPLETION_PROTOCOL_TAG.sub("", detail).strip()
        return "", visual or _NO_DISPLAYABLE_RESULT
    return detail, detail


def _completion_source_label(match: re.Match[str]) -> str:
    """Render a supplied URL's hostname, without navigating or verifying its source."""
    label, raw = match.groups()
    url = raw.rstrip(".,;:!?")
    punctuation = raw[len(url):]
    if label is not None:
        return _completion_reference_text(label) + punctuation
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        hostname = None
    return (hostname or "link") + punctuation


def _completion_reference_text(detail: str) -> str:
    """Keep labels/hostnames while URLs remain only in the local task result."""
    body, marker, catalog = detail.rpartition("\n\nSources:\n")
    if marker and catalog:
        labels: list[str] = []
        seen: set[str] = set()
        for line in catalog.splitlines():
            reference = _COMPLETION_REFERENCE.fullmatch(line.strip())
            # Only the emitted raw-URL catalog is a list of interchangeable host labels.
            if reference is None or reference.group(1) is not None:
                break
            label = _completion_source_label(reference)
            if label.casefold() not in seen:
                labels.append(label)
                seen.add(label.casefold())
        else:
            return (
                _COMPLETION_REFERENCE.sub(_completion_source_label, body)
                + marker + "\n".join(labels)
            )
    # Consume the complete URL token before rendering its optional Markdown label;
    # stopping at a path parenthesis would forward the remainder of that URL.
    return _COMPLETION_REFERENCE.sub(_completion_source_label, detail)


def completed_result_reference(detail: str) -> str:
    """Return the existing bounded, source-safe visual result as untrusted data."""
    _, visual = _completion_projections(detail)
    safe = _COMPLETION_MACHINE_TOKEN.sub("the reference", _completion_reference_text(visual))
    safe = _PRIVATE_RESULT_HANDLE.sub("private reference", safe)
    safe = " ".join(safe.replace("•", " ").split()).strip()
    return _bounded_result(safe) or _NO_DISPLAYABLE_RESULT


def _voice_friendly_completion_text(status: str, detail: str) -> str:
    detail = _completion_reference_text(detail)
    detail = _COMPLETION_MACHINE_TOKEN.sub("the reference", detail)
    detail = " ".join(detail.replace("•", " ").split()).strip()
    if detail:
        detail = detail[0].upper() + detail[1:]
        if detail[-1] not in ".!?":
            detail += "."
    shortened = len(detail) > _MAX_SPOKEN_COMPLETION_CHARS
    if shortened:
        boundary = detail.rfind(". ", 0, _MAX_SPOKEN_COMPLETION_CHARS)
        if boundary < 160:
            boundary = detail.rfind(" ", 0, _MAX_SPOKEN_COMPLETION_CHARS)
        if boundary < 160:
            boundary = _MAX_SPOKEN_COMPLETION_CHARS
        detail = detail[:boundary].rstrip(" ,;:") + "."
    if status == "completed":
        parts = [detail] if detail else []
        if shortened or not detail:
            parts.append("The full result is on screen.")
        return " ".join(parts)
    lead = {
        "failed": "I ran into a problem with that work.",
        "interrupted": "That work stopped.",
    }[status]
    parts = [lead]
    if detail:
        parts.append(detail)
    if shortened or not detail:
        parts.append("The full result is on screen.")
    return " ".join(parts)


def _conversation_completion_text(status: str, voice_detail: str, visual_detail: str) -> str:
    """Give inference bounded findings/time with source labels, excluding full URLs."""

    handoff = _voice_friendly_completion_text(status, voice_detail)
    full_result = _completion_reference_text(visual_detail)
    full_result = _COMPLETION_MACHINE_TOKEN.sub("the reference", full_result)
    full_result = " ".join(full_result.replace("•", " ").split()).strip()
    if len(full_result) > _MAX_CONVERSATION_COMPLETION_DETAIL_CHARS:
        boundary = full_result.rfind(" ", 0, _MAX_CONVERSATION_COMPLETION_DETAIL_CHARS - 3)
        if boundary < _MAX_CONVERSATION_COMPLETION_DETAIL_CHARS // 2:
            boundary = _MAX_CONVERSATION_COMPLETION_DETAIL_CHARS - 3
        full_result = full_result[:boundary].rstrip(" ,;:") + "..."
    normalized_voice = " ".join(voice_detail.split()).strip()
    if full_result == _NO_DISPLAYABLE_RESULT:
        return handoff
    if not full_result or (
        len(full_result) <= _MAX_SPOKEN_COMPLETION_CHARS
        and full_result.casefold() == normalized_voice.casefold()
    ):
        return handoff
    # The detail projection is the complete worker result. Supplying its short voice
    # projection as well makes conversational inference repeat the same finding and
    # sometimes speak transport scaffolding such as "Full background result".
    return full_result


def _browser_safe_task_result(detail: str) -> str:
    return _bounded_result(_PRIVATE_RESULT_HANDLE.sub("private reference", detail))


def _bounded_result(text: str) -> str:
    if len(text) <= 1024:
        return text
    boundary = text.rfind(" ", 0, 1021)
    if boundary < 512:
        boundary = 1021
    return text[:boundary].rstrip() + "..."
