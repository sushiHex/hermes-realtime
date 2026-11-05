"""Shared bounded projections from display text to synthesized speech text."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime

_ISO_INSTANT = re.compile(
    r"(?<![\w/])[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-5][0-9]"
    r"(?:Z|[+-][0-9]{2}:[0-5][0-9])(?![\w:+-]|\.\d)",
)
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_UTC_DATE = (r"(?P<month>" + "|".join(_MONTHS)
             + r")[ \t]+(?P<day>[0-9]{1,2}),?[ \t]+(?P<year>[0-9]{4})")
_UTC_CLOCK = r"(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})(?::(?P<second>[0-9]{2}))?"
_UTC_PROSE = (
    re.compile(r"(?<![\w/])" + _UTC_CLOCK + r"[ \t]+UTC,?[ \t]+on[ \t]+" + _UTC_DATE
               + r"(?![\w/+-]|\.\d)"),
    re.compile(r"(?<![\w/])" + _UTC_DATE + r",?[ \t]+at[ \t]+" + _UTC_CLOCK
               + r"[ \t]+UTC(?![\w:/+-]|\.\d)"),
)
# Only new prose forms exclude literal/source spans; existing ISO behavior stays intact.
_PROSE_LITERALS = re.compile(
    r'https?://\S+|www\.\S+|(?<!`)(?P<ticks>`+)(?!`)[\s\S]*?(?<!`)(?P=ticks)(?!`)|'
    r'"(?:\\.|[^"\\])*"|“[^”]*”|'
    r"(?<!\w)'(?:\\.|(?<=\w)'(?=\w)|[^'\\])*'(?!\w)|"
    r"‘(?:(?<=\w)’(?=\w)|[^’])*’(?!\w)",
    re.IGNORECASE,
)
_TEMPERATURE = re.compile(
    r"(?<![\w.])(?P<value>[+\-−]?(?:\d+(?:\.\d+)?|\.\d+))[ \t]*"
    r"(?P<unit>°[ \t]*[FC]|[℉℃])(?!\w)",
)
_TEMPERATURE_UNITS = {"F": "Fahrenheit", "C": "Celsius", "℉": "Fahrenheit", "℃": "Celsius"}
_BACKTICKS = re.compile(r"`+")


def _temperature_value(match: re.Match[str]) -> str:
    value = match["value"]
    quantity = "degree" if re.fullmatch(r"[+\-−]?0*1(?:\.0+)?", value) else "degrees"
    return f"{value} {quantity} {_TEMPERATURE_UNITS[match['unit'][-1]]}"


def _local_instant(match: re.Match[str]) -> str:
    original = match.group()
    try:
        local = datetime.fromisoformat(original).astimezone()
    except (ValueError, OverflowError, OSError):
        # An invalid/unsupported date remains source text; never guess its instant.
        return original
    return _spoken_local_instant(local)


def _spoken_local_instant(local: datetime) -> str:
    zone = local.tzname() or f"UTC{local.strftime('%z')}"
    clock = f"{local.hour % 12 or 12}:{local.minute:02d}"
    if local.second:
        clock += f":{local.second:02d}"
    period = "AM" if local.hour < 12 else "PM"
    return (f"{clock} {period} {zone} on "
            f"{_MONTHS[local.month - 1]} {local.day}, {local.year}")


def _local_utc_prose(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if any(start < match.end() and match.start() < end for start, end in literals):
            return match.group()
        try:
            instant = datetime(
                int(match["year"]), _MONTHS.index(match["month"]) + 1, int(match["day"]),
                int(match["hour"]), int(match["minute"]), int(match["second"] or 0),
                tzinfo=UTC,
            )
            return _spoken_local_instant(instant.astimezone())
        except (ValueError, OverflowError, OSError):
            return match.group()

    for pattern in _UTC_PROSE:
        literals = tuple(match.span() for match in _PROSE_LITERALS.finditer(text))
        text = pattern.sub(replace, text)
    return text


def project_confirmation_values(text: str) -> str:
    """Expand only explicit numeric units and instants for exact spoken confirmation."""
    projected = _TEMPERATURE.sub(_temperature_value, text)
    projected = _ISO_INSTANT.sub(_local_instant, projected)
    projected = _local_utc_prose(projected)
    if len(projected) > 4096:
        raise ValueError("projected speech text exceeds supported bounds")
    return projected


def project_speech_text(text: str) -> str:
    """Expand explicit units and localize canonical instants without changing source text."""
    projected = project_confirmation_values(strip_markdown_emphasis_for_speech(text))
    return projected


def _is_punctuation(character: str | None) -> bool:
    return character is not None and unicodedata.category(character).startswith("P")


def _scan_markdown_emphasis(text: str) -> tuple[bytearray, bool]:
    """Mark matched emphasis delimiters and report whether any opening stays unmatched."""

    removed = bytearray(len(text))
    openings: dict[tuple[str, int], list[tuple[int, int]]] = {}
    # Index equal-length closing backtick runs once; unmatched runs must not
    # cause repeated suffix searches on the streaming segmentation path.
    code_ends: dict[int, int] = {}
    next_ticks: dict[int, int] = {}
    for ticks in reversed(tuple(_BACKTICKS.finditer(text))):
        length = ticks.end() - ticks.start()
        if length in next_ticks:
            code_ends[ticks.start()] = next_ticks[length]
        next_ticks[length] = ticks.end()
    index = 0
    while index < len(text):
        if index in code_ends:
            index = code_ends[index]
            continue
        marker = text[index]
        if marker not in {"*", "_"}:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] == marker:
            end += 1
        run_length = end - index
        preceding_backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            preceding_backslashes += 1
            cursor -= 1
        if run_length > 3 or preceding_backslashes % 2:
            index = end
            continue

        previous = text[index - 1] if index else None
        following = text[end] if end < len(text) else None
        previous_space = previous is None or previous.isspace()
        following_space = following is None or following.isspace()
        left_flanking = not following_space and (
            not _is_punctuation(following) or previous_space or _is_punctuation(previous)
        )
        right_flanking = not previous_space and (
            not _is_punctuation(previous) or following_space or _is_punctuation(following)
        )
        if marker == "_":
            can_open = left_flanking and (not right_flanking or _is_punctuation(previous))
            can_close = right_flanking and (not left_flanking or _is_punctuation(following))
        else:
            # Prefer preserving literal operators/paths over guessing intraword
            # emphasis. Speech must not silently turn 2*3*4 into 234.
            can_open = left_flanking and (
                previous is None or not (previous.isalnum() or previous in "_/\\")
            )
            can_close = right_flanking and (
                following is None or not (following.isalnum() or following in "_/\\")
            )

        key = (marker, run_length)
        candidates = openings.get(key)
        if can_close and candidates:
            opening_start, opening_end = candidates.pop()
            removed[opening_start:opening_end] = b"\x01" * (opening_end - opening_start)
            removed[index:end] = b"\x01" * run_length
        elif can_open:
            openings.setdefault(key, []).append((index, end))
        index = end
    return removed, any(candidates for candidates in openings.values())


def strip_markdown_emphasis_for_speech(text: str) -> str:
    """Remove matched emphasis delimiters in one bounded linear scan."""

    removed, _ = _scan_markdown_emphasis(text)
    return "".join(character for offset, character in enumerate(text) if not removed[offset])
