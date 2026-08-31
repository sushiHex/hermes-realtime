"""Shared bounded projections from display text to synthesized speech text."""

from __future__ import annotations

import unicodedata


def _is_punctuation(character: str | None) -> bool:
    return character is not None and unicodedata.category(character).startswith("P")


def strip_markdown_emphasis_for_speech(text: str) -> str:
    """Remove matched emphasis delimiters in one bounded linear scan."""

    removed = bytearray(len(text))
    openings: dict[tuple[str, int], list[tuple[int, int]]] = {}
    index = 0
    while index < len(text):
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
            can_open = left_flanking
            can_close = right_flanking

        key = (marker, run_length)
        candidates = openings.get(key)
        if can_close and candidates:
            opening_start, opening_end = candidates.pop()
            removed[opening_start:opening_end] = b"\x01" * (opening_end - opening_start)
            removed[index:end] = b"\x01" * run_length
        elif can_open:
            openings.setdefault(key, []).append((index, end))
        index = end
    return "".join(character for offset, character in enumerate(text) if not removed[offset])
