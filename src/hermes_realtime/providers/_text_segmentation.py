"""Shared bounded text segmentation helpers for streaming inference adapters."""

from __future__ import annotations

import re
import unicodedata

_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")


def _is_punctuation(character: str | None) -> bool:
    return character is not None and unicodedata.category(character).startswith("P")


def _has_open_markdown_emphasis(text: str) -> bool:
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
            candidates.pop()
        elif can_open:
            openings.setdefault(key, []).append((index, end))
        index = end
    return any(candidates for candidates in openings.values())


def first_speakable_sentence_end(text: str) -> int | None:
    """Return the first sentence end outside list markers and open emphasis."""

    for match in _SENTENCE_END.finditer(text):
        line_start = max(text.rfind("\n", 0, match.start()), text.rfind("\r", 0, match.start())) + 1
        marker_prefix = text[line_start : match.start()].strip()
        lexical_characters = sum(character.isalnum() for character in marker_prefix)
        if (
            marker_prefix.isdecimal()
            or lexical_characters <= 2
            or _has_open_markdown_emphasis(text[: match.end()])
        ):
            continue
        return match.end()
    return None
