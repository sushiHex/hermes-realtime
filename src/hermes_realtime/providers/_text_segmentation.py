"""Shared bounded text segmentation helpers for streaming inference adapters."""

from __future__ import annotations

import re

from hermes_realtime.providers._speech_text import _scan_markdown_emphasis

_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")


def first_speakable_sentence_end(text: str) -> int | None:
    """Return the first sentence end outside list markers and open emphasis."""

    for match in _SENTENCE_END.finditer(text):
        line_start = max(text.rfind("\n", 0, match.start()), text.rfind("\r", 0, match.start())) + 1
        marker_prefix = text[line_start : match.start()].strip()
        lexical_characters = sum(character.isalnum() for character in marker_prefix)
        if (
            marker_prefix.isdecimal()
            or lexical_characters <= 2
            or _scan_markdown_emphasis(text[: match.end()])[1]
        ):
            continue
        return match.end()
    return None
