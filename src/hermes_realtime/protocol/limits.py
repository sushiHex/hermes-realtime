"""The unit a text bound is counted in when the browser enforces the same bound.

A JavaScript string's ``length`` is its UTF-16 code unit count, so a character outside the
basic plane counts twice. A bound the browser checks with ``length`` must be counted the same
way here, or the server admits text the browser then refuses. This module imports nothing from
the package.
"""

from __future__ import annotations

_FIRST_ASTRAL = 0x10000


def utf16_units(text: str) -> int:
    """Return what JavaScript reports as ``text.length``, lone surrogates included."""
    if type(text) is not str:
        raise TypeError("text must be an exact built-in string")
    return len(text) + sum(1 for character in text if ord(character) >= _FIRST_ASTRAL)


def truncate_to_utf16_units(text: str, limit: int) -> str:
    """Return the longest prefix of whole characters within ``limit`` UTF-16 code units."""
    if type(text) is not str:
        raise TypeError("text must be an exact built-in string")
    if type(limit) is not int:
        raise TypeError("limit must be an exact integer")
    if limit < 0:
        raise ValueError("limit must not be negative")
    units = 0
    for index, character in enumerate(text):
        units += 2 if ord(character) >= _FIRST_ASTRAL else 1
        if units > limit:
            return text[:index]
    return text
