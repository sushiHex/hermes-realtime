"""Result-only character limits; unrelated request and reason bounds stay separate."""

# Unicode scalars, matching JavaScript Array.from(text).length after validation.
MAX_COMPLETED_RESULT_CHARS = 4096
MAX_LEGACY_RESULT_CHARS = 1024
MAX_PRESENTATION_VOICE_CHARS = 220
MAX_PRESENTATION_DETAIL_CHARS = 1800

# A legacy client receives an explicit limitation instead of a sliced result.
LEGACY_RESULT_NOTICE = (
    "This result exceeds this client's display limit. "
    "Use an updated client to view the complete result."
)
UNUSABLE_RESULT_NOTICE = "The completed result could not be displayed within the supported limit."


def has_valid_result_unicode(text: str) -> bool:
    """UTF-16 surrogate code points are not standalone Unicode scalar values."""
    return not any(0xD800 <= ord(character) <= 0xDFFF for character in text)
