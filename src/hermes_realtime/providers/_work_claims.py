"""Recognize the existing bounded work-claim phrases used by foreground adapters.

This is a reporting guard, not an intent classifier or proof that arbitrary model
prose is truthful. The recognized phrases are deliberately unchanged from Codex.
"""

from __future__ import annotations

import re

_BACKGROUND_WORK_CLAIM = re.compile(
    r"\b(?:i(?:'|’)ll|i\s+will|let\s+me)\s+"
    r"(?:look\s+into|investigate|research|inspect|analy[sz]e)\b|"
    r"\bi(?:'|’)ve\s+(?:started|launched)\b|"
    r"\bi\s+(?:started|launched)\s+(?:a|the)\s+(?:background\s+)?(?:task|work|analysis)\b",
    re.IGNORECASE,
)


def has_background_work_claim(text: str) -> bool:
    return _BACKGROUND_WORK_CLAIM.search(text) is not None
