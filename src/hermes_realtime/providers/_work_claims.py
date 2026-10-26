"""Recognize bounded reports of started work used by foreground adapters.

This is a reporting guard, not an intent classifier or proof that arbitrary model
prose is truthful. Future offers alone do not report accepted or active work and
are ordinary speech; they grant no proposal, confirmation or dispatch authority.
Only the finite started/launched phrases below are recognized. Other unsupported
acceptance, progress, completion or misleading promises still need manual review.
"""

from __future__ import annotations

import re

_BACKGROUND_WORK_CLAIM = re.compile(
    r"\bi(?:'|’)ve\s+(?:started|launched)\b|"
    r"\bi\s+(?:started|launched)\s+(?:a|the)\s+(?:background\s+)?(?:task|work|analysis)\b",
    re.IGNORECASE,
)


def has_background_work_claim(text: str) -> bool:
    return _BACKGROUND_WORK_CLAIM.search(text) is not None
