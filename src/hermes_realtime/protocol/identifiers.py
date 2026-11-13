"""Identifier grammar and private-token rules: the one definition of each.

Every other spelling of a rule defined here is a rendering of it: the Pydantic wire types in
``events``, the validators across the package and ``scripts``, and the browser's regular
expression literals. ``tests/test_identifier_single_source.py`` fails when a rule is restated
instead of imported, and when a browser literal stops matching its rendering.

Exact rules end in ``\\Z`` so that ``match`` and ``fullmatch`` agree. This module imports
nothing from the package.
"""

from __future__ import annotations

import re
from typing import Final

MAX_IDENTIFIER_CHARS: Final = 128
MAX_VOICE_CONVERSATION_ID_CHARS: Final = 64

_HEAD: Final = "[A-Za-z0-9]"
_TAIL: Final = "[A-Za-z0-9_.:-]"
_URL_SAFE: Final = "[A-Za-z0-9_-]"


def _bounded(prefix: str, head: str = _HEAD) -> str:
    """One head character and a tail filling the identifier bound after ``prefix``."""
    return f"{prefix}{head}{_TAIL}{{0,{MAX_IDENTIFIER_CHARS - len(prefix) - 1}}}"


def _exact(grammar: str) -> re.Pattern[str]:
    return re.compile(grammar + r"\Z")


# Unanchored grammar, for renderings that carry their own anchors and length bound.
IDENTIFIER_GRAMMAR: Final = _bounded("")
VOICE_CONVERSATION_ID_GRAMMAR: Final = f"{_URL_SAFE}{{1,{MAX_VOICE_CONVERSATION_ID_CHARS}}}"

# A protocol identifier: event, session, utterance, task and run ids on the bridge.
IDENTIFIER: Final = _exact(IDENTIFIER_GRAMMAR)
# The public task namespace and the private delegation namespace, inside the same bound.
TASK_ID: Final = _exact(_bounded("task_"))
DELEGATION_RUN_ID: Final = _exact(_bounded("deleg_"))
# A turn identity the browser may anchor to; it starts with a letter.
TURN_ID: Final = _exact(_bounded("", head="[A-Za-z]"))
METRIC_LABEL: Final = _exact("[a-z][a-z0-9_-]{0,63}")
WORKER_IDENTITY: Final = _exact(f"worker_{_URL_SAFE}{{8,64}}")
BROWSER_IDENTITY: Final = _exact("browser_[a-f0-9]{16,64}")
VOICE_CONVERSATION_ID: Final = _exact(VOICE_CONVERSATION_ID_GRAMMAR)

# Searched, never matched whole: a delegation run id wherever it occurs in model-visible text.
PRIVATE_RUN_TOKEN: Final = re.compile(f"deleg_{_HEAD}{_TAIL}*")
# Searched and replaced in text bound for the browser. It is a second, different statement of
# "this text names a private handle": wider than PRIVATE_RUN_TOKEN in the prefixes it knows,
# narrower in the tail it consumes.
PRIVATE_HANDLE: Final = re.compile(
    f"(?:deleg_{_URL_SAFE}*|run_{_URL_SAFE}{{8,}}|subagent_{_URL_SAFE}+)"
)
