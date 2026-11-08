"""Recognize model output that imitates a call to an advertised foreground tool.

Real calls reach the runtime only through structured provider channels, never as reply
text, so reply text that looks like a call is an imitation: call syntax is not
speakable. This is a bounded syntactic guard applied per segment to model output in the
provider adapters, beside the background-work claim guard. It is not a semantic
classifier. Host-composed text (proposal readback, Hermes completion summaries) is not
model output and is never checked here.

A segment is an imitation when it contains, matched case-insensitively:

* an advertised tool name as a whole identifier, optionally namespace-qualified (such as
  ``functions.start_work``), immediately followed by ``(``;
* a quoted JSON key ``name``, ``kind`` or ``tool`` whose quoted value is an advertised name;
* an advertised name used as a quoted JSON key whose value opens an object; or
* the XML-style ``<function=NAME`` form.

Quotes and backticks do not exempt a segment because speech strips them. A plain mention
of a tool name without call syntax is allowed. Detection is per segment: a call split
across segments (for example JSON with the arguments before the name) is not detected.
"""

from __future__ import annotations

import re

# The static superset of tool names any foreground turn advertises. Tests keep it equal
# to the Codex dynamic tools and a superset of the Ollama work decision kinds.
ADVERTISED_TOOL_NAMES = frozenset({"start_work", "task_status", "search_knowledge"})

_NAMES = "|".join(sorted(re.escape(name) for name in ADVERTISED_TOOL_NAMES))
_QUOTE = r"""\\{0,3}["']"""
_WS = r"[ \t\r\n]{0,16}"
_CALL = re.compile(rf"(?<![A-Za-z0-9_])(?:{_NAMES})\(", re.IGNORECASE)
_KEYED_NAME = re.compile(
    rf"{_QUOTE}(?:name|kind|tool){_QUOTE}{_WS}:{_WS}{_QUOTE}(?:{_NAMES}){_QUOTE}",
    re.IGNORECASE,
)
_NAME_AS_KEY = re.compile(rf"{_QUOTE}(?:{_NAMES}){_QUOTE}{_WS}:{_WS}\{{", re.IGNORECASE)
_XML_FUNCTION = re.compile(
    rf"<{_WS}function{_WS}={_WS}[\"']?(?:{_NAMES})(?![A-Za-z0-9_])", re.IGNORECASE
)
_FORMS = (_CALL, _KEYED_NAME, _NAME_AS_KEY, _XML_FUNCTION)


def contains_tool_call_imitation(text: str) -> bool:
    if type(text) is not str:
        raise TypeError("model text must be an exact built-in string")
    return any(form.search(text) is not None for form in _FORMS)
