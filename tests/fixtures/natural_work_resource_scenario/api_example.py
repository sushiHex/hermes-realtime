"""Synthetic compatibility example, independent of Hermes's production API.

Both aliases remain callable. Their warning and replacement are defined here;
this example does not imply that Hermes has deprecated either of these names.
"""
from __future__ import annotations

import json
import warnings


def decode_event(payload: str) -> object:
    """Decode one JSON value."""
    return json.loads(payload)


def decode_v1(payload: str) -> object:
    """Deprecated alias; callers should use decode_event."""
    warnings.warn("decode_v1 is deprecated; use decode_event", DeprecationWarning, stacklevel=2)
    return decode_event(payload)


def encode_event(value: object) -> str:
    """Encode one JSON value."""
    return json.dumps(value, separators=(",", ":"))


def encode_v1(value: object) -> str:
    """Deprecated alias; callers should use encode_event."""
    warnings.warn("encode_v1 is deprecated; use encode_event", DeprecationWarning, stacklevel=2)
    return encode_event(value)
