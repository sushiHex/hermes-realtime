"""Content-free evidence lines: one declaration is emitter, parser and oracle (ADR 0007).

Standard library only. Nothing in the package uses this unit yet.
"""

from ._core import (
    EXCEPTION_KINDS,
    SILENT,
    Count,
    Digest,
    ExceptionKind,
    ExceptionKinds,
    Layout,
    Line,
    LineFault,
    LineRefused,
    MaxItems,
    Observation,
    Outcome,
    ParseCategory,
    Refusal,
    Signed,
    Tally,
    catalogue,
    dropped,
    isolated_catalogue,
    parse_line,
    strict,
)

__all__ = [
    "EXCEPTION_KINDS",
    "SILENT",
    "Count",
    "Digest",
    "ExceptionKind",
    "ExceptionKinds",
    "Layout",
    "Line",
    "LineFault",
    "LineRefused",
    "MaxItems",
    "Observation",
    "Outcome",
    "ParseCategory",
    "Refusal",
    "Signed",
    "Tally",
    "catalogue",
    "dropped",
    "isolated_catalogue",
    "parse_line",
    "strict",
]
