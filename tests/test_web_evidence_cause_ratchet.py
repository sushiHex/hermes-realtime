"""Every literal ``ValueError`` category in the web-evidence modules is declared (#550).

The ``[web-evidence]`` marker names a refusal's cause only when the category is a member
of ``_CAUSES``; any other literal is emitted as ``undeclared``. This ratchet collects, by
AST, the first argument of every ``ValueError(...)`` constructed in the modules on the
refusal path and requires the collected set to EQUAL the declared causes plus the
exclusions below. A new category that nobody declared, or a declared one that is no longer
raised, both fail.
"""

from __future__ import annotations

import ast
from pathlib import Path

from hermes_realtime.integration import native_web_evidence

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "hermes_realtime" / "integration"
MODULES = (
    "native_web_evidence.py",
    "evidence_presentation.py",
    "presentation_transport.py",
    "web_evidence_contract.py",
)

# Categories raised in these modules that can never reach a run refusal, with the reason.
EXCLUDED = {
    # Request admission: ``_request_binding_for`` is answered 400/409 by the HTTP
    # middleware before any run row exists; ``web-evidence-generation`` is also the
    # constructor's generation check (registration).
    "web-evidence-contract": "request admission",
    "web-evidence-generation": "request admission and registration",
    "web-evidence-objective": "request admission",
    # Registration and configuration: raised while the plugin installs, before any run.
    "web-evidence-native-auth-required": "registration",
    "presentation-config": "configuration",
    # Startup credential verification: wrapped into RuntimeError and its own marker.
    "credential_config_drift": "registration",
    "credential_runtime_origin": "registration",
    # Host-side attestation parsing: ``attestation_outcome`` returns None, never refuses.
    "duplicate attestation field": "attestation parsing",
}


def _literals(node: ast.AST, where: str) -> list[str]:
    if type(node) is ast.Constant and type(node.value) is str:
        return [node.value]
    if type(node) is ast.IfExp:
        return _literals(node.body, where) + _literals(node.orelse, where)
    return [f"<non-literal at {where}>"]


def _raised_categories() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name in MODULES:
        tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            where = f"{name}:{getattr(node, 'lineno', 0)}"
            if type(node) is ast.Raise and type(node.exc) is ast.Name and (
                    node.exc.id == "ValueError"):
                found.setdefault(f"<bare ValueError at {where}>", []).append(where)
            if not (type(node) is ast.Call and type(node.func) is ast.Name
                    and node.func.id == "ValueError"):
                continue
            if not node.args:
                found.setdefault(f"<empty ValueError at {where}>", []).append(where)
                continue
            for literal in _literals(node.args[0], where):
                found.setdefault(literal, []).append(where)
    return found


def test_every_raised_category_is_declared_or_excluded_and_none_is_stale() -> None:
    # A missing declaration fails with the full list rather than an import error.
    declared = set(getattr(native_web_evidence, "_CAUSES", frozenset()))
    raised = _raised_categories()
    allowed = declared | set(EXCLUDED)
    unmapped = sorted(f"{category} ({', '.join(raised[category])})"
                      for category in set(raised) - allowed)
    stale = sorted(allowed - set(raised))
    assert not unmapped and not stale, (
        "undeclared categories:\n  " + "\n  ".join(unmapped)
        + "\nstale declarations:\n  " + "\n  ".join(stale))


def test_declared_causes_and_exclusions_are_disjoint_and_cover_the_reason_table() -> None:
    causes = native_web_evidence._CAUSES
    assert type(causes) is frozenset
    assert not causes & set(EXCLUDED)
    assert set(native_web_evidence._REASONS) <= causes
    assert not {"none", "undeclared"} & causes


def test_every_presenter_stage_is_declared() -> None:
    tree = ast.parse((PACKAGE / "evidence_presentation.py").read_text(encoding="utf-8"))
    assigned = set()
    for node in ast.walk(tree):
        if type(node) is not ast.Assign:
            continue
        for target in node.targets:
            if (type(target) is ast.Subscript and ast.unparse(target) == "evidence['stage']"):
                assert type(node.value) is ast.Constant, ast.unparse(node)
                assigned.add(node.value.value)
    assert assigned == set(native_web_evidence._STAGES)
    assert "none" not in native_web_evidence._STAGES
