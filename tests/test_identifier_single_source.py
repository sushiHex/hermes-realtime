"""Each shared identifier rule has one definition; every other spelling is checked against it.

Python: no module outside ``protocol/identifiers.py`` may spell an owned rule as a literal.
Browser: every regular expression literal in ``web/src`` is either the rendering of an owned
rule or is listed here by its exact text, so a new or edited literal has to be classified.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from hermes_realtime.protocol import identifiers
from hermes_realtime.protocol.events import IdentifierString, VoiceConversationId

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / "src/hermes_realtime/protocol/identifiers.py"

EXACT = {
    name: getattr(identifiers, name)
    for name in (
        "IDENTIFIER", "TASK_ID", "DELEGATION_RUN_ID", "TURN_ID", "METRIC_LABEL",
        "WORKER_IDENTITY", "BROWSER_IDENTITY", "VOICE_CONVERSATION_ID",
    )
}
SEARCHED = {name: getattr(identifiers, name) for name in ("PRIVATE_RUN_TOKEN", "PRIVATE_HANDLE")}

# A namespace followed by a character class is a restatement of the rule for that namespace.
_NAMESPACE_RULE = re.compile(r"(?:deleg|task|subagent|worker|browser)\)?_\)?\[")
# Literals the scan reports that are not renderings, each named so that it stays a decision.
_KNOWN_LITERALS = {
    ("src/hermes_realtime/integration/codex_access_credentials.py", "[a-z][a-z0-9_-]{0,63}"):
        "a Hermes custom provider name that shares the metric label shape",
    (
        "src/hermes_realtime/conversation/result_projection.py",
        r"(?:\b(?:task_|run_|deleg_)[A-Za-z0-9_.:-]+|",
    ): "the speech scrub of machine tokens; its run-token half is the owner's pattern",
    ("scripts/rehearse_desktop_mvp.py", r"\b(?:run|deleg)_[A-Za-z0-9]"):
        "a rehearsal leak count with its own, weaker, word-boundary rule",
}

# Exact browser rules the page validates; each literal must equal its rendering.
_BROWSER_EXACT = ("TASK_ID", "TURN_ID", "METRIC_LABEL", "WORKER_IDENTITY", "BROWSER_IDENTITY")
# Browser literals that state an owned rule differently. Removing the difference removes the row.
_BROWSER_DIVERGENT = {
    "/(?:deleg_[A-Za-z0-9_-]*|run_[A-Za-z0-9_-]{8,})/": (
        "PRIVATE_HANDLE", "the browser copy does not know the subagent_ prefix",
    ),
}
# Browser literals for facts this module does not own.
_BROWSER_UNOWNED = frozenset(
    {
        r"/^\p{P}$/u",
        r"/\s/u",
        "/^#bootstrap=([A-Za-z0-9_-]{43,128})$/",
        "/^rebind_[A-Za-z0-9-]{9,121}$/",
        "/^approval_[A-Za-z0-9_-]{8,128}$/",
        "/^[A-Za-z0-9_-]{8,128}$/",
        "/^(0|[1-9][0-9]*)$/",
        r"/[\u0000-\u001f\u007f-\u009f]/",
        "/^[0-9a-f]{64}$/",
        "/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/",
        "/^[a-z]{2}_[a-z]+$/",
        "/^proposal_[A-Za-z0-9_-]{8,64}$/",
        "/^[A-Za-z][A-Za-z0-9]{0,63}$/",
        "/deleg_/",
    }
)
_BROWSER_LITERAL = re.compile(
    r"(?<![\w/<*)\]])/(?![/*\s])((?:\\.|\[(?:\\.|[^\]\\\n])*\]|[^/\\\n\[])+)/([dgimsuvy]*)"
    r"(?=\s*(?:\.test\(|\.exec\(|\.source\b|[;,)]))"
)


def _grammar(pattern: re.Pattern[str]) -> str:
    return pattern.pattern.removesuffix(r"\Z")


def _canonical(source: str) -> str:
    """Drop anchors and order character-class members, so equal rules compare equal."""

    def ordered(match: re.Match[str]) -> str:
        body, tokens, index = match.group(1), [], 0
        while index < len(body):
            width = 2 if body[index] == "\\" else 3 if body[index + 1 : index + 2] == "-" else 1
            width = width if index + width <= len(body) else 1
            tokens.append(body[index : index + width])
            index += width
        return "[" + "".join(sorted(tokens)) + "]"

    for prefix in ("^", r"\A"):
        source = source.removeprefix(prefix)
    for suffix in ("$", r"\Z"):
        source = source.removesuffix(suffix)
    return re.sub(r"\[((?:\\.|[^\]\\])*)\]", ordered, source)


def _string_constants(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and type(node.value) is str
    ]


def _restatements(root: Path = ROOT) -> list[str]:
    owned = {_canonical(pattern.pattern): name for name, pattern in (EXACT | SEARCHED).items()}
    found = []
    for top, shapes in (("src/hermes_realtime", True), ("scripts", False)):
        for path in sorted((root / top).rglob("*.py")):
            if path == OWNER:
                continue
            relative = path.relative_to(root).as_posix()
            for line, value in _string_constants(path):
                if (relative, value) in _KNOWN_LITERALS:
                    continue
                name = owned.get(_canonical(value))
                if _NAMESPACE_RULE.search(value) is not None:
                    found.append(f"{relative}:{line}: restates a namespace rule: {value!r}")
                # Scripts reuse the general shapes for unrelated labels of their own.
                elif shapes and name is not None:
                    found.append(f"{relative}:{line}: restates {name}: {value!r}")
    return found


def _known_literals_present(root: Path = ROOT) -> bool:
    return all(
        value in {constant for _, constant in _string_constants(root / relative)}
        for relative, value in _KNOWN_LITERALS
    )


def _browser_literals(root: Path = ROOT) -> set[str]:
    literals = set()
    for path in sorted((root / "web/src").rglob("*.ts")):
        source = path.read_text(encoding="utf-8")
        assert "RegExp(" not in source, f"{path.name} builds a pattern this test cannot read"
        literals.update(
            f"/{match.group(1)}/{match.group(2)}" for match in _BROWSER_LITERAL.finditer(source)
        )
    return literals


def test_owned_rules_have_these_exact_spellings() -> None:
    assert identifiers.MAX_IDENTIFIER_CHARS == 128
    assert {name: pattern.pattern for name, pattern in (EXACT | SEARCHED).items()} == {
        "IDENTIFIER": r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z",
        "TASK_ID": r"task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,122}\Z",
        "DELEGATION_RUN_ID": r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]{0,121}\Z",
        "TURN_ID": r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z",
        "METRIC_LABEL": r"[a-z][a-z0-9_-]{0,63}\Z",
        "WORKER_IDENTITY": r"worker_[A-Za-z0-9_-]{8,64}\Z",
        "BROWSER_IDENTITY": r"browser_[a-f0-9]{16,64}\Z",
        "VOICE_CONVERSATION_ID": r"[A-Za-z0-9_-]{1,64}\Z",
        "PRIVATE_RUN_TOKEN": "deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*",
        "PRIVATE_HANDLE": (
            "(?:deleg_[A-Za-z0-9_-]*|run_[A-Za-z0-9_-]{8,}|subagent_[A-Za-z0-9_-]+)"
        ),
    }


def test_wire_types_render_the_owned_grammar() -> None:
    for wire_type, pattern, bound in (
        (IdentifierString, identifiers.IDENTIFIER, identifiers.MAX_IDENTIFIER_CHARS),
        (
            VoiceConversationId,
            identifiers.VOICE_CONVERSATION_ID,
            identifiers.MAX_VOICE_CONVERSATION_ID_CHARS,
        ),
    ):
        (constraints,) = wire_type.__metadata__
        assert constraints.pattern == f"^{_grammar(pattern)}$"
        assert constraints.max_length == bound


def test_the_owner_imports_nothing_from_the_package() -> None:
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    imported = {
        alias.name if isinstance(node, ast.Import) else f"{'.' * node.level}{node.module or ''}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported == {"__future__", "re", "typing"}


def test_no_python_module_restates_an_owned_rule() -> None:
    assert _restatements() == []
    assert _known_literals_present(), "a listed literal is gone; remove its row"


@pytest.mark.parametrize(
    ("relative", "source", "reported"),
    [
        ("src/hermes_realtime/x.py", 'A = r"task_[A-Za-z0-9]+"\n', "namespace rule"),
        ("scripts/x.py", 'import re\nA = re.compile(r"\\bdeleg_[a-z]*")\n', "namespace rule"),
        ("scripts/x.py", 'A = r"(?:run|deleg)_[a-z]|(?:task_|x)[a-z]"\n', "namespace rule"),
        ("src/hermes_realtime/x.py", 'A = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"\n', "IDENTIFIER"),
        ("src/hermes_realtime/x.py", 'A = r"[A-Za-z][-.0-9:A-Z_a-z]{0,127}\\Z"\n', "TURN_ID"),
        ("src/hermes_realtime/x.py", 'A = r"browser_[0-9a-f]{16,64}"\n', "namespace rule"),
        ("src/hermes_realtime/x.py", 'A = r"[-0-9A-Z_a-z]{1,64}"\n', "VOICE_CONVERSATION_ID"),
    ],
)
def test_a_restated_rule_is_reported(
    tmp_path: Path, relative: str, source: str, reported: str
) -> None:
    for directory in ("src/hermes_realtime", "scripts"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / relative).write_text(source, encoding="utf-8")
    (finding,) = _restatements(tmp_path)
    assert reported in finding


def test_every_browser_pattern_is_an_owned_rendering_or_a_listed_literal() -> None:
    rendered = {f"/^{_grammar(EXACT[name])}$/": name for name in _BROWSER_EXACT}
    found = _browser_literals()

    assert found - set(rendered) - set(_BROWSER_DIVERGENT) - _BROWSER_UNOWNED == set(), (
        "classify the new browser pattern: render an owned rule or list the literal"
    )
    assert (set(rendered) | set(_BROWSER_DIVERGENT) | _BROWSER_UNOWNED) - found == set(), (
        "a listed browser pattern is gone or no longer matches its owned rule"
    )
    for literal, (owner, reason) in _BROWSER_DIVERGENT.items():
        assert literal != f"/{SEARCHED[owner].pattern}/", reason


def test_a_changed_browser_pattern_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "web/src"
    source.mkdir(parents=True)
    (source / "a.ts").write_text(
        "const A = /^task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,121}$/;\n"
        'if (!/^[a-z]+$/.test(x) || y.replace(/\\s+/g, " ") === "a/b/c") {}\n',
        encoding="utf-8",
    )
    assert _browser_literals(tmp_path) == {
        "/^task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,121}$/", "/^[a-z]+$/", r"/\s+/g",
    }
