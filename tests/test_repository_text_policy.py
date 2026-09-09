from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE_DOC = _ROOT / "docs" / "evidence-capture.md"
_HEARING_TERM = re.compile(r"\b(?:hear|hears|heard|hearing|audible|audibility)\b", re.IGNORECASE)
_HEARING_NEGATION = re.compile(
    r"\b(?:no|not|never|cannot|can't|doesn't|does not|do not|without|neither)\b",
    re.IGNORECASE,
)


def _read_evidence_policy() -> str:
    assert _EVIDENCE_DOC.is_file(), (
        "Task 1 requires docs/evidence-capture.md to freeze the evidence-only "
        "threat and scope contract"
    )
    return _EVIDENCE_DOC.read_text(encoding="utf-8")


def _python_import_targets(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            targets.add(module)
            targets.update(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
    return targets


def _is_affirmative_hearing_claim(text: str) -> bool:
    normalized = " ".join(text.split())
    clauses = re.split(
        r"[.!?;:]|\b(?:although|and|but|however|whereas|while)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    return any(
        _HEARING_TERM.search(clause) is not None
        and _HEARING_NEGATION.search(clause) is None
        for clause in clauses
    )


def test_evidence_capture_task_one_contract_is_documented() -> None:
    policy = _read_evidence_policy()
    normalized_policy = " ".join(policy.split())
    required_sections = {
        "## Status and activation",
        "## Qualification boundary",
        "## Threat boundary",
        "## Disclosure and consent",
        "## Queue boundary",
        "## Lifecycle boundary",
        "## Platform boundary",
        "## Purge boundary",
        "## Physical-observation boundary",
        "## Plugin boundary",
        "## No-learning boundary",
    }
    required_contract = {
        "Evidence capture is disabled by default.",
        "Operator enablement is not consent.",
        (
            "On Linux, evidence enable, status, and purge requests fail closed "
            "as `unsupported_platform` before any evidence path is created or opened."
        ),
        (
            "It captures no raw audio and has no profile, learning, or reviewer "
            "import path."
        ),
        (
            "The runtime cannot prove physical speaker identity, typed authorship, "
            "browser playout, audibility, hearing, comprehension, agreement, or truth."
        ),
        (
            "Slice 0 ships no web-app manifest, service worker, offline cache, update "
            "policy, permission-retention policy, or installed PWA behavior."
        ),
        (
            "Windows service installation, Windows Scheduled Task installation, Session 0, "
            "background start, stop/restart policy, reboot recovery, deployment, publishing, "
            "Phase 1, and every learning phase are excluded."
        ),
        (
            "The spool does not authenticate bytes against another process running "
            "as the same OS user."
        ),
        (
            "The package-authored disclosure cannot be replaced by CLI, environment, "
            "operator configuration, or model output."
        ),
        "Foreground conversation never waits for evidence persistence.",
        (
            "Capacity failure drops or taints capture while the conversation "
            "operation continues."
        ),
        "Revocation and owner drain use independent reserved lanes.",
        (
            "Consent is not inherited across process restart, reconnect replacement, "
            "media-incarnation replacement, disclosure change, or revocation."
        ),
        (
            "Purge targets only the manifest-owned database artifacts; it never uses "
            "a glob or recursive delete."
        ),
        (
            "VACUUM and byte-absence checks are not SSD forensic erasure, and Python "
            "strings are not securely zeroized."
        ),
        "Entry-point discovery does not activate the plugin or authorize capture.",
        (
            "Compatibility testing is limited to a supplied local Hermes v0.20 "
            "source `PluginManager` harness."
        ),
        "No Hermes wheel or public release-version compatibility is claimed.",
        (
            "Entry-point discovery provides packaging compatibility only; no v0.20 "
            "in-process bridge dispatch is claimed."
        ),
        (
            "Captured text cannot affect conversation, inference, context, routing, "
            "tools, policy, prompts, tasks, canonical state, memory, skills, code, "
            "Curator, SessionDB, or a profile."
        ),
    }

    missing_sections = sorted(required_sections.difference(policy.splitlines()))
    missing_contract = sorted(
        statement for statement in required_contract if statement not in normalized_policy
    )
    assert not missing_sections, f"missing evidence policy sections: {missing_sections}"
    assert not missing_contract, f"missing evidence policy statements: {missing_contract}"


def test_readme_states_the_public_hermes_compatibility_boundary() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    required = {
        (
            "This repository does not claim compatibility with a publicly "
            "released Hermes Agent version."
        ),
        (
            "On the locally exercised Hermes v0.20 source surface, enabling "
            "the entry point provides discovery and packaging compatibility only; "
            "in-process bridge dispatch remains fail-closed."
        ),
        (
            "The functional work-dispatch route is the separately launched, "
            "authenticated `hermes-realtime-host`."
        ),
    }
    missing = sorted(statement for statement in required if statement not in readme)
    assert not missing, f"missing public Hermes compatibility boundaries: {missing}"


def test_package_metadata_states_the_project_is_unofficial() -> None:
    # The README's non-affiliation callout travels as the PyPI long description,
    # but this one-line summary is what a package index search result and
    # `pip show` render on their own. It is also the only place the name appears
    # without an owner namespace to disambiguate it, so the qualifier is pinned
    # here rather than left to drift.
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    summary = project["project"]["description"]

    # Pinned as an exact literal, and deliberately duplicated in
    # tests/test_qualification_package.py rather than shared through a constant:
    # a shared constant would move with an edit and both assertions would still
    # pass, which is the opposite of what a policy pin is for.
    assert summary == (
        "Unofficial realtime LiveKit conversation runtime for Hermes Agent"
    )


def test_evidence_capture_readme_links_the_status_boundary() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Full Slice 0 qualification is incomplete." in readme
    assert "It remains disabled by default; operator enablement is not consent." in readme
    assert "[Evidence capture boundary](docs/evidence-capture.md)" in readme
    assert "[Implementation status](docs/implementation-status.md)" in readme


def test_public_search_is_documented_as_default_off_and_separately_consented() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    notices = (_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    normalized_notices = " ".join(notices.split())
    required_readme = {
        "Public search is disabled by default.",
        "Operator enablement is not participant consent.",
        "`--enable-public-search`",
        "Bing Search RSS and, for outcome-shaped queries,",
        "separate browser-visible consent",
    }
    missing = sorted(statement for statement in required_readme if statement not in readme)
    assert not missing, f"missing public-search boundaries: {missing}"
    required_notices = {
        "The operator must explicitly pass `--enable-public-search`",
        "A participant must separately accept the browser-visible public-search disclosure",
        (
            "Revocation cannot cancel a lookup admitted while consent was active or recall a "
            "request that was already sent."
        ),
    }
    missing_notices = sorted(
        statement for statement in required_notices if statement not in normalized_notices
    )
    assert not missing_notices, f"missing outbound-search notices: {missing_notices}"


def test_runtime_docs_do_not_describe_ddgs_as_the_production_lookup() -> None:
    prohibited = {
        "docs/hermes-bridge.md": ["bounded DDGS evidence"],
        "docs/local-livekit.md": [
            "Codex/DDGS retrieval",
            "bounded DDGS foreground-evidence",
            "safe final-transcript DDGS lookup",
        ],
        "docs/release-gates.md": ["final-only DDGS"],
        "docs/source-backed-latency.md": [
            "final transcript starts one DDGS lookup",
            "coordinator lookup or direct DDGS lookup",
            "synchronous DDGS call",
            "stable-partial DDGS prefetch",
            "final-transcript DDGS only",
        ],
    }
    violations: list[str] = []
    for relative_path, stale_claims in prohibited.items():
        text = (_ROOT / relative_path).read_text(encoding="utf-8")
        violations.extend(
            f"{relative_path}: {claim}" for claim in stale_claims if claim in text
        )
    assert not violations, f"stale production DDGS claims: {violations}"


def test_evidence_capture_docs_make_no_affirmative_hearing_claim() -> None:
    policy_text = _read_evidence_policy()
    readme_text = (_ROOT / "README.md").read_text(encoding="utf-8")
    violations = [
        path
        for path, text in (("README.md", readme_text), (str(_EVIDENCE_DOC), policy_text))
        if _is_affirmative_hearing_claim(text)
    ]
    assert not violations, f"affirmative physical hearing claims: {violations}"


def test_hearing_claim_matcher_distinguishes_claims_from_disclaimers() -> None:
    assert _is_affirmative_hearing_claim("The output was heard by a participant.")
    assert _is_affirmative_hearing_claim("The listener can hear the response.")
    assert _is_affirmative_hearing_claim("The listener hears the response.")
    assert _is_affirmative_hearing_claim(
        "Audio is not stored, but the listener\nhears the response."
    )
    assert not _is_affirmative_hearing_claim("No person heard the response.")
    assert not _is_affirmative_hearing_claim("The system cannot prove the user heard output.")


def test_evidence_capture_source_has_no_forbidden_import_path() -> None:
    evidence_root = _ROOT / "src" / "hermes_realtime" / "evidence"
    forbidden_target = re.compile(
        r"(?:^|[._])(?:raw_audio|profile|profiles|memory|memories|skill|skills|curator|"
        r"session_db|sessiondb|learning|learned|review|reviewer|reviewers)(?:[._]|$)"
    )
    forbidden_runtime_boundary = re.compile(
        r"^(?:hermes_realtime\.)?(?:speech|livekit|providers)(?:\.|$)|"
        r"^(?:hermes_agent|hermes_cli|gateway)(?:\.|$)"
    )
    evidence_target = re.compile(r"(?:^|\.)evidence(?:\.|$)")
    forbidden_importer = re.compile(
        r"(?:^|[./_-])(?:raw_audio|profile|profiles|memory|memories|skill|skills|curator|"
        r"session_db|sessiondb|learning|learned|review|reviewer|reviewers)(?:[./_-]|$)"
    )
    violations: list[str] = []

    for path in sorted((_ROOT / "src" / "hermes_realtime").rglob("*.py")):
        targets = _python_import_targets(path)
        if evidence_root in path.parents:
            violations.extend(
                f"{path.relative_to(_ROOT)} imports {target}"
                for target in sorted(targets)
                if (
                    forbidden_target.search(
                        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", target).lower()
                    )
                    or forbidden_runtime_boundary.search(target)
                )
            )

        importer_name = path.relative_to(_ROOT).as_posix()
        if forbidden_importer.search(importer_name):
            violations.extend(
                f"{path.relative_to(_ROOT)} imports {target}"
                for target in sorted(targets)
                if evidence_target.search(target)
            )

    assert not violations, "forbidden evidence import paths:\n" + "\n".join(violations)


def test_web_build_inputs_and_generated_text_assets_are_forced_to_lf() -> None:
    root = _ROOT
    required_rules = {
        "/web/**/*.ts text eol=lf",
        "/web/**/*.css text eol=lf",
        "/web/**/*.html text eol=lf",
        "/web/*.mjs text eol=lf",
        "/src/hermes_realtime/client/static/**/*.js text eol=lf",
        "/src/hermes_realtime/client/static/**/*.css text eol=lf",
        "/src/hermes_realtime/client/static/**/*.html text eol=lf",
    }
    configured_rules = {
        line.strip()
        for line in (root / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert required_rules <= configured_rules

    # Hermetic release candidates are created with `git archive` and therefore
    # have no repository metadata. In a checkout, retain the stronger effective
    # attribute assertion so a later conflicting rule cannot silently win.
    if not (root / ".git").exists():
        return

    paths = (
        "web/src/main.ts",
        "web/src/styles.css",
        "web/index.html",
        "web/build.mjs",
        "src/hermes_realtime/client/static/assets/app.js",
        "src/hermes_realtime/client/static/assets/styles.css",
        "src/hermes_realtime/client/static/index.html",
    )
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", *paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    attributes = {
        line.split(": ", maxsplit=2)[0]: line.rsplit(": ", maxsplit=1)[-1]
        for line in result.stdout.splitlines()
    }
    assert attributes == {path: "lf" for path in paths}
