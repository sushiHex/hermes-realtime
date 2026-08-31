"""Checked-in disclosure and browser asset parity contract."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from runpy import run_path

ROOT = Path(__file__).resolve().parents[1]
DISCLOSURE = ROOT / "src" / "hermes_realtime" / "evidence" / "disclosure_v1.txt"
MANIFEST = ROOT / "src" / "hermes_realtime" / "evidence" / "disclosure_manifest_v1.json"
STATIC = ROOT / "src" / "hermes_realtime" / "client" / "static"


def _visible_text(markup: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


def test_canonical_disclosure_is_visible_accessible_and_hash_bound_to_packaged_assets() -> None:
    disclosure = DISCLOSURE.read_bytes()
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert document == {
        "assets": {
            name: hashlib.sha256((STATIC / relative).read_bytes()).hexdigest()
            for name, relative in {
                "app.js": "assets/app.js",
                "index.html": "index.html",
                "styles.css": "assets/styles.css",
            }.items()
        },
        "consentVersion": "realtime-evidence-consent-v1",
        "disclosureDigest": hashlib.sha256(
            b"realtime-evidence-consent-v1\0" + disclosure
        ).hexdigest(),
        "disclosureSha256": hashlib.sha256(disclosure).hexdigest(),
        "version": 1,
    }

    index = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.search(
        r'<section id="evidence-disclosure"[^>]*aria-labelledby="evidence-disclosure-title">',
        index,
    )
    assert '<h2 id="evidence-disclosure-title">Evidence capture disclosure</h2>' in index
    section = re.search(
        r'<section id="evidence-disclosure"[^>]*>(.*?)</section>', index, re.DOTALL
    )
    assert section is not None
    assert "hidden" not in section.group(0)
    assert "aria-hidden" not in section.group(0)
    visible = _visible_text(section.group(1))
    for paragraph in disclosure.decode("utf-8").split("\n\n"):
        assert re.sub(r"\s+", " ", paragraph).strip() in visible

    required = (
        "final microphone transcript text",
        "final typed input text",
        "generated assistant segment text",
        "transport-confirmed assistant text",
        "sensitive personal information",
        "without encryption at rest",
        "retention period and bounded storage quota",
        "current consent epoch",
        "older closed epochs remain until their retention deadline or an explicit full purge",
        "cannot prove zeroization",
        "restart recovery",
        "does not prove physical speaker identity",
        "does not learn",
    )
    disclosure_text = disclosure.decode("utf-8")
    for phrase in required:
        assert phrase in disclosure_text


def test_disclosure_source_and_packaged_web_assets_are_byte_identical_after_build() -> None:
    assert (ROOT / "web" / "index.html").read_bytes() == (STATIC / "index.html").read_bytes()
    assert (ROOT / "web" / "src" / "styles.css").read_bytes() == (
        STATIC / "assets" / "styles.css"
    ).read_bytes()


def test_release_gate_retains_the_disclosure_and_hash_manifest_in_sdists() -> None:
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))

    required = release_gate["required_sdist_paths"]()
    assert "src/hermes_realtime/evidence/disclosure_v1.txt" in required
    assert "src/hermes_realtime/evidence/disclosure_manifest_v1.json" in required


def test_release_gate_validates_the_canonical_disclosure_manifest() -> None:
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))

    release_gate["validate_disclosure_manifest"](ROOT / "src")


def test_release_gate_requires_sdist_built_wheel_disclosure_parity() -> None:
    script = (ROOT / "scripts" / "release_gate.py").read_text(encoding="utf-8")

    assert "def build_sdist_wheel" in script
    assert "sdist-built wheel disclosure resource differs from source" in script
