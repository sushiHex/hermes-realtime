from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from pathlib import Path

from hermes_realtime.search_egress import (
    SEARCH_EGRESS_CONSENT_VERSION,
    SEARCH_EGRESS_DISCLOSURE_BYTES,
    SEARCH_EGRESS_DISCLOSURE_DIGEST,
)

_ROOT = Path(__file__).resolve().parents[1]


class _DisclosureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._container_depth = 0
        self._paragraph_depth = 0
        self._current: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id") == "search-egress-disclosure-copy":
            self._container_depth = 1
            return
        if self._container_depth:
            self._container_depth += 1
            if tag == "p":
                self._paragraph_depth = self._container_depth
                self._current = []

    def handle_endtag(self, tag: str) -> None:
        if not self._container_depth:
            return
        if tag == "p" and self._paragraph_depth == self._container_depth:
            self.paragraphs.append(" ".join("".join(self._current).split()))
            self._paragraph_depth = 0
            self._current = []
        self._container_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._paragraph_depth:
            self._current.append(data)


def _visible_disclosure(path: Path) -> bytes:
    parser = _DisclosureParser()
    parser.feed(path.read_text(encoding="utf-8"))
    assert len(parser.paragraphs) == 4
    return ("\n\n".join(parser.paragraphs) + "\n").encode("utf-8")


def test_search_egress_digest_binds_the_exact_visible_source_and_packaged_disclosure() -> None:
    assert SEARCH_EGRESS_CONSENT_VERSION == "realtime-search-egress-consent-v1"
    assert (
        b"Bing Search RSS and, for outcome-shaped queries, Google News RSS"
        in SEARCH_EGRESS_DISCLOSURE_BYTES
    )
    assert (
        hashlib.sha256(SEARCH_EGRESS_DISCLOSURE_BYTES).hexdigest()
        == SEARCH_EGRESS_DISCLOSURE_DIGEST
    )
    assert _visible_disclosure(_ROOT / "web" / "index.html") == SEARCH_EGRESS_DISCLOSURE_BYTES
    assert (
        _visible_disclosure(
            _ROOT / "src" / "hermes_realtime" / "client" / "static" / "index.html"
        )
        == SEARCH_EGRESS_DISCLOSURE_BYTES
    )
