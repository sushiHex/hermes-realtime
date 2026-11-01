"""Explicit source-bound replay for one owned diagnostic; never a live lookup."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _refuse(category: str) -> Any:
    print("[lookup-frozen-search] " + json.dumps({"category": category}, separators=(",", ":")),
          flush=True)
    raise ValueError(category)


def _pairs(items: Any) -> Any:
    if len(items) != len(dict(items)):
        raise ValueError("frozen_search_schema")
    return dict(items)


class FrozenSearch:
    def __init__(self, result: Any, digest: str, objective: str) -> None:
        self.result, self.digest, self.objective = result, digest, objective.strip()
        self.searches = 0
        self.refused = self.extraction_refused = False
        self.lock = threading.Lock()

    def summary(self) -> dict[str, Any]:
        return {"mode": "frozen", "sha256": self.digest, "searches": self.searches,
                "extraction_refused": self.extraction_refused, "refused": self.refused}

    def search(self, query: Any, limit: Any = 5) -> Any:
        with self.lock:
            if type(query) is not str or query != self.objective or type(limit) is not int or not (
                len(self.result["data"]["web"]) <= limit <= 10
            ):
                self.refused = True
                _refuse("frozen_search_scope")
            if self.searches:
                self.refused = True
                _refuse("frozen_search_spent")
            self.searches = 1
            return copy.deepcopy(self.result)

    def extract(self) -> Any:
        with self.lock:
            self.extraction_refused = self.refused = True
            _refuse("frozen_extract_unavailable")


def load_frozen_search(path: Any, digest: Any, objective: Any) -> FrozenSearch | None:
    if path is None and digest is None:
        return None
    if (not isinstance(path, (str, Path)) or not Path(path).is_absolute()
            or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(objective) is not str or not objective.strip() or len(objective) > 1024):
        _refuse("frozen_search_config")
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(65537)
    except OSError:
        _refuse("frozen_search_unreadable")
    if len(raw) > 65536:
        _refuse("frozen_search_bound")
    if hashlib.sha256(raw).hexdigest() != digest:
        _refuse("frozen_search_digest")
    try:
        result = json.loads(raw, object_pairs_hook=_pairs)
    except (ValueError, RecursionError):
        _refuse("frozen_search_schema")
    if (type(result) is not dict or set(result) != {"success", "data"}
            or result["success"] is not True or type(result["data"]) is not dict
            or set(result["data"]) != {"web"} or type(result["data"]["web"]) is not list
            or not 1 <= len(result["data"]["web"]) <= 10):
        _refuse("frozen_search_schema")
    for index, row in enumerate(result["data"]["web"], 1):
        if (type(row) is not dict or set(row) != {"url", "title", "description", "position"}
                or any(type(row[name]) is not str for name in ("url", "title", "description"))
                or type(row["position"]) is not int or row["position"] != index):
            _refuse("frozen_search_schema")
        try:
            url = urlsplit(row["url"])
        except ValueError:
            _refuse("frozen_search_schema")
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.fragment):
            _refuse("frozen_search_schema")
    return FrozenSearch(result, digest, objective)
