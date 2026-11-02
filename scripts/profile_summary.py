"""Print a fixed allowlist of profile model-routing metadata."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_MAX_CONFIG_BYTES = 65_536
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_.:/@+-]{1,128}\Z")


def _mapping(value: object) -> dict[str, Any]:
    return value if type(value) is dict else {}


def _label(value: object) -> str | None:
    if type(value) is str and _SAFE_LABEL.fullmatch(value):
        return value
    return None


def summarize_config(config: object) -> dict[str, object]:
    root = _mapping(config)
    model = _mapping(root.get("model"))
    plugins = _mapping(root.get("plugins"))
    entries = _mapping(plugins.get("entries"))
    lookup = _mapping(entries.get("ordinary-lookup-experimental"))
    settings = _mapping(lookup.get("settings"))
    presentation = _mapping(settings.get("presentation"))

    native_model = _label(model.get("default"))
    native_provider = _label(model.get("provider"))
    native_mode = _label(model.get("api_mode"))
    presentation_model = _label(presentation.get("model"))
    presentation_provider = _label(presentation.get("provider"))
    presentation_mode = _label(presentation.get("api_mode"))
    return {
        "version": 1,
        "config_read": True,
        "native_model": native_model,
        "native_provider": native_provider,
        "native_mode": native_mode,
        "presentation_model": presentation_model,
        "presentation_provider": presentation_provider,
        "presentation_mode": presentation_mode,
        "same_model": native_model is not None
        and native_model == presentation_model,
        "same_provider": native_provider is not None
        and native_provider == presentation_provider,
        "same_mode": native_mode is not None and native_mode == presentation_mode,
    }


def _read_config(path: Path) -> object:
    with path.open("rb") as stream:
        raw = stream.read(_MAX_CONFIG_BYTES + 1)
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError("config_bound")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    import yaml  # type: ignore[import-untyped]

    return yaml.safe_load(raw)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(
            json.dumps({"version": 1, "config_read": False}, separators=(",", ":"))
        )
        return 2
    try:
        summary = summarize_config(_read_config(Path(args[0])))
    except Exception:
        summary = {"version": 1, "config_read": False}
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if summary["config_read"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
