"""Print routing presence and equality, never raw profile values."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_MAX_CONFIG_BYTES = 65_536


def _mapping(value: object) -> dict[str, Any]:
    return value if type(value) is dict else {}


def _routing_value(value: object) -> str | None:
    if type(value) is str and 0 < len(value) <= 128:
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

    native_model = _routing_value(model.get("default"))
    native_provider = _routing_value(model.get("provider"))
    native_mode = _routing_value(model.get("api_mode"))
    presentation_model = _routing_value(presentation.get("model"))
    presentation_provider = _routing_value(presentation.get("provider"))
    presentation_mode = _routing_value(presentation.get("api_mode"))
    return {
        "version": 2,
        "config_read": True,
        "native_model_configured": native_model is not None,
        "native_provider_configured": native_provider is not None,
        "native_mode_configured": native_mode is not None,
        "presentation_model_configured": presentation_model is not None,
        "presentation_provider_configured": presentation_provider is not None,
        "presentation_mode_configured": presentation_mode is not None,
        "same_model": native_model is not None and native_model == presentation_model,
        "same_provider": native_provider is not None and native_provider == presentation_provider,
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
        print(json.dumps({"version": 2, "config_read": False}, separators=(",", ":")))
        return 2
    try:
        summary = summarize_config(_read_config(Path(args[0])))
    except Exception:
        summary = {"version": 2, "config_read": False}
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if summary["config_read"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
