from __future__ import annotations

import builtins
import importlib
import sys


def test_provider_package_import_does_not_import_numpy(monkeypatch: object) -> None:
    original_import = builtins.__import__

    def reject_numpy(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy" or name.startswith("numpy."):
            raise AssertionError("base provider package imported numpy")
        return original_import(name, *args, **kwargs)

    for name in tuple(sys.modules):
        if name == "hermes_realtime.providers" or name.startswith("hermes_realtime.providers."):
            sys.modules.pop(name)
    monkeypatch.setattr(builtins, "__import__", reject_numpy)

    package = importlib.import_module("hermes_realtime.providers")

    assert "PlaybackEchoGuard" in package.__all__