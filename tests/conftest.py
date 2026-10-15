"""Explicit transport denial for synthetic Ollama streaming tests."""

from pathlib import Path

import pytest

from hermes_realtime.providers.ollama_models import OllamaModelClient
from tests.support import native_git_selection


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--native-git-executable", type=Path, default=None)


def pytest_configure(config: pytest.Config) -> None:
    native_git_selection.set_requested_native_git(config.getoption("native_git_executable"))


def pytest_unconfigure(config: pytest.Config) -> None:
    native_git_selection.set_requested_native_git(None)


@pytest.fixture
def deny_uninjected_ollama_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def denied(self: OllamaModelClient, endpoint: str, body: object) -> bytes:
        raise AssertionError("synthetic Ollama tests must inject a metadata reader")

    monkeypatch.setattr(OllamaModelClient, "_read_request", denied)
