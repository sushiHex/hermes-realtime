"""Carry only pytest's requested Git path across independently loaded helpers."""

from pathlib import Path

_requested_executable: Path | None = None


def set_requested_native_git(executable: Path | None) -> None:
    global _requested_executable
    _requested_executable = executable


def requested_native_git() -> Path | None:
    return _requested_executable
