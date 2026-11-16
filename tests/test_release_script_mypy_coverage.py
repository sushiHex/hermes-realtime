from pathlib import Path
from runpy import run_path

ROOT = Path(__file__).resolve().parents[1]


def _gate() -> dict[str, object]:
    return run_path(str(ROOT / "scripts" / "release_gate.py"))


def _globals(gate: dict[str, object]) -> dict[str, object]:
    return gate["script_mypy_targets"].__globals__  # type: ignore[attr-defined, no-any-return]


def _observed_command(gate: dict[str, object], root: Path) -> tuple[str, ...]:
    calls: list[tuple[str, ...]] = []

    def observe_run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd, env
        calls.append(command)

    _globals(gate)["run"] = observe_run
    try:
        gate["run_script_mypy"](root, {})  # type: ignore[operator]
    finally:
        del _globals(gate)["run"]
    assert len(calls) == 1
    return calls[0]


def test_every_script_is_type_checked_or_justifiably_excluded() -> None:
    gate = _gate()
    excluded = gate["SCRIPT_MYPY_EXCLUDED"]
    assert isinstance(excluded, dict)
    on_disk = {f"scripts/{path.name}" for path in (ROOT / "scripts").glob("*.py")}

    command = _observed_command(gate, ROOT)
    checked = {argument for argument in command if argument.startswith("scripts/")}

    assert command[:6] == ("uv", "run", "--frozen", "--group", "dev", "mypy")
    assert "--follow-imports=skip" in command
    assert checked | set(excluded) == on_disk
    assert not checked & set(excluded)
    assert all(type(why) is str and why and "\n" not in why for why in excluded.values())
    assert "scripts/stage_lookup_policy.py" in checked
    assert "scripts/release_gate.py" in checked
    assert gate["stale_script_mypy_exclusions"](ROOT) == ()  # type: ignore[operator]


def test_stale_exclusion_is_reported(tmp_path: Path) -> None:
    gate = _gate()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "present.py").write_text("", encoding="utf-8")
    _globals(gate)["SCRIPT_MYPY_EXCLUDED"] = {"scripts/gone.py": "removed script"}

    assert gate["stale_script_mypy_exclusions"](tmp_path) == ("scripts/gone.py",)  # type: ignore[operator]


def test_new_script_is_checked_without_registration(tmp_path: Path) -> None:
    gate = _gate()
    (tmp_path / "scripts").mkdir()
    for name in ("brand_new.py", "excused.py"):
        (tmp_path / "scripts" / name).write_text("", encoding="utf-8")
    _globals(gate)["SCRIPT_MYPY_EXCLUDED"] = {"scripts/excused.py": "known debt"}

    assert gate["script_mypy_targets"](tmp_path) == ("scripts/brand_new.py",)  # type: ignore[operator]
    command = _observed_command(gate, tmp_path)
    assert "scripts/brand_new.py" in command
    assert "scripts/excused.py" not in command
