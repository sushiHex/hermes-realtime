"""Isolated build-tool worker; its output alone is never qualification evidence."""

from __future__ import annotations

import sys

_RUNTIME_PURPOSES = {
    "realtime_windows_direct_runtime",
    "realtime_windows_sdist_built_runtime",
    "hermes_v020_pluginmanager_runtime",
}


def main(arguments: list[str]) -> int:
    if (
        len(arguments) != 5
        or arguments[0] not in {"imports", "wheel", "sdist"} | _RUNTIME_PURPOSES
        or any(not value or len(value) > 4096 or "\x00" in value for value in arguments)
        or not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode)
        or sys.prefix != sys.base_prefix
    ):
        return 2
    try:
        import importlib
        import json
        import os
        from pathlib import Path

        kind, package_name, source_name, output_name, report_name = arguments
        packages = Path(package_name).resolve(strict=True)
        source = Path(source_name).resolve(strict=True)
        output = Path(output_name).resolve(strict=True)
        runtime = Path(sys.base_prefix).resolve(strict=True)
        worker = Path(__file__).resolve(strict=True)
        if not all(path.is_dir() for path in (packages, source, output)) or not Path(
            sys.executable
        ).resolve(strict=True).is_relative_to(runtime):
            return 2
        sys.path.insert(0, str(packages))
        names = (
            (
                "hermes_realtime",
                "hermes_realtime.host_launcher",
                "hermes_realtime.launcher",
                "hermes_realtime.hermes_plugin",
            )
            if kind in _RUNTIME_PURPOSES
            else ("hatchling", "packaging", "pathspec", "pluggy", "trove_classifiers")
        )
        for name in names:
            module = importlib.import_module(name)
            origin = module.__file__
            if not isinstance(origin, str) or not Path(origin).resolve(strict=True).is_relative_to(
                packages
            ):
                return 1
        if (
            kind in _RUNTIME_PURPOSES
            and getattr(sys.modules["hermes_realtime"], "__version__", None) != "0.0.3"
        ):
            return 1
        artifact = None
        if kind in {"wheel", "sdist"}:
            build = importlib.import_module("hatchling.build")

            os.chdir(source)
            # The admitted Hatchling 1.27.0 default, made explicit for each build.
            os.environ["SOURCE_DATE_EPOCH"] = "1580601600"
            if kind == "wheel":
                artifact = build.build_wheel(str(output))
                expected = "hermes_realtime-0.0.3-py3-none-any.whl"
            else:
                artifact = build.build_sdist(str(output))
                expected = "hermes_realtime-0.0.3.tar.gz"
            if artifact != expected or not (output / expected).is_file():
                return 1
        origins = []
        for module in list(sys.modules.values()):
            origin = getattr(module, "__file__", None)
            if origin is None:
                continue
            if not isinstance(origin, str):
                return 1
            origins.append(Path(origin).resolve(strict=True))
        if not all(
            path == worker or path.is_relative_to(packages) or path.is_relative_to(runtime)
            for path in origins
        ):
            return 1
        if not all(
            Path(path).resolve().is_relative_to(packages)
            or Path(path).resolve().is_relative_to(runtime)
            for path in sys.path
        ):
            return 1
        observation = {
            "version": 1,
            "kind": kind,
            "pid": os.getpid(),
            "imports": len(names),
            "file_origins": len(origins),
            "source_fallback": False,
            "artifact": artifact,
        }
        with Path(report_name).open("x", encoding="utf-8", newline="\n") as report:
            json.dump(observation, report, sort_keys=True, separators=(",", ":"))
            report.write("\n")
        return 0
    except Exception:
        # Paths, installed module diagnostics and backend exceptions stay private.
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
