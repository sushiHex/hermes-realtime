"""Isolated candidate benchmark worker; output bytes alone grant no authority."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def _run(
    benchmark: ModuleType,
    machine_path: Path,
    report_path: Path,
    observation_path: Path,
    *,
    file_origins: int = 1,
) -> None:
    import os
    import platform

    machine = benchmark.collect_live_machine_manifest()
    machine_sha256 = benchmark.write_machine_manifest_atomic(machine_path, machine)
    machine_bytes = machine_path.read_bytes()
    profiles, saturation, payload_bytes, payload_sha256 = benchmark.run_benchmark()
    report = benchmark.make_report(
        machine_bytes=machine_bytes,
        profiles=profiles,
        saturation=saturation,
        payload_bytes=payload_bytes,
        payload_sha=payload_sha256,
        thresholds_required=True,
    )
    report_sha256 = benchmark.write_report_atomic(
        report_path,
        report,
        machine_manifest_sha256=machine_sha256,
        payload_canonical_bytes=payload_bytes,
        payload_sha256=payload_sha256,
    )
    module_file = getattr(benchmark, "__file__", None)
    if not isinstance(module_file, str):
        raise ValueError("benchmark module origin is unavailable")
    script_sha256 = benchmark.sha256_bytes(Path(module_file).resolve().read_bytes())
    observation = benchmark.canonical_json_bytes(
        {
            "benchmarkReportSha256": report_sha256,
            "benchmarkScriptSha256": script_sha256,
            "fileOrigins": file_origins,
            "machineManifestSha256": machine_sha256,
            "pid": os.getpid(),
            "pythonVersion": platform.python_version(),
            "sourceFallback": False,
            "version": 1,
        }
    )
    with observation_path.open("xb") as stream:
        stream.write(observation)
        stream.flush()
        os.fsync(stream.fileno())
    if observation_path.read_bytes() != observation:
        raise ValueError("benchmark observation reopen mismatch")


def main(arguments: list[str]) -> int:
    if (
        len(arguments) != 4
        or any(not value or len(value) > 4096 or "\x00" in value for value in arguments)
        or not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode)
        or sys.prefix != sys.base_prefix
    ):
        return 2
    try:
        import importlib

        source_name, machine_name, report_name, observation_name = arguments
        source = Path(source_name).resolve(strict=True)
        worker = Path(__file__).resolve(strict=True)
        snapshot = worker.parents[1]
        runtime = Path(sys.base_prefix).resolve(strict=True)
        outputs = tuple(Path(name).resolve(strict=False) for name in arguments[1:])
        if (
            not source.is_dir()
            or source != snapshot / "src"
            or not Path(sys.executable).resolve(strict=True).is_relative_to(runtime)
            or len({path.parent for path in outputs}) != 1
            or not outputs[0].parent.is_dir()
            or any(path.exists() or path.is_symlink() for path in outputs)
        ):
            return 2
        sys.path.insert(0, str(snapshot))
        sys.path.insert(0, str(source))
        benchmark = importlib.import_module("scripts.benchmark_evidence_admission")
        module_file = getattr(benchmark, "__file__", None)
        expected_module = snapshot / "scripts" / "benchmark_evidence_admission.py"
        if (
            not isinstance(module_file, str)
            or Path(module_file).resolve(strict=True) != expected_module
        ):
            return 1
        # Import the candidate modules before timing so every measured operation
        # uses the source-selected implementation in the immutable snapshot.
        importlib.import_module("hermes_realtime.evidence.models")
        importlib.import_module("hermes_realtime.evidence.admission")
        origins = []
        for module in tuple(sys.modules.values()):
            origin = getattr(module, "__file__", None)
            if origin is None:
                continue
            if not isinstance(origin, str):
                return 1
            origins.append(Path(origin).resolve(strict=True))
        if not all(
            path.is_relative_to(snapshot) or path.is_relative_to(runtime) for path in origins
        ):
            return 1
        if not all(
            Path(path).resolve().is_relative_to(snapshot)
            or Path(path).resolve().is_relative_to(runtime)
            for path in sys.path
        ):
            return 1
        _run(benchmark, *outputs, file_origins=len(origins))
        return 0
    except Exception:
        # Native machine details, paths, and benchmark failures stay private.
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
