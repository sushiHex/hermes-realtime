"""The archived benchmark worker closes its invocation and output protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace as Row


def test_worker_refuses_nonisolated_direct_call(tmp_path: Path) -> None:
    from scripts.qualification_benchmark_worker import main

    assert main([str(tmp_path)] * 5) == 2


def test_worker_emits_machine_report_and_process_bound_observation(tmp_path: Path) -> None:
    from scripts import benchmark_evidence_admission as benchmark
    from scripts.qualification_benchmark_worker import _run

    script = Path(benchmark.__file__).resolve()
    machine_path = tmp_path / "machine.json"
    report_path = tmp_path / "report.json"
    observation_path = tmp_path / "observation.json"
    script_digest = hashlib.sha256(script.read_bytes()).hexdigest()
    machine = {
        "schemaVersion": 1,
        "cpuModel": "Synthetic Test CPU",
        "logicalCpuCount": 8,
        "installedRamBytes": 17_179_869_184,
        "windowsEdition": "Windows Test Edition",
        "windowsBuild": "26000.1",
        "windowsArchitecture": "AMD64",
        "pythonFullVersion": "3.11.16",
        "pythonArchitecture": "64bit",
        "acPower": True,
        "powerScheme": "high_performance",
        "benchmarkScriptSha256": script_digest,
    }
    def make_report(**arguments):
        value = benchmark.make_report_for_testing(required=True, passed=True)
        value["benchmarkScriptSha256"] = script_digest
        value["machineManifestSha256"] = hashlib.sha256(arguments["machine_bytes"]).hexdigest()
        value["payloadSha256"] = arguments["payload_sha"]
        return value

    fake = Row(
        __file__=str(script),
        collect_live_machine_manifest=lambda: machine,
        write_machine_manifest_atomic=benchmark.write_machine_manifest_atomic,
        run_benchmark=lambda: ([], {}, 1, "d" * 64),
        make_report=make_report,
        write_report_atomic=benchmark.write_report_atomic,
        canonical_json_bytes=benchmark.canonical_json_bytes,
        sha256_bytes=benchmark.sha256_bytes,
    )

    _run(fake, machine_path, report_path, observation_path)

    observed = benchmark.parse_canonical_json_bytes(observation_path.read_bytes())
    assert observed["pid"] > 0
    assert observed["benchmarkScriptSha256"] == script_digest
    assert observed["machineManifestSha256"] == hashlib.sha256(
        machine_path.read_bytes()
    ).hexdigest()
    assert observed["benchmarkReportSha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
