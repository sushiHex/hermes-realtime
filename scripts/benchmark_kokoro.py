"""Reproduce a single Kokoro ONNX provider/model latency benchmark.

GPU runs require a separate environment containing exactly one ONNX Runtime
package plus its matching NVIDIA runtime libraries. The selected execution
provider is asserted so CUDA cannot silently fall back to CPU.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import onnxruntime as ort
from kokoro_onnx import Kokoro

_MODEL_SAMPLE_RATE_HZ = 24_000
_SHORT_TEXT = "The review is complete. I found two timing issues and corrected both."
_MEDIUM_TEXT = (
    "The assistant should begin speaking promptly, remain interruptible, preserve the "
    "complete answer, and resume only the unspoken suffix after a tentative interruption. "
    "It should also keep transcript identity stable while audio is synthesized and "
    "delivered through the realtime transport."
)
_LONG_TEXT = " ".join([_MEDIUM_TEXT] * 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _summarize(times: list[float], sample_count: int) -> dict[str, float | int]:
    duration = sample_count / _MODEL_SAMPLE_RATE_HZ
    return {
        "runs": len(times),
        "median_seconds": round(statistics.median(times), 4),
        "p95_seconds": round(_percentile(times, 0.95), 4),
        "max_seconds": round(max(times), 4),
        "audio_seconds": round(duration, 4),
        "median_rtf": round(statistics.median(times) / duration, 4),
    }


async def _stream_once(engine: Kokoro, text: str) -> tuple[float, float, int, int]:
    started = time.perf_counter()
    first = 0.0
    chunks = 0
    samples = 0
    async for audio, sample_rate in engine.create_stream(
        text,
        voice="bf_isabella",
        speed=1.0,
        lang="en-gb",
    ):
        if sample_rate != _MODEL_SAMPLE_RATE_HZ:
            raise RuntimeError(f"unexpected sample rate: {sample_rate}")
        chunks += 1
        samples += len(audio)
        if not first:
            first = time.perf_counter() - started
    return first, time.perf_counter() - started, chunks, samples


async def _benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.provider == "cuda":
        ort.preload_dlls(directory="")
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        requested = ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(arguments.model),
        sess_options=options,
        providers=requested,
    )
    engine = Kokoro.from_session(session, str(arguments.voices))
    cold_load = time.perf_counter() - started
    actual = session.get_providers()
    if actual[0] != requested[0]:
        raise RuntimeError(f"selected {actual}, expected {requested}")

    engine.create("Warm up.", voice="bf_isabella", speed=1.0, lang="en-gb")
    workloads: dict[str, Any] = {}
    for label, text in (("short", _SHORT_TEXT), ("medium", _MEDIUM_TEXT)):
        times: list[float] = []
        sample_count = 0
        for _ in range(arguments.runs):
            started = time.perf_counter()
            audio, sample_rate = engine.create(
                text,
                voice="bf_isabella",
                speed=1.0,
                lang="en-gb",
            )
            times.append(time.perf_counter() - started)
            if sample_rate != _MODEL_SAMPLE_RATE_HZ:
                raise RuntimeError(f"unexpected sample rate: {sample_rate}")
            sample_count = len(audio)
        workloads[label] = {"characters": len(text), **_summarize(times, sample_count)}

    first_times: list[float] = []
    total_times: list[float] = []
    chunks = 0
    samples = 0
    for _ in range(arguments.stream_runs):
        first, total, chunks, samples = await _stream_once(engine, _LONG_TEXT)
        first_times.append(first)
        total_times.append(total)
    workloads["long_stream"] = {
        "characters": len(_LONG_TEXT),
        "runs": arguments.stream_runs,
        "chunks": chunks,
        "first_median_seconds": round(statistics.median(first_times), 4),
        "total_median_seconds": round(statistics.median(total_times), 4),
        "audio_seconds": round(samples / _MODEL_SAMPLE_RATE_HZ, 4),
    }
    return {
        "label": arguments.label,
        "ort_version": ort.__version__,
        "model_bytes": arguments.model.stat().st_size,
        "model_sha256": _sha256(arguments.model),
        "voices_bytes": arguments.voices.stat().st_size,
        "voices_sha256": _sha256(arguments.voices),
        "requested_providers": requested,
        "actual_providers": actual,
        "cold_load_seconds": round(cold_load, 4),
        "workloads": workloads,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--runs", type=int, default=10, choices=range(1, 101))
    parser.add_argument("--stream-runs", type=int, default=3, choices=range(1, 21))
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    for path in (arguments.model, arguments.voices):
        if not path.is_file():
            raise FileNotFoundError(path)
    print(json.dumps(asyncio.run(_benchmark(arguments)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
