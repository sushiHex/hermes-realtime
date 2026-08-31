"""Characterize source-backed transcript progressions without exposing transcript text."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from hermes_realtime.providers.current_facts import (
    CurrentFactEvidence,
    CurrentFactLookup,
    contains_private_material,
    foreground_search_query,
)
from hermes_realtime.providers.knowledge_backends import (
    DdgsKnowledgeBackend,
    ExaInstantKnowledgeBackend,
    OpenAIHostedSearchBackend,
    TavilyKnowledgeBackend,
)

_SCHEMA_VERSION = 1
_MIN_BAKEOFF_PAIRS = 2
_DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "source_backed_latency_cases.json"
)
_MIN_STABLE_PREFIX_CHARS = 20
_MIN_CANDIDATE_CHARS = 32
_DEBOUNCE_MS = 120


def _normalized(text: str) -> str:
    if type(text) is not str:
        raise TypeError("transcript text must be an exact string")
    return " ".join(text.split())


def nearest_rank_percentile(values: tuple[float, ...], fraction: float) -> float:
    """Return a nearest-rank percentile with strict finite sample validation."""

    if type(values) is not tuple or not values:
        raise ValueError("latency samples must not be empty")
    if type(fraction) is not float or not 0 < fraction <= 1:
        raise ValueError("percentile fraction must be between zero and one")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in values
    ):
        raise ValueError("latency samples must be finite non-negative numbers")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def stable_prefix_chars(previous: str, current: str) -> int:
    """Count a normalized shared prefix ending at a complete word boundary."""

    left = _normalized(previous)
    right = _normalized(current)
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index].casefold() == right[index].casefold():
        index += 1
    if index == 0:
        return 0
    left_continues_word = index < len(left) and left[index].isalnum()
    right_continues_word = index < len(right) and right[index].isalnum()
    ended_inside_word = left_continues_word or right_continues_word
    if index == len(left) and index < len(right) and right[index].isalnum():
        ended_inside_word = True
    if index == len(right) and index < len(left) and left[index].isalnum():
        ended_inside_word = True
    if ended_inside_word:
        boundary = left.rfind(" ", 0, index)
        return max(0, boundary)
    return index


def speculation_eligible(text: str) -> bool:
    """Fail closed unless text is routed and contains no recognized private material."""

    normalized = _normalized(text)
    if not normalized or len(normalized) > 512 or contains_private_material(normalized):
        return False
    return foreground_search_query(normalized) is not None


def _exact_dict(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact object")
    mapping = cast(dict[object, object], value)
    if any(type(key) is not str for key in mapping):
        raise TypeError(f"{label} keys must be exact strings")
    return cast(dict[str, object], mapping)


def load_corpus(path: Path) -> dict[str, object]:
    """Load and validate the bounded synthetic characterization corpus."""

    if not isinstance(path, Path):
        raise TypeError("corpus path must be a Path")
    payload = _exact_dict(json.loads(path.read_text(encoding="utf-8")), "corpus")
    if payload.get("schemaVersion") != _SCHEMA_VERSION:
        raise ValueError("unsupported corpus schemaVersion")
    if set(payload) != {"schemaVersion", "corpusVersion", "description", "cases"}:
        raise ValueError("corpus has unexpected fields")
    corpus_version = payload.get("corpusVersion")
    description = payload.get("description")
    cases = payload.get("cases")
    if type(corpus_version) is not str or not corpus_version or len(corpus_version) > 80:
        raise ValueError("corpusVersion is invalid")
    if type(description) is not str or not description or len(description) > 240:
        raise ValueError("corpus description is invalid")
    if type(cases) is not list or not 1 <= len(cases) <= 200:
        raise ValueError("corpus cases are invalid")
    seen: set[str] = set()
    for item in cases:
        case = _exact_dict(item, "case")
        if set(case) != {"id", "category", "stt", "partials", "finalAtMs", "final"}:
            raise ValueError("case has unexpected fields")
        case_id = case.get("id")
        category = case.get("category")
        stt = case.get("stt")
        final_at = case.get("finalAtMs")
        final = case.get("final")
        partials = case.get("partials")
        if type(case_id) is not str or not case_id or len(case_id) > 40 or case_id in seen:
            raise ValueError("case id is invalid")
        seen.add(case_id)
        if type(category) is not str or not category or len(category) > 80:
            raise ValueError("case category is invalid")
        if stt not in {"moonshine", "faster-whisper"}:
            raise ValueError("case stt is invalid")
        if type(final_at) is not int or not 0 <= final_at <= 120_000:
            raise ValueError("case finalAtMs is invalid")
        if type(final) is not str or not final.strip() or len(final) > 512:
            raise ValueError("case final is invalid")
        if type(partials) is not list or len(partials) > 100:
            raise ValueError("case partials are invalid")
        prior_at = -1
        for raw_partial in partials:
            partial = _exact_dict(raw_partial, "partial")
            if set(partial) != {"atMs", "text"}:
                raise ValueError("partial has unexpected fields")
            at_ms = partial.get("atMs")
            text = partial.get("text")
            if type(at_ms) is not int or not prior_at < at_ms < final_at:
                raise ValueError("partial atMs is invalid")
            if type(text) is not str or not text.strip() or len(text) > 512:
                raise ValueError("partial text is invalid")
            prior_at = at_ms
    return payload


def _case_sample(raw_case: object) -> dict[str, object]:
    case = _exact_dict(raw_case, "case")
    case_id = cast(str, case["id"])
    category = cast(str, case["category"])
    stt = cast(str, case["stt"])
    final = cast(str, case["final"])
    final_at = cast(int, case["finalAtMs"])
    raw_partials = cast(list[object], case["partials"])
    base: dict[str, object] = {"id": case_id, "category": category, "stt": stt}
    if stt != "moonshine" or not raw_partials:
        return {**base, "status": "sttFinalOnly", "stablePrefixChars": 0, "lastPartialExact": False}
    partial_texts = [cast(str, _exact_dict(item, "partial")["text"]) for item in raw_partials]
    if contains_private_material(final) or any(
        contains_private_material(text) for text in partial_texts
    ):
        return {
            **base,
            "status": "privateRejected",
            "stablePrefixChars": 0,
            "lastPartialExact": False,
        }
    if foreground_search_query(final) is None:
        return {**base, "status": "unrouted", "stablePrefixChars": 0, "lastPartialExact": False}
    if not speculation_eligible(final):
        return {
            **base,
            "status": "privateRejected",
            "stablePrefixChars": 0,
            "lastPartialExact": False,
        }
    last = _exact_dict(raw_partials[-1], "partial")
    last_text = cast(str, last["text"])
    exact = _normalized(last_text).casefold() == _normalized(final).casefold()
    prefix = 0
    if len(raw_partials) >= 2:
        previous = cast(str, _exact_dict(raw_partials[-2], "partial")["text"])
        prefix = stable_prefix_chars(previous, last_text)
    if (
        not exact
        or prefix < _MIN_STABLE_PREFIX_CHARS
        or len(_normalized(last_text)) < _MIN_CANDIDATE_CHARS
        or not speculation_eligible(last_text)
    ):
        return {
            **base,
            "status": "unstable",
            "stablePrefixChars": prefix,
            "lastPartialExact": exact,
        }
    admitted_at = cast(int, last["atMs"]) + _DEBOUNCE_MS
    window = max(0, final_at - admitted_at)
    return {
        **base,
        "status": "eligible",
        "stablePrefixChars": prefix,
        "lastPartialExact": True,
        "speculationWindowMs": window,
    }


def build_characterization_report(
    corpus: dict[str, object],
    *,
    backend: str,
    environment: dict[str, str],
) -> dict[str, object]:
    """Build a transcript-free, complete-outcome characterization report."""

    if type(corpus) is not dict or corpus.get("schemaVersion") != _SCHEMA_VERSION:
        raise ValueError("corpus schemaVersion is invalid")
    if type(backend) is not str or not backend or len(backend) > 40:
        raise ValueError("backend is invalid")
    if type(environment) is not dict or set(environment) != {"python", "os", "head", "diffSha256"}:
        raise ValueError("environment identity is incomplete")
    if any(
        type(value) is not str or not value or len(value) > 160
        for value in environment.values()
    ):
        raise ValueError("environment identity is invalid")
    cases = corpus.get("cases")
    if type(cases) is not list or not cases:
        raise ValueError("corpus cases are invalid")
    samples = [_case_sample(case) for case in cases]
    outcomes = Counter(cast(str, sample["status"]) for sample in samples)
    windows = tuple(
        float(cast(int, sample["speculationWindowMs"]))
        for sample in samples
        if "speculationWindowMs" in sample
    )
    window_summary: dict[str, object]
    if windows:
        window_summary = {
            "count": len(windows),
            "p50": nearest_rank_percentile(windows, 0.50),
            "p95": nearest_rank_percentile(windows, 0.95),
            "max": max(windows),
        }
    else:
        window_summary = {"count": 0, "p50": None, "p95": None, "max": None}
    report: dict[str, object] = {
        "schemaVersion": _SCHEMA_VERSION,
        "mode": "characterize",
        "backend": backend,
        "corpusVersion": corpus.get("corpusVersion"),
        "environment": dict(environment),
        "policy": {
            "minStablePrefixChars": _MIN_STABLE_PREFIX_CHARS,
            "minCandidateChars": _MIN_CANDIDATE_CHARS,
            "debounceMs": _DEBOUNCE_MS,
            "exactFinalMatch": True,
        },
        "summary": {
            "totalCases": len(samples),
            "outcomes": dict(sorted(outcomes.items())),
            "speculationWindowMs": window_summary,
        },
        "samples": samples,
    }
    json.dumps(report, allow_nan=False, ensure_ascii=True)
    return report


def build_bakeoff_backends(
    names: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    allow_metered_api: bool,
) -> dict[str, CurrentFactLookup]:
    """Build explicitly selected arms without implicit auth or billing changes."""

    allowed = {
        "ddgs",
        "tavily-ultra-fast",
        "tavily-fast",
        "exa-instant",
        "openai-hosted-search",
    }
    if (
        type(names) is not tuple
        or not names
        or len(names) > len(allowed)
        or len(set(names)) != len(names)
        or any(type(name) is not str or name not in allowed for name in names)
        or "ddgs" not in names
    ):
        raise ValueError("backend names must be unique supported arms including ddgs")
    if type(allow_metered_api) is not bool:
        raise TypeError("allow_metered_api must be an exact bool")
    metered = tuple(name for name in names if name != "ddgs")
    if metered and not allow_metered_api:
        raise PermissionError("metered arms require --allow-metered-api")

    required_environment = {
        "tavily-ultra-fast": "TAVILY_API_KEY",
        "tavily-fast": "TAVILY_API_KEY",
        "exa-instant": "EXA_API_KEY",
        "openai-hosted-search": "OPENAI_API_KEY",
    }
    for name in metered:
        variable = required_environment[name]
        value = environment.get(variable)
        if type(value) is not str or not value.strip():
            raise RuntimeError(f"{variable} is required for {name}")

    backends: dict[str, CurrentFactLookup] = {}
    for name in names:
        if name == "ddgs":
            backends[name] = DdgsKnowledgeBackend()
        elif name.startswith("tavily-"):
            backends[name] = TavilyKnowledgeBackend(
                api_key=environment["TAVILY_API_KEY"],
                search_depth=name.removeprefix("tavily-"),
            )
        elif name == "exa-instant":
            backends[name] = ExaInstantKnowledgeBackend(
                api_key=environment["EXA_API_KEY"]
            )
        else:
            backends[name] = OpenAIHostedSearchBackend(
                api_key=environment["OPENAI_API_KEY"]
            )
    return backends


async def run_backend_bakeoff(
    corpus: dict[str, object],
    *,
    backends: Mapping[str, CurrentFactLookup],
    environment: dict[str, str],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run paired routed cases and return transcript-free backend outcomes."""

    if type(corpus) is not dict or corpus.get("schemaVersion") != _SCHEMA_VERSION:
        raise ValueError("corpus schemaVersion is invalid")
    if not isinstance(backends, Mapping) or not 1 <= len(backends) <= 8:
        raise ValueError("backends must contain between one and eight arms")
    if any(
        type(name) is not str
        or not name
        or len(name) > 40
        or not callable(getattr(backend, "lookup", None))
        for name, backend in backends.items()
    ):
        raise ValueError("backend arms are invalid")
    if type(environment) is not dict or set(environment) != {
        "python",
        "os",
        "head",
        "diffSha256",
    }:
        raise ValueError("environment identity is incomplete")
    if not callable(clock):
        raise TypeError("clock must be callable")
    raw_cases = corpus.get("cases")
    if type(raw_cases) is not list:
        raise ValueError("corpus cases are invalid")
    paired_cases = []
    for raw_case in raw_cases:
        case = _exact_dict(raw_case, "case")
        final = cast(str, case["final"])
        if speculation_eligible(final):
            paired_cases.append(case)
    if len(paired_cases) < _MIN_BAKEOFF_PAIRS:
        raise ValueError(
            f"bakeoff requires at least {_MIN_BAKEOFF_PAIRS} routed paired cases"
        )

    arm_samples: dict[str, list[dict[str, object]]] = {
        name: [] for name in backends
    }
    case_samples: list[dict[str, object]] = []
    for pair_index, case in enumerate(paired_cases):
        outcomes: dict[str, object] = {}
        query = cast(str, case["final"])
        arm_order = list(backends)
        random.Random(f"foreground-knowledge:{pair_index}:{case['id']}").shuffle(arm_order)
        for name in arm_order:
            backend = backends[name]
            started = clock()
            evidence: CurrentFactEvidence | None = None
            failed = False
            try:
                evidence = await backend.lookup(query)
            except Exception:
                failed = True
            elapsed_ms = max(0.0, (clock() - started) * 1_000.0)
            quality = "error" if evidence is None else evidence.quality
            source_count = 0 if evidence is None else len(evidence.sources)
            backend_error = failed or (evidence is not None and evidence.error is not None)
            sample = {
                "latencyMs": elapsed_ms,
                "quality": quality,
                "sourceCount": source_count,
                "error": backend_error,
            }
            arm_samples[name].append(sample)
            outcomes[name] = sample
        case_samples.append(
            {
                "id": case["id"],
                "category": case["category"],
                "armOrder": arm_order,
                "outcomes": outcomes,
            }
        )

    summaries: dict[str, object] = {}
    for name, backend in backends.items():
        samples = arm_samples[name]
        attempts = len(samples)
        latencies = tuple(
            float(cast(float, sample["latencyMs"])) for sample in samples
        )
        usable = sum(sample["quality"] == "usable" for sample in samples)
        empty = sum(sample["quality"] == "empty" for sample in samples)
        errors = sum(bool(sample["error"]) for sample in samples)
        summaries[name] = {
            "attempts": attempts,
            "retrievalMs": {
                "p50": nearest_rank_percentile(latencies, 0.50),
                "p95": nearest_rank_percentile(latencies, 0.95),
            },
            "supportRate": usable / attempts,
            "emptyRate": empty / attempts,
            "errorRate": errors / attempts,
            "authenticationMode": str(
                getattr(backend, "authentication_mode", "unknown")
            )[:80],
            "costMode": str(getattr(backend, "cost_mode", "unknown"))[:80],
        }
    report: dict[str, object] = {
        "schemaVersion": _SCHEMA_VERSION,
        "mode": "bakeoff",
        "corpusVersion": corpus.get("corpusVersion"),
        "environment": dict(environment),
        "pairedCaseCount": len(paired_cases),
        "backends": summaries,
        "samples": case_samples,
    }
    json.dumps(report, allow_nan=False, ensure_ascii=True)
    return report


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=False,
    )
    return completed.stdout.decode("ascii", errors="strict").strip()


def collect_environment_identity() -> dict[str, str]:
    """Return bounded source/environment identity without including diff contents."""

    try:
        head = _git_output("rev-parse", "--short=12", "HEAD")
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
        untracked_paths = _git_output("ls-files", "--others", "--exclude-standard").splitlines()
        untracked_fingerprint = bytearray()
        for relative in sorted(untracked_paths):
            path = Path(relative)
            if not path.is_file():
                continue
            untracked_fingerprint.extend(relative.encode("utf-8"))
            untracked_fingerprint.extend(b"\0")
            untracked_fingerprint.extend(hashlib.sha256(path.read_bytes()).digest())
        diff_sha = hashlib.sha256(diff + b"\0" + bytes(untracked_fingerprint)).hexdigest()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        head = "unavailable"
        diff_sha = hashlib.sha256(b"unavailable").hexdigest()
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()}-{platform.release()}",
        "head": head,
        "diffSha256": diff_sha,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        choices=(
            "ddgs",
            "tavily-ultra-fast",
            "tavily-fast",
            "exa-instant",
            "openai-hosted-search",
        ),
        help="repeat for paired bakeoff arms; DDGS must be included",
    )
    parser.add_argument(
        "--mode",
        default="characterize",
        choices=("characterize", "bakeoff"),
    )
    parser.add_argument(
        "--allow-metered-api",
        action="store_true",
        help="acknowledge that explicitly selected credentialed arms may incur charges",
    )
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    corpus = load_corpus(arguments.corpus)
    environment_identity = collect_environment_identity()
    names = tuple(arguments.backend or ("ddgs",))
    if arguments.mode == "characterize":
        if names != ("ddgs",) or arguments.allow_metered_api:
            raise ValueError("characterize mode supports only unmetered DDGS")
        report = build_characterization_report(
            corpus,
            backend="ddgs",
            environment=environment_identity,
        )
    else:
        backends = build_bakeoff_backends(
            names,
            environment=os.environ,
            allow_metered_api=arguments.allow_metered_api,
        )
        report = asyncio.run(
            _run_bakeoff_and_close(
                corpus,
                backends=backends,
                environment=environment_identity,
            )
        )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(encoded, encoding="utf-8")
    printable = report.get("summary", report.get("backends"))
    print(json.dumps(printable, sort_keys=True, allow_nan=False))
    return 0


async def _run_bakeoff_and_close(
    corpus: dict[str, object],
    *,
    backends: dict[str, CurrentFactLookup],
    environment: dict[str, str],
) -> dict[str, object]:
    try:
        return await run_backend_bakeoff(
            corpus,
            backends=backends,
            environment=environment,
        )
    finally:
        await asyncio.gather(*(backend.close() for backend in backends.values()))


if __name__ == "__main__":
    sys.exit(main())
