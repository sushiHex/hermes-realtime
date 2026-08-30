from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

_BENCHMARK_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_foreground_knowledge.py"
_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_foreground_knowledge",
    _BENCHMARK_PATH,
)
assert _BENCHMARK_SPEC is not None and _BENCHMARK_SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_BENCHMARK_SPEC)
sys.modules[_BENCHMARK_SPEC.name] = _BENCHMARK
_BENCHMARK_SPEC.loader.exec_module(_BENCHMARK)

build_characterization_report = _BENCHMARK.build_characterization_report
load_corpus = _BENCHMARK.load_corpus
nearest_rank_percentile = _BENCHMARK.nearest_rank_percentile
speculation_eligible = _BENCHMARK.speculation_eligible
stable_prefix_chars = _BENCHMARK.stable_prefix_chars
run_backend_bakeoff = _BENCHMARK.run_backend_bakeoff
build_bakeoff_backends = _BENCHMARK.build_bakeoff_backends
parse_args = _BENCHMARK._parse_args


_FIXTURE = Path(__file__).parent / "fixtures" / "source_backed_latency_cases.json"


def test_nearest_rank_percentile_is_deterministic_and_rejects_bad_samples() -> None:
    assert nearest_rank_percentile((10.0, 40.0, 20.0, 30.0), 0.50) == 20.0
    assert nearest_rank_percentile((10.0, 40.0, 20.0, 30.0), 0.95) == 40.0
    with pytest.raises(ValueError, match="must not be empty"):
        nearest_rank_percentile((), 0.95)
    with pytest.raises(ValueError, match="finite non-negative"):
        nearest_rank_percentile((float("nan"),), 0.95)


def test_stable_prefix_counts_complete_normalized_word_boundary() -> None:
    assert stable_prefix_chars(
        "What is the latest stable Python",
        "What is the latest stable Python release?",
    ) == len("What is the latest stable Python")
    assert stable_prefix_chars("AudioSt", "AudioStream capacity") == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("What is the latest stable Python release?", True),
        ("Find official sources for LiveKit AudioStream documentation.", True),
        ("My password is swordfish; check whether it leaked.", False),
        ("Use bearer redacted-token to look this up.", False),
        (r"Find this under C:\Users\private\secrets.txt", False),
        ("Find official sources for " + "ghp_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", False),
        ("Find official sources for " + "AKIA" + "ABCDEFGHIJKLMNOP", False),
        (r"Find official sources for D:\clients\private\roadmap.txt", False),
        (r"Find official sources for \\private-server\share\roadmap.txt", False),
        ('Find official sources for "A1b2C3d4E5f6G7h8I9j0K1l2"', False),
        ("Tell me a joke about Python.", False),
    ),
)
def test_speculation_eligibility_fails_closed_for_private_or_unrouted_text(
    text: str,
    expected: bool,
) -> None:
    assert speculation_eligible(text) is expected


def test_characterization_report_preserves_complete_outcomes_and_environment() -> None:
    corpus = load_corpus(_FIXTURE)
    report = build_characterization_report(
        corpus,
        backend="ddgs",
        environment={
            "python": "3.11.test",
            "os": "test-os",
            "head": "deadbee",
            "diffSha256": "a" * 64,
        },
    )

    assert report["schemaVersion"] == 1
    assert report["mode"] == "characterize"
    assert report["backend"] == "ddgs"
    assert report["corpusVersion"] == "source-backed-v1"
    assert report["environment"]["diffSha256"] == "a" * 64
    assert report["summary"]["totalCases"] == len(corpus["cases"])
    assert report["summary"]["outcomes"] == {
        "eligible": 3,
        "privateRejected": 3,
        "sttFinalOnly": 1,
        "unrouted": 1,
        "unstable": 1,
    }
    samples = report["samples"]
    assert len(samples) == len(corpus["cases"])
    assert {sample["id"] for sample in samples} == {
        case["id"] for case in corpus["cases"]
    }
    assert all(
        sample["status"]
        in {"eligible", "privateRejected", "unrouted", "unstable", "sttFinalOnly"}
        for sample in samples
    )
    assert report["summary"]["speculationWindowMs"]["count"] == 3
    json.dumps(report, allow_nan=False)


def test_corpus_rejects_unknown_schema_and_transcript_content_in_report(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schemaVersion":2,"cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="schemaVersion"):
        load_corpus(invalid)

    report = build_characterization_report(
        load_corpus(_FIXTURE),
        backend="ddgs",
        environment={"python": "x", "os": "x", "head": "x", "diffSha256": "b" * 64},
    )
    encoded = json.dumps(report)
    assert "swordfish" not in encoded
    assert "sk-123" not in encoded
    assert "secrets.txt" not in encoded


class _FakeBackend:
    authentication_mode = "test_auth"
    cost_mode = "test_free"

    def __init__(self, *, usable: bool) -> None:
        self._usable = usable

    async def lookup(self, query: str):
        current_facts = __import__(
            "hermes_realtime.providers.current_facts",
            fromlist=["CurrentFactEvidence", "CurrentFactSource"],
        )
        sources = ()
        if self._usable:
            sources = (
                current_facts.CurrentFactSource(
                    title="Official result",
                    url="https://example.com/reference",
                    snippet="The current supported value is documented here.",
                ),
            )
        return current_facts.CurrentFactEvidence(
            query=query,
            retrieved_date=datetime.now(UTC).date().isoformat(),
            sources=sources,
            backend="fake",
        )


@pytest.mark.asyncio
async def test_backend_bakeoff_is_paired_bounded_and_transcript_free() -> None:
    ticks = iter((0.0, 0.010, 0.010, 0.030) * 16)
    report = await run_backend_bakeoff(
        load_corpus(_FIXTURE),
        backends={
            "usable-arm": _FakeBackend(usable=True),
            "empty-arm": _FakeBackend(usable=False),
        },
        environment={"python": "x", "os": "x", "head": "x", "diffSha256": "c" * 64},
        clock=lambda: next(ticks),
    )

    assert report["mode"] == "bakeoff"
    assert report["pairedCaseCount"] == 5
    assert report["backends"]["usable-arm"]["attempts"] == 5
    assert report["backends"]["usable-arm"]["supportRate"] == 1.0
    assert report["backends"]["empty-arm"]["emptyRate"] == 1.0
    assert report["backends"]["usable-arm"]["authenticationMode"] == "test_auth"
    assert report["backends"]["usable-arm"]["costMode"] == "test_free"
    encoded = json.dumps(report, allow_nan=False)
    assert "latest stable Python release" not in encoded
    assert "swordfish" not in encoded


@pytest.mark.asyncio
async def test_backend_factory_requires_explicit_metered_auth_and_credentials() -> None:
    with pytest.raises(PermissionError, match="allow-metered-api"):
        build_bakeoff_backends(
            ("ddgs", "tavily-fast"),
            environment={"TAVILY_API_KEY": "secret"},
            allow_metered_api=False,
        )
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        build_bakeoff_backends(
            ("ddgs", "tavily-fast"),
            environment={},
            allow_metered_api=True,
        )

    backends = build_bakeoff_backends(
        ("ddgs", "tavily-fast"),
        environment={"TAVILY_API_KEY": "secret"},
        allow_metered_api=True,
    )

    assert tuple(backends) == ("ddgs", "tavily-fast")
    assert backends["tavily-fast"].authentication_mode == "api_key"
    for backend in backends.values():
        await backend.close()


def test_bakeoff_cli_requires_explicit_repeatable_backend_selection(tmp_path: Path) -> None:
    arguments = parse_args(
        [
            "--mode",
            "bakeoff",
            "--backend",
            "ddgs",
            "--backend",
            "tavily-fast",
            "--allow-metered-api",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert arguments.mode == "bakeoff"
    assert arguments.backend == ["ddgs", "tavily-fast"]
    assert arguments.allow_metered_api is True
