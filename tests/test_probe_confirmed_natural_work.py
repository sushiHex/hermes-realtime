import io
import json

import pytest

from hermes_realtime.conversation.task_proposals import HostTaskProposalRequest
from scripts import probe_confirmed_natural_work as probe


def test_v2_is_hash_bound_exposed_comparison_with_exact_v1_cases():
    v1 = probe.load_fixture("v1")
    v2 = probe.load_fixture("v2")
    assert v2["model_cases"] == v1["model_cases"]
    assert v2["scenario_id"] == "confirmed-natural-work-v2"
    assert v2["development_preregistration"]["comparison_fixture_sha256"] == (
        probe.qualification.FIXTURE_SHA256
    )
    assert set(v2["development_preregistration"]["branch_falsifiers"]) == {
        "natural_current_weather", "cancel", "approval",
    }
    assert v2["development_preregistration"]["expected_branches"] == {
        "natural_current_weather": "proposal", "cancel": "reply", "approval": "reply",
    }
    report = probe.build_report([], fixture_version="v2")
    assert report["scenario_id"] == v2["scenario_id"]
    assert report["fixture_sha256"] == probe.V2_FIXTURE_SHA256
    assert report["evaluation_kind"] == "exposed_development_comparison"
    assert report["denominator"] == report["unrun"] == 8
    assert report["objective_fidelity"] == "pending"
    assert report["natural_routing_qualified"] is False


def test_v2_changed_fixture_is_refused(monkeypatch):
    monkeypatch.setattr(probe, "V2_FIXTURE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="preregistration_binding"):
        probe.load_fixture("v2")


def test_v2_selected_through_main_and_aborted_tail(monkeypatch, tmp_path, capsys):
    versions = []
    fixture_loads = []
    original = probe.load_fixture

    def load(version):
        fixture_loads.append(version)
        return original(version)

    async def failed_attempt(**kwargs):
        versions.append(kwargs["fixture_version"])
        raise RuntimeError("synthetic failure before first case")

    monkeypatch.setattr(probe.harness, "verify_imports", lambda: None)
    monkeypatch.setattr(probe.harness, "source_binding", lambda: {
        "head": "a" * 40, "tree": "b" * 40, "dirty": False,
    })
    monkeypatch.setattr(probe, "run_cases", failed_attempt)
    monkeypatch.setattr(probe, "load_fixture", load)
    receipt = tmp_path / "receipt"
    assert probe.main([
        "--source-head", "a" * 40, "--source-tree", "b" * 40,
        "--model-digest", "c" * 64, "--base-url", "http://127.0.0.1:1",
        "--private-receipt", str(receipt), "--fixture-version", "v2", "--live",
    ]) == 1
    assert versions == ["v2"]
    assert fixture_loads == ["v2", "v2"]
    rows = [json.loads(line) for line in receipt.read_bytes().splitlines()]
    assert len(rows) == 8 and all(row["status"] == "not_run" for row in rows)
    report = json.loads(capsys.readouterr().out.split("[confirmed-natural-probe] ")[1])
    assert report["scenario_id"] == "confirmed-natural-work-v2"
    assert report["fixture_sha256"] == probe.V2_FIXTURE_SHA256


def test_v2_fixture_drift_after_preparation_is_refused(monkeypatch, tmp_path, capsys):
    loads = 0
    original = probe.load_fixture

    def changed_fixture(version):
        nonlocal loads
        loads += 1
        if loads == 2:
            raise ValueError("preregistration_binding")
        return original(version)

    monkeypatch.setattr(probe, "load_fixture", changed_fixture)
    monkeypatch.setattr(probe.harness, "verify_imports", lambda: None)
    monkeypatch.setattr(probe.harness, "source_binding", lambda: {
        "head": "a" * 40, "tree": "b" * 40, "dirty": False,
    })
    assert probe.main([
        "--source-head", "a" * 40, "--source-tree", "b" * 40,
        "--model-digest", "c" * 64, "--base-url", "http://127.0.0.1:1",
        "--private-receipt", str(tmp_path / "receipt"), "--fixture-version", "v2",
    ]) == 1
    assert '"status": "attempt_failed"' in capsys.readouterr().out
    assert not (tmp_path / "receipt").exists()


def test_prompt_timing_sink_keeps_only_numeric_fields():
    sink = probe.PromptTimingSink()
    sink.write("private unrelated diagnostic\n")
    sink.write('[ollama-prompt] {"load_duration_ns":12,"num_ctx":4096,')
    sink.write('"prompt":"private text","timing_invalid_fields":["eval_duration"]}\n')
    assert sink.rows == [{"stage": "readiness", "load_duration_ns": 12, "num_ctx": 4096}]
    assert "private" not in json.dumps(sink.rows)


def test_prompt_timing_sink_rejects_non_numeric_value():
    sink = probe.PromptTimingSink()
    with pytest.raises(ValueError, match="timing_numeric"):
        sink.write('[ollama-prompt] {"load_duration_ns":"private text"}\n')


def test_prompt_timing_sink_rejects_unbounded_line():
    sink = probe.PromptTimingSink()
    with pytest.raises(ValueError, match="timing_line_bound"):
        sink.write("x" * 65537)


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_version", ["v1", "v2"])
async def test_probe_never_offers_or_confirms_and_retains_eight_cases(monkeypatch, fixture_version):
    fixture_loads = []
    original = probe.load_fixture

    def load(version):
        fixture_loads.append(version)
        return original(version)

    class Inference:
        async def configure_context(self, size):
            assert size == 4096

        def context_public_data(self):
            return {"modelDigest": "a" * 64}

        async def stream(self, request, turn_id):
            print('[ollama-prompt] {"load_duration_ns":12,"num_ctx":4096}')
            yield HostTaskProposalRequest("Synthetic proposed work")

        async def close(self):
            pass

    async def preflight(**kwargs):
        pass

    monkeypatch.setattr(probe, "factory", lambda **kwargs: Inference())
    monkeypatch.setattr(probe, "preflight", preflight)
    monkeypatch.setattr(probe, "OllamaStreamingInference", Inference)
    monkeypatch.setattr(probe, "load_fixture", load)
    receipt = io.BytesIO()
    rows = await probe.run_cases(model="phi4-mini:latest", base_url="http://127.0.0.1:1",
                                 private_sink=receipt, model_digest="a" * 64,
                                 fixture_version=fixture_version)
    assert len(rows) == 8
    assert fixture_loads == [fixture_version]
    assert all(row["selection"] == "proposal" for row in rows)
    report = probe.build_report(rows, fixture_version=fixture_version)
    assert report["unwanted_proposals"] == 5
    assert report["missed_proposals"] == 0
    assert report["dispatches"] == 0
    assert report["readiness_attempts"] == report["decision_attempts"] == 8
    assert report["objective_fidelity"] == "pending"
    assert report["actual_source_proof"] == "not_run"
    assert report["natural_routing_qualified"] is False
    assert b"Synthetic proposed work" in receipt.getvalue()
    assert "Synthetic proposed work" not in json.dumps(report)


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["proposal_first", "speech_first"])
async def test_mixed_proposal_and_speech_is_not_a_usable_selection(order):
    class Inference:
        async def stream(self, request, turn_id):
            items = [HostTaskProposalRequest("Synthetic proposed work"),
                     "A misleading spoken claim."]
            for item in items if order == "proposal_first" else reversed(items):
                yield item

    with pytest.raises(ValueError, match="selection_terminal"):
        await probe.collect_decision(Inference(), None, "synthetic", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("outputs,error", [([], "selection_empty"), ([123], "selection_type")])
async def test_missing_or_untyped_selection_refused(outputs, error):
    class Inference:
        async def stream(self, request, turn_id):
            for item in outputs:
                yield item

    with pytest.raises(ValueError, match=error):
        await probe.collect_decision(Inference(), None, "synthetic", {})


@pytest.mark.asyncio
async def test_model_digest_refused_before_readiness_or_decision(monkeypatch):
    class Inference:
        async def configure_context(self, size):
            pass

        def context_public_data(self):
            return {"modelDigest": "b" * 64}

        async def close(self):
            pass

    calls = []

    async def preflight(**kwargs):
        calls.append("readiness")

    monkeypatch.setattr(probe, "factory", lambda **kwargs: Inference())
    monkeypatch.setattr(probe, "preflight", preflight)
    monkeypatch.setattr(probe, "OllamaStreamingInference", Inference)
    receipt = io.BytesIO()
    with pytest.raises(ValueError, match="model_digest_refused"):
        await probe.run_cases(model="phi4-mini:latest", base_url="http://127.0.0.1:1",
                              private_sink=receipt, model_digest="a" * 64)
    assert calls == []
    rows = [json.loads(line) for line in receipt.getvalue().splitlines()]
    assert len(rows) == 1
    assert rows[0]["error"] == "attempt_failed"


@pytest.mark.asyncio
async def test_unexpected_factory_provider_refused(monkeypatch):
    class WrongProvider:
        async def close(self):
            pass

    monkeypatch.setattr(probe, "factory", lambda **kwargs: WrongProvider())
    with pytest.raises(ValueError, match="provider_refused"):
        await probe.run_cases(model="phi4-mini:latest", base_url="http://127.0.0.1:1",
                              private_sink=io.BytesIO(), model_digest="a" * 64)


@pytest.mark.parametrize("fault", ["tree", "digest", "drift"])
def test_prepare_refuses_each_binding_fault(monkeypatch, tmp_path, capsys, fault):
    calls = 0

    def binding():
        nonlocal calls
        calls += 1
        return {"head": "a" * 40, "tree": "d" * 40 if fault == "drift" and calls == 2
                else "b" * 40, "dirty": False}

    monkeypatch.setattr(probe.harness, "verify_imports", lambda: None)
    monkeypatch.setattr(probe.harness, "source_binding", binding)
    result = probe.main([
        "--source-head", "a" * 40, "--source-tree", "c" * 40 if fault == "tree" else "b" * 40,
        "--model-digest", "invalid" if fault == "digest" else "e" * 64,
        "--base-url", "http://127.0.0.1:1", "--private-receipt", str(tmp_path / "receipt"),
    ])
    assert result == 1
    assert '"status": "attempt_failed"' in capsys.readouterr().out
    assert not (tmp_path / "receipt").exists()


def test_aborted_tail_is_explicit_in_receipt_and_eight_case_denominator(
    monkeypatch, tmp_path, capsys,
):
    async def failed_attempt(**kwargs):
        row = {"id": "research", "expected": "proposal", "status": "attempted",
               "error": "attempt_failed", "selection": None, "dispatches": 0, "timings": []}
        kwargs["observations"].append(row)
        probe.harness.write_private_row(kwargs["private_sink"], row)
        raise RuntimeError("synthetic operational failure")

    monkeypatch.setattr(probe.harness, "verify_imports", lambda: None)
    monkeypatch.setattr(probe.harness, "source_binding", lambda: {
        "head": "a" * 40, "tree": "b" * 40, "dirty": False,
    })
    monkeypatch.setattr(probe, "run_cases", failed_attempt)
    receipt_path = tmp_path / "receipt"
    assert probe.main([
        "--source-head", "a" * 40, "--source-tree", "b" * 40,
        "--model-digest", "c" * 64, "--base-url", "http://127.0.0.1:1",
        "--private-receipt", str(receipt_path), "--live",
    ]) == 1
    receipt = [json.loads(line) for line in receipt_path.read_bytes().splitlines()]
    assert len(receipt) == 8
    assert [row["status"] for row in receipt] == ["attempted"] + ["not_run"] * 7
    report = json.loads(capsys.readouterr().out.split("[confirmed-natural-probe] ")[1])
    assert (report["denominator"], report["attempts"], report["unrun"], report["errors"]) == (
        8, 1, 7, 1,
    )
    assert len(report["case_outcomes"]) == 8


@pytest.mark.parametrize("outcome", ["healthy", "partial", "error", "drift"])
def test_live_operational_completion_is_separate_from_qualification(
    monkeypatch, tmp_path, capsys, outcome,
):
    async def synthetic_attempts(**kwargs):
        cases = probe.qualification.load_preregistration()["model_cases"]
        for case in cases[:1] if outcome == "partial" else cases:
            row = {"id": case["id"], "expected": case["expected"], "status": "attempted",
                   "error": "attempt_failed" if outcome == "error" else None,
                   "selection": "reply", "dispatches": 0, "timings": []}
            kwargs["observations"].append(row)
            probe.harness.write_private_row(kwargs["private_sink"], row)
        return kwargs["observations"]

    calls = 0

    def binding():
        nonlocal calls
        calls += 1
        return {"head": "a" * 40, "dirty": False,
                "tree": "d" * 40 if outcome == "drift" and calls == 2 else "b" * 40}

    monkeypatch.setattr(probe.harness, "verify_imports", lambda: None)
    monkeypatch.setattr(probe.harness, "source_binding", binding)
    monkeypatch.setattr(probe, "run_cases", synthetic_attempts)
    code = probe.main([
        "--source-head", "a" * 40, "--source-tree", "b" * 40,
        "--model-digest", "c" * 64, "--base-url", "http://127.0.0.1:1",
        "--private-receipt", str(tmp_path / "receipt"), "--live",
    ])
    report = json.loads(capsys.readouterr().out.split("[confirmed-natural-probe] ")[1])
    assert code == (0 if outcome == "healthy" else 1)
    assert report["status"] == ("completed" if outcome == "healthy" else "attempt_failed")
    assert report["natural_routing_qualified"] is False
    assert report["objective_fidelity"] == report["lifecycle_claims"] == "pending"
    assert report["denominator"] == 8
