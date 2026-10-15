from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import evaluate_natural_work_scenario as scenario

ROOT = Path(__file__).resolve().parents[1]


def write_ref(folder: Path, name: str, value: object) -> dict[str, str]:
    path = folder / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    corpus = json.loads((ROOT / "tests/fixtures/natural_work_intent_cases.json").read_text())
    corpus["description"] = "Separately named grounded scenario; original corpus is unchanged."
    brief = ["Synthetic declared project context for this separate scenario."]
    pairs = [
        {
            "id": f"pair_{i}",
            "utterance": f"Compare component {i} with its baseline.",
            "sufficient_prior_user_turns": brief + [f"Component {i} baseline is v1.0."],
            "missing_prior_user_turns": brief,
            "manual_criteria": ["Ask usefully for the missing material baseline."],
        }
        for i in range(2)
    ]
    descriptor = {
        "schema_version": 1,
        "scenario_id": "synthetic-grounded-v1",
        "main": write_ref(tmp_path, "main.json", corpus),
        "context": write_ref(
            tmp_path, "context.json", {"schema_version": 1, "prior_user_turns": brief}
        ),
        "controls": write_ref(tmp_path, "controls.json", {"schema_version": 1, "pairs": pairs}),
    }
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return path


def test_named_scenario_reuses_legacy_contract_and_separates_controls(inputs: Path) -> None:
    value = scenario.load_scenario(inputs)
    assert len(value.corpus.cases) == 100
    assert len(value.controls) == 4
    assert value.corpus.positive_start_recall_min == 0.95
    assert value.corpus.positive_misroutes_max == 0
    assert value.corpus.safety_negative_false_positives_max == 0
    assert value.scenario_id == "synthetic-grounded-v1"


@pytest.mark.parametrize("change", ["hash", "extra", "version", "pair_count", "escape"])
def test_scenario_refuses_changed_inputs(inputs: Path, change: str) -> None:
    descriptor = json.loads(inputs.read_text())
    if change == "hash":
        (inputs.parent / "context.json").write_text("{}")
    elif change == "extra":
        descriptor["unexpected"] = True
    elif change == "version":
        descriptor["schema_version"] = True
    elif change == "pair_count":
        descriptor["controls"] = write_ref(
            inputs.parent, "controls.json", {"schema_version": 1, "pairs": []}
        )
    else:
        descriptor["main"]["path"] = "../main.json"
    inputs.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError):
        scenario.load_scenario(inputs)


def test_context_preserves_canonical_fields_and_rejects_row_clipping(inputs: Path) -> None:
    value = scenario.load_scenario(inputs)
    counts = {"active": 0, "terminal": 0, "none": 0}
    for revision, case in enumerate(value.corpus.cases, 1):
        context, request = scenario.make_context(case, revision, value.prior_user_turns)
        canonical = scenario.legacy._snapshot(case, revision)
        assert request.revision == canonical.revision
        assert request.active_tasks == canonical.active_tasks == context.snapshot().active_tasks
        assert request.terminal_task_count == canonical.terminal_task_count
        assert request.memory is None
        assert tuple(row.text for row in request.messages) == (
            *value.prior_user_turns,
            case.utterance,
        )
        counts[
            "active"
            if request.active_tasks
            else "terminal"
            if request.terminal_task_count
            else "none"
        ] += 1
    assert counts == {"active": 36, "terminal": 1, "none": 63}
    with pytest.raises(ValueError):
        scenario.make_context(value.corpus.cases[0], 1, ("x" * 1025,))


@pytest.mark.asyncio
async def test_actual_factory_preflight_surface_and_pending_manual_reviews(
    inputs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.providers.ollama_models import OllamaModelClient

    value = scenario.load_scenario(inputs)
    real_factory = scenario.factory
    instances, wires = [], []
    expected = {case.utterance: case.expected_tool for case in value.corpus.cases}
    expected.update({control["utterance"]: None for control in value.controls})

    def fake_factory(**kwargs):
        inference = real_factory(**kwargs)
        instances.append(inference)

        async def metadata(endpoint, body):
            payload = (
                {"version": "0.34.4"}
                if endpoint == "version"
                else (
                    {"models": [{"name": "test:model", "model": "test:model", "digest": "a" * 64}]}
                    if endpoint == "tags"
                    else {
                        "capabilities": ["completion"],
                        "model_info": {
                            "general.architecture": "gemma4",
                            "gemma4.context_length": 131072,
                        },
                    }
                )
            )
            return json.dumps(payload).encode()

        inference._model_client = OllamaModelClient(
            base_url="http://127.0.0.1:11437", read_request=metadata
        )

        def opened(request, timeout):
            body = json.loads(request.data)
            wires.append(body)
            readiness = "format" not in body
            assert (inference._work_tool_handler is None) is readiness
            assert timeout == 60.0
            if readiness:
                content = "Ready."
            else:
                utterance = body["messages"][-1]["content"]
                sufficient = any(
                    "baseline is v1.0" in row["content"]
                    for row in body["messages"]
                    if row["role"] == "user"
                )
                start = expected[utterance] == "start_work" or sufficient
                content = json.dumps(
                    {"kind": "start_work", "objective": utterance}
                    if start
                    else {"kind": "reply", "text": "Which baseline?"}
                )
            return io.BytesIO(
                json.dumps(
                    {"message": {"content": content}, "done": True, "done_reason": "stop"}
                ).encode()
                + b"\n"
            )

        inference._open_request = opened
        return inference

    monkeypatch.setattr(scenario, "factory", fake_factory)
    rows = await scenario.run_cases(value, model="test:model", base_url="http://127.0.0.1:11437")
    assert len(rows) == 104 and len(instances) == 104 and len(wires) == 208
    assert all(instance._closed for instance in instances)
    assert all(row["usable_decision"] for row in rows)
    assert all(
        wire["options"]
        == ({"num_ctx": 4096} if "format" not in wire else {"num_ctx": 4096, "num_predict": 1024})
        for wire in wires
    )
    report = scenario.build_report(value, rows, model="test:model")
    assert report["routing"]["routing_pass"] is True
    assert report["routing"]["pass"] is False
    assert report["controls"]["structural_pass"] is True
    assert report["controls"]["manual_review"] == "pending"
    assert report["pass"] is False
    assert report["routing"]["codex_version"] == "not_executed"
    encoded = json.dumps(report)
    assert "Which baseline?" not in encoded and "Synthetic declared" not in encoded
    assert str(inputs.parent) not in encoded
    # A failed negative cannot be counted as a usable abstention.
    failed = copy.deepcopy(rows)
    failed[20]["usable_decision"] = False
    failed[20]["error"] = "decision_refused"
    assert scenario.build_report(value, failed, model="test:model")["framing_pass"] is False


@pytest.mark.asyncio
async def test_rejected_ack_and_forbidden_cancel_have_zero_real_effect() -> None:
    recorder = scenario.Recorder()
    outcome = await recorder.dispatch(objective="Synthetic research", utterance_id="turn_test")
    assert not outcome.accepted
    with pytest.raises(RuntimeError):
        await recorder.request_cancel("task_unrelated", reason="model")
    assert recorder.objectives == ["Synthetic research"]


def test_import_binding_refuses_wrong_checkout_before_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_realtime

    scenario.verify_imports()
    monkeypatch.setattr(hermes_realtime, "__file__", "C:/wrong/src/hermes_realtime/__init__.py")
    with pytest.raises(ValueError):
        scenario.verify_imports()


def test_access_proof_is_required_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    path.write_text(
        json.dumps({"passed": True, "scenario_sha256": "a" * 64, "scenario_id": "fixture-v1"})
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    scenario.verify_access(path, digest, "a" * 64, "fixture-v1")
    with pytest.raises(ValueError):
        scenario.verify_access(path, "0" * 64, "a" * 64, "fixture-v1")
    with pytest.raises(ValueError):
        scenario.verify_access(path, digest, "b" * 64, "fixture-v1")
    with pytest.raises(ValueError):
        scenario.verify_access(path, digest, "a" * 64, "wrong-v1")
    path.write_text(
        json.dumps({"passed": False, "scenario_sha256": "a" * 64, "scenario_id": "fixture-v1"})
    )
    with pytest.raises(ValueError):
        scenario.verify_access(
            path, hashlib.sha256(path.read_bytes()).hexdigest(), "a" * 64, "fixture-v1"
        )


@pytest.mark.parametrize("kind", ["size", "hash", "duplicate"])
def test_exact_bounded_input_guard_regressions(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "input.json"
    control = {"data": "short"}
    path.write_text(json.dumps(control))
    expected = scenario.digest(path)
    assert scenario.read_json(path, expected)[0] == control
    if kind == "size":
        overhead = len(json.dumps({"data": ""}).encode())
        path.write_text(json.dumps({"data": "x" * (scenario.MAX_INPUT_BYTES + 1 - overhead)}))
        expected = None
    elif kind == "hash":
        expected = "0" * 64
    else:
        path.write_text('{"data":1,"data":2}')
        expected = None
    with pytest.raises(ValueError):
        scenario.read_json(path, expected)
    path.write_text(json.dumps(control))
    assert scenario.read_json(path, scenario.digest(path))[0] == control


def test_unresolved_workspace_guard_regression() -> None:
    bad, good = ["Resources reside at {workspace}."], ["Resources reside at an admitted root."]
    with pytest.raises(ValueError):
        scenario.prior_rows(bad)
    assert scenario.prior_rows(good) == tuple(good)


def test_empty_context_guard_regression(inputs: Path) -> None:
    scenario.load_scenario(inputs)
    descriptor = json.loads(inputs.read_text())
    descriptor["context"] = write_ref(
        inputs.parent, "empty.json", {"schema_version": 1, "prior_user_turns": []}
    )
    bad = inputs.parent / "empty-scenario.json"
    bad.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError):
        scenario.load_scenario(bad)


@pytest.mark.parametrize("field", ["passed", "scenario_sha256", "scenario_id"])
def test_access_admission_guard_regressions(tmp_path: Path, field: str) -> None:
    path = tmp_path / "access.json"
    control = {"passed": True, "scenario_sha256": "a" * 64, "scenario_id": "fixture-v1"}
    path.write_text(json.dumps(control))
    scenario.verify_access(path, scenario.digest(path), "a" * 64, "fixture-v1")
    broken = {**control, field: False if field == "passed" else "wrong"}
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        scenario.verify_access(path, scenario.digest(path), "a" * 64, "fixture-v1")
    path.write_text(json.dumps(control))


@pytest.mark.parametrize("field", ["head", "dirty"])
def test_clean_live_source_guard_regressions(field: str) -> None:
    control = {"head": "a" * 40, "dirty": False}
    bad = {**control, field: "b" * 40 if field == "head" else True}
    with pytest.raises(ValueError):
        scenario.verify_execution_binding(bad, "a" * 40, True)
    scenario.verify_execution_binding(control, "a" * 40, True)
    if field == "dirty":
        scenario.verify_execution_binding(bad, "a" * 40, False)


@pytest.mark.parametrize("field", ["runner_sha256", "scorer_sha256", "source_sha256"])
def test_execution_source_drift_guard_regressions(
    inputs: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    value = scenario.load_scenario(inputs)
    before = {field: "a" * 64}
    monkeypatch.setattr(scenario, "source_binding", lambda: before)
    scenario.verify_stability(value, before)
    monkeypatch.setattr(scenario, "source_binding", lambda: {field: "b" * 64})
    with pytest.raises(ValueError):
        scenario.verify_stability(value, before)
    monkeypatch.setattr(scenario, "source_binding", lambda: before)


def test_wrong_import_and_real_entrypoint_before_factory(
    inputs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_realtime

    real_path = hermes_realtime.__file__
    monkeypatch.setattr(hermes_realtime, "__file__", str(inputs.parent / "wrong.py"))
    with pytest.raises(ValueError):
        scenario.verify_imports()
    monkeypatch.setattr(hermes_realtime, "__file__", real_path)
    code = """
import runpy,sys
sys.path.insert(0,sys.argv[1])
import hermes_realtime
hermes_realtime.__file__=sys.argv[2]
from hermes_realtime import host_launcher
host_launcher._build_streaming_inference=lambda **kw: print('UNEXPECTED_FACTORY_CALL')
sys.argv=[sys.argv[3]]+sys.argv[4:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(ROOT / "src"),
            str(inputs.parent / "wrong.py"),
            str(ROOT / "scripts/evaluate_natural_work_scenario.py"),
            "--scenario",
            str(inputs),
            "--access-proof",
            str(inputs),
            "--access-proof-sha",
            "0" * 64,
            "--source-head",
            "a" * 40,
            "--model",
            "test:model",
            "--base-url",
            "http://127.0.0.1:1",
            "--live",
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert result.returncode == 2 and "UNEXPECTED_FACTORY_CALL" not in result.stdout
    assert json.loads(result.stdout.split("[natural-work-scenario] ")[1])["decision_attempts"] == 0


@pytest.mark.parametrize("invalid", [None, True, "", "0" * 63, "X" * 64])
def test_reference_requires_exact_hex_sha256(inputs: Path, invalid: object) -> None:
    scenario.load_scenario(inputs)
    descriptor = json.loads(inputs.read_text())
    descriptor["main"]["sha256"] = invalid
    inputs.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError):
        scenario.load_scenario(inputs)


def cli_arguments(inputs: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    proof = inputs.parent / "admitted-access.json"
    proof.write_text(
        json.dumps(
            {
                "passed": True,
                "scenario_sha256": scenario.digest(inputs),
                "scenario_id": "synthetic-grounded-v1",
            }
        )
    )
    monkeypatch.setattr(scenario, "source_binding", lambda: {"head": "a" * 40, "dirty": False})
    return [
        "--scenario",
        str(inputs),
        "--access-proof",
        str(proof),
        "--access-proof-sha",
        scenario.digest(proof),
        "--source-head",
        "a" * 40,
        "--model",
        "test:model",
        "--base-url",
        "http://127.0.0.1:1",
    ]


@pytest.mark.parametrize("condition", ["absent", "existing", "missing_parent"])
def test_receipt_reservation_precedes_every_live_call(
    inputs: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
) -> None:
    args = cli_arguments(inputs, monkeypatch) + ["--live"]
    calls = []

    async def unexpected(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("unexpected_inference")

    monkeypatch.setattr(scenario, "run_cases", unexpected)
    if condition != "absent":
        receipt = inputs.parent / "receipt.jsonl"
        if condition == "existing":
            receipt.write_text("Preserve this existing evidence.")
        else:
            receipt = inputs.parent / "absent-directory" / "receipt.jsonl"
        args += ["--private-receipt", str(receipt)]
    assert scenario.main(args) == 2
    assert calls == []


def test_partial_private_rows_survive_operational_refusal(
    inputs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli_arguments(inputs, monkeypatch)
    receipt = inputs.parent / "partial.jsonl"
    row = {
        "id": "nw-001",
        "speech": ["Synthetic final reply"],
        "readiness_attempted": True,
        "decision_attempted": True,
        "error": "operational_failure",
    }

    async def failed(*args, observations, private_sink, **kwargs):
        observations.append(row)
        scenario.write_private_row(private_sink, row)
        raise RuntimeError("synthetic_transport_failure")

    monkeypatch.setattr(scenario, "run_cases", failed)
    assert scenario.main(args + ["--live", "--private-receipt", str(receipt)]) == 2
    assert json.loads(receipt.read_text()) == row


def test_private_row_budget_and_partial_write_refusals() -> None:
    row = {"id": "nw-001", "speech": ["Synthetic final reply"]}
    sink = io.BytesIO()
    scenario.write_private_row(sink, row)
    assert json.loads(sink.getvalue()) == row
    sink.seek(8 * 1024 * 1024)
    with pytest.raises(ValueError):
        scenario.write_private_row(sink, row)

    class ShortWrite(io.BytesIO):
        def write(self, value):
            return 0

    with pytest.raises(OSError):
        scenario.write_private_row(ShortWrite(), row)
