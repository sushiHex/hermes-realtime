"""The evidence markers that exist, as measured by ``tests/test_marker_catalogue.py``.

These tables are data for that test and hold nothing else. Each is bound to the source by
equality, so an entry is added, changed or deleted in the same change that adds, changes or
deletes what it describes. ``python -m tests.test_marker_catalogue`` prints the first five as
measured. The last four tables record debts and are meant to shrink (ADR 0007).
"""

# Marker name -> file -> number of places in that file that write the prefix. A place is a use
# of the prefix as text to write: a helper called from ten places counts once, where it prints.
# Line numbers are deliberately absent, so moving code does not change the table.
EMIT_SITES: dict[str, dict[str, int]] = {
    "approval-decision-refused": {
        "web/src/controller.ts": 1,
        "web/src/main.ts": 2,
    },
    "archived-worker": {
        "scripts/equivalence_process.py": 1,
    },
    "archived-workspace": {
        "scripts/equivalence_process.py": 1,
    },
    "binding-speech": {
        "src/hermes_realtime/conversation/worker.py": 1,
    },
    "bootstrap-capability": {
        "src/hermes_realtime/client/session.py": 1,
    },
    "browser-lease": {
        "src/hermes_realtime/client/runtime.py": 1,
    },
    "browser-model": {
        "src/hermes_realtime/client/session.py": 1,
    },
    "browser-model-admission": {
        "src/hermes_realtime/client/session.py": 1,
    },
    "browser-request": {
        "src/hermes_realtime/client/server.py": 1,
    },
    "browser-start-lock": {
        "src/hermes_realtime/client/session.py": 1,
    },
    "codex-access-credential": {
        "src/hermes_realtime/integration/codex_access_credentials.py": 1,
    },
    "codex-cache-probe": {
        "scripts/probe_codex_prompt_cache.py": 1,
    },
    "codex-error-notification": {
        "src/hermes_realtime/providers/codex_app_server.py": 1,
    },
    "codex-output-refusal": {
        "src/hermes_realtime/providers/codex_app_server.py": 1,
    },
    "codex-send-bound": {
        "src/hermes_realtime/providers/codex_app_server.py": 1,
    },
    "codex-session-auth": {
        "src/hermes_realtime/providers/codex_app_server.py": 1,
    },
    "codex-task-status": {
        "src/hermes_realtime/providers/codex_app_server.py": 1,
    },
    "codex-tool-refusal": {
        "src/hermes_realtime/providers/codex_app_server.py": 2,
    },
    "confirmed-natural-probe": {
        "scripts/probe_confirmed_natural_work.py": 1,
    },
    "confirmed-natural-work": {
        "scripts/qualify_confirmed_natural_work.py": 1,
    },
    "confirmed-preregistration": {
        "scripts/probe_confirmed_natural_work.py": 1,
        "scripts/qualify_confirmed_natural_work.py": 1,
    },
    "confirmed-provider-configuration": {
        "scripts/probe_confirmed_natural_work.py": 1,
    },
    "connection-recovery": {
        "web/src/main.ts": 6,
    },
    "consent-activation": {
        "src/hermes_realtime/evidence/runtime.py": 1,
    },
    "conversation-operation-admission": {
        "src/hermes_realtime/evidence/admission.py": 1,
    },
    "conversation-quiescence": {
        "src/hermes_realtime/evidence/admission.py": 1,
    },
    "conversation-reconfiguration": {
        "src/hermes_realtime/conversation/actions.py": 1,
    },
    "cpu-readiness": {
        "scripts/qualification_cpu_readiness.py": 1,
    },
    "declared-tool-path": {
        "scripts/release_gate.py": 1,
    },
    "desktop-mvp-cleanup": {
        "scripts/rehearse_desktop_mvp.py": 1,
    },
    "desktop-mvp-cleanup-settlement": {
        "scripts/rehearse_desktop_mvp.py": 1,
    },
    "desktop-mvp-listener-refusal": {
        "scripts/rehearse_desktop_mvp.py": 1,
    },
    "desktop-mvp-rehearsal": {
        "scripts/rehearse_desktop_mvp.py": 1,
    },
    "foreground-failure": {
        "src/hermes_realtime/conversation/streaming.py": 1,
    },
    "hermes-bridge-dispatch": {
        "src/hermes_realtime/integration/bridge.py": 1,
    },
    "hermes-bridge-hello": {
        "src/hermes_realtime/integration/bridge.py": 1,
    },
    "hermes-bridge-result": {
        "src/hermes_realtime/integration/bridge.py": 1,
    },
    "hermes-bridge-welcome": {
        "src/hermes_realtime/integration/bridge.py": 1,
    },
    "hermes-continuity": {
        "scripts/qualify_hermes_continuity.py": 1,
    },
    "hermes-continuity-ready": {
        "scripts/qualify_hermes_continuity.py": 1,
    },
    "hermes-dispatch-recovery": {
        "src/hermes_realtime/integration/api.py": 1,
    },
    "hermes-identity": {
        "scripts/real_gate_support.py": 1,
    },
    "hermes-latency": {
        "scripts/hermes_session_latency.py": 1,
    },
    "hermes-lookup-policy": {
        "src/hermes_realtime/integration/api.py": 1,
    },
    "hermes-model-source": {
        "src/hermes_realtime/host_launcher.py": 1,
    },
    "hermes-plugin-dispatch": {
        "src/hermes_realtime/integration/plugin.py": 4,
    },
    "hermes-plugin-result": {
        "src/hermes_realtime/integration/plugin.py": 1,
    },
    "hermes-restart-settlement": {
        "src/hermes_realtime/integration/api.py": 1,
    },
    "hermes-result-refusal": {
        "src/hermes_realtime/integration/api.py": 2,
    },
    "hermes-run-record-lock": {
        "src/hermes_realtime/integration/api.py": 1,
    },
    "hermes-tool-catalog": {
        "src/hermes_realtime/integration/api.py": 1,
        "src/hermes_realtime/integration/tool_catalog.py": 1,
    },
    "hermes-voice-archive": {
        "scripts/qualify_hermes_voice_archive.py": 1,
    },
    "hermes-voice-archive-step": {
        "scripts/qualify_hermes_voice_archive.py": 1,
    },
    "hermes-voice-delete": {
        "scripts/qualify_voice_delete.py": 1,
    },
    "hermes-voice-delete-step": {
        "scripts/qualify_voice_delete.py": 1,
    },
    "hermes-voice-memory": {
        "scripts/qualify_voice_memory.py": 1,
    },
    "hermes-voice-memory-boundary": {
        "scripts/qualify_voice_memory.py": 1,
    },
    "hermes-voice-memory-step": {
        "scripts/qualify_voice_memory.py": 1,
    },
    "hermes-voice-review": {
        "scripts/qualify_voice_review.py": 1,
    },
    "hermes-voice-review-step": {
        "scripts/qualify_voice_review.py": 2,
    },
    "hermes-web-evidence": {
        "src/hermes_realtime/integration/api.py": 1,
    },
    "history-capacity": {
        "web/src/main.ts": 1,
    },
    "host-fatal-exit": {
        "src/hermes_realtime/host_launcher.py": 1,
    },
    "host-speech": {
        "src/hermes_realtime/conversation/streaming.py": 2,
    },
    "host-status": {
        "src/hermes_realtime/conversation/task_status.py": 1,
    },
    "input-control-refusal": {
        "web/src/main.ts": 1,
    },
    "kokoro-mcp-host-refused": {
        "src/hermes_realtime/host_launcher.py": 1,
    },
    "kokoro-mcp-preview-refused": {
        "scripts/preview_cutover.py": 2,
    },
    "kokoro-mcp-refused": {
        "src/hermes_realtime/providers/kokoro_mcp.py": 1,
    },
    "kokoro-startup-witness-refused": {
        "src/hermes_realtime/providers/kokoro_worker.py": 1,
    },
    "kokoro-worker-failure": {
        "src/hermes_realtime/providers/kokoro_worker.py": 1,
    },
    "linux-docker": {
        "scripts/qualification_linux_producer.py": 1,
    },
    "linux-publisher": {
        "scripts/qualification_linux_producer.py": 1,
    },
    "local-model-admission": {
        "src/hermes_realtime/providers/ollama_session.py": 1,
    },
    "local-model-bootstrap": {
        "src/hermes_realtime/client/http.py": 1,
    },
    "local-model-catalog": {
        "src/hermes_realtime/providers/ollama_session.py": 1,
    },
    "local-model-choice": {
        "src/hermes_realtime/providers/ollama_session.py": 1,
    },
    "local-model-close": {
        "src/hermes_realtime/providers/ollama_session.py": 1,
    },
    "local-model-configuration": {
        "src/hermes_realtime/client/session.py": 1,
    },
    "local-model-edit-refusal": {
        "web/src/main.ts": 1,
    },
    "local-model-http": {
        "src/hermes_realtime/client/http.py": 1,
    },
    "local-model-prepare": {
        "src/hermes_realtime/providers/ollama_session.py": 1,
    },
    "local-model-selection": {
        "src/hermes_realtime/host_launcher.py": 1,
    },
    "local-model-selection-refusal": {
        "web/src/local-model-selection.ts": 3,
        "web/src/main.ts": 1,
    },
    "lookup-backend-setup": {
        "scripts/setup_lookup_backend.py": 1,
    },
    "lookup-evidence": {
        "src/hermes_realtime/integration/evidence_presentation.py": 2,
    },
    "lookup-frozen-search": {
        "scripts/lookup_frozen_search.py": 1,
    },
    "lookup-mcp": {
        "scripts/lookup_backend_mcp.py": 1,
    },
    "lookup-model": {
        "scripts/lookup_backend_model.py": 1,
    },
    "lookup-model-capture": {
        "scripts/lookup_model_capture.py": 1,
    },
    "lookup-owned": {
        "scripts/lookup_backend_task.py": 1,
    },
    "lookup-policy-stage": {
        "scripts/stage_lookup_policy.py": 1,
    },
    "lookup-transport-refusal": {
        "scripts/lookup_backend_gateway.py": 1,
    },
    "moonshine-finalization-refusal": {
        "src/hermes_realtime/providers/moonshine.py": 1,
    },
    "native-credential-probe": {
        "src/hermes_realtime/integration/native_credential_probe.py": 1,
    },
    "native-credential-refused": {
        "scripts/preview_cutover.py": 2,
        "src/hermes_realtime/integration/native_web_evidence.py": 1,
    },
    "native-git-discovery": {
        "scripts/candidate_e2e_fast_track.py": 1,
    },
    "native-git-probe": {
        "scripts/candidate_e2e_fast_track.py": 1,
    },
    "native-journey-refusal": {
        "scripts/native_run_witness.py": 1,
    },
    "native-typed-journey": {
        "scripts/native_typed_journey.py": 1,
    },
    "natural-gate-cancel": {
        "scripts/real_natural_work_gate.py": 2,
    },
    "natural-work-scenario": {
        "scripts/evaluate_natural_work_scenario.py": 1,
    },
    "ollama-catalog": {
        "src/hermes_realtime/providers/ollama_models.py": 1,
    },
    "ollama-context": {
        "src/hermes_realtime/host_launcher.py": 1,
        "src/hermes_realtime/providers/ollama.py": 1,
    },
    "ollama-context-observation": {
        "src/hermes_realtime/providers/ollama.py": 1,
    },
    "ollama-discovery": {
        "src/hermes_realtime/providers/ollama_models.py": 1,
    },
    "ollama-metadata": {
        "src/hermes_realtime/providers/ollama_models.py": 1,
    },
    "ollama-profile": {
        "src/hermes_realtime/host_launcher.py": 6,
        "src/hermes_realtime/providers/ollama_models.py": 2,
    },
    "ollama-prompt": {
        "src/hermes_realtime/providers/ollama.py": 1,
    },
    "ollama-refusal": {
        "src/hermes_realtime/providers/ollama.py": 1,
    },
    "ollama-runtime-contract": {
        "src/hermes_realtime/providers/ollama.py": 1,
    },
    "ollama-stream": {
        "src/hermes_realtime/providers/ollama.py": 1,
    },
    "ollama-work": {
        "src/hermes_realtime/providers/ollama.py": 1,
    },
    "output-style": {
        "src/hermes_realtime/client/http.py": 1,
        "src/hermes_realtime/client/session.py": 1,
    },
    "pinned-input": {
        "scripts/local_livekit.py": 1,
    },
    "playback-volume-preference": {
        "web/src/playback-volume.ts": 1,
    },
    "presentation-credential-refused": {
        "src/hermes_realtime/integration/presentation_transport.py": 1,
    },
    "projection-replay-refusal": {
        "src/hermes_realtime/client/projection.py": 1,
    },
    "provider-qualification": {
        "scripts/evaluate_natural_work_routing.py": 1,
    },
    "qualification-checkpoint": {
        "src/hermes_realtime/host_launcher.py": 1,
    },
    "qualification-launch": {
        "scripts/qualification_launch.py": 1,
    },
    "real-hermes-gate": {
        "scripts/real_hermes_api_gate.py": 1,
    },
    "session-model-refusal": {
        "web/src/protocol.ts": 1,
    },
    "speech-cleanup-refusal": {
        "src/hermes_realtime/conversation/streaming.py": 1,
    },
    "speech-consumer-cleanup": {
        "src/hermes_realtime/conversation/streaming.py": 1,
    },
    "speech-source": {
        "src/hermes_realtime/conversation/context.py": 1,
    },
    "speech-stop": {
        "src/hermes_realtime/livekit/adapter.py": 1,
    },
    "status-policy-refusal": {
        "src/hermes_realtime/conversation/output_style.py": 1,
    },
    "storage-finalization": {
        "scripts/storage_process.py": 1,
    },
    "storage-worker": {
        "scripts/storage_process.py": 1,
    },
    "target-resolution": {
        "scripts/natural_work_target_resolution.py": 1,
    },
    "task-command-refusal": {
        "src/hermes_realtime/conversation/commands.py": 1,
    },
    "task-command-refused": {
        "src/hermes_realtime/conversation/work_tools.py": 1,
    },
    "task-command-settlement": {
        "src/hermes_realtime/conversation/work_tools.py": 1,
    },
    "task-control-refusal": {
        "web/src/task-controls.ts": 2,
    },
    "task-proposal": {
        "src/hermes_realtime/conversation/task_proposals.py": 1,
    },
    "task-proposal-card": {
        "web/src/task-proposals.ts": 2,
    },
    "task-proposal-control": {
        "src/hermes_realtime/client/session.py": 1,
    },
    "task-proposal-decision": {
        "web/src/task-proposals.ts": 2,
    },
    "task-proposal-http": {
        "src/hermes_realtime/client/http.py": 1,
    },
    "task-proposal-projection": {
        "src/hermes_realtime/client/projection.py": 1,
    },
    "task-proposal-protocol": {
        "web/src/protocol.ts": 1,
    },
    "task-result-admission": {
        "src/hermes_realtime/conversation/tasks.py": 1,
    },
    "task-result-display": {
        "src/hermes_realtime/conversation/result_projection.py": 1,
    },
    "task-result-projection": {
        "src/hermes_realtime/client/projection.py": 1,
    },
    "task-result-refusal": {
        "src/hermes_realtime/conversation/tasks.py": 1,
    },
    "task-start-policy": {
        "src/hermes_realtime/conversation/task_proposals.py": 1,
    },
    "test-ownership": {
        "scripts/release_gate.py": 1,
    },
    "voice-archive": {
        "src/hermes_realtime/companion/archive.py": 1,
    },
    "voice-archive-lease": {
        "src/hermes_realtime/companion/archive.py": 1,
    },
    "voice-archive-open": {
        "src/hermes_realtime/companion/archive.py": 1,
    },
    "voice-archive-path": {
        "scripts/qualify_voice_archive_path.py": 1,
    },
    "voice-archive-send": {
        "src/hermes_realtime/integration/voice_archive.py": 1,
    },
    "voice-companion": {
        "src/hermes_realtime/companion/host.py": 1,
        "src/hermes_realtime/hermes_plugin.py": 1,
    },
    "voice-deletes": {
        "src/hermes_realtime/integration/voice_tail.py": 1,
    },
    "voice-forget": {
        "src/hermes_realtime/companion/forget.py": 1,
    },
    "voice-forget-send": {
        "src/hermes_realtime/integration/voice_forget.py": 1,
    },
    "voice-memory": {
        "src/hermes_realtime/companion/host.py": 2,
    },
    "voice-memory-receive": {
        "src/hermes_realtime/integration/voice_memory.py": 1,
    },
    "voice-memory-stream": {
        "src/hermes_realtime/integration/bridge.py": 1,
    },
    "voice-preview": {
        "scripts/preview_cutover.py": 1,
    },
    "voice-preview-host-failure": {
        "scripts/preview_cutover.py": 1,
    },
    "voice-review": {
        "src/hermes_realtime/companion/review.py": 1,
    },
    "voice-review-close": {
        "src/hermes_realtime/integration/voice_tail.py": 2,
    },
    "voice-review-send": {
        "src/hermes_realtime/integration/voice_review.py": 1,
    },
    "voice-tail": {
        "src/hermes_realtime/integration/voice_tail.py": 1,
    },
    "voice-tail-lock": {
        "src/hermes_realtime/integration/voice_tail.py": 1,
    },
    "voice-tail-outbox": {
        "src/hermes_realtime/integration/voice_tail.py": 5,
    },
    "web-evidence": {
        "src/hermes_realtime/integration/native_web_evidence.py": 1,
    },
}

# File -> marker -> values at a write site that the scan cannot show to be a literal, a
# number, a flag or a member of a literal set. Each entry is ``field: kind: source``.
# A marker named ``*function`` is a helper that takes the prefix as an argument; its
# entries are what the helper itself adds. ``*`` as a field is the whole payload.
UNPROVEN_VALUES: dict[str, dict[str, tuple[str, ...]]] = {
    "scripts/equivalence_process.py": {
        "archived-worker": (
            "child_events: call: _read_worker_exit_events(progress_read)",
        ),
    },
    "scripts/evaluate_natural_work_routing.py": {
        "provider-qualification": (
            "category: attribute: error.code",
            "completed_cases: attribute: error.completed_cases",
            "secondary_categories: attribute: error.secondary_categories",
        ),
    },
    "scripts/evaluate_natural_work_scenario.py": {
        "natural-work-scenario": (
            "*: bound: report",
            "*: call: build_development_report(value, rows, model=args.model)",
            "*: call: build_report(value, rows, model=args.model)",
            "decision_attempts: other: (row['decision_attempted'] for row in rows)",
            "readiness_attempts: other: (row['readiness_attempted'] for row in rows)",
            "scenario_id: attribute: value.scenario_id",
            "source_binding: call: source_binding()",
        ),
    },
    "scripts/hermes_session_latency.py": {
        "hermes-latency": (
            "failure: class-name: type(error).__name__",
        ),
    },
    "scripts/local_livekit.py": {
        "pinned-input": (
            "category: call: verify_upstream(fetch=_download)",
        ),
    },
    "scripts/lookup_backend_gateway.py": {
        "lookup-transport-refusal": (
            "callerCategory: call: denial_caller_category()",
            "denialCount: subscript: witness['transportDenialCount']",
            "extractRequests: subscript: witness['extractRequests']",
            "metadataRequests: call: witness.get('metadataRequests', 0)",
            "searchRequests: subscript: witness['searchRequests']",
        ),
    },
    "scripts/lookup_backend_mcp.py": {
        "lookup-mcp": (
            "accepted: other: await perform(base_url=base_url, model=model, deadline=deadline, s~",
            "failure_category: subscript: error.args[0]",
        ),
    },
    "scripts/lookup_backend_model.py": {
        "lookup-model": (
            "*: other: {key: state[key] for key in ('category', 'stage', 'inference_request_coun~",
        ),
    },
    "scripts/lookup_model_capture.py": {
        "lookup-model-capture": (
            "*: call: self.summary()",
        ),
    },
    "scripts/native_typed_journey.py": {
        "native-typed-journey": (
            "failure: class-name: type(error).__name__",
            "failure: text: str(error)",
            "osError: call: _os_error_evidence(error)",
        ),
    },
    "scripts/natural_work_target_resolution.py": {
        "target-resolution": (
            "refusal: class-name: type(error).__name__",
            "refusal: text: str(error)",
        ),
    },
    "scripts/preview_cutover.py": {
        "voice-preview": (
            "*: call: _failure_status(error)",
            "*: call: adopt_state(args.plan, args.source)",
            "*: call: launch_preview(args.active)",
            "*: call: make_speech_receipt(load_plan(args.plan))",
            "*: call: preflight(plan, args.home)",
            "*: call: stop_preview(args.output)",
            "category: subscript: result['category']",
            "ports: other: [plan.ports[name] for name in ('api', 'companion', 'livekit', 'host')]",
        ),
        "voice-preview-host-failure": (
            "failure_kind: subscript: _failure_status(error)['failure_kind']",
            "stage: subscript: facts['stage']",
        ),
    },
    "scripts/probe_codex_prompt_cache.py": {
        "codex-cache-probe": (
            "*: call: _failed('launch', 'Interrupted')",
            "*: call: _failed('launch', type(error).__name__)",
            "*: call: asyncio.run(run_probe(transport_factory=transport_factory, model=args.mode~",
        ),
    },
    "scripts/probe_confirmed_natural_work.py": {
        "confirmed-natural-probe": (
            "**: call: fixture_metadata(args.fixture_version)",
            "*: call: build_report(rows, fixture_version=args.fixture_version)",
            "codex_effort: attribute: args.codex_effort",
            "inference_provider: attribute: args.inference_provider",
            "model_digest: attribute: args.model_digest",
            "source_binding.codex_executable_sha256: attribute: args.codex_executable_sha256",
            "source_binding.probe_sha256: call: harness.digest(Path(__file__))",
            "source_binding: call: harness.source_binding()",
        ),
    },
    "scripts/qualification_cpu_readiness.py": {
        "cpu-readiness": (
            "outcome: subscript: report['outcome']",
        ),
    },
    "scripts/qualification_linux_producer.py": {
        "linux-publisher": (
            "*: call: budget.summary(phase=phase, outcome=outcome)",
        ),
    },
    "scripts/qualify_confirmed_natural_work.py": {
        "confirmed-natural-work": (
            "*: call: asyncio.run(run_controls())",
        ),
    },
    "scripts/qualify_hermes_continuity.py": {
        "hermes-continuity": (
            "failure: class-name: type(error).__name__",
            "hermes: call: installed_hermes_identity(hermes.version, PINNED_HERMES / 'source')",
            "passed: call: _passed(observed, model.unattributed, identity)",
            "scenarios: other: await _scenario(name, nonce, model, hermes, directory)",
            "unattributed: attribute: model.unattributed",
        ),
        "hermes-continuity-ready": (
            "version: attribute: hermes_cli.__version__",
        ),
    },
    "scripts/qualify_hermes_voice_archive.py": {
        "hermes-voice-archive": (
            "failure: class-name: type(error).__name__",
            "hermes: call: installed_hermes_identity(str(version), PINNED_HERMES / 'source')",
            "measurements: bound: measured",
            "passed: call: _passed(observed, identity)",
        ),
        "hermes-voice-archive-step": (
            "*: other: await _cap(worker)",
            "*: other: await _crash(worker, options[0])",
            "*: other: await _dedup(worker)",
            "*: other: await _durability(worker)",
            "*: other: await _foreign_step(worker, options[0])",
            "*: other: await _holders(worker)",
            "*: other: await _lease(worker, options[0])",
            "*: other: await _noop(worker)",
            "*: other: await _occupied(worker)",
            "*: other: await _recover(worker)",
            "*: other: await _reopen(worker)",
            "*: other: await _seed(worker)",
            "*: other: await _surface(worker)",
            "*: other: await _verify(worker)",
        ),
    },
    "scripts/qualify_voice_archive_path.py": {
        "voice-archive-path": (
            "cleanup_failures: class-name: type(error).__name__",
            "contender: call: json.loads((hermes_home / 'contend.json').read_text(encoding='utf-~",
            "control: call: json.loads((realtime / 'control.json').read_text(encoding='utf-8'))",
            "failure: class-name: type(error).__name__",
            "fidelity: call: _fidelity(truth, sent, archived, tail)",
            "hermes: call: installed_hermes_identity(version, PINNED_HERMES / 'source')",
            "partition.category: subscript: partition['category']",
            "passed: call: _passed(evidence, identity)",
            "reads: call: _reads(realtime)",
            "sends: call: _sends(sent)",
            "steps.companion_ready: call: companion.ready()",
            "steps.companion_ready_again: call: companion.ready()",
            "steps.companion_unloaded: call: companion.stop()",
            "steps.companion_unloaded_again: call: companion.stop()",
            "steps.control: call: _run(python, realtime, port, token, 'control', str(port))",
            "steps.drain: call: _run(python, realtime, port, token, 'drain', str(port))",
            "steps.fill: call: _run(python, realtime, port, token, 'fill', str(port))",
            "steps.second_process: call: spawn('contend').process.wait(timeout=_STEP_TIMEOUT_SEC~",
            "steps.settle: call: _run(python, realtime, port, token, 'settle', str(port))",
            "steps.trailing: call: _run(python, realtime, port, token, 'trailing', str(port))",
        ),
    },
    "scripts/qualify_voice_delete.py": {
        "hermes-voice-delete": (
            "failure: class-name: type(error).__name__",
            "hermes: call: installed_hermes_identity((home / 'hermes-version.txt').read_text(enc~",
            "observed.absence.complete_after_child_removed: subscript: foreign_result['complete_~",
            "observed.absence.foreign_child_pending: subscript: foreign_result['pending']",
            "observed.absence: subscript: core['absence']",
            "observed.fences: subscript: core['fences']",
            "observed.learned_limit.model_requests: other: (count for (case, _model), count in m~",
            "observed.learned_limit: subscript: core['learned_limit']",
            "observed.recovery.no_resurrection: subscript: again_result['no_resurrection']",
            "observed.recovery.restart_complete: subscript: restarted_result['complete']",
            "observed.recovery.successor_complete: subscript: succession['gateway_after_exit']",
            "observed.running_review: subscript: running['running_review']",
            "observed.scope: subscript: core['scope']",
            "observed.succession: other: await _succession(python, succession_home, url, token)",
            "observed.unattributed: attribute: model.unattributed",
            "passed: call: _passed(observed, identity)",
            "residue: subscript: core['residue']",
        ),
        "hermes-voice-delete-step": (
            "*: call: _host_actor(home, scenario)",
            "*: other: await _core(home)",
            "*: other: await _foreign(home)",
            "*: other: await _restart(home)",
            "*: other: await _running(home)",
        ),
    },
    "scripts/qualify_voice_memory.py": {
        "hermes-voice-memory": (
            "failure: class-name: type(error).__name__",
            "hermes: call: installed_hermes_identity('0.21.0', PINNED_HERMES / 'source')",
            "observed.authority: subscript: opened['authority']",
            "observed.bounds.model_requests: call: model.calls.values()",
            "observed.bounds: subscript: bounds['bounds']",
            "observed.capability: subscript: guards['capability']",
            "observed.fail_closed: subscript: guards['fail_closed']",
            "observed.freshness.finished_before_open: subscript: seed_item['finished']",
            "observed.freshness.post_review_refresh: subscript: opened_item['post_review_refresh~",
            "observed.freshness.reads_turn: subscript: latency['reads_turn']",
            "observed.freshness.visible_at_open: subscript: opened_item['learned']",
            "observed.isolation: subscript: guards['isolation']",
            "observed.latency: subscript: opened['latency']",
            "observed.proposal_boundary: subscript: opened['proposal_boundary']",
            "observed.recall.gap_hours: subscript: gap_item['hours']",
            "observed.recall.newest_correction: subscript: opened_item['newest_correction']",
            "observed.recall.next: subscript: opened_item['readback']",
            "observed.recall.non_ascii_over_cap: subscript: seed_item['non_ascii_over_cap']",
            "observed.recall.restart: subscript: opened_item['readback']",
            "observed.review_model_calls: call: model.calls.values()",
            "observed.unattributed: attribute: model.unattributed",
            "passed: call: _passed(observed, evidence['hermes'])",
            "proposal_boundary: subscript: opened['proposal_boundary']",
        ),
        "hermes-voice-memory-step": (
            "*: other: await _bounds(home)",
            "*: other: await _guards(home)",
            "*: other: await _open_memory(home, refresh=False, gap=True)",
            "*: other: await _open_memory(home, refresh=True, gap=False)",
            "*: other: await _seed(home)",
        ),
    },
    "scripts/qualify_voice_review.py": {
        "hermes-voice-review": (
            "failure: class-name: type(error).__name__",
            "hermes: call: installed_hermes_identity('0.21.0', PINNED_HERMES / 'source')",
            "observed.unattributed: attribute: model.unattributed",
            "observed: other: await _worker_step(python, home, url, scenario)",
            "passed: call: _passed(observed, identity)",
        ),
        "hermes-voice-review-step": (
            "*: other: await _attribution(worker)",
            "*: other: await _boundary(worker)",
            "*: other: await _close_integrity(worker)",
            "*: other: await _confinement(worker)",
            "*: other: await _corrections(worker)",
            "*: other: await _coverage(worker)",
            "*: other: await _guards_cold(home)",
            "*: other: await _guards_warm(worker)",
            "*: other: await _lifecycle(worker)",
            "*: other: await _speech(worker)",
        ),
    },
    "scripts/real_hermes_api_gate.py": {
        "real-hermes-gate": (
            "*: call: _components(output.getvalue())",
            "*: call: log.evidence()",
            "category: attribute: error.category",
            "failure: class-name: type(error).__name__",
        ),
    },
    "scripts/real_natural_work_gate.py": {
        "natural-gate-cancel": (
            "attempts: attribute: handler.cancel_attempts",
        ),
    },
    "scripts/rehearse_desktop_mvp.py": {
        "desktop-mvp-cleanup": (
            "current.job_only: call: dict.fromkeys(states, 0)",
            "current.tracked: call: dict.fromkeys(states, 0)",
            "first_refusal: attribute: self.first_cleanup_refusal",
        ),
        "desktop-mvp-listener-refusal": (
            "category: attribute: error.category",
        ),
        "desktop-mvp-rehearsal": (
            "*: call: step.record()",
            "*: other: await rehearsal.close()",
            "category: attribute: error.category",
            "category: call: take_rehearsal_lock()",
            "notes.missing: call: preflight(args.ollama_model)",
            "outcomes: other: {str(record['step']): record['outcome'] for record in rehearsal.re~",
            "stand_in_requests: attribute: rehearsal.stand_in.requests",
        ),
    },
    "scripts/release_gate.py": {
        "declared-tool-path": (
            "refusal: bound: refusal",
        ),
        "test-ownership": (
            "refusal: bound: refusal",
            "tests: bound: tests",
        ),
    },
    "scripts/setup_lookup_backend.py": {
        "lookup-backend-setup": (
            "*: call: run_owned(prepared)",
            "failure_category: subscript: error.args[0]",
            "failure_kind: class-name: type(error).__name__",
        ),
    },
    "scripts/storage_process.py": {
        "storage-finalization": (
            "*: call: _finalization_refusal_summary(failure)",
        ),
        "storage-worker": (
            "stage_count: bound: count",
            "worker_stage: bound: worker_stage",
        ),
    },
    "src/hermes_realtime/client/http.py": {
        "output-style": (
            "category: subscript: stage[0]",
        ),
    },
    "src/hermes_realtime/client/projection.py": {
        "task-result-projection": (
            "kind: argument: kind",
            "kind: bound: kind",
            "kind: subscript: key[0]",
            "limit: name: MAX_COMPLETED_RESULT_CHARS",
            "limit: name: MAX_LEGACY_RESULT_CHARS",
            "status: subscript: data['status']",
            "units: call: measure(text)",
            "units: name: MAX_COMPLETED_RESULT_CHARS",
            "units: name: MAX_LEGACY_RESULT_CHARS",
        ),
    },
    "src/hermes_realtime/client/runtime.py": {
        "browser-lease": (
            "failure: class-name: type(error).__name__",
        ),
    },
    "src/hermes_realtime/client/server.py": {
        "browser-request": (
            "status: call: error_status(error)",
        ),
    },
    "src/hermes_realtime/companion/archive.py": {
        "*_evidence": (
            "failure: class-name: type(error).__name__",
            "refusal: attribute: refusal.category",
        ),
        "voice-archive-lease": (
            "fence: bound: cause",
        ),
    },
    "src/hermes_realtime/companion/forget.py": {
        "voice-forget": (
            "category: other: await self._remaining(deletion)",
            "failure: class-name: type(error).__name__",
            "refusal: attribute: error.category",
            "refusal: attribute: refusal.category",
        ),
    },
    "src/hermes_realtime/companion/host.py": {
        "voice-companion": (
            "failure: class-name: type(error).__name__",
            "refusal: attribute: refusal.category",
        ),
        "voice-memory": (
            "refusal: attribute: error.category",
        ),
    },
    "src/hermes_realtime/companion/review.py": {
        "voice-review": (
            "count: call: self._store.recover_reviews()",
            "failure: class-name: type(error).__name__",
            "refusal: attribute: refusal.category",
        ),
    },
    "src/hermes_realtime/conversation/streaming.py": {
        "foreground-failure": (
            "failure_kind: call: next((label for cls, label in ((ValueError, 'value'), (TypeErro~",
        ),
    },
    "src/hermes_realtime/conversation/tasks.py": {
        "task-result-admission": (
            "category: bound: category",
            "field: bound: field",
            "source: attribute: update.payload.status.value",
        ),
    },
    "src/hermes_realtime/conversation/work_tools.py": {
        "task-command-refused": (
            "category: attribute: admission.refusal_category",
        ),
    },
    "src/hermes_realtime/evidence/admission.py": {
        "conversation-operation-admission": (
            "kind: attribute: exact_kind.value",
        ),
    },
    "src/hermes_realtime/evidence/runtime.py": {
        "consent-activation": (
            "offset_ms.create_durable: call: _milestone_offset_ms(durability_signal, deadline=wa~",
        ),
    },
    "src/hermes_realtime/host_launcher.py": {
        "host-fatal-exit": (
            "cause: attribute: self._cause",
            "flush: bound: flushed",
            "flush: other: await self._bounded(self._flush_conversation, self._flush_seconds)",
            "input: bound: stopped",
            "input: other: await self._bounded(self._stop_input, self._input_stop_seconds)",
        ),
        "qualification-checkpoint": (
            "cleanup: other: [stage for stage, _ in cleanup_failures]",
            "host_exit: call: _host_exit_category(host_error)",
            "refusal: call: _qualification_checkpoint_failure_category(checkpoint_failures[0])",
        ),
    },
    "src/hermes_realtime/integration/api.py": {
        "hermes-dispatch-recovery": (
            "status: bound: status",
        ),
        "hermes-restart-settlement": (
            "stopped: call: outcomes.count('stopped')",
            "unknown: call: outcomes.count('unknown')",
        ),
        "hermes-result-refusal": (
            "characters: name: MAX_COMPLETED_RESULT_CHARS",
        ),
    },
    "src/hermes_realtime/integration/codex_access_credentials.py": {
        "codex-access-credential": (
            "category: attribute: error.category",
        ),
    },
    "src/hermes_realtime/integration/evidence_presentation.py": {
        "lookup-evidence": (
            "kind: argument: kind",
        ),
    },
    "src/hermes_realtime/integration/voice_archive.py": {
        "voice-archive-send": (
            "cause: class-name: type(error).__name__",
            "fence: call: voice_refusal_label(reply.category)",
            "transient: call: voice_refusal_label(reply.category)",
        ),
    },
    "src/hermes_realtime/integration/voice_forget.py": {
        "voice-forget-send": (
            "cause: class-name: type(error).__name__",
            "outcome: call: voice_refusal_label(reason)",
        ),
    },
    "src/hermes_realtime/integration/voice_memory.py": {
        "voice-memory-receive": (
            "cause: class-name: type(error).__name__",
            "refusal: call: voice_refusal_label(event.category)",
        ),
    },
    "src/hermes_realtime/integration/voice_review.py": {
        "voice-review-send": (
            "cause: class-name: type(error).__name__",
            "refusal: call: voice_refusal_label(reply.category)",
        ),
    },
    "src/hermes_realtime/livekit/adapter.py": {
        "speech-stop": (
            "clamped: bound: clamped",
            "late_ms: attribute: audio.sample_rate_hz",
            "late_ms: bound: position",
            "late_ms: call: word_boundary_stop(audio, position)",
            "outcome: call: _word_stop_outcome(audio, position, stop)",
            "stop_block: attribute: audio.sample_rate_hz",
            "stop_block: bound: position",
            "tail_ms: attribute: audio.sample_rate_hz",
            "tail_ms: bound: position",
            "tail_ms: call: word_boundary_stop(audio, position)",
        ),
    },
    "src/hermes_realtime/providers/codex_app_server.py": {
        "codex-send-bound": (
            "category: argument: method",
        ),
        "codex-tool-refusal": (
            "tool: attribute: call.tool",
        ),
    },
    "src/hermes_realtime/providers/kokoro_mcp.py": {
        "kokoro-mcp-refused": (
            "category: argument: category",
        ),
    },
    "src/hermes_realtime/providers/kokoro_worker.py": {
        "kokoro-worker-failure": (
            "category: call: _FAILURE_CATEGORIES.get(exit_code, 'unknown_worker_failure')",
            "kind: call: next((label for exact_type, label in kinds if type(error) is exact_type~",
        ),
    },
    "src/hermes_realtime/providers/ollama.py": {
        "ollama-prompt": (
            "**: call: completion.get(source_name)",
            "**: call: self.prompt_metadata(messages)",
            "num_ctx: attribute: self._num_ctx",
            "timing_invalid_fields: bound: source_name",
        ),
        "ollama-refusal": (
            "requested_context_tokens: attribute: self._num_ctx",
            "status: attribute: error.code",
        ),
    },
    "src/hermes_realtime/providers/ollama_models.py": {
        "ollama-discovery": (
            "category: bound: category",
        ),
    },
    "web/src/main.ts": {
        "connection-recovery": (
            "*: value: JSON.stringify(refused)",
            "stage: value: stage",
        ),
        "history-capacity": (
            "activeCount: value: activeCount",
        ),
        "input-control-refusal": (
            "category: value: refusal",
            "pending_count: value: typedSubmissionCount",
        ),
        "local-model-selection-refusal": (
            "category: value: failure",
        ),
    },
    "web/src/playback-volume.ts": {
        "playback-volume-preference": (
            "category: value: category",
        ),
    },
    "web/src/task-proposals.ts": {
        "task-proposal-decision": (
            "category: value: category",
        ),
    },
}

# Marker name -> file -> number of places that recognise the prefix by hand.
# ``<any>`` is a regular expression that matches every ``[name] {json}`` line.
READERS: dict[str, dict[str, int]] = {
    "<any>": {
        "scripts/real_hermes_api_gate.py": 1,
        "scripts/rehearse_desktop_mvp.py": 1,
    },
    "foreground-failure": {
        "scripts/lookup_backend_logs.py": 2,
    },
    "hermes-continuity-ready": {
        "scripts/qualify_hermes_continuity.py": 2,
    },
    "hermes-voice-archive-step": {
        "scripts/qualify_hermes_voice_archive.py": 2,
    },
    "hermes-voice-delete-step": {
        "scripts/qualify_voice_delete.py": 4,
    },
    "hermes-voice-memory-step": {
        "scripts/qualify_voice_memory.py": 2,
    },
    "hermes-voice-review-step": {
        "scripts/qualify_voice_review.py": 3,
    },
    "kokoro-mcp-refused": {
        "scripts/lookup_backend_logs.py": 1,
    },
    "lookup-presentation": {
        "scripts/lookup_backend_gateway.py": 2,
    },
    "native-credential-refused": {
        "scripts/lookup_backend_logs.py": 1,
    },
    "ollama-prompt": {
        "scripts/probe_confirmed_natural_work.py": 2,
    },
    "presentation-credential-refused": {
        "scripts/lookup_backend_logs.py": 1,
    },
    "real-hermes-gate": {
        "scripts/rehearse_desktop_mvp.py": 1,
    },
    "voice-archive": {
        "scripts/qualify_hermes_voice_archive.py": 1,
    },
    "voice-archive-lease": {
        "scripts/qualify_hermes_voice_archive.py": 1,
    },
    "voice-archive-open": {
        "scripts/qualify_hermes_voice_archive.py": 1,
    },
    "voice-preview": {
        "scripts/preview_cutover.py": 2,
    },
    "voice-preview-host-failure": {
        "scripts/lookup_backend_logs.py": 1,
    },
    "voice-tail": {
        "scripts/rehearse_desktop_mvp.py": 1,
    },
}

# File -> number of prefixes assembled from parts, which the scan cannot name. The one
# in ``current_facts`` formats a source citation and is not a marker.
COMPUTED_PREFIXES: dict[str, int] = {
    "scripts/rehearse_desktop_mvp.py": 4,
    "src/hermes_realtime/providers/current_facts.py": 1,
}

# Names that are read or declared and that nothing in the scanned trees writes.
NEVER_WRITTEN: set[str] = {
    "lookup-presentation",
    "voice-archive-path-step",
}

# Entries of ``UNPROVEN_VALUES`` that were read in source and are free strings: text that a
# caller, a peer or another package chooses. (file, marker, entry) -> what the text is.
# The rest of that table is not cleared by being absent here: most of it is numbers and fixed
# labels the scan cannot follow, and some was not read. An entry leaves both tables in the
# change that removes the flow.
KNOWN_FREE_STRINGS: dict[tuple[str, str, str], str] = {
    (
        "scripts/qualify_hermes_continuity.py",
        "hermes-continuity-ready",
        "version: attribute: hermes_cli.__version__",
    ): "a version string from another package",
    (
        "scripts/release_gate.py",
        "declared-tool-path",
        "refusal: bound: refusal",
    ): "the class name of any exception that is not the guard's own refusal",
}
