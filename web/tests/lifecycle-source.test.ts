import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const mainSource = readFileSync(
  fileURLToPath(new URL("../src/main.ts", import.meta.url)),
  "utf8",
);

describe("browser lifecycle wiring", () => {
  it("releases residual remote playback before error-resume room teardown", () => {
    const blockStart = mainSource.indexOf("  if (errorResume) {", mainSource.indexOf("async function connect"));
    const blockEnd = mainSource.indexOf("  const localOperation = new AbortController();", blockStart);
    expect(blockStart).toBeGreaterThan(-1);
    expect(blockEnd).toBeGreaterThan(blockStart);

    const cleanupBlock = mainSource.slice(blockStart, blockEnd);
    const releaseIndex = cleanupBlock.indexOf(
      "activeRemoteTrack = releaseRemoteAudioBeforeRoomInvalidation(",
    );
    const roomInvalidationIndex = cleanupBlock.indexOf("room = null");
    const mediaReleaseIndex = cleanupBlock.indexOf("await releaseLocalMedia(residualRoom)");

    expect(releaseIndex).toBeGreaterThan(-1);
    expect(roomInvalidationIndex).toBeGreaterThan(releaseIndex);
    expect(mediaReleaseIndex).toBeGreaterThan(roomInvalidationIndex);
  });

  it("projects transient approvals and task state inside the conversation", () => {
    expect(mainSource).toContain("function projectApprovalCard(");
    expect(mainSource).toContain("function projectTaskStateCard(");
    expect(mainSource).not.toContain('element<HTMLElement>("approval-panel")');
    expect(mainSource).not.toContain('element<HTMLParagraphElement>("task-state")');
    expect(mainSource).toContain("function appendTranscriptBounded(");
    expect(mainSource).toContain("function updateTranscriptBounded(");
    expect(mainSource.match(/appendBounded\(transcript,/g) ?? []).toHaveLength(0);
    expect(mainSource.match(/transcriptRetention\.update\(/g)).toHaveLength(1);
    // Every pending approval keeps its own card: concurrent runs can each hold
    // one, and a live card must survive transcript eviction by repinning.
    expect(mainSource).toContain("const approvalCardViews = new Map<string, ApprovalCardView>()");
    expect(mainSource).toContain("actionableApprovalCard(obsolete) && !repinned.has(obsolete)");
    expect(mainSource).not.toContain("activeApprovalCard");
    expect(mainSource).not.toContain("Superseded by a newer approval request.");
    expect(mainSource).toContain("forgetEvictedTaskCard(obsolete)");
  });

  it("automatically rebinds after an authoritative terminal media disconnect", () => {
    const handlerStart = mainSource.indexOf(
      "activeRoom.on(RoomEvent.ConnectionStateChanged",
    );
    const handlerEnd = mainSource.indexOf("  activeRoom.on(\n    RoomEvent.TrackSubscribed", handlerStart);
    expect(handlerStart).toBeGreaterThan(-1);
    expect(handlerEnd).toBeGreaterThan(handlerStart);
    const handler = mainSource.slice(handlerStart, handlerEnd);
    const terminalStart = handler.indexOf(
      "if (terminalDisconnectIsAuthoritative(state, admittedRooms.has(activeRoom)))",
    );
    const terminalEnd = handler.indexOf("      return;", terminalStart);
    expect(terminalStart).toBeGreaterThan(-1);
    expect(terminalEnd).toBeGreaterThan(terminalStart);
    const terminalBranch = handler.slice(terminalStart, terminalEnd);
    expect(terminalBranch).toContain("void recoverTerminalMediaDisconnect();");
    expect(terminalBranch).not.toContain("void disconnectLocal();");

    const recoveryStart = mainSource.indexOf("async function recoverTerminalMediaDisconnect");
    const recoveryEnd = mainSource.indexOf("\n}\n", recoveryStart);
    expect(recoveryStart).toBeGreaterThan(-1);
    expect(recoveryEnd).toBeGreaterThan(recoveryStart);
    const recovery = mainSource.slice(recoveryStart, recoveryEnd);
    const disconnectIndex = recovery.indexOf("await disconnectLocal()");
    const reconnectIndex = recovery.indexOf("await connect()");
    expect(disconnectIndex).toBeGreaterThan(-1);
    expect(reconnectIndex).toBeGreaterThan(disconnectIndex);
  });

  it("renews server media authority when native reconnect reuses a live microphone", () => {
    const reconnectStart = mainSource.indexOf("async function reverifyMicrophoneAfterNativeReconnect");
    const reconnectEnd = mainSource.indexOf("\nasync function connect", reconnectStart);
    expect(reconnectStart).toBeGreaterThan(-1);
    expect(reconnectEnd).toBeGreaterThan(reconnectStart);
    const reconnect = mainSource.slice(reconnectStart, reconnectEnd);

    expect(reconnect).toContain("announceMediaActivation(activeCredential, incarnation)");
    expect(reconnect).not.toContain("reusedMicrophone\n      ? Promise.resolve()");
  });

  it("derives the strict duplex claim from the expanded session-ready event", () => {
    expect(mainSource).toContain(
      "naturalDuplexProfileEnabled({\n      conversationProfile: event.data.conversationProfile,\n      mode: event.data.mode,\n    })",
    );
  });

  it("rotates a poisoned public-event projection exactly once before reconnecting media", () => {
    const pollingStart = mainSource.indexOf("async function pollPublicEvents");
    const pollingEnd = mainSource.indexOf("\n}\n\ndocument.addEventListener", pollingStart);
    expect(pollingStart).toBeGreaterThan(-1);
    expect(pollingEnd).toBeGreaterThan(pollingStart);
    const polling = mainSource.slice(pollingStart, pollingEnd);

    expect(mainSource).toContain("let projectionResyncAttempted = false;");
    expect(mainSource).toContain("async function projectionResyncBrowserCredential(");
    expect(mainSource).toContain('connectionFetch("/api/v1/projection-resync"');
    expect(mainSource).toContain("if (projectionResyncAttempted) {");
    expect(mainSource).toContain("projectionResyncAttempted = true;");
    expect(mainSource).toContain("async function recoverProjectionResync");
    expect(polling).toContain("await recoverProjectionResync(signal);");
    expect(polling).toContain("return;");

    const recoveryStart = mainSource.indexOf("async function recoverProjectionResync");
    const recoveryEnd = mainSource.indexOf("\n}\n", recoveryStart);
    const recovery = mainSource.slice(recoveryStart, recoveryEnd);
    expect(recovery).toContain("eventSequence = 0;");
    expect(recovery).toContain("await disconnectLocal();");
    expect(recovery).toContain("await connect(true);");
  });

  it("keeps microphone onset presentation-only until server floor commitment", () => {
    const monitorStart = mainSource.indexOf("class MicrophoneSignalMonitor");
    const monitorEnd = mainSource.indexOf("const controller = new ClientController", monitorStart);
    const monitor = mainSource.slice(monitorStart, monitorEnd);
    expect(monitor).toContain("beginProvisionalSpeechYield(observedAt)");
    expect(monitor).toContain("quietProvisionalSpeechYield()");
    expect(monitor).not.toContain("requestSpeechYield(");

    const provisionalStart = mainSource.indexOf("function beginProvisionalSpeechYield");
    const exactStart = mainSource.indexOf("async function requestSpeechYield", provisionalStart);
    const provisional = mainSource.slice(provisionalStart, exactStart);
    expect(provisional).toContain("setSpeechRendererVolume(provisionalSpeechVolume)");
    expect(provisional).toContain("setSpeechRendererVolume(0)");
    expect(provisional).toContain("provisionalSpeechYieldController.quiet(currentSpeechTiming())");
    expect(provisional).not.toContain("/api/v1/yield");

    const projectionStart = mainSource.indexOf("function projectPublicEvent");
    const projectionEnd = mainSource.indexOf("async function pollPublicEvents", projectionStart);
    const projection = mainSource.slice(projectionStart, projectionEnd);
    expect(projection).toContain("parseSpeechAuthority(event.data)");
    expect(projection).toContain('"server_floor_claim_committed"');

    expect(mainSource).toContain("activeRemoteTrack instanceof RemoteAudioTrack");
    expect(mainSource).toContain("webAudioMix: true");
    expect(mainSource).toContain(
      "track.attach(remoteAudio);\n      setSpeechRendererEnabled(locallySilencedSpeech === null)",
    );
    expect(mainSource).toContain(
      "remoteAudio.dataset.streamId = streamId;\n      reconcileProvisionalSpeechAuthority(speechTimings.get(streamId) ?? null)",
    );
    expect(mainSource).toContain("setSpeechRendererEnabled(false)");

    const stopHandlerStart = mainSource.indexOf('stopSpeakingButton.addEventListener("click"');
    const stopHandler = mainSource.slice(stopHandlerStart, stopHandlerStart + 500);
    expect(stopHandler).toContain("requestSpeechYield()");
  });

  it("binds public-search consent to the authenticated browser lifecycle", () => {
    expect(mainSource).toContain('from "./search-egress-controls"');
    expect(mainSource).toContain("const searchEgressControls = mountSearchEgressControls(document");
    expect(mainSource).toContain("searchEgressControls.setInteractive(false)");
    expect(mainSource).toContain(
      'event.kind === "search_egress_status"',
    );
    expect(mainSource).toContain(
      "searchEgressControls.projectStatus(parseSearchEgressStatus(event.data))",
    );
    expect(mainSource).toContain("searchEgressControls.setInteractive(connected)");
    expect(mainSource).toContain("searchEgressControls.reset()");
    const reconnectStart = mainSource.indexOf("const rebindRequestId =");
    const reconnectEnd = mainSource.indexOf("pendingRebindRequestId = null;", reconnectStart);
    const reconnect = mainSource.slice(reconnectStart, reconnectEnd);
    expect(reconnect).toContain("if (!sessionReplaced) resetSessionInputAuthority();");
    expect(mainSource).toContain("async function submitSearchEgressControl(");
    const submitStart = mainSource.indexOf("async function submitSearchEgressControl(");
    const submitEnd = mainSource.indexOf("\n}\n", submitStart);
    const submit = mainSource.slice(submitStart, submitEnd);
    expect(submit).toContain("authorization(activeCredential.token)");
    expect(submit).toContain('credentials: "omit"');
    expect(submit).toContain('referrerPolicy: "no-referrer"');
  });
});
