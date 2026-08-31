import { describe, expect, it, vi } from "vitest";

import {
  ApprovalDecisionController,
  AssistantGenerationAuthority,
  assistantGenerationDisposition,
  authorizeRemoteAudio,
  AssistantTurnAssembler,
  bootstrapRequestParameters,
  BoundedRetention,
  canCommitCredentialRefresh,
  canCommitMicrophoneVerification,
  ClientController,
  connectionPresentation,
  ConnectionAttempt,
  enforceMicrophoneMuteAuthority,
  GenerationAuthority,
  formatSessionModelConfiguration,
  formatSessionTokenUsage,
  ForegroundPollFailurePolicy,
  microphoneCaptureOptions,
  microphoneProcessingTelemetry,
  microphoneDeviceChoices,
  microphoneProcessingDisposition,
  MicrophoneActivityGate,
  MicrophoneReadinessAuthority,
  microphoneMuteSettlementIsCurrent,
  microphoneSignalLevel,
  naturalDuplexProfileEnabled,
  RenderTimingTelemetry,
  markdownEmphasisRuns,
  karaokeGapPendingStates,
  karaokeSegmentPendingStates,
  modelSelectionFor,
  reasoningEffortLabel,
  reboundCredential,
  rebindFailureAllowsFreshBootstrap,
  rebindRequestParameters,
  ResponseLatencyStatistics,
  formatLatencyDuration,
  ObjectiveLatencyTracker,
  parseSpeechTiming,
  partialTranscriptDisposition,
  provisionalSpeechYieldAllowed,
  releaseRemoteAudio,
  releaseRemoteAudioBeforeRoomInvalidation,
  replaceRemoteAudio,
  remoteAudioTrackIsCurrent,
  ProvisionalSpeechYieldController,
  SpeechKaraokeClock,
  SpeechMediaStartRegistry,
  SpeechYieldController,
  SpeechTimingRegistry,
  SerializedAsyncQueue,
  settleMediaActivation,
  settleMicrophoneMuteChoice,
  settleMicrophoneVerification,
  settleReconnectedMicrophone,
  settleStableDisconnect,
  sameSpeechAuthority,
  sessionTogglePresentation,
  takeBootstrapCapability,
  takeOptionalBootstrapCapability,
  terminalDisconnectIsAuthoritative,
  taskMetaPresentation,
  taskStatusPresentation,
  UserTurnAssembler,
  userTranscriptProjectionDisposition,
} from "../src/controller";

describe("background task status presentation", () => {
  it("renders live elapsed time and preserves the final duration", () => {
    expect(taskStatusPresentation("active", 7_400)).toBe("Running · 0:07 elapsed");
    expect(taskStatusPresentation("completed", 62_900)).toBe("Completed in 1:02");
    expect(taskStatusPresentation("failed", 3_200)).toBe("Failed after 0:03");
    expect(taskStatusPresentation("interrupted", 3_200)).toBe("Stopped after 0:03");
  });

  it("rejects unknown states and invalid elapsed time", () => {
    expect(() => taskStatusPresentation("queued", 1_000)).toThrow(/status/i);
    expect(() => taskStatusPresentation("active", Number.NaN)).toThrow(/elapsed/i);
    expect(() => taskStatusPresentation("active", -1)).toThrow(/elapsed/i);
  });

  it("does not describe terminal work as still running", () => {
    expect(taskMetaPresentation("active")).toBe("You can keep talking while this runs.");
    expect(taskMetaPresentation("cancelling")).toBe("Stopping the task now.");
    expect(taskMetaPresentation("completed")).toBe("Background work finished.");
    expect(taskMetaPresentation("failed")).toBe("Background work failed.");
    expect(taskMetaPresentation("interrupted")).toBe("Background work stopped.");
    expect(taskMetaPresentation("rejected")).toBe("Background work was not started.");
  });
});

describe("session toggle presentation", () => {
  it.each([
    ["idle", "Connect", false, "connect"],
    ["disconnected", "Connect", false, "connect"],
    ["stopped", "Connect", false, "connect"],
    ["bootstrapping", "Connecting…", true, "connect"],
    ["connecting", "Connecting…", true, "connect"],
    ["preparing", "Stop session", false, "stop"],
    ["connected", "Stop session", false, "stop"],
    ["reconnecting", "Stop session", false, "stop"],
    ["error", "Connect", false, "connect"],
    ["stopping", "Stopping…", true, "stop"],
  ] as const)("maps %s to one unambiguous action", (state, label, disabled, action) => {
    expect(sessionTogglePresentation(state)).toEqual({ label, disabled, action });
  });

  it("fails closed when a one-shot launch capability has been consumed", () => {
    expect(sessionTogglePresentation("stopped", false)).toEqual({
      label: "Fresh launch required",
      disabled: true,
      action: "connect",
    });
    expect(sessionTogglePresentation("error", false)).toEqual({
      label: "Fresh launch required",
      disabled: true,
      action: "connect",
    });
  });

  it("keeps Stop session available after a rejected remote stop disconnects locally", () => {
    expect(sessionTogglePresentation("disconnected", true, true)).toEqual({
      label: "Stop session",
      disabled: false,
      action: "stop",
    });
  });
});

describe("karaoke unread projection", () => {
  it("dims every segment after the active presentation segment", () => {
    expect(
      karaokeSegmentPendingStates(
        ["chunk-1", "chunk-2", "chunk-3"],
        { chunkId: "speech-chunk-1", segmentId: "chunk-1" },
      ),
    ).toEqual([false, true, true]);
  });

  it("dims punctuation and whitespace after the active word", () => {
    expect(karaokeGapPendingStates(4, 1)).toEqual([false, false, true, true, true]);
  });
});

describe("assistant markdown emphasis", () => {
  it("projects matched movie-title delimiters as emphasized source ranges", () => {
    const source =
      "Try *The Princess Bride*, *The Mitchells vs. the Machines*, or *The LEGO Movie*.";
    const runs = markdownEmphasisRuns(source);

    expect(runs.map((run) => run.text).join("")).toBe(
      "Try The Princess Bride, The Mitchells vs. the Machines, or The LEGO Movie.",
    );
    expect(runs.filter((run) => run.emphasis === "italic").map((run) => run.text)).toEqual([
      "The Princess Bride",
      "The Mitchells vs. the Machines",
      "The LEGO Movie",
    ]);
    expect(runs.every((run) => source.slice(run.sourceStart, run.sourceEnd) === run.text)).toBe(
      true,
    );
  });

  it("keeps unmatched and escaped asterisks literal", () => {
    const source = String.raw`Keep \*literal\* and *unmatched markers`;

    expect(markdownEmphasisRuns(source)).toEqual([
      { emphasis: "none", sourceEnd: source.length, sourceStart: 0, text: source },
    ]);
  });

  it("preserves standard single, double, and triple delimiter semantics", () => {
    const source = "*italic* **bold** ***both*** _also italic_ __also bold__ ___also both___";

    expect(
      markdownEmphasisRuns(source)
        .filter((run) => run.emphasis !== "none")
        .map((run) => [run.text, run.emphasis]),
    ).toEqual([
      ["italic", "italic"],
      ["bold", "bold"],
      ["both", "bold-italic"],
      ["also italic", "italic"],
      ["also bold", "bold"],
      ["also both", "bold-italic"],
    ]);
  });
});

describe("model selection", () => {
  const catalog = {
    version: 1 as const,
    selectedModel: "gpt-5.6-terra",
    selectedEffort: "low",
    models: [
      {
        model: "gpt-5.6-terra",
        displayName: "GPT-5.6 Terra",
        description: "Fast",
        supportedEfforts: ["low", "medium"],
        defaultEffort: "medium",
      },
      {
        model: "gpt-5.6-sol",
        displayName: "GPT-5.6 Sol",
        description: "Deep",
        supportedEfforts: ["high", "xhigh"],
        defaultEffort: "high",
      },
    ],
  };

  it("preserves a supported effort and otherwise uses the model default", () => {
    expect(modelSelectionFor(catalog, "gpt-5.6-terra", "medium")).toEqual({
      model: "gpt-5.6-terra",
      effort: "medium",
    });
    expect(modelSelectionFor(catalog, "gpt-5.6-sol", "medium")).toEqual({
      model: "gpt-5.6-sol",
      effort: "high",
    });
    expect(() => modelSelectionFor(catalog, "unknown", "low")).toThrow(/model/i);
  });

  it("describes ultra as provider effort without claiming unavailable delegation", () => {
    expect(reasoningEffortLabel("max")).toBe("max");
    expect(reasoningEffortLabel("ultra")).toBe("ultra · provider effort");
  });
});

describe("microphone acoustic processing", () => {
  it("reports unsupported and omitted processor states as unknown without device metadata", () => {
    expect(
      microphoneProcessingTelemetry(
        {
          echoCancellation: true,
          autoGainControl: false,
          noiseSuppression: true,
        },
        {
          echoCancellation: true,
          noiseSuppression: false,
          deviceId: "must-not-escape",
        } as MediaTrackSettings,
      ),
    ).toEqual({
      version: 1,
      supported: {
        echoCancellation: true,
        autoGainControl: false,
        noiseSuppression: true,
        voiceIsolation: false,
      },
      applied: {
        echoCancellation: "enabled",
        autoGainControl: "unknown",
        noiseSuppression: "disabled",
        voiceIsolation: "unknown",
      },
    });
  });

  it("records one bounded monotonic renderer lifecycle without replay", () => {
    const timing = new RenderTimingTelemetry("speech_01234567", 1_000);
    timing.attached(1_004);
    timing.playing(1_040);

    expect(timing.advanced(1_055)).toEqual({
      streamId: "speech_01234567",
      subscribedToAttachMs: 4,
      attachToPlayingMs: 36,
      playingToAdvanceMs: 15,
    });
    expect(timing.advanced(1_060)).toBeNull();

    const delayedPlayingEvent = new RenderTimingTelemetry("speech_89abcdef", 2_000);
    delayedPlayingEvent.attached(2_004);
    delayedPlayingEvent.playing(2_040);
    expect(delayedPlayingEvent.advanced(2_030)?.playingToAdvanceMs).toBe(0);
  });

  it("rejects malformed stream identities and regressing renderer clocks", () => {
    expect(() => new RenderTimingTelemetry("private host", 1_000)).toThrow(/stream/i);
    const timing = new RenderTimingTelemetry("speech_01234567", 1_000);
    expect(() => timing.attached(999)).toThrow(/monotonic/i);
    timing.attached(1_001);
    expect(() => timing.playing(1_000)).toThrow(/monotonic/i);
  });

  it("derives a bounded local signal level and holds activity briefly", () => {
    expect(microphoneSignalLevel(new Float32Array([0, 0, 0]))).toBe(0);
    expect(microphoneSignalLevel(new Float32Array([0.03, -0.04, 0]))).toBeCloseTo(
      Math.sqrt((0.03 ** 2 + 0.04 ** 2) / 3),
    );
    expect(
      microphoneSignalLevel(new Float32Array([1, ...new Array<number>(255).fill(0)])),
    ).toBeCloseTo(0.0625);

    const gate = new MicrophoneActivityGate(0.02, 180);
    expect(gate.observe(0.01, 1000)).toBe(false);
    expect(gate.observe(0.03, 1010)).toBe(true);
    expect(gate.observe(0.01, 1189)).toBe(true);
    expect(gate.observe(0.01, 1190)).toBe(false);
    gate.reset();
    expect(gate.observe(0.01, 1050)).toBe(false);
  });

  it("requests acoustic echo cancellation without additional browser processors", () => {
    expect(microphoneCaptureOptions("device-1")).toEqual({
      deviceId: { exact: "device-1" },
      autoGainControl: false,
      echoCancellation: { exact: true },
      noiseSuppression: false,
      voiceIsolation: false,
    });
    expect(microphoneCaptureOptions("")).toEqual({
      autoGainControl: false,
      echoCancellation: { exact: true },
      noiseSuppression: false,
      voiceIsolation: false,
    });
  });

  it("requires browser-confirmed AEC-only capture", () => {
    expect(
      microphoneProcessingDisposition({
        echoCancellation: true,
        autoGainControl: false,
        noiseSuppression: false,
        voiceIsolation: false,
      }),
    ).toBe("aec-only");
    expect(microphoneProcessingDisposition({ echoCancellation: false })).toBe("aec-unavailable");
    expect(microphoneProcessingDisposition({})).toBe("aec-unavailable");
    expect(microphoneProcessingDisposition({ echoCancellation: true })).toBe("aec-only");
    expect(
      microphoneProcessingDisposition({
        echoCancellation: true,
        noiseSuppression: false,
        voiceIsolation: false,
      }),
    ).toBe("aec-only");
    expect(
      microphoneProcessingDisposition({
        echoCancellation: true,
        autoGainControl: false,
        voiceIsolation: false,
      }),
    ).toBe("aec-only");
    expect(
      microphoneProcessingDisposition({
        echoCancellation: true,
        autoGainControl: false,
        noiseSuppression: false,
      }),
    ).toBe("aec-only");
    expect(
      microphoneProcessingDisposition({
        echoCancellation: true,
        autoGainControl: true,
        noiseSuppression: false,
      }),
    ).toBe("additional-processing");
    expect(
      microphoneProcessingDisposition({
        echoCancellation: true,
        noiseSuppression: true,
      }),
    ).toBe("additional-processing");
  });

  it("serializes reconnect activations and cleans the exact stale result", async () => {
    const queue = new SerializedAsyncQueue();
    const order: string[] = [];
    let releaseFirst: (() => void) | undefined;
    const first = queue.run(async () => {
      order.push("first-start");
      await new Promise<void>((resolve) => {
        releaseFirst = resolve;
      });
      order.push("first-end");
      return "track-1";
    });
    const second = queue.run(async () => {
      order.push("second-start");
      return "track-2";
    });

    await vi.waitFor(() => expect(releaseFirst).toBeTypeOf("function"));
    expect(order).toEqual(["first-start"]);
    releaseFirst?.();
    expect(await first).toBe("track-1");
    expect(await second).toBe("track-2");
    expect(order).toEqual(["first-start", "first-end", "second-start"]);

    const cleaned: string[] = [];
    expect(
      await settleMicrophoneVerification(false, "track-1", async (track) => {
        cleaned.push(track);
      }),
    ).toEqual({ committed: false, value: "track-1" });
    expect(cleaned).toEqual(["track-1"]);
  });

  it("rejects a delayed media activation after its authority is superseded", async () => {
    let resolveActivation: (() => void) | undefined;
    const activation = new Promise<void>((resolve) => {
      resolveActivation = resolve;
    });
    let current = true;
    let cleanupCalls = 0;
    const settlement = settleMediaActivation(
      activation,
      () => current,
      async () => {
        cleanupCalls += 1;
      },
    );

    current = false;
    resolveActivation?.();

    await expect(settlement).resolves.toBe(false);
    expect(cleanupCalls).toBe(1);
  });

  it("cleans candidate media when activation rejects", async () => {
    let cleanupCalls = 0;
    const settlement = settleMediaActivation(
      Promise.reject(new Error("activation failed")),
      () => true,
      async () => {
        cleanupCalls += 1;
      },
    );

    await expect(settlement).rejects.toThrow("activation failed");
    expect(cleanupCalls).toBe(1);
  });

  it("deduplicates and bounds repeated microphone enumeration", () => {
    const repeated: Array<Pick<MediaDeviceInfo, "kind" | "deviceId" | "label">> = Array.from(
      { length: 200 },
      (_, index) => ({
      kind: "audioinput" as const,
      deviceId: `device-${index % 3}`,
      label: `Microphone ${index}`,
      }),
    );
    repeated.push({ kind: "videoinput" as const, deviceId: "camera", label: "Camera" });

    expect(microphoneDeviceChoices(repeated)).toEqual([
      { deviceId: "device-0", label: "Microphone 0" },
      { deviceId: "device-1", label: "Microphone 1" },
      { deviceId: "device-2", label: "Microphone 2" },
    ]);
  });

  it("projects only the latest out-of-order microphone enumeration", async () => {
    const authority = new GenerationAuthority();
    const projected: string[] = [];
    let resolveOlder: ((value: string) => void) | undefined;
    let resolveNewer: ((value: string) => void) | undefined;
    const project = async (value: Promise<string>): Promise<void> => {
      const generation = authority.issue();
      const resolved = await value;
      if (authority.owns(generation)) projected.push(resolved);
    };
    const older = project(
      new Promise<string>((resolve) => {
        resolveOlder = resolve;
      }),
    );
    const newer = project(
      new Promise<string>((resolve) => {
        resolveNewer = resolve;
      }),
    );

    resolveNewer?.("newer");
    await newer;
    resolveOlder?.("older");
    await older;

    expect(projected).toEqual(["newer"]);
  });

  it("invalidates pending enumeration authority without an exhaustible counter", () => {
    const authority = new GenerationAuthority();
    const pending = authority.issue();
    (authority as unknown as { generation: number }).generation = Number.MAX_SAFE_INTEGER;

    expect(() => authority.invalidate()).not.toThrow();
    expect(authority.owns(pending)).toBe(false);
  });
});

describe("launch capability", () => {
  it("uses a bearer only for diagnostic bootstrap", () => {
    expect(bootstrapRequestParameters(false, "diagnostic-capability")).toEqual({
      path: "/api/v1/bootstrap",
      bearer: "diagnostic-capability",
    });
    expect(bootstrapRequestParameters(true, null)).toEqual({
      path: "/api/v1/stable-bootstrap",
      bearer: null,
    });
    expect(() => bootstrapRequestParameters(false, null)).toThrow(/fresh bootstrap/i);
  });

  it("scrubs the fragment before exposing the capability to caller code", () => {
    const order: string[] = [];
    const replaceState = vi.fn(() => order.push("scrubbed"));

    const capability = takeBootstrapCapability(
      {
        hash: `#bootstrap=${"a".repeat(43)}`,
        pathname: "/",
        search: "",
      },
      { replaceState },
    );
    order.push("returned");

    expect(capability).toBe("a".repeat(43));
    expect(replaceState).toHaveBeenCalledWith(null, "", "/");
    expect(order).toEqual(["scrubbed", "returned"]);
  });

  it("accepts a stable same-origin launch without scrubbing or a capability", () => {
    const replaceState = vi.fn();
    expect(
      takeOptionalBootstrapCapability(
        { hash: "", pathname: "/", search: "" },
        { replaceState },
      ),
    ).toBeNull();
    expect(replaceState).not.toHaveBeenCalled();
  });

  it("rejects malformed nonempty fragments while preserving diagnostic scrubbing", () => {
    const replaceState = vi.fn();
    expect(() =>
      takeOptionalBootstrapCapability(
        { hash: "#unexpected", pathname: "/", search: "" },
        { replaceState },
      ),
    ).toThrow(/fresh bootstrap/i);
    expect(replaceState).toHaveBeenCalledWith(null, "", "/");
  });
});

describe("stable disconnect settlement", () => {
  it("does not release locally after a successful server stop", async () => {
    const calls: string[] = [];
    const result = await settleStableDisconnect(
      async () => {
        calls.push("stop");
      },
      async () => {
        calls.push("release");
      },
    );

    expect(result).toBe("stopped");
    expect(calls).toEqual(["stop"]);
  });

  it("always releases locally when the server stop fails", async () => {
    const calls: string[] = [];
    const result = await settleStableDisconnect(
      async () => {
        calls.push("stop");
        throw new Error("rejected");
      },
      async () => {
        calls.push("release");
      },
    );

    expect(result).toBe("local-release");
    expect(calls).toEqual(["stop", "release"]);
  });
});

describe("client state authority", () => {
  it("starts a fresh connection after an authoritative stop", () => {
    const controller = new ClientController();
    controller.beginConnect();
    controller.bootstrapReady();
    controller.preparingMicrophone();
    controller.connected();
    controller.beginStop();
    controller.stopped();

    controller.beginConnect();

    expect(controller.state).toBe("bootstrapping");
  });

  it("retries a connection after a failed attempt", () => {
    const controller = new ClientController();
    controller.beginConnect();
    controller.failed();

    controller.beginConnect();

    expect(controller.state).toBe("bootstrapping");
  });

  it("reconnects from a disconnected toggle action", () => {
    const controller = new ClientController();
    controller.beginConnect();
    controller.bootstrapReady();
    controller.disconnected();

    controller.beginConnect();

    expect(controller.state).toBe("bootstrapping");
  });

  it("rejects a second connect while the first bootstrap is active", () => {
    const controller = new ClientController();

    controller.beginConnect();

    expect(controller.state).toBe("bootstrapping");
    expect(() => controller.beginConnect()).toThrow(/cannot connect/i);
  });

  it("does not claim listening until media joins and microphone preparation completes", () => {
    const controller = new ClientController();
    const states: string[] = [];
    controller.subscribe((state) => states.push(state));

    controller.beginConnect();
    controller.bootstrapReady();
    controller.preparingMicrophone();

    expect(controller.state).toBe("preparing");
    expect(connectionPresentation("preparing", null, "waiting")).toEqual({
      label: "Preparing microphone…",
      presentationState: "preparing",
      headline: "Preparing microphone",
      detail: "Please wait before speaking. Audio capture is not ready yet.",
    });
    expect(() => controller.beginConnect()).toThrow(/cannot connect/i);

    controller.connected();
    expect(controller.state).toBe("connected");
    expect(connectionPresentation("connected", true, "waiting")).toEqual({
      label: "Connecting speech path…",
      presentationState: "preparing",
      headline: "Connecting speech path",
      detail: "Please wait before speaking. The server has not confirmed microphone audio yet.",
    });
    expect(connectionPresentation("connected", true, "timed-out")).toEqual({
      label: "Microphone not reaching server",
      presentationState: "error",
      headline: "Speech path unavailable",
      detail: "Reconnect to retry voice, or use typed input below.",
    });
    expect(connectionPresentation("connected", true, "ready")).toEqual({
      label: "Listening",
      presentationState: "connected",
      headline: "Listening — speak naturally",
      detail: "Pause briefly when you need to; your words stay together until your turn ends.",
    });
    expect(connectionPresentation("connected", false, "waiting").label).toBe(
      "Ready to type — microphone unavailable",
    );
    expect(states).toEqual(["idle", "bootstrapping", "connecting", "preparing", "connected"]);
  });

  it("reconnects a disconnected media transport without a second bootstrap", () => {
    const controller = new ClientController();
    controller.beginConnect();
    controller.bootstrapReady();
    controller.preparingMicrophone();
    controller.connected();
    controller.connected();
    controller.disconnected();
    controller.disconnected();
    expect(controller.state).toBe("disconnected");

    controller.beginReconnect();
    expect(controller.state).toBe("reconnecting");
    controller.preparingMicrophone();
    expect(controller.state).toBe("preparing");
    controller.connected();
    expect(controller.state).toBe("connected");
  });

  it("allows a disconnected media transport to terminate its server lease", () => {
    const controller = new ClientController();
    controller.beginConnect();
    controller.bootstrapReady();
    controller.preparingMicrophone();
    controller.connected();
    controller.disconnected();

    controller.beginStop();
    expect(controller.state).toBe("stopping");
    controller.stopFailed(false);
    expect(controller.state).toBe("disconnected");
  });
});

describe("credential refresh ownership", () => {
  it("cannot restore authority after stop or replacement", () => {
    const active = {};
    expect(canCommitCredentialRefresh(active, active, "connected")).toBe(true);
    expect(canCommitCredentialRefresh(active, null, "stopped")).toBe(false);
    expect(canCommitCredentialRefresh(active, {}, "connected")).toBe(false);
    expect(canCommitCredentialRefresh(active, active, "stopping")).toBe(false);
  });
});

describe("terminal reconnect credential rotation", () => {
  const active = {
    version: 1 as const,
    url: "wss://livekit.test",
    roomName: "hermes-local",
    participantIdentity: "browser_0123456789abcdef",
    workerIdentity: "worker_hermes_browser",
    expiresInSeconds: 60,
    token: "old.token.value",
  };

  it("requires a fresh participant identity under unchanged session authority", () => {
    const replacement = {
      ...active,
      participantIdentity: "browser_fedcba9876543210",
      token: "new.token.value",
    };
    expect(reboundCredential(active, replacement)).toBe(replacement);
    expect(() =>
      reboundCredential(active, {
        ...replacement,
        participantIdentity: active.participantIdentity,
      }),
    ).toThrow(/identity/i);
    expect(() =>
      reboundCredential(active, { ...replacement, workerIdentity: "worker_other" }),
    ).toThrow(/authority/i);
    expect(() =>
      reboundCredential(active, { ...replacement, roomName: "other-room" }),
    ).toThrow(/authority/i);
    expect(() =>
      reboundCredential(active, { ...replacement, url: "wss://other.test" }),
    ).toThrow(/authority/i);
  });

  it("uses Tailnet peer authority instead of an expiring browser bearer", () => {
    expect(rebindRequestParameters(true, active, "rebind_0123456789abcdef")).toEqual({
      path: "/api/v1/stable-rebind",
      bearer: null,
      body: JSON.stringify({
        participantIdentity: active.participantIdentity,
        requestId: "rebind_0123456789abcdef",
      }),
    });
    expect(rebindRequestParameters(false, active, "rebind_0123456789abcdef")).toEqual({
      path: "/api/v1/rebind",
      bearer: active.token,
      body: JSON.stringify({ requestId: "rebind_0123456789abcdef" }),
    });
    expect(rebindFailureAllowsFreshBootstrap(409)).toBe(true);
    expect(rebindFailureAllowsFreshBootstrap(403)).toBe(false);
    expect(rebindFailureAllowsFreshBootstrap(503)).toBe(false);
  });

  it("does not let an in-flight connect race terminal disconnect cleanup", () => {
    expect(terminalDisconnectIsAuthoritative("disconnected", true)).toBe(true);
    expect(terminalDisconnectIsAuthoritative("disconnected", false)).toBe(false);
    expect(terminalDisconnectIsAuthoritative("reconnecting", true)).toBe(false);
  });
});

describe("first-class microphone mute", () => {
  it("keeps a settled choice authoritative across reconnect when the track is reused", () => {
    const track = { id: "reused" };

    expect(microphoneMuteSettlementIsCurrent(track, track)).toBe(true);
    expect(microphoneMuteSettlementIsCurrent(track, { id: "replacement" })).toBe(false);
    expect(microphoneMuteSettlementIsCurrent(track, null)).toBe(false);
  });

  it("commits only after muting the exact current track", async () => {
    const track = { id: "current" };
    const calls: string[] = [];

    await expect(
      settleMicrophoneMuteChoice(
        track,
        true,
        async (candidate) => {
          calls.push(`mute:${candidate.id}`);
        },
        async (candidate) => {
          calls.push(`unmute:${candidate.id}`);
        },
        (candidate) => candidate === track,
      ),
    ).resolves.toBe(true);
    expect(calls).toEqual(["mute:current"]);
  });

  it("rejects a track replaced while the mute operation is pending", async () => {
    const track = { id: "stale" };
    let current: typeof track | null = track;

    const result = await settleMicrophoneMuteChoice(
      track,
      false,
      async () => undefined,
      async () => {
        current = null;
      },
      (candidate) => candidate === current,
    );

    expect(result).toBe(false);
  });
});

describe("foreground event polling recovery", () => {
  it("keeps retrying after an iOS background cycle until one foreground success", () => {
    const policy = new ForegroundPollFailurePolicy(5);
    policy.visibilityChanged("hidden");
    policy.visibilityChanged("visible");

    expect(Array.from({ length: 8 }, () => policy.recordFailure())).toEqual(
      Array.from({ length: 8 }, () => "retry"),
    );
    policy.recordSuccess();
    expect(Array.from({ length: 4 }, () => policy.recordFailure())).toEqual([
      "retry",
      "retry",
      "retry",
      "retry",
    ]);
    expect(policy.recordFailure()).toBe("terminal");
  });

  it("fails closed after the visible-session failure budget without a background cycle", () => {
    const policy = new ForegroundPollFailurePolicy(3);
    expect(policy.recordFailure()).toBe("retry");
    expect(policy.recordFailure()).toBe("retry");
    expect(policy.recordFailure()).toBe("terminal");
  });

  it("bounds retries after a background recovery cycle", () => {
    const policy = new ForegroundPollFailurePolicy(3);
    policy.visibilityChanged("hidden");
    policy.visibilityChanged("visible");

    expect(Array.from({ length: 8 }, () => policy.recordFailure())).toEqual(
      Array.from({ length: 8 }, () => "retry"),
    );
    expect(policy.recordFailure()).toBe("terminal");
  });
});

describe("native reconnect microphone authority", () => {
  it("reuses a live microphone without reacquiring permission", async () => {
    const track = { readyState: "live" };
    const calls: string[] = [];

    const result = await settleReconnectedMicrophone(
      track,
      (candidate) => candidate.readyState === "live",
      async () => {
        calls.push("create");
        return { readyState: "live" };
      },
      async () => {
        calls.push("cleanup");
      },
    );

    expect(result).toBe(track);
    expect(calls).toEqual([]);
  });

  it("cleans and replaces an ended microphone", async () => {
    const ended = { readyState: "ended" };
    const replacement = { readyState: "live" };
    const calls: string[] = [];

    const result = await settleReconnectedMicrophone(
      ended,
      (candidate) => candidate.readyState === "live",
      async () => {
        calls.push("create");
        return replacement;
      },
      async (candidate) => {
        expect(candidate).toBe(ended);
        calls.push("cleanup");
      },
    );

    expect(result).toBe(replacement);
    expect(calls).toEqual(["cleanup", "create"]);
  });

  it("reapplies mute authority to a reused live microphone", async () => {
    const track = { readyState: "live" };
    const muted: object[] = [];
    const result = await enforceMicrophoneMuteAuthority(
      track,
      true,
      async (candidate, shouldMute) => {
        expect(shouldMute).toBe(true);
        muted.push(candidate);
      },
      async () => undefined,
    );

    expect(result).toBe(track);
    expect(muted).toEqual([track]);
  });

  it("reapplies unmuted authority to a reused live microphone", async () => {
    const track = { readyState: "live" };
    const applied: boolean[] = [];
    const result = await enforceMicrophoneMuteAuthority(
      track,
      false,
      async (_candidate, shouldMute) => void applied.push(shouldMute),
      async () => undefined,
    );

    expect(result).toBe(track);
    expect(applied).toEqual([false]);
  });

  it("cleans a reused microphone when restoring mute authority fails", async () => {
    const track = { readyState: "live" };
    const cleaned: object[] = [];
    const result = await enforceMicrophoneMuteAuthority(
      track,
      true,
      async (_candidate, shouldMute) => {
        expect(shouldMute).toBe(true);
        throw new Error("mute failed");
      },
      async (candidate) => void cleaned.push(candidate),
    );

    expect(result).toBeNull();
    expect(cleaned).toEqual([track]);
  });

  it("commits only the current room generation while microphone preparation is pending", () => {
    const activeRoom = {};
    expect(
      canCommitMicrophoneVerification(4, 4, activeRoom, activeRoom, "preparing"),
    ).toBe(true);
    expect(
      canCommitMicrophoneVerification(3, 4, activeRoom, activeRoom, "preparing"),
    ).toBe(false);
    expect(canCommitMicrophoneVerification(4, 4, activeRoom, {}, "preparing")).toBe(false);
    expect(
      canCommitMicrophoneVerification(4, 4, activeRoom, activeRoom, "reconnecting"),
    ).toBe(false);
  });
});

describe("voice readiness authority", () => {
  it("rejects a delayed readiness proof from a replaced microphone incarnation", () => {
    const readiness = new MicrophoneReadinessAuthority<object, object>();
    const room = {};
    const originalMicrophone = {};
    readiness.activate(room, originalMicrophone, 1);

    expect(readiness.accepts(7, 1, room, originalMicrophone, "connected")).toBe(true);
    readiness.invalidate();
    const replacementMicrophone = {};
    readiness.activate(room, replacementMicrophone, 2);

    expect(readiness.accepts(7, 1, room, replacementMicrophone, "connected")).toBe(false);
    expect(readiness.accepts(6, 2, room, replacementMicrophone, "connected")).toBe(false);
    expect(readiness.accepts(7, 2, room, replacementMicrophone, "connected")).toBe(true);
    expect(readiness.accepts(8, 2, room, replacementMicrophone, "connected")).toBe(true);
    expect(readiness.accepts(7, 2, room, replacementMicrophone, "connected")).toBe(false);
    expect(readiness.accepts(8, 2, room, replacementMicrophone, "preparing")).toBe(false);
  });

  it("invalidates track ownership and resets session generation separately", () => {
    const readiness = new MicrophoneReadinessAuthority<object, object>();
    const room = {};
    const microphone = {};
    readiness.activate(room, microphone, 1);
    expect(readiness.accepts(7, 1, room, microphone, "connected")).toBe(true);

    readiness.invalidate();
    expect(readiness.accepts(7, 1, room, microphone, "connected")).toBe(false);

    const nextMicrophone = {};
    readiness.activate(room, nextMicrophone, 2);
    readiness.resetSession();
    readiness.activate(room, nextMicrophone, 1);
    expect(readiness.accepts(1, 1, room, nextMicrophone, "connected")).toBe(true);
  });
});

describe("remote audio authority", () => {
  it("enables natural duplex only from an exact server profile claim", () => {
    expect(
      naturalDuplexProfileEnabled({
        conversationProfile: "natural_v1",
        mode: "microphone_or_typed",
      }),
    ).toBe(true);
    expect(
      naturalDuplexProfileEnabled({ conversationProfile: "legacy", mode: "microphone_or_typed" }),
    ).toBe(false);
    expect(
      naturalDuplexProfileEnabled({
        conversationProfile: "natural_v1",
        enabled: true,
        mode: "microphone_or_typed",
      }),
    ).toBe(false);
    expect(naturalDuplexProfileEnabled({ conversationProfile: true })).toBe(false);
  });

  it("matches speech authority by exact public identifiers rather than object identity", () => {
    const first = {
      turnId: "turn_1",
      turnGeneration: 3,
      chunkId: "chunk_1",
      streamId: "stream_persistent",
    };
    const reparsed = { ...first };
    const nextChunk = { ...first, chunkId: "chunk_2" };
    const controller = new SpeechYieldController<typeof first>(sameSpeechAuthority);

    expect(sameSpeechAuthority(first, reparsed)).toBe(true);
    expect(sameSpeechAuthority(first, nextChunk)).toBe(false);
    expect(controller.claim(first)).toBe(first);
    expect(controller.settle(first, reparsed, true)).toBe("silenced");
    expect(controller.claim(first)).toBe(first);
    expect(controller.settle(first, nextChunk, true)).toBe("recover");
  });

  it("serializes exact local-yield claims and recovers only after an unmatched response", () => {
    const controller = new SpeechYieldController<object>();
    const first = {};
    const replacement = {};

    expect(controller.claim(first)).toBe(first);
    expect(controller.claim(first)).toBeNull();
    expect(controller.settle(first, replacement, false)).toBe("recover");
    expect(controller.claim(replacement)).toBe(replacement);
    expect(controller.settle(replacement, replacement, true)).toBe("silenced");
    expect(controller.claim(null)).toBeNull();
  });

  it("keeps microphone onset provisional until server floor authority commits it", () => {
    const first = {
      turnId: "turn_1",
      turnGeneration: 3,
      chunkId: "chunk_1",
      streamId: "stream_1",
    };
    const replacement = { ...first, chunkId: "chunk_2" };
    const controller = new ProvisionalSpeechYieldController<typeof first>(sameSpeechAuthority);

    expect(controller.begin(first)).toEqual({ claim: first, firstOnset: true });
    expect(controller.quiet(first)).toBe(true);
    expect(controller.replace(replacement)).toBe(first);
    expect(controller.pending).toBe(replacement);
    expect(controller.settle(replacement, replacement, true)).toBe("silenced");

    expect(controller.begin(first)).toEqual({ claim: first, firstOnset: true });
    expect(controller.settle(replacement, replacement, true)).toBe("ignored");
    expect(controller.pending).toBe(first);
    expect(controller.settle(first, replacement, true)).toBe("recover");

    expect(controller.begin(first)).toEqual({ claim: first, firstOnset: true });
    expect(controller.settle(first, first, false)).toBe("recover");
    expect(controller.begin(first)).toEqual({ claim: first, firstOnset: true });
    expect(controller.expire()).toBe(first);
    expect(controller.pending).toBeNull();
  });

  it("ducks only an actively playing unsilenced renderer", () => {
    expect(provisionalSpeechYieldAllowed(true, true, true, false, false, false)).toBe(true);
    expect(provisionalSpeechYieldAllowed(false, true, true, false, false, false)).toBe(false);
    expect(provisionalSpeechYieldAllowed(true, false, true, false, false, false)).toBe(false);
    expect(provisionalSpeechYieldAllowed(true, true, false, false, false, false)).toBe(false);
    expect(provisionalSpeechYieldAllowed(true, true, true, true, false, false)).toBe(false);
    expect(provisionalSpeechYieldAllowed(true, true, true, false, true, false)).toBe(false);
    expect(provisionalSpeechYieldAllowed(true, true, true, false, false, true)).toBe(false);
  });

  it("accepts only microphone audio from the exact server worker", () => {
    expect(authorizeRemoteAudio("worker_hermes_browser", "worker_hermes_browser", true, true)).toBe(
      true,
    );
    expect(authorizeRemoteAudio("worker_hermes_browser", "worker_intruder", true, true)).toBe(false);
    expect(authorizeRemoteAudio("worker_hermes_browser", "worker_hermes_browser", false, true)).toBe(
      false,
    );
    expect(authorizeRemoteAudio("worker_hermes_browser", "worker_hermes_browser", true, false)).toBe(
      false,
    );
  });

  it("detaches and clears remote playback before local disconnect invalidates the room", () => {
    const detached: object[] = [];
    const sink = {
      dataset: { streamId: "speech-stream" },
      srcObject: { active: true },
    };
    const track = {
      detach: (element: object) => detached.push(element),
    };

    expect(releaseRemoteAudio(track, sink)).toBeNull();

    expect(detached).toEqual([sink]);
    expect(sink.srcObject).toBeNull();
    expect(sink.dataset).not.toHaveProperty("streamId");
  });

  it("still clears remote playback when LiveKit detach throws", () => {
    const sink = {
      dataset: { streamId: "speech-stream" },
      srcObject: { active: true },
    };
    const track = {
      detach: () => {
        throw new Error("detach failed");
      },
    };

    expect(() => releaseRemoteAudio(track, sink)).not.toThrow();
    expect(sink.srcObject).toBeNull();
    expect(sink.dataset).not.toHaveProperty("streamId");
  });

  it("releases remote playback and karaoke before invalidating room authority", () => {
    const operations: string[] = [];
    let source: object | null = { active: true };
    const dataset = new Proxy(
      { streamId: "speech-stream" },
      {
        deleteProperty(target, property) {
          operations.push("stream-id-cleared");
          return Reflect.deleteProperty(target, property);
        },
      },
    );
    const sink = {
      dataset,
      get srcObject(): object | null {
        return source;
      },
      set srcObject(value: object | null) {
        source = value;
        operations.push("sink-cleared");
      },
    };
    const track = {
      detach: () => {
        operations.push("track-detached");
        throw new Error("detach failed");
      },
    };

    expect(
      releaseRemoteAudioBeforeRoomInvalidation(
        track,
        sink,
        () => operations.push("karaoke-stopped"),
        () => operations.push("room-invalidated"),
      ),
    ).toBeNull();

    expect(operations).toEqual([
      "track-detached",
      "sink-cleared",
      "stream-id-cleared",
      "karaoke-stopped",
      "room-invalidated",
    ]);
  });

  it("still invalidates room authority when karaoke cleanup throws", () => {
    const sink = {
      dataset: { streamId: "speech-stream" },
      srcObject: { active: true },
    };
    let invalidated = false;

    expect(() =>
      releaseRemoteAudioBeforeRoomInvalidation(
        null,
        sink,
        () => {
          throw new Error("karaoke cleanup failed");
        },
        () => {
          invalidated = true;
        },
      ),
    ).not.toThrow();

    expect(invalidated).toBe(true);
    expect(sink.srcObject).toBeNull();
    expect(sink.dataset).not.toHaveProperty("streamId");
  });

  it("rejects stale remote unsubscribe events by room and track identity", () => {
    const room = {};
    const track = {};

    expect(remoteAudioTrackIsCurrent(room, room, track, track)).toBe(true);
    expect(remoteAudioTrackIsCurrent(room, null, track, track)).toBe(false);
    expect(remoteAudioTrackIsCurrent(room, {}, track, track)).toBe(false);
    expect(remoteAudioTrackIsCurrent(room, room, track, null)).toBe(false);
    expect(remoteAudioTrackIsCurrent(room, room, track, {})).toBe(false);
  });

  it("installs an authorized replacement even when old-track detach throws", () => {
    const sink = {
      dataset: { streamId: "old-stream" },
      srcObject: { active: true },
    };
    const current = {
      detach: () => {
        throw new Error("detach failed");
      },
    };
    const replacement = {
      detach: () => undefined,
    };

    expect(replaceRemoteAudio(current, replacement, sink)).toBe(replacement);
    expect(sink.srcObject).toBeNull();
    expect(sink.dataset).not.toHaveProperty("streamId");
  });
});

describe("bounded browser retention", () => {
  it("evicts oldest entries by both count and aggregate character cost", () => {
    const retention = new BoundedRetention<object>(2, 8);
    const first = {};
    const second = {};
    const third = {};

    expect(retention.admit(first, 4)).toEqual([]);
    expect(retention.admit(second, 4)).toEqual([]);
    expect(retention.admit(third, 5)).toEqual([first, second]);
    expect(retention.size).toBe(1);
    expect(retention.cost).toBe(5);
  });

  it("recharges retained mutable DOM and evicts over-budget values", () => {
    const retention = new BoundedRetention<string>(3, 10);
    expect(retention.admit("first", 4)).toEqual([]);
    expect(retention.admit("second", 4)).toEqual([]);

    expect(retention.update("second", 9)).toEqual(["first"]);
    expect(retention.cost).toBe(9);
    expect(retention.update("second", 11)).toEqual(["second"]);
    expect(retention.cost).toBe(0);
  });
});

describe("media connection ownership", () => {
  it("disconnects an in-flight room that resolves after cancellation", () => {
    const disconnected: object[] = [];
    const attempt = new ConnectionAttempt<object>((room) => disconnected.push(room));
    const replacement = new ConnectionAttempt<object>((room) => disconnected.push(room));
    const pendingRoom = {};

    attempt.cancel();

    expect(attempt.owns(attempt)).toBe(false);
    expect(attempt.owns(replacement)).toBe(false);
    expect(replacement.owns(replacement)).toBe(true);
    expect(attempt.admit(pendingRoom, null)).toBe(false);
    expect(disconnected).toEqual([pendingRoom]);
  });
});

describe("approval decisions", () => {
  it("does not consume a sequence until authoritative submission succeeds", async () => {
    const attempts: Array<[number, string, string]> = [];
    let fail = true;
    const approvals = new ApprovalDecisionController(async (sequence, approvalId, decision) => {
      attempts.push([sequence, approvalId, decision]);
      if (fail) throw new Error("temporary failure");
    });

    await expect(
      approvals.submit("approval_0123456789abcdef", "approve"),
    ).rejects.toThrow("temporary failure");
    fail = false;
    await approvals.submit("approval_0123456789abcdef", "reject");

    expect(attempts).toEqual([
      [1, "approval_0123456789abcdef", "approve"],
      [1, "approval_0123456789abcdef", "reject"],
    ]);
    expect(approvals.lastSequence).toBe(1);
    approvals.reset();
    expect(approvals.lastSequence).toBe(0);
    await approvals.submit("approval_fedcba9876543210", "approve");
    expect(attempts.at(-1)).toEqual([1, "approval_fedcba9876543210", "approve"]);
  });
});

describe("objective latency tracking", () => {
  it("derives bounded deltas only from authoritative server markers", () => {
    const tracker = new ObjectiveLatencyTracker();

    expect(tracker.observe("speech_ended", 1000)).toBeNull();
    expect(tracker.observe("transcript_final", 1125, "user")).toEqual({
      name: "speech_end_to_transcript",
      durationMs: 125,
    });
    expect(tracker.observe("first_foreground_token", 1200)).toEqual({
      name: "transcript_to_first_token",
      durationMs: 75,
    });
    expect(tracker.observe("first_playable_audio", 1320)).toEqual({
      name: "first_token_to_audio",
      durationMs: 120,
    });
    expect(tracker.observe("interrupt_requested", 1400)).toBeNull();
    expect(tracker.observe("playback_silenced", 1440)).toEqual({
      name: "interrupt_to_silence",
      durationMs: 40,
    });
    expect(() => tracker.observe("speech_ended", 1300)).toThrow(/regressed/i);
  });

  it("tracks last and arithmetic-mean response start latency in constant space", () => {
    const statistics = new ResponseLatencyStatistics();

    expect(statistics.snapshot).toEqual({ lastMs: null, averageMs: null, sampleCount: 0 });
    statistics.observe({ name: "first_token_to_audio", durationMs: 50 });
    statistics.observe({ name: "transcript_to_first_token", durationMs: 100 });
    statistics.observe({ name: "transcript_to_first_token", durationMs: 300 });

    expect(statistics.snapshot).toEqual({ lastMs: 300, averageMs: 200, sampleCount: 2 });
    expect(formatLatencyDuration(null)).toBe("Waiting for first response");
    expect(formatLatencyDuration(87.4)).toBe("87 ms");
    expect(formatLatencyDuration(1234)).toBe("1.23 s");
  });
});

describe("partial transcript lifecycle", () => {
  it("retains interrupted assistant text but clears transient user/session text", () => {
    expect(partialTranscriptDisposition("interrupt_requested", "assistant")).toBe(
      "retain-interrupted",
    );
    expect(partialTranscriptDisposition("interrupt_requested", "user")).toBe("clear");
    expect(partialTranscriptDisposition("speech_ended", "user")).toBe("clear");
    expect(partialTranscriptDisposition("speech_ended", "assistant")).toBe("keep");
    expect(partialTranscriptDisposition("session_stopped", "assistant")).toBe("clear");
  });
});

describe("session model presentation", () => {
  it("labels configured identity and unavailable telemetry without guessing", () => {
    expect(
      formatSessionModelConfiguration({
        authentication: "subscription",
        contextWindowTokens: null,
        effort: "medium",
        model: "gpt-5.6-terra",
        provider: "openai-codex",
        reportsTokenUsage: false,
        transport: "subscription-app-server",
      }),
    ).toEqual([
      ["Provider", "openai-codex"],
      ["Model", "gpt-5.6-terra"],
      ["Access", "Subscription"],
      ["Reasoning effort", "medium"],
      ["Transport", "subscription-app-server"],
      ["Context window", "Not reported"],
      ["Token usage", "Not reported by transport"],
    ]);
  });

  it("formats cumulative provider counters", () => {
    expect(
      formatSessionTokenUsage({
        cachedInputTokens: 1200,
        contextWindowTokens: 272000,
        inputTokens: 4000,
        outputTokens: 900,
        reasoningOutputTokens: 300,
        totalTokens: 5200,
      }),
    ).toEqual([
      ["Session total", "5,200"],
      ["Input", "4,000"],
      ["Cached input", "1,200"],
      ["Output", "900"],
      ["Reasoning output", "300"],
    ]);
  });
});

describe("stable user turn projection", () => {
  it("projects typed input only from its authoritative transcript event", () => {
    expect(userTranscriptProjectionDisposition("typed-admission")).toBe(
      "await-authoritative",
    );
    expect(userTranscriptProjectionDisposition("authoritative-event")).toBe("project");
  });

  it("keeps the turn open when an older assistant generation is rejected", () => {
    const turn = new UserTurnAssembler();
    turn.admit("First segment.");

    expect(assistantGenerationDisposition(undefined, 1)).toBe("create");
    expect(assistantGenerationDisposition(7, 7)).toBe("current");
    expect(assistantGenerationDisposition(7, 8)).toBe("replace");
    expect(assistantGenerationDisposition(7, 6)).toBe("reject");
    expect(turn.admit("Second segment.").text).toBe("First segment. Second segment.");
  });

  it("assembles consecutive finals until assistant content starts", () => {
    const turn = new UserTurnAssembler();

    expect(turn.admit("So. I.").text).toBe("So. I.");
    expect(turn.admit("Right. Right? You.").text).toBe("So. I. Right. Right? You.");
    expect(turn.admit("I would like you to. Ask me a question.").text).toBe(
      "So. I. Right. Right? You. I would like you to. Ask me a question.",
    );
    expect(turn.close().state).toBe("closed");
    expect(() => turn.admit("A later turn.")).toThrow(/closed/i);
  });
});

describe("stable assistant turn projection", () => {
  it("retains generation authority after the corresponding DOM view is evicted", () => {
    const authority = new AssistantGenerationAuthority(32);

    expect(authority.disposition("turn_001", 7)).toBe("create");
    expect(authority.disposition("turn_001", 6)).toBe("reject");
    expect(authority.disposition("turn_001", 7)).toBe("current");
    expect(authority.disposition("turn_001", 8)).toBe("replace");
  });

  it("assembles delivered segments into one generation-scoped bubble", () => {
    const turn = new AssistantTurnAssembler("turn_001", 7);

    expect(turn.admit("chunk_1", "First answer.").text).toBe("First answer.");
    expect(turn.confirm("chunk_1").deliveredChunkIds).toEqual(["chunk_1"]);
    expect(turn.admit("chunk_2", "Second answer.").text).toBe(
      "First answer. Second answer.",
    );
    expect(turn.complete().state).toBe("complete");
    expect(() => turn.admit("chunk_2", "different")).toThrow(/conflicting/i);
  });

  it("discards an unseen late chunk after completion without throwing", () => {
    const turn = new AssistantTurnAssembler("turn_001", 7);
    turn.admit("chunk_1", "Delivered.");
    turn.complete();

    expect(turn.tryAdmit("chunk_2", "Late.")).toBeNull();
    expect(turn.snapshot.text).toBe("Delivered.");
  });

  it("retains generated content and its delivered boundary when interrupted", () => {
    const turn = new AssistantTurnAssembler("turn_001", 7);
    turn.admit("chunk_1", "Already delivered.");
    turn.confirm("chunk_1");
    turn.admit("chunk_2", "Not heard.");

    const interrupted = turn.interrupt();
    expect(interrupted.state).toBe("interrupted");
    expect(interrupted.text).toBe("Already delivered. Not heard.");
    expect(interrupted.deliveredChunkIds).toEqual(["chunk_1"]);
    expect(interrupted.chunkIds).toEqual(["chunk_1", "chunk_2"]);
  });
});

describe("speech timing event ordering", () => {
  it("starts every attached passage when the second track emits no playing event", () => {
    const starts = new SpeechMediaStartRegistry(4);

    starts.attach("speech_1", 0);
    expect(starts.observeStream("speech_1", 0)).toBeUndefined();
    expect(starts.observeStream("speech_1", 0.02)).toBe(0.02);

    starts.attach("speech_2", 1.25);
    expect(starts.observeStream("speech_2", 1.25)).toBeUndefined();
    expect(starts.observeStream("speech_2", 1.5)).toBe(1.5);
    expect(starts.resolvePlaying("speech_2", 1.52)).toBe(1.5);

    starts.attach("speech_3", 0);
    expect(starts.resolvePlaying("speech_3", 0.01)).toBe(0.01);
  });

  it("rebases each passage on one persistent media clock", () => {
    const starts = new SpeechMediaStartRegistry(4);

    starts.beginPassage("chunk_1", 4.0);
    expect(starts.observePassage("chunk_1", 4.25)).toBe(4.25);
    starts.beginPassage("chunk_1", 8.0);

    expect(starts.getPassage("chunk_1")).toBeUndefined();
    expect(starts.observePassage("chunk_1", 8.0)).toBeUndefined();
    expect(starts.observePassage("chunk_1", 8.1)).toBe(8.1);
  });

  it("retains third-chunk timing until its delayed transcript segment arrives", () => {
    const timing = parseSpeechTiming({
      turnId: "resume_1",
      presentationTurnId: "turn_001",
      turnGeneration: 7,
      chunkId: "speech_chunk_3",
      segmentId: "chunk_3",
      streamId: "speech_3",
      sampleRate: 48000,
      timingSource: "estimated",
      timings: "0,5,0,12000;6,11,12000,24000",
    });
    if (timing === null) throw new Error("timing unexpectedly absent");
    const registry = new SpeechTimingRegistry(32);

    registry.admit(timing);

    expect(registry.get("speech_3")).toBe(timing);
    expect(registry.forSegment("turn_001", 7, "chunk_3")).toEqual([timing]);
    expect(registry.forSegment("turn_001", 7, "chunk_2")).toEqual([]);
  });

  it("retains every chunk timing when a persistent stream identity is reused", () => {
    const first = parseSpeechTiming({
      turnId: "turn_001",
      presentationTurnId: "turn_001",
      turnGeneration: 7,
      chunkId: "speech_chunk_1",
      segmentId: "chunk_1",
      streamId: "persistent_speech",
      sampleRate: 48000,
      timingSource: "estimated",
      timings: "0,5,0,12000",
    });
    const second = parseSpeechTiming({
      turnId: "turn_001",
      presentationTurnId: "turn_001",
      turnGeneration: 7,
      chunkId: "speech_chunk_2",
      segmentId: "chunk_2",
      streamId: "persistent_speech",
      sampleRate: 48000,
      timingSource: "estimated",
      timings: "0,6,0,12000",
    });
    if (first === null || second === null) throw new Error("timing unexpectedly absent");
    const registry = new SpeechTimingRegistry(32);

    registry.admit(first);
    registry.admit(second);

    expect(registry.get("persistent_speech")).toBe(second);
    expect(registry.forSegment("turn_001", 7, "chunk_1")).toEqual([first]);
    expect(registry.forSegment("turn_001", 7, "chunk_2")).toEqual([second]);
  });
});

describe("speech timing", () => {
  it("parses monotonic exact offset timing and follows the media clock", () => {
    const timing = parseSpeechTiming({
      turnId: "turn_001",
      presentationTurnId: "turn_001",
      turnGeneration: 7,
      chunkId: "speech_chunk_1",
      segmentId: "segment_1",
      streamId: "speech_1",
      sampleRate: 48000,
      timingSource: "estimated",
      timings: "0,5,0,12000;6,11,12000,24000",
    });
    expect(timing).not.toBeNull();
    if (timing === null) throw new Error("timing unexpectedly absent");
    const clock = new SpeechKaraokeClock(timing, 12.5);

    expect(clock.wordIndexAt(12.5)).toBe(0);
    expect(clock.wordIndexAt(12.749)).toBe(0);
    expect(clock.wordIndexAt(12.75)).toBe(1);
    expect(clock.wordIndexAt(13.0)).toBe(-1);
  });

  it("rebases a new passage when media currentTime resets after playing fires", () => {
    const timing = parseSpeechTiming({
      turnId: "turn_001",
      presentationTurnId: "turn_001",
      turnGeneration: 7,
      chunkId: "speech_chunk_2",
      segmentId: "segment_2",
      streamId: "speech_2",
      sampleRate: 48000,
      timingSource: "estimated",
      timings: "0,2,0,12000;3,12,12000,24000;13,20,24000,36000",
    });
    expect(timing).not.toBeNull();
    if (timing === null) throw new Error("timing unexpectedly absent");

    // Chromium can expose passage one's terminal media time when the replacement
    // stream's `playing` event fires, then reset currentTime to the new stream origin.
    const clock = new SpeechKaraokeClock(timing, 8);

    expect(clock.wordIndexAt(0)).toBe(0);
    expect(clock.wordIndexAt(0.3)).toBe(1);
    expect(clock.wordIndexAt(0.6)).toBe(2);
  });

  it("represents provider leading silence and inter-word gaps without premature highlighting", () => {
    const timing = parseSpeechTiming({
      turnId: "turn_001",
      presentationTurnId: "turn_001",
      turnGeneration: 7,
      chunkId: "speech_chunk_1",
      segmentId: "segment_1",
      streamId: "speech_1",
      sampleRate: 48000,
      timingSource: "provider",
      timings: "0,5,2400,12000;6,11,16000,24000",
    });
    expect(timing).not.toBeNull();
    const clock = new SpeechKaraokeClock(timing!, 10);
    expect(clock.wordIndexAt(10)).toBe(-2);
    expect(clock.wordIndexAt(10.06)).toBe(0);
    expect(clock.wordIndexAt(10.3)).toBe(-2);
    expect(clock.wordIndexAt(10.34)).toBe(1);
    expect(clock.wordIndexAt(10.5)).toBe(-1);
  });

  it("rejects malformed, regressing, and non-canonical timing", () => {
    const base = {
      turnId: "turn_001",
      presentationTurnId: "turn_001",
      turnGeneration: 7,
      chunkId: "speech_chunk_1",
      segmentId: "segment_1",
      streamId: "speech_1",
      sampleRate: 48000,
      timingSource: "provider",
    };
    expect(parseSpeechTiming({ ...base, timings: "0,5,100,200;6,11,50,300" })).toBeNull();
    expect(parseSpeechTiming({ ...base, timings: "0,5,0,12000;garbage" })).toBeNull();
    expect(parseSpeechTiming({ ...base, timings: "" })).toBeNull();
  });
});
