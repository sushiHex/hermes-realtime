import { describe, expect, it } from "vitest";

import {
  parseBootstrapCredential,
  parseCaptureStatus,
  parseEventBatch,
  parseModelCatalog,
  parseSearchEgressStatus,
  parseSessionModelConfiguration,
  parseSessionTokenUsage,
  parseSpeechAuthority,
  parseSpeechRuntime,
  parseVoiceConfiguration,
} from "../src/protocol";

describe("voice configuration protocol", () => {
  it("accepts an exact bounded provider catalog", () => {
    expect(
      parseVoiceConfiguration({
        selectedVoice: "bf_isabella",
        version: 1,
        voices: ["bf_emma", "bf_isabella"],
      }),
    ).toEqual({
      selectedVoice: "bf_isabella",
      version: 1,
      voices: ["bf_emma", "bf_isabella"],
    });
    expect(() =>
      parseVoiceConfiguration({ selectedVoice: "unknown", version: 1, voices: [] }),
    ).toThrow(/fields/i);
  });
});

describe("evidence capture status protocol", () => {
  it("accepts only the exact public capture status shape", () => {
    const status = {
      available: true,
      captureState: "idle",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: "a".repeat(64),
      retentionHours: 48,
    } as const;
    expect(parseCaptureStatus(status)).toEqual(status);
    expect(() => parseCaptureStatus({ ...status, privateToken: "secret" })).toThrow(/shape/i);
    expect(() => parseCaptureStatus({ ...status, captureState: "unavailable" })).toThrow(
      /fields/i,
    );
  });
});

describe("speech runtime protocol", () => {
  it("accepts only exact bounded STT and TTS runtime identities", () => {
    const runtime = {
      sttModel: "moonshine-v2-small",
      sttProvider: "moonshine",
      ttsModel: "kokoro-v1.0.onnx",
      ttsProvider: "kokoro",
    };
    expect(parseSpeechRuntime(runtime)).toEqual(runtime);
    expect(() => parseSpeechRuntime({ ...runtime, extra: "guess" })).toThrow(/shape/i);
    expect(() => parseSpeechRuntime({ ...runtime, sttModel: "" })).toThrow(/fields/i);
    expect(parseSpeechRuntime({ ...runtime, sttModel: "m".repeat(256) }).sttModel).toHaveLength(256);
    expect(() => parseSpeechRuntime({ ...runtime, sttProvider: "p".repeat(65) })).toThrow(
      /fields/i,
    );
    expect(() => parseSpeechRuntime({ ...runtime, ttsModel: "m".repeat(257) })).toThrow(/fields/i);
  });
});

describe("renderer yield authority protocol", () => {
  it("accepts only the exact public speech authority tuple", () => {
    const authority = {
      chunkId: "chunk_2",
      streamId: "stream_4",
      turnGeneration: 3,
      turnId: "turn_1",
    };
    expect(parseSpeechAuthority(authority)).toEqual(authority);
    expect(() => parseSpeechAuthority({ ...authority, streamId: "" })).toThrow(/authority/i);
    expect(() => parseSpeechAuthority({ ...authority, extra: "stale" })).toThrow(/authority/i);
  });
});

describe("session model protocol", () => {
  it("accepts an exact model catalog and rejects unsupported selected effort", () => {
    const catalog = {
      models: [
        {
          defaultEffort: "medium",
          description: "Fast coding model",
          displayName: "GPT-5.6 Terra",
          model: "gpt-5.6-terra",
          supportedEfforts: ["low", "medium"],
        },
        {
          defaultEffort: "high",
          description: "Deep coding model",
          displayName: "GPT-5.6 Sol",
          model: "gpt-5.6-sol",
          supportedEfforts: ["medium", "high", "xhigh", "ultra"],
        },
      ],
      selectedEffort: "low",
      selectedModel: "gpt-5.6-terra",
      version: 1,
    };

    expect(parseModelCatalog(catalog)).toEqual(catalog);
    expect(() => parseModelCatalog({ ...catalog, selectedEffort: "xhigh" })).toThrow(/selection/i);
    expect(() => parseModelCatalog({ ...catalog, privateEndpoint: "hidden" })).toThrow(/shape/i);
  });

  it("accepts exact authoritative model metadata and rejects extra fields", () => {
    const configuration = {
      authentication: "subscription",
      contextWindowTokens: null,
      effort: "medium",
      model: "gpt-5.6-terra",
      provider: "openai-codex",
      reportsTokenUsage: true,
      transport: "subscription-app-server",
    };
    expect(parseSessionModelConfiguration(configuration)).toEqual(configuration);
    expect(() =>
      parseSessionModelConfiguration({ ...configuration, endpoint: "private" }),
    ).toThrow(/shape/i);
  });

  it("accepts exact cumulative usage and rejects unsafe counters", () => {
    const usage = {
      cachedInputTokens: 50,
      contextWindowTokens: 272000,
      inputTokens: 400,
      outputTokens: 90,
      reasoningOutputTokens: 30,
      totalTokens: 520,
    };
    expect(parseSessionTokenUsage(usage)).toEqual(usage);
    expect(() => parseSessionTokenUsage({ ...usage, totalTokens: -1 })).toThrow(/fields/i);
    expect(() => parseSessionTokenUsage({ ...usage, secret: "x" })).toThrow(/shape/i);
  });
});

describe("bootstrap protocol", () => {
  it("accepts only the exact server-selected credential shape", () => {
    expect(
      parseBootstrapCredential({
        version: 1,
        url: "wss://livekit.test",
        roomName: "hermes-local",
        participantIdentity: "browser_0123456789abcdef",
        workerIdentity: "worker_hermes_browser",
        expiresInSeconds: 60,
        token: "a.b.c",
      }),
    ).toEqual({
      version: 1,
      url: "wss://livekit.test",
      roomName: "hermes-local",
      participantIdentity: "browser_0123456789abcdef",
      workerIdentity: "worker_hermes_browser",
      expiresInSeconds: 60,
      token: "a.b.c",
    });

    expect(() =>
      parseBootstrapCredential({
        version: 1,
        url: "wss://livekit.test",
        roomName: "hermes-local",
        participantIdentity: "browser_0123456789abcdef",
        workerIdentity: "browser_not-a-worker",
        expiresInSeconds: 60,
        token: "a.b.c",
      }),
    ).toThrow(/fields/i);

    expect(() =>
      parseBootstrapCredential({
        version: 1,
        url: "wss://livekit.test",
        roomName: "other-room",
        participantIdentity: "browser_0123456789abcdef",
        workerIdentity: "worker_hermes_browser",
        expiresInSeconds: 60,
        token: "a.b.c",
        apiSecret: "must-not-be-accepted",
      }),
    ).toThrow(/shape/i);
  });
});

describe("public event protocol", () => {
  it("accepts an authoritative capture status event", () => {
    const data = {
      available: true,
      captureState: "active",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: "a".repeat(64),
      retentionHours: 24,
    };
    expect(
      parseEventBatch({
        version: 1,
        events: [{ sequence: 1, kind: "capture_status", monotonicMs: 1, data }],
      }).events[0]?.data,
    ).toEqual(data);
  });

  it("requires authoritative speech identities on session readiness", () => {
    const data = {
      conversationProfile: "natural_v1",
      mode: "microphone_or_typed",
      sttModel: "moonshine-v2-small",
      sttProvider: "moonshine",
      ttsModel: "kokoro-v1.0.onnx",
      ttsProvider: "kokoro",
    };
    expect(
      parseEventBatch({
        version: 1,
        events: [{ sequence: 1, kind: "session_ready", monotonicMs: 1, data }],
      }).events[0]?.data,
    ).toEqual(data);
    expect(() =>
      parseEventBatch({
        version: 1,
        events: [
          { sequence: 1, kind: "session_ready", monotonicMs: 1, data: { ...data, sttModel: "" } },
        ],
      }),
    ).toThrow(/speech runtime/i);
  });

  it("accepts sequenced objective events and rejects unknown fields", () => {
    const batch = parseEventBatch({
      version: 1,
      events: [
        {
          sequence: 1,
          kind: "typed_input_admitted",
          monotonicMs: 12500,
          data: { inputSequence: 1 },
        },
      ],
    });
    expect(batch.events[0]?.kind).toBe("typed_input_admitted");
    expect(() =>
      parseEventBatch({ version: 1, events: [], privateHandle: "deleg_hidden" }),
    ).toThrow(/shape/);
  });

  it("accepts server-confirmed voice input readiness", () => {
    const batch = parseEventBatch({
      version: 1,
      events: [
        {
          sequence: 1,
          kind: "voice_input_ready",
          monotonicMs: 12500,
          data: { generation: 1, mediaIncarnation: 4 },
        },
      ],
    });

    expect(batch.events[0]?.kind).toBe("voice_input_ready");
    expect(() =>
      parseEventBatch({
        version: 1,
        events: [
          {
            sequence: 1,
            kind: "voice_input_ready",
            monotonicMs: 12500,
            data: { generation: 0, mediaIncarnation: 4 },
          },
        ],
      }),
    ).toThrow(/authority/i);
    expect(() =>
      parseEventBatch({
        version: 1,
        events: [
          { sequence: 1, kind: "voice_input_ready", monotonicMs: 12500, data: { generation: 1 } },
        ],
      }),
    ).toThrow(/authority/i);
  });

  it.each([
    "barge_in_non_speech_suppressed",
    "barge_in_verifier_unavailable",
    "echo_suppressed",
    "echo_barge_in_confirmed",
  ])(
    "accepts the bounded %s barge-in diagnostic event kind",
    (kind) => {
      expect(
        parseEventBatch({
          version: 1,
          events: [{ sequence: 1, kind, monotonicMs: 12500, data: {} }],
        }).events[0]?.kind,
      ).toBe(kind);
    },
  );

  it("accepts server voice activity boundaries", () => {
    const batch = parseEventBatch({
      version: 1,
      events: [
        { sequence: 1, kind: "voice_activity_started", monotonicMs: 12500, data: {} },
        { sequence: 2, kind: "voice_activity_ended", monotonicMs: 13000, data: {} },
      ],
    });

    expect(batch.events.map((event) => event.kind)).toEqual([
      "voice_activity_started",
      "voice_activity_ended",
    ]);
  });

  it("accepts bounded turn-scoped speech timing and completion events", () => {
    const batch = parseEventBatch({
      version: 1,
      events: [
        {
          sequence: 1,
          kind: "speech_timing",
          monotonicMs: 12700,
          data: {
            turnId: "turn_001",
            turnGeneration: 7,
            chunkId: "kokoro_abc123",
            streamId: "speech_def456",
            sampleRate: 48000,
            timingSource: "estimated",
            timings: "0,5,0,12000;6,11,12000,24000",
          },
        },
        {
          sequence: 2,
          kind: "assistant_turn_completed",
          monotonicMs: 13000,
          data: { turnId: "turn_001", turnGeneration: 7 },
        },
        {
          sequence: 3,
          kind: "assistant_turn_interrupted",
          monotonicMs: 13100,
          data: { turnId: "turn_002", turnGeneration: 8 },
        },
      ],
    });

    expect(batch.events.map((event) => event.kind)).toEqual([
      "speech_timing",
      "assistant_turn_completed",
      "assistant_turn_interrupted",
    ]);
  });

  it("accepts exact content-free knowledge timing and rejects query fields", () => {
    const data = {
      turnId: "session_7_media_3_utterance_11",
      route: "current_fact",
      backend: "ddgs",
      outcome: "usable",
      sampleCount: 9,
      lastMs: 1200,
      p50Ms: 900,
      p95Ms: 1800,
      lookupElapsedMs: 1500,
      lookupBlockingMs: 500,
      lookupOverlapMs: 1000,
      recoveryUsed: false,
    };
    expect(
      parseEventBatch({
        version: 1,
        events: [{ sequence: 1, kind: "knowledge_timing", monotonicMs: 12700, data }],
      }).events[0]?.data,
    ).toEqual(data);
    const healthData = {
      ...data,
      lookupClosed: false,
      lookupDetachedCalls: 2,
      lookupDetachedCallsTotal: 5,
      lookupSaturationEvents: 1,
    };
    expect(
      parseEventBatch({
        version: 1,
        events: [
          { sequence: 1, kind: "knowledge_timing", monotonicMs: 12700, data: healthData },
        ],
      }).events[0]?.data,
    ).toEqual(healthData);
    expect(() =>
      parseEventBatch({
        version: 1,
        events: [
          {
            sequence: 1,
            kind: "knowledge_timing",
            monotonicMs: 12700,
            data: { ...data, lookupDetachedCalls: 2 },
          },
        ],
      }),
    ).toThrow(/knowledge timing/i);
    expect(() =>
      parseEventBatch({
        version: 1,
        events: [
          {
            sequence: 1,
            kind: "knowledge_timing",
            monotonicMs: 12700,
            data: { ...data, query: "private" },
          },
        ],
      }),
    ).toThrow(/knowledge timing/i);
  });

  it("accepts a bounded partial transcript event", () => {
    const batch = parseEventBatch({
      version: 1,
      events: [
        {
          sequence: 1,
          kind: "transcript_partial",
          monotonicMs: 12500,
          data: { role: "user", text: "Can you give" },
        },
        {
          sequence: 2,
          kind: "transcript_partial",
          monotonicMs: 12600,
          data: { role: "assistant", text: "I can explain" },
        },
      ],
    });

    expect(batch.events[0]?.kind).toBe("transcript_partial");
    expect(batch.events[0]?.data.text).toBe("Can you give");
    expect(batch.events[1]?.data.role).toBe("assistant");
    expect(batch.events[1]?.data.text).toBe("I can explain");
  });

  it("accepts a visual-only background task result", () => {
    const batch = parseEventBatch({
      version: 1,
      events: [
        {
          sequence: 1,
          kind: "task_result",
          monotonicMs: 14000,
          data: {
            status: "completed",
            taskId: "task_release_check",
            text: "Full details: https://example.com/releases/19.1",
          },
        },
      ],
    });

    expect(batch.events[0]?.kind).toBe("task_result");
    expect(batch.events[0]?.data.text).toContain("https://example.com");
  });

  it("rejects malformed visual-only background task results", () => {
    expect(() =>
      parseEventBatch({
        version: 1,
        events: [
          {
            sequence: 1,
            kind: "task_result",
            monotonicMs: 14000,
            data: { text: "missing authority" },
          },
        ],
      }),
    ).toThrow("Public task result is invalid");
  });
});


describe("search egress status parsing", () => {
  it("accepts only the exact content-free status schema and event kind", () => {
    const status = {
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: "b".repeat(64),
      searchEgressState: "idle",
    } as const;
    expect(parseSearchEgressStatus(status)).toEqual(status);
    expect(() => parseSearchEgressStatus({ ...status, query: "forbidden" })).toThrow(/shape/i);
    expect(() =>
      parseSearchEgressStatus({ ...status, searchEgressState: "unavailable" }),
    ).toThrow(/fields/i);
    expect(
      parseEventBatch({
        version: 1,
        events: [
          {
            sequence: 1,
            kind: "search_egress_status",
            monotonicMs: 1,
            data: status,
          },
        ],
      }).events[0]?.kind,
    ).toBe("search_egress_status");
  });
});
