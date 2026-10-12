import { describe, expect, it, vi } from "vitest";
import {
  fetchLocalModelRequest,
  LOCAL_MODEL_METADATA_TIMEOUT_MS,
  LOCAL_MODEL_PREPARATION_TIMEOUT_MS,
  localModelPreparationFor,
  parseLocalModelCatalog,
  parseLocalModelPreparation,
  type LocalModelCatalog,
} from "../src/local-model-selection";

const digest = "a".repeat(64);
const catalog = {
  version: 1,
  available: true,
  provider: "ollama",
  state: "configuring",
  models: [
    { model: "installed:latest", digest, advertisedContextTokens: 32768 },
    { model: "installed:unknown", digest: "b".repeat(64), advertisedContextTokens: null },
  ],
  defaultModel: "installed:latest",
  defaultRequestedContextTokens: 16384,
  minimumContextTokens: 2048,
  maximumContextTokens: 262144,
  preparationTimeoutMs: 120000,
  selection: null,
};

function enabledCatalog(value: unknown): LocalModelCatalog {
  const parsed = parseLocalModelCatalog(value);
  if (!parsed.available) throw new Error("Expected enabled local model catalog");
  return parsed;
}

describe("pre-connect local model contract", () => {
  it("bounds readiness separately from the ordinary short bootstrap timeout", async () => {
    vi.useFakeTimers();
    try {
      const observedSignals: AbortSignal[] = [];
      const pending = fetchLocalModelRequest(
        "/api/v1/local-model-prepare",
        { method: "POST" },
        LOCAL_MODEL_PREPARATION_TIMEOUT_MS,
        null,
        (_input, init) => new Promise<Response>((resolve) => {
          const signal = init?.signal as AbortSignal;
          observedSignals.push(signal);
          signal.addEventListener("abort", () => resolve(new Response(null, { status: 408 })));
        }),
      );
      await vi.advanceTimersByTimeAsync(LOCAL_MODEL_PREPARATION_TIMEOUT_MS);
      expect((await pending).status).toBe(408);
      expect(observedSignals[0]?.aborted).toBe(true);
      expect(LOCAL_MODEL_PREPARATION_TIMEOUT_MS).toBeGreaterThan(10_000);
    } finally {
      vi.useRealTimers();
    }
  });

  it("aborts a stalled response body within the preparation deadline", async () => {
    vi.useFakeTimers();
    try {
      const signals: AbortSignal[] = [];
      const pending = fetchLocalModelRequest(
        "/api/v1/local-model-prepare",
        { method: "POST" },
        LOCAL_MODEL_PREPARATION_TIMEOUT_MS,
        null,
        (_input, init) => {
          const signal = init?.signal as AbortSignal;
          signals.push(signal);
          return Promise.resolve(new Response(new ReadableStream<Uint8Array>({
            start(controller) {
              signal?.addEventListener("abort", () => controller.error(new DOMException("Aborted", "AbortError")));
            },
          })));
        },
      );
      const outcome = expect(pending).rejects.toThrow();
      await Promise.resolve();
      await Promise.resolve();
      await vi.advanceTimersByTimeAsync(LOCAL_MODEL_PREPARATION_TIMEOUT_MS);
      await outcome;
      expect(signals[0]?.aborted).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("rejects oversized response bodies and timeout values beyond the hard ceiling", async () => {
    await expect(fetchLocalModelRequest(
      "/api/v1/local-models",
      { method: "POST" },
      LOCAL_MODEL_METADATA_TIMEOUT_MS,
      null,
      async () => new Response(new Uint8Array(262_145)),
    )).rejects.toThrow(TypeError);
    await expect(fetchLocalModelRequest(
      "/api/v1/local-model-prepare",
      { method: "POST" },
      LOCAL_MODEL_PREPARATION_TIMEOUT_MS + 1,
      null,
      async () => new Response(null),
    )).rejects.toThrow(RangeError);
  });

  it("accepts an exact host catalog without inventing model presets", () => {
    expect(parseLocalModelCatalog(catalog)).toEqual(catalog);
  });

  it("accepts only the exact explicit feature-disabled response", () => {
    expect(parseLocalModelCatalog({ version: 1, available: false })).toEqual({ version: 1, available: false });
    const refusal = vi.spyOn(console, "info").mockImplementation(() => {});
    try {
      expect(() => parseLocalModelCatalog({ version: 1, available: false, error: "private detail" })).toThrow(TypeError);
      expect(refusal.mock.calls).toEqual([
        ['[local-model-selection-refusal] {"category":"invalid_catalog"}'],
      ]);
    } finally {
      refusal.mockRestore();
    }
  });

  it("accepts a recovered frozen selection bound to one installed model digest", () => {
    const frozen = {
      model: "installed:latest",
      digest,
      requestedContextTokens: 8192,
      advertisedContextTokens: 32768,
      effectiveContextTokens: 65536,
    };
    const parsed = parseLocalModelCatalog({ ...catalog, state: "prepared", selection: frozen });
    if (!parsed.available) throw new Error("Expected enabled catalog");
    expect(parsed.selection).toEqual(frozen);
  });

  it("rejects non-string lifecycle state", () => {
    const refusal = vi.spyOn(console, "info").mockImplementation(() => {});
    try {
      expect(() => parseLocalModelCatalog({ ...catalog, state: ["configuring"] })).toThrow(TypeError);
      expect(refusal.mock.calls).toEqual([
        ['[local-model-selection-refusal] {"category":"invalid_catalog"}'],
      ]);
    } finally {
      refusal.mockRestore();
    }
  });

  it("rejects control characters in model identities", () => {
    const refusal = vi.spyOn(console, "info").mockImplementation(() => {});
    try {
      expect(() => parseLocalModelCatalog({ ...catalog, models: [{ ...catalog.models[0], model: "installed:\nlatest" }] })).toThrow(TypeError);
      expect(refusal.mock.calls).toEqual([
        ['[local-model-selection-refusal] {"category":"invalid_catalog"}'],
      ]);
    } finally {
      refusal.mockRestore();
    }
  });

  it("allows distinct installed aliases to share a digest", () => {
    expect(enabledCatalog({ ...catalog, models: [
      catalog.models[0], { ...catalog.models[0], model: "installed:alias" },
    ] }).models).toHaveLength(2);
  });

  it("rejects unknown fields, malformed digests, stale bindings, and invalid capacities with one refusal", () => {
    const refusal = vi.spyOn(console, "info").mockImplementation(() => {});
    try {
      const rejected: unknown[] = [
        { ...catalog, privateEndpoint: "synthetic-private-detail" },
        { ...catalog, state: ["configuring"] },
        { ...catalog, models: [{ ...catalog.models[0], digest: "A".repeat(64) }] },
        { ...catalog, models: [catalog.models[0], catalog.models[0]] },
        { ...catalog, minimumContextTokens: 0 },
        { ...catalog, defaultModel: "missing:model" },
        { ...catalog, state: "prepared", selection: {
          model: "installed:latest", digest: "c".repeat(64), requestedContextTokens: 8192,
          advertisedContextTokens: 32768, effectiveContextTokens: 8192,
        } },
        { ...catalog, state: "prepared", selection: {
          model: "installed:latest", digest, requestedContextTokens: 32769,
          advertisedContextTokens: 32768, effectiveContextTokens: 8192,
        } },
      ];
      for (const value of rejected) {
        refusal.mockClear();
        expect(() => parseLocalModelCatalog(value)).toThrow(TypeError);
        expect(refusal.mock.calls).toEqual([
          ['[local-model-selection-refusal] {"category":"invalid_catalog"}'],
        ]);
      }
    } finally {
      refusal.mockRestore();
    }
  });

  it("requires a fresh explicit valid budget when the default exceeds model capacity or capacity is unknown", () => {
    const limited = { ...catalog, models: [{ ...catalog.models[0], advertisedContextTokens: 8192 }] };
    expect(() => localModelPreparationFor(enabledCatalog(limited), "installed:latest", 16384)).toThrow(TypeError);
    const unknown = { ...catalog, models: [catalog.models[1]], defaultModel: "installed:unknown" };
    const parsedUnknown = enabledCatalog(unknown);
    expect(() => localModelPreparationFor(parsedUnknown, "installed:unknown", null)).toThrow(TypeError);
    expect(localModelPreparationFor(parsedUnknown, "installed:unknown", 4096)).toEqual({
      version: 1,
      model: "installed:unknown",
      digest: "b".repeat(64),
      requestedContextTokens: 4096,
    });
  });

  it("binds each prepared request to the exact current catalog digest and requested capacity", () => {
    const parsed = enabledCatalog(catalog);
    expect(localModelPreparationFor(parsed, "installed:latest", 8192)).toEqual({
      version: 1,
      model: "installed:latest",
      digest,
      requestedContextTokens: 8192,
    });
    expect(() => localModelPreparationFor(parsed, "not-installed", 8192)).toThrow(TypeError);
    expect(() => localModelPreparationFor(parsed, "installed:latest", 2047)).toThrow(TypeError);
    expect(() => localModelPreparationFor(parsed, "installed:latest", 262145)).toThrow(TypeError);
  });

  it("accepts only the exact prepared selection bound to the request and catalog", () => {
    const parsed = enabledCatalog(catalog);
    const request = localModelPreparationFor(parsed, "installed:latest", 8192);
    const response = {
      version: 1,
      selection: {
        model: "installed:latest", digest, requestedContextTokens: 8192,
        advertisedContextTokens: 32768, effectiveContextTokens: 8192,
      },
    };
    expect(parseLocalModelPreparation(response, request, parsed)).toEqual(response.selection);
    expect(() => parseLocalModelPreparation({ ...response, extra: "stale" }, request, parsed)).toThrow(TypeError);
    expect(() => parseLocalModelPreparation({ ...response, selection: { ...response.selection, digest: "c".repeat(64) } }, request, parsed)).toThrow(TypeError);
    const forgedRequest = { ...request, digest: "c".repeat(64) };
    expect(() => parseLocalModelPreparation({ ...response, selection: { ...response.selection, digest: forgedRequest.digest } }, forgedRequest, parsed)).toThrow(TypeError);
  });
});
