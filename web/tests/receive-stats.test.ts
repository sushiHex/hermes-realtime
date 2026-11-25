import { describe, expect, it } from "vitest";

import {
  inboundAudioCounters,
  receiveStatsDelta,
  ReceiveStatsSampler,
  type ReceiveStatsContext,
  type ReceiveStatsSample,
  type StatsReportLike,
} from "../src/receive-stats";

function report(...entries: Record<string, unknown>[]): StatsReportLike {
  return new Map(entries.map((entry, index) => [`entry-${index}`, entry]));
}

function inbound(fields: Record<string, unknown>): Record<string, unknown> {
  return { type: "inbound-rtp", kind: "audio", ...fields };
}

const context: ReceiveStatsContext = {
  duckActive: true,
  audioContext: "running",
  pageUptimeS: 3_600,
  roomUptimeS: 1_200,
};

interface Harness {
  readonly sampler: ReceiveStatsSampler;
  readonly submitted: ReceiveStatsSample[];
  readonly reads: Array<(value: StatsReportLike | undefined) => void>;
  readonly readFailures: Array<(error: unknown) => void>;
  readonly timers: Array<{ ms: number; fire: () => void }>;
  clock: number;
}

function harness(cap = 4): Harness {
  const state = {
    submitted: [] as ReceiveStatsSample[],
    reads: [] as Array<(value: StatsReportLike | undefined) => void>,
    readFailures: [] as Array<(error: unknown) => void>,
    timers: [] as Array<{ ms: number; fire: () => void }>,
    clock: 0,
  };
  const sampler = new ReceiveStatsSampler({
    readStats: () =>
      new Promise<StatsReportLike | undefined>((resolve, reject) => {
        state.reads.push(resolve);
        state.readFailures.push(reject);
      }),
    submit: async (sample) => {
      state.submitted.push(sample);
    },
    now: () => state.clock,
    delay: (ms) =>
      new Promise<void>((resolve) => {
        state.timers.push({ ms, fire: resolve });
      }),
    cap,
  });
  return Object.assign(state, { sampler });
}

async function settle(): Promise<void> {
  for (let index = 0; index < 20; index += 1) await Promise.resolve();
}

function fireTimers(state: Harness, ms: number): void {
  for (const timer of state.timers.filter((entry) => entry.ms === ms)) timer.fire();
}

describe("inbound audio counters", () => {
  it("reads the one inbound audio entry and omits fields it lacks", () => {
    expect(
      inboundAudioCounters(
        report(
          { type: "inbound-rtp", kind: "video", concealedSamples: 99 },
          inbound({
            removedSamplesForAcceleration: 10,
            concealedSamples: 4,
            jitterBufferTargetDelay: 0.5,
            jitterBufferEmittedCount: 4_800,
            silentConcealedSamples: "7",
            trackIdentifier: "not-copied",
          }),
        ),
      ),
    ).toEqual({
      removedSamplesForAcceleration: 10,
      concealedSamples: 4,
      jitterBufferTargetDelay: 0.5,
      jitterBufferEmittedCount: 4_800,
    });
  });

  it("refuses a report without exactly one inbound audio entry", () => {
    expect(inboundAudioCounters(undefined)).toBeNull();
    expect(inboundAudioCounters(report({ type: "outbound-rtp", kind: "audio" }))).toBeNull();
    expect(inboundAudioCounters(report(inbound({}), inbound({})))).toBeNull();
  });
});

describe("receive stats delta", () => {
  it("reports deltas and the mean target delay over the window", () => {
    expect(
      receiveStatsDelta(
        {
          removedSamplesForAcceleration: 100,
          insertedSamplesForDeceleration: 20,
          concealedSamples: 1_000,
          silentConcealedSamples: 900,
          concealmentEvents: 3,
          jitterBufferTargetDelay: 10,
          jitterBufferEmittedCount: 48_000,
        },
        {
          removedSamplesForAcceleration: 580,
          insertedSamplesForDeceleration: 20,
          concealedSamples: 1_240,
          silentConcealedSamples: 960,
          concealmentEvents: 5,
          jitterBufferTargetDelay: 1_450,
          jitterBufferEmittedCount: 72_000,
        },
      ),
    ).toEqual({
      removedSamplesForAcceleration: 480,
      insertedSamplesForDeceleration: 0,
      concealedSamples: 240,
      silentConcealedSamples: 60,
      concealmentEvents: 2,
      meanTargetDelayMs: 60,
    });
  });

  it("leaves a field absent, never zero, when either side lacks it or it went backwards", () => {
    expect(
      receiveStatsDelta(
        { concealedSamples: 50, removedSamplesForAcceleration: 9, jitterBufferTargetDelay: 1 },
        { concealedSamples: 40, insertedSamplesForDeceleration: 3, jitterBufferTargetDelay: 2 },
      ),
    ).toEqual({});
    expect(
      receiveStatsDelta(
        { jitterBufferTargetDelay: 1, jitterBufferEmittedCount: 10 },
        { jitterBufferTargetDelay: 2, jitterBufferEmittedCount: 10 },
      ),
    ).toEqual({});
  });
});

describe("receive stats sampler", () => {
  it("diffs the chunk-start baseline against a read 500 ms after the chunk end", async () => {
    const state = harness();
    state.sampler.chunkStarted();
    state.reads[0]!(report(inbound({ concealedSamples: 10, concealmentEvents: 1 })));
    await settle();
    state.clock = 2_000;
    state.sampler.chunkEnded(() => context);
    await settle();
    expect(state.timers.map((timer) => timer.ms)).toContain(500);
    expect(state.reads).toHaveLength(1);
    state.clock = 2_500;
    fireTimers(state, 500);
    await settle();
    expect(state.reads).toHaveLength(2);
    state.reads[1]!(report(inbound({ concealedSamples: 25, concealmentEvents: 1 })));
    await settle();
    expect(state.submitted).toEqual([
      {
        windowMs: 2_500,
        concealedSamples: 15,
        concealmentEvents: 0,
        duckActive: true,
        audioContext: "running",
        pageUptimeS: 3_600,
        roomUptimeS: 1_200,
      },
    ]);
  });

  it("takes its baseline at the chunk end when no chunk start was seen", async () => {
    const state = harness();
    state.clock = 100;
    state.sampler.chunkEnded(() => ({ ...context, audioContext: null }));
    expect(state.reads).toHaveLength(1);
    state.reads[0]!(report(inbound({ insertedSamplesForDeceleration: 2 })));
    await settle();
    state.clock = 600;
    fireTimers(state, 500);
    await settle();
    state.reads[1]!(report(inbound({ insertedSamplesForDeceleration: 7 })));
    await settle();
    expect(state.submitted).toEqual([
      {
        windowMs: 500,
        insertedSamplesForDeceleration: 5,
        duckActive: true,
        pageUptimeS: 3_600,
        roomUptimeS: 1_200,
      },
    ]);
  });

  it("stops at its cap, counting failed samples too", async () => {
    const state = harness(2);
    state.sampler.chunkEnded(() => context);
    state.sampler.chunkEnded(() => context);
    state.sampler.chunkEnded(() => context);
    state.sampler.chunkStarted();
    state.sampler.chunkEnded(() => context);
    expect(state.reads).toHaveLength(2);
    state.readFailures[0]!(new Error("stats unavailable"));
    await settle();
    state.sampler.chunkEnded(() => context);
    expect(state.reads).toHaveLength(2);
  });

  it("never throws or blocks the end-of-speech handler when stats fail or hang", async () => {
    const throwing = new ReceiveStatsSampler({
      readStats: () => {
        throw new Error("synchronous stats failure");
      },
      submit: async () => {
        throw new Error("submit must not be reached");
      },
      now: () => 0,
      delay: () => Promise.resolve(),
    });
    expect(() => throwing.chunkStarted()).not.toThrow();
    expect(
      throwing.chunkEnded(() => {
        throw new Error("context failure");
      }),
    ).toBeUndefined();
    expect(throwing.chunkEnded(() => context)).toBeUndefined();

    const state = harness();
    let handlerFinished = false;
    state.sampler.chunkEnded(() => context);
    handlerFinished = true;
    expect(handlerFinished).toBe(true);
    // The baseline read hangs; its bound fires and the sample is dropped, not submitted.
    fireTimers(state, 1_000);
    await settle();
    fireTimers(state, 500);
    await settle();
    fireTimers(state, 1_000);
    await settle();
    expect(state.submitted).toEqual([]);

    const rejecting = harness();
    rejecting.sampler.chunkEnded(() => context);
    rejecting.readFailures[0]!(new Error("getStats rejected"));
    await settle();
    fireTimers(rejecting, 500);
    await settle();
    rejecting.reads[1]?.(report(inbound({ concealedSamples: 1 })));
    await settle();
    expect(rejecting.submitted).toEqual([]);
  });
});
