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

const unreadSample: ReceiveStatsSample = {
  windowMs: 0,
  statsRead: false,
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
  it("diffs the chunk-start baseline against a read 500 ms after the publication end", async () => {
    const state = harness();
    state.sampler.chunkStarted();
    state.reads[0]!(report(inbound({ concealedSamples: 10, concealmentEvents: 1 })));
    await settle();
    state.clock = 2_000;
    expect(state.sampler.publicationDelivered(context)).toBe("sampled");
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
        statsRead: true,
        concealedSamples: 15,
        concealmentEvents: 0,
        duckActive: true,
        audioContext: "running",
        pageUptimeS: 3_600,
        roomUptimeS: 1_200,
      },
    ]);
  });

  it("windows a multi-chunk publication from its last chunk's start, in one sample", async () => {
    const state = harness();
    state.sampler.chunkStarted();
    state.reads[0]!(report(inbound({ concealedSamples: 100 })));
    await settle();
    state.clock = 3_000;
    state.sampler.chunkStarted();
    state.reads[1]!(report(inbound({ concealedSamples: 400 })));
    await settle();
    state.clock = 4_000;
    expect(state.sampler.publicationDelivered(context)).toBe("sampled");
    state.clock = 4_500;
    fireTimers(state, 500);
    await settle();
    state.reads[2]!(report(inbound({ concealedSamples: 450 })));
    await settle();
    expect(state.submitted).toEqual([
      {
        windowMs: 1_500,
        statsRead: true,
        concealedSamples: 50,
        duckActive: true,
        audioContext: "running",
        pageUptimeS: 3_600,
        roomUptimeS: 1_200,
      },
    ]);
  });

  it("takes its baseline at the publication end when no chunk start was seen", async () => {
    const state = harness();
    state.clock = 100;
    state.sampler.publicationDelivered({ ...context, audioContext: null });
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
        statsRead: true,
        insertedSamplesForDeceleration: 5,
        duckActive: true,
        pageUptimeS: 3_600,
        roomUptimeS: 1_200,
      },
    ]);
  });

  it("stops at its cap, counting failed samples too, and says so once", async () => {
    const state = harness(2);
    expect(state.sampler.publicationDelivered(context)).toBe("sampled");
    expect(state.sampler.publicationDelivered(context)).toBe("sampled");
    expect(state.sampler.publicationDelivered(context)).toBe("capped");
    state.sampler.chunkStarted();
    expect(state.sampler.publicationDelivered(context)).toBe("ignored");
    expect(state.reads).toHaveLength(2);
    state.readFailures[0]!(new Error("stats unavailable"));
    await settle();
    state.sampler.publicationDelivered(context);
    expect(state.reads).toHaveLength(2);
  });

  it("refuses an invalid context without reading or counting it", () => {
    const state = harness(1);
    expect(state.sampler.publicationDelivered({ ...context, pageUptimeS: Number.NaN })).toBe(
      "invalid-context",
    );
    expect(state.reads).toHaveLength(0);
    expect(state.sampler.publicationDelivered(context)).toBe("sampled");
  });

  it("reports a failed or hung read as statsRead false, never throwing or blocking", async () => {
    const throwingSubmitted: ReceiveStatsSample[] = [];
    const throwing = new ReceiveStatsSampler({
      readStats: () => {
        throw new Error("synchronous stats failure");
      },
      submit: async (sample) => {
        throwingSubmitted.push(sample);
      },
      now: () => 0,
      delay: () => Promise.resolve(),
    });
    expect(() => throwing.chunkStarted()).not.toThrow();
    expect(throwing.publicationDelivered(context)).toBe("sampled");
    await settle();
    expect(throwingSubmitted).toEqual([unreadSample]);

    // The baseline read hangs; its bound fires and the sample still goes out, unread.
    const state = harness();
    state.sampler.publicationDelivered(context);
    fireTimers(state, 1_000);
    await settle();
    fireTimers(state, 500);
    await settle();
    state.reads[1]!(report(inbound({ concealedSamples: 1 })));
    await settle();
    expect(state.submitted).toEqual([unreadSample]);

    // Hung reads are released by their bound, so they never hold the in-flight slots.
    const hung = harness(10);
    for (let index = 0; index < 4; index += 1) hung.sampler.publicationDelivered(context);
    expect(hung.sampler.publicationDelivered(context)).toBe("busy");
    fireTimers(hung, 1_000);
    await settle();
    fireTimers(hung, 500);
    await settle();
    fireTimers(hung, 1_000);
    await settle();
    expect(hung.submitted).toEqual([unreadSample, unreadSample, unreadSample, unreadSample]);
    const readsBefore = hung.reads.length;
    expect(hung.sampler.publicationDelivered(context)).toBe("sampled");
    expect(hung.reads.length).toBe(readsBefore + 1);

    const rejecting = harness();
    rejecting.sampler.publicationDelivered(context);
    rejecting.readFailures[0]!(new Error("getStats rejected"));
    await settle();
    fireTimers(rejecting, 500);
    await settle();
    rejecting.reads[1]!(report(inbound({ concealedSamples: 1 })));
    await settle();
    expect(rejecting.submitted).toEqual([unreadSample]);
  });

  it("still submits when the settle delay itself fails", async () => {
    const submitted: ReceiveStatsSample[] = [];
    const sampler = new ReceiveStatsSampler({
      readStats: async () => report(inbound({ concealedSamples: 1 })),
      submit: async (sample) => {
        submitted.push(sample);
      },
      now: () => 0,
      delay: (ms) => (ms === 500 ? Promise.reject(new Error("timer failed")) : new Promise(() => {})),
    });
    sampler.publicationDelivered(context);
    await settle();
    expect(submitted).toEqual([unreadSample]);
  });
});
