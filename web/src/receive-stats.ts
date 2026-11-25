// Content-free receive-side evidence around the end of each delivered speech publication (#556).
// One sample per delivered publication, windowed from the start of its LAST chunk (each chunk
// start replaces the baseline) to 500 ms after the publication's end. Only integer deltas of the
// speech track's inbound audio counters, whether both stats reads succeeded, two uptimes, the
// duck flag and a closed audio-context enum leave the page: no text, ids, track ids or audio.

const COUNTER_FIELDS = [
  "removedSamplesForAcceleration",
  "insertedSamplesForDeceleration",
  "concealedSamples",
  "silentConcealedSamples",
  "concealmentEvents",
  "jitterBufferTargetDelay",
  "jitterBufferEmittedCount",
] as const;

const DELTA_FIELDS = [
  "removedSamplesForAcceleration",
  "insertedSamplesForDeceleration",
  "concealedSamples",
  "silentConcealedSamples",
  "concealmentEvents",
] as const;

const AUDIO_CONTEXT_STATES = ["running", "suspended", "closed", "interrupted"] as const;

export const MAX_RECEIVE_COUNT = 1_000_000_000;
export const MAX_RECEIVE_WINDOW_MS = 600_000;
export const MAX_RECEIVE_TARGET_DELAY_MS = 60_000;
export const MAX_RECEIVE_UPTIME_S = 31_536_000;
export const RECEIVE_SAMPLE_CAP = 2_048;
const SETTLE_MS = 500;
const READ_TIMEOUT_MS = 1_000;
const MAX_IN_FLIGHT = 4;

export type AudioContextStateName = (typeof AUDIO_CONTEXT_STATES)[number];
export type InboundAudioCounters = Partial<Record<(typeof COUNTER_FIELDS)[number], number>>;
export type ReceiveStatsDelta = Partial<
  Record<(typeof DELTA_FIELDS)[number] | "meanTargetDelayMs", number>
>;

export interface StatsReportLike {
  forEach(callback: (value: unknown) => void): void;
}

export interface ReceiveStatsContext {
  readonly duckActive: boolean;
  readonly audioContext: AudioContextStateName | null;
  readonly pageUptimeS: number;
  readonly roomUptimeS: number;
}

// statsRead is false when either read failed, timed out or found no inbound audio entry; the
// deltas are then all absent, so a failed read never looks like an unobserved publication.
export type ReceiveStatsSample = ReceiveStatsDelta & {
  readonly windowMs: number;
  readonly statsRead: boolean;
  readonly duckActive: boolean;
  readonly audioContext?: AudioContextStateName;
  readonly pageUptimeS: number;
  readonly roomUptimeS: number;
};

export function audioContextStateName(value: unknown): AudioContextStateName | null {
  return (AUDIO_CONTEXT_STATES as readonly unknown[]).includes(value)
    ? (value as AudioContextStateName)
    : null;
}

export function inboundAudioCounters(
  report: StatsReportLike | undefined,
): InboundAudioCounters | null {
  if (report === undefined) return null;
  const entries: Record<string, unknown>[] = [];
  report.forEach((value) => {
    if (typeof value !== "object" || value === null) return;
    const entry = value as Record<string, unknown>;
    if (entry.type === "inbound-rtp" && entry.kind === "audio") entries.push(entry);
  });
  if (entries.length !== 1) return null;
  const counters: InboundAudioCounters = {};
  for (const field of COUNTER_FIELDS) {
    const value = entries[0]![field];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      counters[field] = value;
    }
  }
  return counters;
}

function boundedInteger(value: number, maximum: number): number | undefined {
  const rounded = Math.round(value);
  return Number.isFinite(value) && rounded >= 0 && rounded <= maximum ? rounded : undefined;
}

export function receiveStatsDelta(
  before: InboundAudioCounters,
  after: InboundAudioCounters,
): ReceiveStatsDelta {
  const delta: ReceiveStatsDelta = {};
  for (const field of DELTA_FIELDS) {
    const start = before[field];
    const end = after[field];
    if (start === undefined || end === undefined || end < start) continue;
    const value = boundedInteger(end - start, MAX_RECEIVE_COUNT);
    if (value !== undefined) delta[field] = value;
  }
  const targetStart = before.jitterBufferTargetDelay;
  const targetEnd = after.jitterBufferTargetDelay;
  const emittedStart = before.jitterBufferEmittedCount;
  const emittedEnd = after.jitterBufferEmittedCount;
  if (
    targetStart !== undefined &&
    targetEnd !== undefined &&
    emittedStart !== undefined &&
    emittedEnd !== undefined &&
    targetEnd >= targetStart &&
    emittedEnd > emittedStart
  ) {
    const mean = boundedInteger(
      ((targetEnd - targetStart) / (emittedEnd - emittedStart)) * 1_000,
      MAX_RECEIVE_TARGET_DELAY_MS,
    );
    if (mean !== undefined) delta.meanTargetDelayMs = mean;
  }
  return delta;
}

interface Snapshot {
  readonly at: number;
  readonly counters: InboundAudioCounters | null;
}

export interface ReceiveStatsSamplerOptions {
  readonly readStats: () => Promise<StatsReportLike | undefined>;
  readonly submit: (sample: ReceiveStatsSample) => Promise<void>;
  readonly now: () => number;
  readonly delay: (ms: number) => Promise<void>;
  readonly cap?: number;
}

interface EndedContext {
  readonly duckActive: boolean;
  readonly audioContext: AudioContextStateName | null;
  readonly pageUptimeS: number;
  readonly roomUptimeS: number;
  readonly endedAt: number;
}

// Why a delivered publication was not sampled. "capped" is returned once, then "ignored".
export type ReceiveStatsDisposition =
  | "sampled"
  | "capped"
  | "ignored"
  | "busy"
  | "invalid-context";

function sampleFrom(
  ended: EndedContext,
  before: Snapshot | null,
  after: Snapshot | null,
): ReceiveStatsSample {
  const startAt = before?.at ?? ended.endedAt;
  const endAt = after?.at ?? startAt;
  const span = Math.round(endAt - startAt);
  const windowMs = Number.isFinite(span) ? Math.min(Math.max(span, 0), MAX_RECEIVE_WINDOW_MS) : 0;
  const beforeCounters = before === null ? null : before.counters;
  const afterCounters = after === null ? null : after.counters;
  const statsRead = beforeCounters !== null && afterCounters !== null;
  return {
    windowMs,
    statsRead,
    ...(beforeCounters !== null && afterCounters !== null
      ? receiveStatsDelta(beforeCounters, afterCounters)
      : {}),
    duckActive: ended.duckActive,
    ...(ended.audioContext === null ? {} : { audioContext: ended.audioContext }),
    pageUptimeS: ended.pageUptimeS,
    roomUptimeS: ended.roomUptimeS,
  };
}

// Each call returns at once and never throws. Every read is time-bound, and every admitted
// publication submits exactly one sample from a `finally`: a failed read is reported as
// statsRead=false, never dropped. Admitted attempts count toward the cap.
export class ReceiveStatsSampler {
  private baseline: Promise<Snapshot> | null = null;
  private attempts = 0;
  private capReported = false;
  private inFlight = 0;
  private readonly cap: number;

  constructor(private readonly options: ReceiveStatsSamplerOptions) {
    this.cap = options.cap ?? RECEIVE_SAMPLE_CAP;
  }

  // Each chunk start replaces the baseline, so a publication of several chunks is windowed
  // from its last chunk's start. A stale first-chunk baseline would outlive an interrupted reply.
  chunkStarted(): void {
    if (this.attempts >= this.cap) return;
    this.baseline = this.snapshot();
  }

  publicationDelivered(context: ReceiveStatsContext): ReceiveStatsDisposition {
    if (this.attempts >= this.cap) {
      if (this.capReported) return "ignored";
      this.capReported = true;
      return "capped";
    }
    if (this.inFlight >= MAX_IN_FLIGHT) return "busy";
    let ended: EndedContext;
    try {
      const pageUptimeS = boundedInteger(Math.floor(context.pageUptimeS), MAX_RECEIVE_UPTIME_S);
      const roomUptimeS = boundedInteger(Math.floor(context.roomUptimeS), MAX_RECEIVE_UPTIME_S);
      if (
        pageUptimeS === undefined ||
        roomUptimeS === undefined ||
        typeof context.duckActive !== "boolean"
      ) {
        return "invalid-context";
      }
      ended = {
        duckActive: context.duckActive,
        audioContext: audioContextStateName(context.audioContext),
        pageUptimeS,
        roomUptimeS,
        endedAt: this.options.now(),
      };
    } catch {
      return "invalid-context";
    }
    this.attempts += 1;
    this.inFlight += 1;
    const baseline = this.baseline ?? this.snapshot();
    this.baseline = null;
    void this.complete(baseline, ended)
      .catch(() => undefined)
      .finally(() => {
        this.inFlight -= 1;
      });
    return "sampled";
  }

  private async complete(baseline: Promise<Snapshot>, ended: EndedContext): Promise<void> {
    let before: Snapshot | null = null;
    let after: Snapshot | null = null;
    try {
      await this.options.delay(SETTLE_MS);
      after = await this.snapshot();
      before = await baseline;
    } finally {
      await this.options.submit(sampleFrom(ended, before, after));
    }
  }

  private snapshot(): Promise<Snapshot> {
    let at = 0;
    try {
      at = this.options.now();
      const read = this.options.readStats().then(
        (report) => inboundAudioCounters(report),
        () => null,
      );
      const bound = this.options.delay(READ_TIMEOUT_MS).then(() => null);
      return Promise.race([read, bound]).then(
        (counters) => ({ at, counters }),
        () => ({ at, counters: null }),
      );
    } catch {
      return Promise.resolve({ at, counters: null });
    }
  }
}
