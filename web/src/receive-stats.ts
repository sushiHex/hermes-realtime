// Content-free receive-side evidence around the end of each delivered speech chunk (#556).
// Only integer deltas of the speech track's inbound audio counters, two uptimes, the duck flag
// and a closed audio-context enum leave the page: no text, identifiers, track ids or audio.

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

export type ReceiveStatsSample = ReceiveStatsDelta & {
  readonly windowMs: number;
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

// Each call returns at once and never throws; every read is time-bound and any failure drops
// that one sample. Attempts, failed or not, count toward the cap.
export class ReceiveStatsSampler {
  private baseline: Promise<Snapshot> | null = null;
  private attempts = 0;
  private inFlight = 0;
  private readonly cap: number;

  constructor(private readonly options: ReceiveStatsSamplerOptions) {
    this.cap = options.cap ?? RECEIVE_SAMPLE_CAP;
  }

  chunkStarted(): void {
    if (this.attempts >= this.cap) return;
    this.baseline = this.snapshot();
  }

  chunkEnded(context: () => ReceiveStatsContext): void {
    if (this.attempts >= this.cap || this.inFlight >= MAX_IN_FLIGHT) return;
    this.attempts += 1;
    this.inFlight += 1;
    const baseline = this.baseline ?? this.snapshot();
    this.baseline = null;
    void this.complete(baseline, context)
      .catch(() => undefined)
      .finally(() => {
        this.inFlight -= 1;
      });
  }

  private async complete(
    baseline: Promise<Snapshot>,
    context: () => ReceiveStatsContext,
  ): Promise<void> {
    const ended = context();
    await this.options.delay(SETTLE_MS);
    const after = await this.snapshot();
    const before = await baseline;
    if (before.counters === null || after.counters === null) return;
    const windowMs = boundedInteger(after.at - before.at, MAX_RECEIVE_WINDOW_MS);
    const pageUptimeS = boundedInteger(Math.floor(ended.pageUptimeS), MAX_RECEIVE_UPTIME_S);
    const roomUptimeS = boundedInteger(Math.floor(ended.roomUptimeS), MAX_RECEIVE_UPTIME_S);
    if (
      windowMs === undefined ||
      pageUptimeS === undefined ||
      roomUptimeS === undefined ||
      typeof ended.duckActive !== "boolean"
    ) {
      return;
    }
    const audioContext = audioContextStateName(ended.audioContext);
    await this.options.submit({
      windowMs,
      ...receiveStatsDelta(before.counters, after.counters),
      duckActive: ended.duckActive,
      ...(audioContext === null ? {} : { audioContext }),
      pageUptimeS,
      roomUptimeS,
    });
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
