import type {
  BootstrapCredential,
  ModelCatalog,
  SessionModelConfiguration,
  SessionTokenUsage,
} from "./protocol";

export type ClientState =
  | "idle"
  | "bootstrapping"
  | "connecting"
  | "preparing"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "stopping"
  | "stopped"
  | "error";

export interface ConnectionPresentation {
  readonly label: string;
  readonly presentationState: string;
  readonly headline: string;
  readonly detail: string;
}

export interface SessionTogglePresentation {
  readonly label: string;
  readonly disabled: boolean;
  readonly action: "connect" | "stop";
}

export function sessionTogglePresentation(
  state: ClientState,
  canStartSession = true,
  remoteStopRequired = false,
): SessionTogglePresentation {
  if (typeof canStartSession !== "boolean") {
    throw new TypeError("canStartSession must be a boolean");
  }
  if (typeof remoteStopRequired !== "boolean") {
    throw new TypeError("remoteStopRequired must be a boolean");
  }
  if (state === "bootstrapping" || state === "connecting") {
    return { label: "Connecting…", disabled: true, action: "connect" };
  }
  if (state === "stopping") {
    return { label: "Stopping…", disabled: true, action: "stop" };
  }
  if (
    state === "preparing" ||
    state === "connected" ||
    state === "reconnecting" ||
    remoteStopRequired
  ) {
    return { label: "Stop session", disabled: false, action: "stop" };
  }
  if (!canStartSession) {
    return { label: "Fresh launch required", disabled: true, action: "connect" };
  }
  return { label: "Connect", disabled: false, action: "connect" };
}

export type VoicePathState = "waiting" | "ready" | "timed-out";

export function taskStatusPresentation(status: string, elapsedMs: number): string {
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
    throw new TypeError("task elapsed time must be finite and non-negative");
  }
  const elapsedSeconds = Math.floor(elapsedMs / 1000);
  const minutes = Math.floor(elapsedSeconds / 60);
  const seconds = String(elapsedSeconds % 60).padStart(2, "0");
  const duration = `${minutes}:${seconds}`;
  if (status === "active") return `Running · ${duration} elapsed`;
  if (status === "completed") return `Completed in ${duration}`;
  if (status === "failed") return `Failed after ${duration}`;
  if (status === "interrupted") return `Stopped after ${duration}`;
  if (status === "rejected") return "Not started";
  if (status === "cancelling") return `Stopping · ${duration} elapsed`;
  throw new TypeError("task status is unsupported");
}

export function taskMetaPresentation(status: string): string {
  if (status === "active") return "You can keep talking while this runs.";
  if (status === "cancelling") return "Stopping the task now.";
  if (status === "completed") return "Background work finished.";
  if (status === "failed") return "Background work failed.";
  if (status === "interrupted") return "Background work stopped.";
  if (status === "rejected") return "Background work was not started.";
  throw new TypeError("task status is unsupported");
}

export type SessionModelPresentationRow = readonly [label: string, value: string];

export interface MarkdownEmphasisRun {
  readonly text: string;
  readonly sourceStart: number;
  readonly sourceEnd: number;
  readonly emphasis: "none" | "italic" | "bold" | "bold-italic";
}

const punctuationCharacter = /^\p{P}$/u;

function isPunctuation(character: string | undefined): boolean {
  return character !== undefined && punctuationCharacter.test(character);
}

export function markdownEmphasisRuns(text: string): readonly MarkdownEmphasisRun[] {
  const hidden = new Uint8Array(text.length);
  const italicDelta = new Int32Array(text.length + 1);
  const boldDelta = new Int32Array(text.length + 1);
  const openings = new Map<string, Array<readonly [number, number]>>();
  let index = 0;
  while (index < text.length) {
    const marker = text[index];
    if (marker !== "*" && marker !== "_") {
      index += 1;
      continue;
    }
    let end = index + 1;
    while (end < text.length && text[end] === marker) end += 1;
    const runLength = end - index;
    let precedingBackslashes = 0;
    for (let cursor = index - 1; cursor >= 0 && text[cursor] === "\\"; cursor -= 1) {
      precedingBackslashes += 1;
    }
    if (runLength > 3 || precedingBackslashes % 2 === 1) {
      index = end;
      continue;
    }

    const previous = index > 0 ? text[index - 1] : undefined;
    const following = end < text.length ? text[end] : undefined;
    const previousSpace = previous === undefined || /\s/u.test(previous);
    const followingSpace = following === undefined || /\s/u.test(following);
    const leftFlanking =
      !followingSpace &&
      (!isPunctuation(following) || previousSpace || isPunctuation(previous));
    const rightFlanking =
      !previousSpace &&
      (!isPunctuation(previous) || followingSpace || isPunctuation(following));
    const canOpen =
      marker === "_"
        ? leftFlanking && (!rightFlanking || isPunctuation(previous))
        : leftFlanking;
    const canClose =
      marker === "_"
        ? rightFlanking && (!leftFlanking || isPunctuation(following))
        : rightFlanking;

    const key = `${marker}${runLength}`;
    const candidates = openings.get(key);
    if (canClose && candidates !== undefined && candidates.length > 0) {
      const [openingStart, openingEnd] = candidates.pop()!;
      hidden.fill(1, openingStart, openingEnd);
      hidden.fill(1, index, end);
      if (runLength === 1 || runLength === 3) {
        italicDelta[openingEnd] = (italicDelta[openingEnd] ?? 0) + 1;
        italicDelta[index] = (italicDelta[index] ?? 0) - 1;
      }
      if (runLength === 2 || runLength === 3) {
        boldDelta[openingEnd] = (boldDelta[openingEnd] ?? 0) + 1;
        boldDelta[index] = (boldDelta[index] ?? 0) - 1;
      }
    } else if (canOpen) {
      const retained = candidates ?? [];
      retained.push([index, end]);
      openings.set(key, retained);
    }
    index = end;
  }

  const runs: MarkdownEmphasisRun[] = [];
  let italicDepth = 0;
  let boldDepth = 0;
  let runStart: number | null = null;
  let runEmphasis: MarkdownEmphasisRun["emphasis"] = "none";
  const flush = (end: number): void => {
    if (runStart === null || runStart === end) return;
    runs.push({
      text: text.slice(runStart, end),
      sourceStart: runStart,
      sourceEnd: end,
      emphasis: runEmphasis,
    });
    runStart = null;
  };
  for (let offset = 0; offset < text.length; offset += 1) {
    italicDepth += italicDelta[offset] ?? 0;
    boldDepth += boldDelta[offset] ?? 0;
    if (hidden[offset] === 1) {
      flush(offset);
      continue;
    }
    const emphasis: MarkdownEmphasisRun["emphasis"] =
      italicDepth > 0 && boldDepth > 0
        ? "bold-italic"
        : boldDepth > 0
          ? "bold"
          : italicDepth > 0
            ? "italic"
            : "none";
    if (runStart === null) {
      runStart = offset;
      runEmphasis = emphasis;
    } else if (runEmphasis !== emphasis) {
      flush(offset);
      runStart = offset;
      runEmphasis = emphasis;
    }
  }
  flush(text.length);
  return runs;
}

export function karaokeSegmentPendingStates(
  segmentIds: readonly string[],
  activeTiming: { readonly chunkId: string; readonly segmentId: string },
): readonly boolean[] {
  const activeIndex = segmentIds.indexOf(activeTiming.segmentId);
  return segmentIds.map((_, index) => activeIndex >= 0 && index > activeIndex);
}

export function karaokeGapPendingStates(
  wordCount: number,
  completedWordCount: number,
): readonly boolean[] {
  return Array.from({ length: wordCount + 1 }, (_, index) => index > completedWordCount);
}

export function reasoningEffortLabel(effort: string): string {
  return effort === "ultra" ? "ultra · provider effort" : effort;
}

export function modelSelectionFor(
  catalog: ModelCatalog,
  model: string,
  preferredEffort: string,
): { readonly model: string; readonly effort: string } {
  const selected = catalog.models.find((item) => item.model === model);
  if (selected === undefined) throw new Error("Model is absent from the active catalog");
  return {
    model,
    effort: selected.supportedEfforts.includes(preferredEffort)
      ? preferredEffort
      : selected.defaultEffort,
  };
}

export function formatSessionModelConfiguration(
  configuration: SessionModelConfiguration,
): readonly SessionModelPresentationRow[] {
  return [
    ["Provider", configuration.provider],
    ["Model", configuration.model],
    [
      "Access",
      configuration.authentication === "subscription"
        ? "Subscription"
        : configuration.authentication === "api-token"
          ? "API token"
          : "Local model",
    ],
    ["Reasoning effort", configuration.effort ?? "Not applicable"],
    ["Transport", configuration.transport],
    [
      "Context window",
      configuration.contextWindowTokens === null
        ? "Not reported"
        : `${configuration.contextWindowTokens.toLocaleString("en-US")} tokens`,
    ],
    [
      "Token usage",
      configuration.reportsTokenUsage ? "Reported live" : "Not reported by transport",
    ],
  ];
}

export function formatSessionTokenUsage(
  usage: SessionTokenUsage,
): readonly SessionModelPresentationRow[] {
  const format = (value: number): string => value.toLocaleString("en-US");
  return [
    ["Session total", format(usage.totalTokens)],
    ["Input", format(usage.inputTokens)],
    ["Cached input", format(usage.cachedInputTokens)],
    ["Output", format(usage.outputTokens)],
    ["Reasoning output", format(usage.reasoningOutputTokens)],
  ];
}

export function connectionPresentation(
  state: ClientState,
  microphoneReady: boolean | null,
  voicePathState: VoicePathState,
): ConnectionPresentation {
  if (state === "preparing") {
    return {
      label: "Preparing microphone…",
      presentationState: "preparing",
      headline: "Preparing microphone",
      detail: "Please wait before speaking. Audio capture is not ready yet.",
    };
  }
  if (state === "connected" && microphoneReady === true) {
    if (voicePathState === "waiting") {
      return {
        label: "Connecting speech path…",
        presentationState: "preparing",
        headline: "Connecting speech path",
        detail: "Please wait before speaking. The server has not confirmed microphone audio yet.",
      };
    }
    if (voicePathState === "timed-out") {
      return {
        label: "Microphone not reaching server",
        presentationState: "error",
        headline: "Speech path unavailable",
        detail: "Reconnect to retry voice, or use typed input below.",
      };
    }
    return {
      label: "Listening",
      presentationState: "connected",
      headline: "Listening — speak naturally",
      detail: "Pause briefly when you need to; your words stay together until your turn ends.",
    };
  }
  if (state === "connected" && microphoneReady === false) {
    return {
      label: "Ready to type — microphone unavailable",
      presentationState: "typed-only",
      headline: "Microphone unavailable",
      detail: "Use the message box below, or reconnect after checking microphone permission.",
    };
  }
  const label = state[0]?.toUpperCase() + state.slice(1);
  return {
    label,
    presentationState: state,
    headline: state === "idle" ? "Ready when you are" : label,
    detail:
      state === "idle"
        ? "Connect, then wait for Listening before speaking."
        : "Voice capture is not ready.",
  };
}

export type PartialTranscriptDisposition = "keep" | "clear" | "retain-interrupted";

export function partialTranscriptDisposition(
  eventKind: string,
  role: "user" | "assistant" | null,
): PartialTranscriptDisposition {
  if (eventKind === "session_stopped") return "clear";
  if (eventKind === "interrupt_requested") {
    return role === "assistant" ? "retain-interrupted" : "clear";
  }
  if (eventKind === "speech_ended" && role === "user") return "clear";
  return "keep";
}

export interface LaunchLocation {
  readonly hash: string;
  readonly pathname: string;
  readonly search: string;
}

export interface LaunchHistory {
  replaceState(data: null, unused: "", url: string): void;
}

const CAPABILITY_FRAGMENT = /^#bootstrap=([A-Za-z0-9_-]{43,128})$/;

export function takeBootstrapCapability(
  location: LaunchLocation,
  history: LaunchHistory,
): string {
  const fragment = location.hash;
  history.replaceState(null, "", `${location.pathname}${location.search}`);
  const match = CAPABILITY_FRAGMENT.exec(fragment);
  if (match === null || match[1] === undefined) {
    throw new Error("A fresh bootstrap launch capability is required");
  }
  return match[1];
}

export function takeOptionalBootstrapCapability(
  location: LaunchLocation,
  history: LaunchHistory,
): string | null {
  if (location.hash === "") return null;
  return takeBootstrapCapability(location, history);
}

export interface BootstrapRequestParameters {
  readonly path: "/api/v1/bootstrap" | "/api/v1/stable-bootstrap";
  readonly bearer: string | null;
}

export function bootstrapRequestParameters(
  stableLaunch: boolean,
  capability: string | null,
): BootstrapRequestParameters {
  if (stableLaunch) {
    return { path: "/api/v1/stable-bootstrap", bearer: null };
  }
  if (capability === null) {
    throw new Error("A fresh bootstrap launch capability is required");
  }
  return { path: "/api/v1/bootstrap", bearer: capability };
}

export interface RebindRequestParameters {
  readonly path: "/api/v1/rebind" | "/api/v1/stable-rebind";
  readonly bearer: string | null;
  readonly body: string | null;
}

export function rebindRequestParameters(
  stableLaunch: boolean,
  credential: BootstrapCredential,
  requestId: string,
): RebindRequestParameters {
  if (!/^rebind_[A-Za-z0-9-]{9,121}$/.test(requestId)) {
    throw new Error("rebind request identifier is invalid");
  }
  if (stableLaunch) {
    return {
      path: "/api/v1/stable-rebind",
      bearer: null,
      body: JSON.stringify({
        participantIdentity: credential.participantIdentity,
        requestId,
      }),
    };
  }
  return {
    path: "/api/v1/rebind",
    bearer: credential.token,
    body: JSON.stringify({ requestId }),
  };
}

export function rebindFailureAllowsFreshBootstrap(status: number): boolean {
  return status === 409;
}

export async function settleStableDisconnect(
  stopRemote: () => Promise<void>,
  releaseLocal: () => Promise<void>,
): Promise<"stopped" | "local-release"> {
  try {
    await stopRemote();
    return "stopped";
  } catch {
    await releaseLocal();
    return "local-release";
  }
}

export type ApprovalDecision = "approve" | "reject";

type ApprovalSender = (
  sequence: number,
  approvalId: string,
  decision: ApprovalDecision,
) => Promise<void>;

const APPROVAL_ID = /^approval_[A-Za-z0-9_-]{8,128}$/;

export class ApprovalDecisionController {
  private sequence = 0;
  private readonly inFlight = new Set<string>();
  private tail: Promise<unknown> = Promise.resolve();

  constructor(private readonly send: ApprovalSender) {
    if (typeof send !== "function") throw new TypeError("send must be callable");
  }

  get lastSequence(): number {
    return this.sequence;
  }

  get isSubmitting(): boolean {
    return this.inFlight.size > 0;
  }

  isSubmittingApproval(approvalId: string): boolean {
    return this.inFlight.has(approvalId);
  }

  reset(): void {
    this.sequence = 0;
    this.inFlight.clear();
    this.tail = Promise.resolve();
  }

  async submit(approvalId: string, decision: ApprovalDecision): Promise<void> {
    if (!APPROVAL_ID.test(approvalId)) {
      throw new TypeError("approval request identity is invalid");
    }
    if (decision !== "approve" && decision !== "reject") {
      throw new TypeError("approval decision is invalid");
    }
    // One decision per approval: a decision already sent may have been acted on
    // server-side, so a later click must not supersede it.
    if (this.inFlight.has(approvalId)) {
      throw new Error("an approval decision is already in flight");
    }
    this.inFlight.add(approvalId);
    // Concurrent runs can each hold a pending approval, and the server resolves
    // them by exact id in any order. Different approvals may therefore be decided
    // concurrently, but sequence allocation is serialised: the number is computed
    // only once the previous send has settled, so two in-flight decisions cannot
    // claim the same sequence. A failed send still commits nothing and may retry.
    const run = this.tail.then(
      () => this.#dispatch(approvalId, decision),
      () => this.#dispatch(approvalId, decision),
    );
    this.tail = run.catch(() => undefined);
    try {
      await run;
    } finally {
      this.inFlight.delete(approvalId);
    }
  }

  async #dispatch(approvalId: string, decision: ApprovalDecision): Promise<void> {
    const next = this.sequence + 1;
    await this.send(next, approvalId, decision);
    this.sequence = next;
  }
}

export function canCommitCredentialRefresh<Value extends object>(
  activeCredential: Value,
  currentCredential: Value | null,
  state: ClientState,
): boolean {
  return (
    currentCredential === activeCredential &&
    (state === "preparing" ||
      state === "connected" ||
      state === "reconnecting" ||
      state === "disconnected")
  );
}

export function terminalDisconnectIsAuthoritative(
  mediaState: string,
  roomWasAdmitted: boolean,
): boolean {
  return mediaState === "disconnected" && roomWasAdmitted;
}

export function reboundCredential(
  active: BootstrapCredential,
  replacement: BootstrapCredential,
): BootstrapCredential {
  if (replacement.participantIdentity === active.participantIdentity) {
    throw new Error("rebind credential did not rotate participant identity");
  }
  if (
    replacement.workerIdentity !== active.workerIdentity ||
    replacement.roomName !== active.roomName ||
    replacement.url !== active.url
  ) {
    throw new Error("rebind credential changed session authority");
  }
  return replacement;
}

export async function settleMicrophoneMuteChoice<Value>(
  track: Value,
  muted: boolean,
  mute: (track: Value) => Promise<void>,
  unmute: (track: Value) => Promise<void>,
  current: (track: Value) => boolean,
): Promise<boolean> {
  if (typeof muted !== "boolean") throw new TypeError("muted must be a boolean");
  await (muted ? mute(track) : unmute(track));
  return current(track);
}

export function microphoneMuteSettlementIsCurrent<Value>(
  candidate: Value,
  activeTrack: Value | null,
): boolean {
  return candidate === activeTrack;
}

export type PollFailureDisposition = "retry" | "terminal";

export class ForegroundPollFailurePolicy {
  private failures = 0;
  private recoveringFromBackground = false;

  constructor(private readonly failureLimit: number) {
    if (!Number.isSafeInteger(failureLimit) || failureLimit < 1 || failureLimit > 100) {
      throw new TypeError("failureLimit must be a positive bounded safe integer");
    }
  }

  visibilityChanged(state: "hidden" | "visible"): void {
    if (state !== "hidden" && state !== "visible") {
      throw new TypeError("visibility state is invalid");
    }
    if (state === "hidden") {
      this.failures = 0;
      this.recoveringFromBackground = true;
    }
  }

  recordSuccess(): void {
    this.failures = 0;
    this.recoveringFromBackground = false;
  }

  recordFailure(): PollFailureDisposition {
    this.failures += 1;
    const activeLimit = this.recoveringFromBackground
      ? Math.min(100, this.failureLimit * 3)
      : this.failureLimit;
    return this.failures >= activeLimit ? "terminal" : "retry";
  }
}

export async function settleReconnectedMicrophone<Value>(
  existing: Value | null,
  isLive: (value: Value) => boolean,
  create: () => Promise<Value | null>,
  cleanup: (value: Value) => Promise<void>,
): Promise<Value | null> {
  if (existing !== null && isLive(existing)) return existing;
  if (existing !== null) await cleanup(existing);
  return create();
}

export async function enforceMicrophoneMuteAuthority<Value>(
  track: Value,
  shouldMute: boolean,
  apply: (track: Value, shouldMute: boolean) => Promise<void>,
  cleanup: (track: Value) => Promise<void>,
): Promise<Value | null> {
  if (typeof shouldMute !== "boolean") throw new TypeError("shouldMute must be a boolean");
  if (typeof apply !== "function") throw new TypeError("apply must be callable");
  if (typeof cleanup !== "function") throw new TypeError("cleanup must be callable");
  try {
    await apply(track, shouldMute);
    return track;
  } catch {
    await cleanup(track);
    return null;
  }
}

export function canCommitMicrophoneVerification<Value extends object>(
  expectedGeneration: number,
  currentGeneration: number,
  expectedRoom: Value,
  currentRoom: Value | null,
  state: ClientState,
): boolean {
  return (
    Number.isSafeInteger(expectedGeneration) &&
    expectedGeneration >= 0 &&
    expectedGeneration === currentGeneration &&
    expectedRoom === currentRoom &&
    state === "preparing"
  );
}

export function authorizeRemoteAudio(
  expectedWorkerIdentity: string,
  participantIdentity: string,
  microphoneSource: boolean,
  audioTrack: boolean,
): boolean {
  return (
    participantIdentity === expectedWorkerIdentity &&
    microphoneSource === true &&
    audioTrack === true
  );
}

export type SpeechYieldDisposition = "recover" | "silenced" | "superseded";

export function naturalDuplexProfileEnabled(data: Record<string, unknown>): boolean {
  return (
    Object.keys(data).sort().join(",") === "conversationProfile,mode" &&
    data.conversationProfile === "natural_v1" &&
    data.mode === "microphone_or_typed"
  );
}

export interface SpeechAuthority {
  readonly turnId: string;
  readonly turnGeneration: number;
  readonly chunkId: string;
  readonly streamId: string;
}

export function sameSpeechAuthority(
  left: SpeechAuthority,
  right: SpeechAuthority,
): boolean {
  return (
    left.turnId === right.turnId &&
    left.turnGeneration === right.turnGeneration &&
    left.chunkId === right.chunkId &&
    left.streamId === right.streamId
  );
}

export class SpeechYieldController<Authority extends object> {
  #pending: Authority | null = null;

  constructor(
    private readonly matches: (left: Authority, right: Authority) => boolean = Object.is,
  ) {}

  claim(current: Authority | null): Authority | null {
    if (current === null || this.#pending !== null) return null;
    this.#pending = current;
    return current;
  }

  settle(
    claim: Authority,
    current: Authority | null,
    matched: boolean,
  ): SpeechYieldDisposition {
    if (this.#pending === null || !this.matches(this.#pending, claim)) return "superseded";
    this.#pending = null;
    if (matched && current !== null && this.matches(claim, current)) return "silenced";
    return current === null ? "superseded" : "recover";
  }

  reset(): void {
    this.#pending = null;
  }
}

export type ProvisionalSpeechYieldDisposition = SpeechYieldDisposition | "ignored";

export class ProvisionalSpeechYieldController<Authority extends object> {
  #pending: Authority | null = null;
  #attenuated = false;

  constructor(
    private readonly matches: (left: Authority, right: Authority) => boolean = Object.is,
  ) {}

  get pending(): Authority | null {
    return this.#pending;
  }

  get attenuated(): boolean {
    return this.#attenuated;
  }

  begin(current: Authority | null): { claim: Authority; firstOnset: boolean } | null {
    if (current === null) return null;
    if (this.#pending !== null) {
      if (!this.matches(this.#pending, current) || this.#attenuated) return null;
      this.#attenuated = true;
      return { claim: this.#pending, firstOnset: false };
    }
    this.#pending = current;
    this.#attenuated = true;
    return { claim: current, firstOnset: true };
  }

  quiet(current: Authority | null): boolean {
    if (
      this.#pending === null ||
      current === null ||
      !this.matches(this.#pending, current) ||
      !this.#attenuated
    ) {
      return false;
    }
    this.#attenuated = false;
    return true;
  }

  replace(current: Authority): Authority | null {
    const previous = this.#pending;
    if (previous === null) return null;
    this.#pending = current;
    return previous;
  }

  settle(
    authority: Authority,
    current: Authority | null,
    committed: boolean,
  ): ProvisionalSpeechYieldDisposition {
    const claim = this.#pending;
    if (claim === null || !this.matches(claim, authority)) return "ignored";
    this.#pending = null;
    this.#attenuated = false;
    if (committed && current !== null && this.matches(claim, current)) return "silenced";
    return current === null ? "superseded" : "recover";
  }

  expire(): Authority | null {
    const claim = this.#pending;
    this.#pending = null;
    this.#attenuated = false;
    return claim;
  }

  reset(): void {
    this.expire();
  }
}

export function provisionalSpeechYieldAllowed(
  naturalDuplexEnabled: boolean,
  hasRemoteTrack: boolean,
  hasTiming: boolean,
  rendererPaused: boolean,
  exactYieldPending: boolean,
  locallySilenced: boolean,
): boolean {
  return (
    naturalDuplexEnabled &&
    hasRemoteTrack &&
    hasTiming &&
    !rendererPaused &&
    !exactYieldPending &&
    !locallySilenced
  );
}

export function remoteAudioTrackIsCurrent<RoomValue extends object, TrackValue extends object>(
  expectedRoom: RoomValue,
  currentRoom: RoomValue | null,
  expectedTrack: TrackValue,
  currentTrack: TrackValue | null,
): boolean {
  return expectedRoom === currentRoom && expectedTrack === currentTrack;
}

export function releaseRemoteAudio<
  Sink extends { dataset: { streamId?: string }; srcObject: object | null },
  TrackValue extends { detach: (sink: Sink) => unknown },
>(track: TrackValue | null, sink: Sink): null {
  try {
    track?.detach(sink);
  } catch {
    // Clearing the sink remains authoritative when SDK detach cleanup fails.
  }
  sink.srcObject = null;
  delete sink.dataset.streamId;
  return null;
}

export function replaceRemoteAudio<
  Sink extends { dataset: { streamId?: string }; srcObject: object | null },
  TrackValue extends { detach: (sink: Sink) => unknown },
>(current: TrackValue | null, replacement: TrackValue, sink: Sink): TrackValue {
  releaseRemoteAudio(current, sink);
  return replacement;
}

export function releaseRemoteAudioBeforeRoomInvalidation<
  Sink extends { dataset: { streamId?: string }; srcObject: object | null },
  TrackValue extends { detach: (sink: Sink) => unknown },
>(
  track: TrackValue | null,
  sink: Sink,
  stopKaraoke: () => void,
  invalidateRoom: () => void,
): null {
  const released = releaseRemoteAudio(track, sink);
  try {
    stopKaraoke();
  } catch {
    // UI cleanup cannot retain stale room authority.
  }
  invalidateRoom();
  return released;
}

export function microphoneSignalLevel(samples: Float32Array): number {
  if (!(samples instanceof Float32Array)) throw new TypeError("samples must be Float32Array");
  if (samples.length === 0) return 0;
  let energy = 0;
  for (const sample of samples) {
    if (!Number.isFinite(sample)) throw new TypeError("microphone sample must be finite");
    energy += sample * sample;
  }
  return Math.min(1, Math.sqrt(energy / samples.length));
}

export class MicrophoneActivityGate {
  private activeUntilMs = 0;

  constructor(
    private readonly threshold = 0.02,
    private readonly holdMs = 180,
  ) {
    if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
      throw new RangeError("microphone activity threshold is invalid");
    }
    if (!Number.isFinite(holdMs) || holdMs < 0 || holdMs > 5000) {
      throw new RangeError("microphone activity hold is invalid");
    }
  }

  observe(level: number, monotonicMs: number): boolean {
    if (!Number.isFinite(level) || level < 0 || level > 1) {
      throw new RangeError("microphone signal level is invalid");
    }
    if (!Number.isFinite(monotonicMs) || monotonicMs < 0) {
      throw new RangeError("microphone activity timestamp is invalid");
    }
    if (level >= this.threshold) this.activeUntilMs = monotonicMs + this.holdMs;
    return monotonicMs < this.activeUntilMs;
  }

  reset(): void {
    this.activeUntilMs = 0;
  }
}

export function microphoneCaptureOptions(selectedDeviceId: string): {
  deviceId?: { exact: string };
  autoGainControl: false;
  echoCancellation: { exact: true };
  noiseSuppression: false;
  voiceIsolation: false;
} {
  return {
    ...(selectedDeviceId === "" ? {} : { deviceId: { exact: selectedDeviceId } }),
    autoGainControl: false,
    echoCancellation: { exact: true },
    noiseSuppression: false,
    voiceIsolation: false,
  };
}

export function microphoneProcessingDisposition(
  settings: Pick<
    MediaTrackSettings,
    "echoCancellation" | "autoGainControl" | "noiseSuppression"
  > & { voiceIsolation?: boolean },
): "aec-only" | "aec-unavailable" | "additional-processing" {
  if (settings.echoCancellation !== true) return "aec-unavailable";
  if (
    settings.autoGainControl === true ||
    settings.noiseSuppression === true ||
    settings.voiceIsolation === true
  ) {
    return "additional-processing";
  }
  return "aec-only";
}

export type BrowserProcessorState = "enabled" | "disabled" | "unknown";

export interface MicrophoneProcessingTelemetry {
  readonly version: 1;
  readonly supported: {
    readonly echoCancellation: boolean;
    readonly autoGainControl: boolean;
    readonly noiseSuppression: boolean;
    readonly voiceIsolation: boolean;
  };
  readonly applied: {
    readonly echoCancellation: BrowserProcessorState;
    readonly autoGainControl: BrowserProcessorState;
    readonly noiseSuppression: BrowserProcessorState;
    readonly voiceIsolation: BrowserProcessorState;
  };
}

function processorState(value: unknown): BrowserProcessorState {
  return value === true ? "enabled" : value === false ? "disabled" : "unknown";
}

export function microphoneProcessingTelemetry(
  supported: Pick<
    MediaTrackSupportedConstraints,
    "echoCancellation" | "autoGainControl" | "noiseSuppression"
  > & { voiceIsolation?: boolean },
  settings: Pick<
    MediaTrackSettings,
    "echoCancellation" | "autoGainControl" | "noiseSuppression"
  > & { voiceIsolation?: boolean },
): MicrophoneProcessingTelemetry {
  return {
    version: 1,
    supported: {
      echoCancellation: supported.echoCancellation === true,
      autoGainControl: supported.autoGainControl === true,
      noiseSuppression: supported.noiseSuppression === true,
      voiceIsolation: supported.voiceIsolation === true,
    },
    applied: {
      echoCancellation: processorState(settings.echoCancellation),
      autoGainControl: processorState(settings.autoGainControl),
      noiseSuppression: processorState(settings.noiseSuppression),
      voiceIsolation: processorState(settings.voiceIsolation),
    },
  };
}

export interface RenderTimingRecord {
  readonly streamId: string;
  readonly subscribedToAttachMs: number;
  readonly attachToPlayingMs: number;
  readonly playingToAdvanceMs: number;
}

const AUDIO_STREAM_ID = /^[A-Za-z0-9_-]{8,128}$/;
const MAX_RENDER_STAGE_MS = 120_000;

export class RenderTimingTelemetry {
  private attachedAt: number | null = null;
  private playingAt: number | null = null;
  private emitted = false;

  constructor(
    private readonly streamId: string,
    private readonly subscribedAt: number,
  ) {
    if (!AUDIO_STREAM_ID.test(streamId)) throw new TypeError("audio stream identity is invalid");
    this.requireTimestamp(subscribedAt);
  }

  attached(monotonicMs: number): void {
    this.requireOrdered(monotonicMs, this.subscribedAt);
    this.attachedAt = monotonicMs;
  }

  playing(monotonicMs: number): void {
    if (this.attachedAt === null) throw new Error("renderer attach timing is unavailable");
    this.requireOrdered(monotonicMs, this.attachedAt);
    this.playingAt = monotonicMs;
  }

  advanced(monotonicMs: number): RenderTimingRecord | null {
    if (this.emitted) return null;
    if (this.attachedAt === null || this.playingAt === null) {
      throw new Error("renderer playing timing is unavailable");
    }
    this.requireTimestamp(monotonicMs);
    const record = {
      streamId: this.streamId,
      subscribedToAttachMs: this.boundedDelta(this.attachedAt, this.subscribedAt),
      attachToPlayingMs: this.boundedDelta(this.playingAt, this.attachedAt),
      playingToAdvanceMs: this.boundedDelta(Math.max(monotonicMs, this.playingAt), this.playingAt),
    };
    this.emitted = true;
    return record;
  }

  private requireTimestamp(value: number): void {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError("renderer monotonic timestamp is invalid");
    }
  }

  private requireOrdered(value: number, previous: number): void {
    this.requireTimestamp(value);
    if (value < previous) throw new RangeError("renderer monotonic clock regressed");
  }

  private boundedDelta(value: number, previous: number): number {
    const delta = Math.round(value - previous);
    if (!Number.isSafeInteger(delta) || delta < 0 || delta > MAX_RENDER_STAGE_MS) {
      throw new RangeError("renderer timing delta is outside the supported range");
    }
    return delta;
  }
}

export class GenerationAuthority {
  private current: object = Object.freeze({});

  issue(): object {
    const token = Object.freeze({});
    this.current = token;
    return token;
  }

  owns(candidate: object): boolean {
    return candidate === this.current;
  }

  invalidate(): void {
    this.current = Object.freeze({});
  }
}

export class MicrophoneReadinessAuthority<Room extends object, Microphone extends object> {
  #serverGeneration: number | null = null;
  #mediaIncarnation: number | null = null;
  #room: Room | null = null;
  #microphone: Microphone | null = null;

  activate(room: Room, microphone: Microphone, mediaIncarnation: number): void {
    if (!Number.isSafeInteger(mediaIncarnation) || mediaIncarnation <= 0) {
      throw new RangeError("media incarnation must be a positive safe integer");
    }
    this.#room = room;
    this.#microphone = microphone;
    this.#mediaIncarnation = mediaIncarnation;
  }

  invalidate(): void {
    this.#room = null;
    this.#microphone = null;
    this.#mediaIncarnation = null;
  }

  resetSession(): void {
    this.invalidate();
    this.#serverGeneration = null;
  }

  accepts(
    generation: number,
    mediaIncarnation: number,
    room: Room | null,
    microphone: Microphone | null,
    state: ClientState,
  ): boolean {
    if (
      !Number.isSafeInteger(generation) ||
      generation <= 0 ||
      !Number.isSafeInteger(mediaIncarnation) ||
      mediaIncarnation <= 0 ||
      mediaIncarnation !== this.#mediaIncarnation ||
      room === null ||
      microphone === null ||
      room !== this.#room ||
      microphone !== this.#microphone ||
      state !== "connected"
    ) {
      return false;
    }
    if (this.#serverGeneration !== null && generation < this.#serverGeneration) return false;
    this.#serverGeneration = generation;
    return true;
  }
}

export class SerializedAsyncQueue {
  private tail: Promise<void> = Promise.resolve();

  run<Value>(operation: () => Promise<Value>): Promise<Value> {
    const predecessor = this.tail;
    let release: () => void = () => undefined;
    this.tail = new Promise<void>((resolve) => {
      release = resolve;
    });
    return predecessor.then(operation).finally(release);
  }
}

export async function settleMediaActivation(
  activation: Promise<void>,
  current: () => boolean,
  cleanup: () => Promise<void>,
): Promise<boolean> {
  try {
    await activation;
  } catch (error) {
    await cleanup();
    throw error;
  }
  if (current()) return true;
  await cleanup();
  return false;
}

export async function settleMicrophoneVerification<Value>(
  current: boolean,
  value: Value,
  cleanup: (value: Value) => Promise<void>,
): Promise<{ readonly committed: boolean; readonly value: Value }> {
  if (!current) await cleanup(value);
  return { committed: current, value };
}

export interface MicrophoneDeviceChoice {
  readonly deviceId: string;
  readonly label: string;
}

export function microphoneDeviceChoices(
  devices: readonly Pick<MediaDeviceInfo, "kind" | "deviceId" | "label">[],
  limit = 128,
): readonly MicrophoneDeviceChoice[] {
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 256) {
    throw new RangeError("microphone device limit is invalid");
  }
  const choices: MicrophoneDeviceChoice[] = [];
  const seen = new Set<string>();
  for (const device of devices) {
    if (
      device.kind !== "audioinput" ||
      typeof device.deviceId !== "string" ||
      device.deviceId.length < 1 ||
      device.deviceId.length > 512 ||
      seen.has(device.deviceId)
    ) {
      continue;
    }
    seen.add(device.deviceId);
    const label =
      typeof device.label === "string" && device.label.length > 0
        ? device.label.slice(0, 256)
        : `Microphone ${choices.length + 1}`;
    choices.push({ deviceId: device.deviceId, label });
    if (choices.length === limit) break;
  }
  return choices;
}

export class BoundedRetention<Value> {
  private readonly entries: Array<{ readonly value: Value; cost: number }> = [];
  private totalCost = 0;

  constructor(
    private readonly maxItems: number,
    private readonly maxCost: number,
  ) {
    if (!Number.isSafeInteger(maxItems) || maxItems < 1) {
      throw new RangeError("maxItems must be a positive safe integer");
    }
    if (!Number.isSafeInteger(maxCost) || maxCost < 1) {
      throw new RangeError("maxCost must be a positive safe integer");
    }
  }

  get size(): number {
    return this.entries.length;
  }

  get cost(): number {
    return this.totalCost;
  }

  admit(value: Value, cost: number): Value[] {
    if (!Number.isSafeInteger(cost) || cost < 0 || cost > this.maxCost) {
      throw new RangeError("retained cost is invalid");
    }
    this.entries.push({ value, cost });
    this.totalCost += cost;
    const evicted: Value[] = [];
    while (this.entries.length > this.maxItems || this.totalCost > this.maxCost) {
      const oldest = this.entries.shift();
      if (oldest === undefined) throw new Error("retention accounting is inconsistent");
      this.totalCost -= oldest.cost;
      evicted.push(oldest.value);
    }
    return evicted;
  }

  update(value: Value, cost: number): Value[] {
    if (!Number.isSafeInteger(cost) || cost < 0) {
      throw new RangeError("retained cost is invalid");
    }
    const index = this.entries.findIndex((entry) => Object.is(entry.value, value));
    if (index < 0) throw new Error("retained value is not admitted");
    const entry = this.entries[index];
    if (entry === undefined) throw new Error("retention accounting is inconsistent");
    if (cost > this.maxCost) {
      this.entries.splice(index, 1);
      this.totalCost -= entry.cost;
      return [value];
    }
    this.totalCost += cost - entry.cost;
    entry.cost = cost;
    const evicted: Value[] = [];
    while (this.entries.length > this.maxItems || this.totalCost > this.maxCost) {
      const oldest = this.entries.shift();
      if (oldest === undefined) throw new Error("retention accounting is inconsistent");
      this.totalCost -= oldest.cost;
      evicted.push(oldest.value);
    }
    return evicted;
  }

  remove(value: Value): boolean {
    const index = this.entries.findIndex((entry) => Object.is(entry.value, value));
    if (index < 0) return false;
    const [entry] = this.entries.splice(index, 1);
    if (entry === undefined) throw new Error("retention accounting is inconsistent");
    this.totalCost -= entry.cost;
    return true;
  }
}

export class ConnectionAttempt<Resource extends object> {
  private cancelled = false;

  constructor(private readonly disconnect: (resource: Resource) => void) {
    if (typeof disconnect !== "function") {
      throw new TypeError("disconnect must be callable");
    }
  }

  cancel(): void {
    this.cancelled = true;
  }

  owns(current: ConnectionAttempt<Resource> | null): boolean {
    return !this.cancelled && current === this;
  }

  admit(resource: Resource, current: Resource | null): boolean {
    if (this.cancelled || current !== resource) {
      this.disconnect(resource);
      return false;
    }
    return true;
  }
}

export interface ObjectiveLatency {
  readonly name:
    | "speech_end_to_transcript"
    | "transcript_to_first_token"
    | "first_token_to_audio"
    | "interrupt_to_silence"
    | "completion_to_notification";
  readonly durationMs: number;
}

export interface ResponseLatencySnapshot {
  readonly lastMs: number | null;
  readonly averageMs: number | null;
  readonly sampleCount: number;
}

export class ResponseLatencyStatistics {
  private lastMs: number | null = null;
  private averageMs: number | null = null;
  private sampleCount = 0;

  get snapshot(): ResponseLatencySnapshot {
    return {
      lastMs: this.lastMs,
      averageMs: this.averageMs,
      sampleCount: this.sampleCount,
    };
  }

  observe(latency: ObjectiveLatency): ResponseLatencySnapshot {
    if (!Number.isFinite(latency.durationMs) || latency.durationMs < 0) {
      throw new TypeError("latency duration must be finite and non-negative");
    }
    if (latency.name !== "transcript_to_first_token") return this.snapshot;
    if (this.sampleCount >= Number.MAX_SAFE_INTEGER) {
      throw new Error("latency sample counter exhausted");
    }
    this.sampleCount += 1;
    this.lastMs = latency.durationMs;
    this.averageMs =
      this.averageMs === null
        ? latency.durationMs
        : this.averageMs + (latency.durationMs - this.averageMs) / this.sampleCount;
    return this.snapshot;
  }
}

export function formatLatencyDuration(durationMs: number | null): string {
  if (durationMs === null) return "Waiting for first response";
  if (!Number.isFinite(durationMs) || durationMs < 0) {
    throw new TypeError("latency duration must be finite and non-negative");
  }
  return durationMs < 1000 ? `${Math.round(durationMs)} ms` : `${(durationMs / 1000).toFixed(2)} s`;
}

export class ObjectiveLatencyTracker {
  private lastObservedMs = -1;
  private speechEndedMs: number | null = null;
  private transcriptMs: number | null = null;
  private firstTokenMs: number | null = null;
  private interruptMs: number | null = null;
  private completionMs: number | null = null;

  observe(kind: string, monotonicMs: number, role?: string): ObjectiveLatency | null {
    if (typeof kind !== "string" || kind.length === 0) {
      throw new TypeError("marker kind must be a non-empty string");
    }
    if (!Number.isFinite(monotonicMs) || monotonicMs < 0) {
      throw new TypeError("marker time must be finite and non-negative");
    }
    if (monotonicMs < this.lastObservedMs) {
      throw new Error("authoritative marker time regressed");
    }
    this.lastObservedMs = monotonicMs;

    if (kind === "speech_ended") {
      this.speechEndedMs = monotonicMs;
      return null;
    }
    if (kind === "transcript_final" && role === "user") {
      const latency = this.delta(
        "speech_end_to_transcript",
        this.speechEndedMs,
        monotonicMs,
      );
      this.transcriptMs = monotonicMs;
      return latency;
    }
    if (kind === "first_foreground_token") {
      const latency = this.delta(
        "transcript_to_first_token",
        this.transcriptMs,
        monotonicMs,
      );
      this.firstTokenMs = monotonicMs;
      return latency;
    }
    if (kind === "first_playable_audio") {
      return this.delta("first_token_to_audio", this.firstTokenMs, monotonicMs);
    }
    if (kind === "interrupt_requested") {
      this.interruptMs = monotonicMs;
      return null;
    }
    if (kind === "playback_silenced") {
      return this.delta("interrupt_to_silence", this.interruptMs, monotonicMs);
    }
    if (kind === "completion_received") {
      this.completionMs = monotonicMs;
      return null;
    }
    if (kind === "notification_queued") {
      return this.delta(
        "completion_to_notification",
        this.completionMs,
        monotonicMs,
      );
    }
    return null;
  }

  private delta(
    name: ObjectiveLatency["name"],
    startedMs: number | null,
    finishedMs: number,
  ): ObjectiveLatency | null {
    if (startedMs === null) return null;
    return { name, durationMs: finishedMs - startedMs };
  }
}

export type UserTurnState = "open" | "closed";
export type UserTranscriptProjectionSource = "typed-admission" | "authoritative-event";
export type UserTranscriptProjectionDisposition = "await-authoritative" | "project";

export function userTranscriptProjectionDisposition(
  source: UserTranscriptProjectionSource,
): UserTranscriptProjectionDisposition {
  return source === "authoritative-event" ? "project" : "await-authoritative";
}

export interface UserTurnSnapshot {
  readonly text: string;
  readonly state: UserTurnState;
}

export class UserTurnAssembler {
  private readonly segments: string[] = [];
  private turnState: UserTurnState = "open";

  get snapshot(): UserTurnSnapshot {
    return { text: this.segments.join(" "), state: this.turnState };
  }

  admit(text: string): UserTurnSnapshot {
    if (this.turnState !== "open") throw new Error("user turn is closed");
    if (typeof text !== "string" || text.trim().length < 1 || text.length > 4096) {
      throw new TypeError("user transcript text is invalid");
    }
    this.segments.push(text.trim());
    return this.snapshot;
  }

  close(): UserTurnSnapshot {
    this.turnState = "closed";
    return this.snapshot;
  }
}

export type AssistantGenerationDisposition = "create" | "current" | "replace" | "reject";

export function assistantGenerationDisposition(
  currentGeneration: number | undefined,
  candidateGeneration: number,
): AssistantGenerationDisposition {
  if (!Number.isSafeInteger(candidateGeneration) || candidateGeneration < 1) {
    throw new TypeError("assistant generation is invalid");
  }
  if (currentGeneration === undefined) return "create";
  if (!Number.isSafeInteger(currentGeneration) || currentGeneration < 1) {
    throw new TypeError("current assistant generation is invalid");
  }
  if (candidateGeneration === currentGeneration) return "current";
  return candidateGeneration > currentGeneration ? "replace" : "reject";
}

export class AssistantGenerationAuthority {
  private readonly highWatermarks = new Map<string, number>();

  constructor(private readonly maxTurns = 256) {
    if (!Number.isSafeInteger(maxTurns) || maxTurns < 1 || maxTurns > 4096) {
      throw new RangeError("assistant generation authority capacity is invalid");
    }
  }

  disposition(turnId: string, candidateGeneration: number): AssistantGenerationDisposition {
    if (typeof turnId !== "string" || turnId.length < 1 || turnId.length > 128) {
      throw new TypeError("turn identity is invalid");
    }
    const disposition = assistantGenerationDisposition(
      this.highWatermarks.get(turnId),
      candidateGeneration,
    );
    if (disposition === "create" || disposition === "replace") {
      this.highWatermarks.delete(turnId);
      this.highWatermarks.set(turnId, candidateGeneration);
      while (this.highWatermarks.size > this.maxTurns) {
        const oldest = this.highWatermarks.keys().next().value as string | undefined;
        if (oldest === undefined) break;
        this.highWatermarks.delete(oldest);
      }
    } else if (disposition === "current") {
      // Map insertion order is the LRU key. Without refreshing it here an
      // actively streaming turn ages out as the "oldest" entry, after which a
      // replayed lower generation reads as "create" instead of "reject".
      this.highWatermarks.delete(turnId);
      this.highWatermarks.set(turnId, candidateGeneration);
    }
    return disposition;
  }

  clear(): void {
    this.highWatermarks.clear();
  }
}

export type AssistantTurnState = "live" | "complete" | "interrupted";

export interface AssistantTurnSnapshot {
  readonly turnId: string;
  readonly turnGeneration: number;
  readonly text: string;
  readonly chunkIds: readonly string[];
  readonly deliveredChunkIds: readonly string[];
  readonly unspokenChunkIds: readonly string[];
  readonly state: AssistantTurnState;
}

export class AssistantTurnAssembler {
  private readonly segments = new Map<string, { text: string; delivered: boolean }>();
  private turnState: AssistantTurnState = "live";

  constructor(
    readonly turnId: string,
    readonly turnGeneration: number,
  ) {
    if (typeof turnId !== "string" || turnId.length < 1 || turnId.length > 128) {
      throw new TypeError("turn identity is invalid");
    }
    if (!Number.isSafeInteger(turnGeneration) || turnGeneration < 1) {
      throw new TypeError("turn generation is invalid");
    }
  }

  get snapshot(): AssistantTurnSnapshot {
    const entries = [...this.segments.entries()];
    return {
      turnId: this.turnId,
      turnGeneration: this.turnGeneration,
      text: entries.map(([, segment]) => segment.text).join(" "),
      chunkIds: entries.map(([chunkId]) => chunkId),
      deliveredChunkIds: entries
        .filter(([, segment]) => segment.delivered)
        .map(([chunkId]) => chunkId),
      unspokenChunkIds: entries
        .filter(([, segment]) => !segment.delivered)
        .map(([chunkId]) => chunkId),
      state: this.turnState,
    };
  }

  admit(chunkId: string, text: string): AssistantTurnSnapshot {
    const admitted = this.tryAdmit(chunkId, text);
    if (admitted === null) throw new Error("assistant turn is no longer live");
    return admitted;
  }

  tryAdmit(chunkId: string, text: string): AssistantTurnSnapshot | null {
    this.validateSegment(chunkId, text);
    const existing = this.segments.get(chunkId);
    if (existing !== undefined) {
      if (existing.text !== text) throw new Error("conflicting assistant chunk content");
      return this.snapshot;
    }
    if (this.turnState !== "live") return null;
    this.segments.set(chunkId, { text, delivered: false });
    return this.snapshot;
  }

  confirm(chunkId: string): AssistantTurnSnapshot {
    if (typeof chunkId !== "string" || chunkId.length < 1 || chunkId.length > 128) {
      throw new TypeError("chunk identity is invalid");
    }
    const segment = this.segments.get(chunkId);
    if (segment === undefined) throw new Error("assistant chunk was not admitted");
    segment.delivered = true;
    return this.snapshot;
  }

  complete(): AssistantTurnSnapshot {
    if (this.turnState === "interrupted") return this.snapshot;
    this.turnState = "complete";
    return this.snapshot;
  }

  interrupt(): AssistantTurnSnapshot {
    if (this.turnState === "live") this.turnState = "interrupted";
    return this.snapshot;
  }

  private validateSegment(chunkId: string, text: string): void {
    if (typeof chunkId !== "string" || chunkId.length < 1 || chunkId.length > 128) {
      throw new TypeError("chunk identity is invalid");
    }
    if (typeof text !== "string" || text.trim().length < 1 || text.length > 4096) {
      throw new TypeError("assistant chunk text is invalid");
    }
  }
}

export interface SpeechWordTiming {
  readonly textStart: number;
  readonly textEnd: number;
  readonly startSample: number;
  readonly endSample: number;
}

export interface SpeechTiming {
  readonly turnId: string;
  readonly presentationTurnId: string;
  readonly turnGeneration: number;
  readonly chunkId: string;
  readonly segmentId: string;
  readonly streamId: string;
  readonly sampleRate: number;
  readonly timingSource: "provider" | "estimated";
  readonly words: readonly SpeechWordTiming[];
}

interface MediaStartEntry {
  attachmentTime: number;
  startTime?: number;
}

// Stream attachments and per-passage rebases are two different identifier
// namespaces. Sharing one map let a chunkId collide with a streamId and let a
// long turn's passages evict the stream attachment that resolvePlaying/observe
// still needed, so each namespace gets its own map and its own eviction budget.
export class SpeechMediaStartRegistry {
  readonly #streams = new Map<string, MediaStartEntry>();
  readonly #passages = new Map<string, MediaStartEntry>();

  constructor(private readonly maxEntries = 32) {
    if (!Number.isSafeInteger(maxEntries) || maxEntries < 1 || maxEntries > 4096) {
      throw new RangeError("speech media-start capacity is invalid");
    }
  }

  attach(streamId: string, mediaTimeSeconds: number): void {
    this.#validate(streamId, mediaTimeSeconds);
    if (this.#streams.has(streamId)) return;
    this.#streams.set(streamId, { attachmentTime: mediaTimeSeconds });
    this.#bound(this.#streams);
  }

  beginPassage(passageId: string, mediaTimeSeconds: number): void {
    this.#validate(passageId, mediaTimeSeconds);
    this.#passages.delete(passageId);
    this.#passages.set(passageId, { attachmentTime: mediaTimeSeconds });
    this.#bound(this.#passages);
  }

  #bound(entries: Map<string, MediaStartEntry>): void {
    while (entries.size > this.maxEntries) {
      const oldest = entries.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      entries.delete(oldest);
    }
  }

  observeStream(streamId: string, mediaTimeSeconds: number): number | undefined {
    return this.#observe(this.#streams, streamId, mediaTimeSeconds);
  }

  observePassage(passageId: string, mediaTimeSeconds: number): number | undefined {
    return this.#observe(this.#passages, passageId, mediaTimeSeconds);
  }

  #observe(
    entries: Map<string, MediaStartEntry>,
    id: string,
    mediaTimeSeconds: number,
  ): number | undefined {
    this.#validate(id, mediaTimeSeconds);
    const entry = entries.get(id);
    if (entry === undefined) return undefined;
    if (entry.startTime !== undefined) return entry.startTime;
    if (mediaTimeSeconds === entry.attachmentTime) return undefined;
    entry.startTime = mediaTimeSeconds;
    return entry.startTime;
  }

  resolvePlaying(streamId: string, mediaTimeSeconds: number): number | undefined {
    this.#validate(streamId, mediaTimeSeconds);
    const entry = this.#streams.get(streamId);
    if (entry === undefined) return undefined;
    entry.startTime ??= mediaTimeSeconds;
    return entry.startTime;
  }

  getPassage(passageId: string): number | undefined {
    return this.#passages.get(passageId)?.startTime;
  }

  #validate(streamId: string, mediaTimeSeconds: number): void {
    if (typeof streamId !== "string" || streamId.length < 1 || streamId.length > 128) {
      throw new TypeError("speech stream identity is invalid");
    }
    if (!Number.isFinite(mediaTimeSeconds) || mediaTimeSeconds < 0) {
      throw new TypeError("speech media time is invalid");
    }
  }
}

// A NUL separator cannot appear in any component identifier, so composite keys
// stay unambiguous. streamId is part of the key: resumed or replayed speech
// re-admits the same turn/generation/chunk under a new stream, and without it
// that re-admission rewrites the entry #latestByStream still points at for the
// previous stream, so get(previousStreamId) would return the new stream's timing.
const SPEECH_TIMING_KEY_SEPARATOR = String.fromCharCode(0);

export class SpeechTimingRegistry {
  readonly #entries = new Map<string, SpeechTiming>();
  readonly #latestByStream = new Map<string, string>();

  constructor(private readonly maxEntries = 32) {
    if (!Number.isSafeInteger(maxEntries) || maxEntries < 1 || maxEntries > 4096) {
      throw new RangeError("speech timing capacity is invalid");
    }
  }

  admit(timing: SpeechTiming): void {
    const key = this.#key(timing);
    if (this.#entries.has(key)) this.#entries.delete(key);
    this.#entries.set(key, timing);
    this.#latestByStream.set(timing.streamId, key);
    while (this.#entries.size > this.maxEntries) {
      const oldest = this.#entries.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      const evicted = this.#entries.get(oldest);
      this.#entries.delete(oldest);
      if (evicted !== undefined && this.#latestByStream.get(evicted.streamId) === oldest) {
        const replacement = [...this.#entries.entries()]
          .reverse()
          .find(([, candidate]) => candidate.streamId === evicted.streamId);
        if (replacement === undefined) this.#latestByStream.delete(evicted.streamId);
        else this.#latestByStream.set(evicted.streamId, replacement[0]);
      }
    }
  }

  get(streamId: string): SpeechTiming | undefined {
    const key = this.#latestByStream.get(streamId);
    return key === undefined ? undefined : this.#entries.get(key);
  }

  forSegment(turnId: string, turnGeneration: number, chunkId: string): SpeechTiming[] {
    return [...this.#entries.values()].filter(
      (timing) =>
        timing.presentationTurnId === turnId &&
        timing.turnGeneration === turnGeneration &&
        timing.segmentId === chunkId,
    );
  }

  #key(timing: SpeechTiming): string {
    return [timing.turnId, timing.turnGeneration, timing.chunkId, timing.streamId].join(
      SPEECH_TIMING_KEY_SEPARATOR,
    );
  }
}

const SPEECH_TIMING_KEYS = new Set([
  "turnId",
  "presentationTurnId",
  "turnGeneration",
  "chunkId",
  "segmentId",
  "streamId",
  "sampleRate",
  "timingSource",
  "timings",
]);

export function parseSpeechTiming(data: Readonly<Record<string, unknown>>): SpeechTiming | null {
  if (Object.keys(data).length !== SPEECH_TIMING_KEYS.size) return null;
  if (Object.keys(data).some((key) => !SPEECH_TIMING_KEYS.has(key))) return null;
  const {
    turnId,
    presentationTurnId,
    turnGeneration,
    chunkId,
    segmentId,
    streamId,
    sampleRate,
    timingSource,
    timings,
  } = data;
  if (
    typeof turnId !== "string" ||
    turnId.length < 1 ||
    turnId.length > 128 ||
    typeof presentationTurnId !== "string" ||
    presentationTurnId.length < 1 ||
    presentationTurnId.length > 128 ||
    !Number.isSafeInteger(turnGeneration) ||
    (turnGeneration as number) < 1 ||
    typeof chunkId !== "string" ||
    chunkId.length < 1 ||
    chunkId.length > 128 ||
    typeof segmentId !== "string" ||
    segmentId.length < 1 ||
    segmentId.length > 128 ||
    typeof streamId !== "string" ||
    streamId.length < 1 ||
    streamId.length > 128 ||
    !Number.isSafeInteger(sampleRate) ||
    (sampleRate as number) < 8000 ||
    (sampleRate as number) > 192000 ||
    (timingSource !== "provider" && timingSource !== "estimated") ||
    typeof timings !== "string" ||
    timings.length < 1 ||
    timings.length > 4096
  ) {
    return null;
  }

  const words: SpeechWordTiming[] = [];
  let previousTextEnd = 0;
  let previousSampleEnd = 0;
  for (const encoded of timings.split(";")) {
    const parts = encoded.split(",");
    if (parts.length !== 4) return null;
    const values = parts.map((part) => {
      if (!/^(0|[1-9][0-9]*)$/.test(part)) return null;
      const value = Number(part);
      return Number.isSafeInteger(value) ? value : null;
    });
    if (values.some((value) => value === null)) return null;
    const [textStart, textEnd, startSample, endSample] = values as [number, number, number, number];
    if (
      textStart < previousTextEnd ||
      textStart >= textEnd ||
      startSample < previousSampleEnd ||
      startSample >= endSample
    ) {
      return null;
    }
    words.push({ textStart, textEnd, startSample, endSample });
    previousTextEnd = textEnd;
    previousSampleEnd = endSample;
  }
  if (words.length < 1 || words.length > 4096) return null;
  return {
    turnId,
    presentationTurnId,
    turnGeneration: turnGeneration as number,
    chunkId,
    segmentId,
    streamId,
    sampleRate: sampleRate as number,
    timingSource,
    words,
  };
}

export class SpeechKaraokeClock {
  private lastWordIndex = 0;
  private lastMediaTimeSeconds: number;

  constructor(
    private readonly timing: SpeechTiming,
    private mediaStartSeconds: number,
  ) {
    if (!Number.isFinite(mediaStartSeconds) || mediaStartSeconds < 0) {
      throw new TypeError("media start time is invalid");
    }
    this.lastMediaTimeSeconds = mediaStartSeconds;
  }

  wordIndexAt(mediaTimeSeconds: number): number {
    if (!Number.isFinite(mediaTimeSeconds) || mediaTimeSeconds < 0) {
      throw new TypeError("media time is invalid");
    }
    if (mediaTimeSeconds < this.lastMediaTimeSeconds) {
      this.mediaStartSeconds = mediaTimeSeconds;
    }
    this.lastMediaTimeSeconds = mediaTimeSeconds;
    const elapsedSamples = Math.max(
      0,
      Math.floor((mediaTimeSeconds - this.mediaStartSeconds) * this.timing.sampleRate),
    );
    const words = this.timing.words;
    if (elapsedSamples >= words[words.length - 1]!.endSample) return -1;
    const current = words[this.lastWordIndex];
    if (current !== undefined && elapsedSamples >= current.startSample && elapsedSamples < current.endSample) {
      return this.lastWordIndex;
    }
    let low = 0;
    let high = words.length - 1;
    while (low <= high) {
      const middle = (low + high) >>> 1;
      const word = words[middle]!;
      if (elapsedSamples < word.startSample) high = middle - 1;
      else if (elapsedSamples >= word.endSample) low = middle + 1;
      else {
        this.lastWordIndex = middle;
        return middle;
      }
    }
    return -2;
  }
}

export class ClientController {
  readonly #listeners = new Set<(state: ClientState) => void>();
  #state: ClientState = "idle";

  get state(): ClientState {
    return this.#state;
  }

  subscribe(listener: (state: ClientState) => void): () => void {
    this.#listeners.add(listener);
    listener(this.#state);
    return () => this.#listeners.delete(listener);
  }

  beginConnect(): void {
    if (!new Set<ClientState>(["idle", "disconnected", "stopped", "error"]).has(this.#state)) {
      throw new Error("cannot connect from the current state");
    }
    this.#setState("bootstrapping");
  }

  bootstrapReady(): void {
    this.#transition("bootstrapping", "connecting", "bootstrap is not active");
  }

  preparingMicrophone(): void {
    if (this.#state !== "connecting" && this.#state !== "reconnecting") {
      throw new Error("media connection is not pending");
    }
    this.#setState("preparing");
  }

  connected(): void {
    if (this.#state === "connected") return;
    this.#transition("preparing", "connected", "microphone preparation is not complete");
  }

  reconnecting(): void {
    if (this.#state !== "connected" && this.#state !== "preparing") {
      throw new Error("client is not connected");
    }
    this.#setState("reconnecting");
  }

  beginReconnect(): void {
    this.#transition("disconnected", "reconnecting", "client is not disconnected");
  }

  disconnected(): void {
    if (this.#state === "disconnected") return;
    if (
      !new Set<ClientState>(["connecting", "preparing", "connected", "reconnecting"]).has(
        this.#state,
      )
    ) {
      throw new Error("client has no active connection");
    }
    this.#setState("disconnected");
  }

  beginStop(): void {
    if (
      !new Set<ClientState>([
        "connecting",
        "preparing",
        "connected",
        "reconnecting",
        "disconnected",
      ]).has(this.#state)
    ) {
      throw new Error("client cannot stop from the current state");
    }
    this.#setState("stopping");
  }

  stopped(): void {
    this.#transition("stopping", "stopped", "client is not stopping");
  }

  stopFailed(mediaConnected: boolean): void {
    if (typeof mediaConnected !== "boolean") {
      throw new TypeError("mediaConnected must be a boolean");
    }
    this.#transition(
      "stopping",
      mediaConnected ? "connected" : "disconnected",
      "client is not stopping",
    );
  }

  failed(): void {
    this.#setState("error");
  }

  #transition(expected: ClientState, next: ClientState, message: string): void {
    if (this.#state !== expected) {
      throw new Error(message);
    }
    this.#setState(next);
  }

  #setState(next: ClientState): void {
    this.#state = next;
    for (const listener of this.#listeners) {
      listener(next);
    }
  }
}
