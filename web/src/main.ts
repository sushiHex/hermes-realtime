import {
  ConnectionState,
  RemoteAudioTrack,
  Room,
  RoomEvent,
  Track,
  type LocalTrack,
  type RemoteParticipant,
  type RemoteTrack,
  type RemoteTrackPublication,
} from "livekit-client";

import {
  ApprovalDecisionController,
  AssistantGenerationAuthority,
  AssistantTurnAssembler,
  authorizeRemoteAudio,
  bootstrapRequestParameters,
  BoundedRetention,
  canCommitCredentialRefresh,
  canCommitMicrophoneVerification,
  ClientController,
  connectionPresentation,
  ConnectionAttempt,
  enforceMicrophoneMuteAuthority,
  ForegroundPollFailurePolicy,
  GenerationAuthority,
  formatSessionModelConfiguration,
  formatSessionTokenUsage,
  microphoneCaptureOptions,
  microphoneProcessingTelemetry,
  MicrophoneActivityGate,
  microphoneDeviceChoices,
  microphoneMuteSettlementIsCurrent,
  microphoneProcessingDisposition,
  MicrophoneReadinessAuthority,
  microphoneSignalLevel,
  naturalDuplexProfileEnabled,
  markdownEmphasisRuns,
  karaokeGapPendingStates,
  karaokeSegmentPendingStates,
  modelSelectionFor,
  reasoningEffortLabel,
  reboundCredential,
  rebindFailureAllowsFreshBootstrap,
  rebindRequestParameters,
  ResponseLatencyStatistics,
  RenderTimingTelemetry,
  formatLatencyDuration,
  ObjectiveLatencyTracker,
  parseSpeechTiming,
  partialTranscriptDisposition,
  provisionalSpeechYieldAllowed,
  ProvisionalSpeechYieldController,
  releaseRemoteAudio,
  releaseRemoteAudioBeforeRoomInvalidation,
  replaceRemoteAudio,
  remoteAudioTrackIsCurrent,
  SpeechKaraokeClock,
  SpeechMediaStartRegistry,
  SpeechYieldController,
  SpeechTimingRegistry,
  SerializedAsyncQueue,
  settleMediaActivation,
  settleMicrophoneMuteChoice,
  settleMicrophoneVerification,
  settleReconnectedMicrophone,
  sameSpeechAuthority,
  sessionTogglePresentation,
  takeOptionalBootstrapCapability,
  terminalDisconnectIsAuthoritative,
  taskMetaPresentation,
  taskStatusPresentation,
  UserTurnAssembler,
  userTranscriptProjectionDisposition,
  type ApprovalDecision,
  type MarkdownEmphasisRun,
  type MicrophoneProcessingTelemetry,
  type RenderTimingRecord,
  type SpeechTiming,
  type UserTranscriptProjectionSource,
  type VoicePathState,
} from "./controller";
import {
  EvidenceControlTimeoutError,
  mountEvidenceControls,
} from "./evidence-controls";
import { mountSearchEgressControls } from "./search-egress-controls";
import { mountTypedComposerEnterSubmission } from "./typed-composer";
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
  type BootstrapCredential,
  type ModelCatalog,
  type PublicEvent,
  type SpeechAuthority,
} from "./protocol";

function element<T extends HTMLElement>(id: string): T {
  const value = document.getElementById(id);
  if (!(value instanceof HTMLElement)) {
    throw new Error(`Required client element is missing: ${id}`);
  }
  return value as T;
}

const stateOutput = element<HTMLOutputElement>("connection-state");
const connectionStatus = element<HTMLDivElement>("connection-status");
const microphoneActivity = element<HTMLButtonElement>("microphone-activity");
const microphoneActivityLabel = element<HTMLSpanElement>("microphone-activity-label");
const microphoneLevel = element<HTMLElement>("microphone-level");
const sessionToggleButton = element<HTMLButtonElement>("session-toggle");
const stopSpeakingButton = element<HTMLButtonElement>("stop-speaking");
const speechRendererState = element<HTMLOutputElement>("speech-renderer-state");
const muteButton = microphoneActivity;
const microphoneSelect = element<HTMLSelectElement>("microphone");
const voiceSelect = element<HTMLSelectElement>("voice");
const modelSelect = element<HTMLSelectElement>("model-select");
const effortSelect = element<HTMLSelectElement>("effort-select");
const modelSelectionState = element<HTMLParagraphElement>("model-selection-state");
const typedForm = element<HTMLFormElement>("typed-form");
const typedInput = element<HTMLTextAreaElement>("typed-input");
const sendButton = element<HTMLButtonElement>("send");
const transcript = element<HTMLOListElement>("transcript");
const transcriptEmpty = element<HTMLDivElement>("transcript-empty");
const readinessHeadline = element<HTMLHeadingElement>("readiness-headline");
const readinessDetail = element<HTMLParagraphElement>("readiness-detail");
const markers = element<HTMLOListElement>("markers");
const remoteAudio = element<HTMLAudioElement>("remote-audio");
mountTypedComposerEnterSubmission(typedInput, typedForm, sendButton);
const evidenceControls = mountEvidenceControls(document, {
  submit: submitEvidenceControl,
});
evidenceControls.setInteractive(false);
const searchEgressControls = mountSearchEgressControls(document, {
  submit: submitSearchEgressControl,
});
searchEgressControls.setInteractive(false);
const modelFields = [
  element<HTMLElement>("model-provider"),
  element<HTMLElement>("model-name"),
  element<HTMLElement>("model-access"),
  element<HTMLElement>("model-effort"),
  element<HTMLElement>("model-transport"),
  element<HTMLElement>("model-context"),
  element<HTMLElement>("model-usage-reporting"),
] as const;
const speechFields = [
  element<HTMLElement>("stt-provider"),
  element<HTMLElement>("stt-model"),
  element<HTMLElement>("tts-provider"),
  element<HTMLElement>("tts-model"),
] as const;
const usageFields = [
  element<HTMLElement>("usage-total"),
  element<HTMLElement>("usage-input"),
  element<HTMLElement>("usage-cached"),
  element<HTMLElement>("usage-output"),
  element<HTMLElement>("usage-reasoning"),
] as const;
const latencyLast = element<HTMLElement>("latency-last");
const latencyAverage = element<HTMLElement>("latency-average");
const knowledgeRoute = element<HTMLElement>("knowledge-route");
const knowledgeLatencyFields = [
  element<HTMLElement>("knowledge-last"),
  element<HTMLElement>("knowledge-p50"),
  element<HTMLElement>("knowledge-p95"),
] as const;
const transcriptRetention = new BoundedRetention<HTMLLIElement>(128, 131_072);
const markerRetention = new BoundedRetention<HTMLLIElement>(256, 32_768);
const eventPollIntervalMs = 100;
const voicePathReadinessTimeoutMs = 8_000;
const provisionalSpeechYieldTimeoutMs = 10_000;
const provisionalSpeechQuietReleaseMs = 750;
const provisionalSpeechVolume = 0.18;

class MicrophoneSignalMonitor {
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private analyser: AnalyserNode | null = null;
  private animationFrame: number | null = null;
  private active: boolean | null = null;
  private muted = false;
  private readonly gate = new MicrophoneActivityGate();
  private readonly samples = new Float32Array(256);

  prepare(): void {
    if (this.context !== null) return;
    this.context = new AudioContext({ latencyHint: "interactive" });
    void this.context.resume().catch(() => addMarker("microphone_meter_resume_blocked"));
  }

  attach(track: MediaStreamTrack): void {
    this.detachSource();
    this.prepare();
    const context = this.context;
    if (context === null) throw new Error("microphone signal context is unavailable");
    const analyser = context.createAnalyser();
    analyser.fftSize = this.samples.length;
    analyser.smoothingTimeConstant = 0;
    const source = context.createMediaStreamSource(new MediaStream([track]));
    source.connect(analyser);
    this.source = source;
    this.analyser = analyser;
    microphoneActivity.dataset.enabled = "true";
    this.active = null;
    this.setActive(false);
    this.sample();
  }

  stop(): void {
    this.detachSource();
    const context = this.context;
    this.context = null;
    if (context !== null) void context.close().catch(() => undefined);
    microphoneActivity.dataset.enabled = "false";
    this.active = null;
    this.setActive(false, "Mic off");
    microphoneLevel.style.transform = "scaleX(0)";
    this.gate.reset();
  }

  setMuted(muted: boolean): void {
    if (this.muted !== muted) this.active = null;
    this.muted = muted;
    if (muted) {
      this.setActive(false, "Mic muted");
      microphoneLevel.style.transform = "scaleX(0)";
    }
  }

  get isActive(): boolean {
    return this.active === true;
  }

  private detachSource(): void {
    if (this.animationFrame !== null) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = null;
    this.source?.disconnect();
    this.analyser?.disconnect();
    this.source = null;
    this.analyser = null;
    this.gate.reset();
  }

  private setActive(
    active: boolean,
    label = active ? "Mic active" : "Mic on",
    observedAt = performance.now(),
  ): void {
    if (this.active === active) return;
    const wasInactive = this.active === false;
    this.active = active;
    microphoneActivity.dataset.active = String(active);
    microphoneActivityLabel.textContent = label;
    if (active && wasInactive) beginProvisionalSpeechYield(observedAt);
    if (!active) quietProvisionalSpeechYield();
  }

  private sample = (): void => {
    const analyser = this.analyser;
    if (analyser === null) return;
    if (this.muted) {
      this.setActive(false, "Mic muted");
      microphoneLevel.style.transform = "scaleX(0)";
      this.animationFrame = requestAnimationFrame(this.sample);
      return;
    }
    analyser.getFloatTimeDomainData(this.samples);
    const level = microphoneSignalLevel(this.samples);
    const observedAt = performance.now();
    const active = this.gate.observe(level, observedAt);
    this.setActive(active, undefined, observedAt);
    microphoneLevel.style.transform = `scaleX(${Math.min(1, level / 0.12).toFixed(3)})`;
    this.animationFrame = requestAnimationFrame(this.sample);
  };
}

const controller = new ClientController();
const microphoneSignalMonitor = new MicrophoneSignalMonitor();

function prepareMicrophoneSignalMonitor(): void {
  try {
    microphoneSignalMonitor.prepare();
  } catch {
    addMarker("microphone_meter_unavailable");
  }
}

function attachMicrophoneSignalMonitor(track: LocalTrack): void {
  try {
    microphoneSignalMonitor.attach(track.mediaStreamTrack);
    addMarker("microphone_meter_attached");
  } catch {
    microphoneSignalMonitor.stop();
    addMarker("microphone_meter_unavailable");
  }
}

const objectiveLatency = new ObjectiveLatencyTracker();
const responseLatency = new ResponseLatencyStatistics();
let capability: string | null = null;
let stableLaunch = false;
let credential: BootstrapCredential | null = null;
let pendingRebindRequestId: string | null = null;
let projectionResyncAttempted = false;
let remoteStopRequired = false;
let sessionStopVersion = 0;
let room: Room | null = null;
const admittedRooms = new WeakSet<Room>();
let inputSequence = 0;
let eventSequence = 0;
let microphoneProcessingEvidence: {
  readonly track: LocalTrack;
  readonly value: MicrophoneProcessingTelemetry;
} | null = null;
let operation: AbortController | null = null;
let connectionAttempt: ConnectionAttempt<Room> | null = null;
let eventPolling: AbortController | null = null;
let credentialRefreshTimer: number | null = null;
let selectedVoice: string | null = null;
let selectableModels: ModelCatalog | null = null;
let modelSelectionPending = false;
let microphoneReady: boolean | null = null;
let voicePathState: VoicePathState = "waiting";
let voicePathReadinessTimer: number | null = null;
let microphoneVerificationGeneration = 0;
let mediaIncarnation = 0;
let activeMicrophoneTrack: LocalTrack | null = null;
let userMicrophoneMuted = false;
let microphoneMutePending = false;
const microphoneReadinessAuthority = new MicrophoneReadinessAuthority<Room, LocalTrack>();
const reconnectMicrophoneQueue = new SerializedAsyncQueue();
const audioDiagnosticQueue = new SerializedAsyncQueue();
const microphoneEnumerationAuthority = new GenerationAuthority();
const eventPollFailurePolicy = new ForegroundPollFailurePolicy(5);
const localMediaReleaseTimeoutMs = 1_000;
const stopRequestTimeoutMs = 5_000;
const connectionRequestTimeoutMs = 10_000;
const approvalController = new ApprovalDecisionController(
  async (sequence, approvalId, decision) => {
    const activeCredential = credential;
    if (activeCredential === null) throw new Error("browser session is not connected");
    const response = await fetch("/api/v1/approval", {
      method: "POST",
      headers: {
        ...authorization(activeCredential.token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ approvalId, decision, sequence }),
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
    });
    if (!response.ok) throw new Error("approval decision rejected");
  },
);
const launchStarted = performance.now();

function appendBounded(
  target: HTMLOListElement,
  item: HTMLLIElement,
  retention: BoundedRetention<HTMLLIElement>,
): void {
  target.append(item);
  for (const obsolete of retention.admit(item, retainedDomCost(item))) {
    obsolete.remove();
  }
}

function retainedDomCost(item: HTMLLIElement): number {
  return (item.textContent?.length ?? 0) + item.querySelectorAll("*").length * 32;
}

function syncTranscriptEmptyState(): void {
  transcriptEmpty.hidden = transcript.childElementCount > 0;
}

function addMarker(name: string, started = launchStarted): void {
  const item = document.createElement("li");
  item.textContent = `${name}: ${(performance.now() - started).toFixed(1)} ms`;
  appendBounded(markers, item, markerRetention);
}

function addTranscript(
  role: "You" | "Assistant",
  text: string,
  interrupted = false,
): void {
  const dataRole = role === "You" ? "user" : "assistant";
  if (dataRole === "assistant") closeActiveUserTurn();
  if (partialTranscriptRole === dataRole) {
    clearPartialTranscript();
  }

  const item = document.createElement("li");
  item.dataset.role = dataRole;
  if (interrupted) item.dataset.interrupted = "true";
  const label = document.createElement("strong");
  label.textContent = `${role}${interrupted ? " (interrupted)" : ""}: `;
  item.append(label, document.createTextNode(text));
  appendTranscriptBounded(item);
  syncTranscriptEmptyState();
  item.scrollIntoView({ block: "nearest" });
}

function addBackgroundResult(text: string): void {
  closeActiveUserTurn();
  const item = document.createElement("li");
  item.dataset.role = "task-result";
  const label = document.createElement("strong");
  label.textContent = "Background result: ";
  item.append(label, document.createTextNode(text));
  appendTranscriptBounded(item);
  syncTranscriptEmptyState();
  item.scrollIntoView({ block: "nearest" });
}

interface TaskCardView {
  readonly item: HTMLLIElement;
  readonly status: HTMLParagraphElement;
  readonly meta: HTMLSpanElement;
  readonly startedBrowserMs: number;
  readonly startedServerMs: number;
  currentStatus: string;
  timer: number | null;
}

interface ApprovalCardView {
  readonly approvalId: string;
  readonly item: HTMLLIElement;
  readonly status: HTMLParagraphElement;
  readonly approve: HTMLButtonElement;
  readonly reject: HTMLButtonElement;
}

const taskCardViews = new Map<string, TaskCardView>();
// Concurrent background runs can each hold a pending approval, and the server
// retains and resolves them by exact id in any order. Every actionable request
// therefore needs its own card; a single active reference stranded the older run
// until the whole session was stopped.
const approvalCardViews = new Map<string, ApprovalCardView>();

function actionableApprovalCard(item: HTMLLIElement): boolean {
  for (const view of approvalCardViews.values()) {
    if (view.item === item) return true;
  }
  return false;
}

function forgetEvictedTaskCard(item: HTMLLIElement): void {
  const taskId = item.dataset.taskId;
  if (taskId === undefined) return;
  const view = taskCardViews.get(taskId);
  if (view?.item === item) {
    if (view.timer !== null) window.clearInterval(view.timer);
    taskCardViews.delete(taskId);
  }
}

function disposeTranscriptEvictions(obsoleteItems: readonly HTMLLIElement[]): void {
  const pending = [...obsoleteItems];
  const repinned = new Set<HTMLLIElement>();
  while (pending.length > 0) {
    const obsolete = pending.shift();
    if (obsolete === undefined) break;
    if (actionableApprovalCard(obsolete) && !repinned.has(obsolete)) {
      repinned.add(obsolete);
      transcript.append(obsolete);
      pending.push(...transcriptRetention.admit(obsolete, retainedDomCost(obsolete)));
      continue;
    }
    forgetEvictedTaskCard(obsolete);
    obsolete.remove();
  }
}

function appendTranscriptBounded(item: HTMLLIElement): void {
  transcript.append(item);
  disposeTranscriptEvictions(transcriptRetention.admit(item, retainedDomCost(item)));
}

function updateTranscriptBounded(item: HTMLLIElement): boolean {
  if (!item.isConnected) return false;
  disposeTranscriptEvictions(transcriptRetention.update(item, retainedDomCost(item)));
  return item.isConnected;
}

function admitOperationCard(item: HTMLLIElement): void {
  closeActiveUserTurn();
  appendTranscriptBounded(item);
  syncTranscriptEmptyState();
  item.scrollIntoView({ block: "nearest" });
}

function refreshOperationCard(item: HTMLLIElement): void {
  if (updateTranscriptBounded(item)) item.scrollIntoView({ block: "nearest" });
}

function projectTaskStateCard(event: PublicEvent): void {
  const data = event.data;
  const taskId = data.taskId;
  const status = data.status;
  if (
    typeof taskId !== "string" ||
    !/^task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,122}$/.test(taskId) ||
    typeof status !== "string" ||
    !new Set(["active", "cancelling", "completed", "failed", "interrupted", "rejected"]).has(
      status,
    )
  ) {
    return;
  }
  let view = taskCardViews.get(taskId);
  if (view === undefined || !view.item.isConnected) {
    const item = document.createElement("li");
    item.dataset.role = "operation";
    item.dataset.operation = "task";
    item.dataset.taskId = taskId;
    const label = document.createElement("strong");
    label.textContent = "Execution";
    const title = document.createElement("span");
    title.className = "operation-title";
    title.textContent = "Background task";
    const state = document.createElement("p");
    state.className = "operation-detail";
    state.setAttribute("role", "timer");
    const meta = document.createElement("span");
    meta.className = "operation-meta";
    meta.textContent = "You can keep talking while this runs.";
    item.append(label, title, state, meta);
    view = {
      item,
      status: state,
      meta,
      startedBrowserMs: performance.now(),
      startedServerMs: event.monotonicMs,
      currentStatus: status,
      timer: null,
    };
    taskCardViews.set(taskId, view);
    admitOperationCard(item);
  }
  view.item.dataset.status = status;
  view.currentStatus = status;
  const render = (): void => {
    const elapsedMs =
      view!.currentStatus === "active" || view!.currentStatus === "cancelling"
        ? performance.now() - view!.startedBrowserMs
        : Math.max(0, event.monotonicMs - view!.startedServerMs);
    view!.status.textContent = taskStatusPresentation(view!.currentStatus, elapsedMs);
    view!.meta.textContent = taskMetaPresentation(view!.currentStatus);
  };
  render();
  if (status === "active" || status === "cancelling") {
    if (view.timer === null) view.timer = window.setInterval(render, 1_000);
  } else if (view.timer !== null) {
    window.clearInterval(view.timer);
    view.timer = null;
  }
  refreshOperationCard(view.item);
}

function projectApprovalCard(data: PublicEvent["data"]): void {
  const approvalId = data.approvalId;
  const state = data.state;
  const actionable = data.actionable === true;
  if (typeof approvalId !== "string" || typeof state !== "string") return;

  if (!actionable) {
    const view = approvalCardViews.get(approvalId);
    if (view === undefined) return;
    approvalCardViews.delete(approvalId);
    view.approve.disabled = true;
    view.reject.disabled = true;
    view.item.dataset.status = state;
    view.status.textContent = state === "approve" ? "Approved." : state === "reject" ? "Rejected." : state;
    updateTranscriptBounded(view.item);
    return;
  }

  const command = data.command;
  const description = data.description;
  const taskId = data.taskId;
  if (
    typeof command !== "string" ||
    command.trim().length < 1 ||
    typeof description !== "string" ||
    description.trim().length < 1 ||
    typeof taskId !== "string"
  ) {
    return;
  }

  const existing = approvalCardViews.get(approvalId);
  if (existing !== undefined && existing.item.isConnected) return;

  const item = document.createElement("li");
  item.dataset.role = "operation";
  item.dataset.operation = "approval";
  item.dataset.status = state;
  const label = document.createElement("strong");
  label.textContent = "Approval required";
  const title = document.createElement("span");
  title.className = "operation-title";
  title.textContent = description;
  const status = document.createElement("p");
  status.className = "operation-detail";
  status.textContent = "Review the requested command before continuing.";
  const commandView = document.createElement("code");
  commandView.className = "operation-command";
  commandView.textContent = command;
  const meta = document.createElement("span");
  meta.className = "operation-meta";
  meta.textContent = taskId;
  const controls = document.createElement("div");
  controls.className = "approval-controls";
  const approve = document.createElement("button");
  approve.className = "button approve";
  approve.type = "button";
  approve.textContent = "Approve";
  const reject = document.createElement("button");
  reject.className = "button secondary";
  reject.type = "button";
  reject.textContent = "Reject";
  controls.append(approve, reject);
  item.append(label, title, status, commandView, meta, controls);
  const view = { approvalId, item, status, approve, reject };
  approvalCardViews.set(approvalId, view);
  // Disable this card's controls for the whole round trip. Without this a
  // double-click, or Approve immediately followed by Reject, issues two
  // decisions for the same request. Other pending approvals stay actionable.
  const submitDecision = (decision: "approve" | "reject"): void => {
    if (approve.disabled || reject.disabled) return;
    approve.disabled = true;
    reject.disabled = true;
    void submitApproval(approvalId, decision).catch(() => {
      addMarker("approval_failed");
      // A failed decision consumes no sequence, so it may be retried. Restore
      // the controls only while this request is still awaiting a decision.
      if (approvalCardViews.get(approvalId) === view) {
        approve.disabled = false;
        reject.disabled = false;
      }
    });
  };
  approve.addEventListener("click", () => submitDecision("approve"));
  reject.addEventListener("click", () => submitDecision("reject"));
  admitOperationCard(item);
}

let partialTranscriptItem: HTMLLIElement | null = null;
let partialTranscriptRole: "user" | "assistant" | null = null;
let partialTranscriptText = "";

interface UserTurnView {
  readonly assembler: UserTurnAssembler;
  readonly item: HTMLLIElement;
  readonly content: Text;
}

let activeUserTurn: UserTurnView | null = null;

function closeActiveUserTurn(): void {
  activeUserTurn?.assembler.close();
  activeUserTurn = null;
}

function addUserTranscript(text: string): void {
  if (partialTranscriptRole === "user") clearPartialTranscript();
  if (activeUserTurn === null || !activeUserTurn.item.isConnected) {
    const item = document.createElement("li");
    item.dataset.role = "user";
    const label = document.createElement("strong");
    label.textContent = "You: ";
    const content = document.createTextNode("");
    item.append(label, content);
    appendTranscriptBounded(item);
    activeUserTurn = { assembler: new UserTurnAssembler(), item, content };
    syncTranscriptEmptyState();
  }
  const view = activeUserTurn;
  const snapshot = view.assembler.admit(text);
  view.content.data = snapshot.text;
  updateTranscriptBounded(view.item);
  if (!view.item.isConnected) activeUserTurn = null;
  syncTranscriptEmptyState();
  view.item.scrollIntoView({ block: "nearest" });
}

function projectUserTranscript(source: UserTranscriptProjectionSource, text: string): void {
  if (userTranscriptProjectionDisposition(source) === "project") {
    addUserTranscript(text);
  }
}

function clearPartialTranscript(): void {
  partialTranscriptItem?.remove();
  partialTranscriptItem = null;
  partialTranscriptRole = null;
  partialTranscriptText = "";
  syncTranscriptEmptyState();
}

function retainInterruptedAssistantTranscript(): void {
  if (partialTranscriptRole !== "assistant" || !partialTranscriptText.trim()) {
    clearPartialTranscript();
    return;
  }
  const text = partialTranscriptText;
  clearPartialTranscript();
  addTranscript("Assistant", text, true);
}

function updatePartialTranscript(role: "user" | "assistant", text: string): void {
  if (role === "assistant") closeActiveUserTurn();
  if (partialTranscriptItem !== null && partialTranscriptRole !== role) {
    clearPartialTranscript();
  }
  if (partialTranscriptItem === null) {
    partialTranscriptItem = document.createElement("li");
    partialTranscriptItem.dataset.partialTranscript = "true";
    partialTranscriptItem.dataset.role = role;
    partialTranscriptRole = role;
    transcript.append(partialTranscriptItem);
    syncTranscriptEmptyState();
  }
  partialTranscriptText = text;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You (live): " : "Assistant (live): ";
  const value = document.createElement("em");
  value.textContent = text;
  partialTranscriptItem.replaceChildren(label, value);
  partialTranscriptItem.scrollIntoView({ block: "nearest" });
}

interface AssistantSegmentView {
  readonly text: string;
  readonly markdownRuns: readonly MarkdownEmphasisRun[];
  readonly element: HTMLSpanElement;
  readonly separator: HTMLSpanElement | null;
  wordElements: HTMLSpanElement[];
  gapElements: HTMLSpanElement[];
}

function appendAssistantMarkdownRange(
  target: ParentNode,
  text: string,
  runs: readonly MarkdownEmphasisRun[],
  start: number,
  end: number,
): void {
  for (const run of runs) {
    const overlapStart = Math.max(start, run.sourceStart);
    const overlapEnd = Math.min(end, run.sourceEnd);
    if (overlapStart >= overlapEnd) continue;
    const value = document.createTextNode(text.slice(overlapStart, overlapEnd));
    if (run.emphasis === "none") {
      target.append(value);
      continue;
    }
    if (run.emphasis === "italic") {
      const emphasis = document.createElement("em");
      emphasis.append(value);
      target.append(emphasis);
      continue;
    }
    const strong = document.createElement("strong");
    if (run.emphasis === "bold-italic") {
      const emphasis = document.createElement("em");
      emphasis.append(value);
      strong.append(emphasis);
    } else {
      strong.append(value);
    }
    target.append(strong);
  }
}

interface AssistantTurnView {
  readonly assembler: AssistantTurnAssembler;
  readonly item: HTMLLIElement;
  readonly label: HTMLElement;
  readonly content: HTMLSpanElement;
  readonly segments: Map<string, AssistantSegmentView>;
}

const assistantTurns = new Map<string, AssistantTurnView>();
const assistantGenerationAuthority = new AssistantGenerationAuthority(256);

const speechTimings = new SpeechTimingRegistry(32);
const streamMediaStarts = new SpeechMediaStartRegistry(32);
const speechYieldController = new SpeechYieldController<SpeechTiming>(sameSpeechAuthority);
const provisionalSpeechYieldController = new ProvisionalSpeechYieldController<SpeechAuthority>(
  sameSpeechAuthority,
);
let locallySilencedSpeech: SpeechTiming | null = null;
let provisionalSpeechYieldStartedAt: number | null = null;
let provisionalSpeechYieldTimer: number | null = null;
let provisionalSpeechQuietTimer: number | null = null;
let speechYieldPending = false;
let naturalDuplexEnabled = false;
let activeAssistantTurnId: string | null = null;
let activeKaraokeFrame: number | null = null;
let activeKaraokeStreamId: string | null = null;
let activeKaraokeElements: HTMLSpanElement[] = [];
let activeKaraokeGaps: HTMLSpanElement[] = [];
let activeKaraokeIndex = -1;
let activeRemoteTrack: RemoteTrack | null = null;

function boundedMapSet<Key, Value>(map: Map<Key, Value>, key: Key, value: Value, limit = 32): void {
  if (map.has(key)) map.delete(key);
  map.set(key, value);
  while (map.size > limit) {
    const oldest = map.keys().next().value as Key | undefined;
    if (oldest === undefined) break;
    map.delete(oldest);
  }
}

function refreshAssistantRetention(view: AssistantTurnView): boolean {
  if (!view.item.isConnected) return false;
  updateTranscriptBounded(view.item);
  for (const [turnId, retained] of assistantTurns) {
    if (!retained.item.isConnected) assistantTurns.delete(turnId);
  }
  syncTranscriptEmptyState();
  return view.item.isConnected;
}

function assistantEventIdentity(
  event: PublicEvent,
): { turnId: string; turnGeneration: number; chunkId: string | null } | null {
  const { turnId, turnGeneration, chunkId, segmentId } = event.data;
  const presentationId = segmentId ?? chunkId;
  if (
    typeof turnId !== "string" ||
    turnId.length < 1 ||
    turnId.length > 128 ||
    !Number.isSafeInteger(turnGeneration) ||
    (turnGeneration as number) < 1 ||
    (presentationId !== undefined &&
      (typeof presentationId !== "string" ||
        presentationId.length < 1 ||
        presentationId.length > 128))
  ) {
    return null;
  }
  return {
    turnId,
    turnGeneration: turnGeneration as number,
    chunkId: typeof presentationId === "string" ? presentationId : null,
  };
}

function createAssistantTurn(turnId: string, turnGeneration: number): AssistantTurnView {
  const assembler = new AssistantTurnAssembler(turnId, turnGeneration);
  const item = document.createElement("li");
  item.dataset.role = "assistant";
  item.dataset.partialTranscript = "true";
  item.dataset.turnId = turnId;
  const label = document.createElement("strong");
  label.textContent = "Assistant (live): ";
  const content = document.createElement("span");
  content.className = "assistant-copy";
  item.append(label, content);
  appendTranscriptBounded(item);
  for (const [retainedTurnId, retained] of assistantTurns) {
    if (!retained.item.isConnected) assistantTurns.delete(retainedTurnId);
  }
  syncTranscriptEmptyState();
  const view = { assembler, item, label, content, segments: new Map() };
  assistantTurns.set(turnId, view);
  return view;
}

function assistantTurnView(turnId: string, turnGeneration: number): AssistantTurnView | null {
  let existing = assistantTurns.get(turnId);
  if (existing !== undefined && !existing.item.isConnected) {
    assistantTurns.delete(turnId);
    existing = undefined;
  }
  const disposition = assistantGenerationAuthority.disposition(turnId, turnGeneration);
  if (disposition === "create") return createAssistantTurn(turnId, turnGeneration);
  if (disposition === "reject") return null;
  // The authority's high-watermark already equals this generation, so it will
  // keep answering "current". Returning null once the item has been evicted from
  // the bounded transcript would silently discard every remaining chunk of a live
  // turn, so re-create the view instead of dropping the rest of the response.
  if (disposition === "current") {
    return existing ?? createAssistantTurn(turnId, turnGeneration);
  }
  if (existing !== undefined) {
    transcriptRetention.remove(existing.item);
    existing.item.remove();
    assistantTurns.delete(turnId);
  }
  return createAssistantTurn(turnId, turnGeneration);
}

function renderAssistantTurnState(view: AssistantTurnView): void {
  const snapshot = view.assembler.snapshot;
  view.item.dataset.partialTranscript = snapshot.state === "live" ? "true" : "false";
  if (snapshot.state === "interrupted") view.item.dataset.interrupted = "true";
  else delete view.item.dataset.interrupted;
  view.label.textContent =
    snapshot.state === "live"
      ? "Assistant (live): "
      : snapshot.state === "interrupted"
        ? "Assistant (interrupted): "
        : "Assistant: ";
  const unspoken = new Set(snapshot.unspokenChunkIds);
  for (const [chunkId, segment] of view.segments) {
    segment.element.classList.toggle(
      "assistant-unspoken",
      snapshot.state === "interrupted" && unspoken.has(chunkId),
    );
  }
  if (refreshAssistantRetention(view)) view.item.scrollIntoView({ block: "nearest" });
}

function admitAssistantSegment(event: PublicEvent, text: string): boolean {
  const identity = assistantEventIdentity(event);
  if (identity === null || identity.chunkId === null) return false;
  const view = assistantTurnView(identity.turnId, identity.turnGeneration);
  if (view === null) return false;
  if (view.assembler.tryAdmit(identity.chunkId, text) === null) return false;
  closeActiveUserTurn();
  if (!view.segments.has(identity.chunkId)) {
    const separator = view.segments.size > 0 ? document.createElement("span") : null;
    if (separator !== null) {
      separator.className = "assistant-separator";
      separator.textContent = " ";
    }
    if (separator !== null) view.content.append(separator);
    const element = document.createElement("span");
    element.className = "assistant-segment";
    element.dataset.chunkId = identity.chunkId;
    const markdownRuns = markdownEmphasisRuns(text);
    appendAssistantMarkdownRange(element, text, markdownRuns, 0, text.length);
    view.content.append(element);
    view.segments.set(identity.chunkId, {
      text,
      markdownRuns,
      element,
      separator,
      wordElements: [],
      gapElements: [],
    });
    for (const timing of speechTimings.forSegment(
      identity.turnId,
      identity.turnGeneration,
      identity.chunkId,
    )) {
      applySpeechTiming(timing);
    }
    const activeTiming =
      activeKaraokeStreamId === null ? undefined : speechTimings.get(activeKaraokeStreamId);
    if (
      activeTiming !== undefined &&
      activeTiming.turnId === identity.turnId &&
      activeTiming.turnGeneration === identity.turnGeneration
    ) {
      projectKaraokeSegmentSuffix(view, activeTiming);
    }
  }
  activeAssistantTurnId = identity.turnId;
  renderAssistantTurnState(view);
  return true;
}

function confirmAssistantSegment(event: PublicEvent): boolean {
  const identity = assistantEventIdentity(event);
  if (identity === null || identity.chunkId === null) return false;
  const view = assistantTurns.get(identity.turnId);
  if (view === undefined || view.assembler.turnGeneration !== identity.turnGeneration) return false;
  view.assembler.confirm(identity.chunkId);
  renderAssistantTurnState(view);
  return true;
}

function completeAssistantTurn(event: PublicEvent): void {
  const identity = assistantEventIdentity(event);
  if (identity === null) return;
  const view = assistantTurns.get(identity.turnId);
  if (view === undefined || view.assembler.turnGeneration !== identity.turnGeneration) return;
  view.assembler.complete();
  renderAssistantTurnState(view);
  if (activeAssistantTurnId === identity.turnId) activeAssistantTurnId = null;
}

function interruptAssistantTurn(turnId: string, turnGeneration: number): void {
  const view = assistantTurns.get(turnId);
  if (view === undefined || view.assembler.turnGeneration !== turnGeneration) return;
  view.assembler.interrupt();
  renderAssistantTurnState(view);
  if (activeAssistantTurnId === turnId) {
    activeAssistantTurnId = null;
    stopKaraoke();
  }
}

function interruptActiveAssistantTurn(): void {
  if (activeAssistantTurnId === null) return;
  const view = assistantTurns.get(activeAssistantTurnId);
  if (view === undefined) {
    activeAssistantTurnId = null;
    stopKaraoke();
    return;
  }
  interruptAssistantTurn(
    activeAssistantTurnId,
    view.assembler.turnGeneration,
  );
}

function interruptAssistantTurnEvent(event: PublicEvent): void {
  const identity = assistantEventIdentity(event);
  if (identity === null) return;
  interruptAssistantTurn(identity.turnId, identity.turnGeneration);
}

function currentSpeechTiming(): SpeechTiming | null {
  const streamId = remoteAudio.dataset.streamId;
  if (typeof streamId !== "string") return null;
  return speechTimings.get(streamId) ?? null;
}

function renderSpeechControl(
  state: "ducked" | "idle" | "ready" | "recover" | "silenced" | "yielding",
): void {
  stopSpeakingButton.hidden = !naturalDuplexEnabled;
  speechRendererState.hidden = !naturalDuplexEnabled;
  speechRendererState.textContent = {
    ducked: "Listening — speech lowered",
    idle: "Speech idle",
    ready: "Speech renderer ready",
    recover: "Speech paused — tap Resume speech",
    silenced: "Speech stopped",
    yielding: "Stopping speech…",
  }[state];
  stopSpeakingButton.textContent = state === "recover" ? "Resume speech" : "Stop speaking";
  stopSpeakingButton.dataset.action = state === "recover" ? "resume" : "stop";
  stopSpeakingButton.disabled = !["ducked", "ready", "recover"].includes(state);
}

function clearProvisionalSpeechYieldTimer(): void {
  if (provisionalSpeechYieldTimer === null) return;
  window.clearTimeout(provisionalSpeechYieldTimer);
  provisionalSpeechYieldTimer = null;
}

function clearProvisionalSpeechQuietTimer(): void {
  if (provisionalSpeechQuietTimer === null) return;
  window.clearTimeout(provisionalSpeechQuietTimer);
  provisionalSpeechQuietTimer = null;
}

function setSpeechRendererVolume(volume: number): void {
  if (activeRemoteTrack instanceof RemoteAudioTrack) {
    activeRemoteTrack.setVolume(volume);
  } else {
    remoteAudio.volume = volume;
  }
}

function setSpeechRendererEnabled(enabled: boolean): void {
  if (activeRemoteTrack !== null) {
    activeRemoteTrack.mediaStreamTrack.enabled = enabled;
  }
}

function beginProvisionalSpeechYield(startedAt = performance.now()): void {
  clearProvisionalSpeechQuietTimer();
  const timing = currentSpeechTiming();
  if (
    !provisionalSpeechYieldAllowed(
      naturalDuplexEnabled,
      activeRemoteTrack !== null,
      timing !== null,
      remoteAudio.paused,
      speechYieldPending,
      locallySilencedSpeech !== null,
    )
  ) return;
  const onset = provisionalSpeechYieldController.begin(timing);
  if (onset === null) return;
  setSpeechRendererVolume(provisionalSpeechVolume);
  renderSpeechControl("ducked");
  if (onset.firstOnset) {
    provisionalSpeechYieldStartedAt = startedAt;
    addMarker("microphone_onset_to_attenuation", startedAt);
    clearProvisionalSpeechYieldTimer();
    provisionalSpeechYieldTimer = window.setTimeout(() => {
      provisionalSpeechYieldTimer = null;
      expireProvisionalSpeechYield("local_voice_claim_timeout");
    }, provisionalSpeechYieldTimeoutMs);
  } else {
    addMarker("local_voice_reattenuated");
  }
}

function quietProvisionalSpeechYield(): void {
  const pending = provisionalSpeechYieldController.pending;
  const current = currentSpeechTiming();
  if (
    pending === null ||
    current === null ||
    !sameSpeechAuthority(pending, current) ||
    !provisionalSpeechYieldController.attenuated ||
    provisionalSpeechQuietTimer !== null
  ) return;
  provisionalSpeechQuietTimer = window.setTimeout(() => {
    provisionalSpeechQuietTimer = null;
    if (
      microphoneSignalMonitor.isActive ||
      !provisionalSpeechYieldController.quiet(currentSpeechTiming())
    ) return;
    setSpeechRendererVolume(1);
    renderSpeechControl("ready");
    const startedAt = provisionalSpeechYieldStartedAt;
    if (startedAt !== null) addMarker("microphone_onset_to_recovery", startedAt);
  }, provisionalSpeechQuietReleaseMs);
}

function settleProvisionalSpeechYield(
  authority: SpeechAuthority,
  committed: boolean,
  marker: string,
): void {
  const current = currentSpeechTiming();
  const disposition = provisionalSpeechYieldController.settle(authority, current, committed);
  if (disposition === "ignored") return;
  clearProvisionalSpeechQuietTimer();
  clearProvisionalSpeechYieldTimer();
  const startedAt = provisionalSpeechYieldStartedAt;
  provisionalSpeechYieldStartedAt = null;
  addMarker(marker);
  if (disposition === "silenced" && current !== null) {
    locallySilencedSpeech = current;
    remoteAudio.pause();
    setSpeechRendererEnabled(false);
    setSpeechRendererVolume(0);
    stopKaraoke();
    renderSpeechControl("silenced");
    if (startedAt !== null) addMarker("microphone_onset_to_committed_silence", startedAt);
  } else if (disposition === "recover") {
    setSpeechRendererVolume(1);
    renderSpeechControl("ready");
  } else {
    setSpeechRendererVolume(1);
    renderSpeechControl("idle");
  }
}

function expireProvisionalSpeechYield(marker: string): void {
  if (provisionalSpeechYieldController.expire() === null) return;
  clearProvisionalSpeechQuietTimer();
  clearProvisionalSpeechYieldTimer();
  provisionalSpeechYieldStartedAt = null;
  setSpeechRendererVolume(1);
  addMarker(marker);
  renderSpeechControl(currentSpeechTiming() === null ? "idle" : "ready");
}

function resetProvisionalSpeechYield(): void {
  clearProvisionalSpeechQuietTimer();
  clearProvisionalSpeechYieldTimer();
  provisionalSpeechYieldController.reset();
  provisionalSpeechYieldStartedAt = null;
  setSpeechRendererVolume(1);
}

async function recoverSpeechRenderer(): Promise<void> {
  if (activeRemoteTrack === null || currentSpeechTiming() === null) {
    renderSpeechControl("idle");
    return;
  }
  locallySilencedSpeech = null;
  setSpeechRendererEnabled(true);
  setSpeechRendererVolume(
    provisionalSpeechYieldController.attenuated ? provisionalSpeechVolume : 1,
  );
  try {
    await remoteAudio.play();
    renderSpeechControl("ready");
    addMarker("speech_renderer_recovered");
  } catch {
    renderSpeechControl("recover");
    addMarker("speech_renderer_recovery_required");
  }
}

function parseYieldMatch(value: unknown): boolean {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("yield response is invalid");
  }
  const record = value as Record<string, unknown>;
  if (
    Object.keys(record).sort().join(",") !== "matched,version" ||
    typeof record.matched !== "boolean" ||
    record.version !== 1
  ) {
    throw new Error("yield response is invalid");
  }
  return record.matched;
}

async function requestSpeechYield(): Promise<void> {
  if (!naturalDuplexEnabled) return;
  const activeCredential = credential;
  if (activeCredential === null || activeRemoteTrack === null) return;
  resetProvisionalSpeechYield();
  const claim = speechYieldController.claim(currentSpeechTiming());
  if (claim === null) return;
  locallySilencedSpeech = claim;
  speechYieldPending = true;
  remoteAudio.pause();
  setSpeechRendererEnabled(false);
  setSpeechRendererVolume(0);
  stopKaraoke();
  renderSpeechControl("yielding");
  addMarker("stop_speaking_requested");
  let matched = false;
  try {
    const response = await fetch("/api/v1/yield", {
      method: "POST",
      headers: {
        ...authorization(activeCredential.token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        turnId: claim.turnId,
        turnGeneration: claim.turnGeneration,
        chunkId: claim.chunkId,
        streamId: claim.streamId,
      }),
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
      signal: AbortSignal.timeout(stopRequestTimeoutMs),
    });
    if (!response.ok) throw new Error("yield request rejected");
    matched = parseYieldMatch(await response.json());
  } catch {
    addMarker("speech_yield_failed");
  }
  speechYieldPending = false;
  const disposition = speechYieldController.settle(claim, currentSpeechTiming(), matched);
  if (disposition === "silenced") {
    renderSpeechControl("silenced");
    return;
  }
  if (disposition === "recover") await recoverSpeechRenderer();
}

function reconcileProvisionalSpeechAuthority(timing: SpeechTiming | null): void {
  if (
    !naturalDuplexEnabled ||
    timing === null ||
    remoteAudio.dataset.streamId !== timing.streamId
  ) {
    return;
  }
  if (
    provisionalSpeechYieldController.pending !== null &&
    !sameSpeechAuthority(provisionalSpeechYieldController.pending, timing)
  ) {
    const pending = provisionalSpeechYieldController.pending;
    if (
      pending.turnId === timing.turnId &&
      pending.turnGeneration === timing.turnGeneration
    ) {
      provisionalSpeechYieldController.replace(timing);
      setSpeechRendererVolume(
        provisionalSpeechYieldController.attenuated ? provisionalSpeechVolume : 1,
      );
      addMarker("local_voice_claim_authority_replaced");
    } else {
      expireProvisionalSpeechYield("local_voice_claim_superseded");
    }
  }
  if (
    locallySilencedSpeech !== null &&
    !sameSpeechAuthority(locallySilencedSpeech, timing) &&
    !speechYieldPending
  ) {
    locallySilencedSpeech = null;
  }
  if (locallySilencedSpeech !== null) {
    renderSpeechControl(speechYieldPending ? "yielding" : "silenced");
  } else {
    if (microphoneSignalMonitor.isActive) beginProvisionalSpeechYield();
    renderSpeechControl(provisionalSpeechYieldController.attenuated ? "ducked" : "ready");
    if (remoteAudio.paused) void recoverSpeechRenderer();
  }
}

function applySpeechTiming(timing: SpeechTiming): void {
  speechTimings.admit(timing);
  reconcileProvisionalSpeechAuthority(timing);
  if (!naturalDuplexEnabled) {
    resetProvisionalSpeechYield();
    locallySilencedSpeech = null;
    speechYieldPending = false;
  }
  if (
    activeRemoteTrack !== null &&
    remoteAudio.dataset.streamId === timing.streamId
  ) {
    streamMediaStarts.beginPassage(timing.chunkId, remoteAudio.currentTime);
    window.requestAnimationFrame(() => observeSpeechPassageStart(timing, 0));
  }
  const view = assistantTurns.get(timing.presentationTurnId);
  if (view === undefined || view.assembler.turnGeneration !== timing.turnGeneration) return;
  const segment = view.segments.get(timing.segmentId);
  if (segment === undefined || timing.words.some((word) => word.textEnd > segment.text.length)) return;

  const fragment = document.createDocumentFragment();
  const wordElements: HTMLSpanElement[] = [];
  const gapElements: HTMLSpanElement[] = [];
  let offset = 0;
  for (const word of timing.words) {
    const gap = document.createElement("span");
    gap.className = "karaoke-gap karaoke-pending";
    appendAssistantMarkdownRange(
      gap,
      segment.text,
      segment.markdownRuns,
      offset,
      word.textStart,
    );
    fragment.append(gap);
    gapElements.push(gap);
    const element = document.createElement("span");
    element.className = "karaoke-word karaoke-pending";
    appendAssistantMarkdownRange(
      element,
      segment.text,
      segment.markdownRuns,
      word.textStart,
      word.textEnd,
    );
    fragment.append(element);
    wordElements.push(element);
    offset = word.textEnd;
  }
  const trailingGap = document.createElement("span");
  trailingGap.className = "karaoke-gap karaoke-pending";
  appendAssistantMarkdownRange(
    trailingGap,
    segment.text,
    segment.markdownRuns,
    offset,
    segment.text.length,
  );
  fragment.append(trailingGap);
  gapElements.push(trailingGap);
  segment.element.replaceChildren(fragment);
  segment.wordElements = wordElements;
  segment.gapElements = gapElements;
  if (!refreshAssistantRetention(view)) return;
  maybeStartKaraoke(timing.streamId);
}

function observeSpeechPassageStart(timing: SpeechTiming, attempt: number): void {
  if (
    activeRemoteTrack === null ||
    remoteAudio.dataset.streamId !== timing.streamId ||
    speechTimings.get(timing.streamId) !== timing
  ) {
    return;
  }
  const mediaStart = streamMediaStarts.observePassage(timing.chunkId, remoteAudio.currentTime);
  if (mediaStart !== undefined) {
    maybeStartKaraoke(timing.streamId);
    return;
  }
  if (attempt >= 600) {
    addMarker("speech_passage_media_stalled");
    return;
  }
  window.requestAnimationFrame(() => observeSpeechPassageStart(timing, attempt + 1));
}

function stopKaraoke(complete = false): void {
  if (activeKaraokeFrame !== null) window.cancelAnimationFrame(activeKaraokeFrame);
  activeKaraokeFrame = null;
  activeKaraokeStreamId = null;
  for (const element of activeKaraokeElements) {
    element.classList.remove("karaoke-active", "karaoke-pending");
    if (complete) element.classList.add("karaoke-complete");
  }
  for (const gap of activeKaraokeGaps) gap.classList.remove("karaoke-pending");
  activeKaraokeElements = [];
  activeKaraokeGaps = [];
  activeKaraokeIndex = -1;
}

function projectKaraokeSegmentSuffix(view: AssistantTurnView, activeTiming: SpeechTiming): void {
  const entries = [...view.segments.entries()];
  const states = karaokeSegmentPendingStates(
    entries.map(([chunkId]) => chunkId),
    activeTiming,
  );
  entries.forEach(([, segment], index) => {
    const pending = states[index] ?? false;
    segment.element.classList.toggle("karaoke-pending", pending);
    segment.separator?.classList.toggle("karaoke-pending", pending);
  });
}

function projectKaraokeGaps(segment: AssistantSegmentView, completedWordCount: number): void {
  const states = karaokeGapPendingStates(segment.wordElements.length, completedWordCount);
  segment.gapElements.forEach((gap, index) => {
    gap.classList.toggle("karaoke-pending", states[index] ?? false);
  });
}

function maybeStartKaraoke(streamId: string): void {
  const timing = speechTimings.get(streamId);
  const mediaStart = timing === undefined ? undefined : streamMediaStarts.getPassage(timing.chunkId);
  if (timing === undefined || mediaStart === undefined) return;
  const view = assistantTurns.get(timing.presentationTurnId);
  const segment = view?.segments.get(timing.segmentId);
  if (segment === undefined || segment.wordElements.length !== timing.words.length) return;

  stopKaraoke();
  activeKaraokeStreamId = streamId;
  activeKaraokeElements = segment.wordElements;
  activeKaraokeGaps = segment.gapElements;
  projectKaraokeSegmentSuffix(view!, timing);
  const clock = new SpeechKaraokeClock(timing, mediaStart);
  const tick = (): void => {
    if (activeKaraokeStreamId !== streamId || remoteAudio.dataset.streamId !== streamId) return;
    const index = clock.wordIndexAt(remoteAudio.currentTime);
    if (index === -1) {
      stopKaraoke(true);
      return;
    }
    if (index === -2) {
      if (activeKaraokeIndex >= 0) projectKaraokeGaps(segment, activeKaraokeIndex + 1);
      const previous = activeKaraokeElements[activeKaraokeIndex];
      previous?.classList.remove("karaoke-active", "karaoke-pending");
      previous?.classList.add("karaoke-complete");
      activeKaraokeIndex = -2;
      activeKaraokeFrame = window.requestAnimationFrame(tick);
      return;
    }
    if (index !== activeKaraokeIndex) {
      const firstChanged = Math.max(0, activeKaraokeIndex);
      for (let completed = firstChanged; completed < index; completed += 1) {
        const element = activeKaraokeElements[completed];
        element?.classList.remove("karaoke-active", "karaoke-pending");
        element?.classList.add("karaoke-complete");
      }
      activeKaraokeElements[activeKaraokeIndex]?.classList.remove("karaoke-active");
      const current = activeKaraokeElements[index];
      current?.classList.remove("karaoke-pending");
      current?.classList.add("karaoke-active");
      projectKaraokeGaps(segment, index);
      current?.scrollIntoView({ block: "nearest", inline: "nearest" });
      activeKaraokeIndex = index;
    }
    activeKaraokeFrame = window.requestAnimationFrame(tick);
  };
  activeKaraokeFrame = window.requestAnimationFrame(tick);
}

function addObjectiveMarker(event: PublicEvent): void {
  const item = document.createElement("li");
  item.textContent = `${event.kind}: server monotonic ${event.monotonicMs.toFixed(1)} ms`;
  appendBounded(markers, item, markerRetention);
}

function addObjectiveLatency(event: PublicEvent): void {
  const role = typeof event.data.role === "string" ? event.data.role : undefined;
  const latency = objectiveLatency.observe(event.kind, event.monotonicMs, role);
  if (latency === null) return;
  const item = document.createElement("li");
  item.textContent = `${latency.name}: ${latency.durationMs.toFixed(1)} ms`;
  appendBounded(markers, item, markerRetention);
  const statistics = responseLatency.observe(latency);
  if (latency.name === "transcript_to_first_token") {
    latencyLast.textContent = formatLatencyDuration(statistics.lastMs);
    latencyAverage.textContent = formatLatencyDuration(statistics.averageMs);
  }
}

function projectPublicEvent(event: PublicEvent): void {
  if (event.kind === "session_ready") {
    naturalDuplexEnabled = naturalDuplexProfileEnabled({
      conversationProfile: event.data.conversationProfile,
      mode: event.data.mode,
    });
    const speechRuntime = parseSpeechRuntime({
      sttModel: event.data.sttModel,
      sttProvider: event.data.sttProvider,
      ttsModel: event.data.ttsModel,
      ttsProvider: event.data.ttsProvider,
    });
    [
      speechRuntime.sttProvider,
      speechRuntime.sttModel,
      speechRuntime.ttsProvider,
      speechRuntime.ttsModel,
    ].forEach((value, index) => {
      speechFields[index]!.textContent = value;
    });
    renderSpeechControl("idle");
  } else if (event.kind === "voice_input_ready") {
    confirmServerVoicePath(event.data.generation, event.data.mediaIncarnation);
  } else if (event.kind === "voice_activity_started") {
    addMarker("server_vad_speech_started");
  } else if (event.kind === "voice_activity_ended") {
    addMarker("server_vad_speech_ended");
  } else if (event.kind === "barge_in_non_speech_suppressed") {
    if (Object.keys(event.data).length > 0) {
      settleProvisionalSpeechYield(
        parseSpeechAuthority(event.data),
        false,
        "server_non_speech_released",
      );
    }
  } else if (event.kind === "session_model") {
    const configuration = parseSessionModelConfiguration(event.data);
    formatSessionModelConfiguration(configuration).forEach((row, index) => {
      modelFields[index]!.textContent = row[1];
    });
    if (configuration.reportsTokenUsage) usageFields[0].textContent = "Waiting for first turn";
  } else if (event.kind === "session_usage") {
    const usage = parseSessionTokenUsage(event.data);
    formatSessionTokenUsage(usage).forEach((row, index) => {
      usageFields[index]!.textContent = row[1];
    });
    if (usage.contextWindowTokens !== null) {
      modelFields[5].textContent = `${usage.contextWindowTokens.toLocaleString("en-US")} tokens`;
    }
  } else if (event.kind === "knowledge_timing") {
    const route = event.data.route;
    const backend = event.data.backend;
    const outcome = event.data.outcome;
    const timings = [event.data.lastMs, event.data.p50Ms, event.data.p95Ms];
    if (
      typeof route === "string" &&
      typeof backend === "string" &&
      typeof outcome === "string" &&
      timings.every((value) => typeof value === "number")
    ) {
      knowledgeRoute.textContent = `${route} · ${backend} · ${outcome}`;
      timings.forEach((value, index) => {
        knowledgeLatencyFields[index]!.textContent = formatLatencyDuration(value as number);
      });
    }
  } else if (event.kind === "capture_status") {
    evidenceControls.projectStatus(parseCaptureStatus(event.data));
  } else if (event.kind === "search_egress_status") {
    searchEgressControls.projectStatus(parseSearchEgressStatus(event.data));
  } else if (event.kind === "assistant_text_generated") {
    const text = event.data.text;
    if (typeof text === "string") admitAssistantSegment(event, text);
  } else if (event.kind === "transcript_partial") {
    const text = event.data.text;
    const role = event.data.role;
    if (typeof text === "string" && (role === "user" || role === "assistant")) {
      if (role === "assistant") {
        if (
          (typeof event.data.segmentId === "string" || admitAssistantSegment(event, text)) &&
          partialTranscriptRole === "assistant"
        ) {
          clearPartialTranscript();
        }
      } else {
        updatePartialTranscript(role, text);
      }
    }
  } else if (event.kind === "transcript_final") {
    const text = event.data.text;
    const role = event.data.role;
    if (typeof text === "string" && (role === "user" || role === "assistant")) {
      if (role === "assistant") {
        const identity = assistantEventIdentity(event);
        if (identity === null) {
          addTranscript("Assistant", text);
        } else if (typeof event.data.segmentId === "string") {
          confirmAssistantSegment(event);
        } else if (admitAssistantSegment(event, text)) {
          confirmAssistantSegment(event);
        }
      } else {
        projectUserTranscript("authoritative-event", text);
      }
    }
  } else if (event.kind === "speech_timing") {
    const timing = parseSpeechTiming(event.data);
    if (timing !== null) applySpeechTiming(timing);
  } else if (event.kind === "assistant_turn_completed") {
    completeAssistantTurn(event);
  } else if (event.kind === "assistant_turn_interrupted") {
    interruptAssistantTurnEvent(event);
  } else if (
    event.kind === "speech_ended" ||
    event.kind === "interrupt_requested" ||
    event.kind === "session_stopped"
  ) {
    if (event.kind === "interrupt_requested" && Object.keys(event.data).length > 0) {
      settleProvisionalSpeechYield(
        parseSpeechAuthority(event.data),
        true,
        "server_floor_claim_committed",
      );
    }
    if (event.kind !== "interrupt_requested") {
      if (event.kind === "session_stopped") {
        interruptActiveAssistantTurn();
        closeActiveUserTurn();
      }
      const disposition = partialTranscriptDisposition(event.kind, partialTranscriptRole);
      if (disposition === "retain-interrupted") {
        retainInterruptedAssistantTranscript();
      } else if (disposition === "clear") {
        clearPartialTranscript();
      }
    }
  } else if (event.kind === "task_state") {
    projectTaskStateCard(event);
  } else if (event.kind === "task_result") {
    const text = event.data.text;
    if (typeof text === "string" && text.trim().length > 0) {
      addBackgroundResult(text);
    }
  } else if (event.kind === "approval_state") {
    projectApprovalCard(event.data);
  }
  addObjectiveLatency(event);
  addObjectiveMarker(event);
}

async function pollPublicEvents(signal: AbortSignal): Promise<void> {
  let consecutiveFailures = 0;
  while (!signal.aborted) {
    try {
      const activeCredential = credential;
      if (activeCredential === null) return;
      const response = await fetch("/api/v1/events", {
        method: "POST",
        headers: {
          ...authorization(activeCredential.token),
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ after: eventSequence }),
        cache: "no-store",
        credentials: "omit",
        referrerPolicy: "no-referrer",
        signal,
      });
      if (!response.ok) throw new Error("public event projection rejected");
      const batch = parseEventBatch(await response.json());
      for (const event of batch.events) {
        if (event.sequence !== eventSequence + 1) {
          throw new Error("public event sequence gap");
        }
        projectPublicEvent(event);
        eventSequence = event.sequence;
      }
      consecutiveFailures = 0;
      eventPollFailurePolicy.recordSuccess();
      await new Promise<void>((resolve) =>
        window.setTimeout(resolve, eventPollIntervalMs),
      );
    } catch (error) {
      if (signal.aborted) return;
      consecutiveFailures += 1;
      if (eventPollFailurePolicy.recordFailure() === "terminal") {
        if (controller.state === "stopping") return;
        try {
          await recoverProjectionResync(signal);
          return;
        } catch (resyncError) {
          if (signal.aborted) return;
          error = resyncError;
        }
        const observedStopVersion = sessionStopVersion;
        if (eventPolling?.signal === signal) {
          eventPolling.abort();
          eventPolling = null;
        }
        await disconnectLocal();
        if (observedStopVersion !== sessionStopVersion) return;
        controller.failed();
        throw error;
      }
      addMarker("event_projection_retry");
      await new Promise<void>((resolve) =>
        window.setTimeout(resolve, Math.min(1600, 100 * 2 ** consecutiveFailures)),
      );
    }
  }
}

document.addEventListener("visibilitychange", () => {
  eventPollFailurePolicy.visibilityChanged(document.visibilityState);
});

function setInteractive(connected: boolean): void {
  muteButton.disabled = !connected || activeMicrophoneTrack === null || microphoneMutePending;
  typedInput.disabled = !connected;
  sendButton.disabled = !connected;
  evidenceControls.setInteractive(connected);
  searchEgressControls.setInteractive(connected);
  voiceSelect.disabled = !connected || selectedVoice === null;
  const modelInteractive = connected && selectableModels !== null && !modelSelectionPending;
  modelSelect.disabled = !modelInteractive;
  effortSelect.disabled = !modelInteractive;
}

function renderMicrophoneMute(): void {
  const action = userMicrophoneMuted ? "Unmute microphone" : "Mute microphone";
  muteButton.setAttribute("aria-label", action);
  muteButton.title = action;
  muteButton.setAttribute("aria-pressed", String(userMicrophoneMuted));
  muteButton.dataset.muted = String(userMicrophoneMuted);
  microphoneSignalMonitor.setMuted(userMicrophoneMuted);
}

async function toggleMicrophoneMute(): Promise<void> {
  if (activeMicrophoneTrack === null || controller.state !== "connected" || microphoneMutePending) {
    return;
  }
  const requested = !userMicrophoneMuted;
  microphoneMutePending = true;
  setInteractive(true);
  try {
    await reconnectMicrophoneQueue.run(async () => {
      const track = activeMicrophoneTrack;
      if (track === null || controller.state !== "connected") return;
      const committed = await settleMicrophoneMuteChoice(
        track,
        requested,
        async (candidate) => {
          await candidate.mute();
        },
        async (candidate) => {
          await candidate.unmute();
        },
        (candidate) => microphoneMuteSettlementIsCurrent(candidate, activeMicrophoneTrack),
      );
      if (!committed) return;
      userMicrophoneMuted = requested;
      renderMicrophoneMute();
      addMarker(requested ? "microphone_muted" : "microphone_unmuted");
    });
  } catch {
    addMarker("microphone_mute_failed");
  } finally {
    microphoneMutePending = false;
    setInteractive(controller.state === "connected");
  }
}

function renderModelSelection(
  catalog: ModelCatalog,
  selectedModel = catalog.selectedModel,
  selectedEffort = catalog.selectedEffort,
): void {
  modelSelect.replaceChildren(
    ...catalog.models.map((model) => {
      const option = document.createElement("option");
      option.value = model.model;
      option.textContent = model.displayName;
      option.title = model.description;
      return option;
    }),
  );
  modelSelect.value = selectedModel;
  const model = catalog.models.find((item) => item.model === selectedModel);
  if (model === undefined) throw new Error("selected model is absent from catalog");
  effortSelect.replaceChildren(
    ...model.supportedEfforts.map((effort) => {
      const option = document.createElement("option");
      option.value = effort;
      option.textContent = reasoningEffortLabel(effort);
      return option;
    }),
  );
  effortSelect.value = selectedEffort;
}

async function loadModelConfiguration(): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null) return;
  const response = await fetch("/api/v1/models", {
    method: "POST",
    headers: authorization(activeCredential.token),
    body: null,
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  });
  if (!response.ok) {
    selectableModels = null;
    modelSelectionState.textContent = "Live selection is unavailable for this provider.";
    modelSelectionState.dataset.state = "error";
    setInteractive(controller.state === "connected");
    addMarker("model_catalog_unavailable");
    return;
  }
  selectableModels = parseModelCatalog(await response.json());
  renderModelSelection(selectableModels);
  modelSelectionState.textContent =
    "Selected-model Codex options. Ultra is the highest provider effort.";
  modelSelectionState.dataset.state = "ready";
  setInteractive(controller.state === "connected");
  addMarker(`model_catalog_loaded_${selectableModels.models.length}`);
}

async function changeModelConfiguration(model: string, effort: string): Promise<void> {
  const activeCredential = credential;
  const previous = selectableModels;
  if (activeCredential === null || previous === null || modelSelectionPending) return;
  const requested = modelSelectionFor(previous, model, effort);
  renderModelSelection(previous, requested.model, requested.effort);
  modelSelectionPending = true;
  modelSelectionState.textContent = "Applying after the active turn…";
  delete modelSelectionState.dataset.state;
  setInteractive(controller.state === "connected");
  try {
    const response = await fetch("/api/v1/model", {
      method: "POST",
      headers: {
        ...authorization(activeCredential.token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requested),
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
    });
    if (!response.ok) throw new Error("model selection rejected");
    selectableModels = parseModelCatalog(await response.json());
    renderModelSelection(selectableModels);
    modelSelectionState.textContent = "Accepted. Applies to the next turn.";
    modelSelectionState.dataset.state = "ready";
    addMarker(`model_changed_${selectableModels.selectedModel}_${selectableModels.selectedEffort}`);
  } catch {
    selectableModels = previous;
    renderModelSelection(previous);
    modelSelectionState.textContent = "Selection failed; previous settings retained.";
    modelSelectionState.dataset.state = "error";
    addMarker("model_change_failed");
  } finally {
    modelSelectionPending = false;
    setInteractive(controller.state === "connected");
  }
}

async function loadVoiceConfiguration(): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null) return;
  const response = await fetch("/api/v1/voices", {
    method: "POST",
    headers: authorization(activeCredential.token),
    body: null,
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  });
  if (!response.ok) throw new Error("voice configuration rejected");
  const configuration = parseVoiceConfiguration(await response.json());
  voiceSelect.replaceChildren();
  for (const voice of configuration.voices) {
    const option = document.createElement("option");
    option.value = voice;
    option.textContent = voice;
    voiceSelect.append(option);
  }
  selectedVoice = configuration.selectedVoice;
  if (selectedVoice !== null) voiceSelect.value = selectedVoice;
  voiceSelect.disabled = selectedVoice === null || controller.state !== "connected";
  addMarker(`voice_catalog_loaded_${configuration.voices.length}`);
}

async function changeVoice(voice: string): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null || selectedVoice === null) return;
  const previous = selectedVoice;
  voiceSelect.disabled = true;
  try {
    const response = await fetch("/api/v1/voice", {
      method: "POST",
      headers: {
        ...authorization(activeCredential.token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ voice }),
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
    });
    if (!response.ok) throw new Error("voice selection rejected");
    selectedVoice = voice;
    addMarker(`voice_changed_${voice}`);
  } catch {
    voiceSelect.value = previous;
    addMarker("voice_change_failed");
  } finally {
    voiceSelect.disabled = controller.state !== "connected";
  }
}

function renderConnectionPresentation(state = controller.state): void {
  const presentation = connectionPresentation(state, microphoneReady, voicePathState);
  stateOutput.textContent = presentation.label;
  connectionStatus.dataset.state = presentation.presentationState;
  readinessHeadline.textContent = presentation.headline;
  readinessDetail.textContent = presentation.detail;
}

function clearVoicePathReadinessTimer(): void {
  if (voicePathReadinessTimer === null) return;
  window.clearTimeout(voicePathReadinessTimer);
  voicePathReadinessTimer = null;
}

function waitForServerVoicePath(): void {
  clearVoicePathReadinessTimer();
  if (microphoneReady !== true || voicePathState === "ready") return;
  voicePathState = "waiting";
  renderConnectionPresentation();
  voicePathReadinessTimer = window.setTimeout(() => {
    voicePathReadinessTimer = null;
    if (microphoneReady !== true || voicePathState !== "waiting") return;
    voicePathState = "timed-out";
    renderConnectionPresentation();
    addMarker("voice_input_server_timeout");
  }, voicePathReadinessTimeoutMs);
}

function confirmServerVoicePath(generation: unknown, incarnation: unknown): void {
  if (
    !Number.isSafeInteger(generation) ||
    !Number.isSafeInteger(incarnation) ||
    !microphoneReadinessAuthority.accepts(
      generation as number,
      incarnation as number,
      room,
      activeMicrophoneTrack,
      controller.state,
    )
  ) {
    addMarker("voice_input_server_stale");
    return;
  }
  if (voicePathState === "ready") return;
  clearVoicePathReadinessTimer();
  voicePathState = "ready";
  renderConnectionPresentation();
  addMarker("voice_input_server_confirmed");
}

function canStartSession(): boolean {
  return stableLaunch || capability !== null || credential !== null;
}

function renderSessionToggle(state = controller.state): void {
  const toggle = sessionTogglePresentation(
    state,
    canStartSession(),
    remoteStopRequired,
  );
  sessionToggleButton.textContent = toggle.label;
  sessionToggleButton.disabled = toggle.disabled;
  sessionToggleButton.classList.toggle("primary", toggle.action === "connect");
  sessionToggleButton.classList.toggle("danger", toggle.action === "stop");
}

controller.subscribe((state) => {
  renderConnectionPresentation(state);
  renderSessionToggle(state);
  setInteractive(state === "connected");
});

async function loadMicrophones(): Promise<void> {
  const enumerationGeneration = microphoneEnumerationAuthority.issue();
  let devices: MediaDeviceInfo[];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch {
    if (microphoneEnumerationAuthority.owns(enumerationGeneration)) {
      addMarker("microphone_enumeration_failed");
    }
    return;
  }
  if (!microphoneEnumerationAuthority.owns(enumerationGeneration)) return;
  const selected = microphoneSelect.value;
  const options = microphoneDeviceChoices(devices).map((device) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label;
    return option;
  });
  const systemDefault = document.createElement("option");
  systemDefault.value = "";
  systemDefault.textContent = "System default";
  microphoneSelect.replaceChildren(systemDefault, ...options);
  microphoneSelect.value = options.some((option) => option.value === selected) ? selected : "";
}

function authorization(token: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
  };
}

class SessionRebindRejected extends Error {
  constructor(readonly status: number) {
    super("session rebind rejected");
  }
}

async function connectionFetch(
  input: RequestInfo | URL,
  init: RequestInit,
  parentSignal: AbortSignal,
): Promise<Response> {
  const request = new AbortController();
  const abort = (): void => request.abort();
  if (parentSignal.aborted) abort();
  else parentSignal.addEventListener("abort", abort, { once: true });
  const timeout = window.setTimeout(abort, connectionRequestTimeoutMs);
  try {
    return await fetch(input, { ...init, signal: request.signal });
  } finally {
    window.clearTimeout(timeout);
    parentSignal.removeEventListener("abort", abort);
  }
}

async function bootstrap(signal: AbortSignal): Promise<BootstrapCredential> {
  const request = bootstrapRequestParameters(stableLaunch, capability);
  const response = await connectionFetch(request.path, {
    method: "POST",
    headers: request.bearer === null ? undefined : authorization(request.bearer),
    body: null,
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  }, signal);
  capability = null;
  if (!response.ok) {
    throw new Error("bootstrap denied");
  }
  return parseBootstrapCredential(await response.json());
}

function clearCredentialRefresh(): void {
  if (credentialRefreshTimer !== null) {
    window.clearTimeout(credentialRefreshTimer);
    credentialRefreshTimer = null;
  }
}

function scheduleCredentialRefresh(activeCredential: BootstrapCredential): void {
  clearCredentialRefresh();
  const delayMs = Math.max(10_000, activeCredential.expiresInSeconds * 500);
  credentialRefreshTimer = window.setTimeout(() => {
    void refreshCredential().catch(() => {
      if (!canCommitCredentialRefresh(activeCredential, credential, controller.state)) return;
      addMarker("credential_refresh_failed");
      void disconnectLocal();
    });
  }, delayMs);
}

function nextMediaIncarnation(): number {
  if (mediaIncarnation >= Number.MAX_SAFE_INTEGER) {
    throw new Error("media incarnation authority is exhausted");
  }
  mediaIncarnation += 1;
  return mediaIncarnation;
}

async function announceMediaActivation(
  activeCredential: BootstrapCredential,
  incarnation: number,
  signal?: AbortSignal,
): Promise<void> {
  const request = new AbortController();
  const abort = (): void => request.abort();
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  const timeout = window.setTimeout(abort, stopRequestTimeoutMs);
  try {
    const response = await fetch("/api/v1/media", {
      method: "POST",
      headers: {
        ...authorization(activeCredential.token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ mediaIncarnation: incarnation }),
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
      signal: request.signal,
    });
    if (!response.ok) throw new Error("media activation rejected");
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

async function refreshCredential(): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null) return;
  const response = await fetch("/api/v1/refresh", {
    method: "POST",
    headers: authorization(activeCredential.token),
    body: null,
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  });
  if (!response.ok) throw new Error("credential refresh rejected");
  const refreshed = parseBootstrapCredential(await response.json());
  if (!canCommitCredentialRefresh(activeCredential, credential, controller.state)) return;
  if (
    refreshed.participantIdentity !== activeCredential.participantIdentity ||
    refreshed.workerIdentity !== activeCredential.workerIdentity ||
    refreshed.roomName !== activeCredential.roomName ||
    refreshed.url !== activeCredential.url
  ) {
    throw new Error("credential refresh changed session authority");
  }
  credential = refreshed;
  addMarker("credential_refreshed");
  scheduleCredentialRefresh(refreshed);
}

async function rebindBrowserCredential(
  activeCredential: BootstrapCredential,
  signal: AbortSignal,
  requestId: string,
): Promise<BootstrapCredential> {
  const request = rebindRequestParameters(stableLaunch, activeCredential, requestId);
  const headers =
    request.bearer === null
      ? { "Content-Type": "application/json" }
      : {
          ...authorization(request.bearer),
          "Content-Type": "application/json",
        };
  const response = await connectionFetch(request.path, {
    method: "POST",
    headers,
    body: request.body,
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  }, signal);
  if (!response.ok) throw new SessionRebindRejected(response.status);
  return reboundCredential(
    activeCredential,
    parseBootstrapCredential(await response.json()),
  );
}

async function projectionResyncBrowserCredential(
  activeCredential: BootstrapCredential,
  signal: AbortSignal,
): Promise<BootstrapCredential> {
  if (projectionResyncAttempted) {
    throw new Error("public event projection resync was already attempted");
  }
  projectionResyncAttempted = true;
  const response = await connectionFetch("/api/v1/projection-resync", {
    method: "POST",
    headers: authorization(activeCredential.token),
    body: null,
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  }, signal);
  if (!response.ok) throw new Error("public event projection resync rejected");
  const replacement = parseBootstrapCredential(await response.json());
  if (replacement.participantIdentity === activeCredential.participantIdentity) {
    throw new Error("public event projection resync did not rotate credential authority");
  }
  return replacement;
}

async function recoverProjectionResync(signal: AbortSignal): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null) throw new Error("active credential is unavailable");
  const observedStopVersion = sessionStopVersion;
  const replacement = await projectionResyncBrowserCredential(activeCredential, signal);
  if (signal.aborted || sessionStopVersion !== observedStopVersion) return;
  credential = replacement;
  pendingRebindRequestId = null;
  clearCredentialRefresh();
  eventSequence = 0;
  resetSessionInputAuthority();
  if (eventPolling?.signal === signal) eventPolling = null;
  addMarker("event_projection_resynced");
  await disconnectLocal();
  if (
    signal.aborted ||
    sessionStopVersion !== observedStopVersion ||
    controller.state !== "disconnected"
  ) {
    return;
  }
  await connect(true);
}

function resetSessionInputAuthority(): void {
  inputSequence = 0;
  microphoneProcessingEvidence = null;
  approvalController.reset();
  evidenceControls.reset();
  searchEgressControls.reset();
}

async function submitAudioDiagnostic(
  activeRoom: Room,
  activeTrack: RemoteTrack,
  render: RenderTimingRecord,
): Promise<void> {
  await audioDiagnosticQueue.run(async () => {
    const activeCredential = credential;
    const capture = microphoneProcessingEvidence;
    if (
      activeCredential === null ||
      capture === null ||
      activeMicrophoneTrack !== capture.track ||
      room !== activeRoom ||
      activeRemoteTrack !== activeTrack ||
      remoteAudio.dataset.streamId !== render.streamId
    ) {
      return;
    }
    const request: RequestInit = {
      method: "POST",
      headers: {
        ...authorization(activeCredential.token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        version: 1,
        streamId: render.streamId,
        supported: capture.value.supported,
        applied: capture.value.applied,
        render: {
          subscribedToAttachMs: render.subscribedToAttachMs,
          attachToPlayingMs: render.attachToPlayingMs,
          playingToAdvanceMs: render.playingToAdvanceMs,
        },
      }),
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
    };
    let accepted = false;
    for (let attempt = 0; attempt < 2 && !accepted; attempt += 1) {
      try {
        accepted = (await fetch("/api/v1/audio-diagnostic", request)).ok;
      } catch {
        if (attempt === 1) throw new Error("audio diagnostic delivery failed");
      }
    }
    if (!accepted) throw new Error("audio diagnostic rejected");
  });
}

function bindRoomEvents(activeRoom: Room): void {
  activeRoom.on(RoomEvent.ConnectionStateChanged, (state: ConnectionState) => {
    if (room !== activeRoom) return;
    if (terminalDisconnectIsAuthoritative(state, admittedRooms.has(activeRoom))) {
      addMarker("media_terminally_disconnected");
      void recoverTerminalMediaDisconnect();
      return;
    }
    if (
      state === ConnectionState.Reconnecting &&
      (controller.state === "connected" || controller.state === "preparing")
    ) {
      microphoneVerificationGeneration += 1;
      voicePathState = "waiting";
      clearVoicePathReadinessTimer();
      microphoneReadinessAuthority.invalidate();
      microphoneEnumerationAuthority.invalidate();
      microphoneReady = null;
      controller.reconnecting();
      addMarker("media_reconnecting");
    } else if (
      state === ConnectionState.Connected &&
      controller.state === "reconnecting" &&
      connectionAttempt === null
    ) {
      const verificationGeneration = microphoneVerificationGeneration;
      const observedStopVersion = sessionStopVersion;
      microphoneReady = null;
      controller.preparingMicrophone();
      void reconnectMicrophoneQueue
        .run(() => reverifyMicrophoneAfterNativeReconnect(activeRoom, verificationGeneration))
        .catch(() => {
          addMarker("microphone_reverification_failed");
          if (
            sessionStopVersion === observedStopVersion &&
            connectionAttempt === null &&
            canCommitMicrophoneVerification(
              verificationGeneration,
              microphoneVerificationGeneration,
              activeRoom,
              room,
              controller.state,
            )
          ) {
            void disconnectLocal();
          }
        });
    }
  });
  activeRoom.on(
    RoomEvent.TrackSubscribed,
    (
      track: RemoteTrack,
      publication: RemoteTrackPublication,
      participant: RemoteParticipant,
    ) => {
      const activeCredential = credential;
      if (
        room !== activeRoom ||
        activeCredential === null ||
        !authorizeRemoteAudio(
          activeCredential.workerIdentity,
          participant.identity,
          publication.source === Track.Source.Microphone,
          track.kind === Track.Kind.Audio,
        )
      ) {
        return;
      }
      const streamId = publication.trackName;
      if (typeof streamId !== "string" || streamId.length < 1 || streamId.length > 128) return;
      let renderTiming: RenderTimingTelemetry | null = null;
      try {
        renderTiming = new RenderTimingTelemetry(streamId, performance.now());
      } catch {
        addMarker("audio_diagnostic_stream_unsupported");
      }
      stopKaraoke();
      if (activeRemoteTrack !== null && activeRemoteTrack !== track) {
        setSpeechRendererVolume(1);
      }
      if (activeRemoteTrack !== null && activeRemoteTrack !== track) {
        activeRemoteTrack = replaceRemoteAudio(activeRemoteTrack, track, remoteAudio);
      } else {
        activeRemoteTrack = track;
      }
      remoteAudio.dataset.streamId = streamId;
      reconcileProvisionalSpeechAuthority(speechTimings.get(streamId) ?? null);
      setSpeechRendererVolume(
        locallySilencedSpeech !== null
          ? 0
          : provisionalSpeechYieldController.attenuated
            ? provisionalSpeechVolume
            : 1,
      );
      streamMediaStarts.attach(streamId, remoteAudio.currentTime);
      let playbackPlaying = false;
      let firstAdvanceAt: number | null = null;
      const maybeSubmitTiming = (): void => {
        if (renderTiming === null || !playbackPlaying || firstAdvanceAt === null) return;
        try {
          const record = renderTiming.advanced(firstAdvanceAt);
          if (record === null) return;
          void submitAudioDiagnostic(activeRoom, track, record).catch(() => {
            addMarker("audio_diagnostic_failed");
          });
        } catch {
          renderTiming = null;
          addMarker("audio_diagnostic_timing_invalid");
        }
      };
      const playbackStarted = (): void => {
        if (remoteAudio.dataset.streamId !== streamId) return;
        streamMediaStarts.resolvePlaying(streamId, remoteAudio.currentTime);
        try {
          renderTiming?.playing(performance.now());
        } catch {
          renderTiming = null;
          addMarker("audio_diagnostic_timing_invalid");
        }
        playbackPlaying = true;
        if (microphoneSignalMonitor.isActive) beginProvisionalSpeechYield();
        maybeSubmitTiming();
        maybeStartKaraoke(streamId);
      };
      const observePlayback = (): void => {
        if (remoteAudio.dataset.streamId !== streamId) return;
        if (streamMediaStarts.observeStream(streamId, remoteAudio.currentTime) === undefined) {
          window.requestAnimationFrame(observePlayback);
          return;
        }
        firstAdvanceAt = performance.now();
        maybeSubmitTiming();
        maybeStartKaraoke(streamId);
      };
      remoteAudio.addEventListener("playing", playbackStarted, { once: true });
      track.attach(remoteAudio);
      setSpeechRendererEnabled(locallySilencedSpeech === null);
      setSpeechRendererVolume(
        locallySilencedSpeech !== null
          ? 0
          : provisionalSpeechYieldController.attenuated
            ? provisionalSpeechVolume
            : 1,
      );
      try {
        renderTiming?.attached(performance.now());
      } catch {
        renderTiming = null;
        addMarker("audio_diagnostic_timing_invalid");
      }
      window.requestAnimationFrame(observePlayback);
    },
  );
  activeRoom.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
    if (!remoteAudioTrackIsCurrent(activeRoom, room, track, activeRemoteTrack)) return;
    activeRemoteTrack = releaseRemoteAudio(track, remoteAudio);
    resetProvisionalSpeechYield();
    speechYieldController.reset();
    locallySilencedSpeech = null;
    speechYieldPending = false;
    renderSpeechControl("idle");
    stopKaraoke();
  });
}

async function cleanupMicrophoneTrack(
  activeRoom: Room,
  localTrack: LocalTrack,
): Promise<void> {
  if (microphoneProcessingEvidence?.track === localTrack) {
    microphoneProcessingEvidence = null;
  }
  try {
    localTrack.stop();
  } catch {
    addMarker("microphone_stop_failed");
  }
  try {
    await activeRoom.localParticipant.unpublishTrack(localTrack);
  } catch {
    addMarker("microphone_unpublish_failed");
  }
}

async function enableSelectedMicrophone(
  activeRoom: Room,
  incarnation: number,
): Promise<LocalTrack | null> {
  const selected = microphoneSelect.value;
  let localTrack: LocalTrack | undefined;
  try {
    const publication = await activeRoom.localParticipant.setMicrophoneEnabled(
      true,
      microphoneCaptureOptions(selected),
      { name: `microphone-${incarnation}` },
    );
    localTrack = publication?.track;
    const settings = localTrack?.getSourceTrackSettings();
    let processingEvidence: MicrophoneProcessingTelemetry | null = null;
    if (settings !== undefined) {
      try {
        processingEvidence = microphoneProcessingTelemetry(
          navigator.mediaDevices.getSupportedConstraints(),
          settings,
        );
      } catch {
        addMarker("microphone_processing_diagnostic_unavailable");
      }
    }
    if (
      localTrack === undefined ||
      settings === undefined ||
      microphoneProcessingDisposition(settings) !== "aec-only"
    ) {
      throw new Error("browser did not apply required AEC-only microphone processing");
    }
    if (processingEvidence !== null) {
      microphoneProcessingEvidence = { track: localTrack, value: processingEvidence };
    }
    addMarker("microphone_aec_only_applied");
    addMarker("microphone_published");
    if (userMicrophoneMuted) await localTrack.mute();
    void loadMicrophones();
    return localTrack;
  } catch {
    if (localTrack !== undefined) await cleanupMicrophoneTrack(activeRoom, localTrack);
    addMarker("microphone_unavailable_typed_only");
    return null;
  }
}

async function reverifyMicrophoneAfterNativeReconnect(
  activeRoom: Room,
  verificationGeneration: number,
): Promise<void> {
  if (
    !canCommitMicrophoneVerification(
      verificationGeneration,
      microphoneVerificationGeneration,
      activeRoom,
      room,
      controller.state,
    )
  ) {
    return;
  }
  const activeCredential = credential;
  if (activeCredential === null) throw new Error("active credential is unavailable");
  const existingMicrophone = activeMicrophoneTrack;
  const existingMicrophoneIsLive =
    existingMicrophone !== null && existingMicrophone.mediaStreamTrack.readyState === "live";
  if (existingMicrophone !== null && !existingMicrophoneIsLive) {
    activeMicrophoneTrack = null;
    microphoneReadinessAuthority.invalidate();
    microphoneSignalMonitor.stop();
  }
  let incarnation = mediaIncarnation;
  const activation = await settleReconnectedMicrophone(
    existingMicrophone,
    () => existingMicrophoneIsLive,
    async () => {
      incarnation = nextMediaIncarnation();
      return enableSelectedMicrophone(activeRoom, incarnation);
    },
    (candidate) => cleanupMicrophoneTrack(activeRoom, candidate),
  );
  const reusedMicrophone = activation !== null && activation === existingMicrophone;
  const settled = await settleMicrophoneVerification(
    canCommitMicrophoneVerification(
      verificationGeneration,
      microphoneVerificationGeneration,
      activeRoom,
      room,
      controller.state,
    ),
    activation,
    async (staleTrack) => {
      if (staleTrack !== null) await cleanupMicrophoneTrack(activeRoom, staleTrack);
    },
  );
  if (!settled.committed) return;
  const authoritativeTrack =
    reusedMicrophone && settled.value !== null
      ? await enforceMicrophoneMuteAuthority(
          settled.value,
          userMicrophoneMuted,
          async (track, shouldMute) => {
            if (shouldMute) await track.mute();
            else await track.unmute();
          },
          (track) => cleanupMicrophoneTrack(activeRoom, track),
        )
      : settled.value;
  const selectedMicrophoneReady = authoritativeTrack !== null;
  microphoneReady = selectedMicrophoneReady;
  if (authoritativeTrack !== null) {
    activeMicrophoneTrack = authoritativeTrack;
    microphoneReadinessAuthority.activate(activeRoom, authoritativeTrack, incarnation);
    attachMicrophoneSignalMonitor(authoritativeTrack);
    renderMicrophoneMute();
  } else {
    activeMicrophoneTrack = null;
    microphoneReadinessAuthority.invalidate();
    microphoneSignalMonitor.stop();
    clearVoicePathReadinessTimer();
  }
  const mediaCommitted = await settleMediaActivation(
    announceMediaActivation(activeCredential, incarnation),
    () =>
      canCommitMicrophoneVerification(
        verificationGeneration,
        microphoneVerificationGeneration,
        activeRoom,
        room,
        controller.state,
      ),
    async () => {
      if (authoritativeTrack !== null) {
        await cleanupMicrophoneTrack(activeRoom, authoritativeTrack);
      }
    },
  );
  if (!mediaCommitted) return;
  controller.connected();
  if (authoritativeTrack !== null) waitForServerVoicePath();
  addMarker("media_reconnected");
  addMarker(selectedMicrophoneReady ? "voice_input_published" : "typed_input_ready");
}

async function connect(projectionResync = false): Promise<void> {
  clearPartialTranscript();
  prepareMicrophoneSignalMonitor();
  const startingState = controller.state;
  const disconnectedResume = startingState === "disconnected";
  const errorResume = startingState === "error" && credential !== null;
  const resuming = disconnectedResume || errorResume;
  if (!resuming && credential === null && !stableLaunch && capability === null) {
    addMarker("fresh_launch_required");
    return;
  }
  if (disconnectedResume) {
    controller.beginReconnect();
  } else {
    controller.beginConnect();
    if (errorResume) controller.bootstrapReady();
  }
  if (errorResume) {
    microphoneVerificationGeneration += 1;
    microphoneReadinessAuthority.invalidate();
    activeMicrophoneTrack = null;
    microphoneEnumerationAuthority.invalidate();
    connectionAttempt?.cancel();
    connectionAttempt = null;
    operation?.abort();
    eventPolling?.abort();
    eventPolling = null;
    microphoneSignalMonitor.stop();
    const residualRoom = room;
    activeRemoteTrack = releaseRemoteAudioBeforeRoomInvalidation(
      activeRemoteTrack,
      remoteAudio,
      stopKaraoke,
      () => {
        room = null;
      },
    );
    resetProvisionalSpeechYield();
    speechYieldController.reset();
    locallySilencedSpeech = null;
    speechYieldPending = false;
    renderSpeechControl("idle");
    await releaseLocalMedia(residualRoom);
    try {
      const disconnectOperation = residualRoom?.disconnect();
      void Promise.resolve(disconnectOperation).catch(() =>
        addMarker("media_room_disconnect_failed"),
      );
    } catch {
      addMarker("media_room_disconnect_failed");
    }
  }
  const localOperation = new AbortController();
  operation = localOperation;
  const started = performance.now();
  let attemptedRoom: Room | null = null;
  let attempt: ConnectionAttempt<Room> | null = null;
  let sessionReplaced = false;
  try {
    if (resuming) {
      const previousCredential = credential;
      if (previousCredential === null) throw new Error("resume credential is unavailable");
      if (projectionResync) {
        pendingRebindRequestId = null;
        sessionReplaced = true;
        addMarker("event_projection_reconnected", started);
      } else {
        const rebindRequestId =
          pendingRebindRequestId ?? `rebind_${crypto.randomUUID()}`;
        pendingRebindRequestId = rebindRequestId;
        clearCredentialRefresh();
        eventPolling?.abort();
        eventPolling = null;
        try {
          credential = await rebindBrowserCredential(
            previousCredential,
            localOperation.signal,
            rebindRequestId,
          );
          addMarker("session_rebound", started);
        } catch (error) {
          if (localOperation.signal.aborted) throw error;
          try {
            credential = await rebindBrowserCredential(
              previousCredential,
              localOperation.signal,
              rebindRequestId,
            );
            addMarker("session_rebound_after_retry", started);
          } catch (retryError) {
            if (localOperation.signal.aborted) throw retryError;
            if (!stableLaunch) throw retryError;
            if (
              !(retryError instanceof SessionRebindRejected) ||
              !rebindFailureAllowsFreshBootstrap(retryError.status)
            ) {
              throw retryError;
            }
            credential = await bootstrap(localOperation.signal);
            sessionReplaced = true;
            resetSessionInputAuthority();
            addMarker("session_replaced", started);
          }
        }
        if (!sessionReplaced) resetSessionInputAuthority();
        pendingRebindRequestId = null;
      }
    } else {
      credential = await bootstrap(localOperation.signal);
      pendingRebindRequestId = null;
      resetSessionInputAuthority();
      addMarker("bootstrap_complete", started);
      controller.bootstrapReady();
    }
    const activeCredential = credential;
    if (activeCredential === null) throw new Error("active credential is unavailable");
    const activeRoom = new Room({ adaptiveStream: true, dynacast: true, webAudioMix: true });
    attemptedRoom = activeRoom;
    attempt = new ConnectionAttempt<Room>((obsoleteRoom) => obsoleteRoom.disconnect());
    connectionAttempt = attempt;
    room = activeRoom;
    microphoneEnumerationAuthority.invalidate();
    bindRoomEvents(activeRoom);
    await activeRoom.connect(activeCredential.url, activeCredential.token, {
      autoSubscribe: true,
    });
    if (!attempt.admit(activeRoom, room)) return;
    admittedRooms.add(activeRoom);
    microphoneReady = null;
    voicePathState = "waiting";
    clearVoicePathReadinessTimer();
    if (!resuming || sessionReplaced) {
      microphoneReadinessAuthority.resetSession();
    }
    controller.preparingMicrophone();
    const verificationGeneration = microphoneVerificationGeneration;
    addMarker(
      sessionReplaced ? "room_replaced" : resuming ? "room_reconnected" : "room_connected",
      started,
    );
    scheduleCredentialRefresh(activeCredential);
    const incarnation = nextMediaIncarnation();
    const activation = await enableSelectedMicrophone(activeRoom, incarnation);
    const settled = await settleMicrophoneVerification(
      attempt.owns(connectionAttempt) &&
        canCommitMicrophoneVerification(
          verificationGeneration,
          microphoneVerificationGeneration,
          activeRoom,
          room,
          controller.state,
        ),
      activation,
      async (staleTrack) => {
        if (staleTrack !== null) await cleanupMicrophoneTrack(activeRoom, staleTrack);
      },
    );
    if (!settled.committed) return;
    const selectedMicrophoneReady = settled.value !== null;
    microphoneReady = selectedMicrophoneReady;
    if (settled.value !== null) {
      activeMicrophoneTrack = settled.value;
      microphoneReadinessAuthority.activate(activeRoom, settled.value, incarnation);
      attachMicrophoneSignalMonitor(settled.value);
      renderMicrophoneMute();
    } else {
      activeMicrophoneTrack = null;
      microphoneReadinessAuthority.invalidate();
      microphoneSignalMonitor.stop();
      clearVoicePathReadinessTimer();
    }
    const mediaCommitted = await settleMediaActivation(
      announceMediaActivation(activeCredential, incarnation, localOperation.signal),
      () =>
        attempt !== null &&
        attempt.owns(connectionAttempt) &&
        canCommitMicrophoneVerification(
          verificationGeneration,
          microphoneVerificationGeneration,
          activeRoom,
          room,
          controller.state,
        ),
      async () => {
        if (settled.value !== null) await cleanupMicrophoneTrack(activeRoom, settled.value);
      },
    );
    if (!mediaCommitted) return;
    controller.connected();
    if (settled.value !== null) waitForServerVoicePath();
    addMarker(selectedMicrophoneReady ? "voice_input_published" : "typed_input_ready", started);
    if (eventPolling === null) {
      if (!resuming || sessionReplaced) eventSequence = 0;
      eventPollFailurePolicy.recordSuccess();
      eventPolling = new AbortController();
      void pollPublicEvents(eventPolling.signal).catch(() => {
        if (!eventPolling?.signal.aborted) addMarker("event_projection_failed");
      });
    }
    await loadVoiceConfiguration();
    await loadModelConfiguration();
  } catch {
    if (attempt !== null && !attempt.owns(connectionAttempt)) {
      await closeLocalRoom(attemptedRoom);
      return;
    }
    if (controller.state === "stopping") return;
    const observedStopVersion = sessionStopVersion;
    const detachedAttemptedRoom = room === attemptedRoom ? null : attemptedRoom;
    await disconnectLocal();
    if (detachedAttemptedRoom !== null) await closeLocalRoom(detachedAttemptedRoom);
    if (sessionStopVersion !== observedStopVersion) return;
    if (resuming) {
      controller.disconnected();
      addMarker("media_reconnect_failed", started);
      return;
    }
    clearCredentialRefresh();
    eventPolling?.abort();
    eventPolling = null;
    credential = null;
    capability = null;
    controller.failed();
    addMarker("connection_failed", started);
  } finally {
    if (attempt !== null && attempt.owns(connectionAttempt)) connectionAttempt = null;
    if (operation === localOperation) operation = null;
  }
}

function stopLocalMediaCapture(track: LocalTrack | null): void {
  if (track === null) return;
  try {
    track.stop();
  } catch {
    addMarker("media_track_stop_failed");
  }
  try {
    track.mediaStreamTrack.stop();
  } catch {
    addMarker("media_source_stop_failed");
  }
}

async function releaseLocalMedia(
  activeRoom: Room | null,
  authoritativeTrack: LocalTrack | null = null,
): Promise<void> {
  stopLocalMediaCapture(authoritativeTrack);
  if (activeRoom === null) return;
  const unpublishOperations: Promise<void>[] = [];
  for (const publication of activeRoom.localParticipant.trackPublications.values()) {
    const track = publication.track;
    if (track === undefined) continue;
    stopLocalMediaCapture(track);
    unpublishOperations.push(
      activeRoom.localParticipant.unpublishTrack(track).then(
        () => undefined,
        () => addMarker("media_unpublish_failed"),
      ),
    );
  }
  await Promise.all(unpublishOperations);
}

async function closeLocalRoom(
  activeRoom: Room | null,
  authoritativeTrack: LocalTrack | null = null,
): Promise<void> {
  const cleanupOperations: Promise<void>[] = [
    releaseLocalMedia(activeRoom, authoritativeTrack),
  ];
  if (activeRoom !== null) {
    try {
      cleanupOperations.push(
        Promise.resolve(activeRoom.disconnect()).catch(() =>
          addMarker("media_room_disconnect_failed"),
        ),
      );
    } catch {
      addMarker("media_room_disconnect_failed");
    }
  }
  let releaseTimeout: number | null = null;
  const timeout = new Promise<void>((resolve) => {
    releaseTimeout = window.setTimeout(resolve, localMediaReleaseTimeoutMs);
  });
  const released = Promise.all(cleanupOperations).then(() => undefined);
  if ((await Promise.race([released.then(() => true), timeout.then(() => false)])) === false) {
    addMarker("media_release_timeout");
  }
  if (releaseTimeout !== null) window.clearTimeout(releaseTimeout);
}

async function disconnectLocal(): Promise<void> {
  microphoneVerificationGeneration += 1;
  voicePathState = "waiting";
  clearVoicePathReadinessTimer();
  microphoneReadinessAuthority.invalidate();
  const authoritativeTrack = activeMicrophoneTrack;
  activeMicrophoneTrack = null;
  microphoneEnumerationAuthority.invalidate();
  connectionAttempt?.cancel();
  connectionAttempt = null;
  operation?.abort();
  microphoneSignalMonitor.stop();
  const activeRoom = room;
  activeRemoteTrack = releaseRemoteAudioBeforeRoomInvalidation(
    activeRemoteTrack,
    remoteAudio,
    stopKaraoke,
    () => {
      room = null;
    },
  );
  resetProvisionalSpeechYield();
  speechYieldController.reset();
  locallySilencedSpeech = null;
  speechYieldPending = false;
  renderSpeechControl("idle");
  await closeLocalRoom(activeRoom, authoritativeTrack);
  if (["connecting", "preparing", "connected", "reconnecting"].includes(controller.state)) {
    controller.disconnected();
  }
  microphoneReady = null;
  addMarker("media_disconnected");
}

async function recoverTerminalMediaDisconnect(): Promise<void> {
  const observedStopVersion = sessionStopVersion;
  await disconnectLocal();
  if (
    sessionStopVersion !== observedStopVersion ||
    credential === null ||
    controller.state !== "disconnected"
  ) return;
  addMarker("media_terminal_rebind_started");
  await connect();
}

async function stop(): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null) return;
  const rebindRequestId = pendingRebindRequestId;
  controller.beginStop();
  sessionStopVersion += 1;
  remoteStopRequired = true;
  eventPolling?.abort();
  eventPolling = null;
  await disconnectLocal();
  const stopRequest = new AbortController();
  const stopTimeout = window.setTimeout(() => stopRequest.abort(), stopRequestTimeoutMs);
  try {
    const response = await fetch("/api/v1/stop", {
      method: "POST",
      headers:
        rebindRequestId === null
          ? authorization(activeCredential.token)
          : {
              ...authorization(activeCredential.token),
              "Content-Type": "application/json",
            },
      body: rebindRequestId === null ? null : JSON.stringify({ requestId: rebindRequestId }),
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
      signal: stopRequest.signal,
    });
    if (!response.ok) throw new Error("session stop was rejected");
  } catch (error) {
    controller.stopFailed(false);
    addMarker("session_stop_failed");
    throw error;
  } finally {
    window.clearTimeout(stopTimeout);
  }
  remoteStopRequired = false;
  pendingRebindRequestId = null;
  interruptActiveAssistantTurn();
  clearCredentialRefresh();
  microphoneSignalMonitor.stop();
  credential = null;
  capability = null;
  microphoneReady = null;
  userMicrophoneMuted = false;
  microphoneMutePending = false;
  renderMicrophoneMute();
  typedInput.value = "";
  for (const view of approvalCardViews.values()) {
    view.approve.disabled = true;
    view.reject.disabled = true;
    view.status.textContent = "Session stopped before a decision.";
  }
  approvalCardViews.clear();
  taskCardViews.clear();
  speechFields.forEach((field, index) => {
    field.textContent = index % 2 === 0 ? "Awaiting session" : "—";
  });
  selectableModels = null;
  modelSelectionPending = false;
  microphoneReadinessAuthority.resetSession();
  modelSelect.replaceChildren(new Option("Unavailable", ""));
  effortSelect.replaceChildren(new Option("Unavailable", ""));
  modelSelectionState.textContent = "Session stopped.";
  delete modelSelectionState.dataset.state;
  evidenceControls.reset();
  searchEgressControls.reset();
  controller.stopped();
  addMarker("session_stopped");
}

async function submitTyped(text: string): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null) throw new Error("no active credential");
  const sequence = inputSequence + 1;
  const response = await fetch("/api/v1/input", {
    method: "POST",
    headers: {
      ...authorization(activeCredential.token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sequence, text }),
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  });
  if (!response.ok) throw new Error("typed input rejected");
  inputSequence = sequence;
  projectUserTranscript("typed-admission", text);
  addMarker("typed_input_admitted");
}

async function submitEvidenceControl(
  path: "/api/v1/evidence-consent" | "/api/v1/evidence-revoke",
  body: object,
): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null) throw new Error("no active credential");
  const response = await fetch(path, {
    method: "POST",
    headers: {
      ...authorization(activeCredential.token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  });
  if (!response.ok) {
    if (await isRetryableEvidenceControlTimeout(response, body)) {
      throw new EvidenceControlTimeoutError();
    }
    throw new Error("evidence control rejected");
  }
}

async function submitSearchEgressControl(
  path: "/api/v1/search-egress-consent" | "/api/v1/search-egress-revoke",
  body: object,
): Promise<void> {
  const activeCredential = credential;
  if (activeCredential === null) throw new Error("no active credential");
  const response = await fetch(path, {
    method: "POST",
    headers: {
      ...authorization(activeCredential.token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  });
  if (!response.ok) throw new Error("public search consent control rejected");
}

async function isRetryableEvidenceControlTimeout(
  response: Response,
  body: object,
): Promise<boolean> {
  if (response.status !== 503 || !hasExactEvidenceControlSequence(body)) return false;
  let text: string;
  try {
    text = await response.text();
  } catch {
    return false;
  }
  if (new TextEncoder().encode(text).byteLength > 512) return false;
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    return false;
  }
  if (
    typeof payload !== "object" ||
    payload === null ||
    Array.isArray(payload)
  ) {
    return false;
  }
  const record = payload as Record<string, unknown>;
  const keys = Object.keys(record);
  if (
    keys.length !== 3 ||
    !keys.every((key) => key === "captureState" || key === "error" || key === "sequence")
  ) {
    return false;
  }
  const captureState = record.captureState;
  return (
    record.error === "control_timeout" &&
    record.sequence === body.sequence &&
    typeof captureState === "string" &&
    ["idle", "active", "revoked_purging", "purge_failed", "faulted", "unavailable"].includes(
      captureState,
    )
  );
}

function hasExactEvidenceControlSequence(
  value: object,
): value is { readonly sequence: number } {
  if (!("sequence" in value)) return false;
  const sequence = (value as { readonly sequence: unknown }).sequence;
  return typeof sequence === "number" && Number.isSafeInteger(sequence) && sequence > 0;
}

async function submitApproval(
  approvalId: string,
  decision: ApprovalDecision,
): Promise<void> {
  const view = approvalCardViews.get(approvalId);
  if (view === undefined) {
    throw new Error("no approval is actionable");
  }
  view.approve.disabled = true;
  view.reject.disabled = true;
  view.status.textContent = "Submitting decision…";
  try {
    await approvalController.submit(approvalId, decision);
    addMarker(`approval_${decision}`);
  } catch (error) {
    // Leave the card actionable so the operator can retry this exact request;
    // the authoritative resolution event is what removes it from the map.
    if (approvalCardViews.get(approvalId) === view) {
      view.approve.disabled = false;
      view.reject.disabled = false;
      view.status.textContent = "Decision failed. Try again.";
    }
    throw error;
  }
}

sessionToggleButton.addEventListener("click", () => {
  const action = sessionTogglePresentation(
    controller.state,
    canStartSession(),
    remoteStopRequired,
  ).action;
  if (action === "connect") {
    void connect();
  } else {
    void stop().catch(() => undefined);
  }
});
stopSpeakingButton.addEventListener("click", () => {
  if (stopSpeakingButton.dataset.action === "resume") {
    void recoverSpeechRenderer();
  } else {
    void requestSpeechYield();
  }
});
const rendererInterrupted = (): void => {
  if (
    !naturalDuplexEnabled ||
    locallySilencedSpeech !== null ||
    currentSpeechTiming() === null
  ) {
    return;
  }
  renderSpeechControl("recover");
  addMarker("speech_renderer_interrupted");
};
remoteAudio.addEventListener("pause", rendererInterrupted);
remoteAudio.addEventListener("stalled", rendererInterrupted);
remoteAudio.addEventListener("error", rendererInterrupted);
muteButton.addEventListener("click", () => void toggleMicrophoneMute());
voiceSelect.addEventListener("change", () => void changeVoice(voiceSelect.value));
modelSelect.addEventListener("change", () => {
  const catalog = selectableModels;
  if (catalog === null) return;
  const requested = modelSelectionFor(catalog, modelSelect.value, effortSelect.value);
  void changeModelConfiguration(requested.model, requested.effort);
});
effortSelect.addEventListener("change", () =>
  void changeModelConfiguration(modelSelect.value, effortSelect.value),
);
typedForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = typedInput.value;
  if (text.trim() === "" || text.length > 4096) return;
  sendButton.disabled = true;
  void submitTyped(text)
    .then(() => {
      typedInput.value = "";
    })
    .catch(() => addMarker("typed_input_failed"))
    .finally(() => {
      sendButton.disabled = controller.state !== "connected";
    });
});

try {
  capability = takeOptionalBootstrapCapability(window.location, window.history);
  stableLaunch = capability === null;
  addMarker(capability === null ? "stable_launch_ready" : "launch_capability_loaded");
  renderSessionToggle();
} catch {
  capability = null;
  controller.failed();
  addMarker("fresh_launch_required");
}
