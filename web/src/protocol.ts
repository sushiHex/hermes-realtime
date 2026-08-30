export interface BootstrapCredential {
  readonly version: 1;
  readonly url: string;
  readonly roomName: string;
  readonly participantIdentity: string;
  readonly workerIdentity: string;
  readonly expiresInSeconds: number;
  readonly token: string;
}

export interface VoiceConfiguration {
  readonly version: 1;
  readonly voices: readonly string[];
  readonly selectedVoice: string | null;
}

export interface SpeechRuntime {
  readonly sttModel: string;
  readonly sttProvider: string;
  readonly ttsModel: string;
  readonly ttsProvider: string;
}

export interface SpeechAuthority {
  readonly turnId: string;
  readonly turnGeneration: number;
  readonly chunkId: string;
  readonly streamId: string;
}

export interface SessionModelConfiguration {
  readonly authentication: "subscription" | "api-token" | "local";
  readonly provider: string;
  readonly transport: string;
  readonly model: string;
  readonly effort: string | null;
  readonly contextWindowTokens: number | null;
  readonly reportsTokenUsage: boolean;
}

export interface SelectableModel {
  readonly model: string;
  readonly displayName: string;
  readonly description: string;
  readonly supportedEfforts: readonly string[];
  readonly defaultEffort: string;
}

export interface ModelCatalog {
  readonly version: 1;
  readonly models: readonly SelectableModel[];
  readonly selectedModel: string;
  readonly selectedEffort: string;
}

export interface SessionTokenUsage {
  readonly cachedInputTokens: number;
  readonly contextWindowTokens: number | null;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly reasoningOutputTokens: number;
  readonly totalTokens: number;
}

export interface CaptureStatus {
  readonly available: boolean;
  readonly captureState:
    | "unavailable"
    | "idle"
    | "active"
    | "revoked_purging"
    | "purge_failed"
    | "faulted";
  readonly consentVersion: "realtime-evidence-consent-v1";
  readonly disclosureDigest: string;
  readonly retentionHours: number;
}

export interface SearchEgressStatus {
  readonly available: boolean;
  readonly consentVersion: "realtime-search-egress-consent-v1";
  readonly disclosureDigest: string;
  readonly searchEgressState: "unavailable" | "idle" | "active";
}

const BOOTSTRAP_KEYS = [
  "expiresInSeconds",
  "participantIdentity",
  "roomName",
  "token",
  "url",
  "version",
  "workerIdentity",
] as const;
const ROOM = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
const IDENTITY = /^browser_[a-f0-9]{16,64}$/;
const WORKER_IDENTITY = /^worker_[A-Za-z0-9_-]{8,64}$/;

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    Object.getPrototypeOf(value) === Object.prototype
  );
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index])
  );
}

export function parseSpeechAuthority(value: unknown): SpeechAuthority {
  if (
    !isPlainRecord(value) ||
    !hasExactKeys(value, ["chunkId", "streamId", "turnGeneration", "turnId"])
  ) {
    throw new TypeError("Speech authority shape is invalid");
  }
  const boundedId = (candidate: unknown): candidate is string =>
    typeof candidate === "string" && candidate.length >= 1 && candidate.length <= 128;
  if (
    !boundedId(value.turnId) ||
    !Number.isSafeInteger(value.turnGeneration) ||
    (value.turnGeneration as number) <= 0 ||
    !boundedId(value.chunkId) ||
    !boundedId(value.streamId)
  ) {
    throw new TypeError("Speech authority fields are invalid");
  }
  return value as unknown as SpeechAuthority;
}

const REASONING_EFFORTS = new Set(["none", "low", "medium", "high", "xhigh", "max", "ultra"]);

export function parseModelCatalog(value: unknown): ModelCatalog {
  if (
    !isPlainRecord(value) ||
    !hasExactKeys(value, ["models", "selectedEffort", "selectedModel", "version"])
  ) {
    throw new TypeError("Model catalog shape is invalid");
  }
  if (
    value.version !== 1 ||
    !Array.isArray(value.models) ||
    value.models.length < 1 ||
    value.models.length > 100 ||
    typeof value.selectedModel !== "string" ||
    typeof value.selectedEffort !== "string"
  ) {
    throw new TypeError("Model catalog fields are invalid");
  }
  const models: SelectableModel[] = [];
  for (const item of value.models) {
    if (
      !isPlainRecord(item) ||
      !hasExactKeys(item, [
        "defaultEffort",
        "description",
        "displayName",
        "model",
        "supportedEfforts",
      ]) ||
      typeof item.model !== "string" ||
      item.model.length < 1 ||
      item.model.length > 256 ||
      typeof item.displayName !== "string" ||
      item.displayName.length < 1 ||
      item.displayName.length > 256 ||
      typeof item.description !== "string" ||
      item.description.length > 1024 ||
      !Array.isArray(item.supportedEfforts) ||
      item.supportedEfforts.length < 1 ||
      item.supportedEfforts.length > REASONING_EFFORTS.size ||
      item.supportedEfforts.some(
        (effort) => typeof effort !== "string" || !REASONING_EFFORTS.has(effort),
      ) ||
      new Set(item.supportedEfforts).size !== item.supportedEfforts.length ||
      typeof item.defaultEffort !== "string" ||
      !item.supportedEfforts.includes(item.defaultEffort)
    ) {
      throw new TypeError("Model catalog model fields are invalid");
    }
    models.push(item as unknown as SelectableModel);
  }
  if (new Set(models.map((item) => item.model)).size !== models.length) {
    throw new TypeError("Model catalog models are duplicated");
  }
  const selected = models.find((item) => item.model === value.selectedModel);
  if (selected === undefined || !selected.supportedEfforts.includes(value.selectedEffort)) {
    throw new TypeError("Model catalog selection is invalid");
  }
  return value as unknown as ModelCatalog;
}

export function parseVoiceConfiguration(value: unknown): VoiceConfiguration {
  if (!isPlainRecord(value) || !hasExactKeys(value, ["selectedVoice", "version", "voices"])) {
    throw new TypeError("Voice configuration shape is invalid");
  }
  if (value.version !== 1 || !Array.isArray(value.voices) || value.voices.length > 128) {
    throw new TypeError("Voice configuration is invalid");
  }
  const voices = value.voices;
  if (
    voices.some((voice) => typeof voice !== "string" || !/^[a-z]{2}_[a-z]+$/.test(voice)) ||
    new Set(voices).size !== voices.length ||
    (value.selectedVoice !== null &&
      (typeof value.selectedVoice !== "string" || !voices.includes(value.selectedVoice)))
  ) {
    throw new TypeError("Voice configuration fields are invalid");
  }
  return { version: 1, voices, selectedVoice: value.selectedVoice } as VoiceConfiguration;
}

export function parseSpeechRuntime(value: unknown): SpeechRuntime {
  const keys = ["sttModel", "sttProvider", "ttsModel", "ttsProvider"];
  if (!isPlainRecord(value) || !hasExactKeys(value, keys)) {
    throw new TypeError("Speech runtime shape is invalid");
  }
  const bounded = (key: string, maximum: number): boolean =>
    typeof value[key] === "string" &&
    (value[key] as string).trim().length > 0 &&
    (value[key] as string).length <= maximum;
  if (
    !bounded("sttProvider", 64) ||
    !bounded("sttModel", 256) ||
    !bounded("ttsProvider", 64) ||
    !bounded("ttsModel", 256)
  ) {
    throw new TypeError("Speech runtime fields are invalid");
  }
  return value as unknown as SpeechRuntime;
}

export function parseSessionModelConfiguration(value: unknown): SessionModelConfiguration {
  const keys = [
    "authentication",
    "contextWindowTokens",
    "effort",
    "model",
    "provider",
    "reportsTokenUsage",
    "transport",
  ];
  if (!isPlainRecord(value) || !hasExactKeys(value, keys)) {
    throw new TypeError("Session model configuration shape is invalid");
  }
  const boundedString = (item: unknown, maximum: number): item is string =>
    typeof item === "string" && item.trim().length > 0 && item.length <= maximum;
  if (
    !new Set(["subscription", "api-token", "local"]).has(value.authentication as string) ||
    !boundedString(value.provider, 64) ||
    !boundedString(value.transport, 64) ||
    !boundedString(value.model, 256) ||
    (value.effort !== null &&
      (typeof value.effort !== "string" ||
        !REASONING_EFFORTS.has(value.effort))) ||
    (value.contextWindowTokens !== null &&
      (!Number.isSafeInteger(value.contextWindowTokens) ||
        (value.contextWindowTokens as number) < 1 ||
        (value.contextWindowTokens as number) > 100_000_000)) ||
    typeof value.reportsTokenUsage !== "boolean"
  ) {
    throw new TypeError("Session model configuration fields are invalid");
  }
  return value as unknown as SessionModelConfiguration;
}

export function parseSessionTokenUsage(value: unknown): SessionTokenUsage {
  const keys = [
    "cachedInputTokens",
    "contextWindowTokens",
    "inputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
  ];
  if (!isPlainRecord(value) || !hasExactKeys(value, keys)) {
    throw new TypeError("Session token usage shape is invalid");
  }
  for (const key of keys) {
    if (key === "contextWindowTokens" && value[key] === null) continue;
    if (!Number.isSafeInteger(value[key]) || (value[key] as number) < 0) {
      throw new TypeError("Session token usage fields are invalid");
    }
  }
  if (
    value.contextWindowTokens !== null &&
    ((value.contextWindowTokens as number) < 1 ||
      (value.contextWindowTokens as number) > 100_000_000)
  ) {
    throw new TypeError("Session token usage fields are invalid");
  }
  return value as unknown as SessionTokenUsage;
}

export function parseCaptureStatus(value: unknown): CaptureStatus {
  if (
    !isPlainRecord(value) ||
    !hasExactKeys(value, [
      "available",
      "captureState",
      "consentVersion",
      "disclosureDigest",
      "retentionHours",
    ])
  ) {
    throw new TypeError("Capture status shape is invalid");
  }
  if (
    typeof value.available !== "boolean" ||
    !new Set([
      "unavailable",
      "idle",
      "active",
      "revoked_purging",
      "purge_failed",
      "faulted",
    ]).has(value.captureState as string) ||
    value.available !== (value.captureState !== "unavailable") ||
    value.consentVersion !== "realtime-evidence-consent-v1" ||
    typeof value.disclosureDigest !== "string" ||
    !/^[0-9a-f]{64}$/.test(value.disclosureDigest) ||
    !Number.isSafeInteger(value.retentionHours) ||
    (value.retentionHours as number) < 1 ||
    (value.retentionHours as number) > 168
  ) {
    throw new TypeError("Capture status fields are invalid");
  }
  return value as unknown as CaptureStatus;
}

export function parseSearchEgressStatus(value: unknown): SearchEgressStatus {
  if (
    !isPlainRecord(value) ||
    !hasExactKeys(value, [
      "available",
      "consentVersion",
      "disclosureDigest",
      "searchEgressState",
    ])
  ) {
    throw new TypeError("Search egress status shape is invalid");
  }
  if (
    typeof value.available !== "boolean" ||
    !new Set(["unavailable", "idle", "active"]).has(value.searchEgressState as string) ||
    value.available !== (value.searchEgressState !== "unavailable") ||
    value.consentVersion !== "realtime-search-egress-consent-v1" ||
    typeof value.disclosureDigest !== "string" ||
    !/^[0-9a-f]{64}$/.test(value.disclosureDigest)
  ) {
    throw new TypeError("Search egress status fields are invalid");
  }
  return value as unknown as SearchEgressStatus;
}

export function parseBootstrapCredential(value: unknown): BootstrapCredential {
  if (!isPlainRecord(value)) {
    throw new Error("Invalid bootstrap credential shape");
  }
  const keys = Object.keys(value).sort();
  if (
    keys.length !== BOOTSTRAP_KEYS.length ||
    keys.some((key, index) => key !== BOOTSTRAP_KEYS[index])
  ) {
    throw new Error("Invalid bootstrap credential shape");
  }

  const {
    expiresInSeconds,
    participantIdentity,
    roomName,
    token,
    url,
    version,
    workerIdentity,
  } = value;
  if (
    version !== 1 ||
    typeof url !== "string" ||
    typeof roomName !== "string" ||
    !ROOM.test(roomName) ||
    typeof participantIdentity !== "string" ||
    !IDENTITY.test(participantIdentity) ||
    typeof workerIdentity !== "string" ||
    !WORKER_IDENTITY.test(workerIdentity) ||
    typeof expiresInSeconds !== "number" ||
    !Number.isInteger(expiresInSeconds) ||
    expiresInSeconds < 30 ||
    expiresInSeconds > 300 ||
    typeof token !== "string" ||
    token.length > 8192 ||
    token.split(".").length !== 3
  ) {
    throw new Error("Invalid bootstrap credential fields");
  }

  let endpoint: URL;
  try {
    endpoint = new URL(url);
  } catch {
    throw new Error("Invalid bootstrap credential fields");
  }
  const loopback = endpoint.hostname === "127.0.0.1" || endpoint.hostname === "localhost";
  if (
    (endpoint.protocol !== "wss:" && !(endpoint.protocol === "ws:" && loopback)) ||
    endpoint.username !== "" ||
    endpoint.password !== ""
  ) {
    throw new Error("Invalid bootstrap credential fields");
  }

  return {
    version: 1,
    url,
    roomName,
    participantIdentity,
    workerIdentity,
    expiresInSeconds,
    token,
  };
}

const EVENT_KEYS = ["data", "kind", "monotonicMs", "sequence"] as const;
const EVENT_BATCH_KEYS = ["events", "version"] as const;
const EVENT_KINDS = new Set([
  "approval_state",
  "assistant_turn_completed",
  "assistant_turn_interrupted",
  "assistant_text_generated",
  "barge_in_non_speech_suppressed",
  "barge_in_verifier_unavailable",
  "capture_status",
  "completion_received",
  "echo_barge_in_confirmed",
  "echo_suppressed",
  "transcript_echo_suppressed",
  "first_foreground_token",
  "first_playable_audio",
  "interrupt_requested",
  "knowledge_timing",
  "notification_queued",
  "playback_silenced",
  "session_model",
  "session_usage",
  "session_ready",
  "search_egress_status",
  "session_stopped",
  "speech_timing",
  "speech_ended",
  "task_state",
  "task_result",
  "transcript_final",
  "transcript_partial",
  "typed_input_admitted",
  "voice_activity_ended",
  "voice_activity_started",
  "voice_input_ready",
]);
const PRIVATE_HANDLE = /(?:deleg_[A-Za-z0-9_-]*|run_[A-Za-z0-9_-]{8,})/;

export interface PublicEvent {
  readonly sequence: number;
  readonly kind: string;
  readonly monotonicMs: number;
  readonly data: Readonly<Record<string, string | number | boolean | null>>;
}

export interface EventBatch {
  readonly version: 1;
  readonly events: readonly PublicEvent[];
}

function parseEvent(value: unknown): PublicEvent {
  if (!isPlainRecord(value) || !hasExactKeys(value, EVENT_KEYS)) {
    throw new TypeError("Public event shape is invalid");
  }
  if (!Number.isSafeInteger(value.sequence) || (value.sequence as number) <= 0) {
    throw new TypeError("Public event sequence is invalid");
  }
  if (typeof value.kind !== "string" || !EVENT_KINDS.has(value.kind)) {
    throw new TypeError("Public event kind is invalid");
  }
  if (
    typeof value.monotonicMs !== "number" ||
    !Number.isFinite(value.monotonicMs) ||
    value.monotonicMs < 0
  ) {
    throw new TypeError("Public event timestamp is invalid");
  }
  if (!isPlainRecord(value.data) || Object.keys(value.data).length > 16) {
    throw new TypeError("Public event data shape is invalid");
  }
  for (const [key, item] of Object.entries(value.data)) {
    if (!/^[A-Za-z][A-Za-z0-9]{0,63}$/.test(key)) {
      throw new TypeError("Public event data key is invalid");
    }
    if (
      item !== null &&
      typeof item !== "string" &&
      typeof item !== "boolean" &&
      !(typeof item === "number" && Number.isSafeInteger(item))
    ) {
      throw new TypeError("Public event data primitive is invalid");
    }
    if (typeof item === "string" && (item.length > 4096 || PRIVATE_HANDLE.test(item))) {
      throw new TypeError("Public event data string is invalid");
    }
  }
  if (value.kind === "session_ready") {
    if (
      !hasExactKeys(value.data, [
        "conversationProfile",
        "mode",
        "sttModel",
        "sttProvider",
        "ttsModel",
        "ttsProvider",
      ]) ||
      !new Set(["legacy", "natural_v1"]).has(value.data.conversationProfile as string) ||
      value.data.mode !== "microphone_or_typed"
    ) {
      throw new TypeError("Public session speech runtime shape is invalid");
    }
    parseSpeechRuntime({
      sttModel: value.data.sttModel,
      sttProvider: value.data.sttProvider,
      ttsModel: value.data.ttsModel,
      ttsProvider: value.data.ttsProvider,
    });
  }
  if (value.kind === "capture_status") parseCaptureStatus(value.data);
  if (value.kind === "search_egress_status") parseSearchEgressStatus(value.data);
  if (
    value.kind === "voice_input_ready" &&
    (!hasExactKeys(value.data, ["generation", "mediaIncarnation"]) ||
      !Number.isSafeInteger(value.data.generation) ||
      (value.data.generation as number) <= 0 ||
      !Number.isSafeInteger(value.data.mediaIncarnation) ||
      (value.data.mediaIncarnation as number) <= 0)
  ) {
    throw new TypeError("Public voice readiness authority is invalid");
  }
  if (
    (value.kind === "interrupt_requested" ||
      value.kind === "barge_in_non_speech_suppressed") &&
    Object.keys(value.data).length > 0
  ) {
    parseSpeechAuthority(value.data);
  }
  if (
    value.kind === "task_result" &&
    (!hasExactKeys(value.data, ["status", "taskId", "text"]) ||
      typeof value.data.status !== "string" ||
      !new Set(["completed", "failed", "interrupted"]).has(value.data.status) ||
      typeof value.data.taskId !== "string" ||
      !/^task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,122}$/.test(value.data.taskId) ||
      typeof value.data.text !== "string" ||
      value.data.text.trim().length < 1 ||
      value.data.text.length > 1024)
  ) {
    throw new TypeError("Public task result is invalid");
  }
  if (value.kind === "knowledge_timing") {
    const data = value.data;
    const keys = [
      "backend",
      "lastMs",
      "lookupBlockingMs",
      "lookupElapsedMs",
      "lookupOverlapMs",
      "outcome",
      "p50Ms",
      "p95Ms",
      "recoveryUsed",
      "route",
      "sampleCount",
      "turnId",
    ];
    const healthKeys = [
      "lookupClosed",
      "lookupDetachedCalls",
      "lookupDetachedCallsTotal",
      "lookupSaturationEvents",
    ];
    const millisecondKeys = [
      "lastMs",
      "p50Ms",
      "p95Ms",
      "lookupElapsedMs",
      "lookupBlockingMs",
      "lookupOverlapMs",
    ];
    const validMilliseconds = millisecondKeys.every(
      (key) =>
        Number.isSafeInteger(data[key]) &&
        (data[key] as number) >= 0 &&
        (data[key] as number) <= 600_000,
    );
    const hasHealth = hasExactKeys(data, [...keys, ...healthKeys]);
    const validHealth =
      !hasHealth ||
      (typeof data.lookupClosed === "boolean" &&
        ["lookupDetachedCalls", "lookupDetachedCallsTotal", "lookupSaturationEvents"].every(
          (key) =>
            Number.isSafeInteger(data[key]) &&
            (data[key] as number) >= 0,
        ));
    if (
      (!hasExactKeys(data, keys) && !hasHealth) ||
      typeof data.turnId !== "string" ||
      !/^[A-Za-z][A-Za-z0-9_.:-]{0,127}$/.test(data.turnId) ||
      typeof data.route !== "string" ||
      !/^[a-z][a-z0-9_-]{0,63}$/.test(data.route) ||
      typeof data.backend !== "string" ||
      !/^[a-z][a-z0-9_-]{0,63}$/.test(data.backend) ||
      typeof data.outcome !== "string" ||
      !new Set(["usable", "weak", "empty", "timeout", "failed"]).has(data.outcome) ||
      !Number.isSafeInteger(data.sampleCount) ||
      (data.sampleCount as number) < 1 ||
      (data.sampleCount as number) > 4096 ||
      typeof data.recoveryUsed !== "boolean" ||
      !validHealth ||
      !validMilliseconds ||
      data.lookupElapsedMs !==
        (data.lookupBlockingMs as number) + (data.lookupOverlapMs as number)
    ) {
      throw new TypeError("Public knowledge timing is invalid");
    }
  }
  return value as unknown as PublicEvent;
}

export function parseEventBatch(value: unknown): EventBatch {
  if (!isPlainRecord(value) || !hasExactKeys(value, EVENT_BATCH_KEYS)) {
    throw new TypeError("Event batch shape is invalid");
  }
  if (value.version !== 1 || !Array.isArray(value.events) || value.events.length > 256) {
    throw new TypeError("Event batch is invalid");
  }
  const events = value.events.map(parseEvent);
  for (let index = 1; index < events.length; index += 1) {
    if (events[index]!.sequence !== events[index - 1]!.sequence + 1) {
      throw new TypeError("Event batch sequence is not contiguous");
    }
  }
  return { version: 1, events };
}
