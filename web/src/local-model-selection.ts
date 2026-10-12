export interface LocalModelOption {
  readonly model: string;
  readonly digest: string;
  readonly advertisedContextTokens: number | null;
}

export interface PreparedLocalModelSelection {
  readonly model: string;
  readonly digest: string;
  readonly requestedContextTokens: number;
  readonly advertisedContextTokens: number | null;
  readonly effectiveContextTokens: number | null;
}

export interface LocalModelCatalog {
  readonly version: 1;
  readonly available: true;
  readonly provider: "ollama";
  readonly state: "configuring" | "preparing" | "prepared";
  readonly models: readonly LocalModelOption[];
  readonly defaultModel: string | null;
  readonly defaultRequestedContextTokens: number;
  readonly minimumContextTokens: number;
  readonly maximumContextTokens: number;
  readonly preparationTimeoutMs: number;
  readonly selection: PreparedLocalModelSelection | null;
}

export interface LocalModelUnavailable {
  readonly version: 1;
  readonly available: false;
}

export type LocalModelAvailability = LocalModelCatalog | LocalModelUnavailable;

export class LocalModelProtocolError extends TypeError {}

export interface LocalModelPreparationRequest {
  readonly version: 1;
  readonly model: string;
  readonly digest: string;
  readonly requestedContextTokens: number;
}

const MAX_REPORTED_CAPACITY = 100_000_000;
export const LOCAL_MODEL_METADATA_TIMEOUT_MS = 10_000;
export const LOCAL_MODEL_PREPARATION_TIMEOUT_MS = 125_000;
const MAX_LOCAL_MODEL_RESPONSE_BYTES = 262_144;

async function readBoundedResponse(response: Response): Promise<Response> {
  if (response.body === null) return response;
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      length += part.value.byteLength;
      if (length > MAX_LOCAL_MODEL_RESPONSE_BYTES) {
        await reader.cancel();
        throw new TypeError("Local model response exceeds the bounded size");
      }
      chunks.push(part.value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new Response(length === 0 ? null : body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

export async function fetchLocalModelRequest(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
  parentSignal: AbortSignal | null = null,
  fetcher: typeof fetch = fetch,
): Promise<Response> {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > LOCAL_MODEL_PREPARATION_TIMEOUT_MS) {
    throw new RangeError("Local model request timeout is invalid");
  }
  const request = new AbortController();
  const abort = (): void => request.abort();
  if (parentSignal?.aborted) abort();
  else parentSignal?.addEventListener("abort", abort, { once: true });
  const timeout = globalThis.setTimeout(abort, timeoutMs);
  try {
    const response = await fetcher(input, { ...init, signal: request.signal });
    return await readBoundedResponse(response);
  } finally {
    globalThis.clearTimeout(timeout);
    parentSignal?.removeEventListener("abort", abort);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(value: object, expected: readonly string[]): boolean {
  const keys = Object.keys(value);
  return keys.length === expected.length && keys.every((key) => expected.includes(key));
}

function validModel(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0 && value.length <= 256 &&
    !/[\u0000-\u001f\u007f-\u009f]/.test(value);
}

function validDigest(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function validReportedCapacity(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0 &&
    (value as number) <= MAX_REPORTED_CAPACITY;
}

function validOptionalCapacity(value: unknown): value is number | null {
  return value === null || validReportedCapacity(value);
}

function validRequestedCapacity(value: unknown, minimum: number, maximum: number): value is number {
  return Number.isSafeInteger(value) && (value as number) >= minimum &&
    (value as number) <= maximum;
}

function refusal(): never {
  throw new LocalModelProtocolError("Local model selection protocol is invalid");
}

function parseSelection(value: unknown): PreparedLocalModelSelection | null {
  if (value === null) return null;
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "advertisedContextTokens",
      "digest",
      "effectiveContextTokens",
      "model",
      "requestedContextTokens",
    ]) ||
    !validModel(value.model) ||
    !validDigest(value.digest) ||
    !validRequestedCapacity(value.requestedContextTokens, 2048, 262144) ||
    !validOptionalCapacity(value.advertisedContextTokens) ||
    !validOptionalCapacity(value.effectiveContextTokens) ||
    (value.advertisedContextTokens !== null &&
      (value.requestedContextTokens as number) > (value.advertisedContextTokens as number))
  ) return null;
  return value as unknown as PreparedLocalModelSelection;
}

function parseLocalModelCatalogValue(value: unknown): LocalModelCatalog | null {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "available",
      "defaultModel",
      "defaultRequestedContextTokens",
      "maximumContextTokens",
      "minimumContextTokens",
      "models",
      "preparationTimeoutMs",
      "provider",
      "selection",
      "state",
      "version",
    ]) ||
    value.version !== 1 || value.available !== true || value.provider !== "ollama" ||
    typeof value.state !== "string" ||
    !["configuring", "preparing", "prepared"].includes(value.state) ||
    !Array.isArray(value.models) || value.models.length > 256 ||
    value.defaultRequestedContextTokens !== 16384 ||
    value.minimumContextTokens !== 2048 || value.maximumContextTokens !== 262144 ||
    value.preparationTimeoutMs !== 120000 ||
    (value.defaultModel !== null && !validModel(value.defaultModel))
  ) return null;

  const models: LocalModelOption[] = [];
  for (const candidate of value.models) {
    if (
      !isRecord(candidate) ||
      !hasExactKeys(candidate, ["advertisedContextTokens", "digest", "model"]) ||
      !validModel(candidate.model) || !validDigest(candidate.digest) ||
      !validOptionalCapacity(candidate.advertisedContextTokens)
    ) return null;
    models.push(candidate as unknown as LocalModelOption);
  }
  if (
    new Set(models.map((item) => item.model)).size !== models.length
  ) return null;
  if (value.defaultModel !== null && !models.some((item) => item.model === value.defaultModel)) {
    return null;
  }
  const selection = parseSelection(value.selection);
  if (value.selection !== null && selection === null) return null;
  if (
    (value.state === "prepared") !== (selection !== null) ||
    (selection !== null && !models.some((item) =>
      item.model === selection.model && item.digest === selection.digest &&
      item.advertisedContextTokens === selection.advertisedContextTokens,
    ))
  ) return null;
  return value as unknown as LocalModelCatalog;
}

function parseLocalModelAvailabilityValue(value: unknown): LocalModelAvailability | null {
  if (isRecord(value) && hasExactKeys(value, ["available", "version"]) &&
      value.version === 1 && value.available === false) {
    return value as unknown as LocalModelUnavailable;
  }
  return parseLocalModelCatalogValue(value);
}

export function parseLocalModelCatalog(value: unknown): LocalModelAvailability {
  let accepted = false;
  try {
    const parsed = parseLocalModelAvailabilityValue(value);
    if (parsed === null) return refusal();
    accepted = true;
    return parsed;
  } finally {
    if (!accepted) {
      console.info(`[local-model-selection-refusal] ${JSON.stringify({ category: "invalid_catalog" })}`);
    }
  }
}

export function localModelPreparationFor(
  catalog: LocalModelCatalog,
  model: string,
  requestedContextTokens: number | null,
): LocalModelPreparationRequest {
  let accepted = false;
  try {
    const selected = catalog.models.find((item) => item.model === model);
    if (
      catalog.state !== "configuring" || selected === undefined ||
      !validRequestedCapacity(
        requestedContextTokens,
        catalog.minimumContextTokens,
        catalog.maximumContextTokens,
      ) ||
      (selected.advertisedContextTokens !== null &&
        (requestedContextTokens as number) > selected.advertisedContextTokens)
    ) return refusal();
    accepted = true;
    return {
      version: 1,
      model: selected.model,
      digest: selected.digest,
      requestedContextTokens,
    };
  } finally {
    if (!accepted) {
      console.info(`[local-model-selection-refusal] ${JSON.stringify({ category: "invalid_request" })}`);
    }
  }
}

export function parseLocalModelPreparation(
  value: unknown,
  request: LocalModelPreparationRequest,
  catalog: LocalModelCatalog,
): PreparedLocalModelSelection {
  let accepted = false;
  try {
    if (!isRecord(value) || !hasExactKeys(value, ["selection", "version"]) || value.version !== 1) {
      return refusal();
    }
    const selection = parseSelection(value.selection);
    const model = catalog.models.find((item) => item.model === request.model);
    if (
      selection === null || model === undefined || catalog.state !== "configuring" ||
      request.digest !== model.digest ||
      selection.model !== request.model || selection.digest !== request.digest ||
      selection.requestedContextTokens !== request.requestedContextTokens ||
      selection.advertisedContextTokens !== model.advertisedContextTokens
    ) return refusal();
    accepted = true;
    return selection;
  } finally {
    if (!accepted) {
      console.info(`[local-model-selection-refusal] ${JSON.stringify({ category: "invalid_preparation" })}`);
    }
  }
}
