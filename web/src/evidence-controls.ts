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

export interface EvidenceConsentRequest {
  readonly accepted: true;
  readonly consentVersion: "realtime-evidence-consent-v1";
  readonly disclosureDigest: string;
  readonly retentionHours: number;
  readonly sequence: number;
  readonly sources: { readonly microphone: true; readonly typed: true };
}

export interface EvidenceRevokeRequest {
  readonly sequence: number;
}

export interface EvidenceControls {
  projectStatus(status: CaptureStatus): void;
  reset(): void;
  setInteractive(interactive: boolean): void;
}

interface PendingControl {
  readonly body: EvidenceConsentRequest | EvidenceRevokeRequest;
  readonly expectedState: "active" | "revoked_purging";
  readonly path: "/api/v1/evidence-consent" | "/api/v1/evidence-revoke";
  readonly sequence: number;
  readonly statusCanConfirm: boolean;
  confirmed: boolean;
  retryable: boolean;
}

interface MountEvidenceControlsOptions {
  readonly submit: (
    path: "/api/v1/evidence-consent" | "/api/v1/evidence-revoke",
    body: EvidenceConsentRequest | EvidenceRevokeRequest,
  ) => Promise<void>;
}

/** A bounded public control timeout leaves the exact server operation retryable. */
export class EvidenceControlTimeoutError extends Error {
  constructor() {
    super("evidence control timed out");
    this.name = "EvidenceControlTimeoutError";
  }
}

function requiredElement<T extends HTMLElement>(document: Document, id: string): T {
  const value = document.getElementById(id);
  if (!(value instanceof document.defaultView!.HTMLElement)) {
    throw new Error(`Required evidence control element is missing: ${id}`);
  }
  return value as T;
}

function projectedState(
  status: CaptureStatus | null,
  requestFailed: boolean,
): "unavailable" | "idle" | "active" | "revoked_purging" | "error" {
  if (requestFailed || status?.captureState === "faulted" || status?.captureState === "purge_failed") {
    return "error";
  }
  if (status === null) return "unavailable";
  return status.captureState;
}

function statusMessage(
  state: ReturnType<typeof projectedState>,
  requestInFlight: boolean,
  retryable: boolean,
): string {
  if (requestInFlight) return "Evidence capture request in progress.";
  if (retryable) return "Evidence capture request timed out. Retry the same request.";
  switch (state) {
    case "idle":
      return "Evidence capture is off. You may enable local capture.";
    case "active":
      return "Evidence capture is active.";
    case "revoked_purging":
      return "Consent was revoked. Evidence is being erased.";
    case "error":
      return "Evidence capture encountered an error. Controls are unavailable.";
    default:
      return "Evidence capture is unavailable for this session.";
  }
}

export function mountEvidenceControls(
  document: Document,
  options: MountEvidenceControlsOptions,
): EvidenceControls {
  const statusOutput = requiredElement<HTMLOutputElement>(document, "evidence-capture-status");
  const consentButton = requiredElement<HTMLButtonElement>(document, "evidence-consent");
  const revokeButton = requiredElement<HTMLButtonElement>(document, "evidence-revoke");
  let status: CaptureStatus | null = null;
  let lastAcceptedSequence = 0;
  let pending: PendingControl | null = null;
  let requestInFlight = false;
  let interactive = true;
  let requestFailed = false;
  const consentDefaultLabel = consentButton.textContent;
  const revokeDefaultLabel = revokeButton.textContent;

  const render = (): void => {
    const state = projectedState(status, requestFailed);
    const retry = pending?.retryable === true;
    const retryConsent = retry && pending?.expectedState === "active";
    const retryRevoke = retry && pending?.expectedState === "revoked_purging";
    statusOutput.dataset.captureState = state;
    statusOutput.textContent = statusMessage(state, requestInFlight, retry);
    statusOutput.setAttribute("aria-busy", String(requestInFlight));
    consentButton.disabled =
      !interactive || requestInFlight || (!retryConsent && (pending !== null || state !== "idle"));
    revokeButton.disabled =
      !interactive ||
      requestInFlight ||
      (!retryRevoke && (pending !== null || state !== "active"));
    consentButton.textContent = retryConsent ? "Retry enabling local capture" : consentDefaultLabel;
    revokeButton.textContent = retryRevoke ? "Retry revoking consent and erasing evidence" : revokeDefaultLabel;
    if (retryConsent) {
      consentButton.setAttribute("aria-label", "Retry enabling local capture");
    } else {
      consentButton.removeAttribute("aria-label");
    }
    if (retryRevoke) {
      revokeButton.setAttribute("aria-label", "Retry revoking consent and erasing evidence");
    } else {
      revokeButton.removeAttribute("aria-label");
    }
  };

  const sequence = (): number => {
    if (lastAcceptedSequence >= Number.MAX_SAFE_INTEGER) {
      throw new Error("evidence control sequence authority is exhausted");
    }
    return lastAcceptedSequence + 1;
  };

  const submit = async (
    path: "/api/v1/evidence-consent" | "/api/v1/evidence-revoke",
    body: EvidenceConsentRequest | EvidenceRevokeRequest,
    expectedState: PendingControl["expectedState"],
  ): Promise<void> => {
    if (requestInFlight) return;
    let request = pending;
    if (request === null) {
      request = {
        body,
        expectedState,
        path,
        sequence: body.sequence,
        statusCanConfirm: expectedState === "active",
        confirmed: false,
        retryable: false,
      };
      pending = request;
    } else if (
      !request.retryable ||
      request.path !== path ||
      request.body !== body && request.sequence !== body.sequence
    ) {
      return;
    }
    requestInFlight = true;
    requestFailed = false;
    request.retryable = false;
    render();
    try {
      await options.submit(request.path, request.body);
      lastAcceptedSequence = Math.max(lastAcceptedSequence, request.sequence);
      if (pending === request && request.expectedState === "revoked_purging") {
        pending = null;
      }
    } catch (error) {
      if (!request.confirmed) {
        if (error instanceof EvidenceControlTimeoutError) {
          request.retryable = true;
        } else {
          requestFailed = true;
          if (pending === request) pending = null;
        }
      }
    } finally {
      requestInFlight = false;
      render();
    }
  };

  consentButton.addEventListener("click", () => {
    const retry = pending;
    if (retry?.retryable && retry.expectedState === "active") {
      void submit(retry.path, retry.body, retry.expectedState);
      return;
    }
    const current = status;
    if (
      requestInFlight ||
      !interactive ||
      current === null ||
      !current.available ||
      current.captureState !== "idle"
    ) {
      return;
    }
    const request: EvidenceConsentRequest = {
      accepted: true,
      consentVersion: current.consentVersion,
      disclosureDigest: current.disclosureDigest,
      retentionHours: current.retentionHours,
      sequence: sequence(),
      sources: { microphone: true, typed: true },
    };
    void submit("/api/v1/evidence-consent", request, "active");
  });

  revokeButton.addEventListener("click", () => {
    const retry = pending;
    if (retry?.retryable && retry.expectedState === "revoked_purging") {
      void submit(retry.path, retry.body, retry.expectedState);
      return;
    }
    const current = status;
    if (
      requestInFlight ||
      !interactive ||
      current === null ||
      !current.available ||
      current.captureState !== "active"
    ) {
      return;
    }
    void submit("/api/v1/evidence-revoke", { sequence: sequence() }, "revoked_purging");
  });

  render();
  return {
    projectStatus(nextStatus): void {
      status = nextStatus;
      const current = pending;
      if (
        current !== null &&
        current.statusCanConfirm &&
        nextStatus.captureState === current.expectedState
      ) {
        current.confirmed = true;
        lastAcceptedSequence = Math.max(lastAcceptedSequence, current.sequence);
        if (pending === current) pending = null;
      }
      render();
    },
    reset(): void {
      status = null;
      lastAcceptedSequence = 0;
      pending = null;
      requestInFlight = false;
      requestFailed = false;
      render();
    },
    setInteractive(nextInteractive): void {
      interactive = nextInteractive;
      render();
    },
  };
}
