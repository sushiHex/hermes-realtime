export interface SearchEgressStatus {
  readonly available: boolean;
  readonly consentVersion: "realtime-search-egress-consent-v1";
  readonly disclosureDigest: string;
  readonly searchEgressState: "unavailable" | "idle" | "active";
}

export interface SearchEgressConsentRequest {
  readonly accepted: true;
  readonly consentVersion: "realtime-search-egress-consent-v1";
  readonly disclosureDigest: string;
  readonly sequence: number;
}

export interface SearchEgressRevokeRequest {
  readonly sequence: number;
}

type SearchEgressPath =
  | "/api/v1/search-egress-consent"
  | "/api/v1/search-egress-revoke";

export interface SearchEgressControls {
  projectStatus(status: SearchEgressStatus): void;
  reset(): void;
  setInteractive(interactive: boolean): void;
}

interface MountSearchEgressControlsOptions {
  readonly submit: (
    path: SearchEgressPath,
    body: SearchEgressConsentRequest | SearchEgressRevokeRequest,
  ) => Promise<void>;
}

interface PendingControl {
  readonly body: SearchEgressConsentRequest | SearchEgressRevokeRequest;
  readonly expectedState: "active" | "idle";
  readonly path: SearchEgressPath;
  readonly sequence: number;
}

function requiredElement<T extends HTMLElement>(document: Document, id: string): T {
  const value = document.getElementById(id);
  if (!(value instanceof document.defaultView!.HTMLElement)) {
    throw new Error(`Required search egress control element is missing: ${id}`);
  }
  return value as T;
}

export function mountSearchEgressControls(
  document: Document,
  options: MountSearchEgressControlsOptions,
): SearchEgressControls {
  const statusOutput = requiredElement<HTMLOutputElement>(document, "search-egress-status");
  const consentButton = requiredElement<HTMLButtonElement>(document, "search-egress-consent");
  const revokeButton = requiredElement<HTMLButtonElement>(document, "search-egress-revoke");
  let status: SearchEgressStatus | null = null;
  let lastAcceptedSequence = 0;
  let pending: PendingControl | null = null;
  let requestInFlight = false;
  let requestFailed = false;
  let interactive = true;
  let generation = 0;

  const render = (): void => {
    const state = requestFailed ? "error" : (status?.searchEgressState ?? "unavailable");
    statusOutput.dataset.searchEgressState = state;
    statusOutput.setAttribute("aria-busy", String(requestInFlight));
    if (requestInFlight) {
      statusOutput.textContent = "Public search consent request in progress.";
    } else if (state === "idle") {
      statusOutput.textContent = "Public search is off for this session.";
    } else if (state === "active") {
      statusOutput.textContent = "Public search is allowed for this session.";
    } else if (state === "error") {
      statusOutput.textContent = "Public search consent encountered an error.";
    } else {
      statusOutput.textContent = "Public search is unavailable for this session.";
    }
    consentButton.disabled =
      !interactive || requestInFlight || pending !== null || state !== "idle";
    revokeButton.disabled =
      !interactive || requestInFlight || pending !== null || state !== "active";
  };

  const nextSequence = (): number => {
    if (lastAcceptedSequence >= Number.MAX_SAFE_INTEGER) {
      throw new Error("search egress control sequence authority is exhausted");
    }
    return lastAcceptedSequence + 1;
  };

  const submit = async (control: PendingControl): Promise<void> => {
    if (requestInFlight || pending !== null) return;
    const requestGeneration = generation;
    pending = control;
    requestInFlight = true;
    requestFailed = false;
    render();
    try {
      await options.submit(control.path, control.body);
      if (generation !== requestGeneration) return;
      lastAcceptedSequence = control.sequence;
    } catch {
      if (generation !== requestGeneration) return;
      if (pending === control) {
        requestFailed = true;
      }
    } finally {
      if (generation !== requestGeneration) return;
      requestInFlight = false;
      render();
    }
  };

  consentButton.addEventListener("click", () => {
    const current = status;
    if (
      current === null ||
      !interactive ||
      requestInFlight ||
      pending !== null ||
      !current.available ||
      current.searchEgressState !== "idle"
    ) {
      return;
    }
    const sequence = nextSequence();
    void submit({
      body: {
        accepted: true,
        consentVersion: current.consentVersion,
        disclosureDigest: current.disclosureDigest,
        sequence,
      },
      expectedState: "active",
      path: "/api/v1/search-egress-consent",
      sequence,
    });
  });

  revokeButton.addEventListener("click", () => {
    const current = status;
    if (
      current === null ||
      !interactive ||
      requestInFlight ||
      pending !== null ||
      !current.available ||
      current.searchEgressState !== "active"
    ) {
      return;
    }
    const sequence = nextSequence();
    void submit({
      body: { sequence },
      expectedState: "idle",
      path: "/api/v1/search-egress-revoke",
      sequence,
    });
  });

  render();
  return {
    projectStatus(nextStatus): void {
      status = nextStatus;
      const current = pending;
      if (current !== null && current.expectedState === nextStatus.searchEgressState) {
        lastAcceptedSequence = Math.max(lastAcceptedSequence, current.sequence);
        if (pending === current) pending = null;
      } else if (current !== null && requestFailed && pending === current) {
        pending = null;
      }
      requestFailed = false;
      render();
    },
    reset(): void {
      generation += 1;
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
