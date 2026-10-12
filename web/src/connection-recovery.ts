import { ReloadRebindRefused } from "./controller";

export type ConnectionFailureStage =
  | "bootstrap"
  | "rebind"
  | "media"
  | "microphone"
  | "activation"
  | "configuration"
  | "stop"
  | "worker"
  | "projection";

export type ConnectionFailureCategory =
  | "request-failed"
  | "request-refused"
  | "authorization-refused"
  | "request-timed-out"
  | "service-unavailable";

export interface ConnectionFailure {
  readonly stage: ConnectionFailureStage;
  readonly category: ConnectionFailureCategory;
}

export class ConnectionRequestRejected extends Error {
  constructor(readonly status: number) {
    super("connection request rejected");
  }
}

export function connectionFailureCategory(error: unknown): ConnectionFailureCategory {
  if (!(error instanceof ConnectionRequestRejected) && !(error instanceof ReloadRebindRefused)) {
    return "request-failed";
  }
  if (error.status === 403) return "authorization-refused";
  if (error.status === 408) return "request-timed-out";
  if (error.status === 503) return "service-unavailable";
  return "request-refused";
}

const titles: Record<ConnectionFailureStage, string> = {
  bootstrap: "Session startup failed",
  rebind: "Session reconnect failed",
  media: "Audio connection failed",
  microphone: "Microphone preparation failed",
  activation: "Session activation failed",
  configuration: "Session configuration failed",
  stop: "Session disconnect was not confirmed",
  worker: "Speech worker disconnected",
  projection: "Conversation updates unavailable",
};

const reasons: Record<ConnectionFailureCategory, string> = {
  "request-failed": "",
  "request-refused": "Request refused. ",
  "authorization-refused": "Authorization refused (403). ",
  "request-timed-out": "Request timed out (408). ",
  "service-unavailable": "Host refused the request (503). ",
};

export function renderConnectionRecovery(
  output: HTMLOutputElement,
  failure: ConnectionFailure | null,
  canStartSession: boolean,
  remoteStopRequired: boolean,
): void {
  output.hidden = failure === null;
  if (failure === null) {
    output.textContent = "";
    delete output.dataset.stage;
    delete output.dataset.category;
    return;
  }
  output.dataset.stage = failure.stage;
  output.dataset.category = failure.category;
  const retry = remoteStopRequired
    ? "Local audio is disconnected. Select Disconnect to retry the server stop."
    : !canStartSession
      ? "Open a fresh launch from the host to connect."
      : "Select Connect to retry.";
  const recovery = failure.category === "service-unavailable"
    ? " If it persists, ask the host operator to check recovery; a restart may be needed."
    : !remoteStopRequired && canStartSession
      ? " If it repeats, ask the host operator to check recovery."
      : "";
  output.textContent = `${titles[failure.stage]}. ${reasons[failure.category]}${retry}${recovery}`;
}
