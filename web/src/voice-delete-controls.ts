export type VoiceDeleteState = "unavailable" | "idle" | "pending" | "complete" | "unknown";

type Credential = { token: string; participantIdentity: string };
type Path = "/api/v1/delete-voice-conversation" | "/api/v1/voice-delete-status";

const STATES: readonly string[] = ["unavailable", "idle", "pending", "complete", "unknown"];
const LIMIT = "What Hermes learned from it (memories and skills) stays and may still shape replies. There is no unlearning in the MVP. Delegated tasks remain in Hermes and are managed with Hermes's own session controls. A copy made with Hermes /branch is a separate conversation: deletion stays pending until you delete that copy in Hermes.";

export function parseVoiceDeleteState(value: unknown): VoiceDeleteState {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("voice delete status is invalid");
  }
  const data = value as Record<string, unknown>;
  if (
    Object.keys(data).sort().join(",") !== "state,version" ||
    data.version !== 1 ||
    typeof data.state !== "string" ||
    !STATES.includes(data.state)
  ) {
    throw new TypeError("voice delete status is invalid");
  }
  return data.state as VoiceDeleteState;
}

export class VoiceDeleteControls {
  private available = false;
  private pending = false;
  private inFlight = false;
  private timer: number | null = null;
  private epoch = 0;

  constructor(
    private readonly button: HTMLButtonElement,
    private readonly status: HTMLOutputElement,
    private readonly options: {
      credential: () => Credential | null;
      connected: () => boolean;
      confirm: (message: string) => boolean;
      request: (path: Path, token: string) => Promise<unknown>;
      clear: () => void;
      pollIntervalMs?: number;
    },
  ) {
    button.addEventListener("click", () => void this.delete());
    this.render();
  }

  render(): void {
    // A pending delete never blocks the next one: the current conversation is a new one.
    this.button.disabled = !this.options.connected() || !this.available || this.inFlight;
  }

  reset(): void {
    this.epoch += 1;
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = null;
    this.available = false;
    this.pending = false;
    this.inFlight = false;
    this.status.textContent = "Connect to delete this voice conversation.";
    this.render();
  }

  cleared(): void {
    this.options.clear();
  }

  async delete(): Promise<void> {
    const credential = this.options.credential();
    if (credential === null || !this.options.connected() || !this.available || this.inFlight) return;
    if (!this.options.confirm(`Delete this voice conversation? ${LIMIT}`)) return;
    const epoch = ++this.epoch;
    this.inFlight = true;
    this.status.textContent = "Starting deletion…";
    this.render();
    let recover = false;
    try {
      const state = parseVoiceDeleteState(
        await this.options.request("/api/v1/delete-voice-conversation", credential.token),
      );
      if (state !== "pending" && state !== "complete") throw new TypeError("delete did not start");
      if (
        this.epoch !== epoch ||
        this.options.credential()?.participantIdentity !== credential.participantIdentity ||
        this.options.credential()?.token !== credential.token
      ) return;
      this.present(state);
      if (this.pending) this.schedule(credential.participantIdentity);
    } catch {
      if (this.epoch !== epoch) return;
      this.status.textContent = "Deletion could not be confirmed. Checking status.";
      recover = true;
    } finally {
      if (this.epoch === epoch) {
        this.inFlight = false;
        this.render();
        if (recover) void this.refresh();
      }
    }
  }

  async refresh(): Promise<void> {
    if (this.inFlight) return;
    const credential = this.options.credential();
    if (credential === null || !this.options.connected()) return;
    const epoch = ++this.epoch;
    try {
      const state = parseVoiceDeleteState(
        await this.options.request("/api/v1/voice-delete-status", credential.token),
      );
      if (
        this.epoch !== epoch ||
        this.options.credential()?.participantIdentity !== credential.participantIdentity ||
        this.options.credential()?.token !== credential.token
      ) return;
      this.present(state);
      if (this.pending) this.schedule(credential.participantIdentity);
    } catch {
      if (
        this.epoch !== epoch ||
        this.options.credential()?.participantIdentity !== credential.participantIdentity ||
        this.options.credential()?.token !== credential.token
      ) return;
      this.status.textContent = "Deletion status unavailable. Checking again.";
      if (this.pending) this.schedule(credential.participantIdentity);
    }
  }

  private present(state: VoiceDeleteState): void {
    this.available = state !== "unavailable";
    // A delete this page saw pending is still polled while the companion is away.
    this.pending = state === "pending" || (state === "unavailable" && this.pending);
    this.status.textContent =
      state === "unavailable"
        ? "Voice conversation deletion is unavailable on this host. It needs the Hermes voice companion, and is off while evidence capture keeps its own copy (Revoke consent and erase removes that copy). A deletion already started resumes when the companion is back."
        : state === "pending"
          ? "Deletion pending. Hermes is still verifying the archive."
          : state === "complete"
            ? "Voice conversation deleted."
            : state === "unknown"
              ? "An earlier deletion could not be confirmed: its record was unreadable. Check Hermes's own session controls."
              : "Ready to delete this voice conversation.";
    this.render();
  }

  private schedule(identity: string): void {
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => {
      this.timer = null;
      if (this.pending && this.options.credential()?.participantIdentity === identity) {
        void this.refresh();
      }
    }, this.options.pollIntervalMs ?? 1500);
  }
}
