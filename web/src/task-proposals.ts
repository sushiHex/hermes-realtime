import { parseTaskProposal, type TaskProposal } from "./protocol";

interface Credential { readonly participantIdentity: string; readonly token: string }
interface View {
  readonly data: TaskProposal;
  readonly participantIdentity: string | null;
  readonly item: HTMLLIElement;
  readonly objective: HTMLPreElement;
  readonly status: HTMLParagraphElement;
  readonly confirm: HTMLButtonElement;
  readonly decline: HTMLButtonElement;
  pending: boolean;
  busy: boolean;
}

export class TaskProposalControls {
  private readonly views = new Map<string, View>();
  private sequence = 0;
  private binding: string | null = null;
  private generation: number | null = null;

  constructor(private readonly document: Document, private readonly options: {
    credential: () => Credential | null;
    connected: () => boolean;
    submit: (token: string, body: Record<string, unknown>) => Promise<unknown>;
    admit: (item: HTMLLIElement) => void;
    changed: (item: HTMLLIElement) => void;
  }) { this.refresh(); }

  project(value: unknown): void {
    const data = parseTaskProposal(value);
    let view = this.views.get(data.proposalId);
    if (view !== undefined) {
      if (view.data.objective !== data.objective || view.data.generation !== data.generation ||
          (!view.pending && data.status === "pending")) {
        view.pending = false;
        this.render(view);
        try { throw new Error("task proposal authority changed"); }
        finally { console.info('[task-proposal-card] {"category":"authority_changed","count":1}'); }
      }
    }
    if (data.status === "pending") {
      if (this.generation !== null && data.generation < this.generation) {
        try { throw new Error("task proposal generation is stale"); }
        finally { console.info('[task-proposal-card] {"category":"stale_generation","count":1}'); }
      }
      if (this.generation !== data.generation) this.sequence = 0;
      this.generation = data.generation;
    }
    if (view === undefined) {
      const item = this.document.createElement("li");
      item.dataset.role = "operation";
      item.dataset.operation = "task-proposal";
      item.dataset.proposalId = data.proposalId;
      const label = this.document.createElement("strong");
      label.textContent = "Proposed background task";
      const objective = this.document.createElement("pre");
      objective.className = "proposal-objective";
      objective.id = `task-proposal-objective-${data.proposalId}`;
      objective.textContent = data.objective;
      const status = this.document.createElement("p");
      status.className = "operation-detail";
      status.setAttribute("role", "status");
      const buttons = this.document.createElement("div");
      buttons.className = "approval-controls";
      const confirm = this.document.createElement("button");
      confirm.type = "button"; confirm.className = "button approve";
      confirm.dataset.decision = "confirm"; confirm.textContent = "Confirm task";
      confirm.setAttribute("aria-describedby", objective.id);
      const decline = this.document.createElement("button");
      decline.type = "button"; decline.className = "button secondary";
      decline.dataset.decision = "decline"; decline.textContent = "Decline";
      decline.setAttribute("aria-describedby", objective.id);
      buttons.append(confirm, decline);
      item.append(label, objective, status, buttons);
      view = {data, participantIdentity:this.options.credential()?.participantIdentity ?? null,
        item, objective, status, confirm, decline, pending:data.status === "pending", busy:false};
      this.views.set(data.proposalId, view);
      const current = view;
      confirm.addEventListener("click", () => void this.decide(current, "confirm"));
      decline.addEventListener("click", () => void this.decide(current, "decline"));
      this.options.admit(item);
    }
    view.pending = data.status === "pending";
    view.item.dataset.status = data.status;
    if (!view.busy || !view.pending) view.status.textContent = {
      pending: data.reason === "use_confirmation_card"
        ? "Use Confirm on this task card for this proposal. Review the full objective first."
        : data.reason === "spoken_confirmation_not_ready"
          ? "Spoken confirmation is not ready. Review the full objective and use these controls."
          : "Review the full objective before confirming. No task has been dispatched by this proposal.",
      confirmed: "Confirmation submitted. Waiting for task status.",
      declined: "Proposal declined.", retired: "Proposal is no longer actionable.",
    }[data.status];
    this.refresh();
    this.options.changed(view.item);
  }

  refresh(): void {
    const identity = this.options.credential()?.participantIdentity ?? null;
    if (identity !== this.binding) { this.binding = identity; this.reset(); }
    for (const view of this.views.values()) this.render(view);
  }

  reset(): void {
    this.sequence = 0; this.generation = null;
    for (const view of this.views.values()) this.render(view);
  }

  actionable(item: HTMLLIElement): boolean {
    return [...this.views.values()].some(view => view.item === item && view.pending);
  }

  forget(item: HTMLLIElement): void {
    const id = item.dataset.proposalId;
    if (id !== undefined && this.views.get(id)?.item === item) this.views.delete(id);
  }

  private hasAuthority(view: View): boolean {
    return view.item.isConnected && view.item.contains(view.objective) &&
      view.objective.textContent === view.data.objective && view.pending &&
      view.data.generation === this.generation && this.options.connected() &&
      view.participantIdentity === this.options.credential()?.participantIdentity;
  }

  private render(view: View): void {
    view.confirm.disabled = view.decline.disabled = view.busy || !this.hasAuthority(view);
  }

  private async decide(view: View, decision: "confirm" | "decline"): Promise<void> {
    if (view.busy || !this.hasAuthority(view)) {
      try { return; }
      finally { console.info('[task-proposal-decision] {"category":"unavailable","count":1}'); }
    }
    const credential = this.options.credential()!;
    view.busy = true;
    this.render(view);
    const sequence = ++this.sequence;
    const body = {proposalId:view.data.proposalId, objective:view.data.objective,
      generation:view.data.generation, decision, sequence};
    let category = "refused";
    try {
      const reply = await this.options.submit(credential.token, body);
      if (!this.hasAuthority(view)) { category = "superseded"; return; }
      const ack = reply as Record<string, unknown>;
      if (Object.keys(ack).sort().join(",") !== "decision,generation,proposalId,sequence,version") throw new Error("proposal acknowledgment is invalid");
      if (ack.version !== 1 || ack.proposalId !== body.proposalId || ack.generation !== body.generation ||
          ack.decision !== decision || ack.sequence !== sequence) throw new Error("proposal acknowledgment does not match");
      view.status.textContent = decision === "confirm" ? "Confirmation submitted. Waiting for task status." : "Decline submitted.";
      category = "settled";
    } catch {
      if (this.hasAuthority(view)) view.status.textContent = "Decision not confirmed. Reconnect to reconcile the proposal.";
    } finally {
      console.info(`[task-proposal-decision] ${JSON.stringify({category,count:1})}`);
      this.options.changed(view.item);
    }
  }
}
