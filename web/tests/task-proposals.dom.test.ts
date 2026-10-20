import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { describe, expect, it, vi } from "vitest";
import { TaskProposalControls } from "../src/task-proposals";

const proposal = { proposalId: "proposal_12345678", objective: "Review **this** exact objective\nincluding <synthetic>markup</synthetic> and all details.",
  generation: 7, status: "pending", reason: null };

function mount(submit = vi.fn<(token: string, body: Record<string, unknown>) => Promise<unknown>>(async (_token, body) => ({version:1,
  proposalId:body.proposalId, generation:body.generation, decision:body.decision, sequence:body.sequence}))) {
  const dom = new JSDOM(readFileSync(new URL("../index.html", import.meta.url), "utf8"));
  let credential = { participantIdentity: "browser_0123456789abcdef", token: "synthetic.token" };
  let connected = true;
  const transcript = dom.window.document.querySelector<HTMLOListElement>("#transcript")!;
  const controls = new TaskProposalControls(dom.window.document, {
    credential: () => credential, connected: () => connected, submit,
    admit: item => transcript.append(item), changed: () => {},
  });
  const card = () => transcript.querySelector<HTMLElement>('[data-operation="task-proposal"]')!;
  const confirm = () => card().querySelector<HTMLButtonElement>('[data-decision="confirm"]')!;
  const decline = () => card().querySelector<HTMLButtonElement>('[data-decision="decline"]')!;
  const refresh = (identity = credential.participantIdentity) => {
    credential = {participantIdentity:identity, token:"synthetic.refreshed"}; controls.refresh();
  };
  return { dom, controls, card, confirm, decline, submit, refresh,
    connected: (value: boolean) => { connected = value; controls.refresh(); } };
}

describe("task proposal controls", () => {
  it.each(["confirm", "decline"] as const)("presents the full exact objective before %s and submits once", async decision => {
    const view = mount();
    view.controls.project(proposal);
    expect(view.card().querySelector(".proposal-objective")?.textContent).toBe(proposal.objective);
    expect(view.card().textContent).not.toContain("started");
    const button = decision === "confirm" ? view.confirm() : view.decline();
    expect(button.type).toBe("button");
    expect(view.card().querySelector("#" + button.getAttribute("aria-describedby"))?.textContent).toBe(proposal.objective);
    button.click(); button.click();
    await vi.waitFor(() => expect(view.submit).toHaveBeenCalledOnce());
    expect(view.submit).toHaveBeenCalledWith("synthetic.token", {
      proposalId:proposal.proposalId, objective:proposal.objective, generation:7, decision, sequence:1,
    });
    expect(view.confirm().disabled).toBe(true);
    expect(view.decline().disabled).toBe(true);
    expect(view.card().textContent).not.toContain("started");
    view.controls.project({...proposal, status:decision === "confirm" ? "confirmed" : "declined"});
    expect(view.confirm().disabled).toBe(true);
    view.dom.window.close();
  });

  it("retains one in-flight action through pending replay and ignores stale settlement after rebinding", async () => {
    let resolve!: (reply: Record<string, unknown>) => void;
    const submit = vi.fn(() => new Promise<Record<string, unknown>>(done => {resolve=done;}));
    const view = mount(submit);
    view.controls.project(proposal);
    view.confirm().click();
    view.controls.project(proposal);
    expect(view.dom.window.document.querySelectorAll('[data-operation="task-proposal"]')).toHaveLength(1);
    view.decline().click();
    expect(submit).toHaveBeenCalledOnce();
    view.refresh("browser_fedcba9876543210");
    view.controls.project({...proposal, proposalId:"proposal_new_12345"});
    resolve({version:1, proposalId:proposal.proposalId, generation:7, decision:"confirm", sequence:1});
    await Promise.resolve();
    expect(view.confirm().disabled).toBe(true);
    expect(view.card().textContent).not.toContain("Confirmation submitted");
    view.dom.window.close();
  });

  it("uses refreshed same-participant credentials and disables disconnected controls", async () => {
    const view = mount();
    view.controls.project(proposal);
    view.connected(false);
    view.confirm().click();
    expect(view.submit).not.toHaveBeenCalled();
    view.refresh();
    view.connected(true);
    view.decline().click();
    await vi.waitFor(() => expect(view.submit).toHaveBeenCalledOnce());
    expect(view.submit.mock.calls[0]?.[0]).toBe("synthetic.refreshed");
    view.dom.window.close();
  });

  it.each(["objective", "generation"])("refuses a changed %s on an existing proposal identity", field => {
    const view = mount();
    view.controls.project(proposal);
    expect(() => view.controls.project({...proposal, [field]: field === "objective" ? "Edited" : 8})).toThrow();
    view.confirm().click();
    expect(view.submit).not.toHaveBeenCalled();
    view.dom.window.close();
  });

  it("does not reactivate a settled proposal or a detached card", () => {
    const view = mount();
    view.controls.project(proposal);
    view.controls.project({...proposal, status:"retired"});
    expect(() => view.controls.project(proposal)).toThrow();
    view.confirm().click();
    expect(view.submit).not.toHaveBeenCalled();
    const detached = mount();
    detached.controls.project(proposal);
    const button = detached.confirm();
    detached.card().remove();
    button.click();
    expect(detached.submit).not.toHaveBeenCalled();
    view.dom.window.close(); detached.dom.window.close();
  });

  it.each(["changed", "missing"])("cannot confirm a %s objective presentation", change => {
    const view = mount();
    view.controls.project(proposal);
    const objective = view.card().querySelector(".proposal-objective")!;
    if (change === "changed") objective.textContent = "Edited presentation";
    else objective.remove();
    view.confirm().click();
    expect(view.submit).not.toHaveBeenCalled();
    view.dom.window.close();
  });

  it.each(["rejected", "malformed", "wrong-binding", "extra"])("keeps uncertain %s decisions disabled", async failure => {
    const submit = vi.fn(async () => {
      if (failure === "rejected") throw new Error("synthetic refusal");
      if (failure === "extra") return {version:1, proposalId:proposal.proposalId, generation:7,
        decision:"confirm", sequence:1, extra:true};
      return failure === "malformed" ? {version:1} : {version:1, proposalId:proposal.proposalId,
        generation:8, decision:"confirm", sequence:1};
    });
    const view = mount(submit);
    view.controls.project(proposal);
    view.confirm().click();
    await vi.waitFor(() => expect(view.card().textContent).toContain("not confirmed"));
    expect(view.confirm().disabled).toBe(true);
    expect(view.decline().disabled).toBe(true);
    view.controls.project(proposal);
    expect(view.confirm().disabled).toBe(true);
    view.dom.window.close();
  });

  it("keeps early spoken-confirmation guidance distinct from actionable text controls", () => {
    const view = mount();
    view.controls.project({...proposal, reason:"spoken_confirmation_not_ready"});
    expect(view.card().textContent).toContain("Spoken confirmation is not ready");
    expect(view.confirm().disabled).toBe(false);
    view.dom.window.close();
  });

  it("directs permanently ineligible spoken confirmation to the visible card", () => {
    const view = mount();
    view.controls.project({...proposal, reason:"use_confirmation_card"});
    expect(view.card().textContent).toContain("Use Confirm on this task card");
    expect(view.confirm().disabled).toBe(false);
    view.dom.window.close();
  });

  it("disables old pending controls when a newer generation arrives without an old settlement", () => {
    const view = mount();
    view.controls.project(proposal);
    const old = view.confirm();
    view.controls.project({...proposal, proposalId:"proposal_new_12345", generation:8});
    expect(old.disabled).toBe(true);
    old.click();
    expect(view.submit).not.toHaveBeenCalled();
    view.dom.window.close();
  });

  it("retains sequences through token refresh, resets for the next generation, and refuses stale generations", async () => {
    const view = mount();
    view.controls.project(proposal);
    view.confirm().click();
    await vi.waitFor(() => expect(view.submit).toHaveBeenCalledOnce());
    view.controls.project({...proposal, status:"confirmed"});
    view.refresh();
    view.controls.project({...proposal, proposalId:"proposal_87654321"});
    view.decline = () => view.dom.window.document.querySelector<HTMLButtonElement>('[data-proposal-id="proposal_87654321"] [data-decision="decline"]')!;
    view.decline().click();
    await vi.waitFor(() => expect(view.submit).toHaveBeenCalledTimes(2));
    expect(view.submit.mock.calls[1]?.[1].sequence).toBe(2);
    view.controls.project({...proposal, proposalId:"proposal_new_12345", generation:8});
    expect(() => view.controls.project({...proposal, proposalId:"proposal_stale123", generation:7})).toThrow();
    view.dom.window.document.querySelector<HTMLButtonElement>('[data-proposal-id="proposal_new_12345"] [data-decision="confirm"]')!.click();
    await vi.waitFor(() => expect(view.submit).toHaveBeenCalledTimes(3));
    expect(view.submit.mock.calls[2]?.[1].sequence).toBe(1);
    view.dom.window.close();
  });

  it("starts a new participant sequence even when the host generation number is unchanged", async () => {
    const view = mount();
    view.controls.project(proposal);
    view.confirm().click();
    await vi.waitFor(() => expect(view.submit).toHaveBeenCalledOnce());
    view.controls.project({...proposal, status:"confirmed"});
    view.refresh("browser_fedcba9876543210");
    view.controls.project({...proposal, proposalId:"proposal_new_12345"});
    view.dom.window.document.querySelector<HTMLButtonElement>('[data-proposal-id="proposal_new_12345"] [data-decision="confirm"]')!.click();
    await vi.waitFor(() => expect(view.submit).toHaveBeenCalledTimes(2));
    expect(view.submit.mock.calls[1]?.[1].sequence).toBe(1);
    view.dom.window.close();
  });

  it.each(["version", "proposalId", "generation", "decision", "sequence"])("refuses a mismatched acknowledgment %s", async field => {
    const submit = vi.fn(async () => ({version:1, proposalId:proposal.proposalId, generation:7,
      decision:"confirm", sequence:1, [field]: field === "proposalId" ? "proposal_other123" : 2}));
    const view = mount(submit);
    view.controls.project(proposal);
    view.confirm().click();
    await vi.waitFor(() => expect(view.card().textContent).toContain("not confirmed"));
    expect(view.confirm().disabled).toBe(true);
    view.dom.window.close();
  });
});
