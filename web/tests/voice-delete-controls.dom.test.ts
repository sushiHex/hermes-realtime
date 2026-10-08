import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { VoiceDeleteControls } from "../src/voice-delete-controls";

const markup = readFileSync(fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");
const response = (state: "idle" | "pending" | "complete" | "unknown") => ({ version: 1, state });

function mount(options: {
  confirm: (message: string) => boolean;
  request: (path: string, token: string) => Promise<unknown>;
  clear: () => void;
}) {
  const dom = new JSDOM(markup);
  vi.stubGlobal("window", dom.window);
  const button = dom.window.document.querySelector<HTMLButtonElement>("#delete-voice-conversation");
  const status = dom.window.document.querySelector<HTMLOutputElement>("#voice-delete-status");
  if (!button || !status) throw new Error("voice delete controls missing");
  const controls = new VoiceDeleteControls(button, status, {
    ...options,
    credential: () => ({ token: "synthetic-token", participantIdentity: "participant-a" }),
    connected: () => true,
    pollIntervalMs: 100_000,
  });
  return { dom, button, status, controls };
}

afterEach(() => vi.unstubAllGlobals());

describe("voice conversation deletion control", () => {
  it("keeps delete disabled before capability status and when this host is unavailable", async () => {
    const requests: string[] = [];
    const { dom, button, status, controls } = mount({
      confirm: () => true,
      request: async (path) => {
        requests.push(path);
        return { version: 1, state: "unavailable" };
      },
      clear: () => undefined,
    });

    expect(button.disabled).toBe(true);
    await controls.delete();
    expect(requests).toEqual([]);
    await controls.refresh();
    expect(requests).toEqual(["/api/v1/voice-delete-status"]);
    expect(status.textContent).toContain("Voice conversation deletion is unavailable on this host.");
    expect(status.textContent).toContain("evidence capture keeps its own copy");
    expect(status.textContent).toContain("A deletion already started resumes when the companion is back.");
    expect(button.disabled).toBe(true);
    await controls.delete();
    expect(requests).toEqual(["/api/v1/voice-delete-status"]);
    controls.reset();
    dom.window.close();
  });

  it("requires explicit confirmation and clears the visible conversation only on local clear event", async () => {
    const requests: string[] = [];
    const confirmations: string[] = [];
    let allowed = false;
    let clears = 0;
    const { dom, button, status, controls } = mount({
      confirm: (message) => { confirmations.push(message); return allowed; },
      request: async (path) => {
        if (path === "/api/v1/voice-delete-status") return response("idle");
        requests.push(path);
        return response("pending");
      },
      clear: () => { clears += 1; },
    });
    expect(button.textContent).toContain("Delete this voice conversation");
    expect(dom.window.document.body.textContent).toContain(
      "What Hermes learned from it (memories and skills) stays and may still shape replies. There is no unlearning in the MVP.",
    );
    expect(dom.window.document.body.textContent).toContain(
      "Delegated tasks remain in Hermes and are managed with Hermes's own session controls.",
    );
    await controls.refresh();
    expect(button.disabled).toBe(false);

    await controls.delete();
    expect(requests).toEqual([]);
    expect(confirmations).toHaveLength(1);
    expect(confirmations[0]).toContain(
      "Delegated tasks remain in Hermes and are managed with Hermes's own session controls.",
    );
    allowed = true;
    await controls.delete();
    expect(requests).toEqual(["/api/v1/delete-voice-conversation"]);
    expect(status.textContent).toContain("pending");
    expect(status.textContent).not.toContain("deleted");
    expect(button.disabled).toBe(false);
    expect(clears).toBe(0);

    controls.cleared();
    expect(clears).toBe(1);
    controls.reset();
    dom.window.close();
  });

  it("reports complete only after server status and ignores an older complete response", async () => {
    let resolveOld!: (value: unknown) => void;
    const oldStatus = new Promise<unknown>((resolve) => { resolveOld = resolve; });
    let statusCalls = 0;
    const { dom, button, status, controls } = mount({
      confirm: () => true,
      request: async (path) => {
        if (path === "/api/v1/delete-voice-conversation") return response("pending");
        statusCalls += 1;
        return statusCalls === 1 ? response("idle") : statusCalls === 2 ? oldStatus : response("complete");
      },
      clear: () => undefined,
    });

    await controls.refresh();
    const stale = controls.refresh();
    await controls.delete();
    resolveOld(response("complete"));
    await stale;
    expect(status.textContent).toContain("pending");
    expect(button.disabled).toBe(false);

    await controls.refresh();
    expect(status.textContent).toBe("Voice conversation deleted.");
    expect(button.disabled).toBe(false);
    controls.reset();
    dom.window.close();
  });

  it("keeps a new delete in flight when an old delete settles after reset", async () => {
    let resolveOld!: (value: unknown) => void;
    let resolveNew!: (value: unknown) => void;
    const oldRequest = new Promise<unknown>((resolve) => { resolveOld = resolve; });
    const newRequest = new Promise<unknown>((resolve) => { resolveNew = resolve; });
    let deletes = 0;
    let statusRequests = 0;
    const { dom, button, status, controls } = mount({
      confirm: () => true,
      request: async (path) => {
        if (path === "/api/v1/voice-delete-status") {
          statusRequests += 1;
          return response("idle");
        }
        deletes += 1;
        return deletes === 1 ? oldRequest : newRequest;
      },
      clear: () => undefined,
    });
    await controls.refresh();
    const old = controls.delete();
    controls.reset();
    await controls.refresh();
    const current = controls.delete();
    await controls.refresh();
    expect(statusRequests).toBe(2);
    resolveOld(response("complete"));
    await old;
    expect(status.textContent).toBe("Starting deletion…");
    expect(button.disabled).toBe(true);
    await controls.refresh();
    expect(statusRequests).toBe(2);
    resolveNew(response("pending"));
    await current;
    expect(status.textContent).toContain("pending");
    controls.reset();
    dom.window.close();
  });

  it("lets a later conversation be deleted while an earlier delete is pending", async () => {
    const requests: string[] = [];
    const { dom, button, controls } = mount({
      confirm: () => true,
      request: async (path) => {
        requests.push(path);
        return response("pending");
      },
      clear: () => undefined,
    });
    await controls.refresh();
    expect(button.disabled).toBe(false);
    await controls.delete();
    await controls.delete();
    expect(requests.filter((path) => path === "/api/v1/delete-voice-conversation")).toHaveLength(2);
    controls.reset();
    dom.window.close();
  });

  it("reports an unreadable delete record as unknown, never as ready", async () => {
    const { dom, button, status, controls } = mount({
      confirm: () => true,
      request: async () => ({ version: 1, state: "unknown" }),
      clear: () => undefined,
    });
    await controls.refresh();
    expect(status.textContent).toContain("could not be confirmed");
    expect(status.textContent).not.toContain("Ready");
    expect(button.disabled).toBe(false);
    controls.reset();
    dom.window.close();
  });

  it("keeps polling a pending delete while the companion is away, with delete disabled", async () => {
    const states = ["pending", "unavailable"];
    const { dom, button, status, controls } = mount({
      confirm: () => true,
      request: async () => ({ version: 1, state: states.shift() ?? "unavailable" }),
      clear: () => undefined,
    });
    const timer = () => (controls as unknown as { timer: unknown }).timer;
    await controls.refresh();
    expect(timer()).not.toBeNull();
    await controls.refresh();
    expect(status.textContent).toContain("A deletion already started resumes when the companion is back.");
    expect(button.disabled).toBe(true);
    expect(timer()).not.toBeNull();
    controls.reset();
    dom.window.close();
  });

  it("does not poll an unavailable host it never saw a delete pending on", async () => {
    const { dom, controls } = mount({
      confirm: () => true,
      request: async () => ({ version: 1, state: "unavailable" }),
      clear: () => undefined,
    });
    await controls.refresh();
    expect((controls as unknown as { timer: unknown }).timer).toBeNull();
    controls.reset();
    dom.window.close();
  });

  it("never treats a delete answered with unknown as started", async () => {
    const requests: string[] = [];
    const { dom, status, controls } = mount({
      confirm: () => true,
      request: (path) => {
        requests.push(path);
        if (path === "/api/v1/delete-voice-conversation") return Promise.resolve(response("unknown"));
        // The recovery refresh stays in flight, so the status shows the delete's own outcome.
        return requests.length === 1 ? Promise.resolve(response("idle")) : new Promise<unknown>(() => undefined);
      },
      clear: () => undefined,
    });
    await controls.refresh();
    await controls.delete();
    expect(status.textContent).toBe("Deletion could not be confirmed. Checking status.");
    expect(requests).toEqual([
      "/api/v1/voice-delete-status",
      "/api/v1/delete-voice-conversation",
      "/api/v1/voice-delete-status",
    ]);
    controls.reset();
    dom.window.close();
  });

  it("states that a Hermes /branch copy keeps a deletion pending", async () => {
    const confirmations: string[] = [];
    const { dom, controls } = mount({
      confirm: (message) => { confirmations.push(message); return false; },
      request: async () => response("idle"),
      clear: () => undefined,
    });
    const limit = "A copy made with Hermes /branch is a separate conversation: deletion stays pending until you delete that copy in Hermes.";
    expect(dom.window.document.body.textContent).toContain(limit);
    await controls.refresh();
    await controls.delete();
    expect(confirmations[0]).toContain(limit);
    controls.reset();
    dom.window.close();
  });

  it("ignores an older pending refresh after a newer refresh confirms completion", async () => {
    let resolveOld!: (value: unknown) => void;
    const oldStatus = new Promise<unknown>((resolve) => { resolveOld = resolve; });
    let statusCalls = 0;
    const { dom, button, status, controls } = mount({
      confirm: () => true,
      request: async () => {
        statusCalls += 1;
        return statusCalls === 1 ? oldStatus : response("complete");
      },
      clear: () => undefined,
    });

    const old = controls.refresh();
    await controls.refresh();
    expect(status.textContent).toBe("Voice conversation deleted.");
    expect(button.disabled).toBe(false);
    resolveOld(response("pending"));
    await old;
    expect(status.textContent).toBe("Voice conversation deleted.");
    expect(button.disabled).toBe(false);
    controls.reset();
    dom.window.close();
  });
});
