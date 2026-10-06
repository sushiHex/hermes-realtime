import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { VoiceDeleteControls } from "../src/voice-delete-controls";

const markup = readFileSync(fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");
const response = (state: "idle" | "pending" | "complete") => ({ version: 1, state });

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
  it("requires explicit confirmation and clears the visible conversation only on local clear event", async () => {
    const requests: string[] = [];
    const confirmations: string[] = [];
    let allowed = false;
    let clears = 0;
    const { dom, button, status, controls } = mount({
      confirm: (message) => { confirmations.push(message); return allowed; },
      request: async (path) => { requests.push(path); return response("pending"); },
      clear: () => { clears += 1; },
    });
    expect(button.textContent).toContain("Delete this voice conversation");
    expect(dom.window.document.body.textContent).toContain(
      "What Hermes learned from it (memories and skills) stays and may still shape replies. There is no unlearning in the MVP.",
    );
    expect(dom.window.document.body.textContent).toContain(
      "Delegated tasks remain in Hermes and are managed with Hermes's own session controls.",
    );

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
    expect(button.disabled).toBe(true);
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
        return statusCalls === 1 ? oldStatus : response("complete");
      },
      clear: () => undefined,
    });

    const stale = controls.refresh();
    await controls.delete();
    resolveOld(response("complete"));
    await stale;
    expect(status.textContent).toContain("pending");
    expect(button.disabled).toBe(true);

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
          return response("complete");
        }
        deletes += 1;
        return deletes === 1 ? oldRequest : newRequest;
      },
      clear: () => undefined,
    });
    const old = controls.delete();
    controls.reset();
    const current = controls.delete();
    await controls.refresh();
    expect(statusRequests).toBe(0);
    resolveOld(response("complete"));
    await old;
    expect(status.textContent).toBe("Starting deletion…");
    expect(button.disabled).toBe(true);
    await controls.refresh();
    expect(statusRequests).toBe(0);
    resolveNew(response("pending"));
    await current;
    expect(status.textContent).toContain("pending");
    controls.reset();
    dom.window.close();
  });
});
