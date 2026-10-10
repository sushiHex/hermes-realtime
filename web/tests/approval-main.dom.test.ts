import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const media = vi.hoisted(() => ({ rooms: [] as Array<{ emit(event: string, value: unknown): void }> }));
vi.mock("livekit-client", async (original) => ({
  ...await original<typeof import("livekit-client")>(),
  Room: class {
    localParticipant = { trackPublications: new Map(), setMicrophoneEnabled: async () => undefined };
    handlers = new Map<string, (value: unknown) => void>();
    constructor() { media.rooms.push(this); }
    on(event: string, handler: (value: unknown) => void) { this.handlers.set(event, handler); return this; }
    emit(event: string, value: unknown) { this.handlers.get(event)?.(value); }
    async connect() {}
    async disconnect() {}
  },
}));

let dom: JSDOM;
async function mount(
  decide: () => Promise<Response> = async () => Response.json({ version: 1 }),
  stopRequest: () => Promise<Response> = async () => Response.json({ version: 1 }),
) {
  vi.resetModules();
  media.rooms.length = 0;
  dom = new JSDOM(readFileSync(new URL("../index.html", import.meta.url), "utf8"), {
    url: "http://localhost/", pretendToBeVisual: true,
  });
  for (const key of ["window", "document", "HTMLElement", "HTMLMediaElement", "Option"] as const) {
    vi.stubGlobal(key, key === "window" ? dom.window : dom.window[key]);
  }
  vi.stubGlobal("navigator", { mediaDevices: { enumerateDevices: async () => [] } });
  vi.spyOn(dom.window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  vi.spyOn(console, "info").mockImplementation(() => {});
  let emit: ((events: unknown[]) => void) | null = null;
  let sequence = 0;
  let lease = 0;
  let refresh!: () => void;
  const setTimeout = dom.window.setTimeout.bind(dom.window);
  vi.spyOn(dom.window, "setTimeout").mockImplementation((handler, milliseconds, ...arguments_) => {
    if (milliseconds === 30000 && typeof handler === "function") refresh = () => handler(...arguments_);
    return setTimeout(handler, milliseconds, ...arguments_);
  });
  const mintCredential = () => {
    lease += 1;
    sequence = 0;
    return { version: 1, url: "wss://livekit.test", roomName: "synthetic-room",
      participantIdentity: `browser_${lease.toString().padStart(16, "0")}`,
      workerIdentity: "worker_hermes_browser", expiresInSeconds: 60, token: "synthetic.token.value" };
  };
  const decisions: Array<{ approvalId: string; decision: string; sequence: number }> = [];
  vi.stubGlobal("fetch", vi.fn((path: string, options: RequestInit) => {
    if (path === "/api/v1/stable-bootstrap") {
      return Response.json(mintCredential());
    }
    if (path === "/api/v1/stable-rebind") {
      lease += 1;
      return Response.json({ version: 1, url: "wss://livekit.test", roomName: "synthetic-room",
        participantIdentity: `browser_${lease.toString().padStart(16, "0")}`,
        workerIdentity: "worker_hermes_browser", expiresInSeconds: 60, token: "synthetic.rebound.token" });
    }
    if (path === "/api/v1/projection-resync") return Response.json(mintCredential());
    if (path === "/api/v1/refresh") return Response.json({ version: 1,
      url: "wss://livekit.test", roomName: "synthetic-room",
      participantIdentity: `browser_${lease.toString().padStart(16, "0")}`,
      workerIdentity: "worker_hermes_browser", expiresInSeconds: 60, token: "synthetic.refreshed.token" });
    if (path === "/api/v1/media") return Response.json({ version: 1 });
    if (path === "/api/v1/stop") return stopRequest();
    if (path === "/api/v1/voices") return Response.json({ version: 1, voices: [], selectedVoice: null });
    if (path === "/api/v1/events") return new Promise<Response>((resolve) => {
      emit = (events) => resolve(Response.json({ version: 1, events }));
    });
    if (path === "/api/v1/approval") { decisions.push(JSON.parse(options.body as string)); return decide(); }
    if (path === "/api/v1/voice-delete/status") return Response.json({ version: 1, available: true });
    if (path === "/api/v1/voice-delete") return Response.json({ version: 1 });
    return new Response(null, { status: 503 });
  }));
  await import("../src/main");
  const toggle = dom.window.document.querySelector<HTMLButtonElement>("#session-toggle")!;
  async function connect() {
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLInputElement>("#typed-input")!.disabled).toBe(false));
  }
  async function stop() { toggle.click(); await vi.waitFor(() => expect(toggle.textContent).toBe("Connect")); }
  async function rebind() {
    media.rooms.at(-1)!.emit("participantDisconnected", { identity: "worker_hermes_browser" });
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    await connect();
  }
  async function event(kind: string, data: object) {
    await batch([{ kind, data }]);
  }
  async function batch(updates: readonly { kind: string; data: object }[]) {
    await vi.waitFor(() => expect(emit).not.toBeNull(), { timeout: 2000 });
    const dispatch = emit!;
    emit = null;
    dispatch(updates.map(({ kind, data }) => {
      sequence += 1;
      return { sequence, kind, data, monotonicMs: sequence * 1000 };
    }));
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const approval = (state = "pending", actionable = true, approvalId = "approval_fixture_0001") => event("approval_state", {
    approvalId, taskId: "task_fixture", state, actionable,
    command: "synthetic command", description: "Synthetic approval",
  });
  const card = () => dom.window.document.querySelector<HTMLLIElement>('[data-operation="approval"]')!;
  const approve = () => card().querySelector<HTMLButtonElement>(".approve")!;
  await connect();
  await approval();
  return { connect, stop, rebind, event, batch, approval, card, approve, decisions, toggle,
    refresh: () => refresh() };
}

afterEach(() => { dom?.window.close(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("durable mounted approval identity", () => {
  it.each(["pending", "approve", "reject"])("reconciles %s replay into the same card after stop/start", async (state) => {
    const { connect, stop, approval, card, approve, decisions } = await mount();
    const original = card();
    await stop();
    expect(original.dataset.status).toBe("pending");
    expect(original.textContent).not.toContain("Session stopped before a decision");
    expect(approve().disabled).toBe(true);
    await connect();
    expect(approve().disabled).toBe(true);
    await approval(state, state === "pending");
    expect(dom.window.document.querySelectorAll('[data-operation="approval"]')).toHaveLength(1);
    expect(card()).toBe(original);
    expect(card().dataset.status).toBe(state);
    expect(approve().disabled).toBe(state !== "pending");
    if (state === "pending") {
      approve().click();
      await vi.waitFor(() => expect(decisions).toEqual([
        { approvalId: "approval_fixture_0001", decision: "approve", sequence: 1 },
      ]));
    } else expect(card().textContent).toContain(state === "approve" ? "Approved." : "Rejected.");
  });

  it("keeps a disabled pending approval through voice clear and forgets settled identity", async () => {
    const { connect, stop, event, approval, card, approve } = await mount();
    const original = card();
    await stop();
    await connect();
    await event("voice_conversation_cleared", {});
    expect(card()).toBe(original);
    expect(approve().disabled).toBe(true);
    await approval("approve", false);
    await event("voice_conversation_cleared", {});
    expect(dom.window.document.querySelectorAll('[data-operation="approval"]')).toHaveLength(0);
    await approval();
    expect(card()).not.toBe(original);
  });

  it.each(["settlement", "replacement"])("late failed decision does not restore controls after %s", async (change) => {
    let reject!: (reason: Error) => void;
    const { connect, stop, approval, card, approve } = await mount(() => new Promise<Response>((_resolve, failure) => { reject = failure; }));
    const original = card();
    approve().click();
    await vi.waitFor(() => expect(reject).toBeDefined());
    if (change === "settlement") await approval("reject", false);
    else { await stop(); await connect(); }
    reject(new Error("synthetic old request failure"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(card()).toBe(original);
    expect(approve().disabled).toBe(true);
    if (change === "settlement") expect(card().textContent).toContain("Rejected.");
    else { await approval(); expect(approve().disabled).toBe(false); }
  });

  it("pending replay keeps one decision in flight on its current card", async () => {
    let resolve!: (response: Response) => void;
    const { approval, approve, decisions } = await mount(() => new Promise<Response>((success) => { resolve = success; }));
    approve().click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    await approval();
    expect(approve().disabled).toBe(true);
    approve().click();
    expect(decisions).toHaveLength(1);
    resolve(Response.json({ version: 1 }));
    await new Promise((success) => setTimeout(success, 0));
    expect(approve().disabled).toBe(true);
    await approval();
    expect(approve().disabled).toBe(false);
  });

  it("native reconnect cannot restore settled approval authority", async () => {
    const { approval, approve, toggle, card } = await mount();
    await approval("approve", false);
    media.rooms[0]!.emit("connectionStateChanged", "reconnecting");
    expect(approve().disabled).toBe(true);
    media.rooms[0]!.emit("connectionStateChanged", "connected");
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLInputElement>("#typed-input")!.disabled).toBe(false));
    expect(toggle.textContent).toBe("Disconnect");
    expect(approve().disabled).toBe(true);
    expect(card().textContent).toContain("Approved.");
  });

  it("disables approval authority synchronously while stop acknowledgment is held", async () => {
    let release!: (response: Response) => void;
    const { approve, decisions, toggle } = await mount(undefined, () => new Promise<Response>((resolve) => { release = resolve; }));
    toggle.click();
    expect(approve().disabled).toBe(true);
    approve().click();
    expect(decisions).toEqual([]);
    await vi.waitFor(() => expect(release).toBeDefined());
    release(Response.json({ version: 1 }));
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
  });

  it("refuses a queued approval while remote stop acknowledgment is held", async () => {
    let release!: (response: Response) => void;
    let stopped!: (response: Response) => void;
    const { approval, approve, decisions, toggle } = await mount(
      () => new Promise<Response>((resolve) => { release = resolve; }),
      () => new Promise<Response>((resolve) => { stopped = resolve; }),
    );
    await approval("pending", true, "approval_fixture_0002");
    approve().click();
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    toggle.click();
    await vi.waitFor(() => expect(stopped).toBeDefined());
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(decisions).toHaveLength(1);
    expect(vi.mocked(console.info).mock.calls
      .filter(([message]) => typeof message === "string" && message.startsWith("[approval-decision-refused]")))
      .toEqual([['[approval-decision-refused] {"count":1,"category":"connection_changed"}']]);
    stopped(Response.json({ version: 1 }));
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
  });

  it("does not retarget a queued decision to replacement connection authority", async () => {
    let release!: (response: Response) => void;
    let first = true;
    const { rebind, approval, approve, decisions } = await mount(() => {
      if (first) { first = false; return new Promise<Response>((resolve) => { release = resolve; }); }
      return Promise.resolve(Response.json({ version: 1 }));
    });
    await approval("pending", true, "approval_fixture_0002");
    approve().click();
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    await rebind();
    await approval();
    await approval("pending", true, "approval_fixture_0002");
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(decisions).toEqual([
      { approvalId: "approval_fixture_0001", decision: "approve", sequence: 1 },
    ]);
    expect(vi.mocked(console.info).mock.calls
      .filter(([message]) => typeof message === "string" && message.startsWith("[approval-decision-refused]")))
      .toEqual([['[approval-decision-refused] {"count":1,"category":"connection_changed"}']]);
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions).toEqual([
      { approvalId: "approval_fixture_0001", decision: "approve", sequence: 1 },
      { approvalId: "approval_fixture_0002", decision: "approve", sequence: 2 },
    ]));
  });

  it("forgets settled approval identity with normal bounded history eviction", async () => {
    const { approval, batch, card } = await mount();
    const original = card();
    await approval("approve", false);
    const events = Array.from({ length: 140 }, (_, index) => ({
      kind: "transcript_final", data: { role: "assistant", text: `Synthetic history line ${index}` },
    }));
    await batch(events);
    expect(original.isConnected).toBe(false);
    await approval();
    expect(card()).not.toBe(original);
    expect(dom.window.document.querySelectorAll('[data-operation="approval"]')).toHaveLength(1);
  });

  it.each(["settled", "evicted"])("refuses a queued decision whose card was %s before send", async (change) => {
    let release!: (response: Response) => void;
    let first = true;
    const { approval, batch, approve, decisions } = await mount(() => {
      if (first) { first = false; return new Promise<Response>((resolve) => { release = resolve; }); }
      return Promise.resolve(Response.json({ version: 1 }));
    });
    await approval("pending", true, "approval_fixture_0002");
    approve().click();
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    await approval("reject", false, "approval_fixture_0002");
    if (change === "evicted") await batch(Array.from({ length: 140 }, (_, index) => ({
      kind: "transcript_final", data: { role: "assistant", text: `Synthetic history line ${index}` },
    })));
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(decisions).toEqual([
      { approvalId: "approval_fixture_0001", decision: "approve", sequence: 1 },
    ]);
    expect(vi.mocked(console.info).mock.calls
      .filter(([message]) => typeof message === "string" && message.startsWith("[approval-decision-refused]")))
      .toEqual([['[approval-decision-refused] {"count":1,"category":"connection_changed"}']]);
  });

  it("old successful approval acknowledgment cannot advance the new lease sequence", async () => {
    let release!: (response: Response) => void;
    let first = true;
    const { connect, stop, approval, approve, decisions } = await mount(() => {
      if (first) { first = false; return new Promise<Response>((resolve) => { release = resolve; }); }
      return Promise.resolve(Response.json({ version: 1 }));
    });
    approve().click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    await stop();
    await connect();
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    await approval("pending", true, "approval_fixture_0002");
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions).toEqual([
      { approvalId: "approval_fixture_0001", decision: "approve", sequence: 1 },
      { approvalId: "approval_fixture_0002", decision: "approve", sequence: 1 },
    ]));
  });

  it("authoritative settlement before HTTP success still commits the admitted sequence", async () => {
    let release!: (response: Response) => void;
    let first = true;
    const { approval, approve, decisions, card } = await mount(() => {
      if (first) { first = false; return new Promise<Response>((resolve) => { release = resolve; }); }
      return Promise.resolve(Response.json({ version: 1 }));
    });
    approve().click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    await approval("approve", false);
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(card().textContent).toContain("Approved.");
    expect(approve().disabled).toBe(true);
    await approval("pending", true, "approval_fixture_0002");
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions.map((value: any) => value.sequence)).toEqual([1, 2]));
  });

  it("projection resync refuses a successful old acknowledgment before new sequence allocation", async () => {
    let release!: (response: Response) => void;
    let first = true;
    const { event, approval, approve, decisions } = await mount(() => {
      if (first) { first = false; return new Promise<Response>((resolve) => { release = resolve; }); }
      return Promise.resolve(Response.json({ version: 1 }));
    });
    approve().click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    for (let index = 0; index < 5; index += 1) await event("synthetic_invalid_event", {});
    await vi.waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([path]) => path === "/api/v1/media")).toHaveLength(2));
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    await approval("pending", true, "approval_fixture_0002");
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions.map((value: any) => value.sequence)).toEqual([1, 1]));
  });

  it("same-participant credential refresh preserves a successful decision acknowledgment", async () => {
    let release!: (response: Response) => void;
    let first = true;
    const { refresh, approval, approve, decisions } = await mount(() => {
      if (first) { first = false; return new Promise<Response>((resolve) => { release = resolve; }); }
      return Promise.resolve(Response.json({ version: 1 }));
    });
    approve().click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    refresh();
    await vi.waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([path]) => path === "/api/v1/refresh")).toHaveLength(1));
    await new Promise((resolve) => setTimeout(resolve, 0));
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    await approval("pending", true, "approval_fixture_0002");
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions.map((value: any) => value.sequence)).toEqual([1, 2]));
  });

  it("ordinary rebind does not authorize a stale card from the preceding lease", async () => {
    const { stop, connect, rebind, card, approve } = await mount();
    const original = card();
    await stop();
    await connect();
    await rebind();
    expect(card()).toBe(original);
    expect(approve().disabled).toBe(true);
  });

  it("ordinary rebind transfers a retained pending card without requiring approval replay", async () => {
    const { rebind, card, approve, decisions } = await mount();
    const original = card();
    await rebind();
    expect(card()).toBe(original);
    expect(approve().disabled).toBe(false);
    approve().click();
    await vi.waitFor(() => expect(decisions).toEqual([
      { approvalId: "approval_fixture_0001", decision: "approve", sequence: 1 },
    ]));
  });

  it("ordinary rebind preserves a delayed accepted acknowledgment and its next sequence", async () => {
    let release!: (response: Response) => void;
    let first = true;
    const { rebind, approval, approve, decisions } = await mount(() => {
      if (first) { first = false; return new Promise<Response>((resolve) => { release = resolve; }); }
      return Promise.resolve(Response.json({ version: 1 }));
    });
    approve().click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    await rebind();
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    await approval("pending", true, "approval_fixture_0002");
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(decisions.map((value: any) => value.sequence)).toEqual([1, 2]));
  });

  it.each(["success", "failure"])("old %s cannot release a new same-card pending decision", async (outcome) => {
    let oldSuccess!: (response: Response) => void;
    let oldFailure!: (reason: Error) => void;
    let current!: (response: Response) => void;
    let first = true;
    const { connect, stop, approval, approve, decisions, card } = await mount(() => {
      if (first) { first = false; return new Promise<Response>((resolve, reject) => { oldSuccess = resolve; oldFailure = reject; }); }
      return new Promise<Response>((resolve) => { current = resolve; });
    });
    const original = card();
    approve().click();
    await vi.waitFor(() => expect(decisions).toHaveLength(1));
    await stop();
    await connect();
    await approval();
    expect(approve().disabled).toBe(false);
    approve().click();
    await vi.waitFor(() => expect(decisions).toHaveLength(2));
    if (outcome === "success") oldSuccess(Response.json({ version: 1 }));
    else oldFailure(new Error("synthetic retired acknowledgment failure"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(card()).toBe(original);
    expect(approve().disabled).toBe(true);
    expect(card().textContent).not.toContain("Decision failed");
    await approval();
    expect(approve().disabled).toBe(true);
    current(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(approve().disabled).toBe(true);
    await approval("approve", false);
    expect(card().textContent).toContain("Approved.");
    expect(decisions.map((value: any) => value.sequence)).toEqual([1, 1]);
  });
});
