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

const credential = {
  version: 1, url: "wss://livekit.test", roomName: "synthetic-room",
  participantIdentity: "browser_0123456789abcdef", workerIdentity: "worker_hermes_browser",
  expiresInSeconds: 60, token: "synthetic.token.value",
};
let dom: JSDOM;

async function mount(inputRequest: (body: { sequence: number; text: string }, signal: AbortSignal) => Promise<Response> = async () => Response.json({ version: 1 }), oneShot = false) {
  vi.resetModules();
  media.rooms.length = 0;
  dom = new JSDOM(readFileSync(new URL("../index.html", import.meta.url), "utf8"), {
    url: oneShot ? `http://localhost/#bootstrap=${"a".repeat(43)}` : "http://localhost/", pretendToBeVisual: true,
  });
  for (const key of ["window", "document", "HTMLElement", "HTMLMediaElement", "Option"] as const) {
    vi.stubGlobal(key, key === "window" ? dom.window : dom.window[key]);
  }
  vi.stubGlobal("navigator", { mediaDevices: { enumerateDevices: async () => [] } });
  vi.spyOn(dom.window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  dom.window.HTMLElement.prototype.scrollIntoView = () => {};
  vi.spyOn(console, "info").mockImplementation(() => {});
  let refreshCredential: (() => void) | null = null;
  const setTimeout = dom.window.setTimeout.bind(dom.window);
  vi.spyOn(dom.window, "setTimeout").mockImplementation((handler, milliseconds, ...arguments_) => {
    if (milliseconds === 30000 && typeof handler === "function") refreshCredential = () => handler(...arguments_);
    return setTimeout(handler, milliseconds, ...arguments_);
  });
  let emit: ((events: unknown[]) => void) | null = null;
  let sequence = 0;
  const requests: Array<{ sequence: number; text: string }> = [];
  const approvals: Array<{ sequence: number; approvalId: string; decision: string }> = [];
  vi.stubGlobal("fetch", vi.fn((path: string, options: RequestInit) => {
    if (path === "/api/v1/stable-bootstrap" || path === "/api/v1/bootstrap") return Response.json(credential);
    if (path === "/api/v1/stable-rebind") {
      if (JSON.parse(options.body as string).freshView === true) sequence = 0;
      return Response.json({ ...credential, participantIdentity: "browser_fedcba9876543210", token: "synthetic.rebound.token" });
    }
    if (path === "/api/v1/refresh") return Response.json({ ...credential, token: "synthetic.rotated.token" });
    if (path === "/api/v1/media" || path === "/api/v1/stop") return Response.json({ version: 1 });
    if (path === "/api/v1/approval") {
      approvals.push(JSON.parse(options.body as string));
      return Response.json({ version: 1 });
    }
    if (path === "/api/v1/voices") return Response.json({ version: 1, voices: [], selectedVoice: null });
    if (path === "/api/v1/events") return new Promise<Response>((resolve) => {
      emit = (events) => resolve(Response.json({ version: 1, events }));
    });
    if (path === "/api/v1/input") {
      const body = JSON.parse(options.body as string);
      requests.push(body);
      return inputRequest(body, options.signal as AbortSignal);
    }
    return new Response(null, { status: 503 });
  }));
  await import("../src/main");
  const toggle = dom.window.document.querySelector<HTMLButtonElement>("#session-toggle")!;
  toggle.click();
  await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
  async function event(kind: string, data: object) {
    await vi.waitFor(() => expect(emit).not.toBeNull());
    const dispatch = emit!;
    emit = null;
    sequence += 1;
    dispatch([{ sequence, kind, monotonicMs: sequence * 1000, data }]);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const task = (taskId: string | null, status: string, reason?: string) => event("task_state", { taskId, status, ...(reason === undefined ? {} : { reason }) });
  const approval = async (approvalId: string) => {
    await event("approval_state", { approvalId, state: "pending", actionable: true, taskId: "task_fixture", command: "synthetic command", description: "Synthetic approval" });
    dom.window.document.querySelector<HTMLButtonElement>('[data-operation="approval"]:last-child .approve')!.click();
    await vi.waitFor(() => expect(approvals.at(-1)?.approvalId).toBe(approvalId));
    await new Promise((resolve) => setTimeout(resolve, 0));
  };
  const card = (taskId = "task_fixture") => dom.window.document.querySelector<HTMLLIElement>(`#transcript li[data-task-id="${taskId}"]`)!;
  const cancel = (taskId = "task_fixture") => card(taskId)?.querySelector<HTMLButtonElement>("button")!;
  const submit = (text: string) => {
    dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.value = text;
    dom.window.document.querySelector<HTMLFormElement>("#typed-form")!.dispatchEvent(new dom.window.Event("submit", { cancelable: true }));
  };
  return { requests, approvals, approval, task, card, cancel, submit, toggle, refresh: () => refreshCredential!() };
}

afterEach(() => {
  dom?.window.close();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("mounted task controls", () => {
  it("preserves admitted typed sequence across ordinary browser rebind", async () => {
    const { requests, submit, toggle } = await mount();
    submit("Synthetic input before rebind");
    await vi.waitFor(() => expect(dom.window.document.querySelector("#markers")!.textContent).toContain("typed_input_admitted"));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector("#connection-status")!.getAttribute("data-state")).toBe("typed-only"));
    submit("Synthetic input after rebind");
    await vi.waitFor(() => expect(requests.map((item) => item.sequence)).toEqual([1, 2]));
    const rebind = vi.mocked(fetch).mock.calls.find(([path]) => path === "/api/v1/stable-rebind")!;
    expect(JSON.parse(rebind[1]!.body as string)).not.toHaveProperty("freshView");
  });

  it("preserves admitted approval sequence across ordinary browser rebind", async () => {
    const { approvals, approval, toggle } = await mount();
    await approval("approval_fixture_first");
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector("#connection-status")!.getAttribute("data-state")).toBe("typed-only"));
    await approval("approval_fixture_second");
    expect(approvals.map((item) => item.sequence)).toEqual([1, 2]);
  });

  it("resets admitted input and approval counters on fresh-view recovery", async () => {
    const { requests, approvals, approval, submit, toggle } = await mount(async (body) => {
      if (body.text === "Synthetic uncertain input") throw new Error("synthetic lost acknowledgment");
      return Response.json({ version: 1 });
    });
    await approval("approval_fixture_first");
    submit("Synthetic admitted input");
    await vi.waitFor(() => expect(dom.window.document.querySelector("#markers")!.textContent).toContain("typed_input_admitted"));
    submit("Synthetic uncertain input");
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector("#connection-status")!.getAttribute("data-state")).toBe("typed-only"));
    submit("Synthetic input after reset");
    await approval("approval_fixture_second");
    await vi.waitFor(() => expect(requests.map((item) => item.sequence)).toEqual([1, 2, 1]));
    expect(approvals.map((item) => item.sequence)).toEqual([1, 1]);
  });

  it("requires a fresh-view reconnect after a server-admitted input loses its acknowledgment", async () => {
    let reject!: (reason: Error) => void;
    let spent = false;
    const { requests, task, cancel, submit, toggle } = await mount(() => {
      if (spent) return Promise.resolve(Response.json({ version: 1 }));
      spent = true;
      return new Promise((_resolve, refuse) => { reject = refuse; });
    });
    await task("task_fixture", "active");
    cancel().click();
    submit("Synthetic queued input");
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    // The request spent sequence 1 on the server before its response was lost.
    reject(new Error("synthetic lost acknowledgment"));
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    expect(dom.window.document.querySelector("#transcript")!.textContent).toContain("The previous command may have been admitted.");
    expect(cancel().disabled).toBe(true);
    submit("Synthetic later input");
    cancel().click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(requests).toHaveLength(1);
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector("#connection-status")!.getAttribute("data-state")).toBe("typed-only"));
    const rebind = vi.mocked(fetch).mock.calls.find(([path]) => path === "/api/v1/stable-rebind")!;
    expect(JSON.parse(rebind[1]!.body as string)).toMatchObject({ freshView: true });
    expect(toggle.textContent).toBe("Disconnect");
    submit("Synthetic input after fresh view");
    await vi.waitFor(() => expect(requests).toEqual([
      { sequence: 1, text: "cancel task: task_fixture" },
      { sequence: 1, text: "Synthetic input after fresh view" },
    ]));
  });

  it.each(["success", "failure"])("does not disconnect a replacement binding on late old input %s", async (outcome) => {
    let release!: (response: Response) => void;
    let reject!: (error: Error) => void;
    let first = true;
    const { requests, task, cancel, toggle, submit } = await mount(() => {
      if (!first) return Promise.resolve(Response.json({ version: 1 }));
      first = false;
      return new Promise((resolve, refuse) => { release = resolve; reject = refuse; });
    });
    await task("task_fixture", "active");
    cancel().click();
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector("#connection-status")!.getAttribute("data-state")).toBe("typed-only"));
    const rebind = vi.mocked(fetch).mock.calls.find(([path]) => path === "/api/v1/stable-rebind")!;
    expect(JSON.parse(rebind[1]!.body as string)).toMatchObject({ freshView: true });
    if (outcome === "success") release(Response.json({ version: 1 }));
    else reject(new Error("synthetic old acknowledgment failure"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toggle.textContent).toBe("Disconnect");
    expect(dom.window.document.querySelectorAll('[data-operation="input-control"]')).toHaveLength(1);
    expect(dom.window.document.querySelector("#markers")!.textContent).not.toContain("typed_input_admitted");
    submit("Synthetic input after unknown admission and reconnect");
    await vi.waitFor(() => expect(requests.map((item) => item.sequence)).toEqual([1, 1]));
    expect(toggle.textContent).toBe("Disconnect");
  });

  it("retains remote stop retry authority after late uncertain input failure", async () => {
    let reject!: (error: Error) => void;
    const { requests, submit, toggle } = await mount(() => new Promise((_resolve, refuse) => { reject = refuse; }));
    const dispatch = vi.mocked(fetch).getMockImplementation()!;
    let stops = 0;
    vi.mocked(fetch).mockImplementation(async (path, options) => {
      if (path === "/api/v1/stop") return Response.json({ version: 1 }, { status: ++stops === 1 ? 503 : 200 });
      return dispatch(path, options);
    });
    submit("Synthetic input before stop");
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector("#markers")!.textContent).toContain("session_stop_failed"));
    reject(new Error("synthetic late lost acknowledgment"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toggle.textContent).toBe("Disconnect");
    toggle.click();
    await vi.waitFor(() => expect(stops).toBe(2));
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    expect(dom.window.document.querySelectorAll('[data-operation="input-control"]')).toHaveLength(0);
  });

  it("requires a fresh launch after uncertain input in a one-shot session", async () => {
    const { requests, submit, toggle } = await mount(async () => { throw new Error("synthetic lost response"); }, true);
    submit("Synthetic one-shot input");
    await vi.waitFor(() => expect(toggle.textContent).toBe("Fresh launch required"));
    expect(toggle.disabled).toBe(true);
    expect(dom.window.document.querySelector("#transcript")!.textContent).toContain("Open a fresh launch before sending more input.");
    submit("Synthetic attempted repeat");
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(requests).toHaveLength(1);
  });

  it("routes the real card click once to its exact task through admitted typed input", async () => {
    let release!: (response: Response) => void;
    const { requests, task, cancel, card } = await mount(() => new Promise((resolve) => { release = resolve; }));
    await task("task_fixture", "active");
    expect(cancel()).not.toBeNull();
    cancel().click();
    cancel().click();
    await vi.waitFor(() => expect(requests).toEqual([{ sequence: 1, text: "cancel task: task_fixture" }]));
    expect(card().dataset.status).toBe("active");
    release(Response.json({ version: 1 }));
    await vi.waitFor(() => expect(cancel().disabled).toBe(false));
  });

  it.each(["cancelling", "completed", "interrupted", "failed"])("disables cancellation for %s work", async (status) => {
    const { requests, task, cancel } = await mount();
    await task("task_fixture", "active");
    await task("task_fixture", status);
    expect(cancel().disabled).toBe(true);
    cancel().click();
    expect(requests).toEqual([]);
  });

  it("serializes card and composer submissions with distinct sequences", async () => {
    let release!: (response: Response) => void;
    const { requests, task, cancel, submit } = await mount((body) => body.sequence === 1
      ? new Promise((resolve) => { release = resolve; }) : Promise.resolve(Response.json({ version: 1 })));
    await task("task_fixture", "active");
    cancel().click();
    submit("Synthetic follow-up");
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    release(Response.json({ version: 1 }));
    await vi.waitFor(() => expect(requests).toEqual([
      { sequence: 1, text: "cancel task: task_fixture" }, { sequence: 2, text: "Synthetic follow-up" },
    ]));
  });

  it("refuses queued and late-return input after local disconnection", async () => {
    let release!: (response: Response) => void;
    const { requests, task, cancel, submit, toggle } = await mount(() => new Promise((resolve) => { release = resolve; }));
    await task("task_fixture", "active");
    cancel().click();
    submit("Synthetic queued input");
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(cancel().disabled).toBe(true));
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(requests).toHaveLength(1);
    expect(dom.window.document.querySelector("#markers")!.textContent).not.toContain("typed_input_admitted");
    expect(dom.window.document.querySelector("#transcript")!.textContent).not.toContain("cancel task:");
    expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.value).toBe("Synthetic queued input");
    expect(toggle.textContent).toBe("Connect");
  });

  it("bounds the shared input queue while a prior admission is pending", async () => {
    let release!: (response: Response) => void;
    const { requests, submit } = await mount((body) => body.sequence === 1
      ? new Promise((resolve) => { release = resolve; }) : Promise.resolve(Response.json({ version: 1 })));
    for (let index = 0; index < 10; index += 1) submit(`Synthetic input ${index}`);
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    release(Response.json({ version: 1 }));
    await vi.waitFor(() => expect(requests).toHaveLength(8));
    expect(requests.map((request) => request.sequence)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    const refusals = vi.mocked(console.info).mock.calls.map(([line]) => String(line))
      .filter((line) => line.startsWith("[input-control-refusal]"));
    expect(refusals).toEqual([
      '[input-control-refusal] {"kind":"typed-input","category":"capacity","pending_count":8}',
      '[input-control-refusal] {"kind":"typed-input","category":"capacity","pending_count":8}',
    ]);
  });

  it("refuses composer submission while disconnected", async () => {
    const { requests, submit, toggle } = await mount();
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    submit("Synthetic disconnected input");
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(requests).toEqual([]);
    expect(vi.mocked(console.info).mock.calls.map(([line]) => String(line))).toContain(
      '[input-control-refusal] {"kind":"typed-input","category":"disconnected","pending_count":0}',
    );
  });

  it("rejects old queued and late input after native reconnect to the same room", async () => {
    let release!: (response: Response) => void;
    const { requests, task, cancel, submit, toggle } = await mount(() => new Promise((resolve) => { release = resolve; }));
    await task("task_fixture", "active");
    cancel().click();
    submit("Synthetic input before native reconnect");
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    media.rooms[0]!.emit("connectionStateChanged", "reconnecting");
    media.rooms[0]!.emit("connectionStateChanged", "connected");
    await vi.waitFor(() => expect(dom.window.document.querySelector("#connection-status")!.getAttribute("data-state")).toBe("typed-only"));
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(requests).toHaveLength(1);
    expect(dom.window.document.querySelector("#markers")!.textContent).not.toContain("typed_input_admitted");
    expect(dom.window.document.querySelector("#transcript")!.textContent).not.toContain("cancel task:");
    expect(toggle.textContent).toBe("Connect");
  });

  it("treats an HTTP error after submission as uncertain without claiming task cancellation", async () => {
    const { task, cancel, card } = await mount(async () => new Response(null, { status: 503 }));
    await task("task_fixture", "active");
    cancel().click();
    await vi.waitFor(() => expect(card().textContent).toContain("Cancellation was not confirmed."));
    expect(card().dataset.status).toBe("active");
    expect(dom.window.document.querySelector("#transcript")!.textContent).not.toContain("cancel task:");
    expect(cancel().disabled).toBe(true);
    expect(dom.window.document.querySelector("#transcript")!.textContent).toContain("The previous command may have been admitted.");
  });

  it("bounds a stalled input request and requires reconnect before another cancellation", async () => {
    const { task, cancel, card } = await mount((_body, signal) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new Error("synthetic timeout")), { once: true });
    }));
    await task("task_fixture", "active");
    let timeout: (() => void) | null = null;
    const schedule = vi.mocked(dom.window.setTimeout).getMockImplementation()!;
    vi.mocked(dom.window.setTimeout).mockImplementation((handler, milliseconds, ...arguments_) => {
      if (milliseconds === 5000 && typeof handler === "function") timeout = () => handler(...arguments_);
      return schedule(handler, milliseconds, ...arguments_);
    });
    cancel().click();
    await vi.waitFor(() => expect(timeout).not.toBeNull());
    timeout!();
    await vi.waitFor(() => expect(card().textContent).toContain("Cancellation was not confirmed."));
    expect(cancel().disabled).toBe(true);
    expect(card().dataset.status).toBe("active");
  });

  it("refuses queued and late input after captured credentials rotate", async () => {
    let release!: (response: Response) => void;
    const { requests, task, cancel, submit, refresh, toggle } = await mount(() => new Promise((resolve) => { release = resolve; }));
    await task("task_fixture", "active");
    cancel().click();
    submit("Synthetic input before credential rotation");
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    refresh();
    await vi.waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([path]) => path === "/api/v1/refresh")).toBe(true));
    await new Promise((resolve) => setTimeout(resolve, 0));
    release(Response.json({ version: 1 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(requests).toHaveLength(1);
    expect(dom.window.document.querySelector("#transcript")!.textContent).not.toContain("cancel task:");
    expect(toggle.textContent).toBe("Connect");
  });

  it("does not echo unrecognized command refusal text", async () => {
    const { task } = await mount();
    await task(null, "rejected", "synthetic diagnostic content");
    expect(dom.window.document.querySelector("#transcript")!.textContent).toContain("Task command was refused.");
    expect(dom.window.document.querySelector("#transcript")!.textContent).not.toContain("synthetic diagnostic content");
  });

  it("keeps a retained old-identity card disabled until current authoritative task state", async () => {
    const { task, cancel, toggle, requests } = await mount();
    await task("task_fixture", "active");
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    await vi.waitFor(() => expect(dom.window.document.querySelector("#connection-status")!.getAttribute("data-state")).toBe("typed-only"));
    expect(cancel().disabled).toBe(true);
    cancel().click();
    expect(requests).toEqual([]);
    await task("task_fixture", "active");
    expect(cancel().disabled).toBe(false);
  });

  it.each([
    "No active task to cancel.",
    "Several tasks are active. Use Cancel on the task you want to stop.",
    "Provide an objective after Start task.",
  ])("displays deterministic command guidance without inventing a task (%s)", async (reason) => {
    const { task } = await mount();
    await task(null, "rejected", reason);
    expect(dom.window.document.querySelector("#transcript")!.textContent).toContain(reason);
    expect(dom.window.document.querySelectorAll('[data-operation="task"]')).toHaveLength(0);
  });

  it("does not mark still-running work terminal when cancellation is refused", async () => {
    const { task, card, cancel } = await mount();
    await task("task_fixture", "active");
    await task("task_fixture", "rejected", "task is not active");
    expect(card().dataset.status).toBe("active");
    expect(card().textContent).toContain("Cancellation was refused. The task status has not changed.");
    expect(cancel().disabled).toBe(false);
  });

  it.each(["cancelling", "interrupted", "completed"])("preserves %s state after a replayed cancellation refusal", async (status) => {
    const { task, card, cancel } = await mount();
    await task("task_fixture", status);
    await task("task_fixture", "rejected", "task is not active");
    expect(card().dataset.status).toBe(status);
    expect(cancel().disabled).toBe(true);
  });
});
