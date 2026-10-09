import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const media = vi.hoisted(() => ({
  connect: vi.fn<() => Promise<void>>(),
  disconnect: vi.fn<() => Promise<void>>(),
  rooms: [] as Array<{ emit: (event: string, participant: { identity: string } | string) => void }>,
}));
vi.mock("livekit-client", async (original) => ({
  ...await original<typeof import("livekit-client")>(),
  Room: class {
    localParticipant = {
      trackPublications: new Map(),
      setMicrophoneEnabled: async () => undefined,
    };
    handlers = new Map<string, (participant: { identity: string } | string) => void>();
    constructor() { media.rooms.push(this); }
    on(event: string, handler: (participant: { identity: string } | string) => void) {
      this.handlers.set(event, handler);
      return this;
    }
    emit(event: string, participant: { identity: string } | string) { this.handlers.get(event)?.(participant); }
    connect = media.connect;
    disconnect = media.disconnect;
  },
}));

const credential = {
  version: 1, url: "wss://livekit.test", roomName: "synthetic-room",
  participantIdentity: "browser_0123456789abcdef", workerIdentity: "worker_hermes_browser",
  expiresInSeconds: 60, token: "synthetic.token.value",
};
let dom: JSDOM;

async function mount(request: (path: string) => Response | Promise<Response>, remembered = false) {
  vi.resetModules();
  media.connect.mockReset().mockResolvedValue();
  media.disconnect.mockReset().mockResolvedValue();
  media.rooms.length = 0;
  dom = new JSDOM(readFileSync(new URL("../index.html", import.meta.url), "utf8"), {
    url: "http://localhost/", pretendToBeVisual: true,
  });
  for (const key of ["window", "document", "HTMLElement", "HTMLMediaElement", "Option"] as const) {
    vi.stubGlobal(key, key === "window" ? dom.window : dom.window[key]);
  }
  vi.stubGlobal("navigator", { mediaDevices: { enumerateDevices: async () => [] } });
  vi.spyOn(dom.window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  vi.stubGlobal("fetch", vi.fn((path: string) => request(path)));
  vi.spyOn(console, "info").mockImplementation(() => {});
  if (remembered) {
    dom.window.sessionStorage.setItem("hermes-realtime.stable-session.v1", JSON.stringify({
      identity: credential.participantIdentity, requestId: null,
    }));
  }
  await import("../src/main");
  const toggle = dom.window.document.querySelector<HTMLButtonElement>("#session-toggle")!;
  const recovery = dom.window.document.querySelector<HTMLOutputElement>("#connection-recovery")!;
  return { toggle, recovery, fetch: vi.mocked(fetch) };
}

function normalRequest(path: string): Response | Promise<Response> {
  if (path === "/api/v1/stable-bootstrap") return Response.json(credential);
  if (path === "/api/v1/stable-rebind") return Response.json({ ...credential, participantIdentity: "browser_fedcba9876543210", token: "synthetic.rebound.token" });
  if (path === "/api/v1/media" || path === "/api/v1/stop") return Response.json({ version: 1 });
  if (path === "/api/v1/voices") return Response.json({ version: 1, voices: [], selectedVoice: null });
  // Keep event polling pending; this fixture does not mint server conversation events.
  if (path.startsWith("/api/v1/events")) return new Promise(() => {});
  return new Response(null, { status: 503 });
}

afterEach(() => {
  dom?.window.close();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("mounted connection recovery", () => {
  it("shows safe bootstrap refusal and retry beside retained history", async () => {
    const { toggle, recovery } = await mount(() => new Response(null, { status: 503 }));
    const history = dom.window.document.querySelector("#transcript")!;
    history.innerHTML = "<li>Synthetic prior turn</li>";
    toggle.click();
    expect(toggle.disabled).toBe(true);
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("bootstrap"));
    expect(recovery.dataset.category).toBe("service-unavailable");
    expect(recovery.hidden).toBe(false);
    expect(toggle.textContent).toBe("Connect");
    expect(toggle.disabled).toBe(false);
    expect(history.textContent).toBe("Synthetic prior turn");
  });

  it("labels remembered-session rejection as rebind and keeps retry authority", async () => {
    const { toggle, recovery, fetch } = await mount(() => new Response(null, { status: 503 }), true);
    toggle.click();
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("rebind"));
    expect(recovery.dataset.category).toBe("service-unavailable");
    const firstRequest = JSON.parse(fetch.mock.calls[0]?.[1]?.body as string).requestId;
    toggle.click();
    await vi.waitFor(() => expect(toggle.disabled).toBe(false));
    expect(fetch.mock.calls.length).toBe(2);
    const secondRequest = JSON.parse(fetch.mock.calls[1]?.[1]?.body as string).requestId;
    expect(secondRequest).toBe(firstRequest);
    expect(fetch.mock.calls.map(([path]) => path)).toEqual([
      "/api/v1/stable-rebind", "/api/v1/stable-rebind",
    ]);
  });

  it("labels media failure and keeps the same session available for recovery", async () => {
    const { toggle, recovery } = await mount(normalRequest);
    media.connect.mockRejectedValueOnce(new Error("synthetic private diagnostic"));
    toggle.click();
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("media"));
    expect(recovery.dataset.category).toBe("request-failed");
    expect(recovery.textContent).not.toContain("synthetic private diagnostic");
    expect(toggle.textContent).toBe("Connect");
    expect(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")).not.toBeNull();
  });

  it("stops local audio and offers an explicit retry when the exact worker leaves", async () => {
    const { toggle, recovery, fetch } = await mount(normalRequest);
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("worker"));
    expect(toggle.textContent).toBe("Connect");
    expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(true);
    expect(media.disconnect).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stable-rebind")).toHaveLength(0);
    expect(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")).not.toBeNull();
  });

  it("ignores worker departure before connection admission completes", async () => {
    let activate!: () => void;
    const { toggle, recovery } = await mount((path) => path === "/api/v1/media"
      ? new Promise<Response>((resolve) => { activate = () => resolve(Response.json({ version: 1 })); })
      : normalRequest(path));
    toggle.click();
    await vi.waitFor(() => expect(activate).toBeDefined());
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(media.disconnect).not.toHaveBeenCalled();
    expect(recovery.hidden).toBe(true);
    expect(toggle.disabled).toBe(true);
    activate();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
  });

  it("ignores departure of another participant", async () => {
    const { toggle, recovery } = await mount(normalRequest);
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    media.rooms[0]!.emit("participantDisconnected", { identity: "worker_another_synthetic" });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toggle.textContent).toBe("Disconnect");
    expect(recovery.hidden).toBe(true);
    expect(media.disconnect).not.toHaveBeenCalled();
  });

  it("ignores a stale room departure after a replacement session connects", async () => {
    const { toggle, recovery } = await mount(normalRequest);
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    const oldRoom = media.rooms[0]!;
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    media.disconnect.mockClear();
    oldRoom.emit("participantDisconnected", { identity: credential.workerIdentity });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toggle.textContent).toBe("Disconnect");
    expect(recovery.hidden).toBe(true);
    expect(media.disconnect).not.toHaveBeenCalled();
  });

  it("ignores worker departure during intentional disconnect", async () => {
    const { toggle, recovery } = await mount(normalRequest);
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    const currentRoom = media.rooms[0]!;
    toggle.click();
    currentRoom.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    expect(recovery.hidden).toBe(true);
  });

  it("does not disconnect a newer successful connection when old cleanup completes", async () => {
    const { toggle, recovery } = await mount(normalRequest);
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    let release!: () => void;
    media.disconnect.mockReturnValueOnce(new Promise<void>((resolve) => { release = resolve; }));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(media.disconnect).toHaveBeenCalledTimes(1));
    expect(toggle.textContent).toBe("Connect");
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toggle.textContent).toBe("Disconnect");
    expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false);
    expect(recovery.hidden).toBe(true);
  });

  it("does not overwrite a newer failed reconnect with late worker cleanup", async () => {
    const { toggle, recovery } = await mount((path) => path === "/api/v1/stable-rebind"
      ? new Response(null, { status: 503 }) : normalRequest(path));
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    let release!: () => void;
    media.disconnect.mockReturnValueOnce(new Promise<void>((resolve) => { release = resolve; }));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(media.disconnect).toHaveBeenCalledTimes(1));
    toggle.click();
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("rebind"));
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(recovery.dataset.stage).toBe("rebind");
    expect(recovery.dataset.category).toBe("service-unavailable");
  });

  it("preserves a successful retry after older failed-connect cleanup settles", async () => {
    const { toggle, recovery } = await mount(normalRequest);
    media.connect.mockRejectedValueOnce(new Error("synthetic media failure"));
    let release!: () => void;
    media.disconnect.mockReturnValueOnce(new Promise<void>((resolve) => { release = resolve; }));
    toggle.click();
    await vi.waitFor(() => expect(media.disconnect).toHaveBeenCalledTimes(1));
    expect(toggle.textContent).toBe("Connect");
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toggle.textContent).toBe("Disconnect");
    expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false);
    expect(recovery.hidden).toBe(true);
  });

  it("does not auto-retry over a newer refused explicit reconnect", async () => {
    const { toggle, recovery, fetch } = await mount((path) => path === "/api/v1/stable-rebind"
      ? new Response(null, { status: 503 }) : normalRequest(path));
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    let release!: () => void;
    media.disconnect.mockReturnValueOnce(new Promise<void>((resolve) => { release = resolve; }));
    media.rooms[0]!.emit("connectionStateChanged", "disconnected");
    await vi.waitFor(() => expect(media.disconnect).toHaveBeenCalledTimes(1));
    toggle.click();
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("rebind"));
    const requestsBefore = fetch.mock.calls.length;
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fetch.mock.calls.length).toBe(requestsBefore);
    expect(recovery.dataset.stage).toBe("rebind");
  });

  it.each(["projection-resync", "poll-terminal"] as const)(
    "preserves an explicit retry failure while older %s cleanup settles", async (pathway) => {
      const { toggle, recovery, fetch } = await mount((path) => {
        if (path.startsWith("/api/v1/events")) return Response.json({});
        if (path === "/api/v1/projection-resync") return pathway === "projection-resync"
          ? Response.json({ ...credential, participantIdentity: "browser_fedcba9876543210", token: "synthetic.rotated.token" })
          : new Response(null, { status: 503 });
        if (path === "/api/v1/stable-rebind") return new Response(null, { status: 503 });
        return normalRequest(path);
      });
      let release!: () => void;
      media.disconnect.mockReturnValueOnce(new Promise<void>((resolve) => { release = resolve; }));
      toggle.click();
      await vi.waitFor(() => expect(media.disconnect).toHaveBeenCalledTimes(1), { timeout: 4000 });
      expect(toggle.textContent).toBe("Connect");
      toggle.click();
      await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("rebind"));
      const requestsBefore = fetch.mock.calls.length;
      const stateBefore = dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state;
      release();
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(fetch.mock.calls.length).toBe(requestsBefore);
      expect(recovery.dataset.stage).toBe("rebind");
      expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe(stateBefore);
    },
  );

  it("ignores a resync answer superseded by worker departure before rotation", async () => {
    let answer!: () => void;
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path.startsWith("/api/v1/events")) return Response.json({});
      if (path === "/api/v1/projection-resync") return new Promise<Response>((resolve) => {
        answer = () => resolve(Response.json({ ...credential, participantIdentity: "browser_fedcba9876543210", token: "synthetic.rotated.token" }));
      });
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(answer).toBeDefined(), { timeout: 4000 });
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("worker"));
    const requestsBefore = fetch.mock.calls.length;
    answer();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fetch.mock.calls.length).toBe(requestsBefore);
    expect(recovery.dataset.stage).toBe("worker");
    expect(JSON.parse(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")!).identity).toBe(credential.participantIdentity);
  });

  it("ignores a resync rejection superseded by worker departure", async () => {
    let rejectResync!: () => void;
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path.startsWith("/api/v1/events")) return Response.json({});
      if (path === "/api/v1/projection-resync") return new Promise<Response>((_, reject) => {
        rejectResync = () => reject(new Error("synthetic resync failure"));
      });
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(rejectResync).toBeDefined(), { timeout: 4000 });
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("worker"));
    const stateBefore = dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state;
    const requestsBefore = fetch.mock.calls.length;
    const disconnectsBefore = media.disconnect.mock.calls.length;
    rejectResync();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe(stateBefore);
    expect(recovery.dataset.stage).toBe("worker");
    expect(toggle.textContent).toBe("Connect");
    expect(fetch.mock.calls.length).toBe(requestsBefore);
    expect(media.disconnect.mock.calls.length).toBe(disconnectsBefore);
  });

  it("shows safe projection refusal after terminal polling recovery fails", async () => {
    const { toggle, recovery } = await mount((path) => {
      if (path.startsWith("/api/v1/events")) return Response.json({});
      if (path === "/api/v1/projection-resync") return new Response(null, { status: 503 });
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("projection"), { timeout: 4000 });
    expect(recovery.dataset.category).toBe("service-unavailable");
    expect(recovery.hidden).toBe(false);
    expect(recovery.textContent).toBe("Conversation updates unavailable. Host refused the request (503). Select Connect to retry. If it persists, ask the host operator to check recovery; a restart may be needed.");
    expect(toggle.textContent).toBe("Connect");
  });

  it.each(["voices", "models"] as const)("preserves the new %s catalog when an older response settles", async (catalog) => {
    let answerOld!: () => void;
    let requests = 0;
    const payload = (old: boolean) => catalog === "voices"
      ? { version: 1, voices: [old ? "af_old" : "af_current"], selectedVoice: old ? "af_old" : "af_current" }
      : {
        version: 1, selectedModel: old ? "synthetic-old" : "synthetic-current", selectedEffort: "high",
        models: [{ model: old ? "synthetic-old" : "synthetic-current", displayName: "Synthetic model",
          description: "Synthetic catalog", defaultEffort: "high", supportedEfforts: ["high"] }],
      };
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path === `/api/v1/${catalog}`) {
        if (++requests === 1) return new Promise<Response>((resolve) => {
          answerOld = () => resolve(Response.json(payload(true)));
        });
        return Response.json(payload(false));
      }
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(answerOld).toBeDefined());
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("worker"));
    toggle.click();
    const select = dom.window.document.querySelector<HTMLSelectElement>(catalog === "voices" ? "#voice" : "#model-select")!;
    await vi.waitFor(() => expect(select.value).toBe(catalog === "voices" ? "af_current" : "synthetic-current"));
    const requestsBefore = fetch.mock.calls.length;
    answerOld();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(select.value).toBe(catalog === "voices" ? "af_current" : "synthetic-current");
    expect(fetch.mock.calls.length).toBe(requestsBefore);
    expect(toggle.textContent).toBe("Disconnect");
    expect(recovery.hidden).toBe(true);
  });

  it("retries an unconfirmed stop before allowing Connect and preserves history", async () => {
    let stopRequests = 0;
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path === "/api/v1/stop" && ++stopRequests === 1) return new Response(null, { status: 503 });
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
    const history = dom.window.document.querySelector("#transcript")!;
    history.innerHTML = "<li>Synthetic completed turn</li>";
    toggle.click();
    expect(toggle.textContent).toBe("Disconnecting…");
    expect(toggle.disabled).toBe(true);
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("stop"));
    expect(recovery.dataset.category).toBe("service-unavailable");
    expect(toggle.textContent).toBe("Disconnect");
    expect(toggle.disabled).toBe(false);
    expect(recovery.textContent).toContain("retry the server stop");
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    expect(recovery.hidden).toBe(true);
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stable-bootstrap")).toHaveLength(1);
    expect(stopRequests).toBe(2);
    expect(history.textContent).toBe("Synthetic completed turn");
    expect(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")).toBeNull();
  });
});
