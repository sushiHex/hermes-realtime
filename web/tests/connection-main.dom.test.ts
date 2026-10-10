import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const media = vi.hoisted(() => ({
  connect: vi.fn<() => Promise<void>>(),
  disconnect: vi.fn<() => Promise<void>>(),
  microphone: vi.fn<() => Promise<undefined>>(),
  rooms: [] as Array<{ emit: (event: string, participant: { identity: string } | string) => void }>,
}));
vi.mock("livekit-client", async (original) => ({
  ...await original<typeof import("livekit-client")>(),
  Room: class {
    localParticipant = {
      trackPublications: new Map(),
      setMicrophoneEnabled: media.microphone,
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

async function mount(request: (path: string) => Response | Promise<Response>, remembered = false, oneShot = false) {
  vi.resetModules();
  media.connect.mockReset().mockResolvedValue();
  media.disconnect.mockReset().mockResolvedValue();
  media.microphone.mockReset().mockResolvedValue(undefined);
  media.rooms.length = 0;
  dom = new JSDOM(readFileSync(new URL("../index.html", import.meta.url), "utf8"), {
    url: oneShot ? `http://localhost/#bootstrap=${"a".repeat(43)}` : "http://localhost/", pretendToBeVisual: true,
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

it.each([false, true])("applies stored output style before media admission (remembered: %s)", async (remembered) => {
  let acknowledge!: () => void;
  const { toggle, fetch } = await mount(path => {
    if (path === "/api/v1/output-style") return new Promise<Response>(resolve => {
      acknowledge = () => resolve(Response.json({version:1, selectedStyle:"learning"}));
    });
    return normalRequest(path);
  }, remembered);
  const style = dom.window.document.querySelector<HTMLSelectElement>("#output-style")!;
  style.value = "learning";
  style.dispatchEvent(new dom.window.Event("change"));
  toggle.click();
  await vi.waitFor(() => expect(acknowledge).toBeDefined());
  expect(media.connect).not.toHaveBeenCalled();
  acknowledge();
  await vi.waitFor(() => expect(media.connect).toHaveBeenCalledOnce());
  const selection = fetch.mock.calls.find(([path]) => path === "/api/v1/output-style")!;
  expect(JSON.parse(selection[1]?.body as string)).toEqual({style:"learning"});
  const token = remembered ? "synthetic.rebound.token" : credential.token;
  expect(selection[1]?.headers).toMatchObject({Authorization:`Bearer ${token}`});
});

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
    const { toggle, recovery } = await mount(normalRequest);
    let admit!: () => void;
    media.connect.mockReturnValueOnce(new Promise<void>((resolve) => { admit = resolve; }));
    toggle.click();
    await vi.waitFor(() => expect(media.connect).toHaveBeenCalledTimes(1));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(media.disconnect).not.toHaveBeenCalled();
    expect(recovery.hidden).toBe(true);
    expect(toggle.disabled).toBe(true);
    admit();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Disconnect"));
  });

  it.each(["microphone", "media"] as const)("handles admitted worker departure during pending %s activation", async (phase) => {
    let activateMedia!: () => void;
    let releaseMicrophone!: () => void;
    const { toggle, recovery, fetch } = await mount((path) => phase === "media" && path === "/api/v1/media"
      ? new Promise<Response>((resolve) => { activateMedia = () => resolve(Response.json({ version: 1 })); })
      : normalRequest(path));
    if (phase === "microphone") {
      media.microphone.mockReturnValueOnce(new Promise<undefined>((resolve) => { releaseMicrophone = () => resolve(undefined); }));
    }
    toggle.click();
    if (phase === "media") await vi.waitFor(() => expect(activateMedia).toBeDefined());
    else await vi.waitFor(() => expect(media.microphone).toHaveBeenCalledTimes(1));
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("preparing");
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("worker"));
    expect(toggle.textContent).toBe("Connect");
    expect(media.disconnect).toHaveBeenCalledTimes(1);
    if (phase === "media") activateMedia();
    else releaseMicrophone();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("disconnected");
    expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(true);
    expect(recovery.dataset.stage).toBe("worker");
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/models" || path === "/api/v1/voices")).toHaveLength(0);
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stable-rebind")).toHaveLength(0);
  });

  it("allows Disconnect while microphone permission is pending and rejects late activation", async () => {
    const { toggle, recovery, fetch } = await mount(normalRequest);
    let release!: () => void;
    media.microphone.mockReturnValueOnce(new Promise<undefined>((resolve) => { release = () => resolve(undefined); }));
    toggle.click();
    await vi.waitFor(() => expect(media.microphone).toHaveBeenCalledTimes(1));
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("preparing");
    expect(toggle.textContent).toBe("Disconnect");
    expect(toggle.disabled).toBe(false);
    toggle.click();
    await vi.waitFor(() => expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stop")).toHaveLength(1));
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("stopped"));
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/media")).toHaveLength(0);
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("stopped");
    expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(true);
    expect(recovery.hidden).toBe(true);
  });

  it("allows Disconnect during native reconnect and ignores its late return", async () => {
    const { toggle, recovery, fetch } = await mount(normalRequest);
    toggle.click();
    await vi.waitFor(() => expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/models")).toHaveLength(1));
    const activeRoom = media.rooms[0]!;
    activeRoom.emit("connectionStateChanged", "reconnecting");
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("reconnecting");
    expect(toggle.textContent).toBe("Disconnect");
    expect(toggle.disabled).toBe(false);
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("stopped"));
    const requestsBefore = fetch.mock.calls.length;
    activeRoom.emit("connectionStateChanged", "connected");
    activeRoom.emit("participantDisconnected", { identity: credential.workerIdentity });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fetch.mock.calls.length).toBe(requestsBefore);
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("stopped");
    expect(toggle.textContent).toBe("Connect");
    expect(recovery.hidden).toBe(true);
  });

  it("rejects a late rebind answer after reconnect is stopped", async () => {
    let answer!: () => void;
    const replacement = { ...credential, participantIdentity: "browser_fedcba9876543210", token: "synthetic.late.token" };
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path === "/api/v1/stable-rebind") return new Promise<Response>((resolve) => { answer = () => resolve(Response.json(replacement)); });
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/models")).toHaveLength(1));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("worker"));
    toggle.click();
    await vi.waitFor(() => expect(answer).toBeDefined());
    expect(toggle.textContent).toBe("Disconnect");
    expect(toggle.disabled).toBe(false);
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("stopped"));
    const requestsBefore = fetch.mock.calls.length;
    answer();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(media.rooms).toHaveLength(1);
    expect(fetch.mock.calls.length).toBe(requestsBefore);
    expect(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")).toBeNull();
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("stopped");
    expect(recovery.hidden).toBe(true);
  });

  it("preserves a newer connection when an older stopped rebind rejects", async () => {
    let rejectOld!: () => void;
    const { toggle, recovery, fetch } = await mount((path) => path === "/api/v1/stable-rebind"
      ? new Promise<Response>((_, reject) => { rejectOld = () => reject(new Error("synthetic late refusal")); })
      : normalRequest(path));
    toggle.click();
    await vi.waitFor(() => expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/models")).toHaveLength(1));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery?.dataset.stage).toBe("worker"));
    toggle.click();
    await vi.waitFor(() => expect(rejectOld).toBeDefined());
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("stopped"));
    toggle.click();
    await vi.waitFor(() => expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/models")).toHaveLength(2));
    const disconnectsBefore = media.disconnect.mock.calls.length;
    rejectOld();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toggle.textContent).toBe("Disconnect");
    expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false);
    expect(media.disconnect.mock.calls.length).toBe(disconnectsBefore);
    expect(recovery.hidden).toBe(true);
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

  it("retires a definitive 403 rebind refusal and requires explicit fresh bootstrap", async () => {
    const { toggle, recovery, fetch } = await mount((path) => path === "/api/v1/stable-rebind"
      ? new Response(null, { status: 403 }) : normalRequest(path));
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("worker"));
    toggle.click();
    await vi.waitFor(() => expect(recovery.dataset.category).toBe("authorization-refused"));
    expect(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")).toBeNull();
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stable-rebind")).toHaveLength(1);
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stable-bootstrap")).toHaveLength(1);
    expect(toggle.disabled).toBe(false);
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stable-bootstrap")).toHaveLength(2);
  });

  it("retires a definitive one-shot 409 rebind refusal and requires fresh launch", async () => {
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path === "/api/v1/bootstrap") return Response.json(credential);
      if (path === "/api/v1/rebind") return new Response(null, { status: 409 });
      return normalRequest(path);
    }, false, true);
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("worker"));
    toggle.click();
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("rebind"));
    expect(toggle.disabled).toBe(true);
    expect(toggle.textContent).toBe("Fresh launch required");
    expect(recovery.textContent).toContain("fresh launch from the host");
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/rebind")).toHaveLength(1);
  });

  it("retires old authority before a 409 fallback bootstrap fails", async () => {
    let bootstraps = 0;
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path === "/api/v1/stable-rebind") return new Response(null, { status: 409 });
      if (path === "/api/v1/stable-bootstrap" && ++bootstraps === 2) return new Response(null, { status: 503 });
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("worker"));
    toggle.click();
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("bootstrap"));
    expect(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")).toBeNull();
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stable-rebind")).toHaveLength(1);
    expect(bootstraps).toBe(3);
  });

  it("classifies a pending 409 fallback as fresh bootstrap until admission", async () => {
    let admit!: () => void;
    let bootstraps = 0;
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path === "/api/v1/stable-rebind") return new Response(null, { status: 409 });
      if (path === "/api/v1/stable-bootstrap" && ++bootstraps > 1) return new Promise<Response>((resolve) => { admit = () => resolve(Response.json(credential)); });
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("worker"));
    toggle.click();
    await vi.waitFor(() => expect(admit).toBeDefined());
    expect(toggle.textContent).toBe("Connecting…");
    expect(toggle.disabled).toBe(true);
    expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("bootstrapping");
    expect(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")).toBeNull();
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stop")).toHaveLength(0);
    let releaseMicrophone!: () => void;
    media.microphone.mockReturnValueOnce(new Promise<undefined>((resolve) => { releaseMicrophone = () => resolve(undefined); }));
    admit();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("preparing"));
    expect(toggle.disabled).toBe(false);
    expect(toggle.textContent).toBe("Disconnect");
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLElement>("#connection-status")!.dataset.state).toBe("stopped"));
    releaseMicrophone();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/stop")).toHaveLength(1);
    expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(true);
  });

  it("preserves transient rebind replay authority until a successful answer", async () => {
    let rebinds = 0;
    const { toggle, recovery, fetch } = await mount((path) => path === "/api/v1/stable-rebind" && ++rebinds <= 2
      ? new Response(null, { status: 503 }) : normalRequest(path));
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("worker"));
    toggle.click();
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("rebind"));
    expect(JSON.parse(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")!).identity).toBe(credential.participantIdentity);
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    const requests = fetch.mock.calls.filter(([path]) => path === "/api/v1/stable-rebind");
    expect(requests).toHaveLength(3);
    expect(new Set(requests.map(([, init]) => JSON.parse(init!.body as string).requestId)).size).toBe(1);
  });

  it("does not retire a replacement credential for a late old 403 refusal", async () => {
    let refuseOld!: () => void;
    let rebinds = 0;
    let bootstraps = 0;
    const newer = { ...credential, participantIdentity: "browser_1111111111111111", token: "synthetic.newer.token" };
    const { toggle, recovery } = await mount((path) => {
      if (path === "/api/v1/stable-rebind" && ++rebinds === 1) return new Promise<Response>((resolve) => { refuseOld = () => resolve(new Response(null, { status: 403 })); });
      if (path === "/api/v1/stable-bootstrap" && ++bootstraps > 1) return Response.json(newer);
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("worker"));
    toggle.click();
    await vi.waitFor(() => expect(refuseOld).toBeDefined());
    toggle.click();
    await vi.waitFor(() => expect(toggle.textContent).toBe("Connect"));
    toggle.click();
    await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
    refuseOld();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(JSON.parse(dom.window.sessionStorage.getItem("hermes-realtime.stable-session.v1")!).identity).toBe(newer.participantIdentity);
    expect(toggle.textContent).toBe("Disconnect");
    expect(recovery.hidden).toBe(true);
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

  it("retires a pending terminal poll failure when the worker leaves", async () => {
    let failPoll!: () => void;
    let polls = 0;
    const { toggle, recovery, fetch } = await mount((path) => {
      if (path === "/api/v1/events") {
        if (++polls < 5) return Response.json({});
        return new Promise<Response>((_, reject) => { failPoll = () => reject(new Error("synthetic late poll failure")); });
      }
      if (path === "/api/v1/projection-resync") return Response.json({ ...credential, participantIdentity: "browser_fedcba9876543210", token: "synthetic.rotated.token" });
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(failPoll).toBeDefined(), { timeout: 4000 });
    const oldSignal = fetch.mock.calls.filter(([path]) => path === "/api/v1/events").at(-1)![1]!.signal!;
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("worker"));
    failPoll();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(oldSignal.aborted).toBe(true);
    expect(fetch.mock.calls.filter(([path]) => path === "/api/v1/projection-resync")).toHaveLength(0);
    expect(media.connect).toHaveBeenCalledTimes(1);
    expect(toggle.textContent).toBe("Connect");
    expect(recovery.dataset.stage).toBe("worker");
  });

  it.each([false, true])("ignores retired event JSON after worker departure with replacement %s", async (replace) => {
    let answer!: () => void;
    let polls = 0;
    const { toggle, recovery } = await mount((path) => {
      if (path === "/api/v1/events" && ++polls === 1) {
        const response = Response.json({});
        response.json = () => new Promise((resolve) => { answer = () => resolve({ version: 1, events: [{ sequence: 1, kind: "typed_input_admitted", monotonicMs: 12500, data: { inputSequence: 1 } }] }); });
        return response;
      }
      return normalRequest(path);
    });
    toggle.click();
    await vi.waitFor(() => expect(answer).toBeDefined());
    media.rooms[0]!.emit("participantDisconnected", { identity: credential.workerIdentity });
    await vi.waitFor(() => expect(recovery.dataset.stage).toBe("worker"));
    if (replace) {
      toggle.click();
      await vi.waitFor(() => expect(dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input")!.disabled).toBe(false));
      await vi.waitFor(() => expect(recovery.hidden).toBe(true));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const markers = dom.window.document.querySelector("#markers")!;
    const before = markers.textContent;
    answer();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(markers.textContent).toBe(before);
    expect(markers.textContent).not.toContain("typed_input_admitted");
    expect(toggle.textContent).toBe(replace ? "Disconnect" : "Connect");
    expect(recovery.hidden).toBe(replace);
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
