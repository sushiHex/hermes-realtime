import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";
import { connectionFailureCategory, ConnectionRequestRejected, renderConnectionRecovery } from "../src/connection-recovery";

describe("connection recovery guidance", () => {
  const messages = {
    bootstrap: "Session startup failed",
    rebind: "Session reconnect failed",
    media: "Audio connection failed",
    microphone: "Microphone preparation failed",
    activation: "Session activation failed",
    configuration: "Session configuration failed",
    stop: "Session disconnect was not confirmed",
  } as const;

  it.each(Object.entries(messages))("renders fixed %s guidance beside populated history", (stage, title) => {
    const dom = new JSDOM('<output id="recovery"></output><ol id="transcript"><li>Retained synthetic history</li></ol>');
    const output = dom.window.document.querySelector<HTMLOutputElement>("output")!;
    const history = dom.window.document.querySelector("ol")!;
    const before = history.innerHTML;
    renderConnectionRecovery(output, { stage: stage as keyof typeof messages, category: "request-failed" }, true, stage === "stop");
    expect(output.hidden).toBe(false);
    expect(output.dataset.stage).toBe(stage);
    expect(output.dataset.category).toBe("request-failed");
    expect(output.textContent).toBe(stage === "stop"
      ? `${title}. Local audio is disconnected. Select Disconnect to retry the server stop.`
      : `${title}. Select Connect to retry. If it repeats, ask the host operator to check recovery.`);
    expect(history.innerHTML).toBe(before);
  });

  it.each([
    [400, "request-refused"], [403, "authorization-refused"], [408, "request-timed-out"],
    [409, "request-refused"], [503, "service-unavailable"], [500, "request-refused"],
  ] as const)("classifies HTTP %s without response content", (status, category) => {
    expect(connectionFailureCategory(new ConnectionRequestRejected(status))).toBe(category);
  });

  it("does not display exception text or infer a cause from it", () => {
    expect(connectionFailureCategory(new Error("private synthetic payload"))).toBe("request-failed");
  });

  it("explains service refusal without promising reconnect repairs the host", () => {
    const output = new JSDOM("<output></output>").window.document.querySelector("output")!;
    renderConnectionRecovery(output, { stage: "bootstrap", category: "service-unavailable" }, true, false);
    expect(output.textContent).toBe("Session startup failed. Host refused the request (503). Select Connect to retry. If it persists, ask the host operator to check recovery; a restart may be needed.");
  });

  it("requires a fresh launch when a one-shot launch cannot retry", () => {
    const output = new JSDOM("<output></output>").window.document.querySelector("output")!;
    renderConnectionRecovery(output, { stage: "bootstrap", category: "request-failed" }, false, false);
    expect(output.textContent).toBe("Session startup failed. Open a fresh launch from the host to connect.");
  });

  it("keeps unresolved stop recovery ahead of fresh connection authority", () => {
    const output = new JSDOM("<output></output>").window.document.querySelector("output")!;
    renderConnectionRecovery(output, { stage: "stop", category: "request-failed" }, false, true);
    expect(output.textContent).toBe("Session disconnect was not confirmed. Local audio is disconnected. Select Disconnect to retry the server stop.");
  });

  it("clears stale failure guidance when a new attempt starts", () => {
    const output = new JSDOM("<output></output>").window.document.querySelector("output")!;
    renderConnectionRecovery(output, { stage: "media", category: "request-failed" }, true, false);
    renderConnectionRecovery(output, null, true, false);
    expect(output.hidden).toBe(true);
    expect(output.textContent).toBe("");
    expect(output.dataset.category).toBeUndefined();
    expect(output.dataset.stage).toBeUndefined();
  });
});
