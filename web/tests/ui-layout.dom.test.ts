import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

const markup = readFileSync(new URL("../index.html", import.meta.url), "utf8");

describe("compact session layout", () => {
  it("keeps one conversation heading and one style control beside session controls", () => {
    const { document } = new JSDOM(markup).window;
    expect([...document.querySelectorAll("h1,h2")].filter(
      heading => heading.textContent === "Conversation",
    )).toHaveLength(1);
    expect(document.querySelectorAll("#output-style")).toHaveLength(1);
    expect(document.querySelector(".toolbar #output-style")).not.toBeNull();
    expect(document.querySelector("#typed-input")?.getAttribute("placeholder")).toBe("Type a message…");
    expect(document.querySelector(".composer-hint")).toBeNull();
    expect(document.querySelector<HTMLElement>(".model-controls")?.hidden).toBe(true);
    expect(document.querySelector(".model-controls #model-selection-state")).toBeNull();
  });

  it.each([
    ["evidence-disclosure", "evidence-capture-status", "evidence-consent", ".evidence-disclosure-copy"],
    ["search-egress-disclosure", "search-egress-status", "search-egress-consent", ".evidence-disclosure-copy"],
    ["voice-delete-panel", "voice-delete-status", "delete-voice-conversation", "#voice-delete-limit"],
  ])("keeps %s status discoverable and requires expanding the explanation before its actions", (
    panelId, statusId, actionId, copySelector,
  ) => {
    const { document, Node } = new JSDOM(markup).window;
    const panel = document.getElementById(panelId!) as HTMLDetailsElement;
    expect(panel?.tagName).toBe("DETAILS");
    expect(panel.open).toBe(false);
    const summary = panel.querySelector("summary")!;
    expect(summary.contains(document.getElementById(statusId!))).toBe(true);
    const action = document.getElementById(actionId!)!;
    expect(panel.contains(action)).toBe(true);
    expect(summary.contains(action)).toBe(false);
    const explanation = panel.querySelector(copySelector!)!;
    expect(explanation.compareDocumentPosition(action) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);
    expect(action.getAttribute("aria-describedby")).toContain(statusId);
  });

  it("preserves the complete canonical evidence disclosure in the expandable body", () => {
    const { document } = new JSDOM(markup).window;
    const text = document.querySelector("#evidence-disclosure .evidence-disclosure-copy")!.textContent!;
    const canonical = readFileSync(new URL(
      "../../src/hermes_realtime/evidence/disclosure_v1.txt", import.meta.url,
    ), "utf8");
    const normalized = text.replace(/\s+/g, " ").trim();
    for (const paragraph of canonical.trim().split(/\n\n/)) {
      expect(normalized).toContain(paragraph.replace(/\s+/g, " ").trim());
    }
  });
});
