import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const stylesheet = readFileSync(
  fileURLToPath(new URL("../src/styles.css", import.meta.url)),
  "utf8",
);
const markup = readFileSync(
  fileURLToPath(new URL("../index.html", import.meta.url)),
  "utf8",
);

function declarations(selector: string): string {
  const start = stylesheet.indexOf(`${selector} {`);
  if (start < 0) return "";
  const bodyStart = stylesheet.indexOf("{", start) + 1;
  return stylesheet.slice(bodyStart, stylesheet.indexOf("}", bodyStart));
}

describe("connection status contrast", () => {
  it.each([
    ["typed-only", "var(--warning)"],
    ["error", "var(--danger)"],
  ])("keeps %s text readable while coloring only its dot", (state, color) => {
    const status = declarations(`.connection-status[data-state="${state}"] .status`);
    const dot = declarations(`.connection-status[data-state="${state}"] .status-dot`);

    expect(status).toContain(`color: ${color}`);
    expect(status).not.toContain("background");
    expect(dot).toContain(`background-color: ${color}`);
  });
});

describe("microphone activity", () => {
  it("makes the compact status pill the only accessible mute control", () => {
    for (const id of ["microphone-activity", "microphone-activity-label", "microphone-level"]) {
      expect(markup).toContain(`id="${id}"`);
    }
    expect(markup).toMatch(
      /<button\s+id="microphone-activity"[\s\S]*?type="button"[\s\S]*?aria-pressed="false"/,
    );
    expect(markup).not.toContain('id="mute"');
    expect(markup).toContain('id="microphone-activity-label" aria-live="polite"');
    expect(
      declarations('.microphone-activity[data-active="true"] .microphone-activity-dot'),
    ).toContain("background: var(--success)");
  });
});

describe("session controls", () => {
  it("exposes one authoritative connect/stop toggle", () => {
    expect(markup).toContain('id="session-toggle"');
    expect(markup).not.toContain('id="connect"');
    expect(markup).not.toContain('id="disconnect"');
    expect(markup).not.toContain('id="stop"');
  });
});

describe("karaoke unread projection", () => {
  it("uses the muted shade for pending segments and punctuation gaps", () => {
    expect(declarations(".assistant-segment.karaoke-pending")).toContain("color: var(--muted)");
    expect(declarations(".karaoke-gap.karaoke-pending")).toContain("color: var(--muted)");
  });
});

describe("desktop viewport containment", () => {
  it("fits shell chrome and workspace inside the dynamic viewport", () => {
    expect(declarations("html")).toContain("height: 100%");
    expect(declarations("html")).toContain("overflow: hidden");
    expect(declarations("body")).toContain("height: 100%");
    expect(declarations("body")).toContain("overflow: hidden");

    const shell = declarations(".app-shell");
    expect(shell).toContain("display: grid");
    expect(shell).toContain("height: 100dvh");
    expect(shell).toContain("grid-template-rows: auto auto minmax(0, 1fr)");

    const workspace = declarations(".workspace");
    expect(workspace).toContain("height: auto");
    expect(workspace).toContain("min-height: 0");
  });
});

describe("session model inspector", () => {
  it("provides compact semantic targets for identity, access, and cumulative usage", () => {
    for (const id of [
      "model-select",
      "effort-select",
      "model-selection-state",
      "model-provider",
      "model-name",
      "model-access",
      "model-effort",
      "model-transport",
      "model-context",
      "model-usage-reporting",
      "stt-provider",
      "stt-model",
      "tts-provider",
      "tts-model",
      "usage-total",
      "usage-input",
      "usage-cached",
      "usage-output",
      "usage-reasoning",
      "latency-last",
      "latency-average",
    ]) {
      expect(markup).toContain(`id="${id}"`);
    }
    expect(declarations(".model-stats")).toContain("display: grid");
    expect(declarations(".model-stat")).toContain("grid-template-columns");
  });

  it("keeps idle approval and execution placeholders out of the inspector", () => {
    expect(markup).not.toContain('id="approval-panel"');
    expect(markup).not.toContain('id="task-state"');
    expect(markup).not.toContain("No approval requested.");
    expect(markup).not.toContain("Awaiting server state.");
  });
});
