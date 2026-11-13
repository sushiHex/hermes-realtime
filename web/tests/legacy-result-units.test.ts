import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { MAX_LEGACY_RESULT_CHARS, parseEventBatch } from "../src/protocol";

type Segments = [string, number][];
type Case = { name: string; reason: Segments; text: Segments };

// Shared with tests/test_legacy_result_units.py, which asserts the server publishes each text.
const fixture = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../tests/fixtures/legacy_result_utf16.json", import.meta.url)),
    "utf8",
  ),
) as { limit_utf16_units: number; cases: Case[] };

const expand = (segments: Segments): string =>
  segments.map(([text, repeat]) => text.repeat(repeat)).join("");

const batch = (version: 1 | 2, status: string, text: string) => ({
  version,
  events: [{
    sequence: 1,
    kind: "task_result",
    monotonicMs: 1,
    data: { status, taskId: "task_astral", text },
  }],
});

describe("legacy result length is counted in UTF-16 code units", () => {
  it("shares its bound with the server", () => {
    expect(fixture.limit_utf16_units).toBe(MAX_LEGACY_RESULT_CHARS);
    expect(fixture.cases.length).toBeGreaterThanOrEqual(5);
  });

  for (const item of fixture.cases) {
    for (const status of ["failed", "interrupted"]) {
      it(`admits the published ${status} text: ${item.name}`, () => {
        const text = expand(item.text);
        expect(text.length).toBeLessThanOrEqual(MAX_LEGACY_RESULT_CHARS);
        for (const version of [1, 2] as const) {
          expect(parseEventBatch(batch(version, status, text)).events[0]?.data.text).toBe(text);
        }
      });
    }
  }

  it("refuses an astral text that is within the bound only when counted in scalars", () => {
    const text = "😀".repeat(513);
    expect(Array.from(text)).toHaveLength(513);
    expect(text.length).toBe(1026);
    for (const status of ["failed", "interrupted"]) {
      expect(() => parseEventBatch(batch(2, status, text))).toThrow("Public task result");
    }
    expect(() => parseEventBatch(batch(1, "completed", text))).toThrow("Public task result");
  });
});
