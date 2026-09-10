import { describe, expect, it } from "vitest";
import config from "../vitest.config";

describe("vitest setup", () => {
  it("registers only the ICU warm-up and no other test authority", () => {
    expect(config).toEqual({ test: { setupFiles: ["./tests/setup.ts"] } });
  });

  it("warmed locale formatting before this test file ran", () => {
    const warmup = globalThis.icuWarmup;
    if (warmup === undefined) throw new Error("tests/setup.ts did not run in this process");
    expect(warmup.pid).toBe(process.pid);
    expect(Number.isFinite(warmup.warmupMs)).toBe(true);
  });
});
