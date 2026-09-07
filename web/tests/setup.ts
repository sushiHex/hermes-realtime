// Warm ICU locale formatting before each test file is collected and outside
// every test's timeout authority. Before this setup, formatSessionTokenUsage
// in controller.test.ts was the suite's only executed locale-formatting path,
// so it paid that first touch inside a 5,000 ms test bound. Vitest retains the
// aggregate setup duration; the log adds best-effort per-file attribution.
// Timing history: issue #13.
declare global {
  var icuWarmup: Readonly<{ pid: number; warmupMs: number }> | undefined;
}

const started = performance.now();
(1000).toLocaleString("en-US");
const warmupMs = performance.now() - started;

globalThis.icuWarmup = Object.freeze({ pid: process.pid, warmupMs });
console.log(`[icu-warmup] pid=${process.pid} warmup_ms=${warmupMs.toFixed(3)}`);

export {};
