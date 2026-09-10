import type { ViteUserConfig } from "vitest/config";

// Registers the per-file ICU warm-up. No test authority changes: timeouts,
// environment, include, isolation, and pool all stay at Vitest's defaults.
// tests/setup.test.ts pins this shape. See issue #13.
export default {
  test: {
    setupFiles: ["./tests/setup.ts"],
  },
} satisfies ViteUserConfig;
