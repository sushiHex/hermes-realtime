import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

import {
  EvidenceControlTimeoutError,
  mountEvidenceControls,
} from "../src/evidence-controls";

const markup = readFileSync(
  fileURLToPath(new URL("../index.html", import.meta.url)),
  "utf8",
);

const digest = "a".repeat(64);

describe("mounted evidence capture controls", () => {
  it("retries a timed-out consent with the exact request without treating the timeout as terminal", async () => {
    const dom = new JSDOM(markup, { url: "https://assistant.example.test/" });
    const calls: Array<{ path: string; body: unknown }> = [];
    let attempts = 0;
    const controls = mountEvidenceControls(dom.window.document, {
      submit: async (path, body) => {
        calls.push({ path, body });
        attempts += 1;
        if (attempts === 1) throw new EvidenceControlTimeoutError();
      },
    });
    const status = dom.window.document.querySelector<HTMLOutputElement>(
      "#evidence-capture-status",
    );
    const consent = dom.window.document.querySelector<HTMLButtonElement>("#evidence-consent");
    const revoke = dom.window.document.querySelector<HTMLButtonElement>("#evidence-revoke");

    controls.projectStatus({
      available: true,
      captureState: "idle",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    consent?.click();
    await Promise.resolve();

    expect(status?.dataset.captureState).toBe("idle");
    expect(status?.textContent).toMatch(/retry/i);
    expect(consent?.disabled).toBe(false);
    expect(consent?.getAttribute("aria-label")).toMatch(/retry/i);
    expect(revoke?.disabled).toBe(true);

    consent?.click();
    consent?.click();
    await Promise.resolve();
    expect(calls).toEqual([calls[0], calls[0]]);
    expect(calls[0]).toEqual({
      path: "/api/v1/evidence-consent",
      body: {
        accepted: true,
        consentVersion: "realtime-evidence-consent-v1",
        disclosureDigest: digest,
        retentionHours: 48,
        sequence: 1,
        sources: { microphone: true, typed: true },
      },
    });

    controls.projectStatus({
      available: true,
      captureState: "active",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    expect(consent?.disabled).toBe(true);
    expect(revoke?.disabled).toBe(false);
  });

  it("retries a timed-out revoke with the exact request despite revoked-purging projection", async () => {
    const dom = new JSDOM(markup, { url: "https://assistant.example.test/" });
    const calls: Array<{ path: string; body: unknown }> = [];
    let attempts = 0;
    const controls = mountEvidenceControls(dom.window.document, {
      submit: async (path, body) => {
        calls.push({ path, body });
        attempts += 1;
        if (attempts === 1) throw new EvidenceControlTimeoutError();
      },
    });
    const status = dom.window.document.querySelector<HTMLOutputElement>(
      "#evidence-capture-status",
    );
    const consent = dom.window.document.querySelector<HTMLButtonElement>("#evidence-consent");
    const revoke = dom.window.document.querySelector<HTMLButtonElement>("#evidence-revoke");

    controls.projectStatus({
      available: true,
      captureState: "active",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    revoke?.click();
    await Promise.resolve();
    controls.projectStatus({
      available: true,
      captureState: "revoked_purging",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });

    expect(status?.dataset.captureState).toBe("revoked_purging");
    expect(status?.textContent).toMatch(/retry/i);
    expect(revoke?.disabled).toBe(false);
    expect(revoke?.getAttribute("aria-label")).toMatch(/retry/i);
    expect(consent?.disabled).toBe(true);

    revoke?.click();
    revoke?.click();
    await Promise.resolve();
    expect(calls).toEqual([
      { path: "/api/v1/evidence-revoke", body: { sequence: 1 } },
      { path: "/api/v1/evidence-revoke", body: { sequence: 1 } },
    ]);
    controls.projectStatus({
      available: true,
      captureState: "idle",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    expect(consent?.disabled).toBe(false);
  });

  it("keeps a terminal evidence control failure disabled and in error", async () => {
    const dom = new JSDOM(markup, { url: "https://assistant.example.test/" });
    const controls = mountEvidenceControls(dom.window.document, {
      submit: async () => {
        throw new Error("terminal control rejection");
      },
    });
    const status = dom.window.document.querySelector<HTMLOutputElement>(
      "#evidence-capture-status",
    );
    const consent = dom.window.document.querySelector<HTMLButtonElement>("#evidence-consent");

    controls.projectStatus({
      available: true,
      captureState: "idle",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    consent?.click();
    await Promise.resolve();

    expect(status?.dataset.captureState).toBe("error");
    expect(consent?.disabled).toBe(true);
  });

  it("keeps accessible consent and revoke controls beside the canonical disclosure", async () => {
    const dom = new JSDOM(markup, { url: "https://assistant.example.test/" });
    const calls: Array<{ path: string; body: unknown }> = [];
    let completeFirstRequest: (() => void) | undefined;
    const firstRequest = new Promise<void>((resolve) => {
      completeFirstRequest = resolve;
    });
    const controls = mountEvidenceControls(dom.window.document, {
      submit: async (path, body) => {
        calls.push({ path, body });
        if (calls.length === 1) await firstRequest;
      },
    });
    const disclosure = dom.window.document.querySelector("#evidence-disclosure");
    const status = dom.window.document.querySelector<HTMLOutputElement>(
      "#evidence-capture-status",
    );
    const consent = dom.window.document.querySelector<HTMLButtonElement>("#evidence-consent");
    const revoke = dom.window.document.querySelector<HTMLButtonElement>("#evidence-revoke");

    expect(disclosure?.contains(status)).toBe(true);
    expect(disclosure?.contains(consent)).toBe(true);
    expect(disclosure?.contains(revoke)).toBe(true);
    expect(status?.getAttribute("role")).toBe("status");
    expect(status?.getAttribute("aria-live")).toBe("polite");
    expect(consent?.textContent).toMatch(/enable local capture/i);
    expect(revoke?.textContent).toMatch(/revoke.*erase/i);

    controls.projectStatus({
      available: true,
      captureState: "idle",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    expect(status?.dataset.captureState).toBe("idle");
    expect(consent?.disabled).toBe(false);
    expect(revoke?.disabled).toBe(true);

    consent?.click();
    consent?.click();
    expect(calls).toEqual([
      {
        path: "/api/v1/evidence-consent",
        body: {
          accepted: true,
          consentVersion: "realtime-evidence-consent-v1",
          disclosureDigest: digest,
          retentionHours: 48,
          sequence: 1,
          sources: { microphone: true, typed: true },
        },
      },
    ]);
    expect(consent?.disabled).toBe(true);
    expect(revoke?.disabled).toBe(true);
    completeFirstRequest?.();
    await new Promise<void>((resolve) => dom.window.setTimeout(resolve, 0));
    expect(consent?.disabled).toBe(true);

    controls.projectStatus({
      available: true,
      captureState: "active",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    expect(status?.dataset.captureState).toBe("active");
    expect(consent?.disabled).toBe(true);
    expect(revoke?.disabled).toBe(false);

    revoke?.click();
    await Promise.resolve();
    expect(calls.at(-1)).toEqual({
      path: "/api/v1/evidence-revoke",
      body: { sequence: 2 },
    });
    controls.projectStatus({
      available: true,
      captureState: "revoked_purging",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    expect(status?.dataset.captureState).toBe("revoked_purging");
    expect(consent?.disabled).toBe(true);
    expect(revoke?.disabled).toBe(true);

    controls.projectStatus({
      available: true,
      captureState: "faulted",
      consentVersion: "realtime-evidence-consent-v1",
      disclosureDigest: digest,
      retentionHours: 48,
    });
    expect(status?.dataset.captureState).toBe("error");
    expect(status?.textContent).toMatch(/error/i);
    expect(dom.window.document.documentElement.outerHTML).not.toContain("Bearer ");
  });
});
