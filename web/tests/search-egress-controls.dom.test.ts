import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

import { mountSearchEgressControls } from "../src/search-egress-controls";

const markup = readFileSync(
  fileURLToPath(new URL("../index.html", import.meta.url)),
  "utf8",
);
const digest = "a".repeat(64);

describe("mounted public-search egress controls", () => {
  it("lets authoritative active status recover a lost consent response and revoke", async () => {
    const dom = new JSDOM(markup);
    const submitted: Array<{ path: string; body: object }> = [];
    let rejectConsent!: (reason: Error) => void;
    const uncertainConsent = new Promise<void>((_resolve, reject) => {
      rejectConsent = reject;
    });
    const controls = mountSearchEgressControls(dom.window.document, {
      submit: async (path, body) => {
        submitted.push({ path, body });
        if (submitted.length === 1) await uncertainConsent;
      },
    });
    const consent = dom.window.document.getElementById(
      "search-egress-consent",
    ) as HTMLButtonElement;
    const revoke = dom.window.document.getElementById(
      "search-egress-revoke",
    ) as HTMLButtonElement;
    const status = dom.window.document.getElementById(
      "search-egress-status",
    ) as HTMLOutputElement;

    controls.projectStatus({
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: digest,
      searchEgressState: "idle",
    });
    consent.click();
    await Promise.resolve();
    controls.projectStatus({
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: digest,
      searchEgressState: "active",
    });
    rejectConsent(new Error("response lost after authoritative acceptance"));
    await Promise.resolve();
    await Promise.resolve();

    expect(status.dataset.searchEgressState).toBe("active");
    expect(revoke.disabled).toBe(false);
    revoke.click();
    await Promise.resolve();
    expect(submitted[1]).toEqual({
      path: "/api/v1/search-egress-revoke",
      body: { sequence: 2 },
    });
  });

  it("reconciles a failed consent response from a later authoritative active status", async () => {
    const dom = new JSDOM(markup);
    const submitted: Array<{ path: string; body: object }> = [];
    const controls = mountSearchEgressControls(dom.window.document, {
      submit: async (path, body) => {
        submitted.push({ path, body });
        if (submitted.length === 1) throw new Error("response lost after acceptance");
      },
    });
    const consent = dom.window.document.getElementById(
      "search-egress-consent",
    ) as HTMLButtonElement;
    const revoke = dom.window.document.getElementById(
      "search-egress-revoke",
    ) as HTMLButtonElement;
    const status = dom.window.document.getElementById(
      "search-egress-status",
    ) as HTMLOutputElement;

    controls.projectStatus({
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: digest,
      searchEgressState: "idle",
    });
    consent.click();
    await Promise.resolve();
    await Promise.resolve();
    expect(status.dataset.searchEgressState).toBe("error");

    controls.projectStatus({
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: digest,
      searchEgressState: "active",
    });

    expect(status.dataset.searchEgressState).toBe("active");
    expect(revoke.disabled).toBe(false);
    revoke.click();
    await Promise.resolve();
    expect(submitted[1]).toEqual({
      path: "/api/v1/search-egress-revoke",
      body: { sequence: 2 },
    });
  });

  it("fences an in-flight consent completion across binding reset", async () => {
    const dom = new JSDOM(markup);
    const submitted: Array<{ path: string; body: object }> = [];
    let releaseFirst!: () => void;
    const firstPending = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const controls = mountSearchEgressControls(dom.window.document, {
      submit: async (path, body) => {
        submitted.push({ path, body });
        if (submitted.length === 1) await firstPending;
      },
    });
    controls.setInteractive(true);
    controls.projectStatus({
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: "a".repeat(64),
      searchEgressState: "idle",
    });

    const consent = dom.window.document.getElementById(
      "search-egress-consent",
    ) as HTMLButtonElement;
    consent.click();
    await Promise.resolve();
    controls.reset();
    releaseFirst();
    await Promise.resolve();
    await Promise.resolve();

    controls.setInteractive(true);
    controls.projectStatus({
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: "a".repeat(64),
      searchEgressState: "idle",
    });
    consent.click();
    await Promise.resolve();

    expect(submitted).toHaveLength(2);
    expect(submitted[1]).toEqual({
      path: "/api/v1/search-egress-consent",
      body: {
        accepted: true,
        consentVersion: "realtime-search-egress-consent-v1",
        disclosureDigest: "a".repeat(64),
        sequence: 1,
      },
    });
  });

  it("renders a separate exact disclosure and sequences consent then revoke", async () => {
    const dom = new JSDOM(markup, { url: "https://assistant.example.test/" });
    const calls: Array<{ path: string; body: unknown }> = [];
    const controls = mountSearchEgressControls(dom.window.document, {
      submit: async (path, body) => {
        calls.push({ path, body });
      },
    });
    const disclosure = dom.window.document.querySelector("#search-egress-disclosure");
    const status = dom.window.document.querySelector<HTMLOutputElement>(
      "#search-egress-status",
    );
    const consent = dom.window.document.querySelector<HTMLButtonElement>(
      "#search-egress-consent",
    );
    const revoke = dom.window.document.querySelector<HTMLButtonElement>(
      "#search-egress-revoke",
    );

    expect(disclosure?.textContent).toMatch(/Bing Search/);
    expect(disclosure?.textContent).toMatch(/Google News RSS/);
    expect(disclosure?.textContent).toMatch(/derived from.*speech.*typed/i);
    expect(disclosure?.contains(status)).toBe(true);
    expect(disclosure?.contains(consent)).toBe(true);
    expect(disclosure?.contains(revoke)).toBe(true);

    controls.projectStatus({
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: digest,
      searchEgressState: "idle",
    });
    expect(consent?.disabled).toBe(false);
    expect(revoke?.disabled).toBe(true);
    consent?.click();
    await Promise.resolve();
    expect(calls).toEqual([
      {
        path: "/api/v1/search-egress-consent",
        body: {
          accepted: true,
          consentVersion: "realtime-search-egress-consent-v1",
          disclosureDigest: digest,
          sequence: 1,
        },
      },
    ]);

    controls.projectStatus({
      available: true,
      consentVersion: "realtime-search-egress-consent-v1",
      disclosureDigest: digest,
      searchEgressState: "active",
    });
    expect(consent?.disabled).toBe(true);
    expect(revoke?.disabled).toBe(false);
    revoke?.click();
    await Promise.resolve();
    expect(calls[1]).toEqual({
      path: "/api/v1/search-egress-revoke",
      body: { sequence: 2 },
    });

    controls.reset();
    expect(status?.dataset.searchEgressState).toBe("unavailable");
    expect(consent?.disabled).toBe(true);
    expect(revoke?.disabled).toBe(true);
  });
});
