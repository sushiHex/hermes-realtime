import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { describe, expect, it, vi } from "vitest";
import { PlaybackVolumeControls, PLAYBACK_VOLUME_KEY } from "../src/playback-volume";

function mount(saved?: string, storageBlocked = false) {
  const dom = new JSDOM(readFileSync(new URL("../index.html", import.meta.url), "utf8"), {
    url: "http://localhost/",
  });
  const storage = dom.window.localStorage;
  if (saved !== undefined) storage.setItem(PLAYBACK_VOLUME_KEY, saved);
  const slider = dom.window.document.querySelector<HTMLInputElement>("#speaker-volume")!;
  const mute = dom.window.document.querySelector<HTMLButtonElement>("#speaker-mute")!;
  const status = dom.window.document.querySelector<HTMLOutputElement>("#speaker-volume-status")!;
  const write = vi.fn();
  const control = new PlaybackVolumeControls(slider, mute, status, () => {
    if (storageBlocked) throw new Error("storage blocked");
    return storage;
  }, write);
  const choose = (value: string) => {
    slider.value = value;
    slider.dispatchEvent(new dom.window.Event("input"));
  };
  return { dom, storage, slider, mute, status, write, control, choose };
}

describe("speaker volume preference", () => {
  it("labels native keyboard controls and applies input immediately", () => {
    const view = mount();
    expect(view.slider.type).toBe("range");
    expect(view.slider.labels?.[0]?.textContent).toContain("Speaker");
    expect([view.slider.min, view.slider.max, view.slider.step]).toEqual(["0", "100", "1"]);
    expect(view.slider.closest(".device-controls")?.querySelector("#voice")).not.toBeNull();
    expect(view.write.mock.calls).toEqual([[1]]);
    view.choose("37");
    expect(view.write).toHaveBeenLastCalledWith(0.37);
    expect(view.slider.getAttribute("aria-valuetext")).toBe("37 percent");
    expect(view.status.textContent).toBe("37%");
    expect(view.mute.getAttribute("aria-label")).toBe("Mute speaker");
    expect(view.mute.getAttribute("aria-pressed")).toBe("false");
    view.dom.window.close();
  });

  it("keeps mute independent of the chosen level and restores both on refresh", () => {
    const view = mount();
    view.choose("42");
    view.mute.click();
    expect(view.write).toHaveBeenLastCalledWith(0);
    expect(view.slider.value).toBe("42");
    expect(view.status.textContent).toBe("Muted · 42%");
    expect(view.mute.getAttribute("aria-label")).toBe("Unmute speaker");
    expect(view.mute.getAttribute("aria-pressed")).toBe("true");
    view.choose("65");
    expect(view.write).toHaveBeenLastCalledWith(0);
    const saved = view.storage.getItem(PLAYBACK_VOLUME_KEY)!;
    expect(JSON.parse(saved)).toEqual({ volume: 0.65, muted: true });
    const refreshed = mount(saved);
    expect(refreshed.slider.value).toBe("65");
    expect(refreshed.write).toHaveBeenLastCalledWith(0);
    refreshed.mute.click();
    expect(refreshed.write).toHaveBeenLastCalledWith(0.65);
    expect(JSON.parse(refreshed.storage.getItem(PLAYBACK_VOLUME_KEY)!)).toEqual({ volume: 0.65, muted: false });
    view.dom.window.close();
    refreshed.dom.window.close();
  });

  it("multiplies live changes by attenuation and preserves committed silence", () => {
    const view = mount(JSON.stringify({ volume: 0.5, muted: false }));
    view.control.setAttenuation(0.18);
    expect(view.write).toHaveBeenLastCalledWith(0.09);
    view.choose("20");
    expect(view.write).toHaveBeenLastCalledWith(0.036);
    view.mute.click();
    view.control.setAttenuation(1);
    expect(view.write).toHaveBeenLastCalledWith(0);
    view.mute.click();
    expect(view.write).toHaveBeenLastCalledWith(0.2);
    view.control.setAttenuation(0);
    view.choose("80");
    expect(view.write).toHaveBeenLastCalledWith(0);
    view.control.setAttenuation(1);
    expect(view.write).toHaveBeenLastCalledWith(0.8);
    view.dom.window.close();
  });

  it.each([
    ["missing", undefined], ["malformed JSON", "{"], ["null", "null"],
    ["extra key", '{"volume":0.5,"muted":false,"other":1}'],
    ["string volume", '{"volume":"0.5","muted":false}'],
    ["negative volume", '{"volume":-0.1,"muted":false}'],
    ["excess volume", '{"volume":1.1,"muted":false}'],
    ["non-boolean mute", '{"volume":0.5,"muted":"false"}'],
  ])("falls back to unity for %s storage", (_name, saved) => {
    const view = mount(saved);
    expect(view.write.mock.calls).toEqual([[1]]);
    expect(view.slider.value).toBe("100");
    expect(view.mute.getAttribute("aria-pressed")).toBe("false");
    view.dom.window.close();
  });

  it.each([0, 1])("accepts saved boundary level %s", volume => {
    const view = mount(JSON.stringify({ volume, muted: false }));
    expect(view.write).toHaveBeenLastCalledWith(volume);
    view.dom.window.close();
  });

  it("remains usable and reports when browser storage is blocked", () => {
    const view = mount(undefined, true);
    view.choose("30");
    expect(view.write).toHaveBeenLastCalledWith(0.3);
    expect(view.status.textContent).toBe("30% · not saved");
    view.mute.click();
    expect(view.write).toHaveBeenLastCalledWith(0);
    view.dom.window.close();
  });
});
