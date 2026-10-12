import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";
import { OutputStyleControls, OUTPUT_STYLE_KEY } from "../src/output-style";

function mount(value?: string, blocked = false) {
  const dom = new JSDOM('<select id="style"></select><output id="status"></output>', {url:"http://localhost"});
  if (value !== undefined) dom.window.localStorage.setItem(OUTPUT_STYLE_KEY, value);
  const select = dom.window.document.querySelector<HTMLSelectElement>("select")!;
  const status = dom.window.document.querySelector<HTMLOutputElement>("output")!;
  const control = new OutputStyleControls(select, status, () => {
    if (blocked) throw new Error("blocked");
    return dom.window.localStorage;
  });
  const choose = (style: string) => { select.value = style; select.dispatchEvent(new dom.window.Event("change")); };
  return { dom, select, status, control, choose };
}

describe("output style preference", () => {
  it("restores only bounded styles and persists offline preference in the same origin", () => {
    const view = mount("unknown");
    expect(view.select.value).toBe("default");
    expect([...view.select.options].map(option => option.text)).toEqual(["Default", "Proactive", "Concise", "Explanatory", "Learning"]);
    view.choose("learning");
    expect(view.dom.window.localStorage.getItem(OUTPUT_STYLE_KEY)).toBe("learning");
    expect(mount("learning").select.value).toBe("learning");
  });

  it("reports blocked storage without breaking selection", async () => {
    const view = mount(undefined, true);
    await view.control.sync(async style => ({version:1, selectedStyle:style}), () => true);
    view.choose("concise");
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(view.select.value).toBe("concise");
    expect(view.status.textContent).toContain("not saved");
    expect(view.status.textContent).toContain("next response");
  });

  it("reapplies the saved choice across reconnect and server restart", async () => {
    const view = mount("explanatory");
    const sent: string[] = [];
    const submit = async (style: string) => { sent.push(style); return {version:1, selectedStyle:style}; };
    await view.control.sync(submit, () => true);
    view.control.reset();
    await view.control.sync(submit, () => true);
    expect(sent).toEqual(["explanatory", "explanatory"]);
    expect(view.status.textContent).toContain("next response");
  });

  it("ignores stale acknowledgments and rolls back rejected connected changes", async () => {
    const view = mount("concise");
    let finish!: (value: unknown) => void;
    const first = view.control.sync(() => new Promise(resolve => {finish = resolve;}), () => true);
    view.control.reset();
    view.choose("learning");
    finish({version:1, selectedStyle:"concise"});
    await first;
    expect(view.select.value).toBe("learning");
    expect(view.status.textContent).toContain("Connect to apply");
    let calls = 0;
    await view.control.sync(async style => {
      if (++calls > 1) throw new Error("rejected");
      return {version:1, selectedStyle:style};
    }, () => true);
    view.choose("proactive");
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(view.select.value).toBe("learning");
    expect(view.status.textContent).toBe("Selection not confirmed. Reconnect to reapply the saved preference.");
    expect(view.dom.window.localStorage.getItem(OUTPUT_STYLE_KEY)).toBe("learning");
  });

  it("does not claim unsupported or malformed acknowledgments applied", async () => {
    for (const reply of [null, {version:1, selectedStyle:"unknown"}, {version:1, selectedStyle:"default", extra:true}]) {
      const view = mount();
      await view.control.sync(async () => reply, () => true);
      expect(view.status.textContent).toContain("unavailable");
      expect(view.status.textContent).not.toContain("Applies to the next response");
    }
  });

  it("ignores an acknowledgment after caller loses session ownership", async () => {
    const view = mount("learning");
    let current = true;
    let finish!: (value: unknown) => void;
    const pending = view.control.sync(() => new Promise(resolve => {finish = resolve;}), () => current);
    current = false;
    finish({version:1, selectedStyle:"learning"});
    await pending;
    expect(view.status.textContent).not.toContain("Accepted");
  });

  it("keeps a newer selection pending when an older operation settles", async () => {
    const view = mount();
    let oldFinish!: (value: unknown) => void;
    let newFinish!: (value: unknown) => void;
    const old = view.control.sync(() => new Promise(resolve => {oldFinish = resolve;}), () => true);
    view.control.reset();
    view.choose("learning");
    const next = view.control.sync(() => new Promise(resolve => {newFinish = resolve;}), () => true);
    oldFinish({version:1, selectedStyle:"default"});
    await old;
    expect(view.select.disabled).toBe(true);
    expect(view.status.textContent).toContain("Applying");
    newFinish({version:1, selectedStyle:"learning"});
    await next;
    expect(view.select.disabled).toBe(false);
  });
});
