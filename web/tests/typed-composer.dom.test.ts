import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

import { mountTypedComposerEnterSubmission } from "../src/typed-composer";

function mountedComposer() {
  const dom = new JSDOM(`
    <form id="typed-form">
      <textarea id="typed-input"></textarea>
      <button type="submit">Send</button>
    </form>
  `);
  const form = dom.window.document.querySelector<HTMLFormElement>("#typed-form");
  const input = dom.window.document.querySelector<HTMLTextAreaElement>("#typed-input");
  const button = dom.window.document.querySelector<HTMLButtonElement>("button");
  if (form === null || input === null || button === null) {
    throw new Error("typed composer fixture is incomplete");
  }
  let submissions = 0;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submissions += 1;
  });
  const unmount = mountTypedComposerEnterSubmission(input, form, button);
  return { button, dom, input, submissions: () => submissions, unmount };
}

describe("mounted typed composer", () => {
  it("submits non-empty text when Enter is pressed", () => {
    const { dom, input, submissions, unmount } = mountedComposer();
    input.value = "Send this message";

    const event = new dom.window.KeyboardEvent("keydown", {
      key: "Enter",
      bubbles: true,
      cancelable: true,
    });
    const dispatched = input.dispatchEvent(event);

    expect(dispatched).toBe(false);
    expect(event.defaultPrevented).toBe(true);
    expect(submissions()).toBe(1);
    expect(input.value).toBe("Send this message");
    unmount();
  });

  it("does not resubmit text for a repeated Enter keydown", () => {
    const { dom, input, submissions, unmount } = mountedComposer();
    input.value = "Send once";

    const event = new dom.window.KeyboardEvent("keydown", {
      key: "Enter",
      repeat: true,
      bubbles: true,
      cancelable: true,
    });
    input.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
    expect(submissions()).toBe(0);
    unmount();
  });

  it("does not submit while Enter commits an input-method composition", () => {
    const { dom, input, submissions, unmount } = mountedComposer();
    input.value = "Composing";

    const event = new dom.window.KeyboardEvent("keydown", {
      key: "Enter",
      isComposing: true,
      bubbles: true,
      cancelable: true,
    });
    const dispatchResult = input.dispatchEvent(event);

    expect(dispatchResult).toBe(true);
    expect(event.defaultPrevented).toBe(false);
    expect(submissions()).toBe(0);
    unmount();
  });

  it("does not submit when Safari confirms a composition after compositionend", () => {
    const { dom, input, submissions, unmount } = mountedComposer();
    input.value = "Composing";

    const event = new dom.window.KeyboardEvent("keydown", {
      key: "Enter",
      isComposing: false,
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperty(event, "keyCode", { value: 229 });
    const dispatchResult = input.dispatchEvent(event);

    expect(event.isComposing).toBe(false);
    expect(event.keyCode).toBe(229);
    expect(dispatchResult).toBe(true);
    expect(event.defaultPrevented).toBe(false);
    expect(submissions()).toBe(0);
    unmount();
  });

  it("does not submit while the send control is disabled", () => {
    const { button, dom, input, submissions, unmount } = mountedComposer();
    input.value = "Already sending";
    button.disabled = true;

    const event = new dom.window.KeyboardEvent("keydown", {
      key: "Enter",
      bubbles: true,
      cancelable: true,
    });
    input.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
    expect(submissions()).toBe(0);
    unmount();
  });
});
