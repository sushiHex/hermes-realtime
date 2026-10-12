import { JSDOM } from "jsdom";
import { describe, expect, it, vi } from "vitest";

import { mountTaskCancelControl } from "../src/task-controls";

function fixture() {
  const dom = new JSDOM("<button>Cancel</button>");
  const button = dom.window.document.querySelector<HTMLButtonElement>("button")!;
  let actionable = true;
  const submit = vi.fn(async (_command: string) => {});
  const failure = vi.fn();
  const options = {
    taskId: "task_fixture",
    canCancel: () => actionable,
    submit,
    onFailure: failure,
  };
  const control = mountTaskCancelControl(button, options);
  const click = () => button.dispatchEvent(new dom.window.MouseEvent("click"));
  return { button, control, click, submit, failure, options, setActionable: (value: boolean) => { actionable = value; } };
}

describe("exact task cancellation control", () => {
  it("submits the card's captured public identity without claiming cancellation", async () => {
    const { button, click, submit, options } = fixture();
    options.taskId = "task_replacement";
    click();
    expect(submit).toHaveBeenCalledExactlyOnceWith("cancel task: task_fixture");
    expect(button.disabled).toBe(true);
    expect(button.textContent).toBe("Cancel");
    await Promise.resolve();
    expect(button.disabled).toBe(false);
  });

  it("refuses a disconnected or terminal card even if a click is dispatched", () => {
    const { button, click, submit, control, setActionable } = fixture();
    setActionable(false);
    control.refresh();
    expect(button.disabled).toBe(true);
    click();
    expect(submit).not.toHaveBeenCalled();
  });

  it("admits at most one cancellation while submission is pending", async () => {
    const { click, submit } = fixture();
    let release!: () => void;
    submit.mockImplementationOnce(() => new Promise<void>((resolve) => { release = resolve; }));
    click();
    click();
    expect(submit).toHaveBeenCalledTimes(1);
    release();
    await Promise.resolve();
  });

  it("reports admission failure and restores an actionable control", async () => {
    const { button, click, submit, failure } = fixture();
    submit.mockRejectedValueOnce(new Error("synthetic admission failure"));
    click();
    await Promise.resolve();
    expect(failure).toHaveBeenCalledTimes(1);
    expect(button.disabled).toBe(false);
  });

  it("disposes the handler and keeps the obsolete control disabled", () => {
    const { button, click, submit, control } = fixture();
    control.dispose();
    control.refresh();
    click();
    expect(button.disabled).toBe(true);
    expect(submit).not.toHaveBeenCalled();
  });

  it("does not report a late admission failure to a disposed card", async () => {
    const { button, click, submit, failure, control } = fixture();
    let reject!: (reason: Error) => void;
    submit.mockImplementationOnce(() => new Promise<void>((_resolve, refusal) => { reject = refusal; }));
    click();
    control.dispose();
    reject(new Error("synthetic stale failure"));
    await Promise.resolve();
    expect(failure).not.toHaveBeenCalled();
    expect(button.disabled).toBe(true);
  });

  it.each(["wrong_id", "task_fixture extra", "task_deleg_private"])("refuses an invalid public identity (%s)", (taskId) => {
    const dom = new JSDOM("<button>Cancel</button>");
    const button = dom.window.document.querySelector<HTMLButtonElement>("button")!;
    expect(() => mountTaskCancelControl(button, {
      taskId, canCancel: () => true, submit: async () => {}, onFailure: () => {},
    })).toThrow("invalid public task identity");
  });
});
