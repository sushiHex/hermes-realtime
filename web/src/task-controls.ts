export interface TaskCancelControlOptions {
  readonly taskId: string;
  readonly canCancel: () => boolean;
  readonly submit: (command: string) => Promise<void>;
  readonly onFailure: () => void;
}

export function mountTaskCancelControl(
  button: HTMLButtonElement,
  options: TaskCancelControlOptions,
): { refresh(): void; dispose(): void } {
  const taskId = options.taskId;
  try {
    if (!/^task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,122}$/.test(taskId) || /deleg_/.test(taskId)) {
      throw new Error("invalid public task identity");
    }
  } catch (error) {
    try {
      throw error;
    } finally {
      console.info('[task-control-refusal] {"kind":"cancel","category":"invalid-identity"}');
    }
  }
  let pending = false;
  let disposed = false;
  const refresh = (): void => {
    button.disabled = disposed || pending || !options.canCancel();
  };
  const click = async (): Promise<void> => {
    if (pending || !options.canCancel()) {
      try {
        return;
      } finally {
        console.info('[task-control-refusal] {"kind":"cancel","category":"unavailable"}');
      }
    }
    pending = true;
    refresh();
    try {
      await options.submit(`cancel task: ${taskId}`);
    } catch {
      if (!disposed) options.onFailure();
    } finally {
      pending = false;
      refresh();
    }
  };
  button.addEventListener("click", click);
  refresh();
  return {
    refresh,
    dispose(): void {
      disposed = true;
      button.removeEventListener("click", click);
      refresh();
    },
  };
}
