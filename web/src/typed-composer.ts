export function mountTypedComposerEnterSubmission(
  input: HTMLTextAreaElement,
  form: HTMLFormElement,
  submitButton: HTMLButtonElement,
): () => void {
  const submitOnEnter = (event: KeyboardEvent): void => {
    if (event.key !== "Enter" || event.isComposing) return;
    event.preventDefault();
    if (event.repeat || submitButton.disabled) return;
    form.requestSubmit(submitButton);
  };
  input.addEventListener("keydown", submitOnEnter);
  return () => input.removeEventListener("keydown", submitOnEnter);
}
