export function mountTypedComposerEnterSubmission(
  input: HTMLTextAreaElement,
  form: HTMLFormElement,
  submitButton: HTMLButtonElement,
): () => void {
  const submitOnEnter = (event: KeyboardEvent): void => {
    if (event.key !== "Enter") return;
    // Safari can emit the Enter keydown that confirms an IME candidate after
    // compositionend, when isComposing is already false but keyCode remains 229.
    if (event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    if (event.repeat || submitButton.disabled) return;
    form.requestSubmit(submitButton);
  };
  input.addEventListener("keydown", submitOnEnter);
  return () => input.removeEventListener("keydown", submitOnEnter);
}
