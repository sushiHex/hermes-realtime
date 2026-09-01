export function mountTypedComposerEnterSubmission(
  input: HTMLTextAreaElement,
  form: HTMLFormElement,
  submitButton: HTMLButtonElement,
): () => void {
  const submitOnEnter = (event: KeyboardEvent): void => {
    if (event.key !== "Enter") return;
    // Safari can fire the Enter keydown that confirms an input-method candidate
    // *after* compositionend, so isComposing is already false by the time this
    // runs (WebKit bug 165004). keyCode is still 229 there, which is the signal
    // MDN documents for exactly this event ordering. keyCode is deprecated but
    // remains the only reliable discriminator for this case.
    if (event.isComposing || event.keyCode === 229) return;
    // This is a multi-line composer, so Shift+Enter stays a newline: it must
    // neither submit nor swallow the browser's default insertion.
    if (event.shiftKey) return;
    event.preventDefault();
    if (event.repeat || submitButton.disabled) return;
    form.requestSubmit(submitButton);
  };
  input.addEventListener("keydown", submitOnEnter);
  return () => input.removeEventListener("keydown", submitOnEnter);
}
