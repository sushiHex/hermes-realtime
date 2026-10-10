export const OUTPUT_STYLE_KEY = "hermes-realtime.output-style.v1";
export const OUTPUT_STYLES = ["default", "proactive", "concise", "explanatory", "learning"] as const;
export type OutputStyle = typeof OUTPUT_STYLES[number];
type Submit = (style: OutputStyle) => Promise<unknown>;

function boundedStyle(value: unknown): value is OutputStyle {
  return typeof value === "string" && (OUTPUT_STYLES as readonly string[]).includes(value);
}

function acceptedStyle(value: unknown, requested: OutputStyle): boolean {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return Object.keys(record).sort().join(",") === "selectedStyle,version"
    && record.version === 1 && record.selectedStyle === requested;
}

/** Browser-local communication preference. Session ownership stays with the caller. */
export class OutputStyleControls {
  private selected: OutputStyle = "default";
  private saved = false;
  private revision = 0;
  private submit: Submit | null = null;
  private current: (() => boolean) | null = null;

  constructor(
    private readonly select: HTMLSelectElement,
    private readonly status: HTMLOutputElement,
    private readonly storage: () => Storage,
  ) {
    try {
      const stored = storage().getItem(OUTPUT_STYLE_KEY);
      if (boundedStyle(stored)) this.selected = stored;
      this.saved = true;
    } catch { this.saved = false; }
    select.replaceChildren(...OUTPUT_STYLES.map(style => {
      const option = select.ownerDocument.createElement("option");
      option.value = style;
      option.textContent = style[0]!.toUpperCase() + style.slice(1);
      return option;
    }));
    this.reset();
    select.addEventListener("change", () => {
      if (!boundedStyle(select.value)) return;
      const previous = this.selected;
      this.selected = select.value;
      if (this.submit !== null && this.current !== null && this.current()) {
        void this.apply(this.submit, this.current, previous);
      } else {
        this.persist();
        this.render("Connect to apply this preference.");
      }
    });
  }

  private persist(): void {
    try { this.storage().setItem(OUTPUT_STYLE_KEY, this.selected); this.saved = true; }
    catch { this.saved = false; }
  }

  private render(message: string): void {
    this.select.value = this.selected;
    this.status.textContent = message + (this.saved ? "" : " Preference not saved in this browser.");
  }

  reset(): void {
    this.revision += 1;
    this.submit = null;
    this.current = null;
    this.select.disabled = false;
    this.render("Connect to apply this preference.");
  }

  async sync(submit: Submit, current: () => boolean): Promise<void> {
    this.submit = submit;
    this.current = current;
    await this.apply(submit, current, null);
  }

  private async apply(submit: Submit, current: () => boolean, previous: OutputStyle | null): Promise<void> {
    const operation = ++this.revision;
    const requested = this.selected;
    this.select.disabled = true;
    this.render("Applying preference…");
    try {
      const reply = await submit(requested);
      if (operation !== this.revision || !current()) return;
      if (!acceptedStyle(reply, requested)) throw new Error("style acknowledgment rejected");
      this.persist();
      this.render("Accepted. Applies to the next response.");
    } catch {
      if (operation !== this.revision || !current()) return;
      if (previous !== null) {
        this.selected = previous;
        this.render("Selection failed; previous preference retained.");
      } else {
        this.submit = null;
        this.current = null;
        this.render("Output style unavailable on this host. Preference retained.");
      }
    } finally {
      if (operation === this.revision && current()) this.select.disabled = false;
    }
  }
}
