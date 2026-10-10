/** Owns history following and the one authoritative active-card stack. */
export class ConversationHistory {
  private following = true;
  private readonly dock: HTMLDivElement;
  private readonly stack: HTMLOListElement;
  private readonly active = new Map<HTMLLIElement, HTMLLIElement | null>();
  private readonly observer: ResizeObserver | null;

  constructor(private readonly scroller: HTMLElement) {
    const document = scroller.ownerDocument;
    this.dock = document.createElement("div");
    this.dock.className = "active-task-dock";
    this.stack = document.createElement("ol");
    this.stack.className = "timeline active-task-stack";
    this.stack.setAttribute("role", "region");
    this.stack.setAttribute("aria-label", "Active background tasks");
    this.stack.tabIndex = 0;
    this.dock.append(this.stack);
    scroller.prepend(this.dock);
    scroller.addEventListener("scroll", () => {
      this.following = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop <= 2;
      this.refresh();
    });
    const Observer = scroller.ownerDocument.defaultView?.ResizeObserver;
    this.observer = Observer === undefined ? null : new Observer(() => this.refresh());
    this.observer?.observe(scroller);
    this.refresh();
  }

  isActive(item: HTMLLIElement): boolean {
    return this.active.has(item);
  }

  setActive(item: HTMLLIElement, active: boolean): void {
    if (active) {
      if (!this.active.has(item)) {
        this.active.set(item, null);
        this.observer?.observe(item);
      }
    } else {
      const anchor = this.active.get(item);
      if (anchor !== undefined && anchor !== null) anchor.replaceWith(item);
      this.active.delete(item);
      this.observer?.unobserve(item);
    }
    this.refresh();
  }

  forget(item: HTMLLIElement): void {
    this.active.get(item)?.remove();
    this.active.delete(item);
    this.observer?.unobserve(item);
    this.refresh();
  }

  /** Only the history viewport moves; highlighting has no scroll authority. */
  follow(): void {
    if (this.following) this.scroller.scrollTop = this.scroller.scrollHeight;
    this.refresh();
  }

  refresh(): void {
    this.stack.style.maxHeight = `${this.scroller.clientHeight * 0.4}px`;
    const top = this.scroller.getBoundingClientRect().top;
    for (const [item, anchor] of this.active) {
      if (anchor !== null) {
        anchor.style.height = `${item.getBoundingClientRect().height}px`;
      } else if (item.isConnected && item.getBoundingClientRect().top <= top) {
        const placeholder = item.ownerDocument.createElement("li");
        placeholder.className = "task-history-slot";
        placeholder.setAttribute("aria-hidden", "true");
        placeholder.style.height = `${item.getBoundingClientRect().height}px`;
        item.replaceWith(placeholder);
        this.active.set(item, placeholder);
        // Map insertion order is original admission order, including cards which
        // reach the top together after a large wheel/scrollbar movement.
        this.stack.append(item);
      }
    }
    this.dock.hidden = this.stack.childElementCount === 0;
  }
}
