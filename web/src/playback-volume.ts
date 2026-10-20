export const PLAYBACK_VOLUME_KEY = "hermes-realtime.playback-volume.v1";

export class PlaybackVolumeControls {
  private volume = 1;
  private muted = false;
  private attenuation = 1;
  private saved = true;

  constructor(
    private readonly slider: HTMLInputElement,
    private readonly mute: HTMLButtonElement,
    private readonly status: HTMLOutputElement,
    private readonly storage: () => Pick<Storage, "getItem" | "setItem">,
    private readonly write: (volume: number) => void,
  ) {
    try {
      const preference = JSON.parse(this.storage().getItem(PLAYBACK_VOLUME_KEY) ?? "null") as Record<string, unknown>;
      if (
        Object.keys(preference).sort().join(",") === "muted,volume" &&
        typeof preference.volume === "number" &&
        preference.volume >= 0 && preference.volume <= 1 &&
        typeof preference.muted === "boolean"
      ) {
        this.volume = preference.volume;
        this.muted = preference.muted;
      }
    } catch {
      // Storage may be unavailable; a malformed preference also retains unity gain.
    }
    this.slider.addEventListener("input", () => {
      this.volume = this.slider.valueAsNumber / 100;
      this.persist();
    });
    this.mute.addEventListener("click", () => {
      this.muted = !this.muted;
      this.persist();
    });
    this.render();
  }

  setAttenuation(factor: number): void {
    this.attenuation = factor;
    this.apply();
  }

  private persist(): void {
    try {
      this.storage().setItem(PLAYBACK_VOLUME_KEY, JSON.stringify({ volume: this.volume, muted: this.muted }));
      this.saved = true;
    } catch {
      this.saved = false;
    }
    this.render();
  }

  private apply(): void {
    this.write(this.muted ? 0 : this.volume * this.attenuation);
  }

  private render(): void {
    const percent = Math.round(this.volume * 100);
    this.slider.value = String(percent);
    this.slider.setAttribute("aria-valuetext", `${percent} percent`);
    this.mute.setAttribute("aria-label", this.muted ? "Unmute speaker" : "Mute speaker");
    this.mute.setAttribute("aria-pressed", String(this.muted));
    this.mute.textContent = this.muted ? "Unmute" : "Mute";
    this.status.textContent = `${this.muted ? "Muted · " : ""}${percent}%${this.saved ? "" : " · not saved"}`;
    this.apply();
  }
}
