/**
 * Overlay: what each eye sees (equirectangular dot map) and a stats block.
 */
import type { Ommatidia } from "../eye/ommatidia";

/** luminance: what the eye sees. brain: T4a - T4b per column. */
export type EyeView = "luminance" | "brain";

export class Hud {
  view: EyeView = "luminance";
  private readonly ctx: CanvasRenderingContext2D;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly stats: HTMLElement,
  ) {
    this.ctx = canvas.getContext("2d")!;
  }

  drawEyes(
    ommL: Ommatidia,
    lumL: Float32Array,
    brainL: Float32Array | undefined,
    ommR: Ommatidia,
    lumR: Float32Array,
    brainR: Float32Array | undefined,
  ): void {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#9aa3ad";
    ctx.font = "11px ui-monospace, monospace";
    const view = this.view === "brain" && !brainL ? "luminance" : this.view;
    const label = view === "brain" ? "T4a − T4b" : view;
    ctx.fillText(`eyes (${label})  L ${ommL.count}  R ${ommR.count}`, 6, 12);
    this.drawEye(ommL, lumL, view === "brain" ? brainL : undefined);
    this.drawEye(ommR, lumR, view === "brain" ? brainR : undefined);
  }

  private drawEye(omm: Ommatidia, lum: Float32Array, brain: Float32Array | undefined): void {
    const { ctx, canvas } = this;
    const top = 18;
    const h = canvas.height - top - 4;
    const w = canvas.width - 8;
    // azimuth -170..170 deg across the width, elevation 75..-60 down the height.
    const azSpan = (340 * Math.PI) / 180;
    const elTop = (75 * Math.PI) / 180;
    const elSpan = (135 * Math.PI) / 180;
    for (let i = 0; i < omm.count; i++) {
      const x = 4 + ((omm.az[i]! + azSpan / 2) / azSpan) * w;
      const y = top + ((elTop - omm.el[i]!) / elSpan) * h;
      if (!brain) {
        const v = Math.sqrt(Math.max(0, Math.min(1, lum[i]!)));
        const g = Math.round(40 + v * 215);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
      } else {
        const v = Math.max(-1, Math.min(1, brain[i]! * 3));
        ctx.fillStyle = v >= 0 ? `rgba(255,90,70,${0.15 + v * 0.85})` : `rgba(70,140,255,${0.15 - v * 0.85})`;
      }
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    }
  }

  /** Heat strip of unit rates, grouped; `groups` are [label, indices] in display order. */
  drawNet(canvas: HTMLCanvasElement, rates: Float32Array, groups: Array<[string, Int32Array]>): void {
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const total = groups.reduce((a, [, idx]) => a + idx.length, 0);
    if (!total) return;
    const top = 14;
    const h = canvas.height - top - 2;
    let x = 2;
    const w = canvas.width - 4;
    ctx.font = "10px ui-monospace, monospace";
    let labelEnd = 0; // labels are drawn only where they do not overlap the previous one
    for (const [label, idx] of groups) {
      const gw = Math.max(2, (idx.length / total) * w);
      // one bar per pixel column: average the units that fall in it
      const bins = Math.max(1, Math.floor(gw));
      const per = idx.length / bins;
      for (let b = 0; b < bins; b++) {
        const k0 = Math.floor(b * per);
        const k1 = Math.max(k0 + 1, Math.floor((b + 1) * per));
        let sum = 0;
        for (let k = k0; k < k1 && k < idx.length; k++) sum += rates[idx[k]!]!;
        const v = Math.max(0, Math.min(1, sum / (k1 - k0)));
        const g = Math.round(v * 255);
        ctx.fillStyle = `rgb(${g},${Math.round(g * 0.55)},${Math.round(40 + (1 - v) * 40)})`;
        ctx.fillRect(x + b * (gw / bins), top, gw / bins + 0.5, h);
      }
      if (x + 1 >= labelEnd) {
        ctx.fillStyle = "#9aa3ad";
        ctx.fillText(label, x + 1, 10);
        labelEnd = x + 1 + ctx.measureText(label).width + 6;
      }
      ctx.strokeStyle = "#2a3038";
      ctx.strokeRect(x, top, gw, h);
      x += gw;
    }
  }

  private statCells: HTMLSpanElement[] = [];

  /** Two-column key/value grid; rows are rebuilt only when their number changes. */
  setStats(lines: Array<[string, string | number]>): void {
    if (this.statCells.length !== lines.length * 2) {
      this.stats.replaceChildren();
      this.statCells = [];
      for (let i = 0; i < lines.length; i++) {
        const k = document.createElement("span");
        k.className = "k";
        const v = document.createElement("span");
        v.className = "v";
        this.stats.append(k, v);
        this.statCells.push(k, v);
      }
    }
    for (let i = 0; i < lines.length; i++) {
      const [k, v] = lines[i]!;
      const kc = this.statCells[i * 2]!;
      const vc = this.statCells[i * 2 + 1]!;
      if (kc.textContent !== k) kc.textContent = k;
      const text = typeof v === "number" ? fmt(v) : v;
      if (vc.textContent !== text) vc.textContent = text;
    }
  }
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(3);
}
