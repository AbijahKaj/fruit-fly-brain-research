/**
 * Overlay: what each eye sees (equirectangular dot map) and a stats block.
 */
import type { Ommatidia } from "../eye/ommatidia";

export type EyeView = "luminance" | "highpass";

export class Hud {
  view: EyeView = "luminance";
  private readonly ctx: CanvasRenderingContext2D;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly stats: HTMLElement,
    private readonly ommL: Ommatidia,
    private readonly ommR: Ommatidia,
  ) {
    this.ctx = canvas.getContext("2d")!;
  }

  drawEyes(logL: Float32Array, hpL: Float32Array, logR: Float32Array, hpR: Float32Array): void {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#9aa3ad";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`eyes (${this.view})  L ${this.ommL.count}  R ${this.ommR.count}`, 6, 12);
    this.drawEye(this.ommL, logL, hpL);
    this.drawEye(this.ommR, logR, hpR);
  }

  private drawEye(omm: Ommatidia, log: Float32Array, hp: Float32Array): void {
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
      if (this.view === "luminance") {
        // log(0.02)..log(1.02) -> 0..1
        const v = Math.max(0, Math.min(1, (log[i]! + 3.9) / 3.9));
        const g = Math.round(40 + v * 215);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
      } else {
        const v = Math.max(-1, Math.min(1, hp[i]! * 4));
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
    for (const [label, idx] of groups) {
      const gw = (idx.length / total) * w;
      const cell = gw / idx.length;
      for (let k = 0; k < idx.length; k++) {
        const v = Math.max(0, Math.min(1, rates[idx[k]!]!));
        const g = Math.round(v * 255);
        ctx.fillStyle = `rgb(${g},${Math.round(g * 0.55)},${Math.round(40 + (1 - v) * 40)})`;
        ctx.fillRect(x + k * cell, top, Math.max(1, cell), h);
      }
      ctx.fillStyle = "#9aa3ad";
      ctx.fillText(label, x + 1, 10);
      ctx.strokeStyle = "#2a3038";
      ctx.strokeRect(x, top, gw, h);
      x += gw;
    }
  }

  setStats(lines: Array<[string, string | number]>): void {
    this.stats.textContent = lines
      .map(([k, v]) => `${k.padEnd(12)} ${typeof v === "number" ? fmt(v) : v}`)
      .join("\n");
  }
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(3);
}
