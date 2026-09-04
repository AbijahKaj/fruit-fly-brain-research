/**
 * Rate-model runtime over flat typed arrays.
 *
 *   tau_i dx_i/dt = -x_i + wScale * sum_j W_ij r_j + ext_i + bias_i
 *   r_i = clamp(x_i, 0, rMax)
 *
 * `bias` is the resting drive (flyvis calls it the resting potential). It is
 * set homeostatically for the optic lobe (see net.worker.ts) until a fitted
 * per-type value replaces it.
 *
 * Everything is a Float32Array / Int32Array and `step` is one loop, so the
 * same routine can move to a Web Worker, WASM, or a WebGPU kernel unchanged.
 */
import type { CSR } from "./graph";

export interface RateNetParams {
  /** Multiplies signed synapse counts into drive. */
  wScale: number;
  /** Rate ceiling. */
  rMax: number;
}

export class RateNet {
  readonly n: number;
  readonly x: Float32Array;
  readonly r: Float32Array;
  readonly ext: Float32Array;
  readonly bias: Float32Array;
  readonly tau: Float32Array;
  readonly params: RateNetParams;

  constructor(
    readonly csr: CSR,
    tau: Float32Array,
    params: Partial<RateNetParams> = {},
  ) {
    this.n = csr.n;
    this.x = new Float32Array(this.n);
    this.r = new Float32Array(this.n);
    this.ext = new Float32Array(this.n);
    this.bias = new Float32Array(this.n);
    this.tau = tau;
    this.params = { wScale: 0.005, rMax: 1, ...params };
  }

  reset(): void {
    this.x.fill(0);
    this.r.fill(0);
    this.ext.fill(0);
  }

  /** One Euler step of size dt (seconds). */
  step(dt: number): void {
    const { indptr, pre, w } = this.csr;
    const { x, r, ext, bias, tau } = this;
    const { wScale, rMax } = this.params;
    const n = this.n;
    for (let i = 0; i < n; i++) {
      let drive = 0;
      const end = indptr[i + 1]!;
      for (let k = indptr[i]!; k < end; k++) drive += w[k]! * r[pre[k]!]!;
      const target = wScale * drive + ext[i]! + bias[i]!;
      const a = Math.min(1, dt / tau[i]!);
      x[i] = x[i]! + a * (target - x[i]!);
    }
    for (let i = 0; i < n; i++) {
      const v = x[i]!;
      r[i] = v <= 0 ? 0 : v >= rMax ? rMax : v;
    }
  }

  meanRate(idx: Int32Array): number {
    if (idx.length === 0) return 0;
    let s = 0;
    for (let k = 0; k < idx.length; k++) s += this.r[idx[k]!]!;
    return s / idx.length;
  }
}
