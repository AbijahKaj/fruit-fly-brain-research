/// <reference lib="webworker" />
/**
 * Web Worker hosting a RateNet so the 40k-unit optic lobe integrates off
 * the render thread. Protocol in remote-net.ts.
 */
import { buildCSRWeighted } from "./graph";
import { RateNet } from "./rate-net";

let net: RateNet | undefined;
let netDt = 0.002;
let bias0: Float32Array | undefined;

interface InitMsg {
  type: "init";
  n: number;
  m: number;
  pre: Int32Array;
  post: Int32Array;
  /** Signed, scaled per-edge weight. */
  w: Float32Array;
  tau: Float32Array;
  /** Initial per-unit bias (resting drive). */
  bias: Float32Array;
  wScale: number;
  rMax: number;
  netDt: number;
}
interface StepMsg {
  type: "step";
  ext: Float32Array;
  dt: number;
}
interface SettleMsg {
  type: "settle";
  ext: Float32Array;
  seconds: number;
}
interface ParamsMsg {
  type: "params";
  wScale?: number;
  netDt?: number;
}
interface ResetMsg {
  type: "reset";
}
/**
 * Homeostatic bias: repeatedly settle with the given ext, then nudge the bias
 * of the listed units toward a target resting rate. Puts every unit into a
 * regime where inputs can move it both ways, without per-type hand tuning.
 */
interface HomeostatMsg {
  type: "homeostat";
  ext: Float32Array;
  units: Int32Array;
  /** Target membrane value per listed unit (same length as units). */
  targets: Float32Array;
  rounds: number;
  settleSeconds: number;
  eta: number;
}
export type ToWorker = InitMsg | StepMsg | SettleMsg | ParamsMsg | ResetMsg | HomeostatMsg;
export type FromWorker =
  | { type: "ready" }
  | { type: "rates"; r: Float32Array; ext: Float32Array; steps: number; ms: number }
  | { type: "settled"; r: Float32Array; ext: Float32Array; ms: number }
  | { type: "homeostat"; r: Float32Array; ext: Float32Array; ms: number; meanErr: number; biasMean: number };

self.onmessage = (ev: MessageEvent<ToWorker>): void => {
  const msg = ev.data;
  switch (msg.type) {
    case "init": {
      const csr = buildCSRWeighted(msg.n, msg.m, msg.pre, msg.post, msg.w);
      net = new RateNet(csr, msg.tau, { wScale: msg.wScale, rMax: msg.rMax });
      net.bias.set(msg.bias);
      bias0 = msg.bias;
      netDt = msg.netDt;
      post({ type: "ready" });
      break;
    }
    case "params":
      if (net && msg.wScale !== undefined) net.params.wScale = msg.wScale;
      if (msg.netDt !== undefined) netDt = msg.netDt;
      break;
    case "reset":
      net?.reset();
      if (net && bias0) net.bias.set(bias0);
      break;
    case "homeostat": {
      if (!net) return;
      const t0 = performance.now();
      net.ext.set(msg.ext);
      const steps = Math.round(msg.settleSeconds / netDt);
      // At steady state x = drive + ext + bias, so correcting the bias by the
      // membrane error (not the clamped rate) moves the unit straight to the
      // target regardless of fan-in; the rounds absorb recurrent effects.
      let meanErr = 0;
      for (let round = 0; round < msg.rounds; round++) {
        for (let k = 0; k < steps; k++) net.step(netDt);
        meanErr = 0;
        for (let k = 0; k < msg.units.length; k++) {
          const i = msg.units[k]!;
          const err = msg.targets[k]! - net.x[i]!;
          net.bias[i] = net.bias[i]! + msg.eta * err;
          meanErr += Math.abs(err);
        }
        meanErr /= Math.max(1, msg.units.length);
      }
      for (let k = 0; k < steps; k++) net.step(netDt);
      bias0 = net.bias.slice();
      let biasMean = 0;
      for (let k = 0; k < msg.units.length; k++) biasMean += net.bias[msg.units[k]!]!;
      biasMean /= Math.max(1, msg.units.length);
      const r = net.r.slice();
      post({ type: "homeostat", r, ext: msg.ext, ms: performance.now() - t0, meanErr, biasMean }, [r.buffer, msg.ext.buffer]);
      break;
    }
    case "step": {
      if (!net) return;
      const t0 = performance.now();
      net.ext.set(msg.ext);
      let remaining = msg.dt;
      let steps = 0;
      while (remaining > 1e-6) {
        const h = Math.min(netDt, remaining);
        net.step(h);
        remaining -= h;
        steps++;
      }
      const r = net.r.slice();
      post({ type: "rates", r, ext: msg.ext, steps, ms: performance.now() - t0 }, [r.buffer, msg.ext.buffer]);
      break;
    }
    case "settle": {
      if (!net) return;
      const t0 = performance.now();
      net.ext.set(msg.ext);
      const steps = Math.round(msg.seconds / netDt);
      for (let k = 0; k < steps; k++) net.step(netDt);
      const r = net.r.slice();
      post({ type: "settled", r, ext: msg.ext, ms: performance.now() - t0 }, [r.buffer, msg.ext.buffer]);
      break;
    }
  }
};

function post(m: FromWorker, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(m, transfer);
}
