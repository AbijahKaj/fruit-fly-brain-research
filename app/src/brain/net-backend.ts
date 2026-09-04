/**
 * What a brain needs from the network runtime, whichever device integrates it.
 *
 *   worker  RateNet on a CPU thread (net.worker.ts via remote-net.ts)
 *   gpu     the same equation as WGSL compute shaders (gpu-net.ts)
 *
 * Both are asynchronous: `ext` is filled by the caller, `step(dt)` hands it
 * over when the device is idle, and `r` is the latest rates that came back.
 */
import type { Graph } from "./graph";
import { GpuNet } from "./gpu-net";
import { RemoteNet } from "./remote-net";

export type NetBackendKind = "gpu" | "worker";

export interface NetBackend {
  readonly kind: NetBackendKind;
  readonly n: number;
  readonly ext: Float32Array;
  r: Float32Array;
  ready: boolean;
  busy: boolean;
  /** Device-side integration time for the last step, ms. */
  lastMs: number;
  lastSteps: number;
  /** Sim seconds integrated so far. */
  simTime: number;
  whenReady(): Promise<void>;
  setParams(p: { wScale?: number; netDt?: number }): void;
  reset(): void;
  /** Hand ext to the device if idle; true if a step was issued. */
  step(dt: number): boolean;
  settle(seconds: number): Promise<Float32Array>;
  homeostat(
    units: Int32Array,
    targets: Float32Array,
    rounds: number,
    settleSeconds: number,
    eta: number,
  ): Promise<{ meanErr: number; biasMean: number; ms: number }>;
  meanRate(idx: Int32Array): number;
  dispose(): void;
}

export function gpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && navigator.gpu !== undefined;
}

export function createNet(
  kind: NetBackendKind,
  g: Graph,
  w: Float32Array,
  tau: Float32Array,
  bias: Float32Array,
  wScale: number,
  netDt: number,
  rMax: number,
): NetBackend {
  if (kind === "gpu") return new GpuNet(g, w, tau, bias, wScale, netDt, rMax);
  return new RemoteNet(g, w, tau, bias, wScale, netDt, rMax);
}
