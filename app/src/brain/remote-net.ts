/**
 * Main-thread handle on a RateNet living in net.worker.ts.
 *
 * `ext` is filled by the caller, `step(dt)` ships it to the worker when the
 * worker is idle (otherwise the elapsed time accumulates and goes with the
 * next message, capped). `r` is the most recent rates the worker returned,
 * so readers see one worker-round-trip of latency.
 */
import type { Graph } from "./graph";
import type { FromWorker, ToWorker } from "./net.worker";
import type { NetBackend } from "./net-backend";

export class RemoteNet implements NetBackend {
  readonly kind = "worker" as const;
  readonly n: number;
  readonly ext: Float32Array;
  r: Float32Array;
  ready = false;
  busy = false;
  /** Worker-side integration time for the last step message, ms. */
  lastMs = 0;
  lastSteps = 0;
  /** Sim seconds the worker has integrated so far. */
  simTime = 0;
  private pendingDt = 0;
  private readonly worker: Worker;
  private readonly readyPromise: Promise<void>;
  private settleResolve: ((r: Float32Array) => void) | undefined;
  private homeoResolve: ((info: { meanErr: number; biasMean: number; ms: number }) => void) | undefined;
  private spare: Float32Array | undefined;

  constructor(
    g: Graph,
    w: Float32Array,
    tau: Float32Array,
    bias: Float32Array,
    wScale: number,
    netDt: number,
    rMax = 1,
    readonly maxPendingDt = 0.1,
  ) {
    this.n = g.n;
    this.ext = new Float32Array(g.n);
    this.r = new Float32Array(g.n);
    this.worker = new Worker(new URL("./net.worker.ts", import.meta.url), { type: "module" });
    this.readyPromise = new Promise((resolve) => {
      this.worker.onmessage = (ev: MessageEvent<FromWorker>): void => {
        const m = ev.data;
        if (m.type === "ready") {
          this.ready = true;
          resolve();
        } else if (m.type === "rates") {
          this.r = m.r;
          this.spare = m.ext;
          this.lastMs = m.ms;
          this.lastSteps = m.steps;
          this.busy = false;
        } else if (m.type === "settled") {
          this.r = m.r;
          this.spare = m.ext;
          this.lastMs = m.ms;
          this.busy = false;
          this.settleResolve?.(m.r);
          this.settleResolve = undefined;
        } else if (m.type === "homeostat") {
          this.r = m.r;
          this.spare = m.ext;
          this.lastMs = m.ms;
          this.busy = false;
          this.homeoResolve?.({ meanErr: m.meanErr, biasMean: m.biasMean, ms: m.ms });
          this.homeoResolve = undefined;
        }
      };
    });
    const init: ToWorker = {
      type: "init",
      n: g.n,
      m: g.m,
      pre: g.pre.slice(),
      post: g.post.slice(),
      w: w.slice(),
      tau: tau.slice(),
      bias: bias.slice(),
      wScale,
      rMax,
      netDt,
    };
    this.worker.postMessage(init, [init.pre.buffer, init.post.buffer, init.w.buffer, init.tau.buffer, init.bias.buffer]);
  }

  whenReady(): Promise<void> {
    return this.readyPromise;
  }

  setParams(p: { wScale?: number; netDt?: number }): void {
    this.worker.postMessage({ type: "params", ...p } satisfies ToWorker);
  }

  reset(): void {
    this.worker.postMessage({ type: "reset" } satisfies ToWorker);
    this.r.fill(0);
    this.pendingDt = 0;
  }

  /** Ship ext to the worker if idle; returns true if a message was sent. */
  step(dt: number): boolean {
    this.pendingDt = Math.min(this.maxPendingDt, this.pendingDt + dt);
    if (!this.ready || this.busy) return false;
    const buf = this.takeBuffer();
    buf.set(this.ext);
    const msg: ToWorker = { type: "step", ext: buf, dt: this.pendingDt };
    this.simTime += this.pendingDt;
    this.pendingDt = 0;
    this.busy = true;
    this.worker.postMessage(msg, [buf.buffer]);
    return true;
  }

  /** Integrate `seconds` with the current ext and resolve with the rates. */
  settle(seconds: number): Promise<Float32Array> {
    return new Promise((resolve) => {
      const go = (): void => {
        const buf = this.takeBuffer();
        buf.set(this.ext);
        this.busy = true;
        this.settleResolve = resolve;
        this.simTime += seconds;
        this.worker.postMessage({ type: "settle", ext: buf, seconds } satisfies ToWorker, [buf.buffer]);
      };
      if (this.busy) {
        const wait = (): void => {
          if (this.busy) setTimeout(wait, 5);
          else go();
        };
        wait();
      } else go();
    });
  }

  /** Run the homeostatic bias procedure (see net.worker.ts) with the current ext. */
  homeostat(
    units: Int32Array,
    targets: Float32Array,
    rounds: number,
    settleSeconds: number,
    eta: number,
  ): Promise<{ meanErr: number; biasMean: number; ms: number }> {
    return new Promise((resolve) => {
      const go = (): void => {
        const buf = this.takeBuffer();
        buf.set(this.ext);
        this.busy = true;
        this.homeoResolve = resolve;
        this.simTime += settleSeconds * (rounds + 1);
        this.worker.postMessage(
          { type: "homeostat", ext: buf, units, targets, rounds, settleSeconds, eta } satisfies ToWorker,
          [buf.buffer],
        );
      };
      const wait = (): void => {
        if (this.busy) setTimeout(wait, 5);
        else go();
      };
      wait();
    });
  }

  meanRate(idx: Int32Array): number {
    if (idx.length === 0) return 0;
    let s = 0;
    for (let k = 0; k < idx.length; k++) s += this.r[idx[k]!]!;
    return s / idx.length;
  }

  dispose(): void {
    this.ready = false;
    this.worker.terminate();
  }

  private takeBuffer(): Float32Array {
    const b = this.spare && this.spare.length === this.n ? this.spare : new Float32Array(this.n);
    this.spare = undefined;
    return b;
  }
}
