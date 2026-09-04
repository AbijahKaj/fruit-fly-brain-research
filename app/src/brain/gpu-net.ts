/**
 * The rate model on WebGPU. Same equation as rate-net.ts:
 *
 *   tau_i dx_i/dt = -x_i + wScale * sum_j W_ij r_j + ext_i + bias_i
 *   r_i = clamp(x_i, 0, rMax)
 *
 * Two compute passes per Euler step, both over CSR by postsynaptic unit:
 *
 *   drive      one thread per chunk of <= CHUNK in-edges, partial sums.
 *              Splitting rows keeps pooling cells with tens of thousands of
 *              inputs from stalling a whole workgroup.
 *   integrate  one thread per unit: sum its chunks, update x and r.
 *
 * All state lives on the GPU. Each frame's substeps go in one command buffer
 * and are submitted without waiting: the queue keeps them ordered, so the
 * network runs in real time and only the *view* of r lags. Readbacks go
 * through a small pool of staging buffers; a frame whose pool is empty still
 * integrates, it just does not copy r back. settle/homeostat take the device
 * exclusively and wait for their result.
 */
import type { Graph } from "./graph";
import { buildCSRWeighted } from "./graph";
import type { NetBackend } from "./net-backend";

const CHUNK = 64;
const WG = 256;
/** Staging buffers in flight for frame readbacks. */
const SLOTS = 3;
/** writeBuffer's typing rejects ArrayBufferLike-backed views; ours never share memory. */
const src = (a: ArrayBufferView): BufferSource => a as unknown as BufferSource;

const PARAMS = `struct Params { n: u32, nChunks: u32, dt: f32, wScale: f32, rMax: f32, pad0: f32, pad1: f32, pad2: f32 }`;

const WGSL_DRIVE = /* wgsl */ `
${PARAMS}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> chunkUnit: array<u32>;
@group(0) @binding(2) var<storage, read> unitChunkPtr: array<u32>;
@group(0) @binding(3) var<storage, read> indptr: array<u32>;
@group(0) @binding(4) var<storage, read> idx: array<u32>;
@group(0) @binding(5) var<storage, read> wt: array<f32>;
@group(0) @binding(6) var<storage, read> rIn: array<f32>;
@group(0) @binding(7) var<storage, read_write> partial: array<f32>;

@compute @workgroup_size(${WG})
fn drive(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= P.nChunks) { return; }
  let u = chunkUnit[c];
  let k = c - unitChunkPtr[u];
  let start = indptr[u] + k * ${CHUNK}u;
  let end = min(start + ${CHUNK}u, indptr[u + 1u]);
  var s = 0.0;
  for (var e = start; e < end; e++) { s += wt[e] * rIn[idx[e]]; }
  partial[c] = s;
}
`;

const WGSL_INTEGRATE = /* wgsl */ `
${PARAMS}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> unitChunkPtr: array<u32>;
@group(0) @binding(2) var<storage, read> partial: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> r: array<f32>;
@group(0) @binding(5) var<storage, read> ext: array<f32>;
@group(0) @binding(6) var<storage, read> bias: array<f32>;
@group(0) @binding(7) var<storage, read> tau: array<f32>;

@compute @workgroup_size(${WG})
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  var d = 0.0;
  for (var c = unitChunkPtr[i]; c < unitChunkPtr[i + 1u]; c++) { d += partial[c]; }
  let xi = x[i] + (P.dt / tau[i]) * (-x[i] + P.wScale * d + ext[i] + bias[i]);
  x[i] = xi;
  r[i] = clamp(xi, 0.0, P.rMax);
}
`;

interface Slot {
  r: GPUBuffer;
  x: GPUBuffer;
  /** Two u64 timestamps, when the device supports them. */
  q: GPUBuffer | undefined;
  free: boolean;
}

export class GpuNet implements NetBackend {
  readonly kind = "gpu" as const;
  readonly n: number;
  readonly ext: Float32Array;
  r: Float32Array;
  ready = false;
  /** True while settle/homeostat own the device; frame steps are skipped. */
  busy = false;
  /** GPU time of the last frame's substeps (timestamp query), or wall time to readback without it. */
  lastMs = 0;
  lastSteps = 0;
  simTime = 0;
  /** Frames integrated without a free staging buffer (readback skipped). */
  droppedReadbacks = 0;
  deviceName = "";
  hasTimestamps = false;
  readonly nChunks: number;
  private pendingDt = 0;
  private wScale: number;
  private netDt: number;
  private readonly rMax: number;
  private readonly bias0: Float32Array;
  private readonly bias: Float32Array;
  private readonly readyPromise: Promise<void>;
  private device!: GPUDevice;
  private buf!: Record<"uniform" | "chunkUnit" | "unitChunkPtr" | "indptr" | "idx" | "wt" | "partial" | "x" | "r" | "ext" | "bias" | "tau" | "query", GPUBuffer>;
  private querySet: GPUQuerySet | undefined;
  private drivePipe!: GPUComputePipeline;
  private integPipe!: GPUComputePipeline;
  private driveBind!: GPUBindGroup;
  private integBind!: GPUBindGroup;
  private readonly uniformData = new ArrayBuffer(32);
  private slots: Slot[] = [];
  private exclusive!: Slot;
  private rSpare: Float32Array;
  private seq = 0;
  private lastApplied = 0;
  private readonly csr: { indptr: Uint32Array; idx: Uint32Array; wt: Float32Array; chunkUnit: Uint32Array; unitChunkPtr: Uint32Array };

  constructor(
    g: Graph,
    w: Float32Array,
    private readonly tau: Float32Array,
    bias: Float32Array,
    wScale: number,
    netDt: number,
    rMax = 1,
    readonly maxPendingDt = 0.1,
  ) {
    this.n = g.n;
    this.ext = new Float32Array(g.n);
    this.r = new Float32Array(g.n);
    this.rSpare = new Float32Array(g.n);
    this.wScale = wScale;
    this.netDt = netDt;
    this.rMax = rMax;
    this.bias0 = bias.slice();
    this.bias = bias.slice();

    const csr = buildCSRWeighted(g.n, g.m, g.pre, g.post, w);
    const indptr = Uint32Array.from(csr.indptr);
    const unitChunkPtr = new Uint32Array(g.n + 1);
    for (let i = 0; i < g.n; i++) {
      const deg = indptr[i + 1]! - indptr[i]!;
      unitChunkPtr[i + 1] = unitChunkPtr[i]! + Math.max(1, Math.ceil(deg / CHUNK));
    }
    this.nChunks = unitChunkPtr[g.n]!;
    const chunkUnit = new Uint32Array(this.nChunks);
    for (let i = 0; i < g.n; i++) for (let c = unitChunkPtr[i]!; c < unitChunkPtr[i + 1]!; c++) chunkUnit[c] = i;
    this.csr = { indptr, idx: Uint32Array.from(csr.pre), wt: csr.w, chunkUnit, unitChunkPtr };

    this.readyPromise = this.init();
  }

  private async init(): Promise<void> {
    if (!navigator.gpu) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    this.hasTimestamps = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({ requiredFeatures: this.hasTimestamps ? ["timestamp-query"] : [] });
    this.device = device;
    const info = adapter.info as GPUAdapterInfo | undefined;
    this.deviceName = info ? `${info.vendor} ${info.architecture || info.device || ""}`.trim() : "gpu";
    device.lost.then((l) => console.error("WebGPU device lost:", l.message)).catch(() => undefined);

    const mk = (data: ArrayBufferView | number, usage: GPUBufferUsageFlags): GPUBuffer => {
      const size = typeof data === "number" ? data : data.byteLength;
      const b = device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage });
      if (typeof data !== "number") device.queue.writeBuffer(b, 0, src(data));
      return b;
    };
    const S = GPUBufferUsage.STORAGE;
    const D = GPUBufferUsage.COPY_DST;
    const C = GPUBufferUsage.COPY_SRC;
    const n4 = this.n * 4;
    this.buf = {
      uniform: mk(32, GPUBufferUsage.UNIFORM | D),
      chunkUnit: mk(this.csr.chunkUnit, S | D),
      unitChunkPtr: mk(this.csr.unitChunkPtr, S | D),
      indptr: mk(this.csr.indptr, S | D),
      idx: mk(this.csr.idx, S | D),
      wt: mk(this.csr.wt, S | D),
      partial: mk(this.nChunks * 4, S),
      x: mk(n4, S | D | C),
      r: mk(n4, S | D | C),
      ext: mk(n4, S | D),
      bias: mk(this.bias, S | D),
      tau: mk(this.tau, S | D),
      query: mk(16, GPUBufferUsage.QUERY_RESOLVE | C),
    };
    const slot = (): Slot => ({
      r: mk(n4, GPUBufferUsage.MAP_READ | D),
      x: mk(n4, GPUBufferUsage.MAP_READ | D),
      q: this.hasTimestamps ? mk(16, GPUBufferUsage.MAP_READ | D) : undefined,
      free: true,
    });
    for (let k = 0; k < SLOTS; k++) this.slots.push(slot());
    this.exclusive = slot();
    if (this.hasTimestamps) this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });

    const driveMod = device.createShaderModule({ code: WGSL_DRIVE });
    const integMod = device.createShaderModule({ code: WGSL_INTEGRATE });
    this.drivePipe = device.createComputePipeline({ layout: "auto", compute: { module: driveMod, entryPoint: "drive" } });
    this.integPipe = device.createComputePipeline({ layout: "auto", compute: { module: integMod, entryPoint: "integrate" } });
    const b = this.buf;
    const entries = (list: GPUBuffer[]): GPUBindGroupEntry[] => list.map((buffer, binding) => ({ binding, resource: { buffer } }));
    this.driveBind = device.createBindGroup({
      layout: this.drivePipe.getBindGroupLayout(0),
      entries: entries([b.uniform, b.chunkUnit, b.unitChunkPtr, b.indptr, b.idx, b.wt, b.r, b.partial]),
    });
    this.integBind = device.createBindGroup({
      layout: this.integPipe.getBindGroupLayout(0),
      entries: entries([b.uniform, b.unitChunkPtr, b.partial, b.x, b.r, b.ext, b.bias, b.tau]),
    });
    this.ready = true;
  }

  whenReady(): Promise<void> {
    return this.readyPromise;
  }

  setParams(p: { wScale?: number; netDt?: number }): void {
    if (p.wScale !== undefined) this.wScale = p.wScale;
    if (p.netDt !== undefined) this.netDt = p.netDt;
  }

  reset(): void {
    this.r.fill(0);
    this.pendingDt = 0;
    this.bias.set(this.bias0);
    if (!this.ready) return;
    const zeros = new Float32Array(this.n);
    this.device.queue.writeBuffer(this.buf.x, 0, zeros);
    this.device.queue.writeBuffer(this.buf.r, 0, zeros);
    this.device.queue.writeBuffer(this.buf.bias, 0, src(this.bias));
  }

  /** Integrate the elapsed time now; never waits. Returns false only while settle/homeostat run. */
  step(dt: number): boolean {
    this.pendingDt = Math.min(this.maxPendingDt, this.pendingDt + dt);
    if (!this.ready || this.busy) return false;
    const total = this.pendingDt;
    this.pendingDt = 0;
    this.simTime += total;
    const steps = Math.max(1, Math.ceil(total / this.netDt - 1e-6));
    const slot = this.slots.find((s) => s.free);
    const seq = ++this.seq;
    const t0 = performance.now();
    this.submit(steps, total / steps, slot, false);
    this.lastSteps = steps;
    if (!slot) {
      this.droppedReadbacks++;
      return true;
    }
    slot.free = false;
    this.readback(slot, false)
      .then(({ gpuMs }) => {
        slot.free = true;
        if (seq < this.lastApplied) return; // an older frame resolving late
        this.lastApplied = seq;
        this.lastMs = gpuMs ?? performance.now() - t0;
      })
      .catch(() => undefined); // dispose() aborts in-flight maps
    return true;
  }

  settle(seconds: number): Promise<Float32Array> {
    return this.whenIdle().then(async () => {
      this.busy = true;
      const steps = Math.round(seconds / this.netDt);
      this.simTime += seconds;
      const t0 = performance.now();
      this.submit(steps, this.netDt, this.exclusive, false);
      const { gpuMs } = await this.readback(this.exclusive, false);
      this.lastMs = gpuMs ?? performance.now() - t0;
      this.lastSteps = steps;
      this.busy = false;
      return this.r;
    });
  }

  async homeostat(
    units: Int32Array,
    targets: Float32Array,
    rounds: number,
    settleSeconds: number,
    eta: number,
  ): Promise<{ meanErr: number; biasMean: number; ms: number }> {
    await this.whenIdle();
    this.busy = true;
    const t0 = performance.now();
    const steps = Math.round(settleSeconds / this.netDt);
    this.simTime += settleSeconds * (rounds + 1);
    let meanErr = 0;
    const x = new Float32Array(this.n);
    for (let round = 0; round < rounds; round++) {
      this.submit(steps, this.netDt, this.exclusive, true);
      await this.readback(this.exclusive, true, x);
      meanErr = 0;
      for (let k = 0; k < units.length; k++) {
        const i = units[k]!;
        const err = targets[k]! - x[i]!;
        this.bias[i] = this.bias[i]! + eta * err;
        meanErr += Math.abs(err);
      }
      meanErr /= Math.max(1, units.length);
      this.device.queue.writeBuffer(this.buf.bias, 0, src(this.bias));
    }
    this.submit(steps, this.netDt, this.exclusive, false);
    await this.readback(this.exclusive, false);
    this.bias0.set(this.bias);
    let biasMean = 0;
    for (let k = 0; k < units.length; k++) biasMean += this.bias[units[k]!]!;
    biasMean /= Math.max(1, units.length);
    const ms = performance.now() - t0;
    this.lastMs = ms;
    this.busy = false;
    return { meanErr, biasMean, ms };
  }

  meanRate(idx: Int32Array): number {
    if (idx.length === 0) return 0;
    let s = 0;
    for (let k = 0; k < idx.length; k++) s += this.r[idx[k]!]!;
    return s / idx.length;
  }

  dispose(): void {
    this.ready = false;
    this.device?.destroy();
  }

  private whenIdle(): Promise<void> {
    return this.readyPromise.then(
      () =>
        new Promise<void>((resolve) => {
          const wait = (): void => {
            if (this.busy) setTimeout(wait, 5);
            else resolve();
          };
          wait();
        }),
    );
  }

  /** Upload ext and params, queue `steps` Euler steps of size h, and copies into `slot` if given. */
  private submit(steps: number, h: number, slot: Slot | undefined, copyX: boolean): void {
    const dev = this.device;
    const q = dev.queue;
    q.writeBuffer(this.buf.ext, 0, src(this.ext));
    const u = new DataView(this.uniformData);
    u.setUint32(0, this.n, true);
    u.setUint32(4, this.nChunks, true);
    u.setFloat32(8, h, true);
    u.setFloat32(12, this.wScale, true);
    u.setFloat32(16, this.rMax, true);
    q.writeBuffer(this.buf.uniform, 0, this.uniformData);

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass(
      this.querySet && slot?.q
        ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
        : {},
    );
    const driveGroups = Math.ceil(this.nChunks / WG);
    const unitGroups = Math.ceil(this.n / WG);
    for (let k = 0; k < steps; k++) {
      pass.setPipeline(this.drivePipe);
      pass.setBindGroup(0, this.driveBind);
      pass.dispatchWorkgroups(driveGroups);
      pass.setPipeline(this.integPipe);
      pass.setBindGroup(0, this.integBind);
      pass.dispatchWorkgroups(unitGroups);
    }
    pass.end();
    if (slot) {
      enc.copyBufferToBuffer(this.buf.r, 0, slot.r, 0, this.n * 4);
      if (copyX) enc.copyBufferToBuffer(this.buf.x, 0, slot.x, 0, this.n * 4);
      if (this.querySet && slot.q) {
        enc.resolveQuerySet(this.querySet, 0, 2, this.buf.query, 0);
        enc.copyBufferToBuffer(this.buf.query, 0, slot.q, 0, 16);
      }
    }
    q.submit([enc.finish()]);
  }

  /** Map the slot's staging buffers, publish r (and x), free the slot. */
  private async readback(slot: Slot, readX: boolean, xOut?: Float32Array): Promise<{ gpuMs: number | undefined }> {
    const maps = [slot.r.mapAsync(GPUMapMode.READ)];
    if (readX) maps.push(slot.x.mapAsync(GPUMapMode.READ));
    if (slot.q) maps.push(slot.q.mapAsync(GPUMapMode.READ));
    await Promise.all(maps);
    const next = this.rSpare;
    next.set(new Float32Array(slot.r.getMappedRange()));
    slot.r.unmap();
    this.rSpare = this.r;
    this.r = next;
    if (readX) {
      xOut?.set(new Float32Array(slot.x.getMappedRange()));
      slot.x.unmap();
    }
    let gpuMs: number | undefined;
    if (slot.q) {
      const ts = new BigUint64Array(slot.q.getMappedRange());
      gpuMs = Number(ts[1]! - ts[0]!) / 1e6;
      slot.q.unmap();
    }
    return { gpuMs };
  }
}
