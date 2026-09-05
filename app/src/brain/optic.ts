/**
 * Layer 3 (brain): the per-column MaleCNS optic lobe (optic-v2).
 *
 *   luminance per column -> one virtual photoreceptor (flyvis R1-R6 params)
 *     -> histaminergic drive on the lamina (L1, L2, L3) of that column
 *     -> the extracted graph: lamina -> medulla -> T4/T5 -> lobula plate
 *        -> posterior slope -> DNg02 -> VNC -> wing MNs, and -> LC4/LPLC2 -> DNp
 *     -> readouts: HS left vs right (optomotor), LC4/LPLC2 per eye (looming)
 *
 * Nothing in the motion pathway is hand-written: direction selectivity comes
 * from the wiring (Mi9 leading, Mi4 trailing) plus per-type time constants,
 * biases and per-pair synapse strengths fitted on the GPU box (train/). Units
 * without fitted parameters (most lobula-plate pooling cells) get a
 * homeostatic resting bias.
 *
 * Open: the HS -> posterior slope -> DNg02 hop is not calibrated in this graph
 * (the central brain runs at the default synapse scale and DNg02 sits at a
 * constant rate), so the steering readout sits at HS. `readout: "dng02"` is
 * kept for that work.
 *
 * The network runs on WebGPU (gpu-net.ts) or in a Web Worker (remote-net.ts);
 * rates seen here lag by one round trip.
 */
import type { Ommatidia } from "../eye/ommatidia";
import { unitsWhere, typeName, sideName, type Graph } from "./graph";
import { applyFlyvis, flyvisRestV, hasFlyvisType, isPooling, type FlyvisParams } from "./flyvis";
import { createNet, gpuAvailable } from "./net-backend";
import type { NetBackend, NetBackendKind } from "./net-backend";
import type { Brain, EyeInput, MotorCommand } from "./types";

const LAMINA = /^L[123]$/;
const LPTC = /^(HS[ENST]|VS|VST1|VST2|VSm|H2|DCH|VCH)$/;
const LOOMING = /^(LC4|LPLC2)$/;
const DNG02 = /^DNg02_/;

export interface OpticParams {
  /** Stimulus scale: adapted luminance times this drives the virtual photoreceptor. */
  stimGain: number;
  /**
   * Light adaptation: each column's luminance is divided by its slow running
   * mean so that steady light maps to 0.5, the grey flyvis was trained around.
   */
  adaptTau: number;
  stimMax: number;
  /** Default synapse scale for edges flyvis does not cover. */
  defaultScale: number;
  /** Scale for edges onto lobula-plate cells (thousands of T4/T5 synapses each). */
  lptcScale: number;
  /** Where the turn is read: DNg02 L/R through the real posterior-slope wiring, or HS L/R directly. */
  readout: "dng02" | "hs";
  /** Tonic drive on lobula plate tangential cells. */
  lptcBias: number;
  /** Tonic flight-state drive on DNg02: its posterior-slope input (PS080) is GABAergic, so without a tonic drive inhibition has nothing to act on. */
  dnBias: number;
  /** Homeostatic resting membrane target for units flyvis does not cover. */
  restTarget: number;
  restRounds: number;
  wScale: number;
  outputGain: number;
  readoutSign: number;
  baseAmp: number;
  maxTurn: number;
  netDt: number;
  /** Rate ceiling. flyvis has none; keep it high so it only guards against blow-up. */
  rMax: number;
  loomGain: number;
  loomThreshold: number;
  loomTopK: number;
  loomTieBias: number;
  loomBrake: number;
  offsetTau: number;
  warmSeconds: number;
  readoutFloor: number;
}

export class OpticBrain implements Brain {
  readonly name: string;
  readonly params: OpticParams = {
    stimGain: 1,
    adaptTau: 1.0,
    stimMax: 1.5,
    defaultScale: 0.02,
    lptcScale: 0.001,
    readout: "hs",
    lptcBias: 0.2,
    dnBias: 0.5,
    restTarget: 0.3,
    restRounds: 40,
    wScale: 1,
    outputGain: 0.25,
    readoutSign: -1,
    baseAmp: 0.5,
    maxTurn: 0.5,
    netDt: 0.004,
    rMax: 5,
    /** Looming: turn away from the eye whose LC4/LPLC2 population fires more. 0 = off. */
    loomGain: 5,
    /** Rate above the calibrated rest that counts as looming (dead zone). */
    loomThreshold: 0.1,
    /** Cells per eye that carry the looming readout (the best-placed ones; matches train_loom.py). */
    loomTopK: 5,
    /**
     * The readout offset tracks the readout with this time constant (s): static scene asymmetries
     * and slow drifts of the pooled tonic drive are cancelled, optomotor transients pass.
     */
    offsetTau: 1e9,
    /** Live warm-up in the scene, output off, before the readout offsets are taken (s). Photoreceptor adaptation (tau 1 s) and the pooling cells need about 2 s. */
    warmSeconds: 2.5,
    /** Added to each side's rest in the relative readout (rate units), so a silent side is not infinitely sensitive. */
    readoutFloor: 0.2,
    /** Head-on approaches drive both eyes equally: this fraction of the bilateral level becomes a turn to the right. */
    loomTieBias: 0.5,
    /** Bilateral looming lowers the mean wing amplitude by this times the bilateral level (brake). */
    loomBrake: 0.3,
  };
  readonly net: NetBackend;
  calibrated = false;
  /** Network settled; running live with the output off until the warm-up ends. */
  private armed = false;
  private warm = 0;
  private homeostatDone = false;

  /** Per ommatidium (per side): lamina unit indices, flattened with offsets, and their R->L weight. */
  private readonly lamPtrL: Int32Array;
  private readonly lamIdxL: Int32Array;
  private readonly lamWL: Float32Array;
  private readonly lamPtrR: Int32Array;
  private readonly lamIdxR: Int32Array;
  private readonly lamWR: Float32Array;
  /** Virtual photoreceptor membrane per ommatidium, and slow luminance mean for adaptation. */
  private readonly vR_L: Float32Array;
  private readonly vR_R: Float32Array;
  private readonly meanL: Float32Array;
  private readonly meanR: Float32Array;
  private adapted = false;
  readonly coverage: { types: number; edges: number; units: number };
  private readonly lptc: Int32Array;
  private readonly hsL: Int32Array;
  private readonly hsR: Int32Array;
  private readonly dnL: Int32Array;
  private readonly dnR: Int32Array;
  /** Looming detectors per eye (LC4 + LPLC2), with receptive-field centres from their presynaptic columns. */
  readonly lc: { L: Int32Array; R: Int32Array; az: Float32Array; el: Float32Array };
  private loomRest = { L: 0, R: 0 };
  private topBuf = new Float64Array(0);
  loom = { L: 0, R: 0 };
  readonly groups: Record<string, Int32Array>;
  /** Per-ommatidium T4a and T4b unit lists for the HUD direction map. */
  private readonly t4aByOmm: { L: Int32Array[]; R: Int32Array[] };
  private readonly t4bByOmm: { L: Int32Array[]; R: Int32Array[] };
  private readonly homeoUnits: Int32Array;
  private readonly homeoTargets: Float32Array;
  private turn = 0;
  private brake = 0;
  sideGain = { L: 1, R: 1 };
  /** Slow per-side readout offsets (rest level of each side, adapting). */
  /** Rest level of each side, the warm-up average; the readout is relative to it. */
  offsetL = 0;
  offsetR = 0;
  homeoErr = 0;
  homeoBias = 0;

  constructor(
    readonly graph: Graph,
    readonly fv: FlyvisParams,
    ommL: Ommatidia,
    ommR: Ommatidia,
    backend: NetBackendKind = gpuAvailable() ? "gpu" : "worker",
  ) {
    if (!ommL.col || !ommR.col) throw new Error("OpticBrain needs column-based ommatidia");
    const applied = applyFlyvis(graph, fv, this.params.defaultScale, this.params.lptcScale);
    let nCov = 0;
    for (let i = 0; i < graph.n; i++) nCov += applied.covered[i]!;
    this.coverage = { types: applied.nCoveredTypes, edges: applied.nCoveredEdges, units: nCov };
    this.net = createNet(backend, graph, applied.w, applied.tau, applied.bias, this.params.wScale, this.params.netDt, this.params.rMax);
    this.vR_L = new Float32Array(ommL.count);
    this.vR_R = new Float32Array(ommR.count);
    this.meanL = new Float32Array(ommL.count);
    this.meanR = new Float32Array(ommR.count);

    const byCol = (pred: (t: string) => boolean): Map<number, number[]> => {
      const m = new Map<number, number[]>();
      for (let i = 0; i < graph.n; i++) {
        const c = graph.col[i]!;
        if (c < 0 || !pred(typeName(graph, i))) continue;
        let a = m.get(c);
        if (!a) m.set(c, (a = []));
        a.push(i);
      }
      return m;
    };
    const lamW = fv.photoreceptor.laminaInput;
    const flatten = (omm: Ommatidia, m: Map<number, number[]>): [Int32Array, Int32Array, Float32Array] => {
      const ptr = new Int32Array(omm.count + 1);
      const idx: number[] = [];
      const wts: number[] = [];
      for (let k = 0; k < omm.count; k++) {
        for (const i of m.get(omm.col![k]!) ?? []) {
          idx.push(i);
          wts.push(lamW[typeName(graph, i)] ?? 0);
        }
        ptr[k + 1] = idx.length;
      }
      return [ptr, Int32Array.from(idx), Float32Array.from(wts)];
    };
    const lam = byCol((t) => LAMINA.test(t));
    [this.lamPtrL, this.lamIdxL, this.lamWL] = flatten(ommL, lam);
    [this.lamPtrR, this.lamIdxR, this.lamWR] = flatten(ommR, lam);
    const t4a = byCol((t) => t === "T4a");
    const t4b = byCol((t) => t === "T4b");
    const lists = (omm: Ommatidia, m: Map<number, number[]>): Int32Array[] =>
      Array.from({ length: omm.count }, (_, k) => Int32Array.from(m.get(omm.col![k]!) ?? []));
    this.t4aByOmm = { L: lists(ommL, t4a), R: lists(ommR, t4a) };
    this.t4bByOmm = { L: lists(ommL, t4b), R: lists(ommR, t4b) };

    // tonic drive only for LPTCs whose parameters are not fitted (fitted ones carry their bias)
    this.lptc = unitsWhere(graph, (t) => LPTC.test(t) && !(fv.source.startsWith("fitted") && hasFlyvisType(fv, t)));
    this.hsL = unitsWhere(graph, (t, s) => /^HS[ENS]$/.test(t) && s === "L");
    this.hsR = unitsWhere(graph, (t, s) => /^HS[ENS]$/.test(t) && s === "R");
    this.dnL = unitsWhere(graph, (t, s) => DNG02.test(t) && s === "L");
    this.dnR = unitsWhere(graph, (t, s) => DNG02.test(t) && s === "R");
    const g = (re: RegExp, side?: string): Int32Array =>
      unitsWhere(graph, (t, s) => re.test(t) && (side === undefined || s === side));
    this.lc = { L: g(LOOMING, "L"), R: g(LOOMING, "R"), az: new Float32Array(graph.n).fill(NaN), el: new Float32Array(graph.n).fill(NaN) };
    {
      // RF centre = synapse-weighted mean direction of columnar presynaptic partners.
      const isLC = new Uint8Array(graph.n);
      for (const i of this.lc.L) isLC[i] = 1;
      for (const i of this.lc.R) isLC[i] = 1;
      const sx = new Float32Array(graph.n);
      const sy = new Float32Array(graph.n);
      const sz = new Float32Array(graph.n);
      const cols = graph.columns;
      for (let e = 0; e < graph.m; e++) {
        const b = graph.post[e]!;
        if (!isLC[b]) continue;
        const c = graph.col[graph.pre[e]!]!;
        if (c < 0) continue;
        const w = graph.weight[e]!;
        const a = cols.az[c]!;
        const el = cols.el[c]!;
        sx[b] = sx[b]! + w * Math.sin(a) * Math.cos(el);
        sy[b] = sy[b]! + w * Math.sin(el);
        sz[b] = sz[b]! + w * Math.cos(a) * Math.cos(el);
      }
      for (let i = 0; i < graph.n; i++) {
        if (!isLC[i] || sx[i] === 0 && sy[i] === 0 && sz[i] === 0) continue;
        this.lc.az[i] = Math.atan2(sx[i]!, sz[i]!);
        this.lc.el[i] = Math.atan2(sy[i]!, Math.hypot(sx[i]!, sz[i]!));
      }
    }
    this.groups = {
      L1: g(/^L1$/),
      Mi1: g(/^Mi1$/),
      Mi4: g(/^Mi4$/),
      Mi9: g(/^Mi9$/),
      T4a: g(/^T4a$/),
      T4b: g(/^T4b$/),
      T4c: g(/^T4c$/),
      T4d: g(/^T4d$/),
      T5a: g(/^T5a$/),
      T5b: g(/^T5b$/),
      LPi: g(/^LPi/),
      HS: g(/^HS[ENS]$/),
      HSL: g(/^HS[ENS]$/, "L"),
      HSR: g(/^HS[ENS]$/, "R"),
      VS: g(/^VS/),
      T4aL: g(/^T4a$/, "L"),
      T4aR: g(/^T4a$/, "R"),
      T4bL: g(/^T4b$/, "L"),
      T4bR: g(/^T4b$/, "R"),
      T5aL: g(/^T5a$/, "L"),
      T5aR: g(/^T5a$/, "R"),
      T5bL: g(/^T5b$/, "L"),
      T5bR: g(/^T5b$/, "R"),
      LC4L: g(/^LC4$/, "L"),
      LC4R: g(/^LC4$/, "R"),
      LPLC2L: g(/^LPLC2$/, "L"),
      LPLC2R: g(/^LPLC2$/, "R"),
      DNp: g(/^DNp0[1-6]$/),
      DNg02: g(/^DNg02_/),
      MN: unitsWhere(graph, (_t, _s, r) => r === "output"),
    };
    // Homeostatic bias for the pooling cells (LPTCs, LPi, looming LCs) whose parameters are not
    // fitted, and, with raw flyvis params, for the covered optic types too (targeting the flyvis
    // resting membrane value under grey). Fitted types (including LC4/LPLC2 after train_loom.py)
    // carry their own bias and are left alone; a homeostat would re-centre them on the scene's
    // static contrast and cancel the fitted operating point.
    // The central brain and VNC run unbiased: DNg02 tonic drive only.
    const fitted = fv.source.startsWith("fitted");
    const homeo: number[] = [];
    for (let i = 0; i < graph.n; i++) {
      const t = typeName(graph, i);
      const coveredType = fitted && hasFlyvisType(fv, t);
      if ((isPooling(graph, i) && !coveredType) || (graph.role[i] === 5 && !fitted && flyvisRestV(fv, t) !== undefined)) homeo.push(i);
    }
    this.homeoUnits = Int32Array.from(homeo);
    this.homeoTargets = new Float32Array(this.homeoUnits.length);
    for (let k = 0; k < this.homeoUnits.length; k++) {
      const rv = flyvisRestV(fv, typeName(graph, this.homeoUnits[k]!));
      this.homeoTargets[k] = rv ?? this.params.restTarget;
    }
    let lamCount = 0;
    lam.forEach((a) => (lamCount += a.length));
    this.name = `optic-v2 [${backend}]: ${graph.n} units, ${graph.m} edges, ${graph.columns.count} columns; ${fitted ? "fitted" : "flyvis"} params on ${this.coverage.types} types / ${this.coverage.edges} edges (${lamCount} lamina inputs)`;
    void this.reset();
  }

  reset(): void {
    this.calibrated = false;
    this.armed = false;
    this.turn = 0;
    this.vR_L.fill(0);
    this.vR_R.fill(0);
    this.adapted = false;
    this.net.reset();
    this.injectTonic();
    void this.net.whenReady().then(async () => {
      const p = this.params;
      if (!this.homeostatDone && this.homeoUnits.length > 0) {
        // Once per load: gentle steps so the recurrent network settles rather than oscillates.
        const info = await this.net.homeostat(this.homeoUnits, this.homeoTargets, p.restRounds, 0.25, 0.25);
        this.homeoErr = info.meanErr;
        this.homeoBias = info.biasMean;
        this.homeostatDone = true;
      } else {
        await this.net.settle(0.5);
      }
      // Now run live in the scene with the output off until photoreceptor adaptation and the
      // pooling cells have settled; step() takes the offsets when the warm-up ends.
      this.warm = 0;
      this.armed = true;
    });
  }

  private injectTonic(): void {
    const p = this.params;
    const ext = this.net.ext;
    ext.fill(0);
    // Uniform mid-grey through the photoreceptors so the lamina rests where it would in flyvis.
    const pr = this.fv.photoreceptor;
    const rRest = Math.max(0, pr.restOffset + pr.stimGain * 0.5 * p.stimGain);
    for (let k = 0; k < this.lamIdxL.length; k++) ext[this.lamIdxL[k]!] = this.lamWL[k]! * rRest;
    for (let k = 0; k < this.lamIdxR.length; k++) ext[this.lamIdxR[k]!] = this.lamWR[k]! * rRest;
    for (let k = 0; k < this.lptc.length; k++) ext[this.lptc[k]!] = p.lptcBias;
    for (let k = 0; k < this.dnL.length; k++) ext[this.dnL[k]!] = p.dnBias;
    for (let k = 0; k < this.dnR.length; k++) ext[this.dnR[k]!] = p.dnBias;
  }

  /** Virtual photoreceptors: tau dV/dt = -V + stim + bias, r = relu(V); then R -> lamina. */
  private injectEye(
    lum: Float32Array,
    mean: Float32Array,
    vR: Float32Array,
    ptr: Int32Array,
    idx: Int32Array,
    wts: Float32Array,
    dt: number,
  ): void {
    const p = this.params;
    const pr = this.fv.photoreceptor;
    const ext = this.net.ext;
    const alpha = Math.min(1, dt / pr.tau);
    const aAdapt = Math.min(1, dt / p.adaptTau);
    const n = ptr.length - 1;
    for (let k = 0; k < n; k++) {
      if (!this.adapted) mean[k] = lum[k]!;
      else mean[k] = mean[k]! + aAdapt * (lum[k]! - mean[k]!);
      const stim = Math.min(p.stimMax, (0.5 * lum[k]!) / (mean[k]! + 1e-3));
      const target = pr.restOffset + pr.stimGain * p.stimGain * stim;
      vR[k] = vR[k]! + alpha * (target - vR[k]!);
      const r = Math.max(0, vR[k]!);
      for (let j = ptr[k]!; j < ptr[k + 1]!; j++) ext[idx[j]!] = wts[j]! * r;
    }
  }

  step(input: EyeInput, dt: number): MotorCommand {
    const p = this.params;
    this.net.setParams({ wScale: p.wScale, netDt: p.netDt });
    this.injectTonic();
    this.injectEye(input.lumLeft, this.meanL, this.vR_L, this.lamPtrL, this.lamIdxL, this.lamWL, dt);
    this.injectEye(input.lumRight, this.meanR, this.vR_R, this.lamPtrR, this.lamIdxR, this.lamWR, dt);
    this.adapted = true;
    if (this.armed) this.net.step(dt);
    if (this.armed && !this.calibrated) {
      this.warm += dt;
      if (this.warm >= p.warmSeconds) {
        const sides = this.readoutSides();
        this.offsetL = sides.L;
        this.offsetR = sides.R;
        this.loomRest = { L: this.topkRate(this.lc.L), R: this.topkRate(this.lc.R) };
        this.calibrated = true;
      }
    }

    const sides = this.readoutSides();
    if (this.calibrated) {
      this.offsetL += ((sides.L - this.offsetL) * dt) / p.offsetTau;
      this.offsetR += ((sides.R - this.offsetR) * dt) / p.offsetTau;
    }
    // Deviation from rest, relative to rest: when both sides collapse (fast self-rotation beyond
    // the motion detectors' range) the readout goes to -1 on both sides and cancels, instead
    // of leaving the difference of the two rest levels as a permanent turn command.
    const dL = (this.sideGain.L * (sides.L - this.offsetL)) / (this.offsetL + p.readoutFloor);
    const dR = (this.sideGain.R * (sides.R - this.offsetR)) / (this.offsetR + p.readoutFloor);
    const diff = dL - dR;
    this.loom.L = Math.max(0, this.topkRate(this.lc.L) - this.loomRest.L - p.loomThreshold);
    this.loom.R = Math.max(0, this.topkRate(this.lc.R) - this.loomRest.R - p.loomThreshold);
    // Looming on the left eye -> turn right (turn > 0 = yaw right). A head-on object drives both
    // eyes: break the tie toward the right and brake.
    const bilateral = Math.min(this.loom.L, this.loom.R);
    const avoid = p.loomGain * (this.loom.L - this.loom.R + p.loomTieBias * bilateral);
    this.brake = Math.min(p.baseAmp, p.loomBrake * bilateral);
    const raw = this.calibrated ? p.readoutSign * p.outputGain * diff + avoid : 0;
    this.turn = Math.max(-p.maxTurn, Math.min(p.maxTurn, raw));
    const base = p.baseAmp - this.brake;
    return {
      left: clamp01(base + this.turn / 2),
      right: clamp01(base - this.turn / 2),
    };
  }

  /** Mean rate of the loomTopK most active units in the list. */
  private topkRate(idx: Int32Array): number {
    const k = Math.min(this.params.loomTopK, idx.length);
    if (k === 0) return 0;
    const top = this.topBuf.length === k ? this.topBuf : (this.topBuf = new Float64Array(k));
    top.fill(-Infinity);
    const r = this.net.r;
    for (let j = 0; j < idx.length; j++) {
      const v = r[idx[j]!]!;
      if (v <= top[k - 1]!) continue;
      let p = k - 1;
      while (p > 0 && top[p - 1]! < v) {
        top[p] = top[p - 1]!;
        p--;
      }
      top[p] = v;
    }
    let s = 0;
    for (let j = 0; j < k; j++) s += top[j]!;
    return s / k;
  }

  /** Readout rates per side, for the side-gain calibration in main.ts. */
  /** Where the start-up is: "starting" (backend), "settling", "warm-up" (progress 0..1), or "live". */
  get stage(): { name: "starting" | "settling" | "warm-up" | "live"; progress: number } {
    if (!this.net.ready) return { name: "starting", progress: 0 };
    if (!this.armed) return { name: "settling", progress: 0 };
    if (!this.calibrated) return { name: "warm-up", progress: Math.min(1, this.warm / this.params.warmSeconds) };
    return { name: "live", progress: 1 };
  }

  readoutSides(): { L: number; R: number } {
    const n = this.net;
    return this.params.readout === "hs"
      ? { L: n.meanRate(this.hsL), R: n.meanRate(this.hsR) }
      : { L: n.meanRate(this.dnL), R: n.meanRate(this.dnR) };
  }

  /**
   * Equalise the two eyes: the wiring (and the fit) give the left and right readout
   * populations different gains, so straight flight, which drives both with the same
   * front-to-back flow, would read as a turn. Gains are set so the preferred-direction
   * response of each side is the same; the offset is re-centred afterwards.
   */
  setSideGain(pdL: number, pdR: number): void {
    const m = (pdL + pdR) / 2;
    if (pdL > 1e-3 && pdR > 1e-3) this.sideGain = { L: m / pdL, R: m / pdR };
  }

  /** Per-ommatidium T4a - T4b mean rate (front-to-back minus back-to-front). */
  directionMap(side: "L" | "R", out: Float32Array): void {
    const a = this.t4aByOmm[side];
    const b = this.t4bByOmm[side];
    for (let k = 0; k < out.length; k++) out[k] = this.net.meanRate(a[k]!) - this.net.meanRate(b[k]!);
  }

  telemetry(): Record<string, number> {
    const n = this.net;
    const g = this.groups;
    return {
      turn: this.turn,
      offsetL: this.offsetL,
      offsetR: this.offsetR,
      gainL: this.sideGain.L,
      gainR: this.sideGain.R,
      calib: this.calibrated ? 1 : 0,
      homeoErr: this.homeoErr,
      homeoBias: this.homeoBias,
      netMs: n.lastMs,
      netSteps: n.lastSteps,
      simT: n.simTime,
      L1: n.meanRate(g["L1"]!),
      Mi1: n.meanRate(g["Mi1"]!),
      Mi4: n.meanRate(g["Mi4"]!),
      Mi9: n.meanRate(g["Mi9"]!),
      T4a: n.meanRate(g["T4a"]!),
      T4b: n.meanRate(g["T4b"]!),
      T4c: n.meanRate(g["T4c"]!),
      T4d: n.meanRate(g["T4d"]!),
      T5a: n.meanRate(g["T5a"]!),
      T5b: n.meanRate(g["T5b"]!),
      LPi: n.meanRate(g["LPi"]!),
      hsL: n.meanRate(g["HSL"]!),
      hsR: n.meanRate(g["HSR"]!),
      dng02L: n.meanRate(this.dnL),
      dng02R: n.meanRate(this.dnR),
      lc4L: n.meanRate(g["LC4L"]!),
      lc4R: n.meanRate(g["LC4R"]!),
      lplc2L: n.meanRate(g["LPLC2L"]!),
      lplc2R: n.meanRate(g["LPLC2R"]!),
      loomL: this.loom.L,
      loomR: this.loom.R,
      brake: this.brake,
      dnp: n.meanRate(g["DNp"]!),
    };
  }

  /** Unit label helper for console poking. */
  label(i: number): string {
    return `${typeName(this.graph, i)}_${sideName(this.graph, i)}`;
  }
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}
