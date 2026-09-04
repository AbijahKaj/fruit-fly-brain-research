/**
 * Layer 3 (brain), milestone 3: the real per-column optic lobe.
 *
 *   luminance per column -> one virtual photoreceptor (flyvis R1-R6 params)
 *     -> histaminergic drive on the lamina (L1, L2, L3) of that column
 *     -> the extracted graph: lamina -> medulla -> T4/T5 -> lobula plate
 *        -> posterior slope -> DNg02 -> VNC -> wing MNs
 *     -> readout: HS left vs right (default), or DNg02 left vs right
 *
 * Status: with fitted parameters the loop closes through the lobula plate
 * (HS readout). The HS -> posterior slope -> DNg02 hop that worked in the
 * small milestone-2 graph is not yet calibrated in this graph: the central
 * brain runs at the default synapse scale and DNg02 sits at a constant rate.
 *
 * Compared with milestone 2 the hand-written correlator is gone. Direction
 * selectivity has to come from the wiring (Mi9 leading, Mi4 trailing) plus
 * per-type time constants, biases and per-pair synapse strengths. Those come
 * from flyvis (see flyvis.ts); units flyvis does not cover (lobula plate,
 * central brain) get a homeostatic resting bias.
 *
 * The network runs in a Web Worker (remote-net.ts); rates seen here lag by
 * one worker round trip.
 */
import type { Ommatidia } from "../eye/ommatidia";
import { unitsWhere, typeName, sideName, type Graph } from "./graph";
import { applyFlyvis, flyvisRestV, isPooling, type FlyvisParams } from "./flyvis";
import { RemoteNet } from "./remote-net";
import type { Brain, EyeInput, MotorCommand } from "./types";

const LAMINA = /^L[123]$/;
const LPTC = /^(HS[ENST]|VS|VST1|VST2|VSm|H2|DCH|VCH)$/;
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
  /** Flight-state drive on DNg02 (see connectome.ts). */
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
}

export class OpticBrain implements Brain {
  readonly name: string;
  readonly lattice = "columns" as const;
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
    outputGain: 4,
    readoutSign: -1,
    baseAmp: 0.5,
    maxTurn: 0.5,
    netDt: 0.004,
    rMax: 5,
  };
  readonly net: RemoteNet;
  calibrated = false;
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
  readonly groups: Record<string, Int32Array>;
  /** Per-ommatidium T4a and T4b unit lists for the HUD direction map. */
  private readonly t4aByOmm: { L: Int32Array[]; R: Int32Array[] };
  private readonly t4bByOmm: { L: Int32Array[]; R: Int32Array[] };
  private readonly homeoUnits: Int32Array;
  private readonly homeoTargets: Float32Array;
  private turn = 0;
  private offset = 0;
  homeoErr = 0;
  homeoBias = 0;

  constructor(
    readonly graph: Graph,
    readonly fv: FlyvisParams,
    ommL: Ommatidia,
    ommR: Ommatidia,
  ) {
    if (!ommL.col || !ommR.col) throw new Error("OpticBrain needs column-based ommatidia");
    const applied = applyFlyvis(graph, fv, this.params.defaultScale, this.params.lptcScale);
    let nCov = 0;
    for (let i = 0; i < graph.n; i++) nCov += applied.covered[i]!;
    this.coverage = { types: applied.nCoveredTypes, edges: applied.nCoveredEdges, units: nCov };
    this.net = new RemoteNet(graph, applied.w, applied.tau, applied.bias, this.params.wScale, this.params.netDt, this.params.rMax);
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

    this.lptc = unitsWhere(graph, (t) => LPTC.test(t));
    this.hsL = unitsWhere(graph, (t, s) => /^HS[ENS]$/.test(t) && s === "L");
    this.hsR = unitsWhere(graph, (t, s) => /^HS[ENS]$/.test(t) && s === "R");
    this.dnL = unitsWhere(graph, (t, s) => DNG02.test(t) && s === "L");
    this.dnR = unitsWhere(graph, (t, s) => DNG02.test(t) && s === "R");
    const g = (re: RegExp, side?: string): Int32Array =>
      unitsWhere(graph, (t, s) => re.test(t) && (side === undefined || s === side));
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
      DNg02: g(/^DNg02_/),
      MN: unitsWhere(graph, (_t, _s, r) => r === "output"),
    };
    // Homeostatic bias for the pooling cells (LPTCs, LPi, looming LCs), which flyvis does not
    // cover, and, with raw flyvis params, for the covered optic types too (targeting the flyvis
    // resting membrane value under grey). Fitted params already carry the right optic biases.
    // The central brain and VNC run as in milestone 2: no bias, DNg02 tonic drive only.
    const fitted = fv.source.startsWith("fitted");
    const homeo: number[] = [];
    for (let i = 0; i < graph.n; i++) {
      const t = typeName(graph, i);
      if (isPooling(graph, i) || (graph.role[i] === 5 && !fitted && flyvisRestV(fv, t) !== undefined)) homeo.push(i);
    }
    this.homeoUnits = Int32Array.from(homeo);
    this.homeoTargets = new Float32Array(this.homeoUnits.length);
    for (let k = 0; k < this.homeoUnits.length; k++) {
      const rv = flyvisRestV(fv, typeName(graph, this.homeoUnits[k]!));
      this.homeoTargets[k] = rv ?? this.params.restTarget;
    }
    let lamCount = 0;
    lam.forEach((a) => (lamCount += a.length));
    this.name = `optic-v2: ${graph.n} units, ${graph.m} edges, ${graph.columns.count} columns; ${fitted ? "fitted" : "flyvis"} params on ${this.coverage.types} types / ${this.coverage.edges} edges (${lamCount} lamina inputs)`;
    void this.reset();
  }

  reset(): void {
    this.calibrated = false;
    this.turn = 0;
    this.vR_L.fill(0);
    this.vR_R.fill(0);
    this.adapted = false;
    this.net.reset();
    this.injectTonic();
    void this.net.whenReady().then(async () => {
      const p = this.params;
      if (!this.homeostatDone) {
        // Once per load: gentle steps so the recurrent network settles rather than oscillates.
        const info = await this.net.homeostat(this.homeoUnits, this.homeoTargets, p.restRounds, 0.25, 0.25);
        this.homeoErr = info.meanErr;
        this.homeoBias = info.biasMean;
        this.homeostatDone = true;
      } else {
        await this.net.settle(0.5);
      }
      this.offset = this.readoutDiff();
      this.calibrated = true;
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
    if (!input.lumLeft || !input.lumRight) throw new Error("OpticBrain needs raw luminance input");
    this.injectEye(input.lumLeft, this.meanL, this.vR_L, this.lamPtrL, this.lamIdxL, this.lamWL, dt);
    this.injectEye(input.lumRight, this.meanR, this.vR_R, this.lamPtrR, this.lamIdxR, this.lamWR, dt);
    this.adapted = true;
    if (this.calibrated) this.net.step(dt);

    const diff = this.readoutDiff() - this.offset;
    const raw = this.calibrated ? p.readoutSign * p.outputGain * diff : 0;
    this.turn = Math.max(-p.maxTurn, Math.min(p.maxTurn, raw));
    return {
      left: clamp01(p.baseAmp + this.turn / 2),
      right: clamp01(p.baseAmp - this.turn / 2),
    };
  }

  /** Left minus right at the chosen readout level. */
  private readoutDiff(): number {
    const n = this.net;
    return this.params.readout === "hs"
      ? n.meanRate(this.hsL) - n.meanRate(this.hsR)
      : n.meanRate(this.dnL) - n.meanRate(this.dnR);
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
      offset: this.offset,
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
