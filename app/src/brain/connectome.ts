/**
 * Layer 3 (brain), milestone 2: real MaleCNS wiring behind the Brain interface.
 *
 *   HR correlators (stand-in for T4/T5)
 *     -> external drive on the lobula plate tangential cells (HS, H2)
 *     -> the extracted graph: LPTCs -> posterior slope / brain bridge -> DNg02 (+ other DNs)
 *        -> VNC interneurons -> wing motor neurons
 *     -> readout: DNg02 population rate, left vs right (default), or steering MNs
 *
 * Only the first and last arrows are hand-written. Everything in between is
 * synapse counts and predicted transmitter signs from the connectome.
 *
 * Why the readout sits at DNg02 and not at the motor neurons: in this graph
 * DNg02 projects to left and right wing MNs almost equally (see the
 * laterality check in data/out/flight-v1.report.md), so a rate readout at the
 * MNs is not lateralized. Real turning is a timing code in the steering
 * muscles that a rate model cannot express. DNg02 itself is cleanly
 * lateralized: HS_L -> PS080_L (GABA) -| DNg02_R.
 *
 * Side mapping assumption (readoutSign = -1): a higher DNg02 rate on one side
 * means the fly turns AWAY from that side. HS_L reports "I am yawing right",
 * disinhibits DNg02_L, and the corrective turn is to the left. Set +1 for the
 * ipsilateral reading. Which one the fly actually uses is not settled by the
 * synapse counts here.
 *
 * Injection sign conventions (retinal flow > 0 = toward +azimuth = rightward):
 *   HS cells prefer front-to-back (progressive) motion on their own eye.
 *     right eye progressive = +flowR, left eye progressive = -flowL
 *   H2 prefers back-to-front (regressive) motion on its own eye.
 */
import type { Ommatidia } from "../eye/ommatidia";
import { buildCSR, unitsWhere, unitTau, type Graph } from "./graph";
import { HRCorrelator } from "./hr";
import { RateNet } from "./rate-net";
import type { Brain, EyeInput, MotorCommand } from "./types";

const HS = /^HS[ENS]$/;
const H2 = /^H2$/;
const LPTC = /^(HS[ENST]|VS|VST1|VST2|VSm|H2|DCH|VCH)$/;
const STEERING_MN = /^(b[123]|i[12]|iii[134]|hg[1-4]|tp[12]|tpn|ps[12]) MN$|^MNwm3[56]$/;
const DNG02 = /^DNg02_/;

export interface ConnectomeParams {
  /** Tonic drive on all LPTC inputs so motion can modulate them both ways. */
  inputBias: number;
  /** Drive per unit of retinal flow. */
  inputGain: number;
  /** Cap on |motion modulation| so a transient cannot saturate the LPTCs. */
  inputClip: number;
  /**
   * Tonic "flight state" drive on DNg02. In the fly these DNs are active
   * throughout flight; the graph's main visual relay (PS080) is GABAergic,
   * so without a baseline the inhibition has nothing to act on.
   */
  dnBias: number;
  /** Synapse-count to drive scale in the network. */
  wScale: number;
  /** Wing asymmetry per unit of MN rate difference. */
  outputGain: number;
  /** Where to read the turn from. */
  readout: "dng02" | "mn";
  /** -1: higher left readout turns the fly left (contralateral wing). +1: ipsilateral. */
  readoutSign: number;
  baseAmp: number;
  maxTurn: number;
  /** Network integration step, seconds. */
  netDt: number;
}

export class ConnectomeBrain implements Brain {
  readonly name: string;
  readonly params: ConnectomeParams = {
    inputBias: 0.3,
    inputGain: 10,
    inputClip: 0.6,
    dnBias: 0.5,
    wScale: 0.03,
    outputGain: 4,
    readout: "dng02",
    readoutSign: -1,
    baseAmp: 0.5,
    maxTurn: 0.5,
    netDt: 0.001,
  };

  readonly net: RateNet;
  readonly hrL: HRCorrelator;
  readonly hrR: HRCorrelator;

  private readonly hsL: Int32Array;
  private readonly hsR: Int32Array;
  private readonly h2L: Int32Array;
  private readonly h2R: Int32Array;
  private readonly lptcAll: Int32Array;
  private readonly dnL: Int32Array;
  private readonly dnR: Int32Array;
  private readonly mnL: Int32Array;
  private readonly mnR: Int32Array;
  private turn = 0;
  private stepsPerFrame = 0;
  /** Resting (rL - rR) with zero flow, removed from the readout. */
  private offset = 0;

  readonly lattice = "fibonacci" as const;

  constructor(
    readonly graph: Graph,
    ommL: Ommatidia,
    ommR: Ommatidia,
  ) {
    const csr = buildCSR(graph);
    this.net = new RateNet(csr, unitTau(graph), { wScale: this.params.wScale });
    this.hrL = new HRCorrelator(ommL);
    this.hrR = new HRCorrelator(ommR);

    this.hsL = unitsWhere(graph, (t, s) => HS.test(t) && s === "L");
    this.hsR = unitsWhere(graph, (t, s) => HS.test(t) && s === "R");
    this.h2L = unitsWhere(graph, (t, s) => H2.test(t) && s === "L");
    this.h2R = unitsWhere(graph, (t, s) => H2.test(t) && s === "R");
    this.lptcAll = unitsWhere(graph, (t) => LPTC.test(t));
    this.dnL = unitsWhere(graph, (t, s) => DNG02.test(t) && s === "L");
    this.dnR = unitsWhere(graph, (t, s) => DNG02.test(t) && s === "R");
    this.mnL = unitsWhere(graph, (t, s) => STEERING_MN.test(t) && s === "L");
    this.mnR = unitsWhere(graph, (t, s) => STEERING_MN.test(t) && s === "R");
    this.name = `connectome: ${graph.n} units, ${graph.m} edges`;
  }

  reset(): void {
    this.net.reset();
    this.hrL.reset();
    this.hrR.reset();
    this.turn = 0;
    this.calibrate();
  }

  /** Settle the network with tonic drive only and record the resting readout asymmetry. */
  private calibrate(seconds = 1.0): void {
    this.inject(0, 0);
    const steps = Math.round(seconds / this.params.netDt);
    for (let k = 0; k < steps; k++) this.net.step(this.params.netDt);
    this.offset = this.readoutDiff();
  }

  private readoutDiff(): number {
    const n = this.net;
    return this.params.readout === "mn"
      ? n.meanRate(this.mnL) - n.meanRate(this.mnR)
      : n.meanRate(this.dnL) - n.meanRate(this.dnR);
  }

  private inject(flowL: number, flowR: number): void {
    const p = this.params;
    const ext = this.net.ext;
    const mod = (v: number): number => Math.max(-p.inputClip, Math.min(p.inputClip, p.inputGain * v));
    for (let k = 0; k < this.lptcAll.length; k++) ext[this.lptcAll[k]!] = p.inputBias;
    setAll(ext, this.hsL, p.inputBias + mod(-flowL));
    setAll(ext, this.hsR, p.inputBias + mod(flowR));
    setAll(ext, this.h2L, p.inputBias + mod(flowL));
    setAll(ext, this.h2R, p.inputBias + mod(-flowR));
    setAll(ext, this.dnL, p.dnBias);
    setAll(ext, this.dnR, p.dnBias);
  }

  step(input: EyeInput, dt: number): MotorCommand {
    const p = this.params;
    this.net.params.wScale = p.wScale;
    const flowL = this.hrL.update(input.left, dt);
    const flowR = this.hrR.update(input.right, dt);

    this.inject(flowL, flowR);

    // Integrate the network through this frame.
    let remaining = dt;
    this.stepsPerFrame = 0;
    while (remaining > 1e-6) {
      const h = Math.min(p.netDt, remaining);
      this.net.step(h);
      remaining -= h;
      this.stepsPerFrame++;
    }

    // turn > 0 = yaw right. See the side-mapping note above.
    const raw = p.readoutSign * p.outputGain * (this.readoutDiff() - this.offset);
    this.turn = Math.max(-p.maxTurn, Math.min(p.maxTurn, raw));
    return {
      left: clamp01(p.baseAmp + this.turn / 2),
      right: clamp01(p.baseAmp - this.turn / 2),
    };
  }

  telemetry(): Record<string, number> {
    const n = this.net;
    let active = 0;
    for (let i = 0; i < n.n; i++) if (n.r[i]! > 0) active++;
    return {
      flowL: this.hrL.value,
      flowR: this.hrR.value,
      hsL: n.meanRate(this.hsL),
      hsR: n.meanRate(this.hsR),
      dng02L: n.meanRate(this.dnL),
      dng02R: n.meanRate(this.dnR),
      mnL: n.meanRate(this.mnL),
      mnR: n.meanRate(this.mnR),
      turn: this.turn,
      offset: this.offset,
      activeFrac: active / n.n,
      netSteps: this.stepsPerFrame,
    };
  }
}

function setAll(arr: Float32Array, idx: Int32Array, v: number): void {
  for (let k = 0; k < idx.length; k++) arr[idx[k]!] = v;
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}
