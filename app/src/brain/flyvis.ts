/**
 * Trained parameters from flyvis (Lappalainen et al., Nature 2024), dumped by
 * train/dump_flyvis_params.py, applied to our MaleCNS graph:
 *
 *   per type        tau, bias (resting drive)
 *   per type pair   synapse strength and sign  ->  w = count * sign * strength
 *   photoreceptor   tau, bias, and the summed R1-R6 -> L1/L2/L3 drive
 *
 * Types or pairs flyvis does not know keep the graph's tau, the transmitter
 * sign times a default scale, and get a homeostatic bias instead.
 */
import type { Graph } from "./graph";
import { typeName } from "./graph";

export function hasFlyvisType(fv: FlyvisParams, t: string): boolean {
  return fv.types[fvName(t)] !== undefined;
}

/** flyvis resting membrane value for a type, or undefined. */
export function flyvisRestV(fv: FlyvisParams, t: string): number | undefined {
  return fv.types[fvName(t)]?.restV;
}

export interface FlyvisParams {
  source: string;
  types: Record<string, { tau: number; bias: number; restV?: number }>;
  photoreceptor: {
    tau: number;
    bias: number;
    /** Measured: R activity = restOffset + stimGain * stimulus. */
    restOffset: number;
    stimGain: number;
    laminaInput: Record<string, number>;
  };
  pairs: Array<{ pre: string; post: string; strength: number; sign: number }>;
}

export interface Applied {
  tau: Float32Array;
  bias: Float32Array;
  /** Signed, scaled per-edge weights. */
  w: Float32Array;
  /** 1 where the unit's type has flyvis parameters (bias set), 0 where the homeostat applies. */
  covered: Uint8Array;
  nCoveredTypes: number;
  nCoveredEdges: number;
}

export async function loadFlyvis(url: string): Promise<FlyvisParams> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
  return (await res.json()) as FlyvisParams;
}

/** MaleCNS names that map onto one flyvis type. */
const ALIAS: Record<string, string> = { TmY9a: "TmY9", TmY9b: "TmY9" };
const fvName = (t: string): string => ALIAS[t] ?? t;

export function applyFlyvis(g: Graph, fv: FlyvisParams, defaultScale: number): Applied {
  const n = g.n;
  const tau = new Float32Array(n);
  const bias = new Float32Array(n);
  const covered = new Uint8Array(n);
  const typeCovered = g.types.map((t) => fv.types[fvName(t.name)] !== undefined);
  for (let i = 0; i < n; i++) {
    const t = g.types[g.type[i]!]!;
    const p = fv.types[fvName(t.name)];
    if (p) {
      tau[i] = p.tau;
      bias[i] = p.bias;
      covered[i] = 1;
    } else {
      tau[i] = t.tau;
    }
  }
  const pair = new Map<string, { strength: number; sign: number }>();
  for (const p of fv.pairs) pair.set(`${p.pre} ${p.post}`, p);
  const w = new Float32Array(g.m);
  let nCoveredEdges = 0;
  for (let e = 0; e < g.m; e++) {
    const a = g.pre[e]!;
    const b = g.post[e]!;
    const p = pair.get(`${fvName(typeName(g, a))} ${fvName(typeName(g, b))}`);
    if (p) {
      w[e] = g.weight[e]! * p.sign * p.strength;
      nCoveredEdges++;
    } else {
      w[e] = g.weight[e]! * g.sign[a]! * defaultScale;
    }
  }
  return { tau, bias, w, covered, nCoveredTypes: typeCovered.filter(Boolean).length, nCoveredEdges };
}
